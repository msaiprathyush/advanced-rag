import time

import httpx
import pytest
import respx

from advanced_rag.clients import retry as retry_mod
from advanced_rag.clients.arxiv import ArxivClient, Throttle, parse_feed
from advanced_rag.clients.retry import retry_after_seconds, with_retry

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
 <entry>
  <id>http://arxiv.org/abs/2310.11511v1</id>
  <updated>2023-10-17T17:50:00Z</updated>
  <published>2023-10-17T17:50:00Z</published>
  <title>Self-RAG:  Learning to
   Retrieve</title>
  <summary>  An abstract
  here. </summary>
  <author><name>Akari Asai</name></author>
  <link href="http://arxiv.org/abs/2310.11511v1" rel="alternate" type="text/html"/>
  <link title="pdf" href="http://arxiv.org/pdf/2310.11511v1" rel="related" type="application/pdf"/>
  <category term="cs.CL"/>
 </entry>
</feed>"""


def test_parse_feed():
    (p,) = parse_feed(ATOM)
    assert p.arxiv_id == "2310.11511" and p.version == 1
    assert p.title == "Self-RAG: Learning to Retrieve"
    assert p.abstract == "An abstract here."
    assert p.authors == ["Akari Asai"] and p.categories == ["cs.CL"]


def test_throttle_spacing():
    t = Throttle(0.2)
    t0 = time.monotonic()
    t.wait()
    t.wait()
    t.wait()
    assert time.monotonic() - t0 >= 0.4


def test_retry_after_header_parsed():
    resp = httpx.Response(429, headers={"retry-after": "7"})
    exc = httpx.HTTPStatusError("x", request=httpx.Request("GET", "http://x"), response=resp)
    assert retry_after_seconds(exc) == 7.0


@respx.mock
def test_arxiv_retries_on_503_then_succeeds(monkeypatch):
    monkeypatch.setattr(retry_mod, "wait_random_exponential", lambda **_: lambda _rs: 0)
    route = respx.get("https://export.arxiv.org/api/query").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, content=ATOM)]
    )
    # Rebuild the decorated method with zero waits for the test.
    client = ArxivClient(http=httpx.Client(), throttle=Throttle(0))
    client._get = with_retry("arxiv", attempts=3, max_wait=0.01).__call__(  # type: ignore[method-assign]
        client._get.__wrapped__.__get__(client)
    )
    papers = client.search("cat:cs.CL", max_results=1)
    assert route.call_count == 2 and papers[0].arxiv_id == "2310.11511"


@respx.mock
def test_arxiv_gives_up_after_attempts():
    respx.get("https://export.arxiv.org/api/query").mock(return_value=httpx.Response(404))
    client = ArxivClient(http=httpx.Client(), throttle=Throttle(0))
    with pytest.raises(httpx.HTTPStatusError):  # 404 is not retryable -> immediate failure
        client.search("x")


@respx.mock
def test_unwritable_pdf_cache_does_not_fail_the_fetch(tmp_path):
    respx.get("https://export.arxiv.org/pdf/2310.11511").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.4 fake")
    )
    blocker = tmp_path / "file"
    blocker.write_text("x")  # a *file* where the cache dir should go -> mkdir raises OSError
    client = ArxivClient(http=httpx.Client(), throttle=Throttle(0), pdf_cache_dir=blocker / "cache")
    assert client.fetch_pdf("2310.11511").startswith(b"%PDF")


def _rate_limited(retry_after: str):
    resp = httpx.Response(429, headers={"retry-after": retry_after})
    return httpx.HTTPStatusError("x", request=httpx.Request("GET", "http://x"), response=resp)


def test_retry_budget_fails_fast_instead_of_sleeping_through_a_long_retry_after():
    calls = []

    @with_retry("test", attempts=6, max_wait=30, max_total_s=5)
    def flaky():
        calls.append(1)
        raise _rate_limited("60")  # server says wait 60 s, but the budget is 5 s

    t0 = time.monotonic()
    with pytest.raises(httpx.HTTPStatusError):
        flaky()
    assert len(calls) == 1 and time.monotonic() - t0 < 2  # gave up immediately, did not sleep 60 s


def test_retry_within_budget_still_retries(monkeypatch):
    calls = []

    @with_retry("test", attempts=4, min_wait=0.01, max_wait=0.05, max_total_s=30)
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _rate_limited("0")
        return "ok"

    assert flaky() == "ok" and len(calls) == 3
