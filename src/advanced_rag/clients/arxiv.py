"""Thin arXiv API client: throttled (>= 3 s between calls), retried, parsed into PaperMeta."""

import re
import threading
import time
from collections.abc import Iterable
from pathlib import Path

import feedparser
import httpx

from advanced_rag.clients.retry import with_retry
from advanced_rag.config import get_settings
from advanced_rag.ingestion.models import PaperMeta
from advanced_rag.logging import get_logger

log = get_logger(__name__)

API_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://export.arxiv.org/pdf/{id}"
_ID_RE = re.compile(r"arxiv\.org/abs/(?P<id>.+?)(?:v(?P<v>\d+))?$")


class ArxivAPIError(RuntimeError):
    pass


class Throttle:
    """Process-wide minimum spacing between requests (arXiv asks for 1 request / 3 s)."""

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last = float("-inf")

    def wait(self) -> None:
        with self._lock:
            delay = self._last + self.min_interval_s - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


_default_throttle: Throttle | None = None


def _get_default_throttle() -> Throttle:
    global _default_throttle
    if _default_throttle is None:
        _default_throttle = Throttle(get_settings().arxiv_delay_s)
    return _default_throttle


def parse_feed(xml: str | bytes) -> list[PaperMeta]:
    feed = feedparser.parse(xml)
    papers: list[PaperMeta] = []
    for e in feed.entries:
        if "/api/errors" in e.get("id", ""):
            raise ArxivAPIError(e.get("summary", "arXiv API error"))
        m = _ID_RE.search(e.id)
        if not m:
            continue
        links = e.get("links", [])
        pdf = next((lk.href for lk in links if lk.get("title") == "pdf"), None)
        abs_url = next((lk.href for lk in links if lk.get("rel") == "alternate"), e.id)
        papers.append(
            PaperMeta(
                arxiv_id=m["id"],
                version=int(m["v"] or 1),
                title=" ".join(e.title.split()),
                abstract=" ".join(e.summary.split()),
                authors=[a.name for a in e.get("authors", [])],
                categories=[t.term for t in e.get("tags", [])],
                published=e.published,
                updated=e.get("updated", e.published),
                abs_url=abs_url,
                pdf_url=pdf or f"https://arxiv.org/pdf/{m['id']}",
            )
        )
    return papers


class ArxivClient:
    def __init__(
        self,
        http: httpx.Client | None = None,
        throttle: Throttle | None = None,
        pdf_cache_dir: str | Path | None = None,
    ) -> None:
        s = get_settings()
        self._http = http or httpx.Client(
            headers={"User-Agent": s.arxiv_user_agent}, timeout=60.0, follow_redirects=True
        )
        self._throttle = throttle or _get_default_throttle()
        self._pdf_cache = Path(pdf_cache_dir) if pdf_cache_dir else None

    # arXiv answers 429 with no Retry-After; back off for a long while rather than hammering.
    @with_retry("arxiv", attempts=6, min_wait=10, max_wait=90)
    def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        self._throttle.wait()
        t0 = time.perf_counter()
        resp = self._http.get(url, params=params)
        log.info(
            "external_call",
            service="arxiv",
            url=url,
            status=resp.status_code,
            latency_ms=round((time.perf_counter() - t0) * 1000),
        )
        resp.raise_for_status()
        return resp

    def search(
        self,
        query: str,
        *,
        start: int = 0,
        max_results: int = 50,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> list[PaperMeta]:
        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        return parse_feed(self._get(API_URL, params).content)

    def get_by_ids(self, ids: Iterable[str], batch_size: int = 50) -> list[PaperMeta]:
        ids = [i.strip() for i in ids if i.strip()]
        out: list[PaperMeta] = []
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            params = {"id_list": ",".join(batch), "max_results": len(batch)}
            out.extend(parse_feed(self._get(API_URL, params).content))
        return out

    def fetch_pdf(self, arxiv_id: str, version: int | None = None) -> bytes:
        vid = f"{arxiv_id}v{version}" if version else arxiv_id
        cached = self._pdf_cache / f"{vid.replace('/', '_')}.pdf" if self._pdf_cache else None
        if cached and cached.exists():
            return cached.read_bytes()
        data = self._get(PDF_URL.format(id=vid)).content
        if not data.startswith(b"%PDF"):
            raise ArxivAPIError(f"Response for {vid} is not a PDF")
        if cached:
            try:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(data)
            except OSError as exc:  # a cache is an optimisation; never fail ingestion over it
                log.warning("pdf_cache_write_failed", error=type(exc).__name__)
        return data
