from fastapi.testclient import TestClient

from advanced_rag.api import main
from advanced_rag.api.schemas import QueryRequest, QueryResponse


def client(monkeypatch):
    monkeypatch.setattr(main, "warm_up", lambda: None)
    monkeypatch.setattr(main, "warm_up_reranker", lambda: None)
    main.limiter.reset()  # the 3/min /query limit must not leak between tests
    return TestClient(main.app)


def test_health(monkeypatch):
    with client(monkeypatch) as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"
        assert "x-request-id" in r.headers


def test_ingest_requires_key(monkeypatch):
    monkeypatch.setenv("INGEST_API_KEY", "secret")
    from advanced_rag.config import get_settings

    get_settings.cache_clear()
    with client(monkeypatch) as c:
        assert c.post("/ingest", json={"arxiv_ids": ["2310.11511"]}).status_code == 401
        assert (
            c.post("/ingest", json={"arxiv_ids": ["x"]}, headers={"X-API-Key": "nope"}).status_code
            == 401
        )
    get_settings.cache_clear()


def test_ingest_fails_closed_when_server_key_unset(monkeypatch):
    monkeypatch.setenv("INGEST_API_KEY", "")
    from advanced_rag.config import get_settings

    get_settings.cache_clear()
    with client(monkeypatch) as c:
        assert (
            c.post("/ingest", json={"arxiv_ids": ["x"]}, headers={"X-API-Key": ""}).status_code
            == 401
        )
    get_settings.cache_clear()


def test_query_returns_stubbed_response(monkeypatch):
    stub = QueryResponse(
        answer="A [1].", citations=[], decision="answered", mode="advanced", latency_ms=1
    )
    monkeypatch.setattr(main, "_run_query", lambda req: stub)
    with client(monkeypatch) as c:
        r = c.post("/query", json={"question": "What is Self-RAG?"})
        assert r.status_code == 200 and r.json()["decision"] == "answered"
        assert c.post("/query", json={"question": "hi"}).status_code == 422  # too short


def test_query_upstream_failure_is_502(monkeypatch):
    def boom(req: QueryRequest):
        raise RuntimeError("groq down")

    monkeypatch.setattr(main, "_run_query", boom)
    with client(monkeypatch) as c:
        assert c.post("/query", json={"question": "What is Self-RAG?"}).status_code == 502


def test_homepage_serves_the_chat_ui_with_a_strict_csp(monkeypatch):
    with client(monkeypatch) as c:
        r = c.get("/")
        assert r.status_code == 200 and "Ask the arXiv papers" in r.text
        csp = r.headers["content-security-policy"]
        assert "default-src 'self'" in csp and "unsafe-inline" not in csp
        assert r.headers["x-content-type-options"] == "nosniff"
        assert c.get("/static/app.js").status_code == 200
        assert c.get("/static/app.css").status_code == 200


def test_swagger_moved_off_the_main_path(monkeypatch):
    with client(monkeypatch) as c:
        assert c.get("/docs").status_code == 404
        assert c.get("/api/docs").status_code == 200
        assert c.get("/api/openapi.json").status_code == 200


def test_history_is_bounded(monkeypatch):
    turn = {"role": "user", "content": "hello there"}
    with client(monkeypatch) as c:
        too_many = {"question": "What is Self-RAG?", "history": [turn] * 7}
        too_long = {
            "question": "What is Self-RAG?",
            "history": [{"role": "user", "content": "x" * 2001}],
        }
        bad_role = {
            "question": "What is Self-RAG?",
            "history": [{"role": "system", "content": "hi"}],
        }
        for body in (too_many, too_long, bad_role):
            assert c.post("/query", json=body).status_code == 422


def test_history_reaches_the_query_runner(monkeypatch):
    seen = {}

    def fake(req):
        seen["history"] = [t.model_dump() for t in req.history]
        return QueryResponse(
            answer="A [1].", citations=[], decision="answered", mode="advanced", latency_ms=1
        )

    monkeypatch.setattr(main, "_run_query", fake)
    body = {
        "question": "what are its limits?",
        "history": [{"role": "user", "content": "About Self-RAG"}],
    }
    with client(monkeypatch) as c:
        assert c.post("/query", json=body).status_code == 200
    assert seen["history"] == [{"role": "user", "content": "About Self-RAG"}]


def test_llm_rate_limit_is_a_clear_503_not_a_generic_error(monkeypatch):
    import groq
    import httpx

    def limited(req):
        resp = httpx.Response(429, request=httpx.Request("POST", "http://x"))
        raise groq.RateLimitError("rate limited", response=resp, body=None)

    monkeypatch.setattr(main, "_run_query", limited)
    with client(monkeypatch) as c:
        r = c.post("/query", json={"question": "What is Self-RAG?"})
        assert r.status_code == 503 and r.json()["detail"] == "llm_rate_limited"
        assert r.headers["retry-after"] == "60"
