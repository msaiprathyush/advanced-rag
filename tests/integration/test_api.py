from fastapi.testclient import TestClient

from advanced_rag.api import main
from advanced_rag.api.schemas import QueryRequest, QueryResponse


def client(monkeypatch):
    monkeypatch.setattr(main, "warm_up", lambda: None)
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
