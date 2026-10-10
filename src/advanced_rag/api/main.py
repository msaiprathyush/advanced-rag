"""FastAPI service: /query, /ingest, /health, /ready."""

import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import groq
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from advanced_rag import __version__
from advanced_rag.api.deps import require_ingest_key
from advanced_rag.api.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from advanced_rag.config import get_settings
from advanced_rag.embeddings.models import warm_up
from advanced_rag.logging import configure_logging, get_logger, request_id_var
from advanced_rag.retrieval.retriever import warm_up_reranker

configure_logging(get_settings().log_level, get_settings().log_json)
log = get_logger("api")
limiter = Limiter(key_func=get_remote_address)


@lru_cache
def _graph():
    from advanced_rag.graph.build import build_graph

    return build_graph()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # First request shouldn't pay model-load latency: load all three models at startup, where
    # Cloud Run's startup CPU boost applies and no user is waiting.
    await run_in_threadpool(warm_up)
    await run_in_threadpool(warm_up_reranker)
    yield


# The homepage is the chat UI. Swagger stays available for engineers, but off the main path.
app = FastAPI(
    title="Advanced RAG over arXiv",
    version=__version__,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# No inline script/style anywhere in the UI, so the policy can be strict.
PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=PAGE_HEADERS)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    log.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round((time.perf_counter() - t0) * 1000),
    )
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready() -> dict:
    from advanced_rag.store.qdrant import count_points, get_client

    try:
        return {"status": "ready", "points": count_points(get_client())}
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"vector store unavailable: {type(exc).__name__}"
        ) from exc


def _run_query(req: QueryRequest) -> QueryResponse:
    t0 = time.perf_counter()
    if req.mode == "naive":
        from advanced_rag.generation.naive import naive_answer

        answer, citations = naive_answer(req.question)
        return QueryResponse(
            answer=answer,
            citations=citations,
            decision="answered",
            mode="naive",
            llm_calls=1,
            latency_ms=round((time.perf_counter() - t0) * 1000),
        )
    from advanced_rag.graph.build import run_graph

    out = run_graph(req.question, _graph(), history=[t.model_dump() for t in req.history])
    original = out.get("original_question")
    return QueryResponse(
        answer=out["answer"],
        citations=out.get("citations", []),
        decision=out["decision"],
        mode="advanced",
        decline_reason=out.get("decline_reason"),
        grounded=out.get("grounded"),
        top_score=out.get("top_score"),
        standalone_question=out["question"] if original and out["question"] != original else None,
        rewrites=out.get("rewrites", 0),
        llm_calls=out.get("llm_calls", 0),
        latency_ms=round((time.perf_counter() - t0) * 1000),
        trace=out.get("trace") if req.debug else None,
    )


@app.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
@limiter.limit(get_settings().query_rate_limit)
async def query(request: Request, req: QueryRequest) -> QueryResponse:
    try:
        resp = await run_in_threadpool(_run_query, req)
    except groq.RateLimitError as exc:
        # The free LLM quota is exhausted/busy. Say so (and when to retry) instead of a generic error.
        log.warning("query_llm_rate_limited")
        raise HTTPException(
            status_code=503, detail="llm_rate_limited", headers={"Retry-After": "60"}
        ) from exc
    except Exception as exc:
        log.error("query_failed", error=type(exc).__name__, detail=str(exc)[:300])
        raise HTTPException(status_code=502, detail="upstream service failed") from exc
    log.info(
        "query_done",
        mode=resp.mode,
        decision=resp.decision,
        decline_reason=resp.decline_reason,
        top_score=resp.top_score,
        rewrites=resp.rewrites,
        llm_calls=resp.llm_calls,
        latency_ms=resp.latency_ms,
    )
    return resp


def _run_ingest(req: IngestRequest) -> IngestResponse:
    from advanced_rag.clients.arxiv import ArxivClient
    from advanced_rag.ingestion.pipeline import ingest_papers

    arxiv = ArxivClient()
    if req.arxiv_ids:
        metas = arxiv.get_by_ids(req.arxiv_ids[: req.max_results])
    else:
        metas = arxiv.search(req.query or get_settings().arxiv_query, max_results=req.max_results)
    r = ingest_papers(metas, arxiv=arxiv)
    return IngestResponse(
        new=len(r.new), updated=len(r.updated), skipped=len(r.skipped), failed=r.failed
    )


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_ingest_key)])
async def ingest(req: IngestRequest) -> IngestResponse:
    # Small synchronous batches only; bulk re-indexing belongs to the Cloud Run Job.
    return await run_in_threadpool(_run_ingest, req)
