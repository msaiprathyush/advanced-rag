from typing import Literal

from pydantic import BaseModel, Field

from advanced_rag.generation.citations import Citation


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["naive", "advanced"] = "advanced"
    debug: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    decision: Literal["answered", "declined"]
    mode: str
    decline_reason: str | None = None
    grounded: bool | None = None
    rewrites: int = 0
    llm_calls: int = 0
    latency_ms: int
    trace: list[dict] | None = None


class IngestRequest(BaseModel):
    arxiv_ids: list[str] | None = None
    query: str | None = None
    max_results: int = Field(default=5, ge=1, le=10)


class IngestResponse(BaseModel):
    new: int
    updated: int
    skipped: int
    failed: dict[str, str]
