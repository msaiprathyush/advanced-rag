from typing import Literal

from pydantic import BaseModel, Field

from advanced_rag.generation.citations import Citation


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["naive", "advanced"] = "advanced"
    debug: bool = False
    # Prior turns of the conversation, used only to resolve follow-ups ("what about its limits?").
    # Bounded so a client cannot inflate token cost.
    history: list[Turn] = Field(default_factory=list, max_length=6)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    decision: Literal["answered", "declined"]
    mode: str
    decline_reason: str | None = None
    grounded: bool | None = None
    rewrites: int = 0
    llm_calls: int = 0
    top_score: float | None = None  # best cross-encoder score; used by the drift monitor
    standalone_question: str | None = None  # the follow-up as rewritten for retrieval, if it was
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
