from typing import Literal, TypedDict

from advanced_rag.generation.citations import Citation
from advanced_rag.retrieval.retriever import RetrievedChunk


class RAGState(TypedDict, total=False):
    question: str
    current_query: str
    queries: list[str]  # every query tried, in order
    rewrites: int
    regenerations: int
    docs: list[RetrievedChunk]
    top_score: float | None
    answer: str
    citations: list[Citation]
    grounded: bool | None
    unsupported_claims: list[str]
    decision: Literal["answered", "declined"]
    decline_reason: str
    llm_calls: int
    trace: list[dict]
