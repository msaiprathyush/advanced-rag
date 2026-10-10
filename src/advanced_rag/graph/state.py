from typing import Literal, TypedDict

from advanced_rag.generation.citations import Citation
from advanced_rag.retrieval.retriever import RetrievedChunk


class RAGState(TypedDict, total=False):
    question: str  # the question used for retrieval/generation (standalone after condensing)
    original_question: str  # what the user typed, set only when a follow-up was rewritten
    history: list[dict]  # prior chat turns: {"role": "user"|"assistant", "content": str}
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
    scope: bool  # True when the question asks about the collection, not a paper's content
    llm_calls: int
    trace: list[dict]
