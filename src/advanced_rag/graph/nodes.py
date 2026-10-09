"""Graph nodes. Each is built by a factory so dependencies (retriever, LLM) can be injected."""

import time
from collections.abc import Callable

from pydantic import BaseModel, Field

from advanced_rag.clients.llm import LLMClient
from advanced_rag.config import Settings
from advanced_rag.generation.citations import (
    build_citations,
    cited_indices,
    normalize_citations,
)
from advanced_rag.generation.prompts import (
    DECLINE_TEXT,
    answer_messages,
    groundedness_messages,
    rewrite_messages,
)
from advanced_rag.graph.state import RAGState
from advanced_rag.logging import get_logger
from advanced_rag.retrieval.retriever import Retriever

log = get_logger(__name__)


class GroundednessVerdict(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    reasoning: str = ""


def _trace(state: RAGState, node: str, t0: float, **info) -> list[dict]:
    entry = {"node": node, "ms": round((time.perf_counter() - t0) * 1000), **info}
    return [*state.get("trace", []), entry]


def make_nodes(retriever: Retriever, llm: LLMClient, s: Settings) -> dict[str, Callable]:
    def retrieve(state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        query = state.get("current_query") or state["question"]
        result = retriever.retrieve(query)
        queries = state.get("queries", [])
        return {
            "current_query": query,
            "queries": queries if query in queries else [*queries, query],
            "docs": result.chunks,
            "top_score": result.top_score,
            "trace": _trace(state, "retrieve", t0, query=query, top_score=result.top_score),
        }

    def rewrite_query(state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        new = llm.complete("judge", rewrite_messages(state["question"], state.get("queries", [])))
        new = new.strip().strip('"') or state["question"]
        return {
            "current_query": new,
            "rewrites": state.get("rewrites", 0) + 1,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "trace": _trace(state, "rewrite_query", t0, new_query=new),
        }

    def generate(state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        msgs = answer_messages(state["question"], state["docs"], state.get("unsupported_claims"))
        answer = normalize_citations(llm.complete("generator", msgs))
        return {
            "answer": answer,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "trace": _trace(state, "generate", t0),
        }

    def check_groundedness(state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        answer, docs = state["answer"], state["docs"]
        calls = state.get("llm_calls", 0)
        if answer.strip() == DECLINE_TEXT:
            # The model itself said the context is insufficient: honour it, no judge call needed.
            return {
                "grounded": True,
                "trace": _trace(state, "check_groundedness", t0, verdict="model_declined"),
            }
        if not cited_indices(answer, len(docs)):
            # No valid citations -> ungrounded without spending an LLM call.
            return {
                "grounded": False,
                "unsupported_claims": ["Answer contained no valid [n] citations."],
                "trace": _trace(state, "check_groundedness", t0, verdict="no_citations"),
            }
        try:
            verdict = llm.structured(
                "judge", groundedness_messages(answer, docs), GroundednessVerdict
            )
        except Exception as exc:
            # Fail closed: an unverifiable answer is never returned as grounded.
            log.error("groundedness_judge_failed", error=type(exc).__name__, detail=str(exc)[:200])
            return {
                "grounded": False,
                "unsupported_claims": ["Groundedness could not be verified."],
                "llm_calls": calls + 1,
                "trace": _trace(state, "check_groundedness", t0, verdict="judge_error"),
            }
        return {
            "grounded": verdict.grounded,
            "unsupported_claims": verdict.unsupported_claims,
            "llm_calls": calls + 1,
            "trace": _trace(
                state,
                "check_groundedness",
                t0,
                verdict=verdict.grounded,
                unsupported=len(verdict.unsupported_claims),
            ),
        }

    def regenerate(state: RAGState) -> RAGState:
        return {"regenerations": state.get("regenerations", 0) + 1}

    def finalize(state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        if state["answer"].strip() == DECLINE_TEXT:
            return {
                "decision": "declined",
                "decline_reason": "model_found_insufficient_context",
                "citations": [],
                "trace": _trace(state, "finalize", t0, decision="declined"),
            }
        return {
            "decision": "answered",
            "citations": build_citations(state["answer"], state["docs"]),
            "trace": _trace(state, "finalize", t0, decision="answered"),
        }

    def decline(state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        reason = (
            "no_relevant_sources" if state.get("grounded") is None else "could_not_ground_answer"
        )
        return {
            "answer": DECLINE_TEXT,
            "decision": "declined",
            "decline_reason": reason,
            "citations": build_citations(
                "".join(f"[{i + 1}]" for i in range(len(state.get("docs", [])))),
                state.get("docs", []),
            ),  # sources found, not claimed
            "trace": _trace(state, "decline", t0, reason=reason),
        }

    return {
        "retrieve": retrieve,
        "rewrite_query": rewrite_query,
        "generate": generate,
        "check_groundedness": check_groundedness,
        "regenerate": regenerate,
        "finalize": finalize,
        "decline": decline,
    }
