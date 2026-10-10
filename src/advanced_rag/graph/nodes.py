"""Graph nodes. Each is built by a factory so dependencies (retriever, LLM) can be injected."""

import re
import time
from collections import Counter
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
    condense_messages,
    groundedness_messages,
    overview_messages,
    rewrite_messages,
    scope_messages,
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


OVERVIEW_MAX_TITLES = 80
_QUOTED = re.compile(r"[\"\u201c]([^\"\u201d]{12,})[\"\u201d]")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().rstrip(".,;:").lower())


def is_collection_question(llm: LLMClient, question: str) -> bool:
    """Does the question ask about the collection itself (scope, topics, papers held)?"""
    verdict = llm.complete("judge", scope_messages(question))
    return verdict.strip().upper().startswith("YES")


def titles_verified(body: str, titles: list[str]) -> bool:
    """True only if every quoted title in `body` is (a prefix/part of) a real catalog title."""
    known = [_norm(t) for t in titles]
    return all(any(_norm(q) in k for k in known) for q in _QUOTED.findall(body))


def make_nodes(
    retriever: Retriever,
    llm: LLMClient,
    s: Settings,
    catalog_fn: Callable[[], list[dict]] | None = None,
) -> dict[str, Callable]:
    def condense_question(state: RAGState) -> RAGState:
        """Rewrite a chat follow-up into a standalone question. Only runs when there is history."""
        t0 = time.perf_counter()
        raw = state["question"]
        try:
            standalone = llm.complete("judge", condense_messages(state["history"], raw))
            standalone = standalone.strip().strip('"')
        except Exception as exc:  # a failed rewrite must never fail the request
            log.warning("condense_failed", error=type(exc).__name__)
            standalone = ""
        if not standalone or len(standalone) > 500:
            standalone = raw
        return {
            "original_question": raw,
            "question": standalone,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "trace": _trace(state, "condense_question", t0, standalone=standalone),
        }

    def classify_scope(state: RAGState) -> RAGState:
        """Route every question: about the collection itself, or about paper content?

        "What topics do these papers cover?" matches no passage, so passage search either declines it
        or, worse, summarises whichever chunks happen to match. Routing on the question (one tiny
        judge-model call, ~0.3 s) keeps paraphrases on the same path regardless of retrieval noise.
        """
        t0 = time.perf_counter()
        is_scope = False
        try:
            is_scope = is_collection_question(llm, state["question"]) and bool(
                catalog_fn and catalog_fn()
            )
        except Exception as exc:  # a failed check just means "treat it as a normal question"
            log.warning("scope_check_failed", error=type(exc).__name__)
        return {
            "scope": is_scope,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "trace": _trace(state, "classify_scope", t0, scope=is_scope),
        }

    def corpus_overview(state: RAGState) -> RAGState:
        """Answer from the index catalog (titles, years, categories) instead of passage search."""
        t0 = time.perf_counter()
        papers = catalog_fn() if catalog_fn else []
        years = sorted(p["published"][:4] for p in papers if p.get("published"))
        cats = [
            c for c, _ in Counter(c for p in papers for c in p.get("categories", [])).most_common(3)
        ]
        # Counts and date range are computed, not generated, so they cannot be hallucinated.
        header = f"The index currently holds **{len(papers)} papers**"
        if years:
            header += f" (published {years[0]}\u2013{years[-1]})"
        if cats:
            header += f", mostly in {', '.join(cats)}"
        header += ". Main topics:"
        titles = [p["title"] for p in papers[:OVERVIEW_MAX_TITLES]]
        calls = state.get("llm_calls", 0)
        fallback = "\n".join(f"- \u201c{t}\u201d" for t in titles[:8])
        verified = True
        try:
            body = llm.complete("judge", overview_messages(titles)).strip()
            calls += 1
            # Never show a paper title the model invented: verify, or fall back to real titles.
            verified = bool(body) and titles_verified(body, [p["title"] for p in papers])
            if not verified:
                log.warning("overview_unverified_titles")
                body = fallback
        except Exception as exc:  # fall back to a plain list of the newest papers
            log.warning("overview_failed", error=type(exc).__name__)
            body = fallback
        return {
            "answer": f"{header}\n\n{body}",
            "decision": "answered",
            "citations": [],
            "grounded": None,
            "top_score": None,  # not a retrieval result; keep it out of the drift monitor's stats
            "llm_calls": calls,
            "trace": _trace(state, "corpus_overview", t0, papers=len(papers), verified=verified),
        }

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
        "condense_question": condense_question,
        "classify_scope": classify_scope,
        "corpus_overview": corpus_overview,
        "retrieve": retrieve,
        "rewrite_query": rewrite_query,
        "generate": generate,
        "check_groundedness": check_groundedness,
        "regenerate": regenerate,
        "finalize": finalize,
        "decline": decline,
    }
