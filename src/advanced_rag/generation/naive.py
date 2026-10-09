"""M1 baseline: single dense retrieval -> generate. Kept for ablation / `mode=naive`."""

from advanced_rag.clients.llm import LLMClient, get_llm
from advanced_rag.generation.citations import Citation, build_citations, normalize_citations
from advanced_rag.generation.prompts import answer_messages
from advanced_rag.retrieval.retriever import Retriever


def naive_answer(
    question: str, retriever: Retriever | None = None, llm: LLMClient | None = None
) -> tuple[str, list[Citation]]:
    retriever = retriever or Retriever("dense")
    llm = llm or get_llm()
    chunks = retriever.retrieve(question).chunks
    answer = normalize_citations(llm.complete("generator", answer_messages(question, chunks)))
    return answer, build_citations(answer, chunks)
