import re

from pydantic import BaseModel

from advanced_rag.retrieval.retriever import RetrievedChunk

_CITE_RE = re.compile(r"\[(\d+)\]")
# Some models emit browsing-style markers such as 【4†L1-L4】 instead of [4].
_FULLWIDTH_RE = re.compile(r"【\s*(\d+)[^】]*】")
# "[1, 2]" / "[1,2,3]" -> "[1][2][3]"
_MULTI_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)+)\]")


def normalize_citations(answer: str) -> str:
    """Rewrite citation variants into the canonical [n][m] form used by the parser."""
    answer = _FULLWIDTH_RE.sub(r"[\1]", answer)
    return _MULTI_RE.sub(lambda m: "".join(f"[{n.strip()}]" for n in m.group(1).split(",")), answer)


class Citation(BaseModel):
    index: int
    arxiv_id: str
    title: str
    section: str
    abs_url: str
    snippet: str


def cited_indices(answer: str, n_chunks: int) -> list[int]:
    """Valid 1-based block numbers cited in the answer, in order of first appearance."""
    seen: list[int] = []
    for m in _CITE_RE.finditer(answer):
        i = int(m.group(1))
        if 1 <= i <= n_chunks and i not in seen:
            seen.append(i)
    return seen


def build_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    out = []
    for i in cited_indices(answer, len(chunks)):
        p = chunks[i - 1].payload
        out.append(
            Citation(
                index=i,
                arxiv_id=p["arxiv_id"],
                title=p["title"],
                section=p["section"],
                abs_url=p["abs_url"],
                snippet=p["text"][:300],
            )
        )
    return out
