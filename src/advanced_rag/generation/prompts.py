import re

from langchain_core.messages import HumanMessage, SystemMessage

from advanced_rag.retrieval.retriever import RetrievedChunk

ANSWER_SYSTEM = """You are a research assistant answering questions about ML/AI papers.
Rules:
- Use ONLY the numbered context blocks below. Do not use outside knowledge.
- Put a citation like [1] or [2][3] after every claim, using the block numbers.
  Use ONLY plain ASCII square brackets with the bare number. Never use 【】, †, or line ranges.
- If the context does not contain enough information, reply exactly: "I don't have enough information in the indexed papers to answer that."
- Be concise and precise."""

REWRITE_SYSTEM = """You rewrite search queries for retrieval over arXiv ML/AI paper text.
Return ONE improved query only: keyword-rich, using the terminology papers would use.
It must differ from the previous queries. No explanation, no quotes."""

CONDENSE_SYSTEM = """You turn a follow-up question into ONE standalone question using the chat history.
- Resolve pronouns and references ("it", "that paper", "its limitations") from the history.
- Keep the user's intent. Do NOT answer the question and do NOT add facts.
- If the question is already standalone, return it unchanged.
Return only the question: no explanation, no quotes."""

SCOPE_SYSTEM = """You classify a question about a collection of research papers.
Answer YES if the question asks about the collection itself: what papers, topics, subjects, years or
areas it contains or covers, or what the assistant can answer. Answer NO if it asks about the content
of a specific paper or technique. Reply with exactly one word: YES or NO."""

OVERVIEW_SYSTEM = """You describe what a paper collection covers, using ONLY the numbered title list.
Write 4-6 bullet points. Each bullet names one topic and cites 1-3 representative paper titles
copied verbatim from the list, in quotes. Do not invent papers, results or claims beyond the titles.
No introduction and no closing remarks."""

GROUNDEDNESS_SYSTEM = """You are a strict fact-checker. Given CONTEXT blocks and an ANSWER, decide whether
every factual claim in the ANSWER is supported by the CONTEXT.
- grounded=true only if all claims are supported.
- List each unsupported claim in unsupported_claims.
- An answer saying the information is not available counts as grounded."""

DECLINE_TEXT = "I don't have enough information in the indexed papers to answer that."


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        p = c.payload
        blocks.append(f"[{i}] ({p['title']}, {p['section']})\n{p['text']}")
    return "\n\n".join(blocks)


def answer_messages(
    question: str, chunks: list[RetrievedChunk], unsupported: list[str] | None = None
) -> list:
    extra = ""
    if unsupported:
        claims = "\n".join(f"- {u}" for u in unsupported)
        extra = (
            f"\n\nYour previous draft contained unsupported claims. Do NOT repeat them:\n{claims}"
        )
    return [
        SystemMessage(ANSWER_SYSTEM + extra),
        HumanMessage(f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"),
    ]


def rewrite_messages(question: str, previous: list[str]) -> list:
    prev = "\n".join(f"- {q}" for q in previous)
    return [
        SystemMessage(REWRITE_SYSTEM),
        HumanMessage(f"Original question: {question}\nPrevious queries:\n{prev}"),
    ]


_CITE_MARK = re.compile(r"\[\d+\]")


def condense_messages(history: list[dict], question: str) -> list:
    """Chat transcript + follow-up. Old [n] markers are dropped (they refer to a previous context)
    and assistant turns are truncated, which keeps the call cheap."""
    lines = []
    for turn in history[-6:]:
        text = _CITE_MARK.sub("", turn["content"]).strip()
        if turn["role"] == "assistant":
            text = text[:500]
        lines.append(f"{'User' if turn['role'] == 'user' else 'Assistant'}: {text}")
    transcript = "\n".join(lines)
    return [
        SystemMessage(CONDENSE_SYSTEM),
        HumanMessage(f"Chat history:\n{transcript}\n\nFollow-up question: {question}"),
    ]


def scope_messages(question: str) -> list:
    return [SystemMessage(SCOPE_SYSTEM), HumanMessage(question)]


def overview_messages(titles: list[str]) -> list:
    listing = "\n".join(f"{i}. {t}" for i, t in enumerate(titles, 1))
    return [SystemMessage(OVERVIEW_SYSTEM), HumanMessage(f"Papers:\n{listing}")]


def groundedness_messages(answer: str, chunks: list[RetrievedChunk]) -> list:
    return [
        SystemMessage(GROUNDEDNESS_SYSTEM),
        HumanMessage(f"CONTEXT:\n{format_context(chunks)}\n\nANSWER:\n{answer}"),
    ]
