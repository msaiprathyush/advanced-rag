"""Section-aware chunking with a contextual header on the embedded text."""

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from advanced_rag.ingestion.models import Chunk, PaperMeta

CHUNK_SIZE = 1400  # ~350 tokens, comfortably under bge's 512-token limit
CHUNK_OVERLAP = 200
MIN_CHARS = 200

_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=True
)
_text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def _section_name(metadata: dict) -> str:
    parts = [metadata[k].strip("*# ").strip() for k in ("h1", "h2", "h3") if metadata.get(k)]
    return " > ".join(p for p in parts if p) or "Body"


def _mk(meta: PaperMeta, idx: int, section: str, text: str) -> Chunk:
    return Chunk(
        arxiv_id=meta.arxiv_id,
        version=meta.version,
        chunk_index=idx,
        section=section,
        text=text,
        embed_text=f"{meta.title}\nSection: {section}\n\n{text}",
        title=meta.title,
        authors=meta.authors,
        categories=meta.categories,
        published=meta.published,
        abs_url=meta.abs_url,
        pdf_url=meta.pdf_url,
    )


def chunk_paper(meta: PaperMeta, markdown: str | None) -> list[Chunk]:
    """Abstract chunk first, then body chunks. Falls back to abstract-only if no body text."""
    chunks = [_mk(meta, 0, "Abstract", meta.abstract)]
    if not markdown:
        return chunks
    idx = 1
    for doc in _header_splitter.split_text(markdown):
        section = _section_name(doc.metadata)
        for piece in _text_splitter.split_text(doc.page_content):
            piece = piece.strip()
            if len(piece) < MIN_CHARS:
                continue
            chunks.append(_mk(meta, idx, section, piece))
            idx += 1
    return chunks
