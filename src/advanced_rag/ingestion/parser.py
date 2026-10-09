"""PDF bytes -> markdown text (with section headers), references stripped."""

import re

from advanced_rag.logging import get_logger

log = get_logger(__name__)

_REFS_RE = re.compile(
    r"^\s*(?:#+\s*)?\**\s*(?:\d+\.?\s*)?(references|bibliography)\s*\**\s*$", re.I | re.M
)
_HYPHEN_RE = re.compile(r"(\w)-\n(\w)")


def clean_markdown(md: str) -> str:
    m = _REFS_RE.search(md)
    if m and m.start() > len(md) * 0.3:  # ignore a spurious early match (e.g. in a ToC)
        md = md[: m.start()]
    md = _HYPHEN_RE.sub(r"\1\2", md)
    md = re.sub(r"[ \t]+", " ", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def parse_pdf(pdf_bytes: bytes) -> str | None:
    """Return cleaned markdown, or None if the PDF can't be parsed."""
    try:
        import pymupdf
        import pymupdf4llm

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            md = pymupdf4llm.to_markdown(doc, show_progress=False)
        return clean_markdown(md) or None
    except Exception as exc:  # parser failures must not abort a batch
        log.warning("pdf_parse_failed", error=type(exc).__name__, detail=str(exc)[:200])
        return None
