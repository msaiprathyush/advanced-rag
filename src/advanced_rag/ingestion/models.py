from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class PaperMeta:
    arxiv_id: str  # base id without version, e.g. "2310.11511"
    version: int
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str  # ISO-8601
    updated: str
    abs_url: str
    pdf_url: str


@dataclass
class Chunk:
    arxiv_id: str
    version: int
    chunk_index: int
    section: str
    text: str  # raw text shown to the LLM / user
    embed_text: str  # text actually embedded (with contextual header)
    title: str
    authors: list[str]
    categories: list[str]
    published: str
    abs_url: str
    pdf_url: str
    ingested_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def payload(self) -> dict:
        p = asdict(self)
        p.pop("embed_text")
        return p
