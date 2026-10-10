"""Idempotent ingestion: skip known versions, re-ingest updated ones, isolate per-paper failures."""

import time
from dataclasses import dataclass, field

from qdrant_client import QdrantClient

from advanced_rag.clients.arxiv import ArxivClient
from advanced_rag.embeddings.models import embed_passages, sparse_passages
from advanced_rag.ingestion.chunker import chunk_paper
from advanced_rag.ingestion.models import PaperMeta
from advanced_rag.ingestion.parser import parse_pdf
from advanced_rag.logging import get_logger
from advanced_rag.store import qdrant as store

log = get_logger(__name__)

# 16, not 64: ONNX activation memory scales with batch size. Measured peak RSS over 10 papers was
# 3.4 GB at 64 (OOM-killed the 2 GiB Cloud Run Job) vs 1.27 GB at 16, with flat growth.
EMBED_BATCH = 16


@dataclass
class IngestReport:
    new: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    abstract_only: list[str] = field(default_factory=list)
    chunks: int = 0

    def summary(self) -> dict:
        return {
            "new": len(self.new),
            "updated": len(self.updated),
            "skipped": len(self.skipped),
            "failed": len(self.failed),
            "abstract_only": len(self.abstract_only),
            "chunks": self.chunks,
        }


def _ingest_one(
    meta: PaperMeta, arxiv: ArxivClient, client: QdrantClient, report: IngestReport
) -> None:
    pdf = arxiv.fetch_pdf(meta.arxiv_id, meta.version)
    markdown = parse_pdf(pdf)
    chunks = chunk_paper(meta, markdown)
    if markdown is None:
        report.abstract_only.append(meta.arxiv_id)
        log.warning("abstract_only", arxiv_id=meta.arxiv_id)
    # Commit marker: the abstract chunk (index 0) is written LAST, and existing_versions() only
    # counts papers that have it. A run killed mid-paper therefore leaves a paper that is
    # retried (and its orphan chunks overwritten by the idempotent upsert), never skipped.
    ordered = [*chunks[1:], chunks[0]]
    for i in range(0, len(ordered), EMBED_BATCH):
        batch = ordered[i : i + EMBED_BATCH]
        texts = [c.embed_text for c in batch]
        store.upsert_chunks(batch, embed_passages(texts), sparse_passages(texts), client)
    report.chunks += len(chunks)


def ingest_papers(
    metas: list[PaperMeta],
    *,
    force: bool = False,
    arxiv: ArxivClient | None = None,
    client: QdrantClient | None = None,
) -> IngestReport:
    arxiv = arxiv or ArxivClient()
    client = client or store.get_client()
    store.ensure_collection(client)
    report = IngestReport()
    known = {} if force else store.existing_versions([m.arxiv_id for m in metas], client)

    for meta in metas:
        have = known.get(meta.arxiv_id)
        if have is not None and have >= meta.version:
            report.skipped.append(meta.arxiv_id)
            continue
        t0 = time.perf_counter()
        try:
            if have is not None or force:
                store.delete_paper(meta.arxiv_id, client)
            _ingest_one(meta, arxiv, client, report)
            (report.updated if have is not None else report.new).append(meta.arxiv_id)
            log.info(
                "paper_ingested",
                arxiv_id=meta.arxiv_id,
                version=meta.version,
                latency_ms=round((time.perf_counter() - t0) * 1000),
            )
        except Exception as exc:
            report.failed[meta.arxiv_id] = f"{type(exc).__name__}: {exc}"[:300]
            log.error(
                "paper_ingest_failed", arxiv_id=meta.arxiv_id, error=report.failed[meta.arxiv_id]
            )
    log.info("ingest_summary", **report.summary())
    return report
