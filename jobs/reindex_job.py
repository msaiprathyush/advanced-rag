"""Scheduled maintenance job (Cloud Run Job): pull new arXiv papers, re-embed, enforce retention.

Run: python -m jobs.reindex_job
"""

import sys
import time
from pathlib import Path

from advanced_rag.clients.arxiv import ArxivClient
from advanced_rag.config import get_settings
from advanced_rag.ingestion.pipeline import ingest_papers
from advanced_rag.logging import get_logger
from advanced_rag.store import qdrant as store

log = get_logger("reindex_job")
PAGE = 50
FAIL_EXIT_RATIO = 0.5


def read_pinned(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    lines = (ln.split("#")[0].strip() for ln in p.read_text().splitlines())
    return [ln for ln in lines if ln]


def distinct_papers(client, collection: str) -> dict[str, str]:
    """arxiv_id -> published, across the whole collection (payload-only scroll)."""
    papers: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection,
            limit=512,
            offset=offset,
            with_payload=["arxiv_id", "published"],
            with_vectors=False,
        )
        for p in points:
            papers[p.payload["arxiv_id"]] = p.payload["published"]
        if offset is None:
            return papers


def evict_oldest(client, keep: int, pinned: set[str]) -> list[str]:
    """Retention policy: keep the newest `keep` papers (pinned eval papers never evicted)."""
    collection = get_settings().collection
    papers = distinct_papers(client, collection)
    excess = len(papers) - keep
    if excess <= 0:
        return []
    candidates = sorted((pub, aid) for aid, pub in papers.items() if aid not in pinned)
    victims = [aid for _, aid in candidates[:excess]]
    for aid in victims:
        store.delete_paper(aid, client)
    return victims


def fetch_new(arxiv: ArxivClient, query: str, max_new: int) -> list:
    metas = []
    start = 0
    while len(metas) < max_new:
        page = arxiv.search(query, start=start, max_results=min(PAGE, max_new - len(metas)))
        if not page:
            break
        metas.extend(page)
        start += len(page)
    return metas


def run() -> int:
    s = get_settings()
    t0 = time.perf_counter()
    client = store.get_client()
    store.ensure_collection(client)
    arxiv = ArxivClient(pdf_cache_dir=".cache/pdfs")

    pinned = read_pinned(s.pinned_papers_file)
    metas = arxiv.get_by_ids(pinned) if pinned else []
    metas += fetch_new(arxiv, s.arxiv_query, s.reindex_max_new)
    seen: set[str] = set()
    metas = [m for m in metas if not (m.arxiv_id in seen or seen.add(m.arxiv_id))]

    report = ingest_papers(metas, arxiv=arxiv, client=client)
    evicted = evict_oldest(client, s.max_papers_in_index, set(pinned))
    total = len(report.new) + len(report.updated) + len(report.skipped) + len(report.failed)
    log.info(
        "reindex_summary",
        **report.summary(),
        evicted=len(evicted),
        duration_s=round(time.perf_counter() - t0, 1),
        index_points=store.count_points(client),
    )
    attempted = len(report.new) + len(report.updated) + len(report.failed)
    if attempted and len(report.failed) / attempted > FAIL_EXIT_RATIO:
        return 1  # non-zero -> Cloud Run Job retries / alerting
    return 0 if total or not metas else 1


if __name__ == "__main__":
    sys.exit(run())
