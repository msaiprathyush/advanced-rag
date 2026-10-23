"""CLI: python -m advanced_rag.ingestion --ids-file evals/pinned_papers.txt | --query "..." --max 30"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from advanced_rag.clients.arxiv import ArxivClient
from advanced_rag.config import get_settings
from advanced_rag.ingestion.models import PaperMeta
from advanced_rag.ingestion.pipeline import ingest_papers


def read_ids(path: str) -> list[str]:
    lines = (ln.split("#")[0].strip() for ln in Path(path).read_text().splitlines())
    return [ln for ln in lines if ln]


CACHE_DIR = Path(".cache/pdfs")


def cached_metadata(arxiv: ArxivClient, ids: list[str]) -> list[PaperMeta]:
    """Paper metadata for a fixed ID list, cached next to the PDFs.

    CI already caches the pinned PDFs, but still asked arXiv for metadata on every run, and arXiv
    sometimes rate-limits shared CI runners (a nightly eval failed on it). With this, a cache hit
    means zero arXiv requests.
    """
    key = hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()[:16]
    path = CACHE_DIR / f"metadata-{key}.json"
    if path.exists():
        return [PaperMeta(**m) for m in json.loads(path.read_text())]
    metas = arxiv.get_by_ids(ids)
    if len(metas) == len(ids):  # only cache a complete answer
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(m) for m in metas]))
    return metas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file")
    ap.add_argument("--query")
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not (args.ids_file or args.query):
        ap.error("provide --ids-file or --query")

    arxiv = ArxivClient(pdf_cache_dir=CACHE_DIR)
    if args.ids_file:
        metas = cached_metadata(arxiv, read_ids(args.ids_file))
    else:
        metas = arxiv.search(args.query or get_settings().arxiv_query, max_results=args.max)
    report = ingest_papers(metas, force=args.force, arxiv=arxiv)
    print(json.dumps(report.summary() | {"failed_detail": report.failed}, indent=2))


if __name__ == "__main__":
    main()
