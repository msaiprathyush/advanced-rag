"""CLI: python -m advanced_rag.ingestion --ids-file evals/pinned_papers.txt | --query "..." --max 30"""

import argparse
import json
from pathlib import Path

from advanced_rag.clients.arxiv import ArxivClient
from advanced_rag.config import get_settings
from advanced_rag.ingestion.pipeline import ingest_papers


def read_ids(path: str) -> list[str]:
    lines = (ln.split("#")[0].strip() for ln in Path(path).read_text().splitlines())
    return [ln for ln in lines if ln]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file")
    ap.add_argument("--query")
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not (args.ids_file or args.query):
        ap.error("provide --ids-file or --query")

    arxiv = ArxivClient(pdf_cache_dir=".cache/pdfs")
    if args.ids_file:
        metas = arxiv.get_by_ids(read_ids(args.ids_file))
    else:
        metas = arxiv.search(args.query or get_settings().arxiv_query, max_results=args.max)
    report = ingest_papers(metas, force=args.force, arxiv=arxiv)
    print(json.dumps(report.summary() | {"failed_detail": report.failed}, indent=2))


if __name__ == "__main__":
    main()
