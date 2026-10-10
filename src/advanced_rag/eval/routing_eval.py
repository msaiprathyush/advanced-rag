"""Routing eval: is a question about the collection itself, or about paper content?

Paraphrases of the same intent must route the same way, and near-misses ("What does the Self-RAG
paper talk about?") must NOT be routed to the collection overview.
Run: python -m advanced_rag.eval.routing_eval [--gate 0.9]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from advanced_rag.clients.llm import get_llm
from advanced_rag.graph.nodes import is_collection_question

DATASET = Path("evals/routing_set.jsonl")


def run(dataset: Path = DATASET, pace_s: float = 1.0) -> dict:
    rows = [json.loads(ln) for ln in dataset.read_text().splitlines() if ln.strip()]
    llm = get_llm()
    errors = []
    for r in rows:
        got = is_collection_question(llm, r["question"])
        if got != r["collection"]:
            errors.append({"question": r["question"], "expected": r["collection"], "got": got})
        time.sleep(pace_s)  # stay well under the free tier's tokens-per-minute limit
    pos = [r for r in rows if r["collection"]]
    neg = [r for r in rows if not r["collection"]]
    missed = [e for e in errors if e["expected"]]
    false_routes = [e for e in errors if not e["expected"]]
    return {
        "n": len(rows),
        "accuracy": round(1 - len(errors) / len(rows), 3),
        "collection_recall": round(1 - len(missed) / len(pos), 3),
        "content_specificity": round(1 - len(false_routes) / len(neg), 3),
        "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=float, default=None, help="minimum accuracy to pass")
    ap.add_argument("--dataset", default=str(DATASET))
    args = ap.parse_args()
    report = run(Path(args.dataset))
    print(json.dumps(report, indent=2))
    out = Path("evals/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"routing_eval_{Path(args.dataset).stem}.json").write_text(json.dumps(report, indent=2))
    if args.gate is not None and report["accuracy"] < args.gate:
        print(f"GATE FAILED: routing accuracy {report['accuracy']} < {args.gate}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
