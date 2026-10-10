"""No-LLM retrieval ablation: dense vs hybrid vs hybrid+rerank.

Run: python -m advanced_rag.eval.retrieval_eval [--k 6]
Metrics are paper-level: a hit is any retrieved chunk from a gold paper.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import yaml

from advanced_rag.eval.dataset import load_eval_set
from advanced_rag.retrieval.retriever import Retriever

MODES = ("dense", "hybrid", "hybrid_rerank")


def ranked_paper_ids(chunk_ids: list[str]) -> list[str]:
    seen: list[str] = []
    for a in chunk_ids:
        if a not in seen:
            seen.append(a)
    return seen


def recall_at_k(ranked: list[str], gold: set[str]) -> float:
    return len(gold & set(ranked)) / len(gold)


def mrr(ranked: list[str], gold: set[str]) -> float:
    for i, a in enumerate(ranked, 1):
        if a in gold:
            return 1.0 / i
    return 0.0


def ndcg(ranked: list[str], gold: set[str]) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, a in enumerate(ranked, 1) if a in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), len(ranked)) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(k: int = 6, path: str = "evals/eval_set.jsonl") -> dict:
    rows = [r for r in load_eval_set(path) if r.answerable and r.gold_arxiv_ids]
    results: dict[str, dict] = {}
    for mode in MODES:
        retriever = Retriever(mode)  # type: ignore[arg-type]
        per_q = {"recall": [], "mrr": [], "ndcg": []}
        top_scores: list[float] = []
        for r in rows:
            res = retriever.retrieve(r.question, k=k)
            ranked = ranked_paper_ids([c.arxiv_id for c in res.chunks])
            gold = set(r.gold_arxiv_ids)
            per_q["recall"].append(recall_at_k(ranked, gold))
            per_q["mrr"].append(mrr(ranked, gold))
            per_q["ndcg"].append(ndcg(ranked, gold))
            if res.top_score is not None:
                top_scores.append(res.top_score)
        results[mode] = {m: round(statistics.mean(v), 3) for m, v in per_q.items()}
        if top_scores:
            results[mode]["answerable_top_score_p10"] = round(
                statistics.quantiles(top_scores, n=10)[0], 2
            )
    unanswerable = [r for r in load_eval_set(path) if not r.answerable]
    if unanswerable:
        rr = Retriever("hybrid_rerank")
        scores = [rr.retrieve(r.question, k=k).top_score or 0.0 for r in unanswerable]
        results["unanswerable_top_scores"] = {
            "max": round(max(scores), 2),
            "mean": round(statistics.mean(scores), 2),
        }
    return {"k": k, "n_questions": len(rows), "results": results}


def to_markdown(report: dict) -> str:
    lines = [
        f"Retrieval ablation (paper-level, k={report['k']}, n={report['n_questions']})",
        "",
        "| mode | recall@k | MRR | nDCG |",
        "|---|---|---|---|",
    ]
    for mode in MODES:
        r = report["results"][mode]
        lines.append(f"| {mode} | {r['recall']} | {r['mrr']} | {r['ndcg']} |")
    if "unanswerable_top_scores" in report["results"]:
        u = report["results"]["unanswerable_top_scores"]
        lines += [
            "",
            f"Top rerank score on unanswerable questions: max={u['max']}, mean={u['mean']}",
        ]
        p10 = report["results"]["hybrid_rerank"].get("answerable_top_score_p10")
        if p10 is not None:
            lines.append(f"Answerable top-score p10: {p10} (pick rerank_weak_threshold between)")
    return "\n".join(lines)


def retrieval_gate(report: dict, thresholds: dict) -> list[str]:
    """Deterministic, LLM-free quality gate on the production retrieval mode (hybrid_rerank)."""
    r = report["results"]["hybrid_rerank"]
    failures = []
    for key, metric in (("retrieval_recall", "recall"), ("retrieval_mrr", "mrr")):
        floor = thresholds.get(key)
        if floor is not None and r[metric] < floor:
            failures.append(f"{key}: {r[metric]} < min {floor}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="exit non-zero below evals/thresholds.yaml")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--path", default="evals/eval_set.jsonl")
    args = ap.parse_args()
    report = evaluate(args.k, args.path)
    out = Path("evals/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_ablation.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (out / "retrieval_ablation.md").write_text(md + "\n")
    print(md)
    if args.gate:
        path = Path("evals/thresholds.yaml")
        failures = retrieval_gate(report, yaml.safe_load(path.read_text()) if path.exists() else {})
        for f in failures:
            print(f"GATE FAILED: {f}")
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
