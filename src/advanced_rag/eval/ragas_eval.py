"""RAGAS evaluation + CI threshold gate.

Run: python -m advanced_rag.eval.ragas_eval [--limit N] [--no-gate]

1. Run the full graph over every eval question (answers cached in evals/reports/answers.jsonl so a
   rate-limited run can be resumed).
2. Decline metrics (no LLM): decline rate on unanswerable, false-decline rate on answerable.
3. RAGAS (LLM-as-judge via Groq) on the answered, answerable rows.
4. Compare against evals/thresholds.yaml; exit non-zero if any metric is below its floor.
"""

import argparse
import json
import os
import statistics
import sys
from datetime import date
from pathlib import Path

import yaml

from advanced_rag.eval.dataset import EvalRow, load_eval_set
from advanced_rag.logging import get_logger

log = get_logger("ragas_eval")
REPORTS = Path("evals/reports")
THRESHOLDS = Path("evals/thresholds.yaml")
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def run_graph_rows(rows: list[EvalRow], cache_path: Path) -> dict[str, dict]:
    from advanced_rag.graph.build import build_graph, run_graph

    cache: dict[str, dict] = {}
    if cache_path.exists():
        for ln in cache_path.read_text().splitlines():
            rec = json.loads(ln)
            cache[rec["id"]] = rec
    graph = None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a") as f:
        for r in rows:
            if r.id in cache:
                continue
            graph = graph or build_graph()
            out = run_graph(r.question, graph)
            rec = {
                "id": r.id,
                "answer": out["answer"],
                "decision": out["decision"],
                "grounded": out.get("grounded"),
                "rewrites": out.get("rewrites", 0),
                "llm_calls": out.get("llm_calls", 0),
                "contexts": [c.text for c in out.get("docs", [])],
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            cache[r.id] = rec
            log.info("eval_row_done", id=r.id, decision=rec["decision"])
    return cache


def decline_metrics(rows: list[EvalRow], results: dict[str, dict]) -> dict[str, float]:
    ans = [r for r in rows if r.answerable and r.id in results]
    unans = [r for r in rows if not r.answerable and r.id in results]
    out: dict[str, float] = {}
    if unans:
        out["decline_rate_unanswerable"] = sum(
            results[r.id]["decision"] == "declined" for r in unans
        ) / len(unans)
    if ans:
        out["false_decline_rate_answerable"] = sum(
            results[r.id]["decision"] == "declined" for r in ans
        ) / len(ans)
        out["avg_llm_calls"] = statistics.mean(results[r.id]["llm_calls"] for r in ans)
    return {k: round(v, 3) for k, v in out.items()}


def ragas_scores(rows: list[EvalRow], results: dict[str, dict]) -> dict[str, float]:
    from langchain_groq import ChatGroq
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings.base import BaseRagasEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    from advanced_rag.config import get_settings
    from advanced_rag.embeddings.models import embed_passages, embed_query

    class LocalEmbeddings(BaseRagasEmbeddings):
        """Adapter over our own ONNX embedder: same model as production, no extra wrapper deps."""

        def embed_query(self, text: str) -> list[float]:
            return embed_query(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return embed_passages(texts)

        async def aembed_query(self, text: str) -> list[float]:
            return embed_query(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return embed_passages(texts)

    samples = [
        {
            "user_input": r.question,
            "response": results[r.id]["answer"],
            "retrieved_contexts": results[r.id]["contexts"],
            "reference": r.reference_answer,
        }
        for r in rows
        if r.answerable and r.id in results and results[r.id]["decision"] == "answered"
    ]
    if not samples:
        return {}
    cfg = get_settings()
    # Groq free tier is 8,000 tokens/minute and one RAGAS judge prompt is 3-4k tokens, so the judge
    # must WAIT on 429s (the SDK honours Retry-After) instead of failing the job.
    llm = LangchainLLMWrapper(
        ChatGroq(
            model=cfg.judge_model,
            api_key=cfg.groq_api_key,
            temperature=0,
            max_retries=12,
            timeout=180,
            # gpt-oss spends hidden reasoning tokens from the output budget; without this the
            # faithfulness prompts hit max_tokens before finishing (LLMDidNotFinishException).
            reasoning_effort="low",
            max_tokens=3000,
        )
    )
    emb = LocalEmbeddings()
    result = evaluate(
        EvaluationDataset.from_list(samples),
        metrics=[
            Faithfulness(),
            ResponseRelevancy(strictness=1),  # Groq only supports n=1
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(max_workers=1, max_retries=10, max_wait=90, timeout=400),
        raise_exceptions=False,
    )
    df = result.to_pandas()
    scores: dict[str, float] = {"ragas_samples": len(samples)}
    for col, name in (
        ("faithfulness", "faithfulness"),
        ("answer_relevancy", "answer_relevancy"),
        ("llm_context_precision_with_reference", "context_precision"),
        ("context_recall", "context_recall"),
    ):
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals):
                scores[name] = round(float(vals.mean()), 3)
                scores[f"{name}_n"] = len(vals)
    return scores


# metrics where lower is better
LOWER_IS_BETTER = {"false_decline_rate_answerable"}


MIN_SCORED_RATIO = 0.7  # a metric must be scored on >= 70% of samples to count


def gate(metrics: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    failures = []
    total = metrics.get("ragas_samples")
    for name in RAGAS_METRICS:
        n = metrics.get(f"{name}_n")
        if total and n is not None and n < MIN_SCORED_RATIO * total:
            failures.append(f"{name}: only {n}/{int(total)} samples scored (judge rate-limited?)")
        elif total and n is None:
            failures.append(f"{name}: not scored on any sample")
    for name, limit in thresholds.items():
        if name.startswith("retrieval_"):
            continue  # enforced separately by retrieval_eval --gate
        if name not in metrics:
            failures.append(f"{name}: not computed")
        elif name in LOWER_IS_BETTER and metrics[name] > limit:
            failures.append(f"{name}: {metrics[name]} > max {limit}")
        elif name not in LOWER_IS_BETTER and metrics[name] < limit:
            failures.append(f"{name}: {metrics[name]} < min {limit}")
    return failures


def to_markdown(metrics: dict, thresholds: dict, failures: list[str], n: int) -> str:
    lines = [
        f"### RAG evaluation (n={n})",
        "",
        "| metric | value | threshold | |",
        "|---|---|---|---|",
    ]
    for k, v in metrics.items():
        if k.endswith("_n") or k == "ragas_samples":
            continue
        t = thresholds.get(k)
        bad = any(f.startswith(k + ":") for f in failures)
        sign = "max" if k in LOWER_IS_BETTER else "min"
        lines.append(
            f"| {k} | {v} | {f'{sign} {t}' if t is not None else '-'} | {'FAIL' if bad else 'ok'} |"
        )
    if failures:
        lines += ["", "**Gate failed:**", *[f"- {f}" for f in failures]]
    return "\n".join(lines)


def rotating_subset(rows: list[EvalRow], k: int, day: int) -> list[EvalRow]:
    """K answerable rows (a window that advances with `day`) plus one unanswerable row.

    The Groq free tier allows 200k tokens/day per model, so each run scores a few questions and
    consecutive days cover the full set.
    """
    answerable = [r for r in rows if r.answerable]
    unanswerable = [r for r in rows if not r.answerable]
    start = (day * k) % len(answerable)
    window = [answerable[(start + i) % len(answerable)] for i in range(min(k, len(answerable)))]
    return window + ([unanswerable[day % len(unanswerable)]] if unanswerable else [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--rotate",
        type=int,
        default=None,
        help="score K answerable rows, advancing daily so a week covers the whole set",
    )
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--skip-ragas", action="store_true")
    ap.add_argument("--path", default="evals/eval_set.jsonl")
    args = ap.parse_args()

    rows = load_eval_set(args.path)
    if args.rotate:
        rows = rotating_subset(rows, args.rotate, date.today().toordinal())
    elif args.limit:
        answerable = [r for r in rows if r.answerable][: args.limit]
        rows = answerable + [r for r in rows if not r.answerable][: max(1, args.limit // 5)]
    results = run_graph_rows(rows, REPORTS / "answers.jsonl")

    metrics = decline_metrics(rows, results)
    if not args.skip_ragas:
        metrics |= ragas_scores(rows, results)

    thresholds = yaml.safe_load(THRESHOLDS.read_text()) if THRESHOLDS.exists() else {}
    failures = gate(metrics, thresholds)  # always shown; --no-gate only affects the exit code
    md = to_markdown(metrics, thresholds, failures, len(rows))
    (REPORTS / "ragas_report.json").write_text(
        json.dumps({"metrics": metrics, "failures": failures}, indent=2)
    )
    (REPORTS / "ragas_report.md").write_text(md + "\n")
    print(md)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as f:
            f.write(md + "\n")
    return 1 if failures and not args.no_gate else 0


if __name__ == "__main__":
    sys.exit(main())
