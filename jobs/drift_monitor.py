"""Drift monitor (Cloud Run Job, run on a cheap schedule): re-index ONLY if quality has drifted.

Uses no LLM tokens. Run: python -m jobs.drift_monitor
Exit code is non-zero only when the monitor itself fails or drift persists inside the cooldown.
"""

import sys

import httpx

from advanced_rag.config import get_settings
from advanced_rag.eval.retrieval_eval import evaluate
from advanced_rag.logging import get_logger
from advanced_rag.monitoring import drift

log = get_logger("drift_monitor")


def run() -> int:
    cfg = get_settings()
    report = evaluate(k=cfg.final_k, modes=("hybrid_rerank",), include_unanswerable=False)
    canary = report["results"]["hybrid_rerank"]
    log.info("drift_canary", n_questions=report["n_questions"], **canary)

    traffic = drift.TrafficStats()
    hours = None
    project = None
    with httpx.Client(timeout=30) as http:
        try:
            project = drift.project_id(http)
        except httpx.HTTPError:
            log.warning("drift_not_on_gcp", detail="metadata server unreachable; canary only")
        if project:
            traffic = drift.traffic_stats(
                drift.fetch_query_logs(http, project, cfg.drift_window_days)
            )
            hours = drift.hours_since_last_reindex(http, project, cfg)
        log.info(
            "drift_traffic",
            n_queries=traffic.n_queries,
            decline_rate=traffic.decline_rate,
            median_top_score=traffic.median_top_score,
            hours_since_last_reindex=None if hours is None else round(hours, 1),
        )

        decision = drift.decide(canary, traffic, cfg, hours)
        log.info(
            "drift_decision",
            action=decision.action,
            breaches=decision.breaches,
            notes=decision.notes,
        )
        if decision.action == "reindex":
            if cfg.drift_dry_run or not project:
                log.warning("drift_reindex_skipped", reason="dry_run_or_not_on_gcp")
            else:
                execution = drift.trigger_reindex(http, project, cfg)
                log.warning("drift_reindex_triggered", execution=execution)
        elif decision.action == "cooldown":
            return 1  # drift persists even after a recent re-index: make the failure visible
    return 0


if __name__ == "__main__":
    sys.exit(run())
