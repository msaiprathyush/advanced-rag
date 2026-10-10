"""Drift monitor: decide, cheaply and without LLM calls, whether the RAG system needs a re-index.

Two signals:
  A. Retrieval canary: the fixed eval questions run through retrieval only against the PRODUCTION
     index. Catches index damage / regressions (e.g. evicted or missing papers).
  B. Live-traffic drift: decline rate and median top re-rank score over recent /query traffic, read
     from Cloud Logging. Catches users asking about things the corpus no longer covers (staleness).

Limitation: signal A uses a fixed question set about pinned papers, so it cannot see corpus
staleness; that is what signal B is for, and B needs enough traffic to be meaningful.
"""

import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx

from advanced_rag.config import Settings
from advanced_rag.logging import get_logger

log = get_logger(__name__)

_METADATA = "http://metadata.google.internal/computeMetadata/v1"
_HEADERS = {"Metadata-Flavor": "Google"}


# ---------------------------------------------------------------- pure decision logic


@dataclass
class TrafficStats:
    n_queries: int = 0
    decline_rate: float | None = None
    median_top_score: float | None = None


@dataclass
class Decision:
    breaches: list[str] = field(default_factory=list)
    action: str = "none"  # none | reindex | cooldown | insufficient_data_only
    notes: list[str] = field(default_factory=list)


def traffic_stats(entries: list[dict]) -> TrafficStats:
    """Summarise `query_done` log payloads (advanced-mode queries only)."""
    adv = [e for e in entries if e.get("mode", "advanced") == "advanced"]
    if not adv:
        return TrafficStats()
    declined = sum(e.get("decision") == "declined" for e in adv)
    scores = [float(e["top_score"]) for e in adv if e.get("top_score") is not None]
    return TrafficStats(
        n_queries=len(adv),
        decline_rate=round(declined / len(adv), 3),
        median_top_score=round(statistics.median(scores), 2) if scores else None,
    )


def decide(
    canary: dict[str, float],
    traffic: TrafficStats,
    cfg: Settings,
    hours_since_last_reindex: float | None,
) -> Decision:
    d = Decision()
    if canary["recall"] < cfg.drift_min_recall:
        d.breaches.append(f"canary recall {canary['recall']} < {cfg.drift_min_recall}")
    if canary["mrr"] < cfg.drift_min_mrr:
        d.breaches.append(f"canary mrr {canary['mrr']} < {cfg.drift_min_mrr}")

    if traffic.n_queries < cfg.drift_min_queries:
        d.notes.append(
            f"live-traffic signal skipped: {traffic.n_queries} queries < {cfg.drift_min_queries}"
        )
    else:
        if traffic.decline_rate is not None and traffic.decline_rate > cfg.drift_max_decline_rate:
            d.breaches.append(f"decline rate {traffic.decline_rate} > {cfg.drift_max_decline_rate}")
        if (
            traffic.median_top_score is not None
            and traffic.median_top_score < cfg.drift_min_median_top_score
        ):
            d.breaches.append(
                f"median top score {traffic.median_top_score} < {cfg.drift_min_median_top_score}"
            )

    if not d.breaches:
        return d
    if hours_since_last_reindex is not None and hours_since_last_reindex < cfg.drift_cooldown_hours:
        d.action = "cooldown"
        d.notes.append(
            f"re-index ran {hours_since_last_reindex:.1f} h ago (< {cfg.drift_cooldown_hours} h "
            "cooldown); drift persists after a re-index, so investigate manually"
        )
        return d
    d.action = "reindex"
    return d


# ---------------------------------------------------------------- Google API (metadata token)


def _token(http: httpx.Client) -> str:
    r = http.get(f"{_METADATA}/instance/service-accounts/default/token", headers=_HEADERS)
    r.raise_for_status()
    return r.json()["access_token"]


def project_id(http: httpx.Client) -> str:
    r = http.get(f"{_METADATA}/project/project-id", headers=_HEADERS)
    r.raise_for_status()
    return r.text.strip()


def fetch_query_logs(http: httpx.Client, project: str, days: int) -> list[dict]:
    since = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    flt = (
        'resource.type="cloud_run_revision" AND resource.labels.service_name="advanced-rag-api" '
        f'AND jsonPayload.event="query_done" AND timestamp>="{since}"'
    )
    headers = {"Authorization": f"Bearer {_token(http)}"}
    entries: list[dict] = []
    page_token = None
    for _ in range(10):  # bounded: at most 10 x 1000 entries
        body = {"resourceNames": [f"projects/{project}"], "filter": flt, "pageSize": 1000}
        if page_token:
            body["pageToken"] = page_token
        r = http.post("https://logging.googleapis.com/v2/entries:list", json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
        entries += [e["jsonPayload"] for e in data.get("entries", []) if "jsonPayload" in e]
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return entries


def hours_since_last_reindex(http: httpx.Client, project: str, cfg: Settings) -> float | None:
    url = (
        f"https://run.googleapis.com/v2/projects/{project}/locations/{cfg.gcp_region}"
        f"/jobs/{cfg.reindex_job_name}/executions?pageSize=1"
    )
    r = http.get(url, headers={"Authorization": f"Bearer {_token(http)}"})
    r.raise_for_status()
    execs = r.json().get("executions", [])
    if not execs:
        return None
    created = datetime.fromisoformat(execs[0]["createTime"].replace("Z", "+00:00"))
    return (datetime.now(UTC) - created).total_seconds() / 3600


def trigger_reindex(http: httpx.Client, project: str, cfg: Settings) -> str:
    url = (
        f"https://run.googleapis.com/v2/projects/{project}/locations/{cfg.gcp_region}"
        f"/jobs/{cfg.reindex_job_name}:run"
    )
    r = http.post(url, json={}, headers={"Authorization": f"Bearer {_token(http)}"})
    r.raise_for_status()
    return r.json().get("name", "")
