import httpx
import respx

from advanced_rag.config import Settings
from advanced_rag.monitoring import drift
from advanced_rag.monitoring.drift import TrafficStats, decide, traffic_stats

CFG = Settings(
    drift_min_recall=0.9,
    drift_min_mrr=0.85,
    drift_min_queries=20,
    drift_max_decline_rate=0.35,
    drift_min_median_top_score=2.0,
    drift_cooldown_hours=24,
)
GOOD_CANARY = {"recall": 1.0, "mrr": 0.98}
GOOD_TRAFFIC = TrafficStats(n_queries=50, decline_rate=0.1, median_top_score=5.0)


def q(decision="answered", score=5.0, mode="advanced"):
    return {"decision": decision, "top_score": score, "mode": mode}


def test_traffic_stats_ignores_naive_mode_and_computes_median():
    t = traffic_stats([q(), q("declined", 1.0), q(score=7.0), q(mode="naive", score=-99)])
    assert t.n_queries == 3 and t.decline_rate == 0.333 and t.median_top_score == 5.0
    assert traffic_stats([]) == TrafficStats()


def test_healthy_system_does_nothing():
    d = decide(GOOD_CANARY, GOOD_TRAFFIC, CFG, None)
    assert d.action == "none" and d.breaches == []


def test_canary_breach_triggers_reindex():
    d = decide({"recall": 0.7, "mrr": 0.98}, GOOD_TRAFFIC, CFG, None)
    assert d.action == "reindex" and "canary recall" in d.breaches[0]


def test_rising_decline_rate_triggers_reindex():
    t = TrafficStats(n_queries=50, decline_rate=0.6, median_top_score=5.0)
    assert decide(GOOD_CANARY, t, CFG, 100.0).action == "reindex"


def test_falling_top_scores_trigger_reindex():
    t = TrafficStats(n_queries=50, decline_rate=0.1, median_top_score=0.5)
    d = decide(GOOD_CANARY, t, CFG, None)
    assert d.action == "reindex" and "median top score" in d.breaches[0]


def test_too_little_traffic_never_triggers_on_live_signals():
    t = TrafficStats(n_queries=3, decline_rate=1.0, median_top_score=-9.0)
    d = decide(GOOD_CANARY, t, CFG, None)
    assert d.action == "none" and any("skipped" in n for n in d.notes)


def test_cooldown_prevents_reindex_loops():
    d = decide({"recall": 0.5, "mrr": 0.5}, GOOD_TRAFFIC, CFG, hours_since_last_reindex=2.0)
    assert d.action == "cooldown" and d.breaches


@respx.mock
def test_trigger_reindex_calls_the_run_api_with_the_metadata_token():
    respx.get(f"{drift._METADATA}/instance/service-accounts/default/token").mock(
        return_value=httpx.Response(200, json={"access_token": "tok"})
    )
    route = respx.post(
        "https://run.googleapis.com/v2/projects/p/locations/us-central1/jobs/arxiv-reindex:run"
    ).mock(return_value=httpx.Response(200, json={"name": "operations/abc"}))
    with httpx.Client() as http:
        assert drift.trigger_reindex(http, "p", CFG) == "operations/abc"
    assert route.calls[0].request.headers["authorization"] == "Bearer tok"


@respx.mock
def test_hours_since_last_reindex_parses_execution_time():
    respx.get(f"{drift._METADATA}/instance/service-accounts/default/token").mock(
        return_value=httpx.Response(200, json={"access_token": "tok"})
    )
    respx.get(
        "https://run.googleapis.com/v2/projects/p/locations/us-central1/jobs/arxiv-reindex/executions"
    ).mock(
        return_value=httpx.Response(
            200, json={"executions": [{"createTime": "2020-01-01T00:00:00Z"}]}
        )
    )
    with httpx.Client() as http:
        assert drift.hours_since_last_reindex(http, "p", CFG) > 24
