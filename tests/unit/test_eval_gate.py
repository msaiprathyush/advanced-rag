from advanced_rag.eval.ragas_eval import gate
from advanced_rag.eval.retrieval_eval import mrr, ndcg, recall_at_k


def test_gate_passes_when_all_metrics_meet_thresholds():
    m = {
        "ragas_samples": 10,
        "faithfulness": 0.9,
        "false_decline_rate_answerable": 0.1,
        **{
            f"{n}_n": 9
            for n in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        },
    }
    assert gate(m, {"faithfulness": 0.8, "false_decline_rate_answerable": 0.2}) == []


def test_gate_fails_below_min_and_above_max():
    m = {"faithfulness": 0.5, "false_decline_rate_answerable": 0.5}
    f = gate(m, {"faithfulness": 0.8, "false_decline_rate_answerable": 0.2})
    assert len(f) == 2


def test_gate_rejects_a_score_based_on_too_few_samples():
    # 2 of 30 samples scored (rate-limited judge): a perfect score must NOT pass the gate
    m = {"ragas_samples": 30, "faithfulness": 1.0, "faithfulness_n": 2}
    assert any("only 2/30" in x for x in gate(m, {"faithfulness": 0.8}))


def test_gate_fails_on_missing_metric():
    assert gate({}, {"context_recall": 0.6}) == ["context_recall: not computed"]


def test_retrieval_metrics():
    assert recall_at_k(["a", "b"], {"a", "c"}) == 0.5
    assert mrr(["x", "a"], {"a"}) == 0.5 and mrr(["x"], {"a"}) == 0.0
    assert ndcg(["a", "b"], {"a"}) == 1.0 and 0 < ndcg(["b", "a"], {"a"}) < 1
