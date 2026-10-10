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


def test_retrieval_gate_blocks_regressions():
    from advanced_rag.eval.retrieval_eval import retrieval_gate

    good = {"results": {"hybrid_rerank": {"recall": 0.95, "mrr": 0.9}}}
    bad = {"results": {"hybrid_rerank": {"recall": 0.7, "mrr": 0.9}}}
    th = {"retrieval_recall": 0.9, "retrieval_mrr": 0.85}
    assert retrieval_gate(good, th) == []
    assert retrieval_gate(bad, th) == ["retrieval_recall: 0.7 < min 0.9"]


def test_rotating_subset_advances_daily_and_covers_the_set():
    from advanced_rag.eval.dataset import EvalRow
    from advanced_rag.eval.ragas_eval import rotating_subset

    rows = [EvalRow(id=f"q{i}", question="?", answerable=True) for i in range(10)]
    rows += [EvalRow(id="u0", question="?", answerable=False)]
    seen = set()
    for day in range(3):
        sub = rotating_subset(rows, 4, day)
        assert len([r for r in sub if r.answerable]) == 4 and sub[-1].id == "u0"
        seen |= {r.id for r in sub if r.answerable}
    assert len(seen) == 10  # three runs of 4 cover all 10 questions


def test_ragas_gate_ignores_retrieval_thresholds():
    assert gate({"faithfulness": 0.9}, {"retrieval_recall": 0.9, "faithfulness": 0.8}) == []
