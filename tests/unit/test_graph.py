"""Graph branch coverage with a scripted LLM and a stub retriever (no network, no models)."""

from advanced_rag.config import Settings
from advanced_rag.generation.prompts import DECLINE_TEXT
from advanced_rag.graph.build import build_graph, run_graph
from advanced_rag.graph.nodes import GroundednessVerdict
from advanced_rag.retrieval.retriever import RetrievalResult, RetrievedChunk


def chunk(i: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        payload={
            "arxiv_id": f"2310.1151{i}",
            "title": "Self-RAG",
            "section": "Method",
            "abs_url": "https://arxiv.org/abs/2310.11511",
            "text": "Self-RAG uses reflection tokens to decide when to retrieve.",
        },
        score=0.5,
    )


class StubRetriever:
    """Returns a scripted top score per call (last one repeats)."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.queries: list[str] = []

    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        self.queries.append(query)
        score = self.scores[min(len(self.queries) - 1, len(self.scores) - 1)]
        return RetrievalResult([chunk()], score)


class FakeLLM:
    def __init__(self, answers: list[str], verdicts: list[bool], rewrite: str = "rewritten q"):
        self.answers, self.verdicts, self.rewrite = list(answers), list(verdicts), rewrite
        self.calls: list[str] = []
        self.generate_prompts: list[str] = []

    def complete(self, role, messages):
        if role == "judge":
            self.calls.append("rewrite")
            return self.rewrite
        self.calls.append("generate")
        self.generate_prompts.append(messages[0].content)
        return self.answers.pop(0)

    def structured(self, role, messages, schema):
        self.calls.append("judge")
        ok = self.verdicts.pop(0)
        return GroundednessVerdict(grounded=ok, unsupported_claims=[] if ok else ["bad claim"])


S = Settings(rerank_weak_threshold=0.0, max_rewrites=2, max_regenerations=1)


def run(retriever, llm):
    return run_graph("When does Self-RAG retrieve?", build_graph(retriever, llm, S))


def test_happy_path_answers_with_citations():
    llm = FakeLLM(["It retrieves on demand [1]."], [True])
    out = run(StubRetriever([3.0]), llm)
    assert out["decision"] == "answered"
    assert [c.index for c in out["citations"]] == [1]
    assert llm.calls == ["generate", "judge"] and out["llm_calls"] == 2


def test_weak_retrieval_rewrites_then_answers():
    r = StubRetriever([-5.0, 2.0])
    llm = FakeLLM(["Answer [1]."], [True])
    out = run(r, llm)
    assert out["rewrites"] == 1 and out["decision"] == "answered"
    assert r.queries == ["When does Self-RAG retrieve?", "rewritten q"]
    assert llm.calls[0] == "rewrite"


def test_persistently_weak_retrieval_declines_without_generating():
    llm = FakeLLM([], [])
    out = run(StubRetriever([-9.0]), llm)
    assert out["decision"] == "declined" and out["decline_reason"] == "no_relevant_sources"
    assert out["rewrites"] == S.max_rewrites
    assert "generate" not in llm.calls and out["answer"] == DECLINE_TEXT


def test_ungrounded_then_regenerates_with_feedback_and_passes():
    llm = FakeLLM(["Bad [1].", "Good [1]."], [False, True])
    out = run(StubRetriever([3.0]), llm)
    assert out["decision"] == "answered" and out["regenerations"] == 1
    assert "bad claim" in llm.generate_prompts[1]
    assert out["answer"] == "Good [1]."


def test_ungrounded_twice_declines_instead_of_hallucinating():
    llm = FakeLLM(["Bad [1].", "Still bad [1]."], [False, False])
    out = run(StubRetriever([3.0]), llm)
    assert out["decision"] == "declined" and out["decline_reason"] == "could_not_ground_answer"
    assert out["answer"] == DECLINE_TEXT


def test_missing_citations_fail_without_judge_call():
    llm = FakeLLM(["No citations here.", "Fixed [1]."], [True])
    out = run(StubRetriever([3.0]), llm)
    assert out["decision"] == "answered"
    assert llm.calls == ["generate", "generate", "judge"]  # first draft rejected for free


def test_model_declining_is_respected():
    llm = FakeLLM([DECLINE_TEXT], [])
    out = run(StubRetriever([3.0]), llm)
    assert out["decision"] == "declined" and "judge" not in llm.calls


class BrokenJudge(FakeLLM):
    def structured(self, role, messages, schema):
        raise RuntimeError("tool_use_failed")


def test_judge_failure_fails_closed_instead_of_crashing():
    llm = BrokenJudge(["Draft [1].", "Draft again [1]."], [])
    out = run(StubRetriever([3.0]), llm)
    assert out["decision"] == "declined" and out["decline_reason"] == "could_not_ground_answer"
