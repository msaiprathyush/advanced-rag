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
        if messages[0].content.startswith("You route questions"):
            self.calls.append("scope")  # every question is routed once, before retrieval
            return "NO"
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
    assert llm.calls == ["scope", "generate", "judge"] and out["llm_calls"] == 3


def test_weak_retrieval_rewrites_then_answers():
    r = StubRetriever([-5.0, 2.0])
    llm = FakeLLM(["Answer [1]."], [True])
    out = run(r, llm)
    assert out["rewrites"] == 1 and out["decision"] == "answered"
    assert r.queries == ["When does Self-RAG retrieve?", "rewritten q"]
    assert llm.calls[:2] == ["scope", "rewrite"]


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
    assert llm.calls == ["scope", "generate", "generate", "judge"]  # draft rejected for free


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


def test_followup_is_condensed_into_a_standalone_question_before_retrieval():
    class CondensingLLM(FakeLLM):
        def complete(self, role, messages):
            if role == "judge" and "Follow-up question" in messages[-1].content:
                self.calls.append("condense")
                return "What are the limitations of Self-RAG?"
            return super().complete(role, messages)

    r = StubRetriever([3.0])
    llm = CondensingLLM(["Limits are X [1]."], [True])
    history = [
        {"role": "user", "content": "How does Self-RAG work?"},
        {"role": "assistant", "content": "It uses reflection tokens [1]."},
    ]
    out = run_graph("what are its limitations?", build_graph(r, llm, S), history=history)
    assert r.queries[0] == "What are the limitations of Self-RAG?"  # retrieval saw the rewrite
    assert out["question"] == "What are the limitations of Self-RAG?"
    assert out["original_question"] == "what are its limitations?"
    assert llm.calls[0] == "condense" and out["decision"] == "answered"
    assert out["trace"][0]["node"] == "condense_question"


def test_first_question_makes_no_condense_call():
    llm = FakeLLM(["Answer [1]."], [True])
    out = run_graph("When does Self-RAG retrieve?", build_graph(StubRetriever([3.0]), llm, S))
    assert llm.calls == ["scope", "generate", "judge"] and "original_question" not in out


def test_condense_failure_falls_back_to_the_raw_question():
    class Failing(FakeLLM):
        def complete(self, role, messages):
            if role == "judge" and "Follow-up question" in messages[-1].content:
                raise RuntimeError("groq down")
            return super().complete(role, messages)

    r = StubRetriever([3.0])
    out = run_graph(
        "what about its limits?",
        build_graph(r, Failing(["Ok [1]."], [True]), S),
        history=[{"role": "user", "content": "Tell me about Self-RAG"}],
    )
    assert r.queries[0] == "what about its limits?" and out["decision"] == "answered"


# ---- questions about the collection itself ("what topics do these papers cover?") ----

PAPERS = [
    {
        "arxiv_id": "2610.1",
        "title": "Agentic AutoRAG",
        "published": "2026-10-06T00:00:00Z",
        "categories": ["cs.CL", "cs.IR"],
    },
    {
        "arxiv_id": "2310.11511",
        "title": "Self-RAG",
        "published": "2023-10-17T00:00:00Z",
        "categories": ["cs.CL"],
    },
    {
        "arxiv_id": "2005.11401",
        "title": "Retrieval-Augmented Generation for NLP",
        "published": "2020-05-22T00:00:00Z",
        "categories": ["cs.CL", "cs.LG"],
    },
]


class ScopeLLM(FakeLLM):
    """Tells the three judge-model jobs apart by their system prompt."""

    def __init__(self, scope_reply="YES", overview_reply="- Retrieval: “Self-RAG”", **kw):
        super().__init__(["unused"], [True], **kw)
        self.scope_reply, self.overview_reply = scope_reply, overview_reply

    def complete(self, role, messages):
        head = messages[0].content
        if head.startswith("You route questions"):
            self.calls.append("scope")
            if isinstance(self.scope_reply, Exception):
                raise self.scope_reply
            return self.scope_reply
        if head.startswith("You describe what a paper collection"):
            self.calls.append("overview")
            if isinstance(self.overview_reply, Exception):
                raise self.overview_reply
            return self.overview_reply
        return super().complete(role, messages)


def run_meta(llm, catalog=lambda: PAPERS, score=-7.2):
    r = StubRetriever([score])
    g = build_graph(r, llm, S, catalog=catalog)
    return run_graph("what topics do these papers contain?", g), r


def test_question_about_the_collection_is_answered_from_the_catalog_not_declined():
    llm = ScopeLLM()
    out, r = run_meta(llm)
    assert out["decision"] == "answered" and out["rewrites"] == 0
    assert (
        "**3 papers**" in out["answer"] and "2020–2026" in out["answer"]
    )  # computed, not generated
    assert "cs.CL" in out["answer"] and "Self-RAG" in out["answer"]
    assert out["citations"] == [] and out["top_score"] is None
    assert llm.calls == ["scope", "overview"]  # no retrieval, rewrites, generate or judge
    assert [t["node"] for t in out["trace"]] == ["classify_scope", "corpus_overview"]
    assert r.queries == []  # routed before retrieval


def test_weak_retrieval_that_is_not_about_the_collection_still_rewrites_then_declines():
    llm = ScopeLLM(scope_reply="NO")
    out, _ = run_meta(llm)
    assert out["decision"] == "declined" and out["rewrites"] == S.max_rewrites
    assert llm.calls.count("scope") == 1  # classified once, not again after each rewrite


def test_scope_check_failure_falls_back_to_the_normal_flow():
    out, _ = run_meta(ScopeLLM(scope_reply=RuntimeError("groq down")))
    assert out["decision"] == "declined" and out["rewrites"] == S.max_rewrites


def test_yes_with_an_empty_catalog_is_treated_as_a_normal_question():
    out, _ = run_meta(ScopeLLM(), catalog=lambda: [])
    assert out["decision"] == "declined"


def test_overview_llm_failure_still_returns_a_useful_answer():
    out, _ = run_meta(ScopeLLM(overview_reply=RuntimeError("quota")))
    assert out["decision"] == "answered" and "“Agentic AutoRAG”" in out["answer"]


def test_routing_ignores_retrieval_scores():
    # The same collection question routes to the overview even when some chunks score well,
    # which previously sent it down the content path and summarised random papers.
    for score in (-7.2, 6.5):
        out, r = run_meta(ScopeLLM(), score=score)
        assert out["decision"] == "answered" and r.queries == []
        assert out["trace"][-1]["node"] == "corpus_overview"


def test_overview_never_shows_a_title_the_model_invented():
    invented = "- Retrieval: “A Paper That Does Not Exist In The Index”"
    out, _ = run_meta(ScopeLLM(overview_reply=invented))
    assert out["decision"] == "answered"
    assert "Does Not Exist" not in out["answer"]  # replaced by real titles from the catalog
    assert "“Agentic AutoRAG”" in out["answer"]
    assert out["trace"][-1]["verified"] is False


def test_overview_with_real_titles_is_kept_and_marked_verified():
    out, _ = run_meta(ScopeLLM(overview_reply="- RAG: “Self-RAG” and “Agentic AutoRAG”"))
    assert "- RAG:" in out["answer"] and out["trace"][-1]["verified"] is True
