"""LangGraph: [condense follow-up] -> retrieve -> grade -> [rewrite loop] -> generate -> groundedness."""

from langgraph.graph import END, START, StateGraph

from advanced_rag.clients.llm import LLMClient, get_llm
from advanced_rag.config import Settings, get_settings
from advanced_rag.graph.nodes import make_nodes
from advanced_rag.graph.state import RAGState
from advanced_rag.retrieval.retriever import Retriever


def build_graph(
    retriever: Retriever | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
):
    s = settings or get_settings()
    retriever = retriever or Retriever("hybrid_rerank")
    llm = llm or get_llm()
    n = make_nodes(retriever, llm, s)

    def route_after_retrieve(state: RAGState) -> str:
        top = state.get("top_score")
        if top is not None and top >= s.rerank_weak_threshold:
            return "generate"
        if state.get("rewrites", 0) < s.max_rewrites:
            return "rewrite_query"
        return "decline"

    def route_after_grounding(state: RAGState) -> str:
        if state.get("grounded"):
            return "finalize"
        if state.get("regenerations", 0) < s.max_regenerations:
            return "regenerate"
        return "decline"

    g = StateGraph(RAGState)
    for name, fn in n.items():
        g.add_node(name, fn)
    # Follow-ups are first rewritten into a standalone question; a first question skips that call.
    g.add_conditional_edges(
        START,
        lambda state: "condense_question" if state.get("history") else "retrieve",
        {"condense_question": "condense_question", "retrieve": "retrieve"},
    )
    g.add_edge("condense_question", "retrieve")
    g.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "rewrite_query": "rewrite_query", "decline": "decline"},
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", "check_groundedness")
    g.add_conditional_edges(
        "check_groundedness",
        route_after_grounding,
        {"finalize": "finalize", "regenerate": "regenerate", "decline": "decline"},
    )
    g.add_edge("regenerate", "generate")
    g.add_edge("finalize", END)
    g.add_edge("decline", END)
    return g.compile()


def run_graph(question: str, graph=None, history: list[dict] | None = None) -> RAGState:
    graph = graph or build_graph()
    return graph.invoke(
        {
            "question": question,
            "history": history or [],
            "rewrites": 0,
            "regenerations": 0,
            "llm_calls": 0,
            "trace": [],
        }
    )
