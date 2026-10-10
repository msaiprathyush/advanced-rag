"""Tests for the pure parsing/security helpers in the chat UI's app.js, run in QuickJS (no Node needed)."""

import json
from pathlib import Path

import pytest
import quickjs

APP_JS = Path("src/advanced_rag/api/static/app.js")


@pytest.fixture(scope="module")
def js():
    ctx = quickjs.Context()
    ctx.eval(
        APP_JS.read_text()
    )  # DOM section is guarded by `typeof document`, so only helpers load

    def call(expr: str):
        return json.loads(ctx.eval(f"JSON.stringify({expr})"))

    return call


def lit(s: str) -> str:
    return json.dumps(s)


def test_valid_citations_become_pills_and_invalid_ones_are_dropped(js):
    nodes = js(f"parseInline({lit('Claim [1] and [99].')}, new Set([1, 2]))")
    assert [n["t"] for n in nodes] == ["text", "cite", "text", "text"]
    assert nodes[1]["n"] == 1
    assert "99" not in json.dumps(nodes)


def test_bold_and_code(js):
    nodes = js(f"parseInline({lit('use **bold** and `code`')}, new Set())")
    assert [(n["t"], n.get("v")) for n in nodes if n["t"] != "text"] == [
        ("b", "bold"),
        ("code", "code"),
    ]


def test_hostile_markup_stays_inert_text(js):
    hostile = "<img src=x onerror=alert(1)><script>alert(2)</script> **<b>x</b>** [1]"
    nodes = js(f"parseInline({lit(hostile)}, new Set([1]))")
    # Only these node kinds exist, and the DOM layer renders each via textContent.
    assert {n["t"] for n in nodes} <= {"text", "b", "code", "cite"}
    joined = "".join(n.get("v", "") for n in nodes)
    assert "<img" in joined and "<script>" in joined  # preserved as literal text, never interpreted


def test_mark_tags_from_pdf_extraction_are_removed_but_other_markup_stays_inert(js):
    nodes = js(f"parseInline({lit('token <mark>Retrieve</mark> and <b>x</b>')}, new Set())")
    text = "".join(n.get("v", "") for n in nodes)
    assert text == "token Retrieve and <b>x</b>"


def test_narrow_no_break_space_is_normalised(js):
    nodes = js(f"parseInline({lit('a b')}, new Set())")
    assert nodes[0]["v"] == "a b"


def test_blocks_lists_headings_and_tables(js):
    text = "### Title\nIntro line [1].\n\n- one\n- two **b**\n\n1. first\n2. second\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    blocks = js(f"parseBlocks({lit(text)}, new Set([1]))")
    assert [b["type"] for b in blocks] == ["p", "ul", "ol", "pre"]
    assert blocks[0]["inline"][0]["v"].startswith(
        "Title Intro line"
    )  # '###' stripped, lines joined
    assert len(blocks[1]["items"]) == 2 and len(blocks[2]["items"]) == 2
    assert blocks[3]["text"].count("\n") == 2


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://arxiv.org/abs/2310.11511", True),
        ("javascript:alert(1)", False),
        ("http://arxiv.org/abs/2310.11511", False),
        ("https://evil.com/arxiv.org", False),
        ("https://arxiv.org.evil.com/x", False),
        ("not a url", False),
        ('https://arxiv.org/abs/2310.11511"onmouseover="x', False),
        ("https://arxiv.org/pdf/hep-th/9901001v2", True),
    ],
)
def test_only_https_arxiv_links_are_allowed(js, url, ok):
    assert (js(f"safeArxivUrl({lit(url)})") is not None) is ok


def test_section_label_drops_the_paper_title_prefix(js):
    assert (
        js(f"sectionLabel({lit('Self-RAG: Learning > 3 Method > 3.1 Training')})")
        == "3 Method › 3.1 Training"
    )
    assert js(f"sectionLabel({lit('Abstract')})") == "Abstract"


def test_section_label_strips_markdown_emphasis_markers(js):
    assert js(f"sectionLabel({lit('_C. Adaptive Retrieval_')})") == "C. Adaptive Retrieval"
    assert js(f"sectionLabel({lit('Survey > **IV. Method**')})") == "IV. Method"


def test_steps_describe_the_pipeline_in_plain_language(js):
    trace = [
        {"node": "condense_question", "ms": 700, "standalone": "What are Self-RAG's limitations?"},
        {"node": "retrieve", "ms": 2100, "query": "q", "top_score": 5.24},
        {"node": "generate", "ms": 900},
        {"node": "check_groundedness", "ms": 1100, "verdict": True},
        {"node": "finalize", "ms": 0},
    ]
    steps = js(f"buildSteps({json.dumps(trace)})")
    assert [s["text"][:12] for s in steps] == [
        "Rewrote your",
        "Searched the",
        "Drafted an a",
        "Self-check p",
    ]
    assert "5.2" in steps[1]["text"] and steps[3]["cls"] == "good"
    assert steps[0]["quote"] == "What are Self-RAG's limitations?"


def test_collection_questions_are_explained_and_a_no_verdict_stays_silent(js):
    trace = [
        {"node": "classify_scope", "ms": 200, "scope": True},
        {"node": "corpus_overview", "ms": 900, "papers": 55},
    ]
    steps = js(f"buildSteps({json.dumps(trace)})")
    assert "collection itself" in steps[0]["text"]
    assert "55 papers" in steps[1]["text"] and steps[1]["cls"] == "good"
    assert (
        js(f"buildSteps({json.dumps([{'node': 'classify_scope', 'ms': 1, 'scope': False}])})") == []
    )


def test_failed_self_check_and_decline_are_flagged(js):
    trace = [
        {"node": "check_groundedness", "ms": 1, "verdict": False, "unsupported": 2},
        {"node": "check_groundedness", "ms": 1, "verdict": "no_citations"},
        {"node": "decline", "ms": 0},
    ]
    steps = js(f"buildSteps({json.dumps(trace)})")
    assert all(s.get("cls") == "bad" for s in steps) and "2 unsupported" in steps[0]["text"]


def test_ui_source_never_uses_unsafe_html_sinks():
    src = APP_JS.read_text() + Path("src/advanced_rag/api/static/index.html").read_text()
    for sink in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
    ):
        assert sink not in src, sink
    # strict CSP relies on there being no inline script/style/handlers
    html = Path("src/advanced_rag/api/static/index.html").read_text()
    assert "<style" not in html and " onclick=" not in html and "<script>" not in html
