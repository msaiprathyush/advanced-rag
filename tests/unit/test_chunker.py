from advanced_rag.ingestion.chunker import CHUNK_SIZE, MIN_CHARS, chunk_paper
from advanced_rag.ingestion.models import PaperMeta
from advanced_rag.ingestion.parser import clean_markdown

META = PaperMeta(
    arxiv_id="2310.11511",
    version=1,
    title="Self-RAG",
    abstract="We introduce Self-RAG, which learns to retrieve, generate, and critique.",
    authors=["A"],
    categories=["cs.CL"],
    published="2023-10-17T00:00:00Z",
    updated="2023-10-17T00:00:00Z",
    abs_url="https://arxiv.org/abs/2310.11511",
    pdf_url="https://arxiv.org/pdf/2310.11511",
)

BODY = "Retrieval augmented generation improves factuality. " * 40


def test_references_are_stripped():
    md = f"# Intro\n\n{BODY}\n\n# Method\n\n{BODY}\n\n## References\n\n[1] Foo et al. 2020.\n"
    assert "Foo et al" not in clean_markdown(md)


def test_early_references_match_is_ignored():
    md = "# References\n\n" + BODY * 5
    assert "Retrieval augmented" in clean_markdown(md)


def test_abstract_chunk_first_and_header_on_embed_text():
    chunks = chunk_paper(META, f"# Intro\n\n{BODY}")
    assert chunks[0].section == "Abstract"
    assert chunks[0].text == META.abstract
    assert chunks[1].embed_text.startswith("Self-RAG\nSection: Intro")
    assert "Self-RAG\nSection" not in chunks[1].text  # header only in embedded text
    assert "embed_text" not in chunks[1].payload()


def test_size_bounds_and_sequential_indices():
    chunks = chunk_paper(META, f"# A\n\n{BODY}\n\n# B\n\ntiny\n\n{BODY}")
    body = chunks[1:]
    assert all(MIN_CHARS <= len(c.text) <= CHUNK_SIZE for c in body)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_abstract_only_fallback():
    assert len(chunk_paper(META, None)) == 1


def test_oversized_pdf_falls_back_to_abstract_only(monkeypatch):
    from advanced_rag.config import get_settings
    from advanced_rag.ingestion.parser import parse_pdf

    monkeypatch.setenv("PDF_MAX_MB", "1")
    get_settings.cache_clear()
    try:
        assert parse_pdf(b"%PDF" + b"0" * (2 * 1024 * 1024)) is None
    finally:
        monkeypatch.delenv("PDF_MAX_MB")
        get_settings.cache_clear()


def test_mark_highlight_tags_are_stripped_from_parsed_text():
    assert "<mark>" not in clean_markdown("# A\n\nthe <mark>Retrieve</mark> token " + "x" * 50)
