from advanced_rag.generation.citations import cited_indices, normalize_citations


def test_fullwidth_browsing_style_is_normalised():
    assert normalize_citations("It retrieves on demand【4†L1-L4】 and more【2†L7-L10】.") == (
        "It retrieves on demand[4] and more[2]."
    )


def test_comma_lists_are_expanded():
    assert normalize_citations("Claim [1, 2] and [3,4,5].") == "Claim [1][2] and [3][4][5]."


def test_canonical_form_untouched():
    assert normalize_citations("Fine [1][2].") == "Fine [1][2]."


def test_cited_indices_drops_out_of_range_and_dedupes():
    assert cited_indices("a [2] b [9] c [2] d [1] e [0]", 3) == [2, 1]


def test_normalised_output_yields_valid_citations():
    ans = normalize_citations("Retrieval is on demand【4†L1-L4】, thresholded【2†L7-L10】.")
    assert cited_indices(ans, 6) == [4, 2]
