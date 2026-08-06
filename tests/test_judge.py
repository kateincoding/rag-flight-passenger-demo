"""Tests for the judge's deterministic citation check — the safety-critical part."""
from agents.judge import check_citations

CHUNKS = [
    {"chunk_id": "BAG-001", "text": "carry-on rules"},
    {"chunk_id": "SVC-006", "text": "service dog docs"},
]


def test_valid_citations_pass():
    answer = "You need a health certificate [SVC-006]."
    assert check_citations(answer, CHUNKS) is True


def test_invented_citation_fails():
    answer = "You need a passport [SVC-999]."   # not retrieved
    assert check_citations(answer, CHUNKS) is False


def test_no_citation_passes():
    # Nothing cited is a valid subset (the empty set) — grounding is judged elsewhere.
    answer = "Sorry, the sources do not cover that."
    assert check_citations(answer, CHUNKS) is True


def test_mixed_valid_and_invented_fails():
    answer = "See [BAG-001] and [BAG-042]."
    assert check_citations(answer, CHUNKS) is False


def test_multiple_valid_citations_pass():
    answer = "See [BAG-001] and [SVC-006]."
    assert check_citations(answer, CHUNKS) is True
