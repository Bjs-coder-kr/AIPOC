"""RAG citation filtering tests."""

from __future__ import annotations

from documind.rag.qa import filter_citations


def test_filter_citations_substring() -> None:
    pages = [
        {"page_number": 1, "text": "Hello world. This is a test."},
        {"page_number": 2, "text": "Another page with content."},
    ]
    citations = [
        {"page": 1, "snippet": "world"},
        {"page": 1, "snippet": "missing"},
        {"page": 2, "snippet": "content"},
    ]
    filtered = filter_citations(citations, pages)
    assert len(filtered) == 2
    assert filtered[0]["page"] == 1
    assert "world" in filtered[0]["snippet"]
