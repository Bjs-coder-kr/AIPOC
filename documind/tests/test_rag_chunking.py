"""RAG chunking tests."""

from __future__ import annotations

from documind.rag.chunking import chunk_pages


def test_chunk_pages_basic() -> None:
    pages = [{"page_number": 1, "text": "abc" * 400}]
    chunks = chunk_pages(pages, chunk_size=300, overlap=50)
    assert chunks
    first = chunks[0]
    assert first["page"] == 1
    assert first["start_char"] == 0
    assert first["end_char"] > first["start_char"]
    assert first["text"] == pages[0]["text"][first["start_char"] : first["end_char"]]
