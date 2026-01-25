"""Chunking utilities for RAG."""

from __future__ import annotations

import regex as re

from documind.ai.redact import sanitize_snippet


def _is_text_noisy(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 40:
        return True
    compact = re.sub(r"\s+", "", stripped)
    if len(compact) < 30:
        return True
    letters = len(re.findall(r"\p{L}", stripped))
    numbers = len(re.findall(r"\p{N}", stripped))
    symbols = len(re.findall(r"[^\p{L}\p{N}\s]", stripped))
    non_letter_ratio = (numbers + symbols) / max(1, len(compact))
    return non_letter_ratio > 0.6


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[dict]:
    chunks: list[dict] = []
    for page in pages:
        text = page.get("text", "") or ""
        page_number = page.get("page_number", 0)
        if not text.strip():
            continue
        if _is_text_noisy(text) and len(text) < 400:
            continue
        start = 0
        length = len(text)
        chunk_idx = 0
        while start < length:
            end = min(start + chunk_size, length)
            chunk_text = text[start:end]
            if chunk_text.strip() and not _is_text_noisy(chunk_text):
                chunks.append(
                    {
                        "page": page_number,
                        "page_number": page_number,
                        "chunk_id": f"p{page_number}_c{chunk_idx}",
                        "start_char": start,
                        "end_char": end,
                        "text": chunk_text,
                        "snippet": sanitize_snippet(chunk_text, limit=200),
                    }
                )
                chunk_idx += 1
            if end >= length:
                break
            start = max(0, end - overlap)
    return chunks
