"""RAG Q&A prompt and validation."""

from __future__ import annotations

import json
from typing import Any

from documind.ai.redact import redact_text, sanitize_snippet


def build_context(chunks: list[dict], max_chars: int = 2200) -> str:
    lines: list[str] = []
    total = 0
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text:
            continue
        snippet = redact_text(text)
        page = chunk.get("page", chunk.get("page_number", 0))
        chunk_id = chunk.get("chunk_id", "")
        entry = f"[p{page}|{chunk_id}] {snippet}"
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry)
    return "\n\n".join(lines)


def parse_rag_response(payload: str) -> dict[str, Any] | None:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if "answer" not in data or "citations" not in data:
        return None
    if not isinstance(data.get("citations"), list):
        return None
    return data


def filter_citations(
    citations: list[dict], pages: list[dict], chunks: list[dict] | None = None
) -> list[dict]:
    page_map = {page.get("page_number", 0): page.get("text", "") for page in pages}
    chunk_map: dict[str, dict] = {}
    if chunks:
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                chunk_map[str(chunk_id)] = chunk
    filtered: list[dict] = []
    for cite in citations:
        if not isinstance(cite, dict):
            continue
        page = cite.get("page")
        snippet = cite.get("snippet")
        chunk_id = cite.get("chunk_id")
        score = cite.get("score")
        if not snippet:
            continue
        if chunk_id and str(chunk_id) in chunk_map:
            chunk = chunk_map[str(chunk_id)]
            text = chunk.get("text", "")
            if not text:
                continue
            redacted = redact_text(text)
            start = redacted.find(snippet)
            if start == -1:
                continue
            end = start + len(snippet)
            evidence = sanitize_snippet(text[start:end])
            page_value = chunk.get("page", chunk.get("page_number", page))
            payload = {
                "page": page_value,
                "snippet": evidence,
                "chunk_id": str(chunk_id),
            }
            if isinstance(score, (int, float)):
                payload["score"] = float(score)
            filtered.append(payload)
            continue
        if not page:
            continue
        text = page_map.get(page, "")
        if not text:
            continue
        redacted = redact_text(text)
        start = redacted.find(snippet)
        if start == -1:
            start = text.find(snippet)
        if start == -1:
            continue
        end = start + len(snippet)
        evidence = sanitize_snippet(text[start:end])
        payload = {"page": page, "snippet": evidence}
        if isinstance(score, (int, float)):
            payload["score"] = float(score)
        filtered.append(payload)
    return filtered
