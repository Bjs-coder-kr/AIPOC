"""AI candidate extraction helpers."""

from __future__ import annotations

from typing import Any

from documind.ai.redact import sanitize_snippet


AI_CATEGORIES = {"spelling", "grammar", "readability", "logic", "redundancy"}
AI_SUBTYPE_MAP = {
    "spelling": "AI_SPELLING",
    "grammar": "AI_GRAMMAR",
    "readability": "AI_READABILITY",
    "logic": "AI_LOGIC",
    "redundancy": "AI_REDUNDANCY",
}


class CandidateLimiter:
    def __init__(
        self,
        total_limit: int,
        per_page_limit: int,
        per_category_limit: int | None = 1,
    ) -> None:
        self.total_limit = total_limit
        self.per_page_limit = per_page_limit
        self.per_category_limit = per_category_limit
        self.total_count = 0
        self.page_counts: dict[int, int] = {}
        self.category_counts: dict[str, int] = {}
        self.seen: set[tuple[int, str]] = set()

    def allow(self, candidate: dict[str, Any]) -> bool:
        page = int(candidate.get("page", 0))
        snippet = str(candidate.get("evidence_snippet") or "")
        if not snippet:
            return False
        if (page, snippet) in self.seen:
            return False
        if self.total_count >= self.total_limit:
            return False
        if self.page_counts.get(page, 0) >= self.per_page_limit:
            return False
        category = str(candidate.get("category") or "")
        if self.per_category_limit is not None:
            if self.category_counts.get(category, 0) >= self.per_category_limit:
                return False

        self.seen.add((page, snippet))
        self.total_count += 1
        self.page_counts[page] = self.page_counts.get(page, 0) + 1
        self.category_counts[category] = self.category_counts.get(category, 0) + 1
        return True


def extract_ai_candidate(
    text: str,
    redacted_text: str,
    result: dict[str, Any],
    page_number: int,
) -> dict[str, Any] | None:
    category = result.get("category")
    if category not in AI_CATEGORIES:
        return None
    raw_snippet = str(result.get("evidence_snippet") or "")
    if not raw_snippet.strip():
        return None
    snippet = raw_snippet[:200]
    start = redacted_text.find(snippet)
    if start == -1:
        return None
    end = start + len(snippet)
    if end > len(text):
        return None
    evidence_raw = text[start:end]
    message = sanitize_snippet(str(result.get("message") or ""), limit=200)
    evidence_snippet = sanitize_snippet(evidence_raw, limit=200)
    return {
        "id": f"ai_candidate_p{page_number}_{start}",
        "page": page_number,
        "category": category,
        "subtype": AI_SUBTYPE_MAP.get(category, "AI_MISC"),
        "message": message,
        "evidence": evidence_snippet,
        "evidence_snippet": evidence_snippet,
        "location": {"start": start, "end": end},
        "kind": "NOTE",
        "severity": "GREEN",
        "detector": "ai",
    }
