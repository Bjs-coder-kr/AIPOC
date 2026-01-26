"""Redaction helpers for AI payloads."""

from __future__ import annotations

import regex as re


EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(
    r"(?:\+82[-\s]?)?0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}"
)
RRN_PATTERN = re.compile(r"\b\d{6}-?\d{7}\b")


def _mask_match(match: re.Match) -> str:
    return "*" * len(match.group(0))


def redact_text(text: str) -> str:
    redacted = EMAIL_PATTERN.sub(_mask_match, text)
    redacted = PHONE_PATTERN.sub(_mask_match, redacted)
    redacted = RRN_PATTERN.sub(_mask_match, redacted)
    return redacted


def truncate_text(text: str, limit: int = 200) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def sanitize_snippet(text: str, limit: int = 200) -> str:
    return truncate_text(redact_text(text), limit=limit)
