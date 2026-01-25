"""Text normalization helpers."""

from __future__ import annotations

import regex as re


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[\t ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_pages(pages: list[dict]) -> dict:
    normalized_pages: list[dict] = []
    offset = 0

    for page in pages:
        text = normalize_text(page.get("text", ""))
        start_char = offset
        end_char = start_char + len(text)
        normalized_pages.append(
            {
                "page_number": page.get("page_number", 0),
                "text": text,
                "start_char": start_char,
                "end_char": end_char,
            }
        )
        offset = end_char + 2

    return {"pages": normalized_pages, "normalized_char_count": max(0, offset - 2)}
