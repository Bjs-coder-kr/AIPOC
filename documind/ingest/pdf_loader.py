"""PDF loader using pypdf."""

from __future__ import annotations

from io import BytesIO
import logging

from pypdf import PdfReader


PARTIAL_SCAN_THRESHOLD = 0.2
logger = logging.getLogger(__name__)


def _scan_level(ratio: float, high_threshold: float) -> str:
    if ratio >= high_threshold:
        return "HIGH"
    if ratio >= PARTIAL_SCAN_THRESHOLD:
        return "PARTIAL"
    return "NONE"


def load_pdf(
    file_bytes: bytes,
    file_name: str,
    min_text_len: int = 50,
    scan_like_threshold: float = 0.6,
) -> dict:
    reader = PdfReader(BytesIO(file_bytes), strict=False)
    pages: list[dict] = []
    textless_pages = 0
    total_chars = 0

    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("page=%s extract_text_failed err=%s", idx, exc.__class__.__name__)
            text = ""
        pages.append({"page_number": idx, "text": text})
        if len(text.strip()) < min_text_len:
            textless_pages += 1
        total_chars += len(text)

    page_count = len(pages)
    scan_like_ratio = (textless_pages / page_count) if page_count else 1.0
    scan_like = scan_like_ratio >= scan_like_threshold
    scan_level = _scan_level(scan_like_ratio, scan_like_threshold)

    meta = {
        "file_name": file_name,
        "page_count": page_count,
        "textless_pages": textless_pages,
        "raw_char_count": total_chars,
        "scan_like": scan_like,
        "scan_like_ratio": scan_like_ratio,
        "scan_level": scan_level,
    }

    return {"pages": pages, "meta": meta}
