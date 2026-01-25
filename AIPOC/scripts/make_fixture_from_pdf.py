"""Generate a fixture JSON from PDF(s)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from documind.ingest.pdf_loader import load_pdf
from documind.text.normalize import normalize_pages


EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"\b01[0-9]-?\d{3,4}-?\d{4}\b")
RRN_PATTERN = re.compile(r"\b\d{6}-?\d{7}\b")


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "doc"


def _redact_text(text: str) -> str:
    redacted = EMAIL_PATTERN.sub("***", text)
    redacted = PHONE_PATTERN.sub("***", redacted)
    redacted = RRN_PATTERN.sub("***", redacted)
    return redacted


def _iter_pdfs(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() == ".pdf":
            yield path
        return
    for pdf_path in path.rglob("*.pdf"):
        yield pdf_path


def _build_fixture(pdf_bytes: bytes, file_name: str, max_pages: int | None, redact: bool) -> dict:
    data = load_pdf(pdf_bytes, file_name)
    normalized = normalize_pages(data["pages"])
    pages = normalized["pages"]
    if max_pages is not None:
        pages = pages[:max_pages]

    fixture_pages = []
    for page in pages:
        text = page["text"]
        if redact:
            text = _redact_text(text)
        fixture_pages.append({"page_number": page["page_number"], "text": text})

    return {"pages": fixture_pages}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate fixture JSON from PDF(s).")
    parser.add_argument("path", help="PDF file path or directory.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "tests" / "fixtures"),
        help="Output directory for fixture JSON files.",
    )
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--redact", dest="redact", action="store_true", default=True)
    parser.add_argument("--no-redact", dest="redact", action="store_false")

    args = parser.parse_args()
    target = Path(args.path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for pdf_path in _iter_pdfs(target):
        pdf_bytes = pdf_path.read_bytes()
        slug = _slugify(pdf_path.stem)
        suffix = hashlib.md5(pdf_bytes).hexdigest()[:8]
        if slug == "doc" or len(slug) < 3:
            slug = f"{slug}_{suffix}"
        out_path = output_dir / f"pdf_{slug}.json"
        fixture = _build_fixture(pdf_bytes, pdf_path.name, args.max_pages, args.redact)
        out_path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs.append(out_path)

    for path in outputs:
        print(path)

    return 0 if outputs else 1


if __name__ == "__main__":
    raise SystemExit(main())
