"""RAG evaluation script (manual)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from documind.ai.client import OpenAIClient
from documind.rag.index import build_index, search_index
from documind.rag.qa import build_context, filter_citations
from documind.text.normalize import normalize_pages


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pages(path: Path) -> list[dict]:
    payload = _load_json(path)
    pages = payload.get("pages", [])
    normalized = normalize_pages(pages)
    return normalized.get("pages", [])


def _load_questions(path: Path) -> list[dict]:
    payload = _load_json(path)
    questions = payload if isinstance(payload, list) else payload.get("questions", [])
    return [q for q in questions if isinstance(q, dict)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", required=True, help="Pages fixture JSON path.")
    parser.add_argument("--questions", required=True, help="Questions JSON path.")
    args = parser.parse_args()

    pages_path = Path(args.pages)
    questions_path = Path(args.questions)
    pages = _load_pages(pages_path)
    questions = _load_questions(questions_path)

    client = OpenAIClient()
    if not client.is_available():
        print("OPENAI_API_KEY not set; skipping.")
        return 0

    rag_index = build_index(client, pages)
    if not rag_index:
        print("Failed to build RAG index.")
        return 1

    hits = 0
    expected_total = 0

    for idx, item in enumerate(questions, start=1):
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        language = item.get("language", "ko")
        expected = item.get("expected_pages") or item.get("expected_page")
        expected_pages = []
        if isinstance(expected, list):
            expected_pages = [int(x) for x in expected if isinstance(x, int)]
        elif isinstance(expected, int):
            expected_pages = [expected]

        query_embedding = client.embed_texts([question])
        if not query_embedding:
            print(f"[{idx}] embedding failed")
            continue
        chunks = search_index(rag_index, query_embedding[0], top_k=4)
        context = build_context(chunks)
        response = client.rag_qa(question=question, context=context, language=language)
        citations = []
        answer = {}
        if isinstance(response, dict):
            citations = response.get("citations") or []
            answer = response.get("answer") or {}
        filtered = filter_citations(citations, pages, chunks=chunks)
        pages_found = [c.get("page") for c in filtered if c.get("page")]

        hit = False
        if expected_pages:
            expected_total += 1
            hit = any(page in expected_pages for page in pages_found)
            if hit:
                hits += 1

        answer_text = (
            answer.get("ko") if language == "ko" else answer.get("en")
        ) or ""
        print(
            f"[{idx}] citations={len(filtered)} pages={pages_found} "
            f"answer_len={len(str(answer_text))} hit={'Y' if hit else 'N'}"
        )

    if expected_total:
        hit_rate = hits / expected_total
        print(f"hit_rate={hit_rate:.2f} ({hits}/{expected_total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Example:
# python scripts/rag_eval.py --pages tests/fixtures/pdf_sample.json --questions scripts/rag_questions.json
