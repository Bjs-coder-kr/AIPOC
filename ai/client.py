"""OpenAI client for AI explanations and extra checks."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

logger = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.last_error: str | None = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def summarize_issues(self, issues: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.is_available() or not issues:
            return {}
        payload = {
            "role": "user",
            "content": (
                "Return ONLY JSON.\n"
                "Schema: {\"items\":[{\"id\":\"...\",\"ko\":{\"why\":\"...\",\"impact\":\"...\",\"action\":\"...\"},"
                "\"en\":{\"why\":\"...\",\"impact\":\"...\",\"action\":\"...\"}}]}\n"
                "Write short, practical sentences. Do not add extra keys.\n"
                f"Issues: {json.dumps(issues, ensure_ascii=False)}"
            ),
        }
        data = self._chat([payload], temperature=0.2, max_tokens=800)
        if not data:
            return {}
        content = self._extract_content(data)
        if not content:
            self.last_error = "empty_response"
            return {}
        parsed = self._parse_json(content)
        if not isinstance(parsed, dict):
            return {}
        items = parsed.get("items", [])
        if not isinstance(items, list):
            return {}
        result: dict[str, Any] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            issue_id = item.get("id")
            if issue_id:
                result[str(issue_id)] = item
        return result

    def review_page(
        self, text: str, max_candidates: int = 5, language: str = "ko"
    ) -> list[dict[str, Any]]:
        if not self.is_available() or not text.strip():
            return []
        lang_hint = "Korean" if language == "ko" else "English"
        prompt = (
            "Return ONLY JSON.\n"
            "Schema: {\"candidates\":[{\"category\":\"spelling|grammar|readability|logic|redundancy\","
            "\"subtype\":\"...\",\"message\":\"...\",\"evidence_snippet\":\"...\"}]}\n"
            f"Return at most {max_candidates} candidates.\n"
            "evidence_snippet must be an exact substring from the given text (<=200 chars).\n"
            "Only propose NOTE-level issues. Avoid personal data.\n"
            f"Write message in {lang_hint}.\n"
            f"Text:\n{text}"
        )
        data = self._chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=700,
        )
        if not data:
            return []
        content = self._extract_content(data)
        if not content:
            self.last_error = "empty_response"
            return []
        parsed = self._parse_json(content)
        if not isinstance(parsed, dict):
            return []
        candidates = parsed.get("candidates", [])
        if not isinstance(candidates, list):
            return []
        return [item for item in candidates if isinstance(item, dict)]

    def embed_texts(
        self, texts: list[str], model: str | None = None
    ) -> list[list[float]]:
        if not self.is_available() or not texts:
            return []
        self.last_error = None
        embed_model = model or os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)
        payload = {"model": embed_model, "input": texts}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
        except urllib.error.HTTPError as exc:
            self.last_error = f"http_error_{exc.code}"
            logger.warning("OpenAI embedding HTTP error status=%s", exc.code)
            return []
        except urllib.error.URLError as exc:
            self.last_error = "url_error"
            logger.warning("OpenAI embedding URL error reason=%s", exc.reason)
            return []
        except json.JSONDecodeError:
            self.last_error = "json_parse_failed"
            logger.warning("OpenAI embedding JSON parse failed")
            return []
        except Exception as exc:
            self.last_error = f"request_failed_{exc.__class__.__name__}"
            logger.warning("OpenAI embedding request failed err=%s", exc.__class__.__name__)
            return []
        items = parsed.get("data") or []
        if not isinstance(items, list):
            self.last_error = "invalid_json"
            return []
        embeddings: list[list[float]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            vector = item.get("embedding")
            if isinstance(vector, list):
                embeddings.append([float(x) for x in vector])
        if len(embeddings) != len(texts):
            self.last_error = "invalid_json"
            return []
        return embeddings

    def rag_qa(
        self,
        question: str,
        context: str,
        language: str = "ko",
        caution: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.is_available() or not question.strip():
            return None
        lang_hint = "Korean" if language == "ko" else "English"
        caution_text = f"\nNote: {caution}" if caution else ""
        prompt = (
            "Return ONLY JSON.\n"
            "Schema: {\"answer\":{\"ko\":\"...\",\"en\":\"...\"},"
            "\"citations\":[{\"page\":1,\"snippet\":\"...\",\"chunk_id\":\"p1_c0\",\"score\":0.76}]}\n"
            "Write concise answers based only on the provided context.\n"
            "If context is insufficient, say so in the answer and leave citations empty.\n"
            "Citations must be exact substrings from the context and include chunk_id.\n"
            f"Write the main answer in {lang_hint} and provide both ko/en fields.\n"
            f"{caution_text}\n"
            f"Question: {question}\n\nContext:\n{context}"
        )
        data = self._chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=700,
        )
        if not data:
            return None
        content = self._extract_content(data)
        if not content:
            self.last_error = "empty_response"
            return None
        parsed = self._parse_json(content)
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _chat(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> dict[str, Any] | None:
        if not self.is_available():
            return None
        self.last_error = None
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            self.last_error = f"http_error_{exc.code}"
            logger.warning("OpenAI HTTP error status=%s", exc.code)
            return None
        except urllib.error.URLError as exc:
            self.last_error = "url_error"
            logger.warning("OpenAI URL error reason=%s", exc.reason)
            return None
        except json.JSONDecodeError:
            self.last_error = "json_parse_failed"
            logger.warning("OpenAI response JSON parse failed")
            return None
        except Exception as exc:
            self.last_error = f"request_failed_{exc.__class__.__name__}"
            logger.warning("OpenAI request failed err=%s", exc.__class__.__name__)
            return None

    def _extract_content(self, data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")

    def _parse_json(self, text: str) -> Any:
        try:
            parsed = json.loads(text)
            self.last_error = None
            return parsed
        except json.JSONDecodeError:
            self.last_error = "invalid_json"
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
            self.last_error = None
            return parsed
        except json.JSONDecodeError:
            self.last_error = "invalid_json"
            return None
