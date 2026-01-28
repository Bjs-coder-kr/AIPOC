"""Best practice archive/retrieval manager for target optimization."""

from __future__ import annotations

import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from ..ai.embeddings import EmbeddingFactory
from ..llm.config import get_default_embedding_provider
from ..utils.pydantic_compat import patch_pydantic_v1_for_chromadb


logger = logging.getLogger(__name__)

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

BEST_PRACTICE_COLLECTION_BASE = "optim_best_practices"
BEST_PRACTICE_COLLECTION_VERSION = "v2"
BEST_PRACTICE_COLLECTION = (
    f"{BEST_PRACTICE_COLLECTION_BASE}_{BEST_PRACTICE_COLLECTION_VERSION}"
)

PERSIST_DIR = Path(__file__).resolve().parents[2] / "chroma_best_practices"

_CLIENT_LOCK = threading.Lock()
_CLIENT = None
_COLLECTION_CACHE: dict[str, Any] = {}

_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER_CACHE: dict[str, Any] = {}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(
    r"(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?)?\d{3,4}[-.\s]?\d{4}"
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+")


@dataclass
class BestPracticeItem:
    id: str
    original_text: str
    rewritten_text: str
    score: int
    target_level: str
    keywords: str
    timestamp: str
    model_version: str


def _get_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        patch_pydantic_v1_for_chromadb()
        import chromadb

        _CLIENT = chromadb.PersistentClient(path=str(PERSIST_DIR))
        return _CLIENT


def _get_collection(name: str | None = None, provider: str | None = None):
    # If a specific name is given, use it (for explorer etc)
    # If not, build it from the provider
    if not name:
        base = BEST_PRACTICE_COLLECTION
        # Clean provider name to be safe for filenames/collection names
        # e.g. "Gemini CLI" -> "gemini_cli", "OpenAI API" -> "openai_api"
        # But usually we get "gemini", "openai", "ollama".
        suffix = (provider or "default").lower().replace(" ", "_").replace("-", "_")
        name = f"{base}_{suffix}"

    if name in _COLLECTION_CACHE:
        return _COLLECTION_CACHE[name]
    
    client = _get_client()
    # We must ensure we don't accidentally mix dimensions in the default collection if provider is missing
    # But for now, we assume provider is passed.
    collection = client.get_or_create_collection(name=name)
    _COLLECTION_CACHE[name] = collection
    return collection


def _get_embedder(provider: str):
    if provider in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[provider]
    with _EMBEDDER_LOCK:
        if provider in _EMBEDDER_CACHE:
            return _EMBEDDER_CACHE[provider]
        embedder = EmbeddingFactory.create(provider)
        _EMBEDDER_CACHE[provider] = embedder
        return embedder


@lru_cache(maxsize=512)
def _embed_text(provider: str, text: str) -> tuple[float, ...]:
    embedder = _get_embedder(provider)
    embeddings = embedder.embed_texts([text])
    if not embeddings or not embeddings[0]:
        return tuple()
    return tuple(embeddings[0])


def _mask_pii(text: str) -> str:
    if not text:
        return text
    masked = _EMAIL_RE.sub("***", text)
    masked = _PHONE_RE.sub("***", masked)
    return masked


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s <= 1e-9:
        return [0.0 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]


def archive_best_practice(
    result: dict,
    *,
    embedding_provider: str | None = None,
    min_score: int = 95,
    collection_name: str | None = None,
) -> bool:
    """Archive a high-quality optimization result into Chroma."""
    if not isinstance(result, dict):
        return False
    analysis = result.get("analysis") or {}
    score = int(analysis.get("score") or 0)
    if score < min_score:
        return False

    original_text = (result.get("original_text") or "").strip()
    rewritten_text = (result.get("rewritten_text") or "").strip()
    target_level = (result.get("target_level") or "").strip()
    model_version = (result.get("model_version") or "").strip()
    keywords = result.get("keywords") or []

    if not original_text or not rewritten_text or not target_level:
        logger.warning("Archive skipped due to missing required fields.")
        return False

    masked_original = _mask_pii(original_text)
    masked_rewritten = _mask_pii(rewritten_text)

    metadata = {
        "original_text": masked_original,
        "score": score,
        "target_level": target_level,
        "keywords": ",".join(map(str, keywords)) if isinstance(keywords, list) else str(keywords),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
    }

    embedding_provider = embedding_provider or get_default_embedding_provider()
    embedding = _embed_text(embedding_provider, masked_original)
    if not embedding:
        logger.warning("Archive skipped: embedding failed.")
        return False

    collection = _get_collection(collection_name, provider=embedding_provider)
    try:
        collection.add(
            documents=[masked_rewritten],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())],
            embeddings=[list(embedding)],
        )
        return True
    except Exception as exc:
        logger.error("Failed to archive best practice: %s", exc)
        return False


def retrieve_best_practices(
    original_text: str,
    target_level: str,
    *,
    n: int = 2,
    embedding_provider: str | None = None,
    min_score: int = 92,
    collection_name: str | None = None,
) -> list[BestPracticeItem]:
    """Retrieve best-practice examples using hybrid similarity + BM25."""
    if not original_text or not original_text.strip():
        return []

    masked_query = _mask_pii(original_text.strip())
    collection = _get_collection(collection_name, provider=embedding_provider)

    def _get_docs(where_filter: dict | None):
        try:
            return collection.get(
                where=where_filter,
                include=["documents", "metadatas", "ids"],
            )
        except Exception:
            return None

    where = {"target_level": target_level} if target_level else None
    docs = _get_docs(where)
    if not docs or not docs.get("ids"):
        docs = _get_docs(None)
        where = None
    if not docs or not docs.get("ids"):
        return []

    ids = list(docs.get("ids") or [])
    metadatas = list(docs.get("metadatas") or [])
    documents = list(docs.get("documents") or [])
    if not ids or not metadatas:
        return []

    corpus_texts = [
        str(meta.get("original_text", "")) if isinstance(meta, dict) else ""
        for meta in metadatas
    ]
    tokenized_corpus = [_tokenize(text) for text in corpus_texts]
    bm25_scores: list[float] = []
    if any(tokenized_corpus):
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = list(bm25.get_scores(_tokenize(masked_query)))
    if not bm25_scores or len(bm25_scores) != len(ids):
        bm25_scores = [0.0 for _ in ids]
    bm25_norm = _normalize_scores(bm25_scores)

    embedding_scores: dict[str, float] = {}
    embedding_norm: dict[str, float] = {}
    embedding_provider = embedding_provider or get_default_embedding_provider()
    query_embedding = _embed_text(embedding_provider, masked_query)
    if query_embedding:
        try:
            query = collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=min(len(ids), max(n * 4, 12)),
                where=where,
                include=["distances", "ids"],
            )
            q_ids = query.get("ids", [[]])[0]
            distances = query.get("distances", [[]])[0]
            if q_ids and distances and len(q_ids) == len(distances):
                for doc_id, distance in zip(q_ids, distances):
                    similarity = 1.0 / (1.0 + float(distance))
                    embedding_scores[str(doc_id)] = similarity
                norms = _normalize_scores(list(embedding_scores.values()))
                for doc_id, norm_score in zip(embedding_scores.keys(), norms):
                    embedding_norm[doc_id] = norm_score
        except Exception:
            pass

    scored: list[tuple[float, int]] = []
    for idx, doc_id in enumerate(ids):
        meta = metadatas[idx] or {}
        score = int(meta.get("score") or 0)
        if score < min_score:
            continue
        bm25_score = bm25_norm[idx] if idx < len(bm25_norm) else 0.0
        emb_score = embedding_norm.get(str(doc_id), 0.0)
        combined = (0.6 * emb_score) + (0.4 * bm25_score)
        scored.append((combined, idx))

    scored.sort(key=lambda item: item[0], reverse=True)
    results: list[BestPracticeItem] = []
    for _, idx in scored[:n]:
        meta = metadatas[idx] or {}
        results.append(
            BestPracticeItem(
                id=str(ids[idx]),
                original_text=str(meta.get("original_text", "")),
                rewritten_text=str(documents[idx]) if idx < len(documents) else "",
                score=int(meta.get("score") or 0),
                target_level=str(meta.get("target_level") or ""),
                keywords=str(meta.get("keywords") or ""),
                timestamp=str(meta.get("timestamp") or ""),
                model_version=str(meta.get("model_version") or ""),
            )
        )
    return results
