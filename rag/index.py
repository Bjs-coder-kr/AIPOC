"""Embedding index and search for RAG."""

from __future__ import annotations

import math

from documind.ai.redact import redact_text
from documind.rag.chunking import chunk_pages


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mmr_select(
    embeddings: list[list[float]],
    query_embedding: list[float],
    candidates: list[int],
    scores: list[float],
    top_k: int,
    page_ids: list[int] | None = None,
    lambda_mult: float = 0.7,
) -> list[int]:
    selected: list[int] = []
    remaining = candidates[:]
    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = -1.0
        for idx in remaining:
            relevance = scores[idx]
            diversity = 0.0
            if selected:
                diversity = max(
                    _cosine_similarity(embeddings[idx], embeddings[s_idx])
                    for s_idx in selected
                )
            page_penalty = 0.0
            if page_ids and selected:
                if any(page_ids[idx] == page_ids[s_idx] for s_idx in selected):
                    page_penalty = 0.05
            mmr_score = (
                lambda_mult * relevance - (1 - lambda_mult) * diversity - page_penalty
            )
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def build_index(
    client,
    pages: list[dict],
    chunk_size: int = 900,
    overlap: int = 120,
    batch_size: int = 32,
    embed_model: str | None = None,
) -> dict | None:
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return None
    texts = [redact_text(chunk["text"]) for chunk in chunks]
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_embeddings = client.embed_texts(batch, model=embed_model)
        if not batch_embeddings:
            return None
        embeddings.extend(batch_embeddings)
    if len(embeddings) != len(chunks):
        return None
    return {"chunks": chunks, "embeddings": embeddings}


def search_index(
    index: dict, query_embedding: list[float], top_k: int = 4
) -> list[dict]:
    chunks = index.get("chunks") or []
    embeddings = index.get("embeddings") or []
    if not chunks or not embeddings or not query_embedding:
        return []
    scores: list[float] = []
    page_ids: list[int] = []
    for emb in embeddings:
        scores.append(_cosine_similarity(query_embedding, emb))
    for chunk in chunks:
        page_ids.append(int(chunk.get("page", chunk.get("page_number", 0))))
    scored = sorted(range(len(chunks)), key=lambda idx: scores[idx], reverse=True)
    if not scored:
        return []
    pool_size = min(len(scored), max(top_k * 4, 12))
    candidates = scored[:pool_size]
    selected = _mmr_select(
        embeddings, query_embedding, candidates, scores, top_k, page_ids=page_ids
    )
    results: list[dict] = []
    for idx in selected:
        chunk = dict(chunks[idx])
        chunk["score"] = round(scores[idx], 4)
        results.append(chunk)
    return results
