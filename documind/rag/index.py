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
    user_id: str | None = None,
) -> dict | None:
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return None
    # Add user_id to chunks
    if user_id:
        for chunk in chunks:
            chunk["user_id"] = user_id
            
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
    index: dict, 
    query_embedding: list[float], 
    top_k: int = 4,
    user_filter: str | None = None,
    is_admin: bool = False
) -> list[dict]:
    chunks = index.get("chunks") or []
    embeddings = index.get("embeddings") or []
    if not chunks or not embeddings or not query_embedding:
        return []
    
    # Filter valid indices based on user ownership
    valid_indices = []
    for i, chunk in enumerate(chunks):
        chunk_owner = chunk.get("user_id")
        # Admin sees everything, Owner sees theirs, Public (no owner) seen by anyone
        if is_admin or not chunk_owner or chunk_owner == user_filter:
            valid_indices.append(i)
            
    if not valid_indices:
        return []

    scores: list[float] = []
    page_ids: list[int] = []
    
    # Calculate scores only for valid chunks
    valid_embeddings = [embeddings[i] for i in valid_indices]
    
    for emb in valid_embeddings:
        scores.append(_cosine_similarity(query_embedding, emb))
    
    for i in valid_indices:
         chunk = chunks[i]
         page_ids.append(int(chunk.get("page", chunk.get("page_number", 0))))

    # Sort based on score
    # valid_indices map to 0..len(valid_indices)-1 in scores list
    # We need to keep track of original indices if we want to retrieve chunks later?
    # Actually we can just work with the filtered subsets.
    
    scored_subset_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    
    if not scored_subset_indices:
        return []
    
    pool_size = min(len(scored_subset_indices), max(top_k * 4, 12))
    candidates = scored_subset_indices[:pool_size]
    
    # Run MMR on the limited candidate set
    candidate_embeddings = [valid_embeddings[i] for i in candidates]
    # We need to pass full list of embeddings to MMR? No, just candidates.
    # But MMR needs random access to embeddings[s_idx].
    # Let's simplify: pass filtered embeddings and re-map indices.
    
    selected_subset_indices = _mmr_select(
        valid_embeddings, query_embedding, candidates, scores, top_k, page_ids=page_ids
    )
    
    results: list[dict] = []
    for idx in selected_subset_indices:
        # idx is index in valid_indices list
        original_idx = valid_indices[idx]
        chunk = dict(chunks[original_idx])
        chunk["score"] = round(scores[idx], 4)
        results.append(chunk)
    return results
