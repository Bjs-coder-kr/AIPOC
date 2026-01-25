# ai/embeddings.py
"""
Embedding Provider Abstraction Layer.
Supports OpenAI, Gemini, and Ollama with SQLite caching.
"""

from abc import ABC, abstractmethod
import json
import logging
import os
import hashlib
import urllib.request
import urllib.error
import time

# Optional: google-generativeai
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

from ..utils.db import db_manager

logger = logging.getLogger(__name__)

class Embedder(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_texts(self, texts: list[str], model: str = None) -> list[list[float]]:
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        pass


class OpenAIEmbedder(Embedder):
    """OpenAI Embedding Provider."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        
    @property
    def default_model(self) -> str:
        return "text-embedding-3-small"

    def embed_texts(self, texts: list[str], model: str = None) -> list[list[float]]:
        if not self.api_key:
            logger.error("OpenAI API Key provided.")
            return []
            
        model = model or self.default_model
        payload = {"model": model, "input": texts}
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
                result = json.loads(response.read().decode("utf-8"))
                return [item["embedding"] for item in result["data"]]
        except Exception as e:
            logger.error(f"OpenAI Embedding Error: {e}")
            return []


class GeminiEmbedder(Embedder):
    """Google Gemini Embedding Provider."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if HAS_GEMINI and self.api_key:
            genai.configure(api_key=self.api_key)

    @property
    def default_model(self) -> str:
        return "models/text-embedding-004"

    def embed_texts(self, texts: list[str], model: str = None) -> list[list[float]]:
        if not self.api_key:
            logger.error("Gemini API Key not provided.")
            return []
            
        model = model or self.default_model
        
        # 1. Try using google-generativeai package if available
        if HAS_GEMINI:
            try:
                result = genai.embed_content(
                    model=model,
                    content=texts,
                    task_type="retrieval_document"
                )
                return result["embedding"]
            except Exception as e:
                logger.warning(f"Gemini SDK Error: {e}. Falling back to REST API.")
        
        # 2. Fallback to REST API
        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:embedContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        embeddings = []
        # Gemini REST API takes one string or array? 
        # API docs say: embedContent for single, batchEmbedContents for batch.
        # Simple client implementation: iterate if batch endpoint not easily handy or just use single.
        # To match batch behavior of others, let's look at batchEmbedContents.
        # url = f"https://generativelanguage.googleapis.com/v1beta/{model}:batchEmbedContents?key={self.api_key}"
        # payload = {"requests": [{"model": model, "content": {"parts": [{"text": t}]}} for t in texts]}
        
        # For simplicity/robustness in REST fallback, loop single calls (slower but safer)
        # or just fail if SDK missing? Let's implement single loop for now.
        for text in texts:
            try:
                payload = {
                     "model": model,
                     "content": {"parts": [{"text": text}]}
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    embeddings.append(res["embedding"]["values"])
            except Exception as e:
                 logger.error(f"Gemini REST Error for text chunk: {e}")
                 embeddings.append([]) # Append empty to maintain index
                 
        return embeddings


class OllamaEmbedder(Embedder):
    """Ollama Embedding Provider (Local)."""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host

    @property
    def default_model(self) -> str:
        return "nomic-embed-text"

    def embed_texts(self, texts: list[str], model: str = None) -> list[list[float]]:
        model = model or self.default_model
        url = f"{self.host}/api/embeddings"
        embeddings = []
        
        for text in texts:
            try:
                payload = {"model": model, "prompt": text}
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url, 
                    data=data, 
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    embeddings.append(res["embedding"])
            except Exception as e:
                logger.error(f"Ollama Error for text chunk ({model}): {e}")
                embeddings.append([])
                
        return embeddings


class CachedEmbedder(Embedder):
    """Decorator to add SQLite caching to any Embedder."""
    
    def __init__(self, provider: Embedder):
        self.provider = provider
        
    @property
    def default_model(self) -> str:
        return self.provider.default_model
        
    def embed_texts(self, texts: list[str], model: str = None) -> list[list[float]]:
        model = model or self.default_model
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        # 1. Check Cache
        for idx, text in enumerate(texts):
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cached = db_manager.get_cached_embedding(text_hash, model)
            if cached:
                results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)
        
        if not uncached_texts:
            return results
        
        logger.info(f"🧠 Cache Hit: {len(texts) - len(uncached_texts)}/{len(texts)}. Generating {len(uncached_texts)} new embeddings via {model}...")
        
        # 2. Generate New
        new_embeddings = self.provider.embed_texts(uncached_texts, model)
        
        if len(new_embeddings) != len(uncached_texts):
             logger.error("Embedding count mismatch from provider.")
             return []
             
        # 3. Save to Cache and Merge
        for i, vector in enumerate(new_embeddings):
             if vector: # Only save successful embeddings
                text = uncached_texts[i]
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                db_manager.save_embedding(text_hash, text, vector, model)
                
                original_idx = uncached_indices[i]
                results[original_idx] = vector
        
        # Handle failures (return empty list for failed slots? or filter? client expects parallel list)
        # Fill None with empty list if any failed
        return [r if r is not None else [] for r in results]


class EmbeddingFactory:
    """Factory to create Embedder instances."""
    
    @staticmethod
    def create(provider: str) -> Embedder:
        base_embedder = None
        
        if provider.startswith("OpenAI"):
            base_embedder = OpenAIEmbedder()
        elif provider.startswith("Gemini"):
            base_embedder = GeminiEmbedder()
        elif provider.startswith("Ollama"):
            base_embedder = OllamaEmbedder()
        else:
            # Default fallback
            base_embedder = OpenAIEmbedder()
            
        return CachedEmbedder(base_embedder)
