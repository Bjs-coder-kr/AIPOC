# tests/test_db_embeddings.py
"""Tests for DBManager and Embedding Provider."""

import json
import pytest
from unittest.mock import MagicMock, patch

from documind.utils.db import SQLiteManager
from documind.ai.embeddings import EmbeddingFactory, CachedEmbedder, Embedder

@pytest.fixture
def db_manager(tmp_path):
    """Fixture for independent DB manager with temp DB file."""
    # Patch the global DB_PATH or class logic if possible.
    # Since SQLiteManager is singleton, we need to be careful.
    # For testing, we can patch DB_PATH in utils.db module before init?
    # Or just construct new instance by resetting singleton?
    
    with patch("documind.utils.db.DB_PATH", tmp_path / "test.db"):
        # Reset singleton
        SQLiteManager._instance = None
        db = SQLiteManager()
        yield db
        # Cleanup
        SQLiteManager._instance = None

def test_db_settings(db_manager):
    """Test getting and saving settings."""
    db_manager.save_setting("test_key", "test_value")
    val = db_manager.get_setting("test_key")
    assert val == "test_value"
    
    val_missing = db_manager.get_setting("missing_key", "default")
    assert val_missing == "default"

def test_db_history(db_manager):
    """Test saving analysis history."""
    report = {"summary": "test", "score": 95}
    db_manager.save_history("test.pdf", "hash123", report)
    
    history = db_manager.get_recent_history(1)
    assert len(history) == 1
    assert history[0]["filename"] == "test.pdf"
    
    detail = db_manager.get_history_detail(history[0]["id"])
    assert detail["score"] == 95

def test_embedding_factory():
    """Test factory creates correct instances."""
    e1 = EmbeddingFactory.create("OpenAI")
    assert isinstance(e1, CachedEmbedder)
    assert e1.provider.__class__.__name__ == "OpenAIEmbedder"
    
    e2 = EmbeddingFactory.create("Gemini")
    assert e2.provider.__class__.__name__ == "GeminiEmbedder"

    e3 = EmbeddingFactory.create("Ollama")
    assert e3.provider.__class__.__name__ == "OllamaEmbedder"

def test_cached_embedder(db_manager):
    """Test caching logic."""
    mock_provider = MagicMock(spec=Embedder)
    mock_provider.default_model = "mock-model"
    mock_provider.embed_texts.return_value = [[0.1, 0.2]] # Mock embedding
    
    # Temporarily replace global db_manager with our test fixture?
    # CachedEmbedder imports db_manager from utils.db.
    # We need to patch documind.ai.embeddings.db_manager
    
    with patch("documind.ai.embeddings.db_manager", db_manager):
        embedder = CachedEmbedder(mock_provider)
        texts = ["hello world"]
        
        # First call: Cache miss -> Provider called
        res1 = embedder.embed_texts(texts)
        assert res1 == [[0.1, 0.2]]
        mock_provider.embed_texts.assert_called_once()
        
        # Check DB
        cached = db_manager.get_cached_embedding("fake_hash", "mock-model") # Need real hash
        import hashlib
        h = hashlib.sha256(texts[0].encode("utf-8")).hexdigest()
        cached = db_manager.get_cached_embedding(h, "mock-model")
        assert cached == [0.1, 0.2]
        
        # Second call: Cache hit -> Provider NOT called again
        mock_provider.embed_texts.reset_mock()
        res2 = embedder.embed_texts(texts)
        assert res2 == [[0.1, 0.2]]
        mock_provider.embed_texts.assert_not_called()
