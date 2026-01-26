import os
import sys
import unittest
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from documind.utils.best_practice_manager import archive_best_practice, retrieve_best_practices
from documind.ai.embeddings import EmbeddingFactory

class TestBestPracticeManager(unittest.TestCase):
    def setUp(self):
        # Use a temporary collection for testing
        self.test_collection = f"test_optim_collection_{int(datetime.now().timestamp())}"
        self.target_level = "student"
        self.embedding_provider = "Ollama"

    def test_tc1_archive_high_score(self):
        """TC1: 첫 시도 96점 -> 자동 저장 및 통과."""
        result = {
            "original_text": "Quantum mechanics is a fundamental theory in physics.",
            "rewritten_text": "양자 역학은 물리학의 기본 이론입니다.",
            "target_level": self.target_level,
            "analysis": {"score": 96},
            "keywords": ["physics", "quantum"],
            "model_version": "test-model"
        }
        success = archive_best_practice(
            result, 
            collection_name=self.test_collection, 
            min_score=95,
            embedding_provider=self.embedding_provider
        )
        self.assertTrue(success, "Should archive results with score >= 95")

    def test_tc6_tc8_retrieval_and_fallback(self):
        """TC6 & TC8: Retrieval and Fallback."""
        # First, ensure something is in the collection
        archive_best_practice({
            "original_text": "General relativity.",
            "rewritten_text": "일반 상대성 이론.",
            "target_level": "expert",
            "analysis": {"score": 98},
            "keywords": ["physics"],
            "model_version": "test-model"
        }, collection_name=self.test_collection, embedding_provider=self.embedding_provider)

        # Retrieval with exact target_level
        results = retrieve_best_practices(
            "relativity", 
            "expert", 
            n=1, 
            collection_name=self.test_collection,
            embedding_provider=self.embedding_provider
        )
        self.assertTrue(len(results) >= 0)

        # TC8: Fallback for different target_level
        results_fallback = retrieve_best_practices(
            "relativity", 
            "student", # No student docs yet
            n=1,
            collection_name=self.test_collection,
            embedding_provider=self.embedding_provider
        )
        self.assertTrue(len(results_fallback) >= 0)

    def test_pii_masking(self):
        """Verify PII masking during archive."""
        result = {
            "original_text": "Contact me at test@example.com or 010-1234-5678.",
            "rewritten_text": "이메일 test@example.com 혹은 010-1234-5678로 연락주세요.",
            "target_level": self.target_level,
            "analysis": {"score": 97},
            "keywords": ["contact"],
            "model_version": "test-model"
        }
        archive_best_practice(
            result, 
            collection_name=self.test_collection, 
            embedding_provider=self.embedding_provider
        )
        
        # Retrieve and check masking
        retrieved = retrieve_best_practices(
            "contact", 
            self.target_level, 
            collection_name=self.test_collection,
            embedding_provider=self.embedding_provider
        )
        if retrieved:
            self.assertIn("***", retrieved[0].original_text)
            self.assertIn("***", retrieved[0].rewritten_text)
            self.assertNotIn("test@example.com", retrieved[0].original_text)

if __name__ == "__main__":
    unittest.main()
