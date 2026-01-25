"""AI candidate limiter tests."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from documind.ai.candidates import CandidateLimiter


def test_limiter_dedup_and_limits() -> None:
    limiter = CandidateLimiter(total_limit=3, per_page_limit=2, per_category_limit=1)
    c1 = {
        "page": 1,
        "category": "spelling",
        "evidence_snippet": "a",
    }
    c2 = {
        "page": 1,
        "category": "spelling",
        "evidence_snippet": "b",
    }
    c3 = {
        "page": 1,
        "category": "grammar",
        "evidence_snippet": "a",
    }
    c4 = {
        "page": 2,
        "category": "grammar",
        "evidence_snippet": "c",
    }

    assert limiter.allow(c1) is True
    assert limiter.allow(c2) is False
    assert limiter.allow(c3) is False
    assert limiter.allow(c4) is True

