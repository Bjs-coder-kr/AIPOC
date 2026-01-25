"""AI candidate matching tests."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from documind.ai.candidates import extract_ai_candidate
from documind.ai.redact import redact_text


def test_ai_candidate_location_from_redacted_text() -> None:
    text = "ê²½ë ¥ ?”ì•½: ?°ë½ì²?010-1234-5678, ?´ë©”??test@example.com ?…ë‹ˆ??"
    redacted = redact_text(text)
    snippet = redacted[8:26]
    result = {
        "category": "logic",
        "subtype": "AI_CHECK",
        "message": "?°ë½ì²??•ì‹???•ì¸?˜ì„¸??",
        "evidence_snippet": snippet,
    }

    candidate = extract_ai_candidate(text, redacted, result, page_number=1)
    assert candidate is not None
    assert candidate["location"]["start"] == redacted.find(snippet)
    assert candidate["location"]["end"] == candidate["location"]["start"] + len(snippet)
    assert "*" in candidate["evidence"]
    assert "@" not in candidate["evidence"]
    assert "evidence_snippet" in candidate
    assert candidate["detector"] == "ai"
    assert candidate["subtype"] == "AI_LOGIC"

