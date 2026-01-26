"""AI redaction tests."""

import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from documind.ai.redact import redact_text


def test_redact_length_preserved() -> None:
    text = "?°ë½ì²?010-1234-5678 ?´ë©”??test@example.com ì£¼ë?ë²ˆí˜¸ 900101-1234567"
    redacted = redact_text(text)

    assert len(redacted) == len(text)
    assert re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", redacted) is None
    assert re.search(r"\d{6}-?\d{7}", redacted) is None
    assert re.search(r"0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}", redacted) is None

    phone_match = re.search(r"0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}", text)
    assert phone_match is not None
    start, end = phone_match.span()
    assert set(redacted[start:end]) == {"*"}

