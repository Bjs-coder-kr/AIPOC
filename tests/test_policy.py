"""Issue policy tests."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from documind.quality.pipeline import UNCERTAIN_SUGGESTION, _apply_issue_policies
from documind.schema import Issue, IssueI18n, IssueText, Location


class TestIssuePolicies(unittest.TestCase):
    def _make_issue(self, kind: str, severity: str) -> Issue:
        i18n = IssueI18n(
            ko=IssueText(message="ë©”ì‹œì§€", suggestion="?ë³¸ ?œì•ˆ"),
            en=IssueText(message="Message", suggestion="Original suggestion"),
        )
        return Issue(
            id="issue_1",
            category="readability",
            kind=kind,
            subtype="LONG_SENTENCE",
            severity=severity,
            message=i18n.ko.message,
            evidence="Example evidence.",
            suggestion=i18n.ko.suggestion,
            location=Location(page=1, start_char=0, end_char=10),
            confidence=0.7,
            detector="rule_based",
            i18n=i18n,
        )

    def test_uncertain_overrides_suggestion(self) -> None:
        issue = self._make_issue(kind="WARNING", severity="RED")
        page_profiles = [
            {"page": 1, "type": "RESUME", "confidence": 0.34, "signals": []}
        ]
        _apply_issue_policies(
            [issue],
            page_profiles,
            language="ko",
            pages=[{"page_number": 1, "text": "?˜í”Œ ?ìŠ¤??}],
        )
        self.assertEqual(issue.kind, "NOTE")
        self.assertIsNone(issue.subtype)
        self.assertEqual(issue.page_type, "UNCERTAIN")
        self.assertEqual(issue.suggestion, UNCERTAIN_SUGGESTION["ko"])
        self.assertEqual(issue.i18n.ko.suggestion, UNCERTAIN_SUGGESTION["ko"])
        self.assertEqual(issue.i18n.en.suggestion, UNCERTAIN_SUGGESTION["en"])

    def test_note_forces_green_severity(self) -> None:
        issue = self._make_issue(kind="NOTE", severity="RED")
        _apply_issue_policies([issue], [], language="ko", pages=[])
        self.assertEqual(issue.severity, "GREEN")


if __name__ == "__main__":
    unittest.main()

