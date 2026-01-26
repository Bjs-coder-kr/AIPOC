"""Redundancy detector tests."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from documind.quality.detectors import redundancy


class TestRedundancyDetector(unittest.TestCase):
    def _detect(self, text: str, page_type: str, confidence: float = 0.9) -> list:
        pages = [{"page_number": 1, "text": text}]
        profiles = [{"page": 1, "type": page_type, "confidence": confidence, "signals": []}]
        return redundancy.detect(pages, language="ko", page_profiles=profiles)

    def test_consent_boilerplate_is_note(self) -> None:
        text = (
            "Privacy notice and third party consent are required. "
            "Privacy notice and third party consent are required."
        )
        issues = self._detect(text, "CONSENT", confidence=0.4)
        self.assertTrue(issues)
        issue = issues[0]
        self.assertEqual(issue.page_type, "CONSENT")
        self.assertEqual(issue.kind, "NOTE")
        self.assertEqual(issue.subtype, "BOILERPLATE_REPEAT")

    def test_resume_capability_not_inconsistency(self) -> None:
        text = (
            "Spring can connect to a database program. "
            "Java can connect to a database program."
        )
        issues = self._detect(text, "RESUME")
        if not issues:
            return
        issue = issues[0]
        self.assertNotEqual(issue.subtype, "INCONSISTENCY")
        self.assertEqual(issue.kind, "NOTE")


if __name__ == "__main__":
    unittest.main()

