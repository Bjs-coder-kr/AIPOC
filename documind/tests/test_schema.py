"""Basic schema tests."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from documind.schema import (
    DocumentMeta,
    DocumentProfile,
    PageProfile,
    Issue,
    IssueI18n,
    IssueText,
    Location,
    Report,
)


class TestSchema(unittest.TestCase):
    def test_report_schema_roundtrip(self) -> None:
        i18n = IssueI18n(
            ko=IssueText(
                message="긴 문장 감지.",
                suggestion="문장을 더 짧게 분리하세요.",
            ),
            en=IssueText(
                message="Long sentence detected.",
                suggestion="Split the sentence into shorter ones.",
            ),
        )
        issue = Issue(
            id="readability_long_p1_0",
            category="readability",
            kind="WARNING",
            subtype="LONG_SENTENCE",
            severity="YELLOW",
            message=i18n.ko.message,
            evidence="Example evidence.",
            suggestion=i18n.ko.suggestion,
            location=Location(page=1, start_char=0, end_char=10),
            confidence=0.7,
            detector="rule_based",
            i18n=i18n,
        )
        report = Report(
            document_meta=DocumentMeta(
                file_name="sample.pdf",
                page_count=1,
                textless_pages=0,
                raw_char_count=120,
                normalized_char_count=120,
                scan_like=False,
                scan_like_ratio=0.0,
                scan_level="NONE",
                document_profile=DocumentProfile(
                    type="GENERIC",
                    dominant_type="GENERIC",
                    confidence=0.7,
                    signals=[],
                ),
                page_profiles=[
                    PageProfile(
                        page=1,
                        type="GENERIC",
                        confidence=0.7,
                        signals=[],
                    )
                ],
            ),
            score_confidence="HIGH",
            raw_score=95,
            overall_score=95,
            limitations=[],
            issues=[issue],
        )
        payload = report.model_dump()
        self.assertEqual(payload["overall_score"], 95)
        self.assertEqual(payload["issues"][0]["severity"], "YELLOW")


if __name__ == "__main__":
    unittest.main()

