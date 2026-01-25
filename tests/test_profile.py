"""Profile classifier tests."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from documind.profile.classify import classify_text


class TestProfileClassifier(unittest.TestCase):
    def test_classify_consent(self) -> None:
        text = "개인정보 처리방침 및 제3자 제공, 보유 및 이용 기간에 관한 동의서"
        profile = classify_text(text)
        self.assertEqual(profile["type"], "CONSENT")

    def test_classify_resume(self) -> None:
        text = "이력서 자기소개서 지원서: 성명, 연락처, 학력, 경력, 프로젝트, 기술스택"
        profile = classify_text(text)
        self.assertEqual(profile["type"], "RESUME")


if __name__ == "__main__":
    unittest.main()

