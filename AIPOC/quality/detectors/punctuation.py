"""Rule-based punctuation detector."""

from __future__ import annotations

from dataclasses import dataclass

import regex as re

from documind.schema import Issue, IssueI18n, IssueText, Location


PUNCTUATION_PATTERN = re.compile(r"[!?]{2,}|\.{4,}|,{3,}")
BRACKET_PAIRS = [
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ("「", "」"),
    ("『", "』"),
    ("“", "”"),
    ("‘", "’"),
]
OPENERS = {opener for opener, _ in BRACKET_PAIRS}
CLOSER_TO_OPENER = {closer: opener for opener, closer in BRACKET_PAIRS}
OPENER_TO_CLOSER = {opener: closer for opener, closer in BRACKET_PAIRS}
ENUM_CLOSER_PATTERNS = [
    re.compile(r"\b\d+\s*\)$"),
    re.compile(r"\b(?:ex|예)\s*\)$", re.IGNORECASE),
    re.compile(r"\b[가-힣]\s*\)$"),
]


@dataclass(frozen=True)
class BracketMismatch:
    index: int
    kind: str
    opener: str
    closer: str
    hint: str


def _truncate(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _extract_context(text: str, index: int, radius: int = 60) -> str:
    start = max(0, index - radius)
    end = min(len(text), index + radius)
    return _truncate(text[start:end])


def _is_enum_closer(text: str, idx: int) -> bool:
    window_start = max(0, idx - 12)
    window = text[window_start : idx + 1]
    for pattern in ENUM_CLOSER_PATTERNS:
        match = pattern.search(window)
        if not match:
            continue
        match_start = window_start + match.start()
        if match_start > 0 and text[match_start - 1] in {"(", "（"}:
            return False
        return True
    return False


def find_bracket_mismatch(text: str) -> BracketMismatch | None:
    stack: list[tuple[str, int]] = []
    for idx, char in enumerate(text):
        if char in OPENERS:
            stack.append((char, idx))
            continue
        if char == ")" and _is_enum_closer(text, idx):
            if not (stack and stack[-1][0] == "("):
                continue
        if char in CLOSER_TO_OPENER:
            expected_opener = CLOSER_TO_OPENER[char]
            if not stack or stack[-1][0] != expected_opener:
                hint = f"열림 '{expected_opener}' 누락"
                return BracketMismatch(
                    index=idx,
                    kind="missing_opener",
                    opener=expected_opener,
                    closer=char,
                    hint=hint,
                )
            stack.pop()
    if stack:
        opener, idx = stack[-1]
        closer = OPENER_TO_CLOSER.get(opener, "")
        hint = f"닫힘 '{closer}' 누락" if closer else "닫힘 누락"
        return BracketMismatch(
            index=idx,
            kind="missing_closer",
            opener=opener,
            closer=closer,
            hint=hint,
        )
    return None


def _make_issue(
    page_number: int,
    start: int,
    end: int,
    subtype: str,
    message_ko: str,
    message_en: str,
    suggestion_ko: str,
    suggestion_en: str,
    evidence: str,
) -> Issue:
    i18n = IssueI18n(
        ko=IssueText(message=message_ko, suggestion=suggestion_ko),
        en=IssueText(message=message_en, suggestion=suggestion_en),
    )
    return Issue(
        id=f"punctuation_{subtype.lower()}_p{page_number}_{start}",
        category="logic",
        kind="WARNING",
        subtype=subtype,
        severity="YELLOW",
        message=i18n.ko.message,
        evidence=_truncate(evidence),
        suggestion=i18n.ko.suggestion,
        location=Location(page=page_number, start_char=start, end_char=end),
        confidence=0.6,
        detector="rule_based",
        i18n=i18n,
    )


def detect(pages: list[dict], language: str = "ko") -> list[Issue]:
    issues: list[Issue] = []
    use_lang = "en" if language == "en" else "ko"

    for page in pages:
        text = page.get("text", "")
        if not text.strip():
            continue
        page_number = page.get("page_number", 0)

        mismatch = find_bracket_mismatch(text)
        if mismatch is not None:
            message_ko = (
                "문장부호(괄호/인용부호) 짝이 맞지 않는 부분이 있습니다. "
                f"({mismatch.hint})"
            )
            message_en = "Bracket/quote pairs appear to be unbalanced."
            suggestion_ko = "괄호/인용부호 짝을 확인하세요."
            suggestion_en = "Check bracket/quote pairing."
            issue = _make_issue(
                page_number,
                mismatch.index,
                min(len(text), mismatch.index + 1),
                "BRACKET_MISMATCH",
                message_ko,
                message_en,
                suggestion_ko,
                suggestion_en,
                _extract_context(text, mismatch.index),
            )
            issue.message = issue.i18n.en.message if use_lang == "en" else issue.i18n.ko.message
            issue.suggestion = (
                issue.i18n.en.suggestion if use_lang == "en" else issue.i18n.ko.suggestion
            )
            issues.append(issue)

        match = PUNCTUATION_PATTERN.search(text)
        if match:
            message_ko = "비정상적인 구두점 반복이 있습니다."
            message_en = "Unusual punctuation repetition detected."
            suggestion_ko = "구두점 사용을 확인하세요."
            suggestion_en = "Review punctuation usage."
            issue = _make_issue(
                page_number,
                match.start(),
                match.end(),
                "PUNCTUATION_ANOMALY",
                message_ko,
                message_en,
                suggestion_ko,
                suggestion_en,
                text[match.start() : match.end()],
            )
            issue.message = issue.i18n.en.message if use_lang == "en" else issue.i18n.ko.message
            issue.suggestion = (
                issue.i18n.en.suggestion if use_lang == "en" else issue.i18n.ko.suggestion
            )
            issues.append(issue)

    return issues
