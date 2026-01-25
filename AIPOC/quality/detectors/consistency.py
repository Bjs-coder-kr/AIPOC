"""Rule-based consistency detector."""

from __future__ import annotations

import regex as re

from documind.schema import Issue, IssueI18n, IssueText, Location


DATE_PATTERNS = {
    "dash": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "dot": re.compile(r"\b\d{4}\.\d{2}\.\d{2}\b"),
    "slash": re.compile(r"\b\d{4}/\d{2}/\d{2}\b"),
}
COMMA_NUMBER = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
PLAIN_NUMBER = re.compile(r"\b\d{4,}\b")


def _truncate(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


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
    language: str,
) -> Issue:
    i18n = IssueI18n(
        ko=IssueText(message=message_ko, suggestion=suggestion_ko),
        en=IssueText(message=message_en, suggestion=suggestion_en),
    )
    selected = i18n.en if language == "en" else i18n.ko
    return Issue(
        id=f"consistency_{subtype.lower()}_p{page_number}_{start}",
        category="logic",
        kind="WARNING",
        subtype=subtype,
        severity="YELLOW",
        message=selected.message,
        evidence=_truncate(evidence),
        suggestion=selected.suggestion,
        location=Location(page=page_number, start_char=start, end_char=end),
        confidence=0.55,
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

        date_hits = {key: pattern.search(text) for key, pattern in DATE_PATTERNS.items()}
        used_formats = [key for key, match in date_hits.items() if match]
        if len(used_formats) >= 2:
            match = date_hits[used_formats[1]]
            if match:
                issues.append(
                    _make_issue(
                        page_number,
                        match.start(),
                        match.end(),
                        "DATE_FORMAT_INCONSISTENCY",
                        "날짜 표기 형식이 혼용된 것 같습니다.",
                        "Date formats appear to be inconsistent.",
                        "날짜 표기 형식을 통일해 주세요.",
                        "Standardize date formatting.",
                        match.group(0),
                        use_lang,
                    )
                )

        comma_match = COMMA_NUMBER.search(text)
        plain_match = PLAIN_NUMBER.search(text)
        if comma_match and plain_match:
            issues.append(
                _make_issue(
                    page_number,
                    plain_match.start(),
                    plain_match.end(),
                    "NUMBER_FORMAT_INCONSISTENCY",
                    "숫자 표기 형식이 혼용된 것 같습니다.",
                    "Number formatting appears inconsistent.",
                    "숫자 표기 형식을 통일해 주세요.",
                    "Standardize number formatting.",
                    plain_match.group(0),
                    use_lang,
                )
            )

    return issues
