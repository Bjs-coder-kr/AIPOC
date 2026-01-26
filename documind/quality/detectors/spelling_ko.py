"""Rule-based Korean spelling/spacing detector."""

from __future__ import annotations

from dataclasses import dataclass

import regex as re

from documind.schema import Issue, IssueI18n, IssueText, Location


MAX_ISSUES_PER_PAGE = 5


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern
    correction: str
    subtype: str
    kind: str
    severity: str


def _tail_boundary() -> str:
    return r"(?=[^가-힣]|$)"


RULES: list[Rule] = [
    Rule(re.compile(r"할수있다" + _tail_boundary()), "할 수 있다", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"할수없다" + _tail_boundary()), "할 수 없다", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"할수도" + _tail_boundary()), "할 수도", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"할수" + _tail_boundary()), "할 수", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"될수있다" + _tail_boundary()), "될 수 있다", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"될수없다" + _tail_boundary()), "될 수 없다", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"될수도" + _tail_boundary()), "될 수도", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"될수" + _tail_boundary()), "될 수", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"볼수" + _tail_boundary()), "볼 수", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"수있다" + _tail_boundary()), "수 있다", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"수없다" + _tail_boundary()), "수 없다", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"할것" + _tail_boundary()), "할 것", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"될것" + _tail_boundary()), "될 것", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"있는것" + _tail_boundary()), "있는 것", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"없는것" + _tail_boundary()), "없는 것", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"같은것" + _tail_boundary()), "같은 것", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"좋은것" + _tail_boundary()), "좋은 것", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"할뿐" + _tail_boundary()), "할 뿐", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"될뿐" + _tail_boundary()), "될 뿐", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"있을뿐" + _tail_boundary()), "있을 뿐", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"없을뿐" + _tail_boundary()), "없을 뿐", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"하는대로" + _tail_boundary()), "하는 대로", "SPACING_SUSPECT", "WARNING", "YELLOW"),
    Rule(re.compile(r"되서"), "돼서", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"되요"), "돼요", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"됬어요"), "됐어요", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"됬습니다"), "됐습니다", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"됬다"), "됐다", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"됬"), "됐", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"안되요"), "안 돼요", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"안됩니다"), "안 됩니다", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"안되면"), "안 되면", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"안되서"), "안 돼서", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"안되다"), "안 되다", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"안되는"), "안 되는", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"안됬어요"), "안 됐어요", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"안됬다"), "안 됐다", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"안됬"), "안 됐", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"않되"), "안 되", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"몇\s*일"), "며칠", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"오랫만"), "오랜만", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"웬지"), "왠지", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"왠만"), "웬만", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"할려고"), "하려고", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"할려"), "하려", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"할께요"), "할게요", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"할께"), "할게", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"갈께요"), "갈게요", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"올께요"), "올게요", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"될께요"), "될게요", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
    Rule(re.compile(r"거에요"), "거예요", "COMMON_KO_TYPO", "WARNING", "YELLOW"),
]


def _truncate(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def detect(pages: list[dict], language: str = "ko") -> list[Issue]:
    if language != "ko":
        return []

    issues: list[Issue] = []

    for page in pages:
        text = page.get("text", "")
        if not text.strip():
            continue
        page_number = page.get("page_number", 0)
        page_count = 0

        for rule in RULES:
            for match in rule.pattern.finditer(text):
                if page_count >= MAX_ISSUES_PER_PAGE:
                    break
                wrong = match.group(0)
                message_ko = f"맞춤법 의심 표현: '{wrong}'"
                message_en = f"Common Korean typo detected: '{wrong}'"
                suggestion_ko = f"교정안: '{rule.correction}'"
                suggestion_en = f"Suggested form: '{rule.correction}'"
                i18n = IssueI18n(
                    ko=IssueText(message=message_ko, suggestion=suggestion_ko),
                    en=IssueText(message=message_en, suggestion=suggestion_en),
                )
                issues.append(
                    Issue(
                        id=f"spelling_common_p{page_number}_{match.start()}",
                        category="spelling",
                        kind=rule.kind,
                        subtype=rule.subtype,
                        severity=rule.severity,
                        message=i18n.ko.message,
                        evidence=_truncate(wrong),
                        suggestion=i18n.ko.suggestion,
                        location=Location(
                            page=page_number,
                            start_char=match.start(),
                            end_char=match.end(),
                        ),
                        confidence=0.6 if rule.kind == "ERROR" else 0.5,
                        detector="rule_based",
                        i18n=i18n,
                    )
                )
                page_count += 1
            if page_count >= MAX_ISSUES_PER_PAGE:
                break

    return issues
