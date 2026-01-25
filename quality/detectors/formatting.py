"""Rule-based formatting detector."""

from __future__ import annotations

import regex as re

from documind.schema import Issue, IssueI18n, IssueText, Location


NUMBERED_PATTERN = re.compile(r"^\s*(\d+)[.)]\s+")


def _truncate(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def detect(pages: list[dict], language: str = "ko") -> list[Issue]:
    issues: list[Issue] = []
    use_lang = "en" if language == "en" else "ko"

    for page in pages:
        text = page.get("text", "")
        if not text.strip():
            continue
        page_number = page.get("page_number", 0)

        lines = text.splitlines()
        numbers: list[tuple[int, int, str]] = []
        offset = 0
        for line in lines:
            match = NUMBERED_PATTERN.match(line)
            if match:
                number = int(match.group(1))
                numbers.append((number, offset, line))
            offset += len(line) + 1

        if len(numbers) >= 3:
            for idx in range(1, len(numbers)):
                prev_num = numbers[idx - 1][0]
                current_num, start_offset, line_text = numbers[idx]
                if current_num != prev_num + 1:
                    message_ko = "번호/문항 흐름이 끊긴 것 같습니다."
                    message_en = "Numbered list flow appears to break."
                    suggestion_ko = "번호 순서가 자연스러운지 확인해 주세요."
                    suggestion_en = "Check the numbering sequence."
                    evidence = _truncate(line_text)
                    i18n = IssueI18n(
                        ko=IssueText(message=message_ko, suggestion=suggestion_ko),
                        en=IssueText(message=message_en, suggestion=suggestion_en),
                    )
                    selected = i18n.en if use_lang == "en" else i18n.ko
                    issues.append(
                        Issue(
                            id=f"formatting_bullet_break_p{page_number}_{start_offset}",
                            category="logic",
                            kind="WARNING",
                            subtype="BULLET_FLOW_BREAK",
                            severity="YELLOW",
                            message=selected.message,
                            evidence=evidence,
                            suggestion=selected.suggestion,
                            location=Location(
                                page=page_number,
                                start_char=start_offset,
                                end_char=start_offset + len(line_text),
                            ),
                            confidence=0.5,
                            detector="rule_based",
                            i18n=i18n,
                        )
                    )
                    break

    return issues
