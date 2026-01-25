"""Rule-based readability detector."""

from __future__ import annotations

import regex as re

from documind.schema import Issue, IssueI18n, IssueText, Location


LONG_SENTENCE_THRESHOLD = 120
YELLOW_SENTENCE_THRESHOLD = 150
RED_SENTENCE_THRESHOLD = 200

RESUME_LONG_THRESHOLD = 160
RESUME_YELLOW_THRESHOLD = 220
RESUME_RED_THRESHOLD = 280


def _is_noise_sentence(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    compact = re.sub(r"\s+", "", stripped)
    if len(stripped) < 15 or len(compact) < 10:
        return True
    if re.fullmatch(r"[\W\d_]+", stripped):
        return True
    if re.fullmatch(r"\(?\d+\)?[.)]?", stripped):
        return True
    lowered = stripped.lower()
    if any(token in lowered for token in ["http", "www", ".com", "=", "&", "/"]):
        return True
    letters = len(re.findall(r"\p{L}", stripped))
    numbers = len(re.findall(r"\p{N}", stripped))
    symbols = len(re.findall(r"[^\p{L}\p{N}\s]", stripped))
    non_letter_ratio = (numbers + symbols) / max(1, len(compact))
    return non_letter_ratio > 0.6


def _split_sentences(text: str) -> list[dict]:
    sentences: list[dict] = []
    pattern = re.compile(r".+?(?:[.!?]+|$)", re.DOTALL)

    for match in pattern.finditer(text):
        raw = match.group(0)
        if not raw.strip():
            continue
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        start = match.start() + leading
        end = match.end() - trailing
        sentence_text = text[start:end]
        if sentence_text.strip():
            sentences.append({"text": sentence_text, "start": start, "end": end})

    return sentences


def _severity_for_length(length: int, thresholds: tuple[int, int, int]) -> str:
    long_threshold, yellow_threshold, red_threshold = thresholds
    if length >= red_threshold:
        return "RED"
    if length >= yellow_threshold:
        return "YELLOW"
    return "GREEN"


def _thresholds_for_page_type(page_type: str | None) -> tuple[int, int, int]:
    if page_type == "RESUME":
        return (RESUME_LONG_THRESHOLD, RESUME_YELLOW_THRESHOLD, RESUME_RED_THRESHOLD)
    return (LONG_SENTENCE_THRESHOLD, YELLOW_SENTENCE_THRESHOLD, RED_SENTENCE_THRESHOLD)


def detect(
    pages: list[dict],
    language: str = "ko",
    page_profiles: list[dict] | None = None,
) -> list[Issue]:
    issues: list[Issue] = []

    page_type_map = {}
    if page_profiles:
        page_type_map = {profile["page"]: profile["type"] for profile in page_profiles}

    for page in pages:
        text = page.get("text", "")
        sentences = _split_sentences(text)
        page_type = page_type_map.get(page.get("page_number"), None)
        thresholds = _thresholds_for_page_type(page_type)

        for sentence in sentences:
            length = len(sentence["text"])
            if _is_noise_sentence(sentence["text"]):
                continue
            if length < thresholds[0]:
                continue
            severity = _severity_for_length(length, thresholds)
            start = sentence["start"]
            end = sentence["end"]
            evidence = sentence["text"].strip()
            if len(evidence) > 160:
                evidence = evidence[:157] + "..."

            i18n = IssueI18n(
                ko=IssueText(
                    message=f"긴 문장 감지 ({length}자).",
                    suggestion="문장을 더 짧게 분리하세요.",
                ),
                en=IssueText(
                    message=f"Long sentence detected ({length} chars).",
                    suggestion="Split the sentence into shorter ones.",
                ),
            )
            selected = i18n.en if language == "en" else i18n.ko

            kind = "NOTE" if page_type in {"CONSENT", "TERMS"} else "WARNING"
            issues.append(
                Issue(
                    id=f"readability_long_p{page['page_number']}_{start}",
                    category="readability",
                    kind=kind,
                    subtype="LONG_SENTENCE",
                    severity=severity,
                    message=selected.message,
                    evidence=evidence,
                    suggestion=selected.suggestion,
                    location=Location(
                        page=page["page_number"],
                        start_char=start,
                        end_char=end,
                    ),
                    confidence=0.6,
                    detector="rule_based",
                    i18n=i18n,
                )
            )

    return issues
