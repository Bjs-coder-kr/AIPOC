"""Quality pipeline orchestrator."""

from __future__ import annotations

import regex as re

from documind.ingest.loader import load_document
from documind.profile.classify import (
    classify_pages,
    classify_text,
    dominant_type_from_pages,
)
from documind.quality.detectors import (
    consistency,
    formatting,
    punctuation,
    readability,
    redundancy,
    spelling_ko,
)
from documind.schema import DocumentMeta, Issue, Report
from documind.text.normalize import normalize_pages
from documind.utils.logging import setup_logging


KIND_DEDUCTIONS = {
    "ERROR": 10,
    "WARNING": 5,
    "NOTE": 0,
}

UNCERTAIN_THRESHOLD = 0.35
PAGE_TYPE_OVERRIDE_THRESHOLD = 0.35

SEVERITY_ORDER = {
    "RED": 3,
    "YELLOW": 2,
    "GREEN": 1,
}

UNCERTAIN_SUGGESTION = {
    "ko": "문맥/타입 판정이 불확실하여 참고용입니다. 원문을 확인하세요.",
    "en": "Context/type is uncertain, so this is for reference only. Please review the original text.",
}

FORM_SUGGESTION = {
    "ko": "양식/문항 반복 구조일 수 있으니, 의도 여부만 점검하세요.",
    "en": "This may be a repeated form/question pattern. Verify whether repetition is intended.",
}

CONSENT_SUGGESTION = {
    "ko": "동일 문구 의도 여부만 점검하세요.",
    "en": "Check whether identical wording is intended.",
}

ACTIONABLE_ALLOWLIST = {
    "DATE_FORMAT_INCONSISTENCY",
    "NUMBER_FORMAT_INCONSISTENCY",
}

FORMLIKE_PATTERNS = [
    re.compile(r"상/중/하"),
    re.compile(r"해당\s*될\s*경우"),
    re.compile(r"예\s*/?\s*아니오"),
    re.compile(r"군필"),
    re.compile(r"체크"),
    re.compile(r"선택"),
    re.compile(r"기간"),
    re.compile(r"취득일"),
    re.compile(r"[□■○●]"),
    re.compile(r"\bO\s*/\s*X\b", re.IGNORECASE),
]


def _dedup_issues(issues: list[Issue]) -> list[Issue]:
    deduped: dict[tuple[int, int, int], Issue] = {}
    for issue in issues:
        key = (issue.location.page, issue.location.start_char, issue.location.end_char)
        if key not in deduped:
            deduped[key] = issue
            continue
        existing = deduped[key]
        if SEVERITY_ORDER[issue.severity] > SEVERITY_ORDER[existing.severity]:
            deduped[key] = issue
    return list(deduped.values())


def _score(issues: list[Issue]) -> int:
    score = 100
    for issue in issues:
        if issue.kind in {"ERROR", "WARNING"}:
            score -= KIND_DEDUCTIONS.get(issue.kind, 0)
    return max(0, min(100, score))


def _score_confidence(scan_level: str) -> str:
    if scan_level == "HIGH":
        return "LOW"
    if scan_level == "PARTIAL":
        return "MED"
    return "HIGH"


def _attach_page_profile(issues: list[Issue], page_profiles: list[dict]) -> None:
    profile_map = {
        profile["page"]: (profile["type"], profile.get("confidence"))
        for profile in page_profiles
    }
    for issue in issues:
        page = issue.location.page
        if page not in profile_map:
            continue
        page_type, confidence = profile_map[page]
        if (
            issue.page_type in {None, "UNCERTAIN"}
            and page_type in {"CONSENT", "TERMS", "RESUME", "FORM", "REPORT", "GENERIC"}
            and confidence is not None
            and confidence >= PAGE_TYPE_OVERRIDE_THRESHOLD
        ):
            issue.page_type = page_type
        if issue.page_type_confidence is None and confidence is not None:
            issue.page_type_confidence = float(confidence)


def _apply_uncertain_policy(issue: Issue, language: str) -> None:
    issue.kind = "NOTE"
    issue.subtype = None
    issue.i18n.ko.suggestion = UNCERTAIN_SUGGESTION["ko"]
    issue.i18n.en.suggestion = UNCERTAIN_SUGGESTION["en"]
    issue.suggestion = (
        UNCERTAIN_SUGGESTION["en"] if language == "en" else UNCERTAIN_SUGGESTION["ko"]
    )


def _should_keep_actionable(issue: Issue) -> bool:
    # TODO: 확장 포인트 - 고위험 패턴 allowlist를 유형별로 정교화.
    if issue.subtype in ACTIONABLE_ALLOWLIST and issue.page_type in {"CONSENT", "TERMS"}:
        return True
    return False


def _is_formlike_text(text: str) -> bool:
    if not text:
        return False
    return any(pattern.search(text) for pattern in FORMLIKE_PATTERNS)


def _apply_bracket_mismatch_policy(issue: Issue, page_text: str | None) -> None:
    if issue.page_type in {"CONSENT", "TERMS", "FORM"}:
        issue.kind = "NOTE"
        return
    if issue.page_type != "RESUME" or not page_text:
        return
    if not _is_formlike_text(page_text):
        return
    mismatch = punctuation.find_bracket_mismatch(page_text)
    if not mismatch:
        return
    if mismatch.kind == "missing_opener" and mismatch.closer == ")":
        issue.kind = "NOTE"


def _apply_redundancy_policy(issue: Issue, language: str) -> None:
    if issue.page_type == "FORM":
        issue.kind = "NOTE"
        issue.subtype = "FORM_REPEAT"
        issue.i18n.ko.suggestion = FORM_SUGGESTION["ko"]
        issue.i18n.en.suggestion = FORM_SUGGESTION["en"]
        issue.suggestion = FORM_SUGGESTION["en"] if language == "en" else FORM_SUGGESTION["ko"]
        return
    if issue.page_type in {"CONSENT", "TERMS"}:
        issue.kind = "NOTE"
        issue.subtype = "BOILERPLATE_REPEAT"
        issue.i18n.ko.suggestion = CONSENT_SUGGESTION["ko"]
        issue.i18n.en.suggestion = CONSENT_SUGGESTION["en"]
        issue.suggestion = (
            CONSENT_SUGGESTION["en"] if language == "en" else CONSENT_SUGGESTION["ko"]
        )


def _apply_readability_policy(issue: Issue) -> None:
    if issue.page_type in {"CONSENT", "TERMS", "FORM"}:
        issue.kind = "NOTE"


def _apply_logic_policy(issue: Issue, page_text: str | None) -> None:
    if issue.subtype == "BRACKET_MISMATCH":
        _apply_bracket_mismatch_policy(issue, page_text)
        return
    if issue.page_type == "FORM":
        issue.kind = "NOTE"
        return
    if issue.page_type in {"CONSENT", "TERMS"} and not _should_keep_actionable(issue):
        issue.kind = "NOTE"


def _apply_issue_policies(
    issues: list[Issue], page_profiles: list[dict], language: str, pages: list[dict]
) -> list[Issue]:
    _attach_page_profile(issues, page_profiles)
    page_text_map = {page.get("page_number"): page.get("text", "") for page in pages}
    for issue in issues:
        if (
            issue.page_type_confidence is not None
            and issue.page_type_confidence < UNCERTAIN_THRESHOLD
        ):
            issue.page_type = "UNCERTAIN"
        if issue.page_type == "UNCERTAIN":
            _apply_uncertain_policy(issue, language)
            issue.severity = "GREEN"
            continue

        if issue.category == "redundancy":
            _apply_redundancy_policy(issue, language)
        elif issue.category == "readability":
            _apply_readability_policy(issue)
        else:
            _apply_logic_policy(issue, page_text_map.get(issue.location.page))

        if issue.kind == "NOTE":
            issue.severity = "GREEN"
    return issues


def run_pipeline(file_bytes: bytes, file_name: str, language: str = "ko") -> Report:
    logger = setup_logging()
    language = "en" if language == "en" else "ko"
    logger.info("LOAD")
    loaded = load_document(file_bytes, file_name)

    logger.info("NORMALIZE")
    normalized = normalize_pages(loaded["pages"])

    page_profiles = classify_pages(normalized["pages"])
    logger.info("DETECT_READABILITY")
    issues = readability.detect(
        normalized["pages"],
        language=language,
        page_profiles=page_profiles,
    )

    logger.info("DETECT_REDUNDANCY")
    profile_text = "\n\n".join(page["text"] for page in normalized["pages"][:3])
    document_profile = classify_text(profile_text)
    dominant_type = dominant_type_from_pages(page_profiles)
    if document_profile["confidence"] < 0.6 or dominant_type == "MIXED":
        document_profile["dominant_type"] = "MIXED"
    else:
        document_profile["dominant_type"] = dominant_type
    issues.extend(
        redundancy.detect(
            normalized["pages"],
            language=language,
            page_profiles=page_profiles,
        )
    )
    logger.info("DETECT_PUNCTUATION")
    issues.extend(punctuation.detect(normalized["pages"], language=language))
    logger.info("DETECT_FORMATTING")
    issues.extend(formatting.detect(normalized["pages"], language=language))
    logger.info("DETECT_CONSISTENCY")
    issues.extend(consistency.detect(normalized["pages"], language=language))
    if language == "ko":
        logger.info("DETECT_SPELLING_KO")
        issues.extend(spelling_ko.detect(normalized["pages"], language=language))

    logger.info("SCORE")
    issues = _apply_issue_policies(issues, page_profiles, language, normalized["pages"])
    issues = _dedup_issues(issues)
    raw_score = _score(issues)

    meta_payload = {
        **loaded["meta"],
        "normalized_char_count": normalized["normalized_char_count"],
        "document_profile": document_profile,
        "page_profiles": page_profiles,
    }
    score_confidence = _score_confidence(meta_payload["scan_level"])
    limitations: list[str] = []
    if score_confidence == "LOW":
        limitations = [
            (
                "텍스트 추출량이 부족하여 점수 산정이 제한됩니다."
                if language == "ko"
                else "Insufficient extracted text limits scoring accuracy."
            )
        ]

    report = Report(
        document_meta=DocumentMeta(**meta_payload),
        score_confidence=score_confidence,
        raw_score=raw_score,
        overall_score=None if score_confidence == "LOW" else raw_score,
        limitations=limitations,
        issues=issues,
    )
    return report
