"""Document profile classification."""

from __future__ import annotations

import re
from typing import Dict, List


CONSENT_STRONG = {
    "개인정보 처리방침": 4,
    "제3자 제공": 3,
    "보유 및 이용 기간": 3,
    "수집·이용": 3,
    "수집 이용": 3,
    "동의서": 3,
    "고지": 2,
    "법": 2,
    "약관": 2,
    "privacy policy": 4,
    "third party": 3,
    "retention period": 3,
    "consent form": 3,
    "notice": 2,
    "law": 2,
    "terms": 2,
}

CONSENT_WEAK = {
    "동의": 1,
    "consent": 1,
}

CONSENT_BOOST_SIGNALS = [
    "개인정보 처리방침",
    "제3자 제공",
    "보유 및 이용 기간",
    "수집·이용",
    "동의서",
    "privacy policy",
    "third party",
    "retention period",
    "consent form",
]

RESUME_STRONG = {
    "이력서": 4,
    "자기소개서": 4,
    "지원서": 3,
    "성명": 2,
    "연락처": 2,
    "학력": 3,
    "경력": 3,
    "프로젝트": 3,
    "기술스택": 3,
    "자격증": 3,
    "활동": 2,
    "경험": 2,
    "resume": 3,
    "curriculum vitae": 3,
    "experience": 2,
    "education": 2,
    "skills": 2,
}

RESUME_BOOST_SIGNALS = [
    "학력",
    "경력",
    "프로젝트",
    "기술스택",
    "자격증",
    "education",
    "experience",
    "project",
    "skills",
    "certification",
]

TERMS_STRONG = {
    "약관": 4,
    "이용약관": 4,
    "용어의 정의": 4,
    "목적": 3,
    "회원": 3,
    "서비스의 제공": 3,
    "게시와 개정": 3,
    "면책": 3,
    "책임": 3,
    "서비스": 2,
    "조건": 2,
    "terms": 2,
    "conditions": 2,
    "liability": 3,
    "disclaimer": 3,
}

TERMS_ANCHORS = [
    "약관",
    "이용약관",
]

FORM_STRONG = {
    "설문": 3,
    "설문조사": 3,
    "점검지": 3,
    "체크리스트": 3,
    "문항": 2,
    "응답": 2,
    "참여경로": 2,
    "기타": 1,
    "survey": 2,
    "checklist": 2,
    "questionnaire": 2,
    "response": 2,
}

FORM_LIKERT = [
    "매우 아니다",
    "아니다",
    "보통",
    "그렇다",
    "매우 그렇다",
    "strongly disagree",
    "disagree",
    "neutral",
    "agree",
    "strongly agree",
]

REPORT_STRONG = {
    "보고서": 4,
    "브리프": 4,
    "동향": 3,
    "발간": 2,
    "백서": 4,
    "리포트": 3,
    "brief": 3,
    "report": 2,
    "white paper": 3,
    "issue brief": 3,
    "trend": 2,
}

REPORT_WEAK = {
    "요약": 1,
    "목차": 1,
    "서론": 1,
    "결론": 1,
    "summary": 1,
    "abstract": 1,
}


DOMINANT_THRESHOLD = 0.6
CONSENT_MIN_SCORE = 4
CONSENT_MARGIN = 2
RESUME_MIN_SCORE = 3
TERMS_MIN_SCORE = 2
TERMS_MARGIN = 2
FORM_MIN_SCORE = 3
REPORT_MIN_SCORE = 3
REPORT_MARGIN = 1
REPRESENTATIVE_TOP_N = 5
KEYWORD_TOP_N = 6
UNCERTAIN_THRESHOLD = 0.4
TERMS_ARTICLE_PATTERN = re.compile(r"제\s*\d+\s*조")


def _score_keywords(text: str, keywords: Dict[str, int]) -> tuple[int, List[str]]:
    lowered = text.lower()
    score = 0
    signals: List[str] = []
    for keyword, weight in keywords.items():
        if keyword in lowered:
            score += weight
            signals.append(keyword)
    return score, signals


def _count_keywords(text: str, keywords: List[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def _keyword_hits(text: str, keywords: List[str]) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in keywords if keyword in lowered]


def _representative_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentences: List[str] = []
    pattern = re.compile(r".+?(?:[.!?]+|$)", re.DOTALL)
    for match in pattern.finditer(text):
        sentence = match.group(0).strip()
        if sentence:
            sentences.append(sentence)

    keywords = list(CONSENT_STRONG.keys()) + list(CONSENT_WEAK.keys())
    keywords += list(RESUME_STRONG.keys()) + list(TERMS_STRONG.keys())
    keywords += list(FORM_STRONG.keys())

    keyword_candidates = []
    for item in lines + sentences:
        if len(item) < 12:
            continue
        if any(keyword in item.lower() for keyword in keywords):
            keyword_candidates.append(item)
            continue
        if TERMS_ARTICLE_PATTERN.search(item):
            keyword_candidates.append(item)

    candidates = lines + sentences
    candidates = [item for item in candidates if len(item) >= 20]
    if not candidates:
        candidates = lines or [text.strip()]

    seen = set()
    unique = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)

    longest = sorted(unique, key=len, reverse=True)
    keyword_unique = []
    seen_keyword = set()
    for item in keyword_candidates:
        if item in seen_keyword:
            continue
        seen_keyword.add(item)
        keyword_unique.append(item)

    selected = keyword_unique[:KEYWORD_TOP_N] + longest[:REPRESENTATIVE_TOP_N]
    final = []
    seen = set()
    for item in selected:
        if item in seen:
            continue
        seen.add(item)
        final.append(item)
    return "\n".join(final)


def classify_text(text: str) -> dict:
    consent_score, consent_signals = _score_keywords(text, CONSENT_STRONG)
    weak_score, weak_signals = _score_keywords(text, CONSENT_WEAK)
    consent_score += weak_score
    consent_signals += weak_signals
    consent_strong_hits = _count_keywords(text, CONSENT_BOOST_SIGNALS)
    if consent_strong_hits >= 2:
        consent_score += 5
        for keyword in _keyword_hits(text, CONSENT_BOOST_SIGNALS):
            if keyword not in consent_signals:
                consent_signals.append(keyword)

    resume_score, resume_signals = _score_keywords(text, RESUME_STRONG)
    resume_strong_hits = _count_keywords(text, RESUME_BOOST_SIGNALS)
    if resume_strong_hits:
        resume_score += min(4, resume_strong_hits * 2)
        for keyword in _keyword_hits(text, RESUME_BOOST_SIGNALS):
            if keyword not in resume_signals:
                resume_signals.append(keyword)
    terms_score, terms_signals = _score_keywords(text, TERMS_STRONG)
    terms_article_hits = len(TERMS_ARTICLE_PATTERN.findall(text))
    terms_anchor_hits = _count_keywords(text, TERMS_ANCHORS)
    if terms_article_hits:
        terms_score += min(5, terms_article_hits) * 6
        if "제N조" not in terms_signals:
            terms_signals.append("제N조")
    if terms_article_hits:
        terms_anchor_hits += 1

    form_score, form_signals = _score_keywords(text, FORM_STRONG)
    likert_hits = _count_keywords(text, FORM_LIKERT)
    if likert_hits >= 3:
        form_score += 3
        for keyword in _keyword_hits(text, FORM_LIKERT):
            if keyword not in form_signals:
                form_signals.append(keyword)
    checkbox_hits = len(re.findall(r"[□■○●☑✓]", text))
    if checkbox_hits >= 3:
        form_score += min(4, checkbox_hits)
    question_hits = len(re.findall(r"(?m)^\s*\d+[.)]", text))
    if question_hits >= 3:
        form_score += min(4, question_hits)

    report_score, report_signals = _score_keywords(text, REPORT_STRONG)
    report_weak_score, report_weak_signals = _score_keywords(text, REPORT_WEAK)
    report_score += report_weak_score
    report_signals += report_weak_signals

    scores = {
        "CONSENT": consent_score,
        "RESUME": resume_score,
        "TERMS": terms_score,
        "FORM": form_score,
        "REPORT": report_score,
    }

    best_type = "GENERIC"
    best_score = 0
    best_signals: List[str] = []

    consent_wins = (
        consent_score >= CONSENT_MIN_SCORE
        and consent_score >= max(resume_score, terms_score, form_score, report_score) + CONSENT_MARGIN
    )
    terms_wins = (
        terms_anchor_hits >= 1
        and terms_score >= TERMS_MIN_SCORE
        and terms_score >= max(resume_score, form_score, report_score) + TERMS_MARGIN
    )
    report_wins = (
        report_score >= REPORT_MIN_SCORE
        and report_score >= max(resume_score, form_score, terms_score) + REPORT_MARGIN
    )
    if consent_wins:
        best_type = "CONSENT"
        best_score = consent_score
        best_signals = consent_signals
    elif terms_wins:
        best_type = "TERMS"
        best_score = terms_score
        best_signals = terms_signals
    elif report_wins:
        best_type = "REPORT"
        best_score = report_score
        best_signals = report_signals
    elif form_score >= FORM_MIN_SCORE and form_score >= max(resume_score, terms_score):
        best_type = "FORM"
        best_score = form_score
        best_signals = form_signals
    elif resume_score >= RESUME_MIN_SCORE and resume_score >= terms_score:
        best_type = "RESUME"
        best_score = resume_score
        best_signals = resume_signals
    elif terms_score >= TERMS_MIN_SCORE and terms_anchor_hits >= 1:
        best_type = "TERMS"
        best_score = terms_score
        best_signals = terms_signals
    elif report_score >= REPORT_MIN_SCORE:
        best_type = "REPORT"
        best_score = report_score
        best_signals = report_signals

    score_values = sorted(scores.values(), reverse=True)
    top_score = score_values[0] if score_values else 0
    second_score = score_values[1] if len(score_values) > 1 else 0

    if best_type == "GENERIC":
        confidence = 0.2
    else:
        confidence = (top_score - second_score) / (top_score + second_score + 1e-6)
        confidence = confidence * 1.6 + 0.05
        confidence = max(0.0, min(1.0, confidence))

    return {
        "type": best_type,
        "confidence": confidence,
        "signals": best_signals[:8],
        "scores": scores,
        "consent_strong_hits": consent_strong_hits,
        "terms_article_hits": terms_article_hits,
        "terms_anchor_hits": terms_anchor_hits,
    }


def classify_pages(pages: list[dict]) -> list[dict]:
    profiles: list[dict] = []
    for page in pages:
        page_number = page.get("page_number", 0)
        text = page.get("text", "")
        representative_text = _representative_text(text)
        profile = classify_text(representative_text)
        consent_override = (
            profile["type"] == "CONSENT"
            and profile.get("consent_strong_hits", 0) >= 2
        )
        terms_override = (
            profile["type"] == "TERMS"
            and profile.get("terms_article_hits", 0) >= 2
        )
        if consent_override and profile["confidence"] < UNCERTAIN_THRESHOLD:
            profile["confidence"] = UNCERTAIN_THRESHOLD
        if terms_override and profile["confidence"] < UNCERTAIN_THRESHOLD:
            profile["confidence"] = UNCERTAIN_THRESHOLD
        if profile["confidence"] < UNCERTAIN_THRESHOLD and not (consent_override or terms_override):
            profile["type"] = "UNCERTAIN"
        profiles.append(
            {
                "page": page_number,
                "type": profile["type"],
                "confidence": profile["confidence"],
                "signals": profile["signals"],
                "consent_score": profile.get("scores", {}).get("CONSENT", 0),
                "resume_score": profile.get("scores", {}).get("RESUME", 0),
                "terms_score": profile.get("scores", {}).get("TERMS", 0),
                "form_score": profile.get("scores", {}).get("FORM", 0),
            }
        )
    return profiles


def dominant_type_from_pages(page_profiles: list[dict]) -> str:
    weights: Dict[str, float] = {}
    for profile in page_profiles:
        doc_type = profile.get("type", "GENERIC")
        if doc_type == "UNCERTAIN":
            continue
        confidence = float(profile.get("confidence", 0.0))
        weights[doc_type] = weights.get(doc_type, 0.0) + max(0.1, confidence)

    if not weights:
        return "GENERIC"

    sorted_weights = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    if len(sorted_weights) == 1:
        return sorted_weights[0][0]

    top_type, top_weight = sorted_weights[0]
    second_weight = sorted_weights[1][1]
    total_weight = sum(weights.values())

    if top_weight == second_weight or (top_weight / total_weight) < DOMINANT_THRESHOLD:
        return "MIXED"
    return top_type


def classify_document(text: str) -> dict:
    return classify_text(text)
