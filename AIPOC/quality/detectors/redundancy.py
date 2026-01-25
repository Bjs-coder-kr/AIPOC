"""Rule-based redundancy detector."""

from __future__ import annotations

import regex as re
from rapidfuzz import fuzz

from documind.schema import Issue, IssueI18n, IssueText, Location, MatchedTo


SIMILARITY_THRESHOLD = 90
MAX_SENTENCES = 200
MAX_LOOKAHEAD = 50
CONSENT_CONF_THRESHOLD = 0.35
BOILERPLATE_KEYWORDS = [
    "개인정보",
    "동의",
    "제3자",
    "보유",
    "이용",
    "수집",
    "고지",
    "처리방침",
    "약관",
    "법",
    "privacy",
    "consent",
    "third party",
    "retention",
    "notice",
    "terms",
]

TECH_KEYWORDS = [
    "자바",
    "스프링",
    "파이썬",
    "리액트",
    "도커",
    "쿠버네티스",
    "aws",
    "java",
    "kotlin",
    "mysql",
    "postgresql",
    "postgres",
    "oracle",
    "mssql",
    "sqlserver",
    "jsp",
    "spring",
    "springboot",
    "spring-boot",
    "react",
    "docker",
    "kubernetes",
    "node",
    "nodejs",
    "javascript",
    "typescript",
    "python",
]

TECH_KEYWORD_SET = {keyword.lower() for keyword in TECH_KEYWORDS}

KOREAN_CAPABILITY_PATTERN = re.compile(r"(할\s*수\s*있|수\s*있|가능하)")

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


def _normalize_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[\p{L}\p{N}]+", text.lower())
    return {token for token in tokens if len(token) >= 2}


def _extract_key_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    lowered = text.lower()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9.+#-]*", text):
        tokens.add(token.lower())
    for token in re.findall(r"\d+(?:[./-]\d+)+", text):
        tokens.add(token)
    for token in re.findall(r"\d+", text):
        if len(token) >= 2:
            tokens.add(token)
    for keyword in TECH_KEYWORDS:
        if keyword in lowered:
            tokens.add(keyword)
    return tokens


def _boilerplate_score(text: str) -> int:
    lowered = text.lower()
    return sum(1 for keyword in BOILERPLATE_KEYWORDS if keyword in lowered)


def _is_verbatim(ratio: float, tokens_a: set[str], tokens_b: set[str]) -> bool:
    if ratio < 98:
        return False
    diff = tokens_a.symmetric_difference(tokens_b)
    return len(diff) <= 1


def _is_capability_statement(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if KOREAN_CAPABILITY_PATTERN.search(compact):
        return True
    lowered = text.lower()
    return bool(re.search(r"\b(can|able to|be able to)\b", lowered))


def _is_meaningful_token(token: str) -> bool:
    if re.search(r"\d", token):
        return True
    return token.lower() in TECH_KEYWORD_SET


def _has_meaningful_key_diff(key_a: set[str], key_b: set[str]) -> bool:
    diff = {token.lower() for token in key_a.symmetric_difference(key_b)}
    if not diff:
        return False
    return any(_is_meaningful_token(token) for token in diff)


def _is_inconsistency(
    ratio: float,
    tokens_a: set[str],
    tokens_b: set[str],
    key_a: set[str],
    key_b: set[str],
    is_capability: bool,
    is_boilerplate: bool,
) -> bool:
    if ratio < 90 or ratio > 97:
        return False
    if not tokens_a or not tokens_b:
        return False
    if not key_a or not key_b:
        return False
    if is_capability or is_boilerplate:
        return False
    if not _has_meaningful_key_diff(key_a, key_b):
        return False
    common = tokens_a.intersection(tokens_b)
    return len(common) >= 3


def _severity_for_ratio(ratio: float) -> str:
    if ratio >= 98:
        return "RED"
    if ratio >= 95:
        return "YELLOW"
    return "GREEN"


def detect(
    pages: list[dict],
    language: str = "ko",
    page_profiles: list[dict] | None = None,
) -> list[Issue]:
    sentence_items: list[dict] = []

    for page in pages:
        text = page.get("text", "")
        for sentence in _split_sentences(text):
            if _is_noise_sentence(sentence["text"]):
                continue
            sentence_items.append(
                {
                    "page": page["page_number"],
                    "start": sentence["start"],
                    "end": sentence["end"],
                    "text": sentence["text"].strip(),
                }
            )

    if len(sentence_items) > MAX_SENTENCES:
        return []

    issues: list[Issue] = []
    normalized = [_normalize_sentence(item["text"]) for item in sentence_items]

    page_type_map = {}
    page_conf_map = {}
    if page_profiles:
        page_type_map = {profile["page"]: profile["type"] for profile in page_profiles}
        page_conf_map = {
            profile["page"]: profile.get("confidence") for profile in page_profiles
        }

    for i in range(len(sentence_items)):
        base = normalized[i]
        for j in range(i + 1, min(len(sentence_items), i + 1 + MAX_LOOKAHEAD)):
            ratio = fuzz.ratio(base, normalized[j])
            if ratio < SIMILARITY_THRESHOLD:
                continue
            item = sentence_items[j]
            base_item = sentence_items[i]
            severity = _severity_for_ratio(ratio)
            page_type = page_type_map.get(item["page"], "GENERIC")
            page_confidence = page_conf_map.get(item["page"])
            evidence = item["text"]
            if len(evidence) > 160:
                evidence = evidence[:157] + "..."

            subtype = None
            kind = "WARNING"
            tokens_a = _tokenize(base_item["text"])
            tokens_b = _tokenize(item["text"])
            key_a = _extract_key_tokens(base_item["text"])
            key_b = _extract_key_tokens(item["text"])
            is_capability = _is_capability_statement(
                base_item["text"]
            ) or _is_capability_statement(item["text"])
            boilerplate_score = max(
                _boilerplate_score(base_item["text"]),
                _boilerplate_score(item["text"]),
            )
            is_form = page_type == "FORM"
            is_consent_terms = (
                page_type in {"CONSENT", "TERMS"}
                and (page_confidence is None or page_confidence >= CONSENT_CONF_THRESHOLD)
            )
            is_resume = page_type == "RESUME"
            is_boilerplate = boilerplate_score >= 2
            resume_note_ko = "의도적 반복일 수 있습니다. 표현/용어 통일 여부만 점검하세요."
            resume_note_en = (
                "This may be intentional repetition. Check wording/terminology consistency."
            )
            consent_note_ko = "동일 문구 의도 여부만 점검하세요."
            consent_note_en = "Check whether identical wording is intended."
            form_note_ko = "양식/문항 반복 구조일 수 있으니, 의도 여부만 점검하세요."
            form_note_en = "This may be a repeated form/question pattern. Verify intent."

            if is_form:
                subtype = "FORM_REPEAT"
                kind = "NOTE"
                suggestion_ko = form_note_ko
                suggestion_en = form_note_en
            elif is_consent_terms:
                subtype = "BOILERPLATE_REPEAT"
                kind = "NOTE"
                suggestion_ko = consent_note_ko
                suggestion_en = consent_note_en
            elif _is_verbatim(ratio, tokens_a, tokens_b):
                subtype = "VERBATIM_DUPLICATE"
                kind = "WARNING"
                suggestion_ko = "중복 내용을 제거하거나 병합하세요."
                suggestion_en = "Remove or merge duplicated content."
            elif (
                is_resume
                and _is_inconsistency(
                    ratio,
                    tokens_a,
                    tokens_b,
                    key_a,
                    key_b,
                    is_capability,
                    is_boilerplate,
                )
            ):
                subtype = "INCONSISTENCY"
                kind = "WARNING"
                suggestion_ko = "표현/용어(기술 스택) 일관성 확인"
                suggestion_en = "Check terminology/tech stack consistency."
            elif is_resume:
                kind = "NOTE"
                suggestion_ko = resume_note_ko
                suggestion_en = resume_note_en
            else:
                suggestion_ko = "중복 내용을 제거하거나 병합하세요."
                suggestion_en = "Remove or merge duplicated content."

            if page_type == "UNCERTAIN":
                kind = "NOTE"
                subtype = None

            i18n = IssueI18n(
                ko=IssueText(
                    message=f"문장이 다른 문장과 매우 유사합니다 (유사도 {ratio:.0f}).",
                    suggestion=suggestion_ko,
                ),
                en=IssueText(
                    message=(
                        "Sentence is very similar to another sentence "
                        f"(similarity {ratio:.0f})."
                    ),
                    suggestion=suggestion_en,
                ),
            )
            selected = i18n.en if language == "en" else i18n.ko
            base_snippet = base_item["text"].strip()
            if len(base_snippet) > 80:
                base_snippet = base_snippet[:77] + "..."

            issues.append(
                Issue(
                    id=f"redundancy_similar_p{item['page']}_{item['start']}",
                    category="redundancy",
                    severity=severity,
                    message=selected.message,
                    evidence=evidence,
                    suggestion=selected.suggestion,
                    location=Location(
                        page=item["page"],
                        start_char=item["start"],
                        end_char=item["end"],
                    ),
                    confidence=min(1.0, ratio / 100.0),
                    detector="rule_based",
                    i18n=i18n,
                    similarity=ratio / 100.0,
                    matched_to=MatchedTo(
                        page=base_item["page"],
                        start_char=base_item["start"],
                        end_char=base_item["end"],
                        snippet=base_snippet,
                    ),
                    kind=kind,
                    subtype=subtype,
                    page_type=page_type,
                    page_type_confidence=page_confidence,
                )
            )

    return issues
