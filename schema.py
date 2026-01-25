"""Pydantic models for quality reports."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Location(BaseModel):
    page: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_span(self) -> "Location":
        if self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char")
        return self


class IssueText(BaseModel):
    message: str
    suggestion: str


class IssueI18n(BaseModel):
    ko: IssueText
    en: IssueText


class MatchedTo(BaseModel):
    page: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    snippet: str


class Issue(BaseModel):
    id: str
    category: Literal["spelling", "grammar", "readability", "logic", "redundancy"]
    kind: Literal["ERROR", "WARNING", "NOTE"]
    subtype: Optional[str] = None
    severity: Literal["RED", "YELLOW", "GREEN"]
    message: str
    evidence: str
    suggestion: str
    location: Location
    confidence: float = Field(ge=0.0, le=1.0)
    detector: Literal["rule_based", "llm_based"] = "rule_based"
    i18n: IssueI18n
    similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    matched_to: Optional[MatchedTo] = None
    page_type: Optional[
        Literal["CONSENT", "TERMS", "RESUME", "FORM", "REPORT", "GENERIC", "UNCERTAIN"]
    ] = None
    page_type_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class Score(BaseModel):
    name: str
    value: float
    max_value: float = 100.0


class DocumentProfile(BaseModel):
    type: Literal["CONSENT", "TERMS", "RESUME", "FORM", "REPORT", "GENERIC"]
    confidence: float = Field(ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)
    dominant_type: Literal[
        "CONSENT", "TERMS", "RESUME", "FORM", "REPORT", "GENERIC", "MIXED"
    ]


class PageProfile(BaseModel):
    page: int = Field(ge=1)
    type: Literal["CONSENT", "TERMS", "RESUME", "FORM", "REPORT", "GENERIC", "UNCERTAIN"]
    confidence: float = Field(ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)
    consent_score: Optional[int] = None
    resume_score: Optional[int] = None
    terms_score: Optional[int] = None
    form_score: Optional[int] = None


class DocumentMeta(BaseModel):
    file_name: str
    page_count: int = Field(ge=0)
    textless_pages: int = Field(ge=0)
    raw_char_count: int = Field(ge=0)
    normalized_char_count: int = Field(ge=0)
    scan_like: bool
    scan_like_ratio: float = Field(ge=0.0, le=1.0)
    scan_level: Literal["NONE", "PARTIAL", "HIGH"]
    document_profile: DocumentProfile
    page_profiles: list[PageProfile] = Field(default_factory=list)


class Report(BaseModel):
    document_meta: DocumentMeta
    score_confidence: Literal["HIGH", "MED", "LOW"]
    raw_score: int = Field(ge=0, le=100)
    overall_score: Optional[int] = Field(default=None, ge=0, le=100)
    limitations: list[str] = Field(default_factory=list)
    issues: list[Issue]

    @model_validator(mode="after")
    def _enforce_score_policy(self) -> "Report":
        if self.score_confidence == "LOW":
            self.overall_score = None
        else:
            self.overall_score = self.raw_score
        return self
