"""Streamlit entry point."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import regex as re
import streamlit as st

# --------------------------------------------------------------------------
# Hotfix for ChromaDB compatibility with Pydantic v2
# --------------------------------------------------------------------------
import os
# Set dummy environment variables to bypass chromadb validation defaults.
os.environ.setdefault("CLICKHOUSE_HOST", "localhost")
os.environ.setdefault("CLICKHOUSE_PORT", "8123")
os.environ.setdefault("CHROMA_SERVER_HOST", "localhost")
os.environ.setdefault("CHROMA_SERVER_HTTP_PORT", "8000")
os.environ.setdefault("CHROMA_SERVER_GRPC_PORT", "50051")

from documind.utils.pydantic_compat import patch_pydantic_v1_for_chromadb

patch_pydantic_v1_for_chromadb()
# --------------------------------------------------------------------------



from documind.ai.candidates import CandidateLimiter, extract_ai_candidate
from documind.ai.client import OpenAIClient
from documind.ai.redact import redact_text, truncate_text
from documind.ingest.pdf_loader import PARTIAL_SCAN_THRESHOLD
from documind.ingest.loader import load_document
from documind.llm.config import (
    get_analysis_config,
    get_available_providers,
    get_available_embedding_providers,
    get_default_actor_provider,
    get_default_critic_provider,
    get_default_embedding_provider,
    get_api_key,
    get_api_model,
)
from documind.utils.db import db_manager
from documind.utils.best_practice_manager import archive_best_practice
from documind.profile.classify import (
    classify_pages,
    classify_text,
    dominant_type_from_pages,
)
from documind.quality import pipeline as quality_pipeline
from documind.quality.detectors import (
    consistency,
    formatting,
    punctuation,
    readability,
    redundancy,
    spelling_ko,
)
from documind.quality.pipeline import run_pipeline
from documind.rag.chunking import chunk_pages
from documind.rag.qa import build_context, filter_citations
from documind.schema import DocumentMeta, Report, Issue, IssueI18n, IssueText, Location
from documind.target_optimizer import TargetOptimizer
from documind.text.normalize import normalize_pages
from documind.utils.logging import setup_logging


I18N = {
    "ko": {
        "language_label": "언어",
        "title": "DocuMind 품질 MVP",
        "tab_history": "분석 이력",
        "upload_label": "PDF 업로드",
        "upload_info": "PDF/TXT/MD/DOCX 파일을 업로드하세요.",
        "mode_label": "분석 모드",
        "menu_quality": "분석",
        "menu_optim": "타겟 최적화",
        "menu_anti": "안티테제",
        "menu_sqlite": "SQLite 탐색기",
        "menu_chroma": "ChromaDB 탐색기",
        "mode_quality": "품질 분석",
        "mode_anti": "문서 Q&A (OCR)",
        "mode_optim": "타깃 최적화",
        "optim_provider_label": "LLM 제공자",
        "actor_provider_label": "주 LLM (문서 생성)",
        "critic_provider_label": "보조 LLM (평가/채점)",
        "optim_level_label": "대상",
        "optim_result_title": "최적화 결과",
        "optim_analysis_title": "분석",
        "optim_keywords_label": "키워드",
        "optim_accept_button": "수락 (완료)",
        "optim_retry_button": "1회 재시도 (수정)",
        "optim_autorun_button": "끝까지 자동 진행 (Auto)",
        "optim_decision_title": "사용자 확인 (점수 부족)",
        "optim_decision_prompt": "현재 점수: {score}점 (목표: 90점 이상). 어떻게 할까요?",
        "anti_indexed": "문서 인덱싱 완료.",
        "anti_preview_title": "OCR / 텍스트 추출 결과 미리보기",
        "anti_question_label": "문서에 대해 질문하세요",
        "anti_answer_title": "답변",
        "anti_analysis_title": "문서 분석",
        "anti_summary_button": "핵심 요약",
        "anti_antithesis_button": "안티테제 (비판 분석)",
        "anti_revision_button": "개선된 문서 재작성",
        "anti_revision_missing": "먼저 안티테제를 생성해주세요.",
        "file_label": "파일",
        "analyze_button": "분석하기",
        "success": "분석 완료.",
        "scan_caution": "텍스트 추출량이 제한적일 수 있어 결과 해석에 주의가 필요합니다.",
        "summary_title": "요약",
        "score_label": "점수",
        "score_na": "N/A",
        "confidence_label": "신뢰도",
        "issue_count_label": "전체 이슈",
        "actionable_count_label": "조치 필요 이슈",
        "scan_like_label": "스캔 유사 여부",
        "scan_like_ratio_label": "스캔 유사 비율",
        "scan_level_label": "스캔 단계",
        "low_confidence_warning": "점수는 참고용(텍스트 부족)",
        "issues_title": "이슈",
        "diagnostics_title": "진단",
        "raw_score_label": "원점수",
        "limitations_label": "제약사항",
        "table_severity": "심각도",
        "table_category": "카테고리",
        "table_kind": "레벨",
        "table_subtype": "세부 유형",
        "table_page_type": "페이지 유형",
        "table_page": "페이지",
        "table_message": "메시지",
        "table_suggestion": "제안",
        "issue_summary_label": "왜 뜸?",
        "issue_impact_label": "영향",
        "issue_action_label": "권장 수정",
        "issue_details_label": "원문/근거 보기",
        "filter_category": "카테고리",
        "filter_severity": "심각도",
        "filter_kind": "레벨",
        "filter_include_note": "참고 포함",
        "filter_show_raw": "원문 표시",
        "filter_title": "필터",
        "filter_caption": "기본은 조치 필요(오류/경고)만 표시됩니다.",
        "severity_high_label": "높음",
        "severity_mid_label": "중간",
        "severity_low_label": "낮음",
        "severity_mapping_caption": "심각도 매핑: 높음=RED · 중간=YELLOW · 낮음=GREEN",
        "kind_error_label": "오류",
        "kind_warning_label": "경고",
        "filter_options_label": "옵션",
        "results_title": "결과",
        "results_caption": "선택한 조건에 맞는 이슈만 표시됩니다.",
        "severity_breakdown_title": "심각도별 오류 목록",
        "severity_breakdown_caption": "상세 내용은 이슈 탭에서 확인하세요.",
        "filter_all": "전체",
        "evidence_label": "근거",
        "no_issues": "조건에 맞는 이슈가 없습니다.",
        "page_count_label": "페이지 수",
        "textless_pages_label": "텍스트 없는 페이지",
        "raw_char_count_label": "추출 글자수",
        "normalized_char_count_label": "정규화 글자수",
        "char_rank_title": "글자수 랭킹",
        "char_rank_order_label": "정렬",
        "char_rank_order_asc": "오름차순",
        "char_rank_order_desc": "내림차순",
        "page_label": "페이지",
        "char_count_label": "글자수",
        "profile_title": "문서 타입",
        "profile_type_label": "문서 유형",
        "dominant_type_label": "주요 유형",
        "profile_confidence_label": "신뢰도",
        "profile_signals_label": "근거 키워드",
        "page_profiles_title": "페이지별 유형",
        "page_type_label": "페이지 유형",
        "page_type_confidence_label": "신뢰도",
        "page_signals_label": "근거 키워드",
        "consent_score_label": "동의서 점수",
        "resume_score_label": "이력서 점수",
        "matched_to_label": "매칭 문장",
        "matched_to_sentence": "비슷한 문장이 p{page}에 또 있습니다: {snippet}",
        "similarity_label": "유사도",
        "download_title": "다운로드",
        "download_help": "report.json에는 한/영 번역 메시지(ko/en), score_confidence, limitations가 포함됩니다.",
        "download_button": "report.json 다운로드",
        "error": "분석에 실패했습니다. 콘솔 로그를 확인하세요.",
        "no_report": "PDF를 업로드하고 분석을 실행하세요.",
        "note_hint": "참고 포함을 켜면 참고 이슈를 볼 수 있습니다.",
        "ai_explain_toggle": "AI로 쉽게 설명(베타)",
        "ai_review_toggle": "AI 추가 점검(놓친 이슈 후보)",
        "ai_explain_title": "AI 설명(베타)",
        "ai_why_label": "왜 문제인지",
        "ai_impact_label": "영향",
        "ai_action_label": "권장 수정",
        "ai_review_title": "AI 추가 점검(베타)",
        "ai_review_caption": "AI 후보는 참고용(NOTE)입니다.",
        "ai_review_limit_note": "AI 후보는 참고용(NOTE)이며 페이지/문서 상태에 따라 최대 {limit}개까지만 생성됩니다.",
        "ai_review_empty": "AI 후보 0건",
        "ai_explain_empty": "AI 설명 결과 0건",
        "ai_explain_error_prefix": "AI 설명 호출 실패",
        "ai_review_error_prefix": "AI 추가 점검 호출 실패",
        "ai_diag_title": "AI 합의 진단",
        "ai_diag_caption": "내부 진단 + RAG 근거 기반 GPT/Gemini 교차 검증 결과입니다.",
        "ai_diag_missing_key": "GPT/Gemini API 키가 필요합니다.",
        "ai_diag_partial_key": "{provider} 키가 없어 {fallback} 결과만 표시합니다.",
        "ai_diag_unavailable": "AI 진단 결과가 없습니다.",
        "ai_diag_consensus_notes": "합의 메모",
        "ai_diag_gpt_label": "GPT 결과",
        "ai_diag_gemini_label": "Gemini 결과",
        "ai_diag_critique_label": "상호 비판",
        "ai_toggle_locked_note": "AI 옵션 변경은 재분석이 필요합니다.",
        "download_ai_button": "report_ai.json 다운로드",
        "download_ai_help": "report_ai.json에는 AI 진단/설명/추가 점검 결과가 포함됩니다.",
        "processing_title": "분석 중...",
        "processing_subtitle": "분석이 끝날 때까지 잠시만 기다려 주세요.",
        "tab_qa": "질문(Q&A) (베타)",
        "qa_title": "질문(Q&A) (베타)",
        "qa_question_label": "질문",
        "qa_question_placeholder": "문서에 대해 질문을 입력하세요.",
        "qa_char_count": "글자수: {count}/{limit}",
        "qa_ask_button": "질문하기",
        "qa_need_analysis": "분석이 완료된 뒤 질문할 수 있습니다.",
        "qa_need_key": "AI 키가 설정되어야 질문할 수 있습니다.",
        "qa_cooldown": "질문 쿨다운: {seconds}초 후 다시 시도하세요.",
        "qa_processing_search": "근거 검색 중...",
        "qa_processing_rewrite": "질문 정제 중...",
        "qa_processing_answer": "답변 생성 중...",
        "qa_answer_title": "답변",
        "qa_citations_title": "근거",
        "qa_no_citations": "근거를 찾지 못했습니다. 질문을 더 구체화해 주세요 (예: 보유기간/동의거부 불이익…).",
        "qa_empty": "답변을 생성할 수 없습니다.",
        "unsupported_file_type": "지원하지 않는 파일 형식입니다. PDF/TXT/MD만 가능합니다.",
        "text_empty_error": "텍스트가 비어 있어 분석할 수 없습니다. 스캔 PDF일 수 있습니다.",
        "anti_pdf_only_error": "문서 Q&A(OCR)는 PDF만 지원합니다.",
        "tab_summary": "요약",
        "tab_issues": "이슈",
        "tab_diagnostics": "진단",
        "tab_download": "다운로드",
        "tab_help": "가이드/Help",
        "quick_guide_title": "Quick Guide",
        "quick_guide_body": """<ul>
<li>지원 파일: PDF/TXT/MD</li>
<li>OCR 미지원 (스캔 PDF는 텍스트가 부족할 수 있음)</li>
<li>개인정보 주의: evidence에 원문 일부가 포함됨</li>
<li>용어 요약: <code>kind</code>=레벨, <code>subtype</code>=세부 유형, <code>page_type</code>=페이지 유형</li>
<li>현 버전 한계: 맞춤법은 제한적 룰 기반, 문법/논리 진단은 미완</li>
</ul>""",
        "quick_guide_caption": "NOTE는 기본 숨김입니다. 필요 시 '참고 포함' 토글을 켜세요.",
        "help_content": """### 지원 범위\n- 업로드 파일: PDF/TXT/MD\n- OCR 미지원: 스캔 PDF는 텍스트가 적어 결과 신뢰도가 낮아질 수 있습니다.\n\n### 개인정보 주의\n- 이슈 evidence에 원문 일부가 포함됩니다. 민감 정보가 있으면 공유/보관에 주의하세요.\n\n### 결과 해석\n- document_profile: 문서 유형 추정 결과 (dominant_type 포함)\n- score_confidence: 텍스트 추출량/스캔 여부 기반 신뢰도\n- kind/subtype: NOTE/WARNING 구분 및 세부 유형(BOILERPLATE_REPEAT/INCONSISTENCY 등)\n\n### 용어 사전 (Glossary)\n- kind: 이슈 레벨 (오류/경고/참고)\n- subtype: 세부 유형 (긴 문장/정형 문구 반복/표현 불일치 등)\n- page_type: 페이지 유형 (이력/동의/약관/일반/불확실)\n\n### 현 버전 한계\n- 맞춤법은 제한적 룰 기반이며, 문법/논리 진단은 미완입니다.""",
        # Auth
        "login_title": "로그인",
        "signup_title": "회원가입",
        "username_label": "아이디",
        "password_label": "비밀번호",
        "login_button": "로그인",
        "signup_button": "가입하기",
        "logout_button": "로그아웃",
        "welcome_msg": "환영합니다, {username}님! ({role})",
        "login_success": "로그인 성공!",
        "login_failed": "로그인 실패: 아이디 또는 비밀번호를 확인하세요.",
        "signup_success": "회원가입 성공! 이제 로그인해주세요.",
        "signup_failed": "회원가입 실패: 이미 존재하는 아이디일 수 있습니다.",
        "auth_required": "앱을 사용하려면 로그인이 필요합니다.",
        "menu_analyzer": "분석기 (Analyzer)",
        "menu_sqlite": "SQLite 탐색기",
        "menu_chroma": "ChromaDB 탐색기",
        "db_explorer_title": "데이터베이스 탐색기",
        "doc_filename": "파일명",
        "doc_user": "사용자",
        "doc_date": "분석 일시",
        "rag_content": "정제된 RAG 텍스트",
    },
    "en": {
        "language_label": "Language",
        "title": "DocuMind Quality MVP",
        "upload_label": "Upload a PDF",
        "upload_info": "Upload a PDF/TXT/MD file to start analysis.",
        "mode_label": "Mode",
        "mode_quality": "Quality",
        "mode_anti": "Document Q&A (OCR)",
        "mode_optim": "Target Optimization",
        "optim_provider_label": "LLM Provider",
        "actor_provider_label": "Actor LLM (Generation)",
        "critic_provider_label": "Critic LLM (Evaluation)",
        "optim_level_label": "Target",
        "optim_result_title": "Optimized Result",
        "optim_analysis_title": "Analysis",
        "optim_keywords_label": "Keywords",
        "optim_accept_button": "Accept",
        "optim_retry_button": "Retry",
        "optim_decision_title": "Review Required",
        "optim_decision_prompt": "Score is {score}. Accept this result?",
        "anti_indexed": "Document indexed.",
        "anti_preview_title": "OCR / Extracted text preview",
        "anti_question_label": "Ask a question about the document",
        "anti_answer_title": "Answer",
        "anti_analysis_title": "Document analysis",
        "anti_summary_button": "Summary",
        "anti_antithesis_button": "Antithesis (critical analysis)",
        "anti_revision_button": "Rewritten document",
        "anti_revision_missing": "Generate antithesis first.",
        "file_label": "File",
        "analyze_button": "Analyze",
        "success": "Analysis complete.",
        "scan_caution": "Text extraction may be limited; interpret results with care.",
        "summary_title": "Summary",
        "score_label": "Score",
        "score_na": "N/A",
        "confidence_label": "Confidence",
        "issue_count_label": "Total issues",
        "actionable_count_label": "Actionable issues",
        "scan_like_label": "scan_like",
        "scan_like_ratio_label": "scan_like_ratio",
        "scan_level_label": "scan_level",
        "low_confidence_warning": "Score is for reference only (insufficient text).",
        "issues_title": "Issues",
        "diagnostics_title": "Diagnostics",
        "raw_score_label": "raw_score",
        "limitations_label": "limitations",
        "table_severity": "Severity",
        "table_category": "Category",
        "table_kind": "Kind",
        "table_subtype": "Subtype",
        "table_page_type": "Page type",
        "table_page": "Page",
        "table_message": "Message",
        "table_suggestion": "Suggestion",
        "issue_summary_label": "Why?",
        "issue_impact_label": "Impact",
        "issue_action_label": "Recommended action",
        "issue_details_label": "View evidence",
        "filter_category": "Category",
        "filter_severity": "Severity",
        "filter_kind": "Kind",
        "filter_include_note": "Include notes",
        "filter_show_raw": "Raw values",
        "filter_title": "Filters",
        "filter_caption": "By default, only actionable issues (ERROR/WARNING) are shown.",
        "severity_high_label": "High",
        "severity_mid_label": "Medium",
        "severity_low_label": "Low",
        "severity_mapping_caption": "Severity mapping: High=RED · Medium=YELLOW · Low=GREEN",
        "kind_error_label": "Error",
        "kind_warning_label": "Warning",
        "filter_options_label": "Options",
        "results_title": "Results",
        "results_caption": "Only issues matching the selected filters are shown.",
        "severity_breakdown_title": "Severity breakdown",
        "severity_breakdown_caption": "See the Issues tab for details.",
        "filter_all": "All",
        "evidence_label": "Evidence",
        "no_issues": "No issues match the current filters.",
        "page_count_label": "page_count",
        "textless_pages_label": "textless_pages",
        "raw_char_count_label": "raw_char_count",
        "normalized_char_count_label": "normalized_char_count",
        "char_rank_title": "Character count ranking",
        "char_rank_order_label": "Order",
        "char_rank_order_asc": "Ascending",
        "char_rank_order_desc": "Descending",
        "page_label": "Page",
        "char_count_label": "Characters",
        "profile_title": "Document profile",
        "profile_type_label": "Document type",
        "dominant_type_label": "dominant_type",
        "profile_confidence_label": "confidence",
        "profile_signals_label": "signals",
        "page_profiles_title": "Page profiles",
        "page_type_label": "page_type",
        "page_type_confidence_label": "page_type_confidence",
        "page_signals_label": "signals",
        "consent_score_label": "CONSENT score",
        "resume_score_label": "RESUME score",
        "matched_to_label": "Matched sentence",
        "matched_to_sentence": "A similar sentence appears on p{page}: {snippet}",
        "similarity_label": "Similarity",
        "download_title": "Download",
        "download_help": "report.json includes translated messages (ko/en), score_confidence, and limitations.",
        "download_button": "Download report.json",
        "error": "Analysis failed. Check the console logs for details.",
        "no_report": "Upload a PDF and run analysis.",
        "note_hint": "Enable 'Include notes' to view note issues.",
        "ai_explain_toggle": "AI-friendly explanation (beta)",
        "ai_review_toggle": "AI extra review (missed candidates)",
        "ai_explain_title": "AI explanation (beta)",
        "ai_why_label": "Why it matters",
        "ai_impact_label": "Impact",
        "ai_action_label": "Recommended action",
        "ai_review_title": "AI extra review (beta)",
        "ai_review_caption": "AI candidates are informational (NOTE).",
        "ai_review_limit_note": "AI candidates are NOTE-only; up to {limit} may be generated depending on the document state.",
        "ai_review_empty": "AI candidates: 0",
        "ai_explain_empty": "AI explanations: 0",
        "ai_explain_error_prefix": "AI explanation failed",
        "ai_review_error_prefix": "AI review failed",
        "ai_diag_title": "AI consensus diagnosis",
        "ai_diag_caption": "Cross-checked GPT/Gemini diagnosis using internal checks and RAG evidence.",
        "ai_diag_missing_key": "GPT/Gemini API keys are required.",
        "ai_diag_partial_key": "{provider} key missing; showing {fallback} only.",
        "ai_diag_unavailable": "AI diagnosis result is unavailable.",
        "ai_diag_consensus_notes": "Consensus notes",
        "ai_diag_gpt_label": "GPT result",
        "ai_diag_gemini_label": "Gemini result",
        "ai_diag_critique_label": "Cross critiques",
        "ai_toggle_locked_note": "Changing AI options requires re-analysis.",
        "download_ai_button": "Download report_ai.json",
        "download_ai_help": "report_ai.json includes AI diagnosis, explanations, and extra review results.",
        "processing_title": "Analyzing...",
        "processing_subtitle": "Please wait until analysis completes.",
        "tab_qa": "Q&A (beta)",
        "qa_title": "Q&A (beta)",
        "qa_question_label": "Question",
        "qa_question_placeholder": "Ask a question about the document.",
        "qa_char_count": "Characters: {count}/{limit}",
        "qa_ask_button": "Ask",
        "qa_need_analysis": "Run analysis before asking a question.",
        "qa_need_key": "Set an API key to enable Q&A.",
        "qa_cooldown": "Q&A cooldown: try again in {seconds}s.",
        "qa_processing_search": "Retrieving evidence...",
        "qa_processing_rewrite": "Refining question...",
        "qa_processing_answer": "Generating answer...",
        "qa_answer_title": "Answer",
        "qa_citations_title": "Citations",
        "qa_no_citations": "No citations found. Please make the question more specific (e.g., retention period, refusal impact...).",
        "qa_empty": "Unable to generate an answer.",
        "unsupported_file_type": "Unsupported file type. Only PDF/TXT/MD are supported.",
        "text_empty_error": "Text is empty, so analysis cannot proceed. The file may be a scanned PDF.",
        "anti_pdf_only_error": "Document Q&A (OCR) supports PDF only.",
        "tab_summary": "Summary",
        "tab_issues": "Issues",
        "tab_diagnostics": "Diagnostics",
        "tab_download": "Download",
        "tab_history": "History",
        "tab_help": "Guide/Help",
        "quick_guide_title": "Quick Guide",
        "quick_guide_body": """<ul>
<li>Supported files: PDF/TXT/MD/DOCX</li>
<li>OCR not supported (scanned PDFs may have little text)</li>
<li>Privacy caution: evidence includes snippets of original text</li>
<li>Terms: <code>kind</code>=level, <code>subtype</code>=detail, <code>page_type</code>=page category</li>
<li>Current limits: spelling is limited rule-based; grammar/logic checks are not implemented</li>
</ul>""",
        "quick_guide_caption": "NOTE is hidden by default. Enable 'Include NOTE' if needed.",
        "help_content": """### Scope\n- Supported file: PDF/TXT/MD/DOCX\n- OCR not supported: scanned PDFs may reduce confidence.\n\n### Privacy\n- Evidence includes original text snippets. Handle sensitive data carefully.\n\n### How to read results\n- document_profile: estimated document type (includes dominant_type)\n- score_confidence: confidence based on text extraction/scan likelihood\n- kind/subtype: NOTE/WARNING and detailed types (BOILERPLATE_REPEAT/INCONSISTENCY)\n\n### Glossary\n- kind: issue level (ERROR/WARNING/NOTE)\n- subtype: detailed type (LONG_SENTENCE/BOILERPLATE_REPEAT/INCONSISTENCY)\n- page_type: page category (RESUME/CONSENT/TERMS/GENERIC/UNCERTAIN)\n\n### Current limitations\n- Spelling is limited rule-based; grammar/logic checks are not implemented.""",
        # Auth
        "login_title": "Login",
        "signup_title": "Sign Up",
        "username_label": "Username",
        "password_label": "Password",
        "login_button": "Login",
        "signup_button": "Sign Up",
        "logout_button": "Logout",
        "welcome_msg": "Welcome, {username}! ({role})",
        "login_success": "Login successful!",
        "login_failed": "Login failed: Check username or password.",
        "signup_success": "Registration successful! Please login.",
        "signup_failed": "Registration failed: Username may already exist.",
        "auth_required": "Authentication required.",
        # DB Explorer
        "menu_analyzer": "Analyzer",
        "menu_sqlite": "SQLite Explorer",
        "menu_chroma": "ChromaDB Explorer",
        "db_explorer_title": "Database Explorer",
        "doc_filename": "Filename",
        "doc_user": "User",
        "doc_date": "Analysis Date",
        "rag_content": "Refined RAG Content",
    },
}

SUPPORTED_FILE_TYPES = ["pdf", "txt", "md", "docx"]
TEXT_FILE_TYPES = {"txt", "md"}
SCAN_LIKE_THRESHOLD = 0.6
MIN_TEXT_LEN = 50
RAG_COLLECTION_NAME = "documind_rag"
RAG_CHUNK_SIZE = 900
RAG_CHUNK_OVERLAP = 120
RAG_EMBED_BATCH_SIZE = 32
RAG_QUERY_LIMIT = 3
RAG_QUERY_MAX_CHARS = 120
AI_INTERNAL_MAX_ISSUES = 24
AI_DIAG_MAX_ISSUES = 12
AI_DIAG_MAX_CONCERNS = 6

LANG_LABELS = {
    "ko": "한국어",
    "en": "English",
}

LABEL_MAP = {
    "kind": {
        "ko": {
            "ERROR": "오류(조치 필요)",
            "WARNING": "경고(검토 필요)",
            "NOTE": "참고",
        }
    },
    "subtype": {
        "ko": {
            "LONG_SENTENCE": "긴 문장",
            "BOILERPLATE_REPEAT": "정형 문구 반복",
            "VERBATIM_DUPLICATE": "완전 중복",
            "INCONSISTENCY": "표현/용어 불일치",
            "FORM_REPEAT": "양식 반복",
            "COMMON_KO_TYPO": "맞춤법/띄어쓰기",
            "SPACING_SUSPECT": "띄어쓰기 의심",
            "BRACKET_MISMATCH": "문장부호(괄호 짝 불일치)",
            "AI_SPELLING": "AI 맞춤법",
            "AI_GRAMMAR": "AI 문법",
            "AI_READABILITY": "AI 가독성",
            "AI_LOGIC": "AI 논리",
            "AI_REDUNDANCY": "AI 중복",
            "PUNCTUATION_ANOMALY": "구두점 이상",
            "BULLET_FLOW_BREAK": "번호 흐름 이상",
            "DATE_FORMAT_INCONSISTENCY": "날짜 표기 불일치",
            "NUMBER_FORMAT_INCONSISTENCY": "숫자 표기 불일치",
        },
        "en": {
            "AI_SPELLING": "AI Spelling",
            "AI_GRAMMAR": "AI Grammar",
            "AI_READABILITY": "AI Readability",
            "AI_LOGIC": "AI Logic",
            "AI_REDUNDANCY": "AI Redundancy",
        },
    },
    "page_type": {
        "ko": {
            "RESUME": "이력/자기소개",
            "CONSENT": "동의/고지",
            "TERMS": "약관",
            "FORM": "설문/점검지",
            "REPORT": "보고서/브리프",
            "GENERIC": "일반",
            "UNCERTAIN": "불확실",
            "MIXED": "혼합",
        }
    },
}

SHORT_LABEL_MAP = {
    "kind": {
        "ko": {
            "ERROR": "오류",
            "WARNING": "경고",
            "NOTE": "참고",
        }
    },
    "subtype": {
        "ko": {
            "LONG_SENTENCE": "긴 문장",
            "BOILERPLATE_REPEAT": "정형 문구",
            "VERBATIM_DUPLICATE": "완전 중복",
            "INCONSISTENCY": "표현 불일치",
            "FORM_REPEAT": "양식 반복",
            "COMMON_KO_TYPO": "맞춤법",
            "SPACING_SUSPECT": "띄어쓰기",
            "BRACKET_MISMATCH": "괄호 짝 불일치",
            "AI_SPELLING": "AI 맞춤법",
            "AI_GRAMMAR": "AI 문법",
            "AI_READABILITY": "AI 가독성",
            "AI_LOGIC": "AI 논리",
            "AI_REDUNDANCY": "AI 중복",
            "PUNCTUATION_ANOMALY": "구두점",
            "BULLET_FLOW_BREAK": "번호",
            "DATE_FORMAT_INCONSISTENCY": "날짜",
            "NUMBER_FORMAT_INCONSISTENCY": "숫자",
        },
        "en": {
            "AI_SPELLING": "AI Spell",
            "AI_GRAMMAR": "AI Gram",
            "AI_READABILITY": "AI Read",
            "AI_LOGIC": "AI Logic",
            "AI_REDUNDANCY": "AI Redun",
        },
    },
    "page_type": {
        "ko": {
            "RESUME": "이력",
            "CONSENT": "동의",
            "TERMS": "약관",
            "FORM": "설문",
            "REPORT": "보고서",
            "GENERIC": "일반",
            "UNCERTAIN": "불확실",
            "MIXED": "혼합",
        }
    },
}

CATEGORY_LABELS = {
    "ko": {
        "spelling": "맞춤법",
        "grammar": "문법",
        "readability": "가독성",
        "logic": "논리",
        "redundancy": "중복",
    },
    "en": {
        "spelling": "Spelling",
        "grammar": "Grammar",
        "readability": "Readability",
        "logic": "Logic",
        "redundancy": "Redundancy",
    },
}

ISSUE_SUMMARY = {
    "ko": {
        "INCONSISTENCY": "같은 의미를 다른 용어로 작성해 일관성이 흔들릴 수 있어요.",
        "LONG_SENTENCE": "문장이 길어 핵심이 한 번에 읽히지 않을 수 있어요.",
        "BOILERPLATE_REPEAT": "정형 문구가 반복되는 구간이에요. 의도된 반복인지 확인하세요.",
        "VERBATIM_DUPLICATE": "같은 문장이 반복됩니다. 필요하면 삭제/통합하세요.",
        "FORM_REPEAT": "양식/문항 구조가 반복되는 구간이에요. 의도된 반복인지 확인하세요.",
        "COMMON_KO_TYPO": "흔한 맞춤법/띄어쓰기 오류가 있는지 확인하세요.",
        "SPACING_SUSPECT": "띄어쓰기 오류가 있는지 확인하세요.",
        "BRACKET_MISMATCH": "괄호/따옴표 짝이 맞지 않는 부분이 있을 수 있어요.",
        "PUNCTUATION_ANOMALY": "구두점이 과도하게 반복된 구간이 있을 수 있어요.",
        "BULLET_FLOW_BREAK": "번호/문항 흐름이 끊긴 것처럼 보일 수 있어요.",
        "DATE_FORMAT_INCONSISTENCY": "날짜 표기 형식이 섞여 있을 수 있어요.",
        "NUMBER_FORMAT_INCONSISTENCY": "숫자 표기 형식이 섞여 있을 수 있어요.",
        "redundancy": "비슷한 문장이 반복될 수 있어요.",
        "readability": "문장 구조가 길거나 복잡해 읽기 어려울 수 있어요.",
        "default": "검토가 필요한 부분이 있습니다.",
    },
    "en": {
        "INCONSISTENCY": "Different terms may be used for the same meaning.",
        "LONG_SENTENCE": "The sentence is long and may reduce readability.",
        "BOILERPLATE_REPEAT": "Boilerplate text appears to repeat; verify if intended.",
        "VERBATIM_DUPLICATE": "The same sentence repeats; remove or merge if needed.",
        "FORM_REPEAT": "Form/question structure may repeat; verify if intended.",
        "COMMON_KO_TYPO": "Common Korean spelling/spacing issues may be present.",
        "SPACING_SUSPECT": "Potential spacing issues detected.",
        "BRACKET_MISMATCH": "Bracket/quote pairs may be unbalanced.",
        "PUNCTUATION_ANOMALY": "Punctuation repeats unusually.",
        "BULLET_FLOW_BREAK": "Numbered list flow may be broken.",
        "DATE_FORMAT_INCONSISTENCY": "Date formats may be inconsistent.",
        "NUMBER_FORMAT_INCONSISTENCY": "Number formats may be inconsistent.",
        "redundancy": "Similar sentences may be repeating.",
        "readability": "Sentence structure may be long or complex.",
        "default": "This area may need review.",
    },
}

IMPACT_BY_KIND = {
    "ko": {
        "ERROR": "수정하지 않으면 문서 신뢰도에 큰 영향을 줄 수 있어요.",
        "WARNING": "검토 후 용어/문장 구조를 정리하면 더 좋아져요.",
        "NOTE": "참고용입니다. 문맥상 정상일 수도 있어요.",
    },
    "en": {
        "ERROR": "If left as-is, it can significantly affect document credibility.",
        "WARNING": "A quick review can improve clarity and consistency.",
        "NOTE": "For reference only; it may be acceptable in context.",
    },
}

SEVERITY_LABELS = {
    "ko": {
        "RED": "높음",
        "YELLOW": "중간",
        "GREEN": "낮음",
    },
    "en": {
        "RED": "High",
        "YELLOW": "Medium",
        "GREEN": "Low",
    },
}
SEVERITY_ICONS = {
    "RED": "🔴",
    "YELLOW": "🟡",
    "GREEN": "🟢",
}

CUSTOM_CSS = """
<style>
.block-container {
  max-width: 1200px;
  margin: 0 auto;
  padding-top: 1.5rem;
  padding-bottom: 2rem;
}
[data-testid="stToolbar"], #MainMenu, footer {
  visibility: hidden;
  height: 0;
}
[data-testid="stFileUploader"] {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 14px;
  padding: 1rem;
}
[data-testid="stMetric"] {
  background: #111827 !important;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 14px;
  padding: 0.75rem;
  min-height: 110px;
  width: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {
  color: #e5e7eb !important;
}
[data-testid="stCheckbox"] label,
[data-testid="stToggle"] label {
  font-size: 0.85rem;
}
[data-testid="stCheckbox"],
[data-testid="stToggle"] {
  margin-bottom: 0.55rem;
}
#ai-panel-marker {
  display: block;
  height: 0;
  line-height: 0;
}
div[data-testid="stMarkdownContainer"]:has(#ai-panel-marker),
div.element-container:has(#ai-panel-marker) {
  margin: 0;
  padding: 0;
  height: 0;
}
/* === AI panel row: center the row contents === */
[data-testid="stHorizontalBlock"]:has(#ai-panel-marker) {
  align-items: center !important;
}

/* Keep right column big button styling but retarget selector (remove > dependency) */
[data-testid="stHorizontalBlock"]:has(#ai-panel-marker)
  [data-testid="stColumn"]:last-child {
  display: flex !important;
  align-items: stretch !important;
  padding: 0.5rem !important;
}
[data-testid="stHorizontalBlock"]:has(#ai-panel-marker)
  [data-testid="stColumn"]:last-child [data-testid="stElementContainer"] {
  height: 100% !important;
  display: flex !important;
  align-items: stretch !important;
}
[data-testid="stHorizontalBlock"]:has(#ai-panel-marker)
  [data-testid="stColumn"]:last-child [data-testid="stButton"] {
  flex: 1 1 auto !important;
  height: 100% !important;
  display: flex !important;
}
[data-testid="stHorizontalBlock"]:has(#ai-panel-marker)
  [data-testid="stColumn"]:last-child [data-testid="stButton"] button {
  height: 100% !important;
  width: 100% !important;
  margin-top: 0 !important;
  min-height: 140px !important;
  padding: 1.2rem 1rem !important;
  font-size: 1.05rem !important;
  box-sizing: border-box !important;
}
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] section {
  width: 100%;
  max-width: none;
}
[data-baseweb="segmented-control"] {
  justify-content: center;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 999px;
  padding: 2px;
}
[data-baseweb="segmented-control"] button {
  border-radius: 999px;
  color: #94a3b8;
}
[data-baseweb="segmented-control"] button[aria-pressed="true"] {
  background: rgba(59, 130, 246, 0.25);
  border: 1px solid rgba(59, 130, 246, 0.45);
  color: #e5e7eb;
}
.vertical-divider {
  height: 100%;
  min-height: 64px;
  border-left: 1px solid rgba(255, 255, 255, 0.12);
  margin: 0 auto;
}
.card-spacer {
  height: 0.6rem;
}
.quick-guide {
  font-size: 0.95rem;
  line-height: 1.6;
}
.quick-guide ul {
  margin: 0 0 0.5rem 1.1rem;
}
.processing-overlay {
  position: fixed;
  inset: 0;
  background: rgba(2, 6, 23, 0.6);
  z-index: 10000;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: all;
  cursor: wait;
}
.processing-modal {
  background: #0f172a;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 16px;
  padding: 1.4rem 2rem;
  text-align: center;
  min-width: 260px;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
}
.processing-spinner {
  width: 34px;
  height: 34px;
  border: 3px solid rgba(229, 231, 235, 0.35);
  border-top-color: #e5e7eb;
  border-radius: 50%;
  margin: 0 auto 0.6rem;
  animation: processing-spin 0.9s linear infinite;
}
.rag-processing {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.6rem 0.9rem;
  border-radius: 10px;
  background: rgba(59, 130, 246, 0.12);
  color: #e5e7eb;
  font-size: 0.9rem;
}
.rag-processing .processing-spinner {
  width: 18px;
  height: 18px;
  border-width: 2px;
  margin: 0;
}
.processing-title {
  font-size: 1.05rem;
  font-weight: 600;
  color: #e5e7eb;
}
#qa-panel-marker {
  display: block;
  height: 0;
  line-height: 0;
}
[data-testid="stVerticalBlock"]:has(#qa-panel-marker) [data-testid="stTextArea"] textarea {
  width: 100% !important;
  min-width: 100% !important;
  max-width: none !important;
}
[data-testid="stVerticalBlock"]:has(#qa-panel-marker) [data-testid="stTextArea"] {
  width: 100% !important;
  max-width: none !important;
}
[data-testid="stVerticalBlock"]:has(#qa-panel-marker)
  [data-testid="stElementContainer"]:has([data-testid="stTextArea"]) {
  width: 100% !important;
  max-width: none !important;
}
[data-testid="stVerticalBlock"]:has(#qa-panel-marker) [data-testid="stButton"] button {
  height: auto !important;
  min-height: 44px !important;
  padding: 0.6rem 1rem !important;
}
.processing-subtitle {
  font-size: 0.9rem;
  color: rgba(229, 231, 235, 0.75);
  margin-top: 0.35rem;
}
.char-rank-title {
  font-size: 1.55rem;
  font-weight: 700;
  margin: 0.2rem 0 0.4rem 0;
}
@keyframes processing-spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
"""

logger = setup_logging()
logger.info(
    "Streamlit config path: %s",
    (Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"),
)

st.set_page_config(
    page_title="DocuMind Quality MVP",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "lang" not in st.session_state:
    st.session_state["lang"] = "ko"
if "auth_status" not in st.session_state:
    st.session_state["auth_status"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None
if "report" not in st.session_state:
    st.session_state["report"] = None
if "page_char_counts" not in st.session_state:
    st.session_state["page_char_counts"] = None
if "file_info" not in st.session_state:
    st.session_state["file_info"] = None
if "file_hash" not in st.session_state:
    st.session_state["file_hash"] = None
if "is_running" not in st.session_state:
    st.session_state["is_running"] = False
if "normalized_pages" not in st.session_state:
    st.session_state["normalized_pages"] = None
if "ai_explanations" not in st.session_state:
    st.session_state["ai_explanations"] = None
if "ai_candidates" not in st.session_state:
    st.session_state["ai_candidates"] = None
if "ai_status" not in st.session_state:
    st.session_state["ai_status"] = {"explain": None, "review": None}
if "ai_errors" not in st.session_state:
    st.session_state["ai_errors"] = {"explain": None, "review": None}
if "ai_cache" not in st.session_state:
    st.session_state["ai_cache"] = {}
if "ai_call_count" not in st.session_state:
    st.session_state["ai_call_count"] = {"explain": 0, "review": 0}
if "last_ai_run_ts" not in st.session_state:
    st.session_state["last_ai_run_ts"] = 0.0
if "ai_diag_cache" not in st.session_state:
    st.session_state["ai_diag_cache"] = {}
if "ai_diag_result" not in st.session_state:
    st.session_state["ai_diag_result"] = None
if "ai_diag_status" not in st.session_state:
    st.session_state["ai_diag_status"] = None
if "ai_diag_errors" not in st.session_state:
    st.session_state["ai_diag_errors"] = {"gpt": None, "gemini": None, "final": None}
if "last_ai_diag_ts" not in st.session_state:
    st.session_state["last_ai_diag_ts"] = 0.0
if "rag_index_cache" not in st.session_state:
    st.session_state["rag_index_cache"] = {}
if "rag_last_question" not in st.session_state:
    st.session_state["rag_last_question"] = ""
if "rag_last_result" not in st.session_state:
    st.session_state["rag_last_result"] = None
if "rag_status" not in st.session_state:
    st.session_state["rag_status"] = None
if "rag_error" not in st.session_state:
    st.session_state["rag_error"] = None
if "last_rag_run_ts" not in st.session_state:
    st.session_state["last_rag_run_ts"] = 0.0
if "rag_running" not in st.session_state:
    st.session_state["rag_running"] = False
if "analysis_mode" not in st.session_state:
    st.session_state["analysis_mode"] = "quality"
analysis_config = get_analysis_config()
ARCHIVE_THRESHOLD = analysis_config.get("archive_threshold", 95)
default_provider = analysis_config.get("default_provider", "Gemini CLI")
if "optim_provider" not in st.session_state:
    st.session_state["optim_provider"] = default_provider
if "optim_level" not in st.session_state:
    st.session_state["optim_level"] = "public"
    
# Initialize Settings from DB
if "actor_provider" not in st.session_state:
    saved = db_manager.get_setting("actor_provider")
    st.session_state["actor_provider"] = saved if saved else get_default_actor_provider()

if "critic_provider" not in st.session_state:
    saved = db_manager.get_setting("critic_provider")
    st.session_state["critic_provider"] = saved if saved else get_default_critic_provider()

if "embedding_provider" not in st.session_state:
    saved = db_manager.get_setting("embedding_provider")
    default_embed = saved if saved else get_default_embedding_provider()
    embed_options = get_available_embedding_providers()
    st.session_state["embedding_provider"] = (
        default_embed if default_embed in embed_options else get_default_embedding_provider()
    )
if "optim_result" not in st.session_state:
    st.session_state["optim_result"] = None
if "optim_error" not in st.session_state:
    st.session_state["optim_error"] = None
if "optim_session" not in st.session_state:
    st.session_state["optim_session"] = None
if "optim_state" not in st.session_state:
    st.session_state["optim_state"] = None
if "optim_engine" not in st.session_state:
    st.session_state["optim_engine"] = None
if "anti_docs" not in st.session_state:
    st.session_state["anti_docs"] = None
if "anti_indexed" not in st.session_state:
    st.session_state["anti_indexed"] = False
if "anti_error" not in st.session_state:
    st.session_state["anti_error"] = None
if "anti_chunks" not in st.session_state:
    st.session_state["anti_chunks"] = None
if "anti_llm" not in st.session_state:
    st.session_state["anti_llm"] = None
if "anti_retriever" not in st.session_state:
    st.session_state["anti_retriever"] = None

try:
    AI_COOLDOWN_SECONDS = max(0, int(os.getenv("AI_COOLDOWN_SECONDS", "20")))
except ValueError:
    AI_COOLDOWN_SECONDS = 10
try:
    QA_COOLDOWN_SECONDS = max(0, int(os.getenv("QA_COOLDOWN_SECONDS", "20")))
except ValueError:
    QA_COOLDOWN_SECONDS = 20

t = I18N.get(st.session_state["lang"], I18N["ko"])


def _file_ext(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower().lstrip(".")


def _is_text_file(filename: str | None) -> bool:
    return _file_ext(filename) in TEXT_FILE_TYPES


def _decode_text_bytes(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def _split_text_pages(text: str) -> list[str]:
    if "\f" in text:
        parts = [part.strip() for part in text.split("\f")]
        return [part for part in parts if part]
    return [text.strip()]


def _scan_level_for_ratio(ratio: float) -> str:
    if ratio >= SCAN_LIKE_THRESHOLD:
        return "HIGH"
    if ratio >= PARTIAL_SCAN_THRESHOLD:
        return "PARTIAL"
    return "NONE"


def _build_text_meta(pages: list[dict], file_name: str) -> dict:
    textless_pages = sum(
        1 for page in pages if len((page.get("text") or "").strip()) < MIN_TEXT_LEN
    )
    raw_char_count = sum(len(page.get("text") or "") for page in pages)
    page_count = len(pages)
    scan_like_ratio = (textless_pages / page_count) if page_count else 1.0
    scan_like = scan_like_ratio >= SCAN_LIKE_THRESHOLD
    scan_level = _scan_level_for_ratio(scan_like_ratio)
    return {
        "file_name": file_name,
        "page_count": page_count,
        "textless_pages": textless_pages,
        "raw_char_count": raw_char_count,
        "scan_like": scan_like,
        "scan_like_ratio": scan_like_ratio,
        "scan_level": scan_level,
    }


def _run_quality_text(
    file_bytes: bytes, file_name: str, language: str
) -> tuple[Report | None, list[dict], list[dict], str | None]:
    language = "en" if language == "en" else "ko"
    text = _decode_text_bytes(file_bytes)
    if not text.strip():
        return None, [], [], "empty_text"
    page_texts = _split_text_pages(text)
    if not page_texts:
        return None, [], [], "empty_text"
    pages = [
        {"page_number": idx + 1, "text": page_text}
        for idx, page_text in enumerate(page_texts)
    ]
    meta = _build_text_meta(pages, file_name)
    normalized = normalize_pages(pages)
    page_profiles = classify_pages(normalized["pages"])
    issues = readability.detect(
        normalized["pages"],
        language=language,
        page_profiles=page_profiles,
    )

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
    issues.extend(punctuation.detect(normalized["pages"], language=language))
    issues.extend(formatting.detect(normalized["pages"], language=language))
    issues.extend(consistency.detect(normalized["pages"], language=language))
    if language == "ko":
        issues.extend(spelling_ko.detect(normalized["pages"], language=language))

    issues = quality_pipeline._apply_issue_policies(
        issues, page_profiles, language, normalized["pages"]
    )
    issues = quality_pipeline._dedup_issues(issues)
    raw_score = quality_pipeline._score(issues)

    meta_payload = {
        **meta,
        "normalized_char_count": normalized["normalized_char_count"],
        "document_profile": document_profile,
        "page_profiles": page_profiles,
    }
    score_confidence = quality_pipeline._score_confidence(meta_payload["scan_level"])
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
    page_char_counts = [
        {"page": page["page_number"], "char_count": len(page["text"])}
        for page in normalized["pages"]
    ]
    return report, normalized["pages"], page_char_counts, None


def _extract_text_for_optim(file_bytes: bytes, file_name: str) -> str:
    if _is_text_file(file_name):
        text = _decode_text_bytes(file_bytes)
        page_texts = _split_text_pages(text)
        pages = [
            {"page_number": idx + 1, "text": page_text}
            for idx, page_text in enumerate(page_texts)
        ]
        normalized = normalize_pages(pages)
        return "\n\n".join(
            page["text"] for page in normalized["pages"] if page["text"].strip()
        )
    loaded = load_pdf(file_bytes, file_name)
    normalized = normalize_pages(loaded["pages"])
    return "\n\n".join(
        page["text"] for page in normalized["pages"] if page["text"].strip()
    )


def _label_for(value: str | None, group: str, lang: str) -> str:
    if value is None:
        return ""
    lang_map = LABEL_MAP.get(group, {}).get(lang)
    if lang_map and value in lang_map:
        return lang_map[value]
    if lang != "ko":
        return value
    return LABEL_MAP.get(group, {}).get("ko", {}).get(value, value)


def _format_value(value: str | None, group: str, lang: str, show_raw: bool) -> str:
    if value is None:
        return ""
    label = _label_for(value, group, lang)
    if lang == "ko" and show_raw and label != value:
        return f"{label} ({value})"
    return label


def _short_label_for(value: str | None, group: str, lang: str) -> str:
    if value is None:
        return ""
    lang_map = SHORT_LABEL_MAP.get(group, {}).get(lang)
    if lang_map and value in lang_map:
        return lang_map[value]
    if lang != "ko":
        return value
    return SHORT_LABEL_MAP.get(group, {}).get("ko", {}).get(value, value)


def _table_label(value: str | None, group: str, lang: str, show_raw: bool) -> str:
    short_label = _short_label_for(value, group, lang)
    if not value:
        return ""
    if lang == "ko" and show_raw and short_label != value:
        return f"{short_label} ({value})"
    return short_label


def _category_label(category: str, lang: str) -> str:
    return CATEGORY_LABELS.get(lang, CATEGORY_LABELS["en"]).get(category, category)


def _severity_label(severity: str, lang: str, show_raw: bool) -> str:
    label = SEVERITY_LABELS.get(lang, SEVERITY_LABELS["en"]).get(severity, severity)
    if show_raw and label != severity:
        return f"{label} ({severity})"
    return label


def _issue_detail_label(issue, lang: str) -> str:
    subtype_label = _label_for(issue.subtype, "subtype", lang)
    return subtype_label or _category_label(issue.category, lang)


def _issue_summary(issue, lang: str) -> str:
    mapping = ISSUE_SUMMARY.get(lang, ISSUE_SUMMARY["en"])
    if issue.subtype and issue.subtype in mapping:
        return mapping[issue.subtype]
    if issue.category in mapping:
        return mapping[issue.category]
    return mapping["default"]


def _issue_impact(issue, lang: str) -> str:
    return IMPACT_BY_KIND.get(lang, IMPACT_BY_KIND["en"]).get(
        issue.kind, IMPACT_BY_KIND["en"]["NOTE"]
    )


def _issue_action(issue, lang: str) -> str:
    if issue.subtype == "INCONSISTENCY":
        return (
            "표현/용어를 하나로 통일해 주세요."
            if lang == "ko"
            else "Standardize terminology/wording."
        )
    if issue.subtype == "BOILERPLATE_REPEAT":
        return (
            "표준/고지 문구 반복일 수 있으니 동일 표현 여부만 점검하세요."
            if lang == "ko"
            else "This may be boilerplate; verify whether identical wording is intended."
        )
    if issue.subtype == "VERBATIM_DUPLICATE":
        return (
            "같은 문장이 반복됩니다. 필요하면 삭제/통합하세요."
            if lang == "ko"
            else "The same sentence repeats; remove or merge if needed."
        )
    return issue.suggestion


def _multi_select_control(
    label: str,
    options: list[tuple[str, str]],
    default_values: list[str],
    key: str,
    help_text: str | None = None,
    help_map: dict[str, str] | None = None,
) -> list[str]:
    option_labels = [option_label for _, option_label in options]
    default_labels = [
        option_label
        for value, option_label in options
        if value in default_values
    ]
    if hasattr(st, "segmented_control"):
        try:
            selected = st.segmented_control(
                label,
                options=option_labels,
                default=default_labels,
                selection_mode="multi",
                key=key,
                help=help_text,
            )
            if isinstance(selected, str):
                selected = [selected]
            return [value for value, option_label in options if option_label in selected]
        except Exception:
            pass

    st.markdown(f"**{label}**")
    cols = st.columns(len(options))
    selected_values: list[str] = []
    for (value, option_label), col in zip(options, cols):
        with col:
            if st.checkbox(
                option_label,
                value=value in default_values,
                key=f"{key}_{value}",
                help=help_map.get(value) if help_map else None,
            ):
                selected_values.append(value)
    return selected_values


def _render_empty_state(message: str) -> None:
    st.info(message)


def _ai_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _is_text_noisy(text: str) -> bool:
    if not text:
        return True
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 120:
        return True
    if re.search(r"\s{6,}", text):
        return True
    density = len(compact) / max(1, len(text))
    return density < 0.55


def _ai_candidate_limit(pages: list[dict], scan_level: str) -> int:
    total_limit = 3 if scan_level in {"HIGH", "PARTIAL"} else 8
    if any(_is_text_noisy(page.get("text", "")) for page in pages):
        total_limit = min(total_limit, 3)
    return total_limit


def _ai_cache_key(file_hash: str, lang: str, explain: bool, review: bool) -> str:
    return f"{file_hash}:{lang}:explain={int(explain)}:review={int(review)}"


def _rag_cache_key(file_hash: str, lang: str, embedding_provider: str) -> str:
    return f"{file_hash}:{lang}:{embedding_provider}:rag"


def _rag_owner_key(
    user_id: str | None, file_hash: str, lang: str, embedding_provider: str
) -> str:
    owner = user_id or "anonymous"
    return f"{owner}:{file_hash}:{lang}:{embedding_provider}"


def _rag_where_filter(
    owner_key: str,
    file_hash: str,
    lang: str,
    embedding_provider: str,
    is_admin: bool,
) -> dict:
    if is_admin:
        return {
            "file_hash": file_hash,
            "lang": lang,
            "embedding_provider": embedding_provider,
        }
    return {"owner_key": owner_key}


def _get_chroma_collection():
    import chromadb

    persist_dir = str(Path(__file__).resolve().parents[3] / "chroma_raw")
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        return client.get_or_create_collection(
            name=RAG_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    except TypeError:
        return client.get_or_create_collection(name=RAG_COLLECTION_NAME)


def _chroma_owner_exists(collection, owner_key: str) -> bool:
    try:
        existing = collection.get(where={"owner_key": owner_key})
    except Exception:
        return False
    if not isinstance(existing, dict):
        return False
    ids = existing.get("ids") or []
    return len(ids) > 0


def _build_chroma_index(
    client: OpenAIClient,
    pages: list[dict],
    owner_key: str,
    file_name: str,
    file_hash: str,
    lang: str,
    embedding_provider: str,
    user_id: str | None,
) -> object | None:
    collection = _get_chroma_collection()
    if _chroma_owner_exists(collection, owner_key):
        return collection
    chunks = chunk_pages(pages, chunk_size=RAG_CHUNK_SIZE, overlap=RAG_CHUNK_OVERLAP)
    if not chunks:
        return None
    texts = [redact_text(chunk["text"]) for chunk in chunks]
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), RAG_EMBED_BATCH_SIZE):
        batch = texts[start : start + RAG_EMBED_BATCH_SIZE]
        batch_embeddings = client.embed_texts(batch)
        if not batch_embeddings:
            return None
        embeddings.extend(batch_embeddings)
    if len(embeddings) != len(chunks):
        return None
    documents: list[str] = []
    valid_embeddings: list[list[float]] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    for chunk, embedding in zip(chunks, embeddings):
        if not embedding:
            continue
        page_number = int(chunk.get("page", chunk.get("page_number", 0)))
        documents.append(redact_text(chunk.get("text", "")))
        metadatas.append(
            {
                "owner_key": owner_key,
                "user_id": user_id or "anonymous",
                "file_hash": file_hash,
                "lang": lang,
                "embedding_provider": embedding_provider,
                "source": file_name,
                "filename": file_name,
                "page": page_number,
                "chunk_id": chunk.get("chunk_id", ""),
                "start_char": int(chunk.get("start_char", 0)),
                "end_char": int(chunk.get("end_char", 0)),
            }
        )
        ids.append(f"{owner_key}:{chunk.get('chunk_id', len(ids))}")
        valid_embeddings.append(embedding)
    if not documents:
        return None
    try:
        collection.add(
            documents=documents,
            embeddings=valid_embeddings,
            metadatas=metadatas,
            ids=ids,
        )
    except Exception:
        return None
    return collection


def _dedup_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for query in queries:
        normalized = query.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _keyword_query(question: str) -> str | None:
    keywords = _extract_keywords(question)
    if not keywords:
        return None
    return " ".join(keywords[:8])


def _expand_rag_queries(
    client: OpenAIClient, question: str, language: str
) -> list[str]:
    base_queries = [question]
    keyword_query = _keyword_query(question)
    if keyword_query:
        base_queries.append(keyword_query)
    if not client.is_available():
        return _dedup_queries(base_queries)[:RAG_QUERY_LIMIT]
    lang_hint = "Korean" if language == "ko" else "English"
    prompt = (
        "Return ONLY JSON.\n"
        "Schema: {\"queries\": [\"...\"]}\n"
        f"Generate up to {max(1, RAG_QUERY_LIMIT - 1)} short search queries.\n"
        "Use keyword-style phrases, avoid full sentences.\n"
        f"Write in {lang_hint}.\n"
        f"Question: {question}"
    )
    try:
        data = client._chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=240,
        )
        if not data:
            return _dedup_queries(base_queries)[:RAG_QUERY_LIMIT]
        content = client._extract_content(data)
        parsed = client._parse_json(content)
        queries = []
        if isinstance(parsed, dict):
            raw_queries = parsed.get("queries") or []
            if isinstance(raw_queries, list):
                for query in raw_queries:
                    if not isinstance(query, str):
                        continue
                    trimmed = query.strip()
                    if not trimmed:
                        continue
                    queries.append(trimmed[:RAG_QUERY_MAX_CHARS])
        return _dedup_queries(base_queries + queries)[:RAG_QUERY_LIMIT]
    except Exception:
        return _dedup_queries(base_queries)[:RAG_QUERY_LIMIT]


def _distance_to_score(distance: float | None) -> float:
    if distance is None:
        return 0.0
    try:
        dist = float(distance)
    except (TypeError, ValueError):
        return 0.0
    if dist <= 1.0:
        return 1.0 - dist
    return 1.0 / (1.0 + dist)


def _search_chroma(
    collection,
    query_embeddings: list[list[float]],
    top_k: int,
    where_filter: dict,
) -> list[dict]:
    if not query_embeddings:
        return []
    per_query = min(max(top_k * 4, 8), 24)
    try:
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=per_query,
            where=where_filter,
            include=["documents", "metadatas", "distances", "ids"],
        )
    except Exception:
        return []
    if not isinstance(results, dict):
        return []
    ids_list = results.get("ids") or []
    docs_list = results.get("documents") or []
    metas_list = results.get("metadatas") or []
    dists_list = results.get("distances") or []
    combined: dict[str, dict] = {}
    for idx, ids in enumerate(ids_list):
        docs = docs_list[idx] if idx < len(docs_list) else []
        metas = metas_list[idx] if idx < len(metas_list) else []
        dists = dists_list[idx] if idx < len(dists_list) else []
        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            if not doc:
                continue
            doc_key = str(doc_id)
            score = _distance_to_score(dist)
            existing = combined.get(doc_key)
            if existing is None or score > existing["score"]:
                combined[doc_key] = {
                    "doc": doc,
                    "meta": meta or {},
                    "score": score,
                }
    ranked = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
    results: list[dict] = []
    for item in ranked[:top_k]:
        meta = item.get("meta") or {}
        page = int(meta.get("page") or meta.get("page_number") or 0)
        chunk_id = str(meta.get("chunk_id") or "")
        results.append(
            {
                "text": item.get("doc", ""),
                "page": page,
                "page_number": page,
                "chunk_id": chunk_id,
                "score": round(float(item.get("score", 0.0)), 4),
            }
        )
    return results


def _ai_diag_cache_key(file_hash: str, lang: str, embedding_provider: str) -> str:
    return f"{file_hash}:{lang}:{embedding_provider}:diag"


def _gpt_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _gemini_available() -> bool:
    return bool(get_api_key("gemini"))


def _parse_json_payload(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _normalize_ai_issue(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    severity = str(item.get("severity") or "").upper()
    if severity not in {"RED", "YELLOW", "GREEN"}:
        return None
    category = str(item.get("category") or "").lower()
    if category not in {"spelling", "grammar", "readability", "logic", "redundancy"}:
        category = "readability"
    try:
        page = int(item.get("page") or 0)
    except (TypeError, ValueError):
        page = 0
    message_ko = str(item.get("message_ko") or item.get("message") or "").strip()
    message_en = str(item.get("message_en") or "").strip()
    suggestion_ko = str(item.get("suggestion_ko") or "").strip()
    suggestion_en = str(item.get("suggestion_en") or "").strip()
    if not message_ko and not message_en:
        return None
    return {
        "severity": severity,
        "category": category,
        "page": page,
        "message_ko": message_ko,
        "message_en": message_en,
        "suggestion_ko": suggestion_ko,
        "suggestion_en": suggestion_en,
    }


def _normalize_ai_result(payload: dict) -> dict | None:
    if not isinstance(payload, dict):
        return None
    score_raw = payload.get("overall_score")
    try:
        score = int(float(score_raw))
    except (TypeError, ValueError):
        score = None
    if score is None or score < 0 or score > 100:
        score = None
    summary_ko = str(payload.get("summary_ko") or "").strip()
    summary_en = str(payload.get("summary_en") or "").strip()
    diagnostics_ko = str(payload.get("diagnostics_ko") or "").strip()
    diagnostics_en = str(payload.get("diagnostics_en") or "").strip()
    consensus_ko = str(payload.get("consensus_notes_ko") or "").strip()
    consensus_en = str(payload.get("consensus_notes_en") or "").strip()
    issues: list[dict] = []
    for item in payload.get("issues") or []:
        normalized = _normalize_ai_issue(item)
        if normalized:
            issues.append(normalized)
        if len(issues) >= AI_DIAG_MAX_ISSUES:
            break
    if not issues:
        issues = []
    return {
        "overall_score": score,
        "summary_ko": summary_ko,
        "summary_en": summary_en,
        "diagnostics_ko": diagnostics_ko,
        "diagnostics_en": diagnostics_en,
        "issues": issues,
        "consensus_notes_ko": consensus_ko,
        "consensus_notes_en": consensus_en,
    }


def _convert_ai_issues(ai_issues: list[dict], language: str) -> list[Issue]:
    results: list[Issue] = []
    severity_to_kind = {"RED": "ERROR", "YELLOW": "WARNING", "GREEN": "NOTE"}
    for idx, item in enumerate(ai_issues):
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity") or "").upper()
        if severity not in severity_to_kind:
            continue
        category = str(item.get("category") or "readability")
        if category not in {"spelling", "grammar", "readability", "logic", "redundancy"}:
            category = "readability"
        page = item.get("page") or 1
        try:
            page = max(1, int(page))
        except (TypeError, ValueError):
            page = 1
        message_ko = str(item.get("message_ko") or "").strip()
        message_en = str(item.get("message_en") or "").strip()
        suggestion_ko = str(item.get("suggestion_ko") or "").strip()
        suggestion_en = str(item.get("suggestion_en") or "").strip()
        message = message_en if language == "en" else message_ko
        suggestion = suggestion_en if language == "en" else suggestion_ko
        if not message:
            message = message_en or message_ko or "AI diagnosis"
        if not suggestion:
            suggestion = suggestion_en or suggestion_ko or ""
        i18n = IssueI18n(
            ko=IssueText(message=message_ko or message, suggestion=suggestion_ko or ""),
            en=IssueText(message=message_en or message, suggestion=suggestion_en or ""),
        )
        evidence = message
        results.append(
            Issue(
                id=f"ai_final_{idx}_p{page}",
                category=category,
                kind=severity_to_kind[severity],
                subtype=None,
                severity=severity,
                message=message,
                evidence=evidence,
                suggestion=suggestion,
                location=Location(page=page, start_char=0, end_char=max(1, len(evidence))),
                confidence=0.7,
                detector="llm_based",
                i18n=i18n,
            )
        )
        if len(results) >= AI_DIAG_MAX_ISSUES:
            break
    return results


def _build_internal_diagnosis_payload(report: Report, language: str) -> dict:
    severity_rank = {"RED": 3, "YELLOW": 2, "GREEN": 1}
    issues = sorted(
        report.issues,
        key=lambda item: (severity_rank.get(item.severity, 0), item.location.page),
        reverse=True,
    )
    items: list[dict] = []
    for issue in issues[:AI_INTERNAL_MAX_ISSUES]:
        text = issue.i18n.ko if language == "ko" else issue.i18n.en
        items.append(
            {
                "severity": issue.severity,
                "category": issue.category,
                "page": issue.location.page,
                "message": redact_text(text.message),
                "suggestion": redact_text(text.suggestion),
                "evidence": redact_text(issue.evidence),
            }
        )
    meta = report.document_meta
    return {
        "file_name": meta.file_name,
        "page_count": meta.page_count,
        "scan_level": meta.scan_level,
        "scan_like": meta.scan_like,
        "document_profile": {
            "type": meta.document_profile.type,
            "dominant_type": meta.document_profile.dominant_type,
            "confidence": meta.document_profile.confidence,
        },
        "internal_score": report.raw_score,
        "issues": items,
    }


def _build_issue_queries(issues: list, language: str) -> list[str]:
    severity_rank = {"RED": 3, "YELLOW": 2, "GREEN": 1}
    sorted_issues = sorted(
        issues,
        key=lambda item: (severity_rank.get(item.severity, 0), item.location.page),
        reverse=True,
    )
    queries: list[str] = []
    for issue in sorted_issues[:RAG_QUERY_LIMIT]:
        text = issue.i18n.ko if language == "ko" else issue.i18n.en
        message = text.message.strip()
        suggestion = text.suggestion.strip()
        if not message:
            continue
        query = f"{message} {suggestion}".strip()
        queries.append(query[:RAG_QUERY_MAX_CHARS])
    return _dedup_queries(queries)


def _build_rag_context_for_diagnosis(
    client: OpenAIClient,
    pages: list[dict],
    report: Report,
    file_name: str,
    file_hash: str,
    language: str,
    embedding_provider: str,
    user_id: str | None,
    owner_key: str,
) -> str:
    if not pages:
        return ""
    collection = _build_chroma_index(
        client,
        pages,
        owner_key,
        file_name,
        file_hash,
        language,
        embedding_provider,
        user_id,
    )
    if collection is None:
        return ""
    queries = _build_issue_queries(report.issues, language)
    if not queries:
        return ""
    embeddings = client.embed_texts(queries)
    if not embeddings:
        return ""
    where_filter = _rag_where_filter(owner_key, file_hash, language, embedding_provider, False)
    chunks = _search_chroma(collection, embeddings, top_k=6, where_filter=where_filter)
    if not chunks:
        return ""
    return build_context(chunks)


def _build_ai_diag_prompt(
    internal_payload: dict, rag_context: str, language: str
) -> str:
    lang_hint = "Korean" if language == "ko" else "English"
    internal_json = json.dumps(internal_payload, ensure_ascii=False)
    prompt = (
        "Return ONLY JSON.\n"
        "Schema: {\n"
        "  \"overall_score\": 0-100,\n"
        "  \"summary_ko\": \"...\",\n"
        "  \"summary_en\": \"...\",\n"
        "  \"diagnostics_ko\": \"...\",\n"
        "  \"diagnostics_en\": \"...\",\n"
        "  \"issues\": [\n"
        "    {\n"
        "      \"severity\": \"RED|YELLOW|GREEN\",\n"
        "      \"category\": \"spelling|grammar|readability|logic|redundancy\",\n"
        "      \"page\": 1,\n"
        "      \"message_ko\": \"...\",\n"
        "      \"message_en\": \"...\",\n"
        "      \"suggestion_ko\": \"...\",\n"
        "      \"suggestion_en\": \"...\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        f"Rules: limit issues to {AI_DIAG_MAX_ISSUES}. Keep messages short and practical.\n"
        "Use internal diagnostics as hints and verify with context. If evidence is weak, lower severity.\n"
        f"Write the main narrative in {lang_hint} while filling both ko/en fields.\n"
        f"Internal diagnostics JSON:\n{internal_json}\n"
    )
    if rag_context:
        prompt += f"\nRAG context:\n{rag_context}\n"
    return prompt


def _build_ai_critique_prompt(
    self_payload: dict, other_payload: dict
) -> str:
    self_json = json.dumps(self_payload, ensure_ascii=False)
    other_json = json.dumps(other_payload, ensure_ascii=False)
    return (
        "Return ONLY JSON.\n"
        "Schema: {\"concerns\": [\"...\"], \"missing_checks\": [\"...\"], \"overstatements\": [\"...\"]}\n"
        f"Limit each list to {AI_DIAG_MAX_CONCERNS} short bullets.\n"
        "Identify disagreements, missing checks, or overstatements in the other result.\n"
        f"Your result:\n{self_json}\n"
        f"Other result:\n{other_json}\n"
    )


def _call_gemini_text(prompt: str) -> tuple[str | None, str | None]:
    api_key = get_api_key("gemini")
    model = get_api_model("gemini") or "gemini-2.5-pro"
    if not api_key:
        return None, "missing_key"
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1200},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=40) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        candidates = parsed.get("candidates") or []
        if not candidates:
            return None, "empty_response"
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            return None, "empty_response"
        text = str(parts[0].get("text") or "").strip()
        return text, None
    except urllib.error.HTTPError as exc:
        return None, f"http_error_{exc.code}"
    except urllib.error.URLError:
        return None, "url_error"
    except json.JSONDecodeError:
        return None, "invalid_json"
    except Exception as exc:
        return None, f"request_failed_{exc.__class__.__name__}"


def _run_gpt_diagnosis(prompt: str) -> tuple[dict | None, str | None]:
    client = OpenAIClient()
    data = client._chat([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=1200)
    if not data:
        return None, client.last_error or "empty_response"
    content = client._extract_content(data)
    if not content:
        return None, "empty_response"
    parsed = _parse_json_payload(content)
    normalized = _normalize_ai_result(parsed or {})
    return (normalized, None) if normalized else (None, "invalid_json")


def _run_gemini_diagnosis(prompt: str) -> tuple[dict | None, str | None]:
    content, error = _call_gemini_text(prompt)
    if not content:
        return None, error or "empty_response"
    parsed = _parse_json_payload(content)
    normalized = _normalize_ai_result(parsed or {})
    return (normalized, None) if normalized else (None, "invalid_json")


def _run_gpt_critique(self_payload: dict, other_payload: dict) -> tuple[dict | None, str | None]:
    prompt = _build_ai_critique_prompt(self_payload, other_payload)
    client = OpenAIClient()
    data = client._chat([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=600)
    if not data:
        return None, client.last_error or "empty_response"
    content = client._extract_content(data)
    if not content:
        return None, "empty_response"
    parsed = _parse_json_payload(content)
    return (parsed, None) if parsed else (None, "invalid_json")


def _run_gemini_critique(self_payload: dict, other_payload: dict) -> tuple[dict | None, str | None]:
    prompt = _build_ai_critique_prompt(self_payload, other_payload)
    content, error = _call_gemini_text(prompt)
    if not content:
        return None, error or "empty_response"
    parsed = _parse_json_payload(content)
    return (parsed, None) if parsed else (None, "invalid_json")


def _build_ai_final_prompt(
    internal_payload: dict,
    rag_context: str,
    gpt_payload: dict | None,
    gemini_payload: dict | None,
    gpt_critique: dict | None,
    gemini_critique: dict | None,
    average_score: int | None,
    language: str,
) -> str:
    lang_hint = "Korean" if language == "ko" else "English"
    internal_json = json.dumps(internal_payload, ensure_ascii=False)
    gpt_json = json.dumps(gpt_payload or {}, ensure_ascii=False)
    gemini_json = json.dumps(gemini_payload or {}, ensure_ascii=False)
    gpt_crit = json.dumps(gpt_critique or {}, ensure_ascii=False)
    gemini_crit = json.dumps(gemini_critique or {}, ensure_ascii=False)
    prompt = (
        "Return ONLY JSON.\n"
        "Schema: {\n"
        "  \"overall_score\": 0-100,\n"
        "  \"summary_ko\": \"...\",\n"
        "  \"summary_en\": \"...\",\n"
        "  \"diagnostics_ko\": \"...\",\n"
        "  \"diagnostics_en\": \"...\",\n"
        "  \"issues\": [\n"
        "    {\n"
        "      \"severity\": \"RED|YELLOW|GREEN\",\n"
        "      \"category\": \"spelling|grammar|readability|logic|redundancy\",\n"
        "      \"page\": 1,\n"
        "      \"message_ko\": \"...\",\n"
        "      \"message_en\": \"...\",\n"
        "      \"suggestion_ko\": \"...\",\n"
        "      \"suggestion_en\": \"...\"\n"
        "    }\n"
        "  ],\n"
        "  \"consensus_notes_ko\": \"...\",\n"
        "  \"consensus_notes_en\": \"...\"\n"
        "}\n"
        f"Rules: limit issues to {AI_DIAG_MAX_ISSUES}. Use average_score if provided.\n"
        f"Write the main narrative in {lang_hint} while filling both ko/en fields.\n"
        f"Average score: {average_score}\n"
        f"Internal diagnostics JSON:\n{internal_json}\n"
        f"GPT JSON:\n{gpt_json}\n"
        f"Gemini JSON:\n{gemini_json}\n"
        f"GPT critique:\n{gpt_crit}\n"
        f"Gemini critique:\n{gemini_crit}\n"
    )
    if rag_context:
        prompt += f"\nRAG context:\n{rag_context}\n"
    return prompt


def _merge_ai_results(gpt_payload: dict | None, gemini_payload: dict | None) -> dict | None:
    if not gpt_payload and not gemini_payload:
        return None
    gpt_score = gpt_payload.get("overall_score") if gpt_payload else None
    gemini_score = gemini_payload.get("overall_score") if gemini_payload else None
    scores = [score for score in (gpt_score, gemini_score) if isinstance(score, int)]
    avg_score = int(sum(scores) / len(scores)) if scores else None
    combined_issues: list[dict] = []
    seen: set[tuple] = set()
    for payload in (gpt_payload, gemini_payload):
        if not payload:
            continue
        for item in payload.get("issues", []):
            key = (item.get("severity"), item.get("category"), item.get("message_ko"))
            if key in seen:
                continue
            seen.add(key)
            combined_issues.append(item)
    combined_issues = combined_issues[:AI_DIAG_MAX_ISSUES]
    summary_ko = ""
    summary_en = ""
    diagnostics_ko = ""
    diagnostics_en = ""
    for payload in (gpt_payload, gemini_payload):
        if not payload:
            continue
        if not summary_ko:
            summary_ko = payload.get("summary_ko", "")
        if not summary_en:
            summary_en = payload.get("summary_en", "")
        if not diagnostics_ko:
            diagnostics_ko = payload.get("diagnostics_ko", "")
        if not diagnostics_en:
            diagnostics_en = payload.get("diagnostics_en", "")
    return {
        "overall_score": avg_score,
        "summary_ko": summary_ko,
        "summary_en": summary_en,
        "diagnostics_ko": diagnostics_ko,
        "diagnostics_en": diagnostics_en,
        "issues": combined_issues,
        "consensus_notes_ko": "",
        "consensus_notes_en": "",
    }


def _build_ai_issue_payload(issues: list) -> list[dict]:
    payload = []
    for issue in issues[:30]:
        payload.append(
            {
                "id": issue.id,
                "category": issue.category,
                "subtype": issue.subtype,
                "message": redact_text(issue.message),
                "evidence": redact_text(issue.evidence),
                "suggestion": redact_text(issue.suggestion),
            }
        )
    return payload


def _rag_top_k(pages: list[dict], scan_level: str) -> int:
    base = 6 if scan_level == "NONE" else 3
    if any(_is_text_noisy(page.get("text", "")) for page in pages):
        return max(2, min(base, 3))
    return base


def _generate_ai_explanations(client: OpenAIClient, issues: list) -> dict:
    try:
        payload = _build_ai_issue_payload(issues)
        return client.summarize_issues(payload)
    except Exception:
        return {}


def _generate_ai_candidates(
    client: OpenAIClient,
    pages: list[dict],
    scan_level: str,
    language: str,
) -> list[dict]:
    try:
        if not pages:
            return []
        total_limit = _ai_candidate_limit(pages, scan_level)

        candidates: list[dict] = []
        limiter = CandidateLimiter(
            total_limit=total_limit,
            per_page_limit=2,
            per_category_limit=1,
        )
        for page in pages:
            if len(candidates) >= total_limit:
                break
            text = page.get("text", "")
            if not text.strip():
                continue
            max_for_page = max(1, total_limit - len(candidates))
            text_for_ai = truncate_text(text, limit=3000)
            redacted_text = redact_text(text_for_ai)
            results = client.review_page(
                redacted_text, max_candidates=max_for_page, language=language
            )
            for result in results:
                if len(candidates) >= total_limit:
                    break
                candidate = extract_ai_candidate(
                    text=text,
                    redacted_text=redacted_text,
                    result=result,
                    page_number=page.get("page_number", 0),
                )
                if candidate is None:
                    continue
                if limiter.allow(candidate):
                    candidates.append(candidate)
        return candidates
    except Exception:
        return [] 


def _run_rag_qa(
    client: OpenAIClient,
    question: str,
    pages: list[dict],
    rag_collection,
    owner_key: str,
    top_k: int,
    language: str,
    scan_level: str,
    status_callback=None,
    file_hash: str | None = None,
    embedding_provider: str | None = None,
    is_admin: bool = False,
) -> dict | None:
    if not question.strip() or rag_collection is None:
        return None
    if status_callback:
        status_callback("rewrite")
    queries = _expand_rag_queries(client, question, language)
    query_embeddings = client.embed_texts(queries)
    if not query_embeddings:
        return None
    if status_callback:
        status_callback("search")
    where_filter = _rag_where_filter(
        owner_key,
        file_hash or "",
        language,
        embedding_provider or "",
        is_admin,
    )
    chunks = _search_chroma(rag_collection, query_embeddings, top_k, where_filter)
    if not chunks:
        return None
    context = build_context(chunks)
    caution = None
    if scan_level in {"HIGH", "PARTIAL"}:
        caution = (
            "텍스트 품질이 낮아 참고용으로만 답변하세요."
            if language == "ko"
            else "Text quality is low; answer as reference only."
        )
    if status_callback:
        status_callback("answer")
    response = client.rag_qa(
        question=question, context=context, language=language, caution=caution
    )
    if not response:
        return None
    answer = response.get("answer") if isinstance(response, dict) else None
    if not isinstance(answer, dict):
        answer = {}
    citations = response.get("citations") if isinstance(response, dict) else []
    if not isinstance(citations, list):
        citations = []
    filtered = filter_citations(citations, pages, chunks=chunks)
    answer = _apply_rag_answer_guard(answer, filtered, question, language)
    return {
        "question": question,
        "answer": {
            "ko": str(answer.get("ko", "")).strip(),
            "en": str(answer.get("en", "")).strip(),
        },
        "citations": filtered,
    }


def _advance_optim_session(decision: str | None = None):
    session = st.session_state.get("optim_session") or {}
    generator = session.get("generator")
    if generator is None:
        return None, True

    state = None
    done = False
    try:
        while True:
            if decision is not None:
                state = generator.send(decision)
                decision = None
            else:
                state = next(generator)
            session["state"] = state
            st.session_state["optim_state"] = state
            if getattr(state, "decision_required", False):
                break
            if getattr(state, "status", "") == "PASS":
                done = True
                break
    except StopIteration as stop:
        state = stop.value or state
        session["state"] = state
        st.session_state["optim_state"] = state
        done = True

    st.session_state["optim_session"] = session
    return state, done


def _archive_optim_result(result: dict | None) -> None:
    if not result:
        return
    score = result.get("analysis", {}).get("score", 0) or 0
    if score < ARCHIVE_THRESHOLD:
        return
    success = archive_best_practice(
        result,
        embedding_provider=st.session_state.get(
            "embedding_provider", get_default_embedding_provider()
        ),
        min_score=ARCHIVE_THRESHOLD,
    )
    if success:
        st.toast(f"🏆 High Score ({score})! Archived to Best Practices.", icon="💾")


def _append_optim_to_anti_docs(result: dict | None, filename: str | None) -> None:
    if not result or not filename:
        return
    score = result.get("analysis", {}).get("score", 0) or 0
    if score < ARCHIVE_THRESHOLD:
        return
    rewritten = result.get("rewritten_text", "")
    if not rewritten:
        return
    try:
        from langchain_core.documents import Document
    except Exception:
        return
    new_doc = Document(
        page_content=rewritten,
        metadata={
            "source": f"Optimized: {filename}",
            "score": score,
            "level": st.session_state.get("optim_level"),
        },
    )
    if "anti_docs" not in st.session_state or st.session_state["anti_docs"] is None:
        st.session_state["anti_docs"] = []
    st.session_state["anti_docs"].append(new_doc)
    st.session_state["rag_index_cache"] = {}
    st.session_state["anti_indexed"] = True


def _status_from_result(result, error: str | None) -> str:
    if error:
        return "error"
    if result:
        return "ok"
    return "empty"


def _ai_error_message(error: str | None, lang: str) -> str | None:
    if not error:
        return None
    if error.startswith("cooldown_"):
        remaining = error.replace("cooldown_", "")
        return (
            f"AI 호출 쿨다운: {remaining}초 후 다시 시도"
            if lang == "ko"
            else f"AI cooldown: try again in {remaining}s."
        )
    if error.startswith("http_error_"):
        code = error.replace("http_error_", "")
        if code in {"401", "403"}:
            return (
                f"API 키 인증 문제가 있습니다 ({code})."
                if lang == "ko"
                else f"API key authorization failed ({code})."
            )
        if code == "429":
            return (
                f"요청 한도/레이트리밋에 도달했습니다 ({code})."
                if lang == "ko"
                else f"Rate limit exceeded ({code})."
            )
        if code.startswith("5"):
            return (
                f"일시적 서버 오류입니다 ({code})."
                if lang == "ko"
                else f"Temporary server error ({code})."
            )
        return (
            f"요청 실패 ({code})." if lang == "ko" else f"Request failed ({code})."
        )
    if error in {"url_error", "request_failed_TimeoutError"} or "timeout" in error:
        return "네트워크/타임아웃 문제입니다." if lang == "ko" else "Network/timeout error."
    if error in {"json_parse_failed", "invalid_json", "empty_response"}:
        return (
            "응답 형식이 올바르지 않습니다."
            if lang == "ko"
            else "Response format invalid."
        )
    return "요청 실패" if lang == "ko" else "Request failed."


def _extract_keywords(question: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9가-힣]+", question)
    return [token for token in tokens if len(token) >= 2]


def _append_notice(text: str, notice: str) -> str:
    if not notice:
        return text
    if not text:
        return notice
    if notice in text:
        return text
    return f"{text}\n\n{notice}"


def _apply_rag_answer_guard(
    answer: dict, citations: list[dict], question: str, language: str
) -> dict:
    ko_text = str(answer.get("ko", "")).strip()
    en_text = str(answer.get("en", "")).strip()
    if not citations:
        ko_notice = "근거를 찾지 못해 참고용으로 답변합니다."
        en_notice = "No supporting evidence was found; this answer is for reference."
        return {
            "ko": _append_notice(ko_text, ko_notice),
            "en": _append_notice(en_text, en_notice),
        }
    return {"ko": ko_text, "en": en_text}


def _processing_overlay_html(title: str, subtitle: str) -> str:
    return f"""
    <div class="processing-overlay">
      <div class="processing-modal">
        <div class="processing-spinner"></div>
        <div class="processing-title">{title}</div>
        <div class="processing-subtitle">{subtitle}</div>
      </div>
    </div>
    """


def _rag_processing_html(message: str) -> str:
    return (
        "<div class='rag-processing'>"
        "<div class='processing-spinner'></div>"
        f"{message}</div>"
    )


def _get_anti_retriever():
    if st.session_state.get("anti_llm") and st.session_state.get("anti_retriever"):
        return st.session_state["anti_llm"], st.session_state["anti_retriever"]
    from documind.anti.rag.claude import get_claude
    from documind.anti.vectorstore.chroma_raw import get_chroma, save_raw_docs

    chunks = st.session_state.get("anti_chunks") or []
    if not chunks:
        st.session_state["anti_error"] = "no_chunks"
        return None, None
    try:
        save_raw_docs(chunks)
        db = get_chroma()
        retriever = db.as_retriever(search_kwargs={"k": 3})
    except Exception as exc:
        st.session_state["anti_error"] = f"vectorstore_failed:{exc.__class__.__name__}"
        return None, None

    llm = get_claude()
    st.session_state["anti_llm"] = llm
    st.session_state["anti_retriever"] = retriever
    return llm, retriever



header_left, header_right = st.columns([4, 1])
with header_left:
    st.title(t["title"])
with header_right:
    lang = st.radio(
        label=t["language_label"],
        options=["ko", "en"],
        format_func=lambda code: LANG_LABELS[code],
        key="lang",
        horizontal=True,
    )

t = I18N.get(lang, I18N["ko"])

# --------------------------------------------------------------------------
# Authentication Check
# --------------------------------------------------------------------------
def login_screen():
    st.markdown("---")
    tab1, tab2 = st.tabs([t["login_title"], t["signup_title"]])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input(t["username_label"])
            password = st.text_input(t["password_label"], type="password")
            submit = st.form_submit_button(t["login_button"])
            
            if submit:
                user = db_manager.authenticate_user(username, password)
                if user:
                    st.session_state["auth_status"] = True
                    st.session_state["username"] = user["username"]
                    st.session_state["role"] = user["role"]
                    st.success(t["login_success"])
                    st.rerun()
                else:
                    st.error(t["login_failed"])
    
    with tab2:
        with st.form("signup_form"):
            new_user = st.text_input(t["username_label"], key="new_user")
            new_pass = st.text_input(t["password_label"], type="password", key="new_pass")
            submit_signup = st.form_submit_button(t["signup_button"])
            
            if submit_signup:
                if db_manager.register_user(new_user, new_pass):
                    st.success(t["signup_success"])
                else:
                    st.error(t["signup_failed"])

if not st.session_state.get("auth_status"):
    login_screen()
    st.stop()

# Logout Button and Welcome Message in Sidebar
with st.sidebar:
    st.write(t["welcome_msg"].format(username=st.session_state["username"], role=st.session_state["role"]))
    
    # --------------------------------------------------------------------------
    # Sidebar Navigation Menu
    # --------------------------------------------------------------------------
    st.markdown("---")
    menu = st.radio(
        "Menu",
        options=["quality", "optim", "anti", "sqlite", "chroma"],
        format_func=lambda x: t.get(f"menu_{x}", x),
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    if st.button(t["logout_button"]):
        st.session_state["auth_status"] = False
        st.session_state["username"] = None
        st.session_state["role"] = None
        st.rerun()

# --------------------------------------------------------------------------
# DB Explorer UI Functions
# --------------------------------------------------------------------------
def render_sqlite_explorer():
    st.header(t["menu_sqlite"])
    username = st.session_state["username"]
    is_admin = (st.session_state["role"] == "admin")
    
    # Tabs for History and Users (Admin only)
    tabs = ["Analysis History"]
    if is_admin:
        tabs.append("User List")
    
    selected_tab = st.radio("Select View", tabs, horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    if selected_tab == "Analysis History":
        st.subheader("Analysis History")
        history = db_manager.get_user_history(username, is_admin=is_admin, limit=100)
        if not history:
            st.info("No data in SQLite history.")
        else:
            import pandas as pd
            df = pd.DataFrame(history)
            # Rename columns for localized display if needed
            cols = {
                "id": "ID",
                "filename": t["doc_filename"],
                "user_id": t["doc_user"],
                "created_at": t["doc_date"]
            }
            df = df.rename(columns=cols)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Detail View for Download
            st.subheader("Detail View")
            all_ids = [item["id"] for item in history]
            selected_id = st.selectbox(
                "Select ID to view full content",
                options=all_ids,
                key="sqlite_history_detail_select"
            )
            
            if selected_id:
                detail = db_manager.get_history_detail(selected_id)
                if detail:
                    # Check if it's an optimization result with rewritten_text
                    # The data structure is: {"mode": "optim", "result": {"rewritten_text": ...}}
                    # Or direct {"rewritten_text": ...} for older entries
                    result_data = detail.get("result") or detail
                    rewritten_text = result_data.get("rewritten_text")
                    if rewritten_text:
                        from documind.utils.export import create_txt_bytes, create_pdf_bytes
                        fname_base = f"sqlite_export_{selected_id}"
                        
                        sq_col1, sq_col2 = st.columns(2)
                        with sq_col1:
                            st.download_button(
                                label="💾 Download TXT",
                                data=create_txt_bytes(rewritten_text),
                                file_name=f"{fname_base}.txt",
                                mime="text/plain",
                                key=f"sqlite_dl_txt_{selected_id}"
                            )
                        with sq_col2:
                            st.download_button(
                                label="📄 Download PDF",
                                data=create_pdf_bytes(rewritten_text),
                                file_name=f"{fname_base}.pdf",
                                mime="application/pdf",
                                key=f"sqlite_dl_pdf_{selected_id}"
                            )
                        
                        st.text_area("Rewritten Text", value=rewritten_text, height=300)
                    else:
                        st.info("이 항목은 최적화 결과가 아닙니다 (다운로드 불가)." if t.get("lang") == "ko" else "This item is not an optimization result (no download available).")
                    
                    st.json(detail)
                else:
                    st.warning("Failed to load details.")
            
    elif selected_tab == "User List" and is_admin:
        st.subheader("Registered Users")
        users = db_manager.get_all_users()
        if not users:
            st.info("No users found.")
        else:
            import pandas as pd
            df = pd.DataFrame(users)
            st.dataframe(df, use_container_width=True, hide_index=True)

def render_chroma_explorer():
    st.header(t["menu_chroma"])
    username = st.session_state["username"]
    is_admin = (st.session_state["role"] == "admin")
    
    import chromadb
    from pathlib import Path
    
    # Path calculation relative to this file
    # analy_app.py is in AIPOC/app/views/
    persist_dir_raw = str(Path(__file__).resolve().parents[3] / "chroma_raw")
    persist_dir_best = str(Path(__file__).resolve().parents[3] / "chroma_best_practices")
    
    tab_rag, tab_best = st.tabs(["📚 RAG Documents", "🏆 Best Practices"])

    with tab_rag:
        try:
            client = chromadb.PersistentClient(path=persist_dir_raw)
            collection_names = []
            try:
                collection_names = [col.name for col in client.list_collections()]
            except Exception:
                collection_names = []
            if not collection_names:
                collection_names = ["langchain", RAG_COLLECTION_NAME]
            collection_name = st.selectbox(
                "Collection",
                options=collection_names,
                key="chroma_collection_select",
            )
            try:
                collection = client.get_collection(name=collection_name)
            except Exception:
                st.info(f"No '{collection_name}' collection found in ChromaDB.")
                collection = None
            if collection:
                if is_admin:
                    results = collection.get()
                else:
                    results = collection.get(where={"user_id": username})

                if not results or not results.get("documents"):
                    st.info("No RAG data for current user.")
                else:
                    docs = results["documents"]
                    metas = results["metadatas"]
                    ids = results["ids"]

                    display_data = []
                    for i in range(len(docs)):
                        m = metas[i] or {}
                        display_data.append({
                            "ID": ids[i][:8] + "...",
                            t["doc_filename"]: m.get("source", m.get("filename", "N/A")),
                            "Page": m.get("page", "N/A"),
                            t["doc_user"]: m.get("user_id", "N/A"),
                            t["rag_content"]: docs[i][:200] + "..." if len(docs[i]) > 200 else docs[i]
                        })

                    import pandas as pd
                    df = pd.DataFrame(display_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    st.subheader("Detail View")
                    selected_id = st.selectbox(
                        "Select ID to view full content",
                        options=ids,
                        key=f"chroma_detail_select_{collection_name}",
                    )
                    if selected_id:
                        idx = ids.index(selected_id)
                        st.text_area("Full Content", value=docs[idx], height=300)
        except Exception as e:
            st.error(f"Failed to load RAG ChromaDB: {e}")

    with tab_best:
        try:
            client_best = chromadb.PersistentClient(path=persist_dir_best)
            explorer_provider = st.radio(
                "DB Provider",
                options=get_available_embedding_providers(),
                index=0,
                key="bp_explorer_provider_select",
                horizontal=True,
                help="Select which embedding provider's database to inspect."
            )
            
            suffix = (explorer_provider or "default").lower().replace(" ", "_").replace("-", "_")
            coll_name = f"optim_best_practices_v2_{suffix}"
            
            st.info(f"Viewing Best Practices for Provider: **{explorer_provider}** ({coll_name})")
            
            try:
                collection_best = client_best.get_collection(name=coll_name)
            except Exception:
                st.warning(
                    f"No collection found for {explorer_provider}. Try running an optimization first!"
                )
                collection_best = None
            
            if collection_best:
                results_best = collection_best.get(include=["documents", "metadatas"])
                    
                if not results_best or not results_best.get("documents"):
                    st.info("No best practices archived yet.")
                else:
                    docs_b = results_best["documents"]
                    metas_b = results_best["metadatas"]
                    ids_b = results_best["ids"]
                    
                    display_data_b = []
                    for i in range(len(docs_b)):
                        m = metas_b[i] or {}
                        display_data_b.append({
                            "ID": ids_b[i][:8] + "...",
                            "Score": m.get("score", "N/A"),
                            "Level": m.get("target_level", "N/A"),
                            "Model": m.get("model_version", "N/A"),
                            "Timestamp": m.get("timestamp", "N/A")[:19] if m.get("timestamp") else "N/A",
                            "Content": docs_b[i][:150] + "..." if len(docs_b[i]) > 150 else docs_b[i]
                        })
                    
                    import pandas as pd
                    df_b = pd.DataFrame(display_data_b)
                    st.dataframe(df_b, use_container_width=True, hide_index=True)
                    
                    st.subheader("Detail View")
                    selected_id_b = st.selectbox(
                        "Select ID to view full content",
                        options=ids_b,
                        key="chroma_best_detail_select",
                    )
                    if selected_id_b:
                        idx_b = ids_b.index(selected_id_b)
                        
                        # Export Buttons for Best Practice
                        from documind.utils.export import create_txt_bytes, create_pdf_bytes
                        bp_text = docs_b[idx_b]
                        bp_filename = f"best_practice_{ids_b[idx_b][:8]}"
                        
                        bp_col1, bp_col2 = st.columns(2)
                        with bp_col1:
                            st.download_button(
                                label="💾 Download TXT",
                                data=create_txt_bytes(bp_text),
                                file_name=f"{bp_filename}.txt",
                                mime="text/plain",
                                key=f"bp_dl_txt_{ids_b[idx_b]}"
                            )
                        with bp_col2:
                            st.download_button(
                                label="📄 Download PDF",
                                data=create_pdf_bytes(bp_text),
                                file_name=f"{bp_filename}.pdf",
                                mime="application/pdf",
                                key=f"bp_dl_pdf_{ids_b[idx_b]}"
                            )
                        
                        st.text_area("Rewritten Text", value=docs_b[idx_b], height=300)
                        st.json(metas_b[idx_b])
        except Exception as e:
            st.error(f"Failed to load Best Practices ChromaDB: {e}")

# --------------------------------------------------------------------------
# Main Content Branching
# --------------------------------------------------------------------------
if menu == "sqlite":
    render_sqlite_explorer()
    st.stop()
elif menu == "chroma":
    render_chroma_explorer()
    st.stop()

# menu == "analyzer" continues below...
overlay_placeholder = st.empty()
if st.session_state["is_running"]:
    overlay_placeholder.markdown(
        _processing_overlay_html(t["processing_title"], t["processing_subtitle"]),
        unsafe_allow_html=True,
    )
else:
    overlay_placeholder.empty()

with st.container():
    uploaded_file = st.file_uploader(t["upload_label"], type=SUPPORTED_FILE_TYPES)
    ai_available = _ai_available()
    toggle_locked = st.session_state["report"] is not None or st.session_state["is_running"]
    with st.container(border=True):
        toggle_col, action_col = st.columns([3, 1])
        with toggle_col:
            st.markdown(
                "<span id='ai-panel-marker' style='display:block;height:0;line-height:0;'></span>",
                unsafe_allow_html=True,
            )
            # Mode selection is now handled by the sidebar menu
            mode_key = menu
            
            # Display current mode title for clarity
            st.subheader(t.get(f"menu_{mode_key}", mode_key))

            if mode_key == "quality" and ai_available:
                ai_explain_enabled = st.toggle(
                    t["ai_explain_toggle"],
                    value=False,
                    key="ai_explain_toggle",
                    disabled=toggle_locked,
                )
                ai_review_enabled = st.toggle(
                    t["ai_review_toggle"],
                    value=False,
                    key="ai_review_toggle",
                    disabled=toggle_locked,
                )
                if toggle_locked:
                    st.caption(t["ai_toggle_locked_note"])
            else:
                ai_explain_enabled = False
                ai_review_enabled = False
            if mode_key == "optim":
                # DRY: 중앙 집중화된 프로바이더 목록 사용
                provider_options = get_available_providers()
                
                # 주/보조 LLM 선택 (Actor-Critic 분리)
                llm_col1, llm_col2 = st.columns(2)
                with llm_col1:
                    st.selectbox(
                        t["actor_provider_label"],
                        options=provider_options,
                        key="actor_provider",
                        help="문서 생성/재작성을 담당" if st.session_state.get("lang") == "ko" else "Generates/rewrites content"
                    )
                with llm_col2:
                    st.selectbox(
                        t["critic_provider_label"],
                        options=provider_options,
                        key="critic_provider",
                        help="평가/채점을 담당" if st.session_state.get("lang") == "ko" else "Evaluates and scores"
                    )
                
                # Embedding Provider Selection
                embed_options = get_available_embedding_providers()
                st.selectbox(
                    "임베딩 모델" if st.session_state.get("lang") == "ko" else "Embedding Model",
                    options=embed_options,
                    key="embedding_provider",
                    help="RAG 및 분석에 사용될 임베딩 모델"
                )
                
                # Save settings when changed (using callback would be better but simple logic here)
                if st.session_state["actor_provider"]:
                    db_manager.save_setting("actor_provider", st.session_state["actor_provider"])
                if st.session_state["critic_provider"]:
                    db_manager.save_setting("critic_provider", st.session_state["critic_provider"])
                if st.session_state["embedding_provider"]:
                    db_manager.save_setting("embedding_provider", st.session_state["embedding_provider"])

                st.selectbox(
                    t["optim_level_label"],
                    options=["public", "student", "worker", "expert"],
                    key="optim_level",
                )
        with action_col:
            run_clicked = st.button(t["analyze_button"])

    if uploaded_file is None:
        st.session_state["file_info"] = None
        st.session_state["file_hash"] = None
        st.session_state["report"] = None
        st.session_state["page_char_counts"] = None
        st.session_state["normalized_pages"] = None
        st.session_state["ai_explanations"] = None
        st.session_state["ai_candidates"] = None
        st.session_state["ai_status"] = {"explain": None, "review": None}
        st.session_state["ai_errors"] = {"explain": None, "review": None}
        st.session_state["ai_cache"] = {}
        st.session_state["ai_diag_cache"] = {}
        st.session_state["ai_diag_result"] = None
        st.session_state["ai_diag_status"] = None
        st.session_state["ai_diag_errors"] = {"gpt": None, "gemini": None, "final": None}
        st.session_state["rag_index_cache"] = {}
        st.session_state["rag_last_question"] = ""
        st.session_state["rag_last_result"] = None
        st.session_state["rag_status"] = None
        st.session_state["rag_error"] = None
        st.session_state["anti_docs"] = None
        st.session_state["anti_indexed"] = False
        st.session_state["anti_error"] = None
        st.session_state["anti_chunks"] = None
        st.session_state["anti_llm"] = None
        st.session_state["anti_retriever"] = None
        st.session_state["optim_result"] = None
        st.session_state["optim_error"] = None
        st.session_state["optim_session"] = None
        st.session_state["optim_state"] = None
        st.session_state["optim_engine"] = None
        st.session_state.pop("antithesis", None)
    else:
        file_size = getattr(uploaded_file, "size", None)
        file_info = (uploaded_file.name, file_size)
        if file_info != st.session_state["file_info"]:
            st.session_state["file_info"] = file_info
            st.session_state["file_hash"] = None
            st.session_state["report"] = None
            st.session_state["page_char_counts"] = None
            st.session_state["normalized_pages"] = None
            st.session_state["ai_explanations"] = None
            st.session_state["ai_candidates"] = None
            st.session_state["ai_status"] = {"explain": None, "review": None}
            st.session_state["ai_errors"] = {"explain": None, "review": None}
            st.session_state["ai_cache"] = {}
            st.session_state["ai_diag_cache"] = {}
            st.session_state["ai_diag_result"] = None
            st.session_state["ai_diag_status"] = None
            st.session_state["ai_diag_errors"] = {"gpt": None, "gemini": None, "final": None}
            st.session_state["rag_index_cache"] = {}
            st.session_state["rag_last_question"] = ""
            st.session_state["rag_last_result"] = None
            st.session_state["rag_status"] = None
            st.session_state["rag_error"] = None
            st.session_state["anti_docs"] = None
            st.session_state["anti_indexed"] = False
            st.session_state["anti_error"] = None
            st.session_state["anti_chunks"] = None
            st.session_state["anti_llm"] = None
            st.session_state["anti_retriever"] = None
            st.session_state["optim_result"] = None
            st.session_state["optim_error"] = None
            st.session_state["optim_session"] = None
            st.session_state["optim_state"] = None
            st.session_state["optim_engine"] = None
            st.session_state.pop("antithesis", None)

        if run_clicked:
            report = None
            page_char_counts = None
            ai_explanations = None
            ai_candidates = None
            ai_status = {"explain": None, "review": None}
            ai_errors = {"explain": None, "review": None}
            ai_diag_result = None
            ai_diag_status = None
            ai_diag_errors = {"gpt": None, "gemini": None, "final": None}
            try:
                st.session_state["is_running"] = True
                overlay_placeholder.markdown(
                    _processing_overlay_html(
                        t["processing_title"], t["processing_subtitle"]
                    ),
                    unsafe_allow_html=True,
                )
                file_bytes = uploaded_file.getvalue()
                if not st.session_state["file_hash"]:
                    st.session_state["file_hash"] = hashlib.sha256(file_bytes).hexdigest()[:12]
                if mode_key == "quality":
                    report = run_pipeline(
                        file_bytes,
                        uploaded_file.name,
                        language=lang,
                    )
                    loaded = load_document(file_bytes, uploaded_file.name)
                    normalized = normalize_pages(loaded["pages"])
                    page_char_counts = [
                        {
                            "page": page["page_number"],
                            "char_count": len(page["text"]),
                        }
                        for page in normalized["pages"]
                    ]
                    st.session_state["normalized_pages"] = normalized["pages"]
                    if ai_available and report is not None and (ai_explain_enabled or ai_review_enabled):
                        cache_key = _ai_cache_key(
                            st.session_state["file_hash"],
                            lang,
                            ai_explain_enabled,
                            ai_review_enabled,
                        )
                        cached = st.session_state["ai_cache"].get(cache_key)
                        if cached:
                            ai_explanations = cached.get("ai_explanations")
                            ai_candidates = cached.get("ai_candidates")
                            ai_status = cached.get("ai_status", ai_status)
                            ai_errors = cached.get("ai_errors", ai_errors)
                            logger.debug("AI cache hit key=%s", cache_key)
                        else:
                            now = time.time()
                            elapsed = now - st.session_state["last_ai_run_ts"]
                            cooldown_hit = (
                                AI_COOLDOWN_SECONDS > 0 and elapsed < AI_COOLDOWN_SECONDS
                            )
                            if cooldown_hit:
                                remaining = max(
                                    1, int(AI_COOLDOWN_SECONDS - elapsed + 0.999)
                                )
                                cooldown_error = f"cooldown_{remaining}"
                                if ai_explain_enabled:
                                    ai_errors["explain"] = cooldown_error
                                    ai_status["explain"] = "cooldown"
                                if ai_review_enabled:
                                    ai_errors["review"] = cooldown_error
                                    ai_status["review"] = "cooldown"
                                logger.debug(
                                    "AI cooldown hit remaining=%s", remaining
                                )
                            else:
                                ai_ran = False
                                if ai_explain_enabled:
                                    client = OpenAIClient()
                                    ai_explanations = _generate_ai_explanations(
                                        client, report.issues
                                    )
                                    st.session_state["ai_call_count"]["explain"] += 1
                                    logger.debug(
                                        "AI explain calls=%s",
                                        st.session_state["ai_call_count"]["explain"],
                                    )
                                    ai_errors["explain"] = client.last_error
                                    ai_status["explain"] = _status_from_result(
                                        ai_explanations, ai_errors["explain"]
                                    )
                                    ai_ran = True
                                if ai_review_enabled:
                                    client = OpenAIClient()
                                    ai_candidates = _generate_ai_candidates(
                                        client,
                                        normalized["pages"],
                                        report.document_meta.scan_level,
                                        lang,
                                    )
                                    st.session_state["ai_call_count"]["review"] += 1
                                    logger.debug(
                                        "AI review calls=%s",
                                        st.session_state["ai_call_count"]["review"],
                                    )
                                    ai_errors["review"] = client.last_error
                                    ai_status["review"] = _status_from_result(
                                        ai_candidates, ai_errors["review"]
                                    )
                                    ai_ran = True
                                if ai_ran:
                                    st.session_state["last_ai_run_ts"] = time.time()
                                st.session_state["ai_cache"][cache_key] = {
                                    "ai_explanations": ai_explanations,
                                    "ai_candidates": ai_candidates,
                                    "ai_status": ai_status,
                                    "ai_errors": ai_errors,
                                }
                    if report is not None:
                        embedding_provider = (
                            st.session_state.get("embedding_provider") or "OpenAI"
                        )
                        gpt_ok = _gpt_available()
                        gemini_ok = _gemini_available()
                        if gpt_ok or gemini_ok:
                            diag_cache_key = _ai_diag_cache_key(
                                st.session_state["file_hash"], lang, embedding_provider
                            )
                            cached_diag = st.session_state["ai_diag_cache"].get(
                                diag_cache_key
                            )
                            if cached_diag:
                                ai_diag_result = cached_diag.get("ai_diag_result")
                                ai_diag_status = cached_diag.get("ai_diag_status")
                                ai_diag_errors = cached_diag.get(
                                    "ai_diag_errors", ai_diag_errors
                                )
                            else:
                                now = time.time()
                                elapsed = now - st.session_state["last_ai_diag_ts"]
                                cooldown_hit = (
                                    AI_COOLDOWN_SECONDS > 0
                                    and elapsed < AI_COOLDOWN_SECONDS
                                )
                                if cooldown_hit:
                                    remaining = max(
                                        1, int(AI_COOLDOWN_SECONDS - elapsed + 0.999)
                                    )
                                    ai_diag_status = "cooldown"
                                    ai_diag_errors["final"] = f"cooldown_{remaining}"
                                else:
                                    internal_payload = _build_internal_diagnosis_payload(
                                        report, lang
                                    )
                                    owner_key = _rag_owner_key(
                                        st.session_state.get("username"),
                                        st.session_state["file_hash"],
                                        lang,
                                        embedding_provider,
                                    )
                                    source_name = (
                                        uploaded_file.name
                                        if uploaded_file is not None
                                        else report.document_meta.file_name
                                    )
                                    rag_client = OpenAIClient(
                                        embedding_provider=embedding_provider
                                    )
                                    rag_context = _build_rag_context_for_diagnosis(
                                        rag_client,
                                        normalized["pages"],
                                        report,
                                        source_name,
                                        st.session_state["file_hash"],
                                        lang,
                                        embedding_provider,
                                        st.session_state.get("username"),
                                        owner_key,
                                    )
                                    prompt = _build_ai_diag_prompt(
                                        internal_payload, rag_context, lang
                                    )
                                    gpt_payload = None
                                    gemini_payload = None
                                    gpt_critique = None
                                    gemini_critique = None
                                    if gpt_ok:
                                        gpt_payload, ai_diag_errors["gpt"] = _run_gpt_diagnosis(
                                            prompt
                                        )
                                    if gemini_ok:
                                        gemini_payload, ai_diag_errors["gemini"] = _run_gemini_diagnosis(
                                            prompt
                                        )
                                    scores = [
                                        payload.get("overall_score")
                                        for payload in (gpt_payload, gemini_payload)
                                        if payload and isinstance(payload.get("overall_score"), int)
                                    ]
                                    average_score = (
                                        int(sum(scores) / len(scores)) if scores else None
                                    )
                                    if gpt_payload and gemini_payload:
                                        gpt_critique, _ = _run_gpt_critique(
                                            gpt_payload, gemini_payload
                                        )
                                        gemini_critique, _ = _run_gemini_critique(
                                            gemini_payload, gpt_payload
                                        )
                                    final_payload = None
                                    if gpt_payload or gemini_payload:
                                        final_prompt = _build_ai_final_prompt(
                                            internal_payload,
                                            rag_context,
                                            gpt_payload,
                                            gemini_payload,
                                            gpt_critique,
                                            gemini_critique,
                                            average_score,
                                            lang,
                                        )
                                        if gpt_ok:
                                            final_payload, ai_diag_errors["final"] = _run_gpt_diagnosis(
                                                final_prompt
                                            )
                                        if final_payload is None and gemini_ok:
                                            final_payload, ai_diag_errors["final"] = _run_gemini_diagnosis(
                                                final_prompt
                                            )
                                    if final_payload is None:
                                        final_payload = _merge_ai_results(
                                            gpt_payload, gemini_payload
                                        )
                                    ai_diag_result = {
                                        "final": final_payload,
                                        "gpt": gpt_payload,
                                        "gemini": gemini_payload,
                                        "gpt_critique": gpt_critique,
                                        "gemini_critique": gemini_critique,
                                        "average_score": average_score,
                                        "rag_context": rag_context,
                                    }
                                    ai_diag_status = "ok" if final_payload else "error"
                                    st.session_state["last_ai_diag_ts"] = time.time()
                                    st.session_state["ai_diag_cache"][diag_cache_key] = {
                                        "ai_diag_result": ai_diag_result,
                                        "ai_diag_status": ai_diag_status,
                                        "ai_diag_errors": ai_diag_errors,
                                    }
                elif mode_key == "anti":
                    from documind.anti.ingest.pdf_loader import load_pdf_with_ocr
                    from documind.anti.ingest.splitter import split_docs
                    from documind.ingest.loader import load_document
                    from langchain_core.documents import Document
                    import pytesseract

                    tesseract_ok = True
                    try:
                        pytesseract.get_tesseract_version()
                    except Exception:
                        tesseract_ok = False

                    lower_name = uploaded_file.name.lower()
                    if not tesseract_ok and lower_name.endswith(".pdf"):
                        st.session_state["anti_error"] = "tesseract_missing"
                        st.session_state["anti_indexed"] = False
                        st.session_state["anti_docs"] = None
                        st.session_state["anti_chunks"] = None
                    else:
                        docs = []
                        if lower_name.endswith(".pdf"):
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(file_bytes)
                                tmp_path = tmp.name
                            docs = load_pdf_with_ocr(tmp_path)
                        else:
                            # TXT/MD/DOCX -> Unified Loader -> Document
                            loaded = load_document(file_bytes, uploaded_file.name)
                            for p in loaded["pages"]:
                                if p["text"].strip():
                                    docs.append(Document(
                                        page_content=p["text"],
                                        metadata={"page": p["page_number"], "source": uploaded_file.name}
                                    ))
                            
                        chunks = split_docs(docs)
                        st.session_state["anti_docs"] = docs
                        st.session_state["anti_chunks"] = chunks
                        st.session_state["anti_indexed"] = True
                        st.session_state["anti_error"] = None
                else:
                    loaded = load_document(file_bytes, uploaded_file.name)
                    normalized = normalize_pages(loaded["pages"])
                    text = "\n\n".join(
                        page["text"] for page in normalized["pages"] if page["text"].strip()
                    )
                    if not text.strip():
                        st.session_state["optim_error"] = "empty_text"
                        st.warning(t["scan_caution"])
                    else:
                        st.session_state["optim_result"] = None
                        st.session_state["optim_error"] = None
                        st.session_state["optim_state"] = None
                        st.session_state["optim_session"] = None
                        st.session_state["optim_engine"] = None

                        # Progress Callback for Overlay
                        def _optim_progress(status, step, total, score, feedback, message):
                            title = f"{t['processing_title']}"
                            if total > 0:
                                title += f" ({step}/{total})"
                            
                            subtitle_parts = []
                            if message:
                                subtitle_parts.append(message)
                            if score:
                                subtitle_parts.append(f"<b>Score: {score}</b>")
                            
                            subtitle = " <br> ".join(subtitle_parts)
                            overlay_placeholder.markdown(
                                _processing_overlay_html(title, subtitle),
                                unsafe_allow_html=True,
                            )

                        # Actor-Critic: 주 LLM (생성) + 보조 LLM (평가)
                        optimizer = TargetOptimizer(
                            st.session_state["actor_provider"],
                            embedding_provider=st.session_state.get("embedding_provider"),
                        )
                        session = optimizer.start_interactive(
                            text=text,
                            target_level=st.session_state["optim_level"],
                            progress_callback=_optim_progress,
                            critic_provider=st.session_state["critic_provider"],
                        )
                        st.session_state["optim_session"] = session
                        st.session_state["optim_engine"] = optimizer

                        state, done = _advance_optim_session()
                        if done:
                            result = optimizer.finalize_interactive(session, state)
                            st.session_state["optim_result"] = result
                            st.session_state["optim_error"] = None
                            st.session_state["optim_state"] = None
                            st.session_state["optim_session"] = None
                            st.session_state["optim_engine"] = None
                            _archive_optim_result(result)
                            _append_optim_to_anti_docs(result, uploaded_file.name)
                    st.session_state["mode_last_run"] = mode_key
            except Exception:
                logger.exception("Pipeline failed")
                if mode_key == "anti" and st.session_state.get("anti_error"):
                    pass
                else:
                    st.error(t["error"])
                if mode_key == "anti":
                    # DB Save for Anti mode (if chunks exist)
                    chunks = st.session_state.get("anti_chunks")
                    if chunks and uploaded_file and st.session_state.get("file_hash"):
                         try:
                            # Save basic info that anti-analysis was done
                            anti_report = {
                                "mode": "anti",
                                "chunk_count": len(chunks),
                                "status": "indexed"
                            }
                            db_manager.save_history_with_user(
                                f"[ANTI] {uploaded_file.name}",
                                st.session_state["file_hash"],
                                anti_report,
                                st.session_state["username"]
                            )
                         except Exception as e:
                            logger.error(f"Failed to save anti history: {e}")

                    if not st.session_state.get("anti_error"):
                        st.session_state["anti_error"] = "pipeline_failed"
                    st.session_state["anti_indexed"] = False
                if mode_key == "optim":
                    # DB Save for Optim mode is handled inside process_optim_mode usually, 
                    # but if it failed here, we might not save. 
                    # If success, it should be saved where result is generated.
                    # See below for success case handling.
                    st.session_state["optim_error"] = "pipeline_failed"
            finally:
                st.session_state["is_running"] = False
                overlay_placeholder.empty()

            if mode_key == "quality" and report is not None:
                st.session_state["report"] = report
                st.session_state["page_char_counts"] = page_char_counts
                st.session_state["ai_explanations"] = ai_explanations
                st.session_state["ai_candidates"] = ai_candidates
                st.session_state["ai_status"] = ai_status
                st.session_state["ai_errors"] = ai_errors
                st.session_state["ai_diag_result"] = ai_diag_result
                st.session_state["ai_diag_status"] = ai_diag_status
                st.session_state["ai_diag_errors"] = ai_diag_errors
                
                # Save Analysis History to DB
                if uploaded_file and st.session_state.get("file_hash"):
                    try:
                        db_manager.save_history_with_user(
                            uploaded_file.name,
                            st.session_state["file_hash"],
                            report.model_dump(),
                            st.session_state["username"]
                        )
                    except Exception as e:
                        logger.error(f"Failed to save history: {e}")

    if mode_key == "anti":
        anti_docs = st.session_state.get("anti_docs") or []
        anti_error = st.session_state.get("anti_error")
        if anti_error:
            if anti_error == "tesseract_missing":
                st.error("Tesseract가 설치되어 있지 않습니다. OCR을 사용하려면 Tesseract를 설치해 주세요.")
            elif anti_error.startswith("vectorstore_failed"):
                st.error("벡터스토어 초기화에 실패했습니다. 터미널 로그를 확인해 주세요.")
            else:
                st.warning(t["error"])
        if not anti_docs:
            _render_empty_state(t["no_report"])
        else:
            st.success(t["anti_indexed"])
            st.subheader(t["anti_preview_title"])
            for doc in anti_docs:
                page = doc.metadata.get("page")
                source = doc.metadata.get("source", "pdf")
                label = "OCR" if source == "ocr" else "PDF"
                with st.expander(f"{label} | Page {page}"):
                    st.text(doc.page_content[:3000])

            question = st.text_input(t["anti_question_label"])
            if question:
                from documind.anti.rag.chain import get_rag_chain

                llm, retriever = _get_anti_retriever()
                if not llm or not retriever:
                    st.stop()
                rag_chain = get_rag_chain(llm, retriever)
                with st.spinner("답변 생성 중..." if lang == "ko" else "Generating answer..."):
                    answer = rag_chain.invoke(question)
                st.markdown(f"### {t['anti_answer_title']}")
                st.write(answer)

            st.divider()
            st.subheader(t["anti_analysis_title"])
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(t["anti_summary_button"]):
                    from documind.anti.rag.chain import get_rag_chain

                    llm, retriever = _get_anti_retriever()
                    if not llm or not retriever:
                        st.stop()
                    rag_chain = get_rag_chain(llm, retriever)
                    with st.spinner("요약 중..." if lang == "ko" else "Summarizing..."):
                        answer = rag_chain.invoke("이 문서의 핵심 내용을 요약해줘")
                    st.write(answer)
            with col2:
                if st.button(t["anti_antithesis_button"]):
                    from documind.anti.rag.chain import get_antithesis_chain

                    llm, retriever = _get_anti_retriever()
                    if not llm or not retriever:
                        st.stop()
                    antithesis_chain = get_antithesis_chain(llm, retriever)
                    with st.spinner("비판적으로 분석 중..." if lang == "ko" else "Analyzing critically..."):
                        antithesis = antithesis_chain.invoke("이 문서 전체를 비판적으로 분석해줘")
                    st.session_state["antithesis"] = antithesis
                    st.markdown(f"### {t['anti_antithesis_button']}")
                    st.write(antithesis)
            with col3:
                if st.button(t["anti_revision_button"]):
                    if "antithesis" not in st.session_state:
                        st.warning(t["anti_revision_missing"])
                    else:
                        from documind.anti.rag.chain import get_revision_chain

                        llm, retriever = _get_anti_retriever()
                        if not llm or not retriever:
                            st.stop()
                        revision_chain = get_revision_chain(llm, retriever)
                        with st.spinner("문서 개선 중..." if lang == "ko" else "Rewriting..."):
                            revised = revision_chain.invoke({
                                "antithesis": st.session_state["antithesis"]
                            })
                        st.markdown(f"### {t['anti_revision_button']}")
                        st.write(revised)
        st.stop()

    if mode_key == "optim":
        optim_result = st.session_state.get("optim_result")
        optim_error = st.session_state.get("optim_error")
        optim_state = st.session_state.get("optim_state")
        if optim_error == "pipeline_failed":
            st.warning(t["error"])
        if optim_error == "empty_text":
            st.warning(t["scan_caution"])
        if optim_state and getattr(optim_state, "decision_required", False):
            st.subheader(t["optim_decision_title"])
            st.info(t["optim_decision_prompt"].format(score=optim_state.current_score))
            
            # Buttons moved to top
            action_col1, action_col2, action_col3 = st.columns(3)
            accept_clicked = action_col1.button(t["optim_accept_button"], type="primary")
            retry_clicked = action_col2.button(t["optim_retry_button"])
            autorun_clicked = action_col3.button(t.get("optim_autorun_button", "끝까지 자동 진행"))
            
            if accept_clicked or retry_clicked or autorun_clicked:
                st.session_state["is_running"] = True
                overlay_placeholder.markdown(
                    _processing_overlay_html(
                        t["processing_title"], t["processing_subtitle"]
                    ),
                    unsafe_allow_html=True,
                )
                
                decision = "retry"
                if accept_clicked:
                    decision = "accept"
                elif autorun_clicked:
                    decision = "auto_run"
                
                state, done = _advance_optim_session(decision)
                st.session_state["is_running"] = False
                overlay_placeholder.empty()
                if done:
                    optimizer = st.session_state.get("optim_engine")
                    session = st.session_state.get("optim_session")
                    if optimizer and session:
                        result = optimizer.finalize_interactive(session, state)
                        st.session_state["optim_result"] = result
                        st.session_state["optim_error"] = None
                        st.session_state["optim_state"] = None
                        st.session_state["optim_session"] = None
                        st.session_state["optim_engine"] = None
                        _archive_optim_result(result)
                        _append_optim_to_anti_docs(result, uploaded_file.name)
                st.rerun()

            st.subheader(t["optim_result_title"])
            st.write(optim_state.current_text)
            
            if getattr(optim_state, "feedback", ""):
                 with st.expander(t["optim_analysis_title"]):
                    st.json(
                        {
                            "score": optim_state.current_score,
                            "feedback": optim_state.feedback,
                            "status": optim_state.status,
                        }
                    )

        elif not optim_result:
            _render_empty_state(t["no_report"])
        else:
            st.subheader(t["optim_result_title"])
            st.write(optim_result.get("rewritten_text", ""))
            
            # Download Section
            st.markdown("---")
            st.caption("결과 다운로드")
            
            from documind.utils.export import create_txt_bytes, create_docx_bytes, create_pdf_bytes, create_zip_bytes
            
            out_text = optim_result.get("rewritten_text", "")
            base_name = f"optimized_{uploaded_file.name.rsplit('.', 1)[0]}"
            
            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
            
            with d_col1:
                st.download_button(
                    label="TXT",
                    data=create_txt_bytes(out_text),
                    file_name=f"{base_name}.txt",
                    mime="text/plain"
                )
            with d_col2:
                st.download_button(
                    label="DOCX",
                    data=create_docx_bytes(out_text),
                    file_name=f"{base_name}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            with d_col3:
                st.download_button(
                    label="PDF",
                    data=create_pdf_bytes(out_text),
                    file_name=f"{base_name}.pdf",
                    mime="application/pdf"
                )
            with d_col4:
                # ALL (ZIP)
                all_files = {
                    f"{base_name}.txt": create_txt_bytes(out_text),
                    f"{base_name}.docx": create_docx_bytes(out_text),
                    f"{base_name}.pdf": create_pdf_bytes(out_text),
                }
                st.download_button(
                    label="ALL (ZIP)",
                    data=create_zip_bytes(all_files),
                    file_name=f"{base_name}.zip",
                    mime="application/zip",
                    type="primary"
                )


            # Save Optim History to DB
            if optim_result and uploaded_file and st.session_state.get("file_hash"):
                 try:
                    # Avoid duplicated saving by checking session flag or rely on DB constraint (if any)
                    # For now, simple save.
                    optim_report = {
                        "mode": "optim",
                        "score": optim_result.get("analysis", {}).get("score"),
                        "result_keys": list(optim_result.keys())
                    }
                    # We might want to store the full result if it's not too huge, 
                    # but 'optimization' result can be large. Let's store truncated or essential parts.
                    # Or just store the full thing as JSON.
                    # SQLite TEXT limit is huge, so it's fine.
                    full_report_to_save = {
                        "mode": "optim",
                        "result": optim_result
                    }
                    
                    # We should check if we already saved this run
                    save_key = f"optim_saved_{st.session_state['file_hash']}_{st.session_state['optim_level']}"
                    if not st.session_state.get(save_key):
                        db_manager.save_history_with_user(
                            f"[OPTIM] {uploaded_file.name}",
                            st.session_state["file_hash"],
                            full_report_to_save,
                            st.session_state["username"]
                        )
                        st.session_state[save_key] = True
                 except Exception as e:
                    logger.error(f"Failed to save optim history: {e}")

            analysis = optim_result.get("analysis")
            score_suffix = ""
            if analysis and "score" in analysis:
                score_suffix = f" (Score: {analysis['score']})"
            
            if analysis:
                with st.expander(f"{t['optim_analysis_title']}{score_suffix}"):
                    st.json(analysis)
            
            # Diff View
            original = optim_result.get("original_text", "")
            rewritten = optim_result.get("rewritten_text", "")
            if original and rewritten:
                with st.expander(t.get("optim_diff_title", "변경 사항 비교 (Diff)") + score_suffix):
                    import difflib
                    diff = difflib.unified_diff(
                        original.splitlines(),
                        rewritten.splitlines(),
                        fromfile='Original',
                        tofile='Optimized',
                        lineterm=''
                    )
                    diff_text = "\n".join(list(diff))
                    if diff_text:
                        st.code(diff_text, language="diff")
                    else:
                        st.info("변경 사항이 없습니다." if lang == "ko" else "No changes.")

            keywords = optim_result.get("keywords") or []
            if keywords:
                st.caption(t["optim_keywords_label"])
                st.write(", ".join(map(str, keywords)))
        st.stop()

    report = st.session_state.get("report")
    normalized_pages = st.session_state.get("normalized_pages") or []
    if ai_available and (ai_explain_enabled or ai_review_enabled):
        ai_errors = st.session_state.get("ai_errors") or {}
        cooldown_error = None
        for key in ("explain", "review"):
            err = ai_errors.get(key)
            if err and err.startswith("cooldown_"):
                cooldown_error = err
                break
        if cooldown_error:
            message = _ai_error_message(cooldown_error, lang)
            if message:
                st.info(message)

    summary_tab, issues_tab, diagnostics_tab, qa_tab, download_tab, history_tab, help_tab = st.tabs(
        [
            t["tab_summary"],
            t["tab_issues"],
            t["tab_diagnostics"],
            t["tab_qa"],
            t["tab_download"],
            t.get("tab_history", "History"),
            t["tab_help"],
        ]
    )

    with summary_tab:
        if report is None:
            _render_empty_state(t["no_report"])
        else:
            meta = report.document_meta
            issues = report.issues
            ai_diag = st.session_state.get("ai_diag_result") or {}
            ai_final = ai_diag.get("final") if isinstance(ai_diag, dict) else None
            ai_issues = (
                ai_final.get("issues")
                if isinstance(ai_final, dict)
                else None
            )
            use_ai = bool(ai_final and isinstance(ai_issues, list))
            score_display = (
                ai_final.get("overall_score")
                if use_ai and ai_final.get("overall_score") is not None
                else (
                    report.overall_score
                    if report.overall_score is not None
                    else t["score_na"]
                )
            )
            if use_ai:
                actionable_count = sum(
                    1
                    for issue in ai_issues
                    if issue.get("severity") in {"RED", "YELLOW"}
                )
                total_issue_count = len(ai_issues)
            else:
                actionable_count = sum(
                    1 for issue in issues if issue.kind in {"ERROR", "WARNING"}
                )
                total_issue_count = len(issues)
            metric_left, metric_mid, metric_right, metric_extra = st.columns(4, gap="small")
            metric_left.metric(t["score_label"], score_display)
            metric_mid.metric(t["confidence_label"], report.score_confidence)
            metric_right.metric(t["actionable_count_label"], actionable_count)
            metric_extra.metric(t["issue_count_label"], total_issue_count)

            if use_ai:
                st.subheader(t["ai_diag_title"])
                st.caption(t["ai_diag_caption"])
                summary_text = (
                    ai_final.get("summary_en", "")
                    if lang == "en"
                    else ai_final.get("summary_ko", "")
                )
                if summary_text:
                    st.write(summary_text)
                notes_text = (
                    ai_final.get("consensus_notes_en", "")
                    if lang == "en"
                    else ai_final.get("consensus_notes_ko", "")
                )
                if notes_text:
                    st.caption(f"{t['ai_diag_consensus_notes']}: {notes_text}")
            else:
                gpt_ok = _gpt_available()
                gemini_ok = _gemini_available()
                if not gpt_ok and not gemini_ok:
                    st.info(t["ai_diag_missing_key"])
                elif not gpt_ok:
                    st.info(
                        t["ai_diag_partial_key"].format(
                            provider="GPT", fallback="Gemini"
                        )
                    )
                elif not gemini_ok:
                    st.info(
                        t["ai_diag_partial_key"].format(
                            provider="Gemini", fallback="GPT"
                        )
                    )
                final_error = (st.session_state.get("ai_diag_errors") or {}).get("final")
                if final_error:
                    message = _ai_error_message(final_error, lang) or t["ai_diag_unavailable"]
                    st.warning(message)

            if meta.scan_level in {"HIGH", "PARTIAL"}:
                st.warning(t["scan_caution"])

            if report.overall_score is None:
                st.warning(t["low_confidence_warning"])

            severity_groups = {"RED": [], "YELLOW": [], "GREEN": []}
            if use_ai:
                for issue in ai_issues:
                    severity = issue.get("severity")
                    if severity in severity_groups:
                        severity_groups[severity].append(issue)
            else:
                for issue in issues:
                    if issue.severity in severity_groups:
                        severity_groups[issue.severity].append(issue)
            st.subheader(t["severity_breakdown_title"])
            st.caption(t["severity_breakdown_caption"])
            for severity in ("RED", "YELLOW", "GREEN"):
                icon = SEVERITY_ICONS.get(severity, "")
                label = _severity_label(severity, lang, show_raw=False)
                grouped = severity_groups.get(severity, [])
                if not grouped:
                    st.caption(f"{icon} {label}: 0")
                    continue
                with st.expander(
                    f"{icon} {label} ({len(grouped)})", expanded=(severity == "RED")
                ):
                    lines = []
                    for issue in grouped:
                        if use_ai:
                            category_label = _category_label(
                                issue.get("category"), lang
                            )
                            message = (
                                issue.get("message_en", "")
                                if lang == "en"
                                else issue.get("message_ko", "")
                            )
                            suggestion = (
                                issue.get("suggestion_en", "")
                                if lang == "en"
                                else issue.get("suggestion_ko", "")
                            )
                            page = issue.get("page", 0)
                            lines.append(
                                f"- p{page} · {category_label} · {message} → {suggestion}"
                            )
                        else:
                            category_label = _category_label(issue.category, lang)
                            action_text = _issue_action(issue, lang)
                            lines.append(
                                f"- p{issue.location.page} · {category_label} · "
                                f"{issue.message} → {action_text}"
                            )
                    st.markdown("\n".join(lines))

    with issues_tab:
        if report is None:
            _render_empty_state(t["no_report"])
        else:
            meta = report.document_meta
            issues = report.issues
            ai_diag = st.session_state.get("ai_diag_result") or {}
            ai_final = ai_diag.get("final") if isinstance(ai_diag, dict) else None
            ai_issues = (
                ai_final.get("issues")
                if isinstance(ai_final, dict)
                else None
            )
            if isinstance(ai_issues, list) and ai_issues:
                issues = _convert_ai_issues(ai_issues, lang)
            page_type_map = {profile.page: profile.type for profile in meta.page_profiles}

            categories = sorted({issue.category for issue in issues})
            category_options = [t["filter_all"]] + categories
            st.subheader(t["filter_title"])
            with st.container(border=True):
                st.markdown(f"**{t['filter_category']}**")
                selected_category = st.selectbox(
                    t["filter_category"],
                    category_options,
                    format_func=(
                        lambda value: value
                        if value == t["filter_all"]
                        else _category_label(value, lang)
                    ),
                    label_visibility="collapsed",
                )

                sev_col, kind_col, option_col = st.columns([1, 1, 1])
                with sev_col:
                    st.markdown(f"**{t['filter_severity']}**")
                    show_red = st.checkbox(
                        f"{t['severity_high_label']} (RED)",
                        value=True,
                        key="sev_red",
                    )
                    show_yellow = st.checkbox(
                        f"{t['severity_mid_label']} (YELLOW)",
                        value=True,
                        key="sev_yellow",
                    )
                    show_green = st.checkbox(
                        f"{t['severity_low_label']} (GREEN)",
                        value=True,
                        key="sev_green",
                    )
                with kind_col:
                    st.markdown(f"**{t['filter_kind']}**")
                    show_error = st.checkbox(
                        f"{t['kind_error_label']} (ERROR)",
                        value=True,
                        key="kind_error",
                    )
                    show_warning = st.checkbox(
                        f"{t['kind_warning_label']} (WARNING)",
                        value=True,
                        key="kind_warning",
                    )
                with option_col:
                    st.markdown(f"**{t['filter_options_label']}**")
                    include_note = st.toggle(
                        t["filter_include_note"],
                        value=False,
                        help="NOTE issues are informational." if lang == "en" else "참고 이슈를 포함합니다.",
                    )
                    show_raw = st.toggle(
                        t["filter_show_raw"],
                        value=False,
                        help="Show raw enum values." if lang == "en" else "원문 enum 값을 표시합니다.",
                    )

            st.caption(t["filter_caption"])
            st.caption(t["severity_mapping_caption"])

            selected_severity = []
            if show_red:
                selected_severity.append("RED")
            if show_yellow:
                selected_severity.append("YELLOW")
            if show_green:
                selected_severity.append("GREEN")

            selected_kinds = []
            if show_error:
                selected_kinds.append("ERROR")
            if show_warning:
                selected_kinds.append("WARNING")
            if include_note:
                selected_kinds.append("NOTE")

            filtered_issues = []
            for issue in issues:
                if selected_category != t["filter_all"] and issue.category != selected_category:
                    continue
                if issue.severity not in selected_severity:
                    continue
                if issue.kind not in selected_kinds:
                    continue
                filtered_issues.append(issue)

            if not filtered_issues:
                note_exists = any(issue.kind == "NOTE" for issue in issues)
                if note_exists and not include_note:
                    st.info(f"{t['no_issues']} {t['note_hint']}")
                else:
                    _render_empty_state(t["no_issues"])
            else:
                table_rows = []
                for issue in filtered_issues:
                    text = issue.i18n.en if lang == "en" else issue.i18n.ko
                    page_type_value = issue.page_type or page_type_map.get(
                        issue.location.page
                    )
                    table_rows.append(
                        {
                            t["table_severity"]: _severity_label(
                                issue.severity, lang, show_raw
                            ),
                            t["table_kind"]: _table_label(
                                issue.kind, "kind", lang, show_raw
                            ),
                            t["table_subtype"]: _table_label(
                                issue.subtype, "subtype", lang, show_raw
                            ),
                            t["table_page_type"]: _table_label(
                                page_type_value, "page_type", lang, show_raw
                            ),
                            t["table_page"]: issue.location.page,
                            t["table_message"]: text.message,
                        }
                    )
                st.markdown(f"### {t['results_title']}")
                st.caption(t["results_caption"])

                table_height = min(360, 36 * (len(table_rows) + 1))
                st.dataframe(
                    table_rows,
                    
                    hide_index=True,
                    height=table_height,
                    column_config={
                        t["table_page"]: st.column_config.NumberColumn(width="small"),
                        t["table_kind"]: st.column_config.TextColumn(width="small"),
                        t["table_subtype"]: st.column_config.TextColumn(width="small"),
                        t["table_page_type"]: st.column_config.TextColumn(width="small"),
                        t["table_message"]: st.column_config.TextColumn(width="large"),
                    },
                )

                for issue in filtered_issues:
                    level_label = _label_for(issue.kind, "kind", lang)
                    detail_label = _issue_detail_label(issue, lang)
                    summary_text = _issue_summary(issue, lang)
                    impact_text = _issue_impact(issue, lang)
                    action_text = _issue_action(issue, lang)
                    header = (
                        f"{t['page_label']} {issue.location.page} · "
                        f"{level_label} · {detail_label}"
                    )
                    with st.container(border=True):
                        st.markdown(f"**{header}**")
                        st.caption(
                            f"{t['table_severity']}: "
                            f"{_severity_label(issue.severity, lang, show_raw)}"
                        )
                        st.markdown(f"**{t['issue_summary_label']}** {summary_text}")
                        st.markdown(f"**{t['issue_impact_label']}** {impact_text}")
                        st.markdown(f"**{t['issue_action_label']}** {action_text}")
                        ai_explanations = st.session_state.get("ai_explanations") or {}
                        if ai_explain_enabled and ai_explanations.get(issue.id):
                            ai_item = ai_explanations.get(issue.id, {})
                            ai_lang = "en" if lang == "en" else "ko"
                            ai_text = ai_item.get(ai_lang, {})
                            if ai_text:
                                st.markdown(f"**{t['ai_explain_title']}**")
                                st.markdown(
                                    f"- {t['ai_why_label']}: {ai_text.get('why', '')}"
                                )
                                st.markdown(
                                    f"- {t['ai_impact_label']}: {ai_text.get('impact', '')}"
                                )
                                st.markdown(
                                    f"- {t['ai_action_label']}: {ai_text.get('action', '')}"
                                )
                    st.markdown("<div class='card-spacer'></div>", unsafe_allow_html=True)

                    with st.expander(t["issue_details_label"]):
                        st.write(issue.evidence)
                        if issue.category == "redundancy" and issue.matched_to is not None:
                            st.write(
                                t["matched_to_sentence"].format(
                                    page=issue.matched_to.page,
                                    snippet=issue.matched_to.snippet,
                                )
                            )
                        if issue.similarity is not None:
                            st.write(
                                f"{t['similarity_label']}: {issue.similarity:.2f}"
                            )
                        page_type = issue.page_type or page_type_map.get(issue.location.page)
                        page_conf = issue.page_type_confidence
                        if page_type:
                            page_type_display = _format_value(
                                page_type, "page_type", lang, show_raw
                            )
                            if page_conf is not None:
                                st.write(
                                    f"{t['page_type_label']}: {page_type_display} "
                                    f"({t['page_type_confidence_label']} {page_conf:.2f})"
                                )
                            else:
                                st.write(f"{t['page_type_label']}: {page_type_display}")
                        if show_raw:
                            raw_page_type = page_type if page_type else "-"
                            raw_subtype = issue.subtype if issue.subtype else "-"
                            st.write(
                                f"raw: kind={issue.kind}, subtype={raw_subtype}, "
                                f"page_type={raw_page_type}"
                            )

                if ai_review_enabled:
                    ai_candidates = st.session_state.get("ai_candidates") or []
                    ai_limit = _ai_candidate_limit(
                        normalized_pages, meta.scan_level
                    )
                    st.caption(t["ai_review_limit_note"].format(limit=ai_limit))
                    if ai_candidates:
                        st.markdown(f"### {t['ai_review_title']}")
                        st.caption(t["ai_review_caption"])
                        ai_rows = []
                        for candidate in ai_candidates:
                            ai_rows.append(
                                {
                                    t["table_page"]: candidate.get("page"),
                                    t["table_category"]: _category_label(
                                        candidate.get("category"), lang
                                    ),
                                    t["table_subtype"]: _table_label(
                                        candidate.get("subtype"), "subtype", lang, show_raw
                                    ),
                                    t["table_message"]: candidate.get("message"),
                                    t["evidence_label"]: candidate.get("evidence"),
                                }
                            )
                        st.dataframe(
                            ai_rows,
                            
                            hide_index=True,
                            column_config={
                                t["table_page"]: st.column_config.NumberColumn(width="small"),
                                t["table_subtype"]: st.column_config.TextColumn(width="small"),
                                t["table_message"]: st.column_config.TextColumn(width="large"),
                                t["evidence_label"]: st.column_config.TextColumn(width="large"),
                            },
                        )

    with diagnostics_tab:
        if report is None:
            _render_empty_state(t["no_report"])
        else:
            ai_diag = st.session_state.get("ai_diag_result") or {}
            ai_final = ai_diag.get("final") if isinstance(ai_diag, dict) else None
            if isinstance(ai_final, dict):
                st.subheader(t["ai_diag_title"])
                diag_text = (
                    ai_final.get("diagnostics_en", "")
                    if lang == "en"
                    else ai_final.get("diagnostics_ko", "")
                )
                if diag_text:
                    st.write(diag_text)
                notes_text = (
                    ai_final.get("consensus_notes_en", "")
                    if lang == "en"
                    else ai_final.get("consensus_notes_ko", "")
                )
                if notes_text:
                    st.caption(f"{t['ai_diag_consensus_notes']}: {notes_text}")
                gpt_payload = ai_diag.get("gpt")
                gemini_payload = ai_diag.get("gemini")
                gpt_critique = ai_diag.get("gpt_critique")
                gemini_critique = ai_diag.get("gemini_critique")
                if gpt_payload or gemini_payload:
                    with st.expander(t["ai_diag_gpt_label"]):
                        if gpt_payload:
                            st.json(gpt_payload)
                        else:
                            st.info(t["ai_diag_unavailable"])
                    with st.expander(t["ai_diag_gemini_label"]):
                        if gemini_payload:
                            st.json(gemini_payload)
                        else:
                            st.info(t["ai_diag_unavailable"])
                if gpt_critique or gemini_critique:
                    with st.expander(t["ai_diag_critique_label"]):
                        if gpt_critique:
                            st.write("GPT")
                            st.json(gpt_critique)
                        if gemini_critique:
                            st.write("Gemini")
                            st.json(gemini_critique)
            meta = report.document_meta
            metric_items = [
                (t["page_count_label"], meta.page_count),
                (t["textless_pages_label"], meta.textless_pages),
                (t["scan_like_ratio_label"], round(meta.scan_like_ratio, 2)),
                (t["scan_level_label"], meta.scan_level),
                (t["raw_char_count_label"], meta.raw_char_count),
                (t["normalized_char_count_label"], meta.normalized_char_count),
                (t["raw_score_label"], report.raw_score),
            ]
            for row_start in range(0, len(metric_items), 3):
                row_cols = st.columns(3, gap="small")
                for offset, col in enumerate(row_cols):
                    idx = row_start + offset
                    if idx >= len(metric_items):
                        continue
                    label, value = metric_items[idx]
                    col.metric(label, value)
            if report.limitations:
                st.write({t["limitations_label"]: report.limitations})

            st.subheader(t["profile_title"])
            st.write(
                {
                    t["profile_type_label"]: _label_for(
                        meta.document_profile.type, "page_type", lang
                    ),
                    t["dominant_type_label"]: _label_for(
                        meta.document_profile.dominant_type, "page_type", lang
                    ),
                    t["profile_confidence_label"]: round(meta.document_profile.confidence, 2),
                    t["profile_signals_label"]: meta.document_profile.signals,
                }
            )

            if meta.page_profiles:
                st.subheader(t["page_profiles_title"])
                profile_rows = [
                    {
                        t["page_label"]: profile.page,
                        t["profile_type_label"]: _label_for(
                            profile.type, "page_type", lang
                        ),
                        t["profile_confidence_label"]: round(profile.confidence, 2),
                        t["consent_score_label"]: (
                            profile.consent_score if profile.consent_score is not None else 0
                        ),
                        t["resume_score_label"]: (
                            profile.resume_score if profile.resume_score is not None else 0
                        ),
                        t["page_signals_label"]: profile.signals[:3],
                    }
                    for profile in meta.page_profiles
                ]
                st.dataframe(profile_rows, hide_index=True)

            page_char_counts = st.session_state.get("page_char_counts") or []
            if page_char_counts:
                st.markdown(
                    f"<div class='char-rank-title'>{t['char_rank_title']}</div>",
                    unsafe_allow_html=True,
                )
                sorted_counts = sorted(
                    page_char_counts,
                    key=lambda item: item["char_count"],
                    reverse=True,
                )
                rows = [
                    {
                        t["page_label"]: item["page"],
                        t["char_count_label"]: item["char_count"],
                    }
                    for item in sorted_counts[:5]
                ]
                st.dataframe(rows, hide_index=True)

    with qa_tab:
        st.subheader(t["qa_title"])
        if report is None:
            st.info(t["qa_need_analysis"])
        elif not ai_available:
            st.info(t["qa_need_key"])
        else:
            st.markdown("<span id='qa-panel-marker'></span>", unsafe_allow_html=True)
            max_q_len = 150
            rag_status_placeholder = st.empty()
            question = st.text_area(
                t["qa_question_label"],
                placeholder=t["qa_question_placeholder"],
                key="rag_question_input",
                max_chars=max_q_len,
                height=120,
                disabled=st.session_state["rag_running"],
            )
            ask_clicked = st.button(
                t["qa_ask_button"],
                
                disabled=st.session_state["rag_running"],
            )
            if ask_clicked:
                st.session_state["rag_running"] = True
                st.session_state["rag_status"] = None
                st.session_state["rag_error"] = None
                if not question.strip():
                    st.session_state["rag_status"] = "empty"
                    st.session_state["rag_running"] = False
                    rag_status_placeholder.empty()
                else:
                    now = time.time()
                    elapsed = now - st.session_state["last_rag_run_ts"]
                    if QA_COOLDOWN_SECONDS > 0 and elapsed < QA_COOLDOWN_SECONDS:
                        remaining = max(1, int(QA_COOLDOWN_SECONDS - elapsed + 0.999))
                        st.session_state["rag_status"] = "cooldown"
                        st.session_state["rag_error"] = f"cooldown_{remaining}"
                        st.session_state["rag_running"] = False
                        rag_status_placeholder.empty()
                    else:
                        rag_status_placeholder.markdown(
                            _rag_processing_html(t["qa_processing_search"]),
                            unsafe_allow_html=True,
                        )
                        if not st.session_state["file_hash"] and uploaded_file is not None:
                            st.session_state["file_hash"] = hashlib.sha256(
                                uploaded_file.getvalue()
                            ).hexdigest()[:12]
                        embedding_provider = st.session_state.get("embedding_provider") or "OpenAI"
                        rag_key = _rag_cache_key(
                            st.session_state["file_hash"], lang, embedding_provider
                        )
                        rag_state = st.session_state["rag_index_cache"].get(rag_key)
                        owner_key = _rag_owner_key(
                            st.session_state.get("username"),
                            st.session_state["file_hash"],
                            lang,
                            embedding_provider,
                        )
                        if rag_state is None:
                            client = OpenAIClient(embedding_provider=embedding_provider)
                            source_name = (
                                uploaded_file.name
                                if uploaded_file is not None
                                else report.document_meta.file_name
                            )
                            rag_collection = _build_chroma_index(
                                client,
                                normalized_pages,
                                owner_key,
                                source_name,
                                st.session_state["file_hash"],
                                lang,
                                embedding_provider,
                                st.session_state.get("username"),
                            )
                            if rag_collection is None:
                                st.session_state["rag_status"] = "error"
                                st.session_state["rag_error"] = client.last_error or "rag_index_failed"
                            else:
                                rag_state = {"owner_key": owner_key}
                                st.session_state["rag_index_cache"][rag_key] = rag_state
                        if rag_state is not None:
                            client = OpenAIClient(embedding_provider=embedding_provider)
                            rag_collection = _get_chroma_collection()
                            top_k = _rag_top_k(
                                normalized_pages, report.document_meta.scan_level
                            )
                            def _rag_status(stage: str) -> None:
                                if stage == "rewrite":
                                    rag_status_placeholder.markdown(
                                        _rag_processing_html(
                                            t["qa_processing_rewrite"]
                                        ),
                                        unsafe_allow_html=True,
                                    )
                                elif stage == "search":
                                    rag_status_placeholder.markdown(
                                        _rag_processing_html(
                                            t["qa_processing_search"]
                                        ),
                                        unsafe_allow_html=True,
                                    )
                                elif stage == "answer":
                                    rag_status_placeholder.markdown(
                                        _rag_processing_html(
                                            t["qa_processing_answer"]
                                        ),
                                        unsafe_allow_html=True,
                                    )
                            result = _run_rag_qa(
                                client,
                                question,
                                normalized_pages,
                                rag_collection,
                                owner_key,
                                top_k,
                                lang,
                                report.document_meta.scan_level,
                                status_callback=_rag_status,
                                file_hash=st.session_state["file_hash"],
                                embedding_provider=embedding_provider,
                                is_admin=(st.session_state["role"] == "admin"),
                            )
                            st.session_state["rag_last_question"] = question
                            st.session_state["rag_last_result"] = result
                            st.session_state["rag_error"] = client.last_error
                            if result and result.get("answer"):
                                st.session_state["rag_status"] = "ok"
                            else:
                                st.session_state["rag_status"] = (
                                    "error" if client.last_error else "empty"
                                )
                            st.session_state["last_rag_run_ts"] = time.time()
                        st.session_state["rag_running"] = False
                rag_status_placeholder.empty()

            rag_status = st.session_state.get("rag_status")
            rag_error = st.session_state.get("rag_error")
            if rag_status == "cooldown" and rag_error:
                seconds = rag_error.replace("cooldown_", "")
                st.info(t["qa_cooldown"].format(seconds=seconds))
            elif rag_status == "error" and rag_error:
                message = _ai_error_message(rag_error, lang) or t["qa_empty"]
                st.warning(message)

            rag_result = st.session_state.get("rag_last_result")
            if rag_result and rag_result.get("answer"):
                answer = rag_result.get("answer", {})
                answer_text = (
                    answer.get("en", "")
                    if lang == "en"
                    else answer.get("ko", "")
                )
                if answer_text:
                    with st.container(border=True):
                        st.markdown(f"**{t['qa_answer_title']}**")
                        st.write(answer_text)
                citations = rag_result.get("citations") or []
                if citations:
                    with st.container(border=True):
                        st.markdown(f"**{t['qa_citations_title']}**")
                        for cite in citations:
                            page = cite.get("page")
                            snippet = cite.get("snippet")
                            if not page or not snippet:
                                continue
                            with st.expander(f"p{page}"):
                                st.write(snippet)
                else:
                    st.info(t["qa_no_citations"])
            elif rag_status == "empty":
                st.info(t["qa_empty"])

    with download_tab:
        if report is None:
            _render_empty_state(t["no_report"])
        else:
            st.write(t["download_help"])
            json_payload = json.dumps(report.model_dump(), indent=2, ensure_ascii=False)
            st.download_button(
                t["download_button"],
                data=json_payload,
                file_name="report.json",
                mime="application/json",
            )
            ai_explanations = st.session_state.get("ai_explanations")
            ai_candidates = st.session_state.get("ai_candidates")
            ai_status = st.session_state.get("ai_status") or {}
            ai_errors = st.session_state.get("ai_errors") or {}
            rag_result = st.session_state.get("rag_last_result")
            if ai_available and (ai_explain_enabled or ai_review_enabled or rag_result):
                explain_status = ai_status.get("explain")
                review_status = ai_status.get("review")
                explain_error = ai_errors.get("explain")
                review_error = ai_errors.get("review")
                explain_error_msg = _ai_error_message(explain_error, lang)
                review_error_msg = _ai_error_message(review_error, lang)
                if ai_explain_enabled:
                    if explain_status == "cooldown" and explain_error:
                        st.info(explain_error_msg)
                    elif explain_status == "error" and explain_error:
                        st.warning(
                            f"{t['ai_explain_error_prefix']}: {explain_error_msg} ({explain_error})"
                        )
                    elif explain_status == "empty":
                        st.info(t["ai_explain_empty"])
                if ai_review_enabled:
                    if review_status == "cooldown" and review_error:
                        st.info(review_error_msg)
                    elif review_status == "error" and review_error:
                        st.warning(
                            f"{t['ai_review_error_prefix']}: {review_error_msg} ({review_error})"
                        )
                    elif review_status == "empty":
                        st.info(t["ai_review_empty"])

                st.write(t["download_ai_help"])
                ai_payload = json.dumps(
                    {
                        "enabled": {
                            "explain": bool(ai_explain_enabled),
                            "review": bool(ai_review_enabled),
                        },
                        "status": {
                            "explain": explain_status or "empty",
                            "review": review_status or "empty",
                        },
                        "errors": {
                            "explain": (
                                f"{explain_error_msg} ({explain_error})"
                                if explain_error
                                else None
                            ),
                            "review": (
                                f"{review_error_msg} ({review_error})"
                                if review_error
                                else None
                            ),
                        },
                        "ai_explanations": ai_explanations or {},
                        "ai_candidates": ai_candidates or [],
                        "ai_diagnosis": st.session_state.get("ai_diag_result"),
                        "ai_diagnosis_status": st.session_state.get("ai_diag_status"),
                        "ai_diagnosis_errors": st.session_state.get("ai_diag_errors"),
                        "rag": {
                            "question": st.session_state.get("rag_last_question"),
                            "answer": (
                                st.session_state.get("rag_last_result", {}) or {}
                            ).get("answer"),
                            "citations": (
                                st.session_state.get("rag_last_result", {}) or {}
                            ).get("citations", []),
                            "status": st.session_state.get("rag_status"),
                            "error": st.session_state.get("rag_error"),
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                st.download_button(
                    t["download_ai_button"],
                    data=ai_payload,
                    file_name="report_ai.json",
                    mime="application/json",
                )

    with history_tab:
        st.header("최근 분석 이력" if lang == "ko" else "Recent Analysis History")
        try:
            history_items = db_manager.get_user_history(
                st.session_state["username"], 
                is_admin=(st.session_state["role"] == "admin"), 
                limit=10
            )
            if not history_items:
                st.info("이력이 없습니다." if lang == "ko" else "No history found.")
            else:
                for item in history_items:
                    with st.expander(f"{item['created_at']} - {item['filename']}"):
                        # Lazy fetch detail for buttons
                        detail = db_manager.get_history_detail(item["id"])
                        
                        if detail:
                            # 1. Download Buttons (if Optim result)
                            rewritten_text = detail.get("rewritten_text")
                            if rewritten_text:
                                from documind.utils.export import create_txt_bytes, create_pdf_bytes
                                h_col1, h_col2 = st.columns(2)
                                valid_id = item["id"]
                                fname_base = f"optim_{valid_id}"
                                
                                with h_col1:
                                    st.download_button(
                                        label="💾 Download TXT",
                                        data=create_txt_bytes(rewritten_text),
                                        file_name=f"{fname_base}.txt",
                                        mime="text/plain",
                                        key=f"hist_dl_txt_{valid_id}"
                                    )
                                with h_col2:
                                    st.download_button(
                                        label="📄 Download PDF",
                                        data=create_pdf_bytes(rewritten_text),
                                        file_name=f"{fname_base}.pdf",
                                        mime="application/pdf",
                                        key=f"hist_dl_pdf_{valid_id}"
                                    )
                            
                            # 2. Load Report Button (Existing Logic)
                            if st.button("Load Report Context", key=f"load_hist_{item['id']}"):
                                # We need to handle both Report objects and Optim results
                                # Optim results are dicts, Quality reports are Report objects.
                                # Current logic assumed Quality Report only.
                                # Let's try to restore 'report' if it exists
                                if "document_meta" in detail: # Likely a Report object dict
                                    st.session_state["report"] = Report(**detail)
                                    st.success("Analysis Report Loaded!")
                                elif "rewritten_text" in detail: # Optim result
                                    # Restore optim state
                                    st.session_state["optim_result"] = detail
                                    st.success("Optimization Result Loaded via 'Analyze' Tab!")
                                st.rerun()
                        else:
                            st.warning("Failed to load details.")
        except Exception as e:
            st.error(f"Error loading history: {e}")

    with help_tab:
        st.markdown(t["help_content"])
