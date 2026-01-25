# 📋 Project 모듈 → AI POC 통합 이식 계획서

> **Version**: 1.0.0  
> **Date**: 2026-01-26  
> **Purpose**: project/module 및 frontend/services를 AI POC(AIPOC)에 혼합 이식하기 위한 구조 분석 및 계획  
> **Branch**: feature/implement-target

---

## 🎯 목표

**project 폴더**의 핵심 모듈을 **AI POC(AIPOC)**에 통합하여:
1. Target Optimizer (타겟 최적화) 기능 이식
2. Actor-Critic 피드백 루프 통합
3. LLM 프로바이더 시스템 병합
4. 분석 파이프라인(antithesis, quality, summary, target) 연동

---

## 📁 Project 폴더 상세 구조

```plaintext
/Volumes/D/projects/poc/project/
│
├── 📂 module/                      # [Core] 재사용 가능한 핵심 모듈
│   ├── __init__.py                 # 모듈 익스포트: TargetOptimizer, generate_with_critic_loop
│   ├── INTEGRATION_GUIDE.md        # 통합 가이드 문서
│   ├── requirements.txt            # 의존성 (최소화)
│   │
│   ├── 📂 actor_critic/            # ⭐ Actor-Critic 피드백 루프
│   │   ├── __init__.py             # call_critic, generate_with_critic_loop 익스포트
│   │   └── orchestrator.py         # 핵심 오케스트레이터
│   │
│   ├── 📂 target_optimizer/        # ⭐ 타겟 최적화
│   │   ├── __init__.py             # TargetOptimizer, get_persona, TargetPersona 익스포트
│   │   ├── optimizer.py            # 메인 옵티마이저 클래스
│   │   ├── personas.py             # 페르소나 정의 (PUBLIC, STUDENT, WORKER, EXPERT)
│   │   └── guardrail.py            # 타겟 가드레일
│   │
│   ├── 📂 llm/                     # LLM 프로바이더
│   │   ├── __init__.py             # call_llm, LLM_CONFIG 익스포트
│   │   ├── config.py               # LLM 설정 관리
│   │   └── providers.py            # 다중 프로바이더 (Gemini/Claude CLI, API)
│   │
│   └── 📂 utils/                   # 유틸리티
│       ├── __init__.py
│       └── json_utils.py           # JSON 파싱/복구
│
├── 📂 frontend/                    # [App] Streamlit 애플리케이션
│   ├── main.py                     # 앱 엔트리포인트
│   ├── llm_config.py               # LLM 설정 (프론트엔드용)
│   │
│   ├── 📂 services/                # 비즈니스 로직
│   │   ├── api_client.py           # 외부 API 클라이언트
│   │   ├── llm_engine.py           # LLM 엔진 래퍼
│   │   │
│   │   ├── 📂 analysis/            # 분석 파이프라인
│   │   │   ├── __init__.py
│   │   │   ├── chunker.py          # 문서 청킹
│   │   │   │
│   │   │   ├── 📂 antithesis/      # 안티테제 분석
│   │   │   │   ├── antithesis.py   # 반론 생성
│   │   │   │   └── rag.py          # RAG 검색 통합
│   │   │   │
│   │   │   ├── 📂 quality/         # 품질 검증
│   │   │   ├── 📂 summary/         # 요약 생성
│   │   │   │
│   │   │   └── 📂 target/          # ⭐ 타겟 분석 (프론트엔드용)
│   │   │       ├── target.py       # 타겟 분석 로직
│   │   │       ├── personas.py     # 페르소나 (프론트엔드 버전)
│   │   │       ├── guardrail.py    # 가드레일
│   │   │       ├── evaluator.py    # 평가기
│   │   │       └── memory.py       # 컨텍스트 메모리
│   │   │
│   │   └── 📂 llm/                 # LLM 서비스
│   │       ├── orchestrator.py     # LLM 오케스트레이터
│   │       ├── pipeline.py         # 파이프라인
│   │       ├── providers.py        # 프로바이더
│   │       └── query_analyzer.py   # 쿼리 분석
│   │
│   └── 📂 utils/                   # 유틸리티
│       ├── chroma_client.py        # ChromaDB 클라이언트
│       ├── db.py                   # DB 유틸리티
│       ├── json_utils.py           # JSON 유틸
│       ├── text_processor.py       # 텍스트 처리
│       ├── ui_helpers.py           # UI 헬퍼 (25KB)
│       └── 📂 embeddings/          # 임베딩 관련
│
├── 📂 chroma_db/                   # Vector DB 저장소
├── 📄 documind.db                  # SQLite DB
└── 🧪 test_e2e_pipeline.py         # E2E 테스트
```

---

## 🔧 핵심 모듈 API 분석

### 1. Actor-Critic (`module/actor_critic/`)

#### `orchestrator.py` - 핵심 함수

```python
def call_critic(
    provider: str, 
    target_text: str, 
    prompt_type: str, 
    prompt_factory=None, 
    persona_guide=None
) -> dict:
    """
    Critic LLM으로 텍스트 평가
    Returns: {"score": int 0~100, "feedback": str}
    """

def generate_with_critic_loop(
    actor_provider: str,
    prompt_template: str,
    context_text: str,
    context_type: str = "Summary",
    max_retries: int = None,
    progress_callback = None,
    critic_provider: str = None,
    critic_prompt_factory = None,
    persona_guide: dict = None
) -> str:
    """
    Actor-Critic 피드백 루프
    1. Actor 생성 → 2. Critic 평가 → 3. 피드백 기반 재생성
    4. 임계값 미달 시 Best-of-N 선택
    """
```

#### 동작 흐름

```
┌──────────────────────────────────────────────────────────────┐
│                   Actor-Critic Loop                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────┐      생성       ┌─────────┐                   │
│   │  Actor  │ ─────────────► │  Draft  │                   │
│   │  (LLM)  │                │         │                   │
│   └─────────┘                └────┬────┘                   │
│        ▲                          │                         │
│        │                          ▼                         │
│        │                    ┌─────────┐                     │
│   피드백 │                    │ Critic  │                     │
│   기반   │ ◄─────────────── │  (LLM)  │                     │
│   재생성 │   {score, feedback} └─────────┘                     │
│        │                          │                         │
│        │                          ▼                         │
│        │                   score ≥ 90?                      │
│        │                     /      \                       │
│        │               Yes /        \ No                    │
│        │                  ▼          ▼                      │
│        │            ┌────────┐   재시도 (max 3회)            │
│        └───────────┤ Output │                              │
│                     └────────┘                              │
│                                                              │
│   Config: max_retries=3, score_threshold=90                 │
└──────────────────────────────────────────────────────────────┘
```

---

### 2. Target Optimizer (`module/target_optimizer/`)

#### `optimizer.py` - TargetOptimizer 클래스

```python
class TargetOptimizer:
    """타겟 페르소나에 맞춰 문서 최적화"""
    
    def __init__(self, provider: str):
        """LLM 프로바이더 설정"""
    
    def analyze(
        self,
        text: str,
        target_level: str = "public",  # public/student/worker/expert
        progress_callback = None,
        critic_provider: str = None,
        chunk_size: int = 3000
    ) -> dict:
        """
        메인 분석 파이프라인
        Returns: {
            "rewritten_text": str,
            "analysis": dict,
            "keywords": list
        }
        """
    
    # 내부 메서드
    def _split_text(text, chunk_size) -> list
    def _merge_results(results) -> dict
    def _calculate_complexity(text) -> float
    def _route_strategy(score, persona) -> str
    def _execute_plan_and_solve(...) -> dict
    def _execute_direct_rewrite(...) -> dict

# 편의 함수
def generate_target_rewrite(provider, text, level, progress_callback, critic_provider) -> dict
```

#### `personas.py` - 페르소나 정의

```python
class TargetPersona(Enum):
    PUBLIC = "public"    # 일반인
    STUDENT = "student"  # 대학생
    WORKER = "worker"    # 직장인
    EXPERT = "expert"    # 전문가

PERSONA_GUIDES = {
    TargetPersona.PUBLIC: {
        "role": "일반 대중",
        "tone": "친근하고 쉬운",
        "vocabulary": "일상 어휘",
        "complexity_limit": "초등~중등",
        ...
    },
    ...
}

def get_persona(level: str) -> TargetPersona
```

#### `guardrail.py` - 타겟 가드레일

```python
class TargetGuardrail:
    """타겟 최적화 결과 검증"""
    
    def validate(source: str, target: str, anchors: list) -> dict:
        """
        검증 항목:
        - Anchor 보존 (고유명사, 숫자)
        - 의미 유사도
        - 환각 여부
        """
```

---

### 3. LLM 프로바이더 (`module/llm/`)

#### `providers.py` - 다중 프로바이더 지원

```python
SUPPORTED_PROVIDERS = [
    "Gemini CLI",
    "Claude CLI", 
    "Codex",
    "Gemini API",
    "Claude API",
    "OpenAI API"
]

def call_llm(provider: str, prompt: str) -> str:
    """통합 LLM 호출 인터페이스"""
    
def call_gemini_cli(prompt: str) -> str
def call_claude_cli(prompt: str) -> str
def call_gemini_api(prompt: str) -> str
def call_claude_api(prompt: str) -> str
def call_openai_api(prompt: str) -> str
```

#### `config.py` - 설정 관리

```python
LLM_CONFIG = {
    "analysis": {
        "max_retries": 3,
        "score_threshold": 90,
        "default_provider": "Gemini CLI"
    },
    "target_optimizer": {
        "chunk_size": 3000,
        "complexity_threshold": {...}
    }
}
```

---

## 🔄 AI POC 대응 구조 비교

| Project 모듈 | AI POC 대응 | 통합 방안 |
|-------------|-------------|-----------|
| `module/actor_critic/` | `AIPOC/actor_critic/` | 직접 병합 가능 |
| `module/target_optimizer/` | `AIPOC/target_optimizer/` | 확장 이식 필요 |
| `module/llm/` | `AIPOC/llm/` | 프로바이더 병합 |
| `frontend/services/analysis/target/` | `AIPOC/target_optimizer/` | 기능 통합 |
| `frontend/services/llm/` | `AIPOC/llm/` | API 클라이언트 추가 |
| `frontend/utils/` | `AIPOC/utils/` | 유틸리티 병합 |

---

## 📝 통합 계획

### Phase 1: 핵심 모듈 이식

```
[ ] 1. module/actor_critic/orchestrator.py → AIPOC/actor_critic/
    - generate_with_critic_loop 함수 병합
    - persona_guide 파라미터 지원 확인
    
[ ] 2. module/target_optimizer/ → AIPOC/target_optimizer/
    - TargetOptimizer 클래스 이식
    - personas.py 4개 페르소나 적용
    - guardrail.py 통합
    
[ ] 3. module/llm/ → AIPOC/llm/
    - 프로바이더 목록 병합
    - config.py 설정 통합
```

### Phase 2: 프론트엔드 서비스 연동

```
[ ] 4. frontend/services/analysis/target/ 검토
    - target.py 로직 분석
    - memory.py 활용 여부 결정
    
[ ] 5. frontend/services/llm/ 참조
    - pipeline.py 패턴 참고
    - orchestrator.py 통합 패턴
```

### Phase 3: 유틸리티 및 테스트

```
[ ] 6. utils 병합
    - json_utils.py 중복 제거
    - chroma_client.py 통합
    
[ ] 7. 테스트 작성
    - E2E 파이프라인 테스트
    - 페르소나별 최적화 테스트
```

---

## ⚠️ 주의사항

### 충돌 가능 영역

1. **페르소나 정의 차이**
   - `module/target_optimizer/personas.py` vs `frontend/services/analysis/target/personas.py`
   - STRING_TO_ENUM 매핑 확인 필요

2. **LLM 설정 중복**
   - `module/llm/config.py` vs `frontend/llm_config.py`
   - 설정 통합 필요

3. **가드레일 구현**
   - `module/target_optimizer/guardrail.py` vs `frontend/services/analysis/target/guardrail.py`
   - Anchor 추출 로직 비교

### 의존성 확인

```
module/actor_critic/
└── depends on: module/llm, module/utils

module/target_optimizer/
└── depends on: module/llm, module/actor_critic
```

---

## 📊 파일 크기 참고

| 파일 | 크기 | 복잡도 |
|------|------|--------|
| `module/target_optimizer/optimizer.py` | 12.6KB | 높음 |
| `module/llm/providers.py` | 8.1KB | 중간 |
| `module/actor_critic/orchestrator.py` | 7.1KB | 중간 |
| `frontend/services/analysis/target/target.py` | 13.4KB | 높음 |
| `frontend/utils/ui_helpers.py` | 25.5KB | 높음 |

---

> **Note**: 이 문서는 project 폴더를 AI POC에 통합 이식하기 위한 계획 문서입니다.
> 실제 작업 시 각 Phase별로 상세 검토가 필요합니다.
