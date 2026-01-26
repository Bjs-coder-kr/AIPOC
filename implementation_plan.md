# 타겟 최적화 고도화 계획 (Final Comprehensive Plan)

사용자의 피드백을 전면 수용하여 상세하고 구체적으로 보강된 최종 실행 계획입니다.

## 1. 스펙 확정 (Specifications)

### 1.1 데이터 스키마 (ChromaDB)
*   **Collection Name**: `optim_best_practices`
*   **Fields**:
    *   `document`: 최적화된 결과 텍스트 (Rewritten Text) - *임베딩 대상 아님*
    *   `metadata`:
        *   `original_text`: 원문 (변경 전) - *검색 Query 및 유사도 측정 대상*
        *   `score`: 품질 점수 (Integer, 0-100)
        *   `target_level`: 타겟 페르소나 (public, student, expert) - *1차 필터*
        *   `keywords`: 주요 키워드 (Comma separated string)
        *   `timestamp`: 생성 일시 (ISO Format)
        *   `model_version`: 사용된 모델 버전 (예: "gpt-4o-2024-08")
    *   `id`: UUID (v4)

### 1.2 정책 및 기준 (Policies)
*   **저장 기준 (Archive)**:
    *   점수 **95점 이상**.
    *   *편향 방지*: 도메인/타겟별 분포를 주기적으로 확인하여 특정 주제만 쌓이지 않도록 모니터링 (운영 단계).
*   **재활용 기준 (Recall)**:
    *   검색된 유사 사례 중 점수 **92점 이상**인 것만 프롬프트에 주입.
    *   검색 결과가 없거나 점수가 낮으면 프롬프트에 예시를 넣지 않음.
    *   **하이브리드 검색 전략**: 코사인 유사도 검색 + 키워드 매칭(BM25) 병행하여 검색 정확도 향상.
    *   **Fallback 전략**: `target_level` 필터로 검색 결과가 0건일 경우, 필터를 완화하여 다른 레벨의 우수 사례도 참고 가능하도록 함.
*   **사용자 확인 (Interactive)**: 점수 **85점 ~ 89점** 구간. (User Acceptance Required)
*   **자동 통과 (Auto-Pass)**: 90점 이상.
*   **최대 시도 (Max Retries)**: **1차 5회** + **2차 3회** (조건부).
*   **최고 점수 선택 (Best-of-N) 및 추가 재시도 정책**:
    *   5회 시도 후 `max(score) >= 75`인 경우: 해당 결과 채택 (동점 시 최신 시도 우선).
    *   5회 시도 후 `max(score) < 75`인 경우: **추가 3회 재시도** 진행.
        *   추가 3회 시도 조건은 1차 5회와 동일 (RAG 검색, 우수 점수 자동 저장 등).
        *   추가 3회 후에도 75점 미만이면 `max(score)` 중 최고점 선택 (품질 경고 표시).
    *   동점일 경우 항상 **가장 최신 시도(Latest Attempt)** 우선.

### 1.3 프롬프트 정책
*   **Few-shot 삽입**: 검색된 우수 사례 중 **최대 2개**만 사용. (Token Budget 및 혼란 방지)
*   **토큰 및 길이 제한**:
    *   `MAX_EXAMPLE_TOKENS = 1000`: 예시 블록 전체의 토큰 상한.
    *   `MAX_SINGLE_EXAMPLE_LENGTH = 500`: 개별 예시의 최대 문자 길이.
    *   예시가 상한을 초과할 경우: 후반부 truncate 또는 해당 예시 제외 (짧은 예시 우선 선택).
*   **형식**: "지침"과 "예시"를 명확히 분리하여 주입.
    ```text
    [Optimized Examples for Reference]
    Do not copy the content, but follow the style and tone.

    <example_1>
    Original: ...
    Rewritten: ...
    </example_1>
    ```

---

## 2. RAG 저장/검색 구현 (Archive & Retrieval)

### 2.1 매니저 모듈 신설 (`documind/utils/best_practice_manager.py`)
*   `archive_best_practice(result_dict)`:
    *   필수 필드 검증 후 저장.
    *   **개인정보 보호**: 저장 전 전화번호, 이메일 등 PII(Personally Identifiable Information) 패턴이 감지되면 마스킹(`***`) 처리 옵션 적용.
    *   **컬렉션 버전 관리**: 컬렉션 이름에 버전 suffix 추가 (예: `optim_best_practices_v1`).
    *   스키마 변경 시 마이그레이션 스크립트 작성 필요.
*   `retrieve_best_practices(original_text, target_level, n=3)`:
    *   **1차 필터**: `target_level` 일치 여부 (Strict Filtering).
    *   **2차 검색 (하이브리드)**:
        *   코사인 유사도 검색 (임베딩 기반).
        *   키워드 매칭 (BM25 알고리즘 병행).
        *   두 결과를 조합하여 최종 후보 선정.
    *   **3차 정렬**: `score` 높은 순 정렬 -> 상위 2개(`n`) 반환.
    *   **Fallback**: `target_level` 필터 결과가 0건이면, 필터 완화하여 다른 레벨 검색.
*   **성능 최적화**:
    *   **임베딩 캐시**: `lru_cache`를 적용하여 동일한 텍스트에 대한 중복 임베딩 연산 방지.
    *   **Lazy Loading**: ChromaDB 클라이언트는 최초 요청 시에만 초기화.
*   **동시성 처리**:
    *   Streamlit 멀티유저 환경 대비 `threading.Lock` 또는 connection pool 사용.
    *   또는 각 요청마다 새 클라이언트 인스턴스 생성 (성능 trade-off 고려).

---

## 3. 인터랙티브 오케스트레이션 (Interactive Orchestration)

### 3.1 Orchestrator 개편 (`documind/actor_critic/orchestrator.py`)
*   **Generator Pattern 도입**: `generate_with_critic_loop`를 제너레이터 함수로 변경.
*   **상태 모델 (State Model)**:
    ```python
    @dataclass
    class OptimizerState:
        attempt: int        # 현재 시도 횟수
        max_retries: int    # 최대 횟수
        current_score: int  # 현재 점수 (Score)
        current_text: str   # 현재 결과물 (Draft)
        feedback: str       # 비평 내용 (Critique)
        status: str         # "PASS", "FAIL", "WAIT_CONFIRM"
        decision_required: bool # True if user input needed
    ```
*   **API 계약 (Interface Contract)**:
    *   `next()`: 다음 시도 진행. `OptimizerState` 반환.
    *   `send('accept')`: 현재 결과로 확정하고 루프 종료.
    *   `send('retry')`: 비평 반영하여 재시도.

---

## 4. UI 갱신 (`documind/app/views/analy_app.py`)

*   **상태 표시 UI**: 진행 단계(Progress)와 현재 점수를 시각적으로 표시 (예: 게이지 바).
*   **이벤트 핸들링**:
    *   `WAIT_CONFIRM` 상태일 때만 [Accept] / [Retry] 버튼 활성화.
    *   버튼 클릭 시 `st.session_state`를 업데이트하고 `st.rerun()`을 통해 오케스트레이터의 다음 `step` 실행.

---

## 5. 운영 및 품질 관리 (Operations & QA)

### 5.1 테스트 계획
*   **회귀 테스트 (Regression Test)**: RAG 기능을 켰을 때와 껐을 때의 평균 점수 및 소요 시간 비교 측정.
*   **분기 테스트**: 인터랙티브 모드에서 Accept/Retry 선택 시 로직이 정확히 분기되는지 단위 테스트.
*   **시나리오 기반 테스트 케이스**:
    *   **TC1**: 첫 시도 96점 → 자동 저장 및 통과.
    *   **TC2**: 1차 88점(대기) → Accept 선택 → 저장 안 됨, 해당 결과 채택.
    *   **TC3**: 1~4차 실패, 5차 92점 → 5차 결과 채택 및 자동 저장.
    *   **TC4**: 1차~5차 모두 75점 미만 → 추가 3회 재시도 진행 확인.
    *   **TC5**: 추가 3회 후에도 75점 미만 → `max(score)` 선택 및 품질 경고 표시.
    *   **TC6**: RAG 검색 결과 0건 → 프롬프트에 예시 없음 확인.
    *   **TC7**: 동일 `original_text` 2회 요청 → 임베딩 캐시 hit 확인.
    *   **TC8**: `target_level` 필터 결과 0건 → Fallback으로 다른 레벨 검색 확인.

### 5.2 품질 유지
*   **데이터 검증**: 저장된 우수 사례가 실제로 우수한지 관리자(Admin)가 주기적으로 검토할 수 있는 'Best Practice 뷰어' 기능(SQLite 탐색기 확장) 고려.
*   **설정값 관리**: `CHECK_THRESHOLD` (85), `PASS_THRESHOLD` (90), `ARCHIVE_THRESHOLD` (95) 등 주요 파라미터는 `analy_app.py` 상단 상수(Constants) 또는 별도 설정 파일로 분리하여 튜닝 용이하게 함.

---

## 6. 작업 진행 순서
1.  **DB API 구현**: `best_practice_manager.py` 작성 (ChromaDB 컬렉션/스키마 정의).
2.  **Orchestrator 리팩토링**: Generator 기반 인터랙티브 구조로 변경.
3.  **Optimizer 연동**: RAG 검색 로직 및 프롬프트 주입 연결.
4.  **UI 구현**: Streamlit 상태 관리 및 제어 버튼 연동.
5.  **검증**: 시나리오별(96점 저장, 88점 대기, 92점 2회차 통과 등) 동작 확인.

---

## 7. 리뷰 지적사항 및 해결 제안 (Review Findings & Fixes)

아래는 본 계획에 대해 지적된 위험 요소와 대응 방안을 명시적으로 정리한 항목입니다. 다른 LLM이 평가하기 쉽게 “문제 → 해결” 구조로 작성했습니다.

| 지적사항 (Issue) | 해결 제안 (Fix) |
| --- | --- |
| **저장 기준(95점 이상)만으로는 데이터 편향/누적 오염 위험** | 도메인/타겟 분포 모니터링, 중복 제거(유사도 0.95+ 저장 생략), 관리자 검토 뷰어 도입 |
| **RAG 검색 규칙 불명확** | `target_level` 1차 필터 → 하이브리드 검색 (코사인 유사도 + BM25 키워드 매칭) → 점수 기반 정렬 |
| **RAG 검색 결과 부족 시 대응 부재** | `target_level` 필터 결과 0건 시 Fallback 전략: 다른 레벨의 우수 사례도 참고 가능하도록 필터 완화 |
| **프롬프트 주입 위치/규칙 미정** | Few-shot 최대 2개, 예시 블록을 지침과 분리, `MAX_EXAMPLE_TOKENS=1000`, `MAX_SINGLE_EXAMPLE_LENGTH=500` 명시, 초과 시 truncate 또는 제외 |
| **인터랙티브 루프의 API 계약 부재** | `OptimizerState` 상태 모델 정의, `accept/retry` 이벤트 계약, `WAIT_CONFIRM` 플로우 정의 |
| **Best-of 선택 기준 불명확 및 품질 보장 부족** | 5회 시도 후 `max(score) >= 75` 시 채택, `< 75` 시 추가 3회 재시도 (동일 조건), 최종적으로도 75 미만이면 `max(score)` 선택하되 품질 경고 표시. 동점 시 최신 시도 우선 |
| **성능/비용 리스크 고려 부족** | 임베딩 캐시(`lru_cache`), Chroma 클라이언트 Lazy Loading, 검색 결과가 없을 때 프롬프트 주입 생략 |
| **동시성 처리 미흡 (멀티유저 환경)** | `threading.Lock` 또는 connection pool 사용, 또는 요청마다 새 클라이언트 인스턴스 생성 (성능 trade-off 고려) |
| **ChromaDB 스키마 변경 시 마이그레이션 계획 부재** | 컬렉션 이름에 버전 suffix 추가 (예: `optim_best_practices_v1`), 스키마 변경 시 마이그레이션 스크립트 작성 |
| **개인정보 노출 위험** | 저장 전 PII 마스킹 옵션 도입 (전화번호/이메일 탐지) |
| **테스트/운영 계획 부재** | RAG on/off 회귀 테스트, 인터랙티브 분기 테스트, 8개 시나리오 기반 테스트 케이스 (TC1~TC8) 정의, 점수 분포/지연 로그 수집 |
