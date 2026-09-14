# TASKS.md

진행 상태 표기:

- `[x]` 구현 + 테스트 완료
- `[~]` 부분 구현 (비고 참조)
- `[ ]` 미구현

## Phase 0. Project bootstrap

- [x] Python package structure (`src/`, `jobs/`, `tests/`, `app/`)
- [x] dependency management (`requirements.txt`, `pyproject.toml`)
- [x] config loading (`src/utils/config.py`, 타입 지정 dataclass)
- [x] logging (`src/utils/logging.py`, UTC 타임스탬프)
- [x] SQLite initialization (`src/storage/db.py`, `src/storage/schema.sql`)
- [ ] `.env` support for optional credentials — 현재 사용하는 엔드포인트가
      모두 public이라 불러올 credential이 없다. 인증이 필요한 소스를 추가할 때 구현한다.

## Phase 1. Data ingestion

- [x] Binance historical 1D BTCUSDT loader (`src/data/binance.py`)
- [x] incremental upsert (`src/data/ingest.py`, overlap 재조회 포함)
- [x] realtime Binance WebSocket price consumer (`src/data/binance_stream.py`,
      재연결 + heartbeat + 지속성 throttling + 구현성 검사)
- [x] REST fallback current-price loader (`src/data/realtime.py`,
      `get_current_price()` 해석 순서 구현)
- [x] Coinbase BTC-USD cross-check loader (`src/data/coinbase.py`)
- [x] data quality validation (`src/data/validation.py`)

## Phase 2. Feature pipeline

- [x] return features
- [x] trend features
- [x] volatility features
- [x] volume features
- [x] regime labels (`src/features/regime.py`, causal)
- [x] leakage unit tests (`tests/test_leakage.py`)
- [x] feature versioning (`features.version`, `features` 테이블 키에 포함)

## Phase 3. Baselines

- [x] no-change baseline (`src/models/baselines.py`)
- [x] drift baseline
- [x] rolling-return baseline
- [x] baseline evaluator (`src/evaluation/baseline_eval.py`,
      `jobs/evaluate_baselines.py`, 리포트 `src/monitoring/baseline_report.py`)
- [x] 결과 기록 — MODEL_SPEC.md 6.2절. `no_change`가 모든 horizon에서 기준선이다.

## Phase 4. ML model

- [x] horizon grid generation (`src/forecast/horizons.py`)
- [x] target generation (`src/models/targets.py`) — Phase 3에서 필요해 앞당겼다.
      baseline을 채점하려면 label이 있어야 한다.
- [ ] train/validation split engine
- [ ] expanding/rolling window engine
- [ ] LightGBM quantile model
- [ ] XGBoost alternative where supported
- [ ] model serialization
- [ ] model registry — 스키마(`model_registry`)만 준비됨

## Phase 5. Validation

- [ ] Walk-Forward CV — fold 생성기. purge 규칙은 `splits.py`에 준비됨
- [x] outer/final test isolation (`src/validation/splits.py`, purge 규칙 포함)
- [x] regime tagging (`src/features/regime.py`)
- [~] cycle coverage report — 데이터 품질 리포트에 연도/regime 분포 포함.
      모델 성능 분해는 Phase 5에서 추가.
- [x] point metrics (`src/evaluation/metrics.py`) — MAE/RMSE/bias/sMAPE/MASE
- [x] directional metrics — 전체/상승/하락. median이 정확히 0인 예측은
      방향 판단을 하지 않은 것으로 처리한다(`direction_calls`).
- [x] quantile metrics — quantile별 pinball loss + `pinball_mean`(CRPS 근사)
- [x] interval coverage — coverage/width/relative width + Winkler interval score
- [ ] training window comparison

## Phase 6. Forecast generation

- [ ] daily production forecast job
- [ ] quantile predictions
- [ ] price conversion from log return
- [ ] PCHIP visualization interpolation
- [ ] forecast persistence — 스키마만 준비됨

## Phase 7. Realization and monitoring

- [ ] target-date resolver
- [ ] actual price join
- [ ] forecast realization table — 스키마만 준비됨
- [ ] production metric aggregation
- [ ] performance drift detection

## Phase 8. Weekly model review

- [ ] weekly review job
- [ ] retraining trigger rules
- [ ] candidate model training
- [ ] incumbent vs candidate comparison
- [ ] promotion gate
- [ ] rejection logging

## Phase 9. Streamlit

- [ ] Dashboard
- [ ] Prediction Log
- [ ] Model Performance
- [ ] historical/future combined chart
- [ ] interval bands
- [ ] current model summary
- [ ] forecast revision chart

## Phase 10. Testing

- [x] data tests (`tests/test_validation.py`, `tests/test_storage.py`,
      `tests/test_realtime.py`)
- [x] leakage tests (`tests/test_leakage.py`) — 인과성이 필요한 모든 패키지를
      AST로 검사하고, 미래 참조 예외가 `src/models/targets.py` 하나로
      유지되는지도 검사한다.
- [x] target alignment tests (`tests/test_baselines.py`) — 가격 복원 왕복,
      마지막 h개 행 NaN, feature 행렬과의 분리
- [x] interval ordering tests (`tests/test_evaluation.py`)
- [x] forecast horizon tests (`tests/test_timeutils_and_horizons.py`)
- [ ] model reproducibility tests
- [ ] promotion gate tests
- [ ] Streamlit smoke test

## Phase 11. Containerization

- [ ] Dockerfile
- [ ] docker-compose optional
- [ ] non-root runtime
- [ ] persistent DB volume
- [ ] healthcheck
- [ ] scheduled job execution strategy

## Definition of Done

- [x] full historical backfill works — 2017-08-17 ~ 현재, 결측 0일
- [ ] daily forecast can be reproduced from a fixed cutoff
- [x] no future leakage tests fail
- [x] independent outer test is protected — split v1 동결(2024-01-01), horizon purge 강제 + `assert_no_test_contamination()` 가드
- [ ] production model is versioned
- [ ] weekly candidate cannot overwrite production without promotion gate
- [ ] Streamlit shows historical actual + future forecast interval graph
