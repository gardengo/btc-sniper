# ARCHITECTURE.md

## 1. 전체 구조

```text
Binance Historical OHLCV ──┐
                           ├─> Data Layer ─> Feature Layer
Binance Realtime WebSocket ┘                    │
                                                 ▼
                                       Forecast Model Layer
                                                 │
                        ┌────────────────────────┼──────────────────────┐
                        ▼                        ▼                      ▼
                  Point/Median              Quantiles             Baseline
                        └────────────────────────┼──────────────────────┘
                                                 ▼
                                         Forecast Store
                                                 │
                 ┌───────────────────────────────┼────────────────────────┐
                 ▼                               ▼                        ▼
          Streamlit Dashboard             Performance Log          Model Registry
                 │                               │                        │
                 └───────────────────────────────┼────────────────────────┘
                                                 ▼
                                      Weekly Model Review
                                                 │
                                       Candidate Training
                                                 │
                                      Walk-Forward Validation
                                                 │
                                    Candidate vs Production
                                                 │
                                         Promote / Reject
```

학습 cutoff는 outer test를 한 번 평가하기 전과 후가 다르다. 평가 전에는 purge
경계에서 멈추고, `jobs/final_evaluation.py`가 평가를 기록한 뒤에야 해당 설계에
한해 최신 데이터까지 학습할 수 있다 (VALIDATION_SPEC.md 4.5절). 순서를 뒤집으면
예약된 블록이 읽히지도 못한 채 사라지고, 그렇게 만들어진 모델은 겉보기에 전혀
이상하지 않다.

## 2. 주요 컴포넌트

### 2.1 Data Ingestion

- Binance REST: historical 1D OHLCV
- Binance WebSocket: realtime current price
- Coinbase: cross-exchange sanity check

### 2.2 Data Validation

검증 항목:

- timestamp monotonicity
- duplicate timestamp
- missing daily candle
- OHLC 관계 (`low <= open/close <= high`)
- volume >= 0
- non-finite values
- extreme unexplained jumps
- exchange source consistency

### 2.3 Feature Pipeline

- raw market data -> point-in-time features
- transformation metadata/version 관리
- train/test split 이후 각 fold에서 transform fitting

### 2.4 Forecast Engine

입력: 마지막 확정 일봉까지의 feature sequence

forecast는 모델 단독 출력이 아니라 모델과 baseline의 horizon별 가중 평균이다
(MODEL_SPEC.md 7.2절). 검증에서 모델이 우위를 보인 구간에만 모델 가중치가
들어간다. 각 point는 자기 출처(`model`/`blend`/`baseline`)를 함께 저장하므로
저장된 예측값은 항상 무엇이 만들었는지 추적할 수 있다.

출력:

- point/median forecast
- 7개 quantile forecast
- horizon별 예상 가격
- prediction interval

### 2.5 Evaluation Engine

각 예측에 대해 target date가 도착하면 실제값을 연결하고 평가 지표를 계산한다.

baseline, 트리 모델, 운영 예측이 모두 같은 경로(`src/evaluation/evaluator.py`)로
채점된다. 지표를 두 번 구현하면 결국 서로 다른 값을 내놓게 되고, 그러면
"candidate가 baseline보다 X만큼 낫다"가 비교가 아니라 두 개의 다른 계산이 된다.

### 2.6 Model Registry

학습은 절대 승격시키지 않는다. `register()`는 `candidate`만 받고 `production`은
`promote()`만 쓸 수 있다 (CLAUDE.md 2.3절). production 모델은 동시에 하나만
존재하며, 승격 시 기존 모델 은퇴가 같은 트랜잭션에서 일어난다. production이 둘인
순간이 있으면 일일 forecast가 어느 모델이 자기 출력을 만들었는지 말할 수 없게 된다.

outer test 지표는 model_version당 한 번만 기록할 수 있고, 두 번째 시도는
거부된다 (VALIDATION_SPEC.md 4.4절). 이 규칙을 실제로 강제할 수 있는 곳은
registry뿐이다.

승격 판단은 `src/models/promotion.py`가 내리고, registry는 그 판단을 집행한다.
둘을 나눈 이유는 게이트가 순수 함수여야 테스트할 수 있기 때문이다 — 실제
production 모델을 바꾸지 않고도 "이 숫자면 어떤 결정이 나오는가"를 물을 수 있어야
한다. 판단의 **근거**도 함께 기록된다. 거부에는 사유가 필수이고, 게이트가 실패한
검사 이름과 그 숫자를 사유로 만들어준다. 사유 없는 거부는 나중에 아무도 검토할 수
없는 결정이다.


모델 버전별:

- training cutoff
- feature version
- algorithm
- parameters
- validation metrics
- final test metrics
- production start/end
- status

를 저장한다.

## 3. 저장 구조

SQLite 권장.

```text
data/btc_forecast.db
```

논리 테이블:

- market_ohlcv
- realtime_price
- features
- forecasts
- forecast_realizations
- model_runs
- model_registry
- performance_metrics
- data_quality_checks

구현된 물리 스키마는 `src/storage/schema.sql`에 있고, 논리 테이블과의 대응은
`DATA_SPEC.md` section 8에 정리되어 있다. `forecasts`는
`forecasts` / `forecast_points` / `forecast_quantiles`로 정규화되고,
`features`는 long format으로 저장된다. 이유는 DATA_SPEC 문서에 기록되어 있다.

## 4. Streamlit 페이지

### Dashboard

- 실시간 BTC 가격
- 최근 과거 가격
- 1년 forecast graph
- 50/80/95% prediction interval
- 현재 production model 정보

### Prediction Log

- 날짜별 forecast
- horizon별 predicted / actual
- error
- interval coverage
- 평가 완료/미완료 상태

### Model Performance

- MAE / RMSE / MASE 또는 sMAPE
- direction accuracy
- quantile loss / pinball loss
- interval coverage
- regime별 성능
- model version 비교
- 12개월 forecast revision 추이

## 5. 실행 구조

초기 버전은 하나의 Python 프로젝트로 시작하되, 책임을 다음처럼 나눈다.

```text
src/
  data/        binance.py binance_stream.py coinbase.py http.py ingest.py
               realtime.py types.py validation.py
  features/    indicators.py groups.py regime.py pipeline.py
  models/      targets.py baselines.py base.py dataset.py lightgbm_model.py
               forecaster.py training.py registry.py promotion.py
  validation/  splits.py folds.py
  evaluation/  metrics.py evaluator.py baseline_eval.py walk_forward.py
               final_test.py
  storage/     schema.sql db.py repositories.py
  forecast/    horizons.py quantiles.py blending.py generate.py interpolate.py
  monitoring/  data_report.py baseline_report.py validation_report.py
               realization.py drift.py performance_report.py markdown.py
               review_report.py final_report.py
  utils/       config.py logging.py timeutils.py provenance.py
app/
  streamlit_app.py          (Phase 9)
jobs/
  update_market_data.py     구현됨
  build_features.py         구현됨
  data_quality_report.py    구현됨
  stream_realtime_price.py  구현됨
  evaluate_baselines.py     구현됨
  train_model.py            구현됨
  walk_forward.py           구현됨
  generate_forecast.py      구현됨
  evaluate_forecasts.py     구현됨
  final_evaluation.py       구현됨
  weekly_model_review.py    구현됨
tests/
```

`src/models/targets.py`는 이 저장소에서 **유일하게 미래를 참조하도록 허용된
모듈**이다. label은 정의상 `t+h` 가격을 읽어야 한다. `src/features/`,
`src/evaluation/`, `src/models/baselines.py`, `src/validation/`은 모두 인과적이어야
하며, `tests/test_leakage.py`가 AST 파싱으로 이를 강제하고 예외가 그 한 파일로
유지되는지도 함께 검사한다.

`src`는 import 가능한 패키지 루트다. job은 프로젝트 루트에서
`python -m jobs.<name>` 형태로 실행한다.
