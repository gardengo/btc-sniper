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

출력:

- point/median forecast
- 7개 quantile forecast
- horizon별 예상 가격
- prediction interval

### 2.5 Evaluation Engine

각 예측에 대해 target date가 도착하면 실제값을 연결하고 평가 지표를 계산한다.

### 2.6 Model Registry

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
  models/      (Phase 3-4)
  validation/  splits.py  (fold 생성기는 Phase 5)
  evaluation/  (Phase 5)
  storage/     schema.sql db.py repositories.py
  forecast/    horizons.py
  monitoring/  data_report.py
  utils/       config.py logging.py timeutils.py
app/
  streamlit_app.py          (Phase 9)
jobs/
  update_market_data.py     구현됨
  build_features.py         구현됨
  data_quality_report.py    구현됨
  stream_realtime_price.py  구현됨
  generate_forecast.py      (Phase 6)
  evaluate_forecasts.py     (Phase 7)
  weekly_model_review.py    (Phase 8)
tests/
```

`src`는 import 가능한 패키지 루트다. job은 프로젝트 루트에서
`python -m jobs.<name>` 형태로 실행한다.
