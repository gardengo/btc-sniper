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
btc_forecast.db
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
  data/
  features/
  models/
  validation/
  evaluation/
  storage/
  forecast/
  monitoring/
  utils/
app/
  streamlit_app.py
jobs/
  update_market_data.py
  generate_forecast.py
  evaluate_forecasts.py
  weekly_model_review.py
```
