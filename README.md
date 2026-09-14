# BTC Sniper

비트코인(BTC)의 과거 시장 데이터를 이용해 향후 1년의 가격 경로를 예측하고, 과거 실제 가격과 미래 예측 분포를 하나의 그래프로 연결해 보여주는 시계열 예측·모니터링 프로젝트.

## 핵심 목표

- 실시간 BTC 현재가 표시
- 일 단위 기준으로 미래 1년 가격 경로 예측
- 미래 예측은 단일 가격선이 아니라 중앙 예측 + 예측구간으로 표시
- 과거 실제 BTC 가격과 미래 예측을 하나의 연속적인 그래프로 시각화
- Walk-Forward Validation으로 시간 순서를 보존한 모델 평가
- BTC 상승/하락/회복/횡보 등 여러 시장 국면에서 성능 평가
- Validation/Test 반복 사용에 따른 과적합 방지
- 실운영 중 예측값과 실제값을 저장하고 장기 성능 추적
- 주 1회 재학습 후보 검토, 검증 통과한 경우에만 Production 모델 승격
- 알림 기능은 범위에 포함하지 않음

## 권장 데이터 전략

- Canonical historical market data: Binance Spot BTCUSDT 1D OHLCV
- Realtime price: Binance Spot WebSocket
- Cross-exchange sanity check: Coinbase BTC-USD
- 모든 학습 기준 시간은 UTC로 통일

Binance Spot REST API는 `/api/v3/klines`로 Kline/Candlestick 데이터를 제공하고 1d/3d/1w/1M 등을 지원한다. Spot WebSocket은 실시간 시장 스트림을 제공한다. Coinbase Advanced Trade는 BTC-USD candle REST API와 ticker WebSocket을 제공하므로 교차검증용으로 활용할 수 있다.

## 주요 문서

- `CLAUDE.md` : Claude Code 개발 규칙 및 절대 금지사항
- `ARCHITECTURE.md` : 전체 시스템 구조
- `DATA_SPEC.md` : 데이터 수집/정규화/저장 규칙
- `MODEL_SPEC.md` : Feature, target, model, forecast interval 규칙
- `VALIDATION_SPEC.md` : Walk-Forward / cycle coverage / test 규칙
- `OPERATING_SPEC.md` : 실운영 / 주간 재학습 / 모델 승격 규칙
- `TASKS.md` : 단계별 개발 작업 목록
- `config.yaml` : 운영 가능한 모든 핵심 파라미터

## 실행 방법

Python 3.12 이상이 필요하다 (검증 환경: 3.14.7 / Windows).

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS / Linux
```

프로젝트 루트에서 job을 모듈로 실행한다.

```bash
# 1. 시장 데이터 수집 + 검증 + SQLite 저장
python -m jobs.update_market_data --full-refresh   # 최초 전체 backfill
python -m jobs.update_market_data                  # 이후 증분 업데이트

# 2. Point-in-time feature 생성 (+ leakage 검증)
python -m jobs.build_features --check-leakage

# 3. 데이터 품질 리포트 (reports/data_quality_report.md)
python -m jobs.data_quality_report

# 4. Baseline 평가 (reports/baseline_evaluation.md)
python -m jobs.evaluate_baselines                  # 필수 평가 horizon 6개
python -m jobs.evaluate_baselines --all-horizons   # 전체 77개 grid

# 5. 후보 모델 학습 (candidate로만 등록된다)
python -m jobs.train_model                         # 필수 평가 horizon 6개
python -m jobs.train_model --window rolling_4y
python -m jobs.train_model --horizons all          # 전체 grid (약 10분)
python -m jobs.train_model --check-reproducible

# 6. Walk-forward 검증 (inner block 전용)
python -m jobs.walk_forward                        # 선택된 설정으로 검증
python -m jobs.walk_forward --compare-windows      # 학습 window 비교
python -m jobs.walk_forward --compare-params       # 사전 선언된 하이퍼파라미터 비교

# 7. 일일 forecast 생성 (production 모델 필요)
python -m jobs.generate_forecast
python -m jobs.generate_forecast --model-version <version>   # 연구용
python -m jobs.generate_forecast --origin 2025-06-30         # 특정 날짜 backfill

# 8. 도래한 target 확정 + production 지표/드리프트
python -m jobs.evaluate_forecasts

# 테스트
python -m pytest
```

production 지표는 첫 일일 실행부터 쌓인다. 30일 horizon은 30일, 365일 horizon은
1년이 지나야 말을 한다. 과거 날짜로 forecast를 backfill해도 지름길이 되지 않는다.
그 날짜까지 학습된 모델은 이미 답을 봤기 때문이다. `drop_in_sample()`이 그런
행을 자동으로 제외한다 (`OPERATING_SPEC.md` 3.2절).

forecast는 **마지막 확정 일봉**에 anchor한다. 실시간 가격은 `current_price`로
함께 저장되지만 anchor가 되지 않는다 (`CLAUDE.md` 2.4절). 장중 움직임으로
1년 forecast가 재생성되지 않는다.

**현재 검증 결과 (MODEL_SPEC.md 6.5절)**: 트리 모델은 h=1일에서만 baseline을
일관되게 이긴다(4/4 fold, +2.2%). h>=30일부터는 무조건부 경험 분포가 이기고
horizon이 길수록 격차가 커진다. 튜닝 문제가 아니라 독립 관측 수의 한계다.
Phase 6 forecast job은 이 점을 반영해야 한다.

학습은 모델을 production으로 만들지 않는다. `train_model`이 만드는 것은 항상
`candidate`이고, 승격은 주간 리뷰의 별도 판단이다 (`CLAUDE.md` 2.3절).
모델 아티팩트는 `artifacts/models/`에 저장되며 git에는 올리지 않는다.

`evaluate_baselines`는 inner block에서만 채점한다. Outer test는 설계를 동결한
모델을 단 한 번 평가할 때까지 열지 않는다 (`VALIDATION_SPEC.md` 8.2절).
결과 요약은 `MODEL_SPEC.md` 6.2절에 기록되어 있다.

실시간 현재가는 별도의 상주 프로세스로 돌린다. 위 일 단위 job과 동시에
실행해도 안전하다 (`realtime_price` 테이블에만 쓴다).

```bash
python -m jobs.stream_realtime_price              # 중단할 때까지 상주
python -m jobs.stream_realtime_price --duration 60  # 60초만 실행
python -m jobs.stream_realtime_price --once         # REST로 현재가 1회 조회
```

`update_market_data`와 `data_quality_report`는 차단성(blocking) 데이터 품질
오류가 있으면 exit code 1을 반환한다. 이 상태에서는 forecast를 생성하지 않는다
(`OPERATING_SPEC.md` section 9).

## 저장 위치

| 경로 | 내용 |
| --- | --- |
| `data/btc_forecast.db` | SQLite 데이터베이스 (git에서 제외) |
| `reports/` | 생성된 리포트 (git에서 제외) |
| `logs/` | 실행 로그 (git에서 제외) |
| `config.yaml` | 모든 운영 파라미터 |

## 현재 구현 상태

`TASKS.md`에 단계별 상태가 정리되어 있다. 요약하면 데이터 수집 /
검증 / feature 파이프라인까지 완료되었고, 모델 · forecast · Streamlit은
아직 구현되지 않았다.
