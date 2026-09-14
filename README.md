# BTC Long-Horizon Forecast Monitor

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
