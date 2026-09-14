# CLAUDE.md

## 1. 프로젝트 역할

이 프로젝트는 BTC의 미래 가격 경로를 예측하는 연구/모니터링 시스템이다. 자동매매 프로젝트가 아니다. 주문, 포트폴리오 최적화, 알림 기능은 구현하지 않는다.

Claude Code는 이 문서와 세부 설계 문서를 우선 규칙으로 취급한다.

## 2. 가장 중요한 원칙

### 2.1 시간순서 절대 보존

- 랜덤 train/test split 금지.
- 예측 시점 이후에 공개된 데이터가 feature에 들어가면 안 된다.
- rolling/expanding 통계는 반드시 해당 시점까지의 정보만 사용한다.
- scaler, imputer, feature selector, hyperparameter 등 학습되는 변환은 각 fold의 train 구간에서만 fit한다.

### 2.2 Validation/Test 오염 금지

- Validation은 모델/feature/hyperparameter 선택에 사용할 수 있다.
- Final/Outer Test는 모델 선택에 사용하지 않는다.
- Test 결과가 좋지 않다는 이유로 설계 변경 후 같은 Test를 다시 평가하지 않는다.
- Test를 반복 확인하는 행위는 금지한다.
- 운영 데이터가 쌓여도 과거의 독립 Test를 임의로 재사용하여 모델을 선택하지 않는다.

### 2.3 Production 모델 보호

- Candidate 모델 생성과 Production 승격은 별도 단계다.
- 주간 재학습 후보가 기존 모델보다 낫지 않으면 Production 모델을 유지한다.
- 통계적으로 충분한 개선이 확인되지 않은 경우 교체하지 않는다.
- 새 모델이 만들어졌다는 이유만으로 자동 승격하지 않는다.

### 2.4 실시간 가격과 예측 갱신 분리

- Dashboard 현재가는 실시간 업데이트.
- Forecast는 일봉이 확정된 뒤 하루 1회 생성/갱신.
- 장중 움직임으로 장기 forecast를 매 틱 재생성하지 않는다.

## 3. 예측 정의

모델은 가격 자체보다 미래 로그수익률(log return)을 예측한다.

`log_return_h = log(Close[t+h] / Close[t])`

추론 시:

`forecast_price_h = Close[t] * exp(predicted_log_return_h)`

예측 horizon은 다음 규칙으로 구성한다.

- 1~30일: 매일
- 31~90일: 3일 간격
- 91~180일: 7일 간격
- 181~365일: 14일 간격

총 horizon은 365일 이내에서 생성한다.

## 4. 예측 분포

각 horizon에 대해 최소 다음 quantile을 생성한다.

- 2.5%
- 10%
- 25%
- 50%
- 75%
- 90%
- 97.5%

시각화에서는:

- Median = 50%
- 50% interval = 25~75%
- 80% interval = 10~90%
- 95% interval = 2.5~97.5%

을 표시한다.

## 5. 그래프 규칙

- 과거: 실제 BTC close
- 현재: 마지막 확정 일봉 close 및 별도 실시간 current price
- 미래: median + prediction interval
- 과거/미래 경계에 명확한 NOW marker를 표시
- 미래 horizon 점들을 시각적으로 매끄럽게 연결할 때 shape-preserving interpolation(PCHIP 권장)을 사용한다.
- 일반 cubic spline으로 과도한 overshoot가 발생하는 방식은 금지한다.
- 보간값은 시각화용이며 모델의 추가 예측값으로 저장하지 않는다.

## 6. 모델 개발 순서

1. Naive baseline
2. Feature engineering 기반 트리 모델
3. Walk-Forward Validation
4. 성능/안정성 평가
5. 필요하면 딥러닝 후보 실험

딥러닝은 필수가 아니다. 단순 모델이 충분하면 더 복잡한 모델을 사용하지 않는다.

## 7. 시장 사이클 규칙

BTC의 '4년 주기'를 사실로 가정하지 않는다. 대신 여러 시장 국면과 과거 cycle을 평가 범위에 포함한다.

최소한 Bull / Bear / Recovery / Sideways / High-volatility / Low-volatility 구간의 성능을 별도로 기록한다.

## 8. 외부 데이터 규칙

초기 버전은 Binance OHLCV + 가격/거래량 기반 feature를 우선한다.

성능 부족 시 추가 데이터 확장은 다음 순서를 따른다.

1. Market regime / volatility feature
2. On-chain / derivatives feature
3. Macro / cross-asset feature
4. Deep learning

외부 feature를 추가할 때는 해당 데이터가 실제 예측 시점에 이용 가능했는지를 확인하고 point-in-time 정합성을 보장한다.

## 9. 코드 품질

- Python type hint 사용
- 단위 테스트 작성
- 데이터 검증과 모델 로직 분리
- 재현 가능한 random seed
- 로그와 오류 원인 명확히 기록
- 하드코딩된 API key 금지
- 모든 시간은 UTC 내부 저장

## 10. 구현 시 우선 읽을 문서

1. `ARCHITECTURE.md`
2. `DATA_SPEC.md`
3. `MODEL_SPEC.md`
4. `VALIDATION_SPEC.md`
5. `OPERATING_SPEC.md`
6. `TASKS.md`
7. `config.yaml`
