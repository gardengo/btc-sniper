# OPERATING_SPEC.md

## 1. Runtime behavior

### Realtime

- continuously update current BTC price for Streamlit
- do not retrain from every tick
- do not change the model forecast on every tick

### Daily

After the daily BTC candle is fully closed in UTC:

1. ingest/validate new candle
2. update features
3. generate new 1D~365D forecast grid
4. save prediction
5. resolve any forecast targets whose actual values are now available
6. update performance metrics

## 2. Weekly model review

Run once per week.

Sequence:

```text
Data quality check
 -> performance update
 -> drift/regime review
 -> retraining necessity check
 -> if necessary train candidate
 -> inner validation
 -> compare against incumbent
 -> promote or reject
```

## 3. Retraining trigger

Retraining is not automatic just because one week passed.

Candidate training is considered when one or more configured conditions are met:

- scheduled weekly review window
- recent performance degradation over a minimum sample size
- feature/data distribution drift exceeds threshold
- enough new observations accumulated since previous model

If none are met, keep incumbent.

## 4. Minimum sample safeguards

Do not make replacement decisions from tiny samples.

For each horizon, the number of fully realized production forecasts must exceed the configured minimum before using that horizon's production metric for promotion decisions.

## 5. Candidate promotion

```text
Candidate
  |
  +-- validation failed -> reject
  |
  +-- incumbent not beaten -> reject
  |
  +-- interval coverage materially worse -> reject
  |
  +-- unstable by regime/fold -> reject
  |
  +-- passes all checks -> promote
```

Production model is immutable until promotion.

## 6. Model freeze

Each production version has:

- training cutoff
- promotion timestamp
- feature version
- config version

Production predictions must always be reproducible from those records.

## 7. Forecast log lifecycle

At forecast creation save:

- forecast_origin
- target horizon dates
- current/anchor price
- all quantile forecasts
- model version

When actual target closes become available, update realization rows rather than creating a second prediction.

## 8. Streamlit behavior

### Dashboard

Show:

- realtime current BTC price
- last forecast refresh time
- production model version
- historical actual price
- future median forecast
- 50/80/95% intervals
- Now divider
- 1M / 3M / 6M / 12M key forecast summary

### Prediction Log

Filters:

- forecast date
- horizon
- evaluation status

Columns:

- forecast date
- target date
- horizon
- predicted median
- actual price
- error
- interval inclusion

### Model Performance

Show:

- rolling MAE
- RMSE
- direction accuracy
- quantile loss
- interval coverage
- regime-specific metrics
- model version history
- 365D forecast revision history

## 9. Operational failure behavior

- API outage: retry + log + preserve last known good state
- WebSocket outage: reconnect; use REST fallback for UI current price
- missing daily candle: do not generate a new forecast until data is complete
- model load failure: keep last good production model
- candidate training failure: never affect production model

## 10. No notification system

Telegram, KakaoTalk, email, webhook 등 알림은 구현하지 않는다.
