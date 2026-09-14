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

### 3.1 Feature drift is measured against a calibrated null, not a fixed PSI

Implementation: `src/monitoring/drift.py`.

The conventional PSI thresholds (0.1 warn / 0.25 alert) come from credit scoring,
where the current sample is roughly an independent draw. Daily market features
are not: they are strongly autocorrelated, so a contiguous 90-day window is one
regime rather than a random sample of five years.

Measured on this dataset, a 90-day window taken from **inside the training period
itself** -- by construction no drift at all -- puts **48 of 64 features past
0.25**. A trigger that fires on every run teaches the reader to ignore it, which
is worse than having no trigger.

So thresholds are calibrated: contiguous windows are drawn from the reference
period, scored against the rest of it, and each feature is flagged only when it
exceeds what a no-drift window of its own length already produces. That cut the
real-data alert count from 48 features to 12.

Per-feature calibration alone still over-reports, and the measurement says so:
about **16% of features** flag on no-drift windows against a nominal 5%. The
recent window is always at the edge of the reference rather than inside it, and
the features are strongly correlated with each other so flags arrive in clusters.
Both are structural, not fixable by moving a number.

The trigger therefore asks the joint question -- *does this window flag more
features than a no-drift window does?* -- comparing against the calibrated alert
**count**. On the same no-drift checks that fires 1 time in 4 rather than 4 in 4.

Feature drift remains a weak signal. It is one of several triggers, the weekly
review adjudicates, and it must never be the sole reason to replace anything.

### 3.2 Production metrics start empty, and that is correct

Production performance accumulates from the first daily run forward. A 30-day
horizon says nothing for 30 days; a 365-day horizon says nothing for a year.

Backfilling forecasts over past dates does not shortcut this. A model trained
through those dates has already seen the answers, so the realizations look
excellent and mean nothing. `realization.drop_in_sample()` excludes any forecast
whose origin sits inside its own model's training window, and reports how many it
dropped -- the exclusion is enforced rather than left to whoever runs the
backfill to remember.

The independent record of out-of-sample performance is the outer test
(VALIDATION_SPEC.md section 4), evaluated once. The production log is the ongoing
record, and it starts now.

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

Implemented by `repositories.upsert_realizations`, which upserts on
`(forecast_id, horizon_days)`. Re-running the daily job is idempotent, and a
pending row becomes evaluated in place without leaving its earlier state behind.

Row status is `pending` or `fully_evaluated` -- one target date either arrived or
it did not. `partially_evaluable` describes a whole forecast, whose 1-day horizon
may resolve a year before its 365-day horizon (VALIDATION_SPEC.md section 12).

A target date inside the data range with no stored candle stays `pending`. That
is a data problem, not a resolved forecast, and it is logged as one.

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
