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

Implemented by `jobs/weekly_model_review.py`, which writes
`reports/weekly_model_review.md`.

### 2.1 Building a candidate is not promoting one

The default run trains a candidate, registers it, evaluates the gate and writes
the report. It does not touch the production model. `--apply` is the only mode
that writes `model_registry.status`, and it is a separate flag because promotion
changes what every subsequent forecast is made of.

A candidate is built only when a retraining trigger fired (section 3) or
`--force` was passed. "A week has passed" is not a trigger.

### 2.2 The review stops on blocking data errors

A candidate trained on data that failed validation carries the fault into every
forecast it produces, so the data-quality step runs first and a blocking error
ends the review before training. Warnings do not: an extreme daily move is a real
market event, not a data problem, and removing it would be the actual error.

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

### 5.1 Three routes, because there are three different questions

Implemented in `src/models/promotion.py`. The route is chosen from a
**configuration fingerprint** that hashes algorithm, hyperparameters, training
window, feature version, horizon-grid version and seed -- and deliberately
excludes the training cutoff, so "the same design trained through a later date"
is recognised as such.

| route | when | judged on |
| --- | --- | --- |
| `bootstrap` | no production model exists | is serving this better than serving the baseline alone? |
| `configuration_change` | the design differs from the incumbent | does it beat the incumbent on identical folds, by the margin, in every fold? |
| `data_refresh` | same design, later cutoff | is it provably the same design, genuinely fresher, and sane? |

### 5.2 A data refresh cannot be judged on inner validation

Folds are laid out backwards from the frozen `inner_validation_end`
(VALIDATION_SPEC.md section 2.1) and every feature is causal, so a walk-forward
run over the inner block returns **the same numbers it returned before the new
data arrived**: the extra training data lies entirely after the last scored
origin.

This is not a gap to patch. Sliding the inner block forward so the newest data
gets scored is the validation overfitting section 5 of VALIDATION_SPEC.md
prohibits, arriving one week at a time.

So a refresh is promoted on **equivalence plus freshness**, not on a measured
improvement: the configuration is provably unchanged, the cutoff has genuinely
advanced, the refit reproduces the incumbent's recorded inner-block numbers
exactly, and the new model's live forecast is plausible. The claim that it is
*better* is a prior -- a model fitted through last week has seen the market that
produced the price it forecasts from -- and the report says so rather than
dressing it up as a measurement.

The equivalence check earns its place by being expected to pass: on a frozen
block with causal features the numbers should match to the last bit, so a
mismatch means history was revised, the feature pipeline changed, or a fit is not
deterministic.

### 5.3 The gate judges only where the model is actually used

`forecast.blend` gives the model a weight that falls to zero by 30 days
(MODEL_SPEC.md section 6.5), so the served 90-day forecast contains none of the
model. Rejecting a candidate for a 90-day number would be rejecting it on
evidence about something nobody ships, and promoting it for one would be worse.

Horizons with weight 0 are reported and marked `decision_horizon = False`. The
exception is VALIDATION_SPEC.md section 10's rule that a low-power horizon cannot
justify a promotion but can still veto one: a *catastrophic* result at any served
horizon blocks, including a low-power one.

For the same reason the `bootstrap` bar is asymmetric. The model must earn its
place somewhere it is used -- beating the baseline by the margin in every fold at
at least one weighted horizon -- and must not do damage anywhere it is used.
Requiring a full margin at every weighted horizon would reject a model that
merely ties where it is trusted, and a tie does no harm; requiring nothing would
promote a tree that reproduces the baseline, which is strictly worse than the
baseline because it is the same forecast with more moving parts.

### 5.4 Coverage is judged by distance from nominal

A 95% interval that covers 99% of outcomes is miscalibrated exactly as much as
one covering 91%; it is simply wrong in the direction that feels safe. The gate
compares `|coverage - nominal|` against the reference's, not the raw number.

### 5.5 Rejections are recorded

`registry.reject` requires a reason and the gate produces one naming the check
and the number it failed at. "We looked and kept the incumbent" and "nobody ran
the review" are otherwise indistinguishable from an unchanged model version.

## 6. Model freeze

Each production version has:

- training cutoff
- promotion timestamp
- feature version
- config version

Production predictions must always be reproducible from those records.

### 6.1 What the training cutoff is allowed to be

Before the design has had its single outer-test evaluation, every model stops at
the purge boundary (`origin + horizon + embargo < outer_test_start`). A
consequence worth stating plainly: however much new data arrives, retraining
produces the **identical model version**, and the weekly review will say so.

That is correct. A model trained past the boundary destroys the reserved block
without ever reading it, and looks completely normal afterwards.

`jobs/final_evaluation.py` spends the block once and records the result, which
releases the cutoff for that design only. From then on the served model trains
through the newest resolved label and can never be tested again
(VALIDATION_SPEC.md section 4.5).

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
