# VALIDATION_SPEC.md

## 1. Core principle

This is a time-series forecasting system. Random cross-validation is prohibited.

## 2. Walk-Forward Validation

Use expanding-window or rolling-window folds.

Generic pattern:

```text
Train 1 -> Validation 1
Train 2 -> Validation 2
Train 3 -> Validation 3
...
```

Each validation prediction must be generated only from information available at its forecast origin.

## 3. One-year horizon validation

A 365-day forecast cannot be fully scored until 365 future days have occurred.

Therefore historical evaluation uses pseudo-live origins:

```text
origin_t -> forecast 1..365 days -> compare with actuals
origin_t+K -> forecast 1..365 days -> compare with actuals
...
```

Origins must be spaced to avoid excessive overlap. Default origin spacing: 30 days for research evaluation, configurable.

## 4. Final / Outer Test

Reserve a chronologically later block that is untouched during model selection.

For each model family:

1. use inner walk-forward validation for feature/model/hyperparameter selection
2. freeze design
3. train final candidate using all eligible pre-test data
4. evaluate once on outer test
5. do not modify the design based on that outer test result

## 5. Validation overfitting protection

The following are prohibited:

- tuning a feature based on repeated inspection of one fixed validation year
- choosing hyperparameters from final test results
- selecting a production model from historical test windows that have already been repeatedly optimized

When a validation choice has been made, record the decision and freeze it before moving to final test.

## 6. Multi-cycle coverage

Validation reports must include performance by historical regime and cycle.

Minimum tags:

- Bull
- Bear
- Recovery
- Sideways
- High Volatility
- Low Volatility

Cycle analysis is descriptive/evaluative. Do not assume Bitcoin has a deterministic four-year pattern.

## 7. Required metrics

### Point forecast

- MAE
- RMSE
- sMAPE or MASE

### Return forecast

- MAE of log return
- RMSE of log return

### Direction

- overall directional accuracy
- up-direction accuracy
- down-direction accuracy

### Probabilistic forecast

- pinball loss for each quantile
- interval coverage
- interval width
- coverage-vs-width tradeoff

### Stability

- metric dispersion across folds
- worst-fold metric
- regime-specific metric

## 8. Baseline comparison

A complex model must be compared against naive baselines.

Report:

- improvement vs no-change
- improvement vs drift
- improvement consistency across folds

## 9. Training window comparison

At minimum compare:

- expanding
- rolling 4Y
- rolling 5Y
- rolling 8Y

Select the window only from inner validation. The final test remains untouched.

## 10. Acceptance criteria

A candidate is eligible for promotion when all configured minimum conditions are met:

- beats incumbent on primary validation metric by configured margin
- does not materially worsen interval coverage
- performance improvement is present across a minimum number of folds
- no catastrophic degradation in a key regime
- reproducibility checks pass

## 11. Reproducibility

Every evaluation run records:

- dataset snapshot/cutoff
- code version/commit
- model version
- feature version
- config version
- random seed
- metrics

## 12. Evaluation states

Each forecast realization is one of:

- `pending`
- `partially_evaluable`
- `fully_evaluated`

A 365D forecast remains pending/partial until its target date arrives.
