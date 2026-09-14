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

### 2.1 Fold layout

Implemented in `src/validation/folds.py`. For horizon `h` a fold is

```text
train:      [window_start, validation_start - h - embargo)
validation: [validation_start, validation_start + validation_days)
```

The `h + embargo` gap is not cosmetic. A training example at origin `t` carries
the label `log(Close[t+h] / Close[t])`, so without the gap the last training
labels read prices from inside the validation block. This is the section 4.1
purge rule applied one level in, and `assert_fold_is_clean()` enforces both it
and the outer-test rule immediately before every fit.

Folds are laid out **backwards from `inner_validation_end`**, so the most recent
validation block is always complete and always present. Laying them forward from
the start of the data leaves a ragged remainder at the end -- the most
market-relevant period -- and quietly changes what every fold sees as soon as one
more day of history arrives.

A fold with fewer than `validation.walk_forward.min_train_origins` training
origins is dropped rather than trained: a model fitted on a hundred overlapping
rows produces a metric that looks real and is not.

Training windows (`expanding`, `rolling_4y/5y/8y`) set the fold's start;
MODEL_SPEC.md section 7 is explicit that the choice between them is made
empirically on inner validation, never by assuming the four-year cycle.

## 3. One-year horizon validation

A 365-day forecast cannot be fully scored until 365 future days have occurred.

Therefore historical evaluation uses pseudo-live origins:

```text
origin_t -> forecast 1..365 days -> compare with actuals
origin_t+K -> forecast 1..365 days -> compare with actuals
...
```

Origins must be spaced to avoid excessive overlap. Default origin spacing: 30 days for research evaluation, configurable.

### 3.1 Spacing is the horizon, capped

`validation.validation_origin_spacing_days` is a **cap**, not a flat rule. What
actually removes overlap is spacing origins by the horizon: at `h` days apart the
target windows are exactly disjoint. The implemented rule is therefore

```text
spacing = min(horizon, validation_origin_spacing_days)
```

- `h <= cap`: spacing `h`, giving genuinely independent observations.
- `h > cap`:  spacing `cap`, accepting overlap because spacing a 365-day horizon
  by 365 days would leave a handful of origins and no measurable metric at all.
  The overlap is then reported honestly through `independent_windows`.

A flat 30-day spacing would discard 96% of usable origins at `h=1`, where
consecutive origins barely overlap to begin with, and would make the
best-measured horizon the noisiest one.

Implementation: `src/evaluation/baseline_eval.py::origin_spacing_days`.

### 3.2 Effective sample size

Overlap is accounted for as

```text
independent_windows = origin_count * min(spacing, horizon) / horizon
```

which reduces to `origin_count / horizon` for daily origins and to
`origin_count` once spacing reaches the horizon. Implementation:
`src/validation/splits.py::independent_window_count`.

## 4. Final / Outer Test

Reserve a chronologically later block that is untouched during model selection.

For each model family:

1. use inner walk-forward validation for feature/model/hyperparameter selection
2. freeze design
3. train final candidate using all eligible pre-test data
4. evaluate once on outer test
5. do not modify the design based on that outer test result

### 4.1 Eligibility and purge rule

"Eligible pre-test data" is not "any origin before the test start".

A training example at origin `t` for horizon `h` carries the label
`log(Close[t+h] / Close[t])`. The label **reads a price at `t+h`**. If `t+h`
falls inside the outer test block, the model has been trained on the answer it
is about to be tested on, even though its feature row at `t` is entirely in the
past.

The eligibility rule is therefore:

```text
origin + horizon + embargo_days < outer_test_start
```

This purge is horizon-dependent. At h=1 it costs one day of training data; at
h=365 it costs a full year. The same rule applies between the train and
validation blocks of every inner walk-forward fold.

Enforcement lives in `src/validation/splits.py`:
`eligible_training_origins()` selects origins, and
`assert_no_test_contamination()` must be called immediately before fitting.

### 4.2 Frozen split (split_version v1)

Decided 2026-09-14, before any model was trained.

| Block | Range | Purpose |
| --- | --- | --- |
| warmup | 2017-08-17 .. 2018-08-15 | feature warmup, no usable origins |
| inner (train + walk-forward validation) | 2018-08-16 .. 2023-12-31 | all model, feature and hyperparameter selection |
| outer test | 2024-01-01 .. latest data | evaluated once per frozen design |

Resulting sizes on the 2026-09-13 data snapshot:

| horizon | train origins | independent train windows | test origins scoreable | independent test windows |
| --- | --- | --- | --- | --- |
| 1d | 1,963 | 1963.0 | 986 | 986.0 |
| 7d | 1,957 | 279.6 | 980 | 140.0 |
| 30d | 1,934 | 64.5 | 957 | 31.9 |
| 90d | 1,874 | 20.8 | 897 | 10.0 |
| 180d | 1,784 | 9.9 | 807 | 4.5 |
| 365d | 1,599 | 4.4 | 622 | 1.7 |

Rationale:

- **2024-01-01 balances the two scarce resources.** It leaves ~66% of usable
  origins for the inner block and reserves ~33% for the outer test, which is a
  conventional and defensible fraction.
- **The test block is long enough where power exists.** 987 test origin-days
  give ~986 near-independent observations at 1d, ~32 at 30d and ~10 at 90d.
  Those horizons can support a real verdict.
- **Regime coverage is not the deciding factor.** Every candidate boundary from
  2023-01-01 to 2025-01-01 yields a test block containing bull, bear, recovery,
  sideways, high- and low-volatility days, so coverage does not discriminate.
- **The inner block covers every regime too**, including the 2018 bear, the
  2020 crash and recovery, the 2021 cycle and the 2022 deep bear, so
  walk-forward folds can be tagged by regime as section 6 requires.
- **A later boundary starves the test; an earlier one starves training.**
  2025-01-01 leaves only 256 fully scoreable 365d test origins; 2023-01-01
  removes a year from an already short training history.

`outer_test_end` is `null`, meaning the test block rolls forward as new data
arrives. This is deliberate: new unseen data strengthens the test rather than
weakening it. The protection against re-use is section 4.4, not a fixed end
date.

### 4.3 Long-horizon statistical power

Bitcoin has roughly eight years of usable daily history. At a 365-day horizon
that is about **4.4 independent training windows and 1.7 independent test
windows** -- and no choice of boundary changes this materially, because the
constraint is the length of the dataset, not the split.

Consequences, which are requirements rather than caveats:

- Horizons at or above `validation.low_power_horizon_days` (180) are reported
  with an explicit low-power flag.
- A 365-day result must never be the sole basis for promoting or rejecting a
  model. Section 10 acceptance criteria are decided on horizons where the
  evidence exists, with long horizons used as a sanity check for catastrophic
  behaviour only.
- Interval calibration matters more than point accuracy at long horizons. A
  365-day forecast whose 95% interval is honest is useful; a 365-day median
  claiming precision is not.

### 4.4 Outer test re-use protection

CLAUDE.md section 2.2 forbids repeatedly inspecting the test set.

- Each `model_version` may evaluate the outer test at most
  `validation.max_outer_test_evaluations_per_model_version` times (default 1).
- The evaluation is recorded in `model_registry.test_metrics` together with the
  exact test window that was used.
- A design changed in response to a test result is a **new** design: it needs a
  new `model_version`, and its own single evaluation.
- Moving `outer_test_start` invalidates every previously recorded outer-test
  metric and requires a new `split_version`. It must never be moved to obtain a
  better number.

Ongoing honest measurement after the test comes from logged production
forecasts (`forecasts` / `forecast_realizations`), not from re-running the
outer test.

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

### 8.1 The reference and the ratio

`no_change` is the reference baseline (`src/models/baselines.REFERENCE_BASELINE`).
`mase` is the ratio of a forecast's mean absolute log-return error to the
reference's **on the same origins and the same horizon**; below 1.0 beats it,
and `improvement_vs_no_change = 1 - mase`.

This deviates from textbook MASE, which scales by in-sample naive-1 errors. At a
365-day horizon that denominator would be roughly 19x too small and the number
meaningless. Every model in this project is scored through the same function, so
the ratio is comparable across models even though it is not comparable to MASE
values published elsewhere.

### 8.2 Baselines are scored on the inner block only

Baselines have no fitted parameters that could overfit a test set, but computing
their outer-test numbers during development would still tell *the developer*
what that block looks like, which is what section 2.2 of CLAUDE.md forbids.
`evaluate_baselines()` refuses any scope other than the inner block; the
baseline's outer-test numbers are produced once, alongside the final model's
single evaluation.

Current baseline results are recorded in MODEL_SPEC.md section 6.2.

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

Horizons flagged low-power by section 4.3 are excluded from the primary
decision metric. They are still reported, and a catastrophic result there can
still veto a promotion, but they cannot by themselves justify one.

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
