# MODEL_SPEC.md

## 1. Forecast objective

Primary target is future log return.

```text
Y(t,h) = log(Close[t+h] / Close[t])
```

where `h` is the forecast horizon in days.

Price reconstruction:

```text
P(t,h) = Close[t] * exp(Y_hat(t,h))
```

## 2. Forecast horizon grid

Use a dense horizon grid for a smooth 1-year forecast:

- 1~30 days: every 1 day
- 31~90 days: every 3 days
- 91~180 days: every 7 days
- 181~365 days: every 14 days

The grid is used for both model outputs and evaluation. Visualization may apply PCHIP interpolation between forecast points.

### 2.1 Band endpoint rule

The band widths are not exact multiples of their spacing: `180 - 90 = 90` is not
divisible by 7, and `365 - 180 = 185` is not divisible by 14. Stepping forward
from each band start would therefore drop the band endpoints, and the grid would
not contain 180 or 365 - both of which section 15 of the project brief and
`VALIDATION_SPEC.md` require as evaluation horizons.

Each band is therefore **anchored on its end day** and stepped backwards until it
would reach the previous band:

```text
band 1:  1..30    step 1   -> 1,2,3,...,30
band 2:  31..90   step 3   -> 33,36,...,90
band 3:  91..180  step 7   -> 96,103,...,180
band 4:  181..365 step 14  -> 183,197,...,365
```

This yields **77 horizons**. Spacing inside each band is exactly as specified,
the endpoints 30 / 90 / 180 / 365 are always present, and the only irregularity
is one shorter gap at each band boundary (90->96 is 6 days, 180->183 is 3 days).

`forecast.horizon_grid_version` in `config.yaml` versions this rule. Changing the
rule requires a new version so old forecasts stay interpretable.

Implementation: `src/forecast/horizons.py`.

## 3. Quantiles

For each horizon, predict:

```text
q02_5
q10
q25
q50
q75
q90
q97_5
```

From these:

- 50% interval = q25 ~ q75
- 80% interval = q10 ~ q90
- 95% interval = q02_5 ~ q97_5

Validate that quantiles are ordered. If the chosen model produces quantile crossing, enforce ordering only as a clearly logged post-processing step and measure its frequency.

## 4. Initial feature groups

### Price / return

- daily log return
- 3d / 7d / 14d / 30d / 90d returns
- rolling drawdown
- distance to rolling highs/lows

### Trend

- SMA / EMA across multiple windows
- price/MA ratios
- moving-average slope
- momentum

### Volatility

- ATR
- rolling standard deviation
- realized volatility
- high-low range
- volatility regime

### Volume

- volume change
- volume moving averages
- volume ratio
- quote volume
- trade count
- taker-buy ratio where reliable

### Regime

- trend regime
- volatility regime
- drawdown regime
- optional bull/bear/recovery/sideways labeling for evaluation, not necessarily as a model feature

## 5. Feature engineering discipline

Feature candidates must be grouped and ablated.

Do not add all possible indicators without evidence.

Evaluate:

1. Price/return baseline
2. + trend
3. + volatility
4. + volume
5. + regime

Keep a simpler feature set when complexity does not improve out-of-sample performance consistently.

## 6. Candidate models

### Baselines

Three baselines, implemented in `src/models/baselines.py`:

| name | location it predicts | hypothesis |
| --- | --- | --- |
| `no_change` | `0` | the price is a driftless random walk |
| `drift` | long-run mean daily log return x h | the long-run trend continues |
| `rolling_return` | recent mean daily log return x h | recent momentum continues |

#### 6.1 Baselines predict distributions, not points

A point-only baseline cannot be compared on pinball loss or interval coverage,
which VALIDATION_SPEC.md section 7 requires. The comparison would then silently
exclude exactly the part of a forecast that matters most at long horizons.

Every baseline is therefore `location + spread`, where the **spread is shared**:
the empirical quantiles of *past* h-day log returns observed up to the origin,
recentered so their median is 0. Holding the spread fixed makes a baseline
comparison a clean test of the location claim, and stops a difference in pinball
loss coming from two different interval recipes.

The spread is empirical rather than Gaussian because BTC returns are fat-tailed
and skewed; a `sigma * sqrt(h)` band would understate the tails it exists to
cover. It is causal: at origin `t` it uses `log(Close[s] / Close[s-h])` for
`s <= t`, and both prices in every term are at or before `t`.

Because empirical quantiles are monotone in the level and the location is a
constant per-origin shift, these forecasts cannot produce crossing quantiles.

#### 6.2 Baseline results on split v1 (inner block, 2018-08-16 .. 2023-12-31)

Reproduce with `python -m jobs.evaluate_baselines`.

| horizon | best by `pinball_mean` | `no_change` MAE (log return) | `drift` MASE | `rolling_return` MASE |
| --- | --- | --- | --- | --- |
| 1d | drift (by 1e-4, noise) | 0.0234 | 1.001 | 1.008 |
| 7d | no_change | 0.0660 | 1.002 | 1.027 |
| 30d | no_change | 0.1611 | 1.011 | 1.226 |
| 90d | no_change | 0.3345 | 1.012 | 1.393 |
| 180d | no_change | 0.4687 | 1.066 | 1.889 |
| 365d | no_change | 0.7196 | 1.127 | 2.472 |

Three findings that constrain Phase 4:

1. **`no_change` is the bar, and it is not a weak one.** Nothing beats it
   anywhere except by noise at h=1. A tree model that does not beat it on
   `pinball_mean` is not an improvement however good its headline MAE looks.
2. **Correcting the bias made the forecast worse.** `no_change` is
   systematically low at long horizons (`return_bias` -0.36 at 365d, because BTC
   rose over the inner block), and `drift` removes almost all of that bias
   (-0.36 -> +0.01) yet has *worse* MAE (0.72 -> 0.81). Log returns are
   right-skewed, so MAE rewards the median and the mean drift overshoots it.
   Chasing unbiasedness is not the same as chasing accuracy here.
3. **Interval calibration degrades with horizon, badly.** 95% coverage runs
   0.97 at 1d but 0.65 at 365d. This is the metric Phase 4 should try hardest to
   improve, and it is consistent with VALIDATION_SPEC.md section 4.3: at 365d
   there are only ~4 independent windows to estimate a tail from.

### Primary ML candidates

- LightGBM quantile regression — implemented (`src/models/lightgbm_model.py`)
- XGBoost regression / quantile-capable objective where supported by pinned version
  — **not implemented yet**, deliberately. CLAUDE.md section 6 says not to add
  complexity before the simpler option has been shown insufficient, and XGBoost
  would add a dependency without answering a question LightGBM has not already
  answered. `src/models/base.py` keeps the algorithm behind a one-class interface
  with a name-based registry, so adding it later is one file, not a refactor.

Use one model per horizon initially for reliability and debuggability.

#### 6.3 Implementation notes

One regressor per **(horizon, quantile)** pair: 7 quantiles x the horizons
trained. Independent, individually inspectable, individually replaceable.

- **No scaler.** Trees do not need one, which removes an entire class of
  leakage: there is no fitted transform that could be fitted across a fold
  boundary (CLAUDE.md section 2.1).
- **Determinism is forced.** `deterministic`, `force_row_wise` and
  `num_threads=1` are set by the code and cannot be overridden from config.
  Without them LightGBM's histogram construction varies with thread scheduling
  and VALIDATION_SPEC.md section 11 becomes unenforceable. Note that the seed
  only has an effect once bagging is enabled -- with `subsample=1.0` GBDT is
  already deterministic, so a seed-only reproducibility check would pass
  vacuously.
- **Hyperparameters are deliberately conservative.** At h=365 the ~1,600
  training rows carry only about 4 independent observations (VALIDATION_SPEC.md
  section 4.3). A tree deep enough to fit them would be memorising overlapping
  windows. Tuning happens on inner validation in Phase 5, never on the test.
- **Serialisation is LightGBM's text format**, bundled as one gzipped JSON file
  per model version. Not pickle: a pickle is executable, ties the artifact to
  one Python version, and would make loading a stored model a code-execution
  decision. One file per version also keeps an artifact atomic -- a half-written
  directory of boosters is a model that loads and silently mixes versions.
- **Quantile crossing is repaired and counted.** Independently fitted quantiles
  can cross; `predict_horizon` sorts each row and reports both the number of
  violating adjacent pairs and the number of affected rows. A rising crossing
  rate means the quantile fits disagree about the same input, which is a real
  signal that hiding the repair would hide.

### Optional deep learning

Only after tree-based models fail to meet acceptance criteria:

- LSTM
- Temporal CNN/TCN
- Temporal Transformer

Deep learning must beat the production baseline on validation stability, not just one split.

#### 6.4 Hyperparameter selection (frozen 2026-09-14)

Four sets were declared up front in `models.lightgbm_candidates` and run once with
`python -m jobs.walk_forward --compare-params` over 4 folds and 6 horizons. They
were declared before running rather than hill-climbed: adjusting one number and
re-checking the same folds is the validation overfitting VALIDATION_SPEC.md
section 5 prohibits.

Mean `pinball_mean` across folds (lower is better):

| horizon | under_regularised | moderate | strong | stumps |
| --- | --- | --- | --- | --- |
| 1d | 0.0074 | 0.0071 | **0.0071** | 0.0071 |
| 7d | 0.0247 | 0.0215 | **0.0211** | 0.0211 |
| 30d | 0.0655 | 0.0512 | **0.0478** | 0.0466 |
| 90d | 0.1256 | 0.1052 | **0.0940** | 0.0913 |
| 180d | 0.2374 | 0.1527 | **0.1376** | 0.1385 |
| 365d | 0.2732 | 0.2454 | **0.2348** | 0.2366 |

**`strong` is selected** and frozen into `models.lightgbm`. It is the only set
that beats the baseline in all four folds at h=1 while remaining at worst tied at
h=7; `stumps` is marginally better at 30-90d but is depth-1 and wins there only by
collapsing toward the baseline.

The first guess (`under_regularised`: 300 rounds, 15 leaves, lambda 1.0) was
wrong and is kept as a negative control. Its 95% interval coverage was **0.62
against a nominal 0.95** at h=30, and it was 80% worse than the baseline at
h=180. That is the signature of fitting extreme quantiles to noise.

#### 6.5 What walk-forward actually found (split v1, inner block)

Full results in `reports/walk_forward_validation.md`. Verdict per horizon for the
selected configuration, against the `no_change` baseline on `pinball_mean`:

| horizon | folds won | mean improvement | verdict |
| --- | --- | --- | --- |
| 1d | 4/4 | +2.2% | beats baseline in every fold |
| 7d | 2/4 | +0.2% | mixed, not consistent |
| 30d | 0/4 | -7.9% | does not beat baseline |
| 90d | 0/4 | -9.2% | does not beat baseline |
| 180d | 2/4 | -8.0% | low power, not decisive |
| 365d | 1/3 | -52.7% | low power, not decisive |

Three conclusions, which constrain everything after this:

1. **Feature-based trees add real value only at h=1.** A consistent 4/4-fold win
   is a genuine result, and it is small. At h=7 the model is indistinguishable
   from doing nothing.
2. **From h=30 outward the unconditional distribution wins, and the gap grows
   with horizon.** This is not a tuning failure. Estimating a *conditional* 97.5th
   percentile of a 30-day return needs tail observations, and with ~53
   independent windows there is roughly one of them. Regularisation improves the
   model precisely by pushing it toward the baseline, and the limit of that
   process is being the baseline. More data would help; more model capacity
   cannot.
3. **The production forecast must not be "the tree" at long horizons.** Phase 6
   should serve the baseline distribution, or a shrunk blend, wherever the tree
   has not demonstrated an advantage. Shipping a worse forecast because it came
   from a model would be the whole point of the exercise, inverted.

Per CLAUDE.md section 6, deep learning is not the answer to this. A model class
with more capacity fails for the same reason and harder.

## 7. Training window candidates

Compare:

- Expanding window
- Rolling 4-year
- Rolling 5-year
- Rolling 8-year
- available-history window

Do not assume the 4-year BTC cycle is stable enough to choose one window a priori.

### 7.1 Comparison result (2026-09-14): expanding, because nothing distinguishes them

`python -m jobs.walk_forward --compare-windows`, 4 folds x 6 horizons.

**Only two of the four candidates are actually distinguishable.** The usable
history is about 8 years and the inner block is 5.4 of them, so a 5- or 8-year
rolling window reaches back past the start of the data and *is* the expanding
window. `rolling_5y` and `rolling_8y` produced numerically identical results to
`expanding` -- not similar, identical. The comparison report now says so
explicitly, because four columns of the same numbers otherwise read as "we
evaluated this and it did not matter" when it was never evaluated at all.

Between the two that differ, `rolling_4y` was marginally better at 7-90d
(mean `pinball_mean` 0.0921 vs 0.0940 at 90d) and marginally worse at 180-365d.
The differences are far inside fold-to-fold spread.

**`expanding` stays the default.** No evidence supports discarding data, and
CLAUDE.md section 6 says not to add complexity without it. This choice is
recorded and frozen here per VALIDATION_SPEC.md section 5; revisiting it needs a
new reason, not a new run.

## 8. Prediction interval training

Primary implementation should prefer quantile regression so the interval is learned from the target distribution rather than generated by arbitrary fixed percentage bands.

## 9. Visualization interpolation

Prediction points should be converted to a future daily visualization grid using PCHIP/shape-preserving interpolation separately for each quantile curve.

The interpolation must not create a price lower than the lower quantile or higher than the upper quantile at each displayed date.

## 10. Required model metadata

Implemented as `src/models/base.py::ModelMetadata` (frozen dataclass, so a model
cannot reach the registry with half of it missing) and persisted to
`model_registry`. `training_rows` is supplemented by `training_rows_by_horizon`,
because the purge costs more data at longer horizons and a single row count
would misreport every horizon but one.

Every trained model must save:

- model_id
- model_version
- algorithm
- library versions
- feature_version
- training_start
- training_cutoff
- horizon_grid_version
- hyperparameters
- random_seed
- validation metrics
- test metrics
- training data row count
- status

## 11. Acceptance philosophy

There is no hard-coded claim such as “BTC direction accuracy must exceed 60%”. Acceptance thresholds must be configuration-driven and evaluated relative to baselines and historical stability.
