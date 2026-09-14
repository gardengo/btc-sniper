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

- LightGBM quantile regression
- XGBoost regression / quantile-capable objective where supported by pinned version

Use one model per horizon initially for reliability and debuggability.

### Optional deep learning

Only after tree-based models fail to meet acceptance criteria:

- LSTM
- Temporal CNN/TCN
- Temporal Transformer

Deep learning must beat the production baseline on validation stability, not just one split.

## 7. Training window candidates

Compare:

- Expanding window
- Rolling 4-year
- Rolling 5-year
- Rolling 8-year
- available-history window

Do not assume the 4-year BTC cycle is stable enough to choose one window a priori.

## 8. Prediction interval training

Primary implementation should prefer quantile regression so the interval is learned from the target distribution rather than generated by arbitrary fixed percentage bands.

## 9. Visualization interpolation

Prediction points should be converted to a future daily visualization grid using PCHIP/shape-preserving interpolation separately for each quantile curve.

The interpolation must not create a price lower than the lower quantile or higher than the upper quantile at each displayed date.

## 10. Required model metadata

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
