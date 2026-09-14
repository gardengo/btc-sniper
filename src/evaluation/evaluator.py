"""Turning aligned forecasts and actuals into the metric tables.

This is the single scoring path for everything the project forecasts: baselines
now, tree models in Phase 5, logged production forecasts in Phase 7. Feeding
them all through one evaluator is what makes "candidate beats baseline by X"
a real comparison rather than two differently-computed numbers.

Regime tagging uses the regime **at the forecast origin**, not over the target
window. That is the causal reading and the decision-relevant one: "how does this
model do when we are in a bear market *today*" is answerable at forecast time,
whereas labelling by the outcome window would score the model partly on
information it could not have had (VALIDATION_SPEC.md section 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.evaluation import metrics as m
from src.forecast.quantiles import assert_ordered, quantile_label, resolve_interval
from src.validation.splits import independent_window_count

ALL_REGIMES: str = "all"


class EvaluationError(RuntimeError):
    """Raised when a sample cannot be scored."""


@dataclass(frozen=True)
class ForecastSample:
    """One model's forecasts for one horizon, aligned with what happened.

    Every series is indexed by forecast origin date. ``actual_log_return`` is
    ``NaN`` wherever the target date has not arrived; those origins are dropped
    by the metric functions rather than imputed.
    """

    model_version: str
    horizon_days: int
    origin_close: pd.Series
    actual_log_return: pd.Series
    predicted_quantiles: pd.DataFrame
    median_level: float = 0.50
    origin_spacing_days: int = 1
    regimes: pd.DataFrame | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.horizon_days < 1:
            raise EvaluationError("horizon_days must be >= 1")
        if self.median_level not in self.predicted_quantiles.columns:
            raise EvaluationError(
                f"median level {self.median_level} missing from predicted quantiles "
                f"{list(self.predicted_quantiles.columns)}"
            )

    @property
    def levels(self) -> tuple[float, ...]:
        return tuple(sorted(float(c) for c in self.predicted_quantiles.columns))

    @property
    def median(self) -> pd.Series:
        return self.predicted_quantiles[self.median_level]

    @property
    def origins(self) -> pd.DatetimeIndex:
        return self.predicted_quantiles.index

    def scoreable(self) -> pd.DatetimeIndex:
        """Origins whose target date has arrived and whose forecast is complete."""
        complete = self.predicted_quantiles.dropna(axis=0, how="any").index
        resolved = self.actual_log_return.dropna().index
        return complete.intersection(resolved).sort_values()

    def restrict(self, origins: pd.Index) -> "ForecastSample":
        return ForecastSample(
            model_version=self.model_version,
            horizon_days=self.horizon_days,
            origin_close=self.origin_close.reindex(origins),
            actual_log_return=self.actual_log_return.reindex(origins),
            predicted_quantiles=self.predicted_quantiles.reindex(origins),
            median_level=self.median_level,
            origin_spacing_days=self.origin_spacing_days,
            regimes=None if self.regimes is None else self.regimes.reindex(origins),
        )

    def actual_price(self) -> pd.Series:
        return self.origin_close * np.exp(self.actual_log_return)

    def predicted_price(self) -> pd.Series:
        return self.origin_close * np.exp(self.median)


def evaluate(
    sample: ForecastSample,
    *,
    intervals: tuple[float, ...] = (0.50, 0.80, 0.95),
    reference_median: pd.Series | None = None,
) -> dict[str, float]:
    """All VALIDATION_SPEC.md section 7 metrics for one sample.

    ``reference_median`` is the no-change baseline's median on the same origins;
    supplying it adds the scaled error ratio (`mase`) and the improvement
    fraction that section 8 requires.
    """
    scoreable = sample.scoreable()
    subset = sample.restrict(scoreable)
    actual = subset.actual_log_return
    median = subset.median

    results: dict[str, float] = {
        "sample_size": float(len(scoreable)),
        "independent_windows": round(
            independent_window_count(
                len(scoreable), sample.horizon_days, sample.origin_spacing_days
            ),
            2,
        ),
        "return_mae": m.mae(actual, median),
        "return_rmse": m.rmse(actual, median),
        "return_bias": m.bias(actual, median),
        "price_mae": m.mae(subset.actual_price(), subset.predicted_price()),
        "price_rmse": m.rmse(subset.actual_price(), subset.predicted_price()),
        "price_smape": m.smape(subset.actual_price(), subset.predicted_price()),
    }
    results.update(m.directional_accuracy(actual, median))

    pinballs: list[float] = []
    for level in subset.levels:
        loss = m.pinball_loss(actual, subset.predicted_quantiles[level], level)
        results[f"pinball_{quantile_label(level)}"] = loss
        if not np.isnan(loss):
            pinballs.append(loss)
    # The mean pinball loss over a symmetric quantile grid approximates CRPS and
    # is the single number that ranks whole predictive distributions.
    results["pinball_mean"] = float(np.mean(pinballs)) if pinballs else float("nan")
    results["quantile_crossings"] = float(
        assert_ordered(subset.predicted_quantiles.dropna(axis=0, how="any"), subset.levels)
    )

    for interval in intervals:
        try:
            low_level, high_level = resolve_interval(interval, subset.levels)
        except Exception:  # noqa: BLE001 - an unavailable interval is reported, not fatal
            continue
        tag = f"{int(round(interval * 100))}"
        lower = subset.predicted_quantiles[low_level]
        upper = subset.predicted_quantiles[high_level]
        coverage = m.interval_coverage(actual, lower, upper)
        results[f"coverage_{tag}"] = coverage
        results[f"coverage_error_{tag}"] = coverage - interval
        results[f"width_{tag}"] = m.interval_width(lower, upper)
        results[f"rel_width_{tag}"] = m.relative_price_width(lower, upper)
        results[f"interval_score_{tag}"] = m.interval_score(actual, lower, upper, interval)

    if reference_median is not None:
        reference = reference_median.reindex(scoreable)
        ratio = m.mase(actual, median, reference)
        results["mase"] = ratio
        results["improvement_vs_no_change"] = (
            float("nan") if np.isnan(ratio) else 1.0 - ratio
        )
    return results


def regime_masks(
    regimes: pd.DataFrame | None, wanted: tuple[str, ...]
) -> dict[str, pd.Index]:
    """Origin subsets for each requested regime label.

    Both regime axes are searched, so ``bear`` matches the direction column and
    ``high_volatility`` the volatility column without the caller needing to know
    which axis a label lives on.
    """
    if regimes is None or regimes.empty:
        return {}
    masks: dict[str, pd.Index] = {}
    for label in wanted:
        matched = pd.Series(False, index=regimes.index)
        for column in regimes.columns:
            matched |= regimes[column].astype("object") == label
        selected = regimes.index[matched.fillna(False).to_numpy()]
        if len(selected):
            masks[label] = selected
    return masks


def evaluate_sample_table(
    sample: ForecastSample,
    *,
    intervals: tuple[float, ...] = (0.50, 0.80, 0.95),
    reference_median: pd.Series | None = None,
    regimes: tuple[str, ...] = (),
    low_power: bool = False,
) -> pd.DataFrame:
    """Tidy metric rows for one sample: overall plus one block per regime."""
    rows: list[dict[str, Any]] = []

    def _emit(regime: str, subset: ForecastSample, reference: pd.Series | None) -> None:
        scored = subset.scoreable()
        values = evaluate(subset, intervals=intervals, reference_median=reference)
        period_start = scored.min().strftime("%Y-%m-%d") if len(scored) else None
        period_end = scored.max().strftime("%Y-%m-%d") if len(scored) else None
        for name, value in values.items():
            rows.append(
                {
                    "model_version": subset.model_version,
                    "horizon_days": subset.horizon_days,
                    "regime": regime,
                    "metric_name": name,
                    "metric_value": None if pd.isna(value) else float(value),
                    "sample_size": int(len(scored)),
                    "period_start": period_start,
                    "period_end": period_end,
                    "low_power": low_power,
                }
            )

    _emit(ALL_REGIMES, sample, reference_median)
    for label, origins in regime_masks(sample.regimes, regimes).items():
        subset = sample.restrict(sample.origins.intersection(origins))
        reference = None if reference_median is None else reference_median.reindex(subset.origins)
        _emit(label, subset, reference)

    return pd.DataFrame(rows)


def pivot_metrics(
    table: pd.DataFrame, metric_names: tuple[str, ...], *, regime: str = ALL_REGIMES
) -> pd.DataFrame:
    """Reshape tidy metrics into model x horizon rows for a report table."""
    subset = table[(table["regime"] == regime) & (table["metric_name"].isin(metric_names))]
    if subset.empty:
        return pd.DataFrame()
    wide = subset.pivot_table(
        index=["horizon_days", "model_version"],
        columns="metric_name",
        values="metric_value",
        sort=False,
    ).reset_index()
    ordered = ["horizon_days", "model_version"] + [
        name for name in metric_names if name in wide.columns
    ]
    return wide[ordered].sort_values(["horizon_days", "model_version"]).reset_index(drop=True)


def summarise_stability(
    table: pd.DataFrame, metric_name: str, *, regimes: tuple[str, ...]
) -> Mapping[str, float]:
    """Dispersion of one metric across regimes, for the stability requirement."""
    subset = table[
        (table["metric_name"] == metric_name) & (table["regime"].isin(regimes))
    ]
    return m.dispersion(subset["metric_value"])
