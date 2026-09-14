"""Naive reference forecasts.

MODEL_SPEC.md section 6 requires three baselines and VALIDATION_SPEC.md
section 8 requires every candidate to be reported against them. They exist to
answer one question before any tree model is trained: *does the complicated
thing actually beat doing nothing?*

Shared structure
----------------
All three baselines predict a **distribution**, not a point. A point-only
baseline cannot be compared on pinball loss or interval coverage, which are
required metrics, so the comparison would silently exclude exactly the part of
the forecast that matters most at long horizons.

Each prediction is ``location + spread``:

* **location** is the predicted median log return and is the only thing that
  differs between the baselines. `no_change` claims 0, `drift` claims the
  long-run average, `rolling_return` claims the recent average, and `empirical`
  claims the historical *median* -- which is what actually minimises absolute
  error, and which none of the other three can express.
* **spread** is shared: the empirical quantiles of *past* h-day log returns,
  recentered so their median is 0.

Holding the spread fixed is deliberate. It makes a baseline comparison a clean
test of the location claim, and it means any difference in pinball loss between
them comes from the median they predict rather than from two different interval
recipes.

Causality
---------
The spread at origin ``t`` uses ``log(Close[s] / Close[s-h])`` for ``s <= t``.
Both prices in every term are at or before ``t``, so nothing after the origin is
read -- an h-day return *ending* at ``t`` is fully observed at ``t``. The same
holds for the drift and rolling-return means, which are built from 1-day
returns. `tests/test_leakage.py` verifies this by truncation.

Because the quantiles are empirical and monotone in the level, and the location
is a constant shift per origin, these forecasts cannot produce crossing
quantiles.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.config import BaselinesConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

MEDIAN_LEVEL: float = 0.50


class BaselineError(RuntimeError):
    """Raised when a baseline cannot produce a forecast."""


@dataclass(frozen=True)
class BaselinePrediction:
    """One baseline's forecast for one horizon, indexed by forecast origin."""

    name: str
    horizon_days: int
    location: pd.Series
    quantiles: pd.DataFrame
    levels: tuple[float, ...]

    @property
    def median(self) -> pd.Series:
        """Predicted median log return."""
        if MEDIAN_LEVEL in self.quantiles.columns:
            return self.quantiles[MEDIAN_LEVEL]
        return self.location

    def valid_origins(self) -> pd.DatetimeIndex:
        """Origins where every quantile is defined."""
        return self.quantiles.dropna(axis=0, how="any").index

    def restrict(self, origins: pd.DatetimeIndex) -> "BaselinePrediction":
        """Same prediction narrowed to ``origins``."""
        return BaselinePrediction(
            name=self.name,
            horizon_days=self.horizon_days,
            location=self.location.reindex(origins),
            quantiles=self.quantiles.reindex(origins),
            levels=self.levels,
        )

    def prices(self, origin_close: pd.Series) -> pd.DataFrame:
        """Quantiles converted to price space (MODEL_SPEC.md section 1)."""
        close = origin_close.reindex(self.quantiles.index)
        return self.quantiles.apply(lambda column: close * np.exp(column))


def _log_close(close: pd.Series) -> pd.Series:
    if close.empty:
        raise BaselineError("cannot forecast from an empty close series")
    if not close.index.is_monotonic_increasing:
        raise BaselineError("close series must be sorted ascending by date")
    values = close.astype("float64")
    if (values <= 0).any():
        raise BaselineError("close series contains non-positive prices")
    return np.log(values)


def _past_return_window(
    log_close: pd.Series, horizon_days: int, settings: BaselinesConfig
):
    """Rolling/expanding view over past h-day log returns, ending at each origin."""
    past = log_close.diff(horizon_days)
    if settings.quantile_lookback_days is None:
        return past.expanding(min_periods=settings.quantile_min_samples)
    return past.rolling(
        settings.quantile_lookback_days, min_periods=settings.quantile_min_samples
    )


def empirical_spread(
    log_close: pd.Series,
    horizon_days: int,
    levels: tuple[float, ...],
    settings: BaselinesConfig,
) -> pd.DataFrame:
    """Zero-centred empirical quantiles of past ``horizon_days`` log returns.

    Column ``q`` at row ``t`` is ``quantile_q(R_t) - median(R_t)`` where ``R_t``
    is every h-day log return observed up to and including ``t``. Subtracting
    the median is what makes it a pure spread: the location is supplied by the
    baseline, so the two parts never both claim a trend.
    """
    window = _past_return_window(log_close, horizon_days, settings)
    centre = window.quantile(MEDIAN_LEVEL)
    return pd.DataFrame(
        {level: window.quantile(level) - centre for level in levels},
        index=log_close.index,
    )


def observation_counts(
    log_close: pd.Series, horizon_days: int, settings: BaselinesConfig
) -> pd.Series:
    """How many past h-day returns each origin's spread was estimated from.

    The observations overlap, so the independent count is roughly this divided
    by the horizon -- the number VALIDATION_SPEC.md section 4.3 cares about.
    """
    return _past_return_window(log_close, horizon_days, settings).count()


class Baseline(ABC):
    """A naive forecast defined entirely by the location it predicts."""

    name: str

    def __init__(self, settings: BaselinesConfig) -> None:
        self.settings = settings

    @abstractmethod
    def location(self, log_close: pd.Series, horizon_days: int) -> pd.Series:
        """Predicted median log return at each origin. Must be causal."""

    def describe(self) -> str:
        return self.name

    def predict(
        self, close: pd.Series, horizon_days: int, levels: tuple[float, ...]
    ) -> BaselinePrediction:
        if horizon_days < 1:
            raise BaselineError("horizon_days must be >= 1")
        if not levels:
            raise BaselineError("at least one quantile level is required")
        ordered = tuple(sorted(levels))
        log_close = _log_close(close)
        location = self.location(log_close, horizon_days)
        spread = empirical_spread(log_close, horizon_days, ordered, self.settings)
        return BaselinePrediction(
            name=self.name,
            horizon_days=horizon_days,
            location=location,
            quantiles=spread.add(location, axis=0),
            levels=ordered,
        )


class EmpiricalQuantileBaseline(Baseline):
    """The next h days look like the last h-day periods did.

    Location is the **historical median** of past h-day log returns, so combined
    with the shared zero-centred spread this is simply the unconditional
    empirical distribution, uncentred.

    It exists because the other three all predict a median derived from a *mean*
    or from zero, and none of them predicts the quantity that minimises absolute
    error: the median itself. BTC's h-day return distribution is strongly
    right-skewed at long horizons, so its mean and its median are far apart, and
    a forecast family that cannot express "the typical year was up 20%" is
    missing an obvious hypothesis rather than rejecting it.
    """

    name = "empirical"

    def location(self, log_close: pd.Series, horizon_days: int) -> pd.Series:
        window = _past_return_window(log_close, horizon_days, self.settings)
        return window.quantile(MEDIAN_LEVEL).rename("location")


class NoChangeBaseline(Baseline):
    """The price is a driftless random walk: the best guess is today's price."""

    name = "no_change"

    def location(self, log_close: pd.Series, horizon_days: int) -> pd.Series:
        return pd.Series(0.0, index=log_close.index, name="location")


class DriftBaseline(Baseline):
    """Random walk with drift: extrapolate the long-run average daily return."""

    name = "drift"

    def location(self, log_close: pd.Series, horizon_days: int) -> pd.Series:
        daily = log_close.diff(1)
        mean_daily = daily.expanding(min_periods=self.settings.drift_min_samples).mean()
        return (mean_daily * horizon_days).rename("location")

    def describe(self) -> str:
        return f"{self.name}(min_samples={self.settings.drift_min_samples})"


class RollingReturnBaseline(Baseline):
    """Recent momentum continues: extrapolate the recent average daily return."""

    name = "rolling_return"

    def location(self, log_close: pd.Series, horizon_days: int) -> pd.Series:
        window = self.settings.rolling_return_window_days
        daily = log_close.diff(1)
        mean_daily = daily.rolling(window, min_periods=window).mean()
        return (mean_daily * horizon_days).rename("location")

    def describe(self) -> str:
        return f"{self.name}(window={self.settings.rolling_return_window_days}d)"


BASELINE_CLASSES: dict[str, type[Baseline]] = {
    NoChangeBaseline.name: NoChangeBaseline,
    DriftBaseline.name: DriftBaseline,
    RollingReturnBaseline.name: RollingReturnBaseline,
    EmpiricalQuantileBaseline.name: EmpiricalQuantileBaseline,
}

# The reference every "improvement vs baseline" number is measured against.
REFERENCE_BASELINE: str = NoChangeBaseline.name


def build_baselines(settings: BaselinesConfig) -> list[Baseline]:
    """Instantiate the baselines named in ``baselines.enabled``."""
    unknown = [name for name in settings.enabled if name not in BASELINE_CLASSES]
    if unknown:
        raise BaselineError(
            f"unknown baselines in config: {unknown}; "
            f"available: {sorted(BASELINE_CLASSES)}"
        )
    return [BASELINE_CLASSES[name](settings) for name in settings.enabled]


def common_valid_origins(
    predictions: list[BaselinePrediction], candidates: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """Origins every baseline can score, so the comparison is like-for-like.

    The baselines warm up at different speeds -- `no_change` is defined from the
    first row, `drift` needs ``drift_min_samples`` observations. Evaluating each
    on its own valid range would compare different time periods and call the
    difference model quality.
    """
    shared = candidates
    for prediction in predictions:
        shared = shared.intersection(prediction.valid_origins())
    return shared.sort_values()
