"""Shape-preserving interpolation of the forecast curve, for display only.

CLAUDE.md section 5 and MODEL_SPEC.md section 9 both require this and both
constrain it:

* PCHIP, not a natural cubic spline. A cubic spline through a sparse horizon grid
  overshoots between knots and invents local maxima the model never predicted --
  on a price chart that reads as a forecast of a rally that nothing produced.
  PCHIP is monotone-preserving between knots and cannot overshoot.
* The result is **visualisation only**. Interpolated values are never stored as
  forecast points and never scored. `interpolate_quantiles` returns a frame the
  chart consumes; nothing writes it to `forecast_points`.

Two details that matter
-----------------------
**Interpolate in log-return space, then convert.** The model's output is a log
return; interpolating there and exponentiating keeps every displayed price
positive by construction, and makes the curve's shape independent of the price
level.

**Anchor at h=0.** A forecast has to start from the last closed price, so
``(0, 0.0)`` is included as a knot for every quantile. Without it the curve
starts at h=1 and the chart shows a gap, or worse, a jump, exactly at the NOW
marker.

Quantiles are interpolated independently, so the curves can in principle touch
between knots; the result is re-sorted per date, which is what MODEL_SPEC.md
section 9 requires -- no displayed price outside the band it belongs to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from src.utils.logging import get_logger

logger = get_logger(__name__)

ORIGIN_ANCHOR: tuple[int, float] = (0, 0.0)


class InterpolationError(ValueError):
    """Raised when a forecast curve cannot be interpolated."""


def daily_grid(max_horizon_days: int, *, include_origin: bool = True) -> np.ndarray:
    """Day offsets the chart draws, from the origin to ``max_horizon_days``."""
    if max_horizon_days < 1:
        raise InterpolationError("max_horizon_days must be >= 1")
    start = 0 if include_origin else 1
    return np.arange(start, max_horizon_days + 1, dtype="int64")


def interpolate_curve(
    horizons: np.ndarray, values: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """PCHIP through ``(horizons, values)``, evaluated at ``targets``."""
    if horizons.size != values.size:
        raise InterpolationError("horizons and values must have the same length")
    if horizons.size < 2:
        raise InterpolationError("at least two knots are needed to interpolate")
    if not np.all(np.diff(horizons) > 0):
        raise InterpolationError("horizons must be strictly increasing")
    if not np.isfinite(values).all():
        raise InterpolationError("cannot interpolate through non-finite values")
    return PchipInterpolator(horizons, values, extrapolate=False)(targets)


def interpolate_quantiles(
    quantiles: pd.DataFrame,
    *,
    max_horizon_days: int | None = None,
    anchor_at_origin: bool = True,
) -> pd.DataFrame:
    """Daily log-return curves for each quantile level.

    ``quantiles`` is indexed by horizon in days with one column per quantile
    level. The result is indexed by day offset over the whole daily grid.
    """
    if quantiles.empty:
        raise InterpolationError("cannot interpolate an empty forecast")
    knots = quantiles.sort_index()
    if knots.index.min() < 1:
        raise InterpolationError("forecast horizons must start at 1 day or later")

    horizons = knots.index.to_numpy(dtype="float64")
    limit = int(max_horizon_days or knots.index.max())
    targets = daily_grid(limit, include_origin=anchor_at_origin).astype("float64")

    curves: dict[float, np.ndarray] = {}
    for level in knots.columns:
        values = knots[level].to_numpy(dtype="float64")
        if anchor_at_origin:
            grid = np.concatenate([[float(ORIGIN_ANCHOR[0])], horizons])
            series = np.concatenate([[ORIGIN_ANCHOR[1]], values])
        else:
            grid, series = horizons, values
        curves[level] = interpolate_curve(grid, series, targets)

    frame = pd.DataFrame(curves, index=pd.Index(targets.astype("int64"), name="day"))
    return enforce_band_order(frame)


def enforce_band_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort each row's quantiles ascending.

    MODEL_SPEC.md section 9: no displayed value may sit outside the band it
    belongs to. Independent interpolation can produce a touch between knots even
    when every knot is ordered, so the guarantee is applied to the drawn curve
    rather than assumed from the inputs.
    """
    ordered = sorted(frame.columns)
    values = frame[ordered].to_numpy(copy=True)
    values.sort(axis=1)
    result = frame.copy()
    result[ordered] = values
    return result


def to_price_curves(
    curves: pd.DataFrame, origin_close: float, origin_date: pd.Timestamp
) -> pd.DataFrame:
    """Attach calendar dates and convert the log-return curves to prices."""
    if origin_close <= 0:
        raise InterpolationError("origin close must be positive")
    prices = curves.apply(lambda column: origin_close * np.exp(column))
    prices.insert(
        0,
        "date",
        [origin_date + pd.Timedelta(days=int(day)) for day in curves.index],
    )
    return prices


def interpolated_forecast_curve(
    quantiles: pd.DataFrame,
    origin_close: float,
    origin_date: pd.Timestamp,
    *,
    max_horizon_days: int | None = None,
) -> pd.DataFrame:
    """Chart-ready daily price bands. Never persisted, never scored."""
    curves = interpolate_quantiles(quantiles, max_horizon_days=max_horizon_days)
    return to_price_curves(curves, origin_close, origin_date)
