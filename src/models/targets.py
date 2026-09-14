"""Forecast target construction.

MODEL_SPEC.md section 1 defines the target::

    Y(t, h) = log(Close[t+h] / Close[t])

and the inverse used at inference time::

    P(t, h) = Close[t] * exp(Y_hat(t, h))

**This module is the only place in the codebase allowed to look forward in
time.** A target *must* read a future price -- that is what makes it a label.
Everything under `src/features/` is forbidden from doing so, and
`tests/test_leakage.py` enforces that separation by parsing the feature sources.
The safety property is therefore not "nobody shifts backwards" but "the forward
shift happens here, on the label, and never touches a feature column".

Two consequences follow and are enforced below:

* the last ``h`` rows of a horizon-``h`` target are ``NaN``; they are not yet
  observable and must never be filled, dropped silently, or carried forward.
* a target column must never be fed back into a feature matrix. `assert_disjoint`
  is a cheap guard for callers that hold both frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

TARGET_COLUMN_PREFIX: str = "target_log_return_"


class TargetError(RuntimeError):
    """Raised when targets cannot be built from the given series."""


def target_column(horizon_days: int) -> str:
    """Canonical column name for a horizon's target."""
    return f"{TARGET_COLUMN_PREFIX}{int(horizon_days)}d"


def _validate_close(close: pd.Series) -> pd.Series:
    if close.empty:
        raise TargetError("cannot build targets from an empty close series")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TargetError("close series must be indexed by date")
    if not close.index.is_monotonic_increasing:
        raise TargetError("close series must be sorted ascending by date")
    if close.index.has_duplicates:
        raise TargetError("close series index contains duplicate dates")
    values = close.astype("float64")
    if not np.isfinite(values.to_numpy()).all():
        raise TargetError("close series contains non-finite values")
    if (values <= 0).any():
        raise TargetError("close series contains non-positive prices")
    return values


def forward_log_return(close: pd.Series, horizon_days: int) -> pd.Series:
    """``log(Close[t+h] / Close[t])`` aligned on the forecast origin ``t``.

    The shift is positional, not calendar-based, which is only correct because
    the daily series is validated to have no calendar gaps upstream
    (`src/data/validation.py`). The last ``h`` entries are ``NaN``.
    """
    if horizon_days < 1:
        raise TargetError("horizon_days must be >= 1")
    values = _validate_close(close)
    log_close = np.log(values)
    # The one deliberate forward shift in the codebase; see the module docstring.
    future = log_close.shift(-horizon_days)
    target = future - log_close
    target.name = target_column(horizon_days)
    return target


def build_targets(close: pd.Series, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Wide target matrix: one column per horizon, indexed by forecast origin."""
    if not horizons:
        raise TargetError("horizons must not be empty")
    ordered = tuple(sorted({int(h) for h in horizons}))
    frame = pd.DataFrame(
        {target_column(h): forward_log_return(close, h) for h in ordered},
        index=close.index,
    )
    logger.debug(
        "built %d target columns over %d rows (%s..%s)",
        frame.shape[1],
        frame.shape[0],
        close.index.min().date(),
        close.index.max().date(),
    )
    return frame


def target_dates(origins: pd.DatetimeIndex, horizon_days: int) -> pd.DatetimeIndex:
    """Calendar dates the labels of ``origins`` resolve on."""
    if horizon_days < 1:
        raise TargetError("horizon_days must be >= 1")
    return origins + pd.Timedelta(days=horizon_days)


def observable_origins(
    close: pd.Series, horizon_days: int, *, as_of: pd.Timestamp | None = None
) -> pd.DatetimeIndex:
    """Origins whose horizon-``h`` label has already happened.

    An origin is scoreable only once ``t + h`` exists in the data. Origins past
    that point stay `pending` in the sense of VALIDATION_SPEC.md section 12.
    """
    if horizon_days < 1:
        raise TargetError("horizon_days must be >= 1")
    index = close.index
    cutoff = (as_of if as_of is not None else index.max()) - pd.Timedelta(
        days=horizon_days
    )
    return index[index <= cutoff]


def assert_disjoint(features: pd.DataFrame, targets: pd.DataFrame) -> None:
    """Fail if a target column has leaked into the feature matrix.

    Cheap enough to call before every fit. A target inside the feature matrix is
    a total leak -- the model would read the answer directly -- and it is an easy
    mistake to make when both frames are merged for convenience.
    """
    overlap = sorted(set(features.columns) & set(targets.columns))
    if overlap:
        raise TargetError(
            f"target columns present in the feature matrix: {overlap}"
        )
    suspicious = sorted(
        c for c in features.columns if str(c).startswith(TARGET_COLUMN_PREFIX)
    )
    if suspicious:
        raise TargetError(
            f"feature matrix contains target-shaped columns: {suspicious}"
        )
