"""Causal technical-indicator primitives.

Every function here is **strictly causal**: the value at row ``t`` depends only
on rows ``<= t``. That is the core requirement of CLAUDE.md section 2.1 and
DATA_SPEC.md section 6, so this module deliberately avoids:

* ``shift(-n)`` / any negative shift
* ``rolling(..., center=True)``
* ``bfill()`` / ``interpolate()`` in a way that pulls a future value backwards

Warmup rows are left as NaN instead of being filled. A NaN says "not computable
from past data yet", which is the honest answer; filling it would invent
information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON: float = 1e-12


def safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """``log(numerator / denominator)`` with non-positive inputs mapped to NaN."""
    num = numerator.where(numerator > 0)
    den = denominator.where(denominator > 0)
    return np.log(num / den)


def log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    """Backward-looking log return over ``periods`` days."""
    if periods < 1:
        raise ValueError("periods must be >= 1")
    return safe_log_ratio(close, close.shift(periods))


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over a trailing window ending at the current row."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average; ``adjust=False`` keeps it recursive/causal."""
    if span < 1:
        raise ValueError("span must be >= 1")
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder_ema(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing (used by RSI and ATR)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Trailing sample standard deviation."""
    return series.rolling(window=window, min_periods=window).std(ddof=1)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing, in ``[0, 100]``."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    average_gain = wilder_ema(gain, window)
    average_loss = wilder_ema(loss, window)
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + relative_strength))
    # average_loss == 0 with a positive average gain is a pure uptrend -> RSI 100.
    result = result.where(~((average_loss == 0.0) & (average_gain > 0.0)), 100.0)
    return result.where(~((average_loss == 0.0) & (average_gain == 0.0)), 50.0)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: the classic three-way maximum against the previous close."""
    previous_close = close.shift(1)
    candidates = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return candidates.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    return wilder_ema(true_range(high, low, close), window)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line and histogram."""
    if fast >= slow:
        raise ValueError("fast span must be shorter than slow span")
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger middle, upper and lower bands."""
    middle = sma(close, window)
    deviation = rolling_std(close, window) * num_std
    return middle, middle + deviation, middle - deviation


def realized_volatility(
    daily_log_return: pd.Series, window: int, annualization_days: int = 365
) -> pd.Series:
    """Annualised realised volatility from trailing daily log returns."""
    return rolling_std(daily_log_return, window) * np.sqrt(annualization_days)


def rolling_percentile_rank(
    series: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    """Percentile rank of the current value within its own trailing window.

    The window *includes* the current observation and nothing after it, so the
    result is causal. Returns a value in ``[0, 1]``.
    """
    periods = min_periods if min_periods is not None else window

    def rank(values: np.ndarray) -> float:
        current = values[-1]
        if not np.isfinite(current):
            return np.nan
        finite = values[np.isfinite(values)]
        if finite.size < 2:
            return np.nan
        return float((finite <= current).sum() - 1) / float(finite.size - 1)

    return series.rolling(window=window, min_periods=periods).apply(rank, raw=True)


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    """Trailing maximum, current row included."""
    return series.rolling(window=window, min_periods=window).max()


def rolling_min(series: pd.Series, window: int) -> pd.Series:
    """Trailing minimum, current row included."""
    return series.rolling(window=window, min_periods=window).min()


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Trailing z-score of a series against its own rolling mean/std."""
    mean = series.rolling(window=window, min_periods=window).mean()
    std = rolling_std(series, window)
    return (series - mean) / std.replace(0.0, np.nan)
