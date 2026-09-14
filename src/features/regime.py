"""Market-regime labelling.

VALIDATION_SPEC.md section 6 requires every validation report to be broken down
by regime, with at minimum: bull, bear, recovery, sideways, high_volatility and
low_volatility.

Two independent axes are produced:

direction
    ``bull`` / ``bear`` / ``recovery`` / ``sideways``, from the drawdown against
    the trailing 1-year high combined with the trailing 90-day return.
volatility
    ``high_volatility`` / ``normal_volatility`` / ``low_volatility``, from where
    the current 30-day realised volatility sits inside its own trailing
    distribution.

Both axes are **causal**: the label for day ``t`` uses only data up to ``t``.
That matters twice over -- it keeps the one-hot columns usable as model
features, and it keeps a regime-conditioned evaluation honest, because a label
computed with hindsight would leak the future into the performance breakdown.

CLAUDE.md section 7: no four-year-cycle assumption is encoded anywhere here.
The labels are descriptive statistics of the realised path, nothing more.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.features import indicators as ind

DIRECTION_BULL: str = "bull"
DIRECTION_BEAR: str = "bear"
DIRECTION_RECOVERY: str = "recovery"
DIRECTION_SIDEWAYS: str = "sideways"
DIRECTION_LABELS: tuple[str, ...] = (
    DIRECTION_BULL,
    DIRECTION_BEAR,
    DIRECTION_RECOVERY,
    DIRECTION_SIDEWAYS,
)

VOLATILITY_HIGH: str = "high_volatility"
VOLATILITY_NORMAL: str = "normal_volatility"
VOLATILITY_LOW: str = "low_volatility"
VOLATILITY_LABELS: tuple[str, ...] = (VOLATILITY_HIGH, VOLATILITY_LOW)
ALL_VOLATILITY_LABELS: tuple[str, ...] = (
    VOLATILITY_HIGH,
    VOLATILITY_NORMAL,
    VOLATILITY_LOW,
)


def _param(params: Mapping[str, Any], key: str, default: Any) -> Any:
    value = params.get(key, default)
    return default if value is None else value


def compute_regime_labels(
    frame: pd.DataFrame, params: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """Label every day with its direction and volatility regime.

    Returns a frame with the intermediate statistics alongside the two label
    columns, so a report can show *why* a day was labelled the way it was.
    """
    settings = dict(params or {})
    close = frame["close"]

    fast_window = int(_param(settings, "trend_fast_window", 50))
    slow_window = int(_param(settings, "trend_slow_window", 200))
    drawdown_window = int(_param(settings, "drawdown_window", 365))
    return_window = int(_param(settings, "direction_return_window", 90))
    return_threshold = float(_param(settings, "direction_return_threshold", 0.10))
    bear_drawdown = float(_param(settings, "bear_drawdown_threshold", 0.20))
    volatility_window = int(_param(settings, "volatility_window", 30))
    rank_window = int(_param(settings, "volatility_rank_window", 365))
    rank_min_periods = int(_param(settings, "volatility_rank_min_periods", 180))
    high_quantile = float(_param(settings, "high_volatility_quantile", 0.70))
    low_quantile = float(_param(settings, "low_volatility_quantile", 0.30))

    trend_score = ind.safe_log_ratio(ind.sma(close, fast_window), ind.sma(close, slow_window))
    drawdown = ind.safe_log_ratio(close, ind.rolling_max(close, drawdown_window))
    direction_return = ind.log_return(close, return_window)

    daily_return = ind.log_return(close, 1)
    realized_vol = ind.realized_volatility(daily_return, volatility_window)
    volatility_rank = ind.rolling_percentile_rank(
        realized_vol, rank_window, min_periods=rank_min_periods
    )

    # Direction: deep drawdown splits bear from recovery; otherwise the 90-day
    # return splits bull from sideways.
    deep_drawdown = drawdown <= -abs(bear_drawdown)
    rising = direction_return > return_threshold
    falling = direction_return < -return_threshold

    direction = pd.Series(np.nan, index=frame.index, dtype="object")
    known = drawdown.notna() & direction_return.notna()
    direction[known & deep_drawdown & rising] = DIRECTION_RECOVERY
    direction[known & deep_drawdown & ~rising] = DIRECTION_BEAR
    direction[known & ~deep_drawdown & rising] = DIRECTION_BULL
    direction[known & ~deep_drawdown & falling] = DIRECTION_BEAR
    direction[known & ~deep_drawdown & ~rising & ~falling] = DIRECTION_SIDEWAYS

    volatility = pd.Series(np.nan, index=frame.index, dtype="object")
    rank_known = volatility_rank.notna()
    volatility[rank_known] = VOLATILITY_NORMAL
    volatility[rank_known & (volatility_rank >= high_quantile)] = VOLATILITY_HIGH
    volatility[rank_known & (volatility_rank <= low_quantile)] = VOLATILITY_LOW

    return pd.DataFrame(
        {
            "trend_score": trend_score,
            "drawdown": drawdown,
            "direction_return": direction_return,
            "realized_vol": realized_vol,
            "volatility_rank": volatility_rank,
            "direction": direction,
            "volatility": volatility,
        },
        index=frame.index,
    )


def regime_summary(labels: pd.DataFrame) -> pd.DataFrame:
    """Day counts and share per regime label, for the data-quality report."""
    rows: list[dict[str, Any]] = []
    total = len(labels)
    for axis in ("direction", "volatility"):
        counts = labels[axis].value_counts(dropna=False)
        for label, count in counts.items():
            rows.append(
                {
                    "axis": axis,
                    "label": "unlabelled (warmup)" if pd.isna(label) else str(label),
                    "days": int(count),
                    "share": float(count) / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows)
