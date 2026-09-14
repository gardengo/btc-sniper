"""Feature groups.

MODEL_SPEC.md section 5 requires features to be grouped so they can be ablated
in the order price/return -> trend -> volatility -> volume -> regime. Each
builder below returns a frame for exactly one group, and the pipeline
concatenates the groups that are enabled in config.

Scaling conventions used throughout:

* Ratios between two prices are expressed as log ratios, so they are symmetric
  and roughly stationary across BTC's several orders of magnitude of price.
* Levels that would otherwise be non-stationary (a raw SMA, a raw volume) are
  always divided by a contemporaneous level before being emitted.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.features import indicators as ind

GROUP_PRICE_RETURN: str = "price_return"
GROUP_TREND: str = "trend"
GROUP_VOLATILITY: str = "volatility"
GROUP_VOLUME: str = "volume"
GROUP_REGIME: str = "regime"

ALL_GROUPS: tuple[str, ...] = (
    GROUP_PRICE_RETURN,
    GROUP_TREND,
    GROUP_VOLATILITY,
    GROUP_VOLUME,
    GROUP_REGIME,
)


def _param(params: Mapping[str, Any], key: str, default: Any) -> Any:
    value = params.get(key, default)
    return default if value is None else value


def build_price_return_features(
    frame: pd.DataFrame, params: Mapping[str, Any]
) -> pd.DataFrame:
    """Raw return, distance-to-extreme and drawdown features."""
    close, high, low = frame["close"], frame["high"], frame["low"]
    out: dict[str, pd.Series] = {}

    for period in _param(params, "return_periods", [1, 3, 7, 14, 30, 90]):
        out[f"ret_{period}d"] = ind.log_return(close, int(period))

    for window in _param(params, "extreme_windows", [30, 90, 365]):
        window = int(window)
        out[f"dist_high_{window}d"] = ind.safe_log_ratio(close, ind.rolling_max(high, window))
        out[f"dist_low_{window}d"] = ind.safe_log_ratio(close, ind.rolling_min(low, window))

    for window in _param(params, "drawdown_windows", [90, 365]):
        window = int(window)
        out[f"drawdown_{window}d"] = ind.safe_log_ratio(close, ind.rolling_max(close, window))

    return pd.DataFrame(out, index=frame.index)


def build_trend_features(frame: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    """Moving-average position, slope, MACD and RSI."""
    close = frame["close"]
    out: dict[str, pd.Series] = {}

    sma_windows = [int(w) for w in _param(params, "sma_windows", [7, 14, 30, 50, 100, 200])]
    ema_windows = [int(w) for w in _param(params, "ema_windows", [12, 26, 50, 200])]
    slope_window = int(_param(params, "ma_slope_window", 20))

    simple_averages = {window: ind.sma(close, window) for window in sma_windows}
    for window, series in simple_averages.items():
        out[f"price_sma_ratio_{window}d"] = ind.safe_log_ratio(close, series)

    for window in ema_windows:
        out[f"price_ema_ratio_{window}d"] = ind.safe_log_ratio(close, ind.ema(close, window))

    for window in sorted(simple_averages):
        if window >= 50:
            series = simple_averages[window]
            out[f"sma_slope_{window}d"] = ind.safe_log_ratio(series, series.shift(slope_window))

    if 50 in simple_averages and 200 in simple_averages:
        out["sma_ratio_50_200"] = ind.safe_log_ratio(
            simple_averages[50], simple_averages[200]
        )

    macd_params = _param(params, "macd", {"fast": 12, "slow": 26, "signal": 9})
    macd_line, signal_line, histogram = ind.macd(
        close,
        fast=int(macd_params.get("fast", 12)),
        slow=int(macd_params.get("slow", 26)),
        signal=int(macd_params.get("signal", 9)),
    )
    out["macd_line_norm"] = macd_line / close
    out["macd_signal_norm"] = signal_line / close
    out["macd_hist_norm"] = histogram / close

    rsi_window = int(_param(params, "rsi_window", 14))
    # Centred on 0 and scaled to roughly [-1, 1] so trees see a symmetric split point.
    out[f"rsi_{rsi_window}"] = (ind.rsi(close, rsi_window) - 50.0) / 50.0

    return pd.DataFrame(out, index=frame.index)


def build_volatility_features(
    frame: pd.DataFrame, params: Mapping[str, Any]
) -> pd.DataFrame:
    """Realised volatility, ATR, intraday range and Bollinger geometry."""
    close, high, low = frame["close"], frame["high"], frame["low"]
    out: dict[str, pd.Series] = {}

    daily_return = ind.log_return(close, 1)
    annualization = int(_param(params, "realized_vol_annualization_days", 365))
    windows = [int(w) for w in _param(params, "volatility_windows", [7, 14, 30, 90])]
    volatilities = {
        window: ind.realized_volatility(daily_return, window, annualization)
        for window in windows
    }
    for window, series in volatilities.items():
        out[f"realized_vol_{window}d"] = series

    ordered = sorted(volatilities)
    for short, long in zip(ordered, ordered[1:]):
        out[f"vol_ratio_{short}_{long}"] = ind.safe_log_ratio(
            volatilities[short], volatilities[long]
        )

    atr_window = int(_param(params, "atr_window", 14))
    out[f"atr_{atr_window}d_pct"] = ind.atr(high, low, close, atr_window) / close

    hl_range = (high - low) / close
    out["hl_range_pct"] = hl_range
    out["hl_range_pct_ma14"] = ind.sma(hl_range, 14)

    bollinger_params = _param(params, "bollinger", {"window": 20, "num_std": 2.0})
    bb_window = int(bollinger_params.get("window", 20))
    middle, upper, lower = ind.bollinger(
        close, bb_window, float(bollinger_params.get("num_std", 2.0))
    )
    band_width = (upper - lower).replace(0.0, np.nan)
    out["bb_width"] = band_width / middle
    out["bb_position"] = (close - lower) / band_width

    return pd.DataFrame(out, index=frame.index)


def build_volume_features(frame: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    """Volume level, dispersion and taker-flow features."""
    volume = frame["volume"]
    out: dict[str, pd.Series] = {}

    out["volume_change_1d"] = ind.safe_log_ratio(volume, volume.shift(1))

    windows = [int(w) for w in _param(params, "volume_windows", [7, 30, 90])]
    averages = {window: ind.sma(volume, window) for window in windows}
    for window, series in averages.items():
        out[f"volume_ratio_{window}d"] = ind.safe_log_ratio(volume, series)

    ordered = sorted(averages)
    for short, long in zip(ordered, ordered[1:]):
        out[f"volume_ma_ratio_{short}_{long}"] = ind.safe_log_ratio(
            averages[short], averages[long]
        )

    if "quote_volume" in frame:
        quote_volume = frame["quote_volume"]
        out["quote_volume_ratio_30d"] = ind.safe_log_ratio(
            quote_volume, ind.sma(quote_volume, 30)
        )

    if "trade_count" in frame:
        trade_count = frame["trade_count"].astype("float64")
        out["trade_count_ratio_30d"] = ind.safe_log_ratio(
            trade_count, ind.sma(trade_count, 30)
        )

    if "taker_buy_base" in frame:
        # Share of volume initiated by buyers; 0.5 is balanced flow.
        taker_share = frame["taker_buy_base"] / volume.replace(0.0, np.nan)
        taker_share = taker_share.where((taker_share >= 0.0) & (taker_share <= 1.0))
        out["taker_buy_share"] = taker_share - 0.5
        out["taker_buy_share_ma7_dev"] = taker_share - ind.sma(taker_share, 7)

    return pd.DataFrame(out, index=frame.index)


def build_regime_features(frame: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    """Numeric regime descriptors plus one-hot regime membership.

    The string labels themselves live in :mod:`src.features.regime`; only
    numeric columns can be stored in the ``features`` table, and MODEL_SPEC.md
    section 4 treats the labels as an evaluation artefact first and a model
    feature second.
    """
    from src.features.regime import compute_regime_labels, DIRECTION_LABELS, VOLATILITY_LABELS

    labels = compute_regime_labels(frame, params)
    out: dict[str, pd.Series] = {
        "regime_trend_score": labels["trend_score"],
        "regime_drawdown": labels["drawdown"],
        "regime_direction_return": labels["direction_return"],
        "regime_vol_rank": labels["volatility_rank"],
    }
    for label in DIRECTION_LABELS:
        out[f"regime_is_{label}"] = (labels["direction"] == label).astype("float64").where(
            labels["direction"].notna()
        )
    for label in VOLATILITY_LABELS:
        out[f"regime_is_{label}"] = (labels["volatility"] == label).astype("float64").where(
            labels["volatility"].notna()
        )
    return pd.DataFrame(out, index=frame.index)


GROUP_BUILDERS: dict[str, Any] = {
    GROUP_PRICE_RETURN: build_price_return_features,
    GROUP_TREND: build_trend_features,
    GROUP_VOLATILITY: build_volatility_features,
    GROUP_VOLUME: build_volume_features,
    GROUP_REGIME: build_regime_features,
}
