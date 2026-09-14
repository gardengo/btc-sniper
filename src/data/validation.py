"""Market-data quality validation.

Implements the checks listed in ARCHITECTURE.md section 2.2 and the pass/fail
rules of DATA_SPEC.md section 7.

Two severities matter operationally:

``error``
    A structural defect. The daily pipeline must stop; OPERATING_SPEC.md
    section 9 forbids generating a forecast on incomplete data.
``warning``
    Something worth seeing but not a reason to halt. Large price jumps land
    here on purpose: DATA_SPEC.md section 7 explicitly forbids deleting
    legitimate crash/pump candles.

This module is pure. It reads a DataFrame and returns results; persisting them
is the caller's job.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import DataQualityConfig
from src.utils.logging import get_logger
from src.utils.timeutils import (
    MS_PER_DAY,
    last_closed_daily_open_ms,
    ms_to_date_str,
    utc_now_iso,
)

logger = get_logger(__name__)

SEVERITY_INFO: str = "info"
SEVERITY_WARNING: str = "warning"
SEVERITY_ERROR: str = "error"
STATUS_PASS: str = "pass"
STATUS_FAIL: str = "fail"

MAX_LISTED_EXAMPLES: int = 20


@dataclass(frozen=True)
class QualityCheckRow:
    """One check outcome, shaped for the ``data_quality_checks`` table."""

    run_id: str
    checked_at: str
    source: str
    symbol: str
    timeframe: str
    check_name: str
    severity: str
    status: str
    message: str
    details: dict[str, Any] | None = None
    rows_checked: int = 0
    range_start: str | None = None
    range_end: str | None = None

    @property
    def is_blocking(self) -> bool:
        """True when this outcome must stop the pipeline."""
        return self.status == STATUS_FAIL and self.severity == SEVERITY_ERROR


@dataclass
class QualityReport:
    """Collected outcomes for one validation run."""

    run_id: str
    source: str
    symbol: str
    timeframe: str
    checked_at: str
    rows: list[QualityCheckRow] = field(default_factory=list)

    def add(
        self,
        check_name: str,
        *,
        passed: bool,
        message: str,
        severity_on_fail: str = SEVERITY_ERROR,
        details: dict[str, Any] | None = None,
        rows_checked: int = 0,
        range_start: str | None = None,
        range_end: str | None = None,
    ) -> QualityCheckRow:
        """Record one outcome and return it."""
        row = QualityCheckRow(
            run_id=self.run_id,
            checked_at=self.checked_at,
            source=self.source,
            symbol=self.symbol,
            timeframe=self.timeframe,
            check_name=check_name,
            severity=SEVERITY_INFO if passed else severity_on_fail,
            status=STATUS_PASS if passed else STATUS_FAIL,
            message=message,
            details=details,
            rows_checked=rows_checked,
            range_start=range_start,
            range_end=range_end,
        )
        self.rows.append(row)
        log = logger.info if passed else (
            logger.error if severity_on_fail == SEVERITY_ERROR else logger.warning
        )
        log("[%s] %s: %s", "PASS" if passed else "FAIL", check_name, message)
        return row

    @property
    def errors(self) -> list[QualityCheckRow]:
        return [row for row in self.rows if row.is_blocking]

    @property
    def warnings(self) -> list[QualityCheckRow]:
        return [
            row
            for row in self.rows
            if row.status == STATUS_FAIL and row.severity == SEVERITY_WARNING
        ]

    @property
    def passed_checks(self) -> list[QualityCheckRow]:
        return [row for row in self.rows if row.status == STATUS_PASS]

    @property
    def ok(self) -> bool:
        """True when no blocking error was recorded."""
        return not self.errors

    def summary(self) -> str:
        return (
            f"{len(self.passed_checks)} passed, {len(self.warnings)} warnings, "
            f"{len(self.errors)} errors"
        )


def new_run_id(prefix: str = "dq") -> str:
    """Short unique identifier for one validation run."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _examples(values: pd.Index | pd.Series) -> list[str]:
    listed = [
        value.strftime("%Y-%m-%d") if isinstance(value, pd.Timestamp) else str(value)
        for value in list(values)[:MAX_LISTED_EXAMPLES]
    ]
    return listed


def validate_ohlcv(
    frame: pd.DataFrame,
    *,
    source: str,
    symbol: str,
    timeframe: str,
    config: DataQualityConfig,
    run_id: str | None = None,
    now: datetime | None = None,
    expect_current: bool = True,
) -> QualityReport:
    """Run every OHLCV quality check over a date-indexed frame.

    ``expect_current`` controls the staleness check; a historical-only backfill
    can disable it.
    """
    report = QualityReport(
        run_id=run_id or new_run_id(),
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        checked_at=utc_now_iso(),
    )
    row_count = len(frame)
    range_start = frame.index.min().strftime("%Y-%m-%d") if row_count else None
    range_end = frame.index.max().strftime("%Y-%m-%d") if row_count else None

    def add(name: str, passed: bool, message: str, **kwargs: Any) -> None:
        report.add(
            name,
            passed=passed,
            message=message,
            rows_checked=row_count,
            range_start=range_start,
            range_end=range_end,
            **kwargs,
        )

    if row_count == 0:
        add("non_empty", False, "no rows available for validation")
        return report
    add("non_empty", True, f"{row_count} rows between {range_start} and {range_end}")

    price_columns = ["open", "high", "low", "close"]

    # --- duplicates -------------------------------------------------------
    if config.require_no_duplicate_keys:
        duplicated = frame.index[frame.index.duplicated(keep=False)].unique()
        add(
            "duplicate_primary_key",
            len(duplicated) == 0,
            "no duplicate candle dates"
            if len(duplicated) == 0
            else f"{len(duplicated)} duplicated candle dates",
            details={"dates": _examples(duplicated)} if len(duplicated) else None,
        )

    # --- ordering ---------------------------------------------------------
    if config.require_monotonic_open_time:
        monotonic = bool(frame["open_time_ms"].is_monotonic_increasing)
        add(
            "monotonic_open_time",
            monotonic,
            "open_time is strictly increasing"
            if monotonic
            else "open_time is not monotonically increasing",
        )

    # --- required fields present -----------------------------------------
    missing_mask = frame[price_columns].isna().any(axis=1)
    missing_dates = frame.index[missing_mask]
    add(
        "missing_ohlc_values",
        len(missing_dates) == 0,
        "all OHLC values present"
        if len(missing_dates) == 0
        else f"{len(missing_dates)} rows missing an OHLC value",
        details={"dates": _examples(missing_dates)} if len(missing_dates) else None,
    )

    # --- finiteness -------------------------------------------------------
    numeric = frame[price_columns + ["volume"]].to_numpy(dtype="float64", na_value=np.nan)
    non_finite_mask = ~np.isfinite(numeric).all(axis=1)
    non_finite_dates = frame.index[non_finite_mask]
    add(
        "non_finite_values",
        len(non_finite_dates) == 0,
        "all price/volume values are finite"
        if len(non_finite_dates) == 0
        else f"{len(non_finite_dates)} rows contain NaN or infinite values",
        details={"dates": _examples(non_finite_dates)} if len(non_finite_dates) else None,
    )

    # --- OHLC relationship ------------------------------------------------
    if config.require_valid_ohlc_relationship:
        body_high = frame[["open", "close"]].max(axis=1)
        body_low = frame[["open", "close"]].min(axis=1)
        invalid = (frame["high"] < body_high) | (frame["low"] > body_low) | (
            frame["high"] < frame["low"]
        )
        invalid_dates = frame.index[invalid.fillna(True)]
        add(
            "ohlc_relationship",
            len(invalid_dates) == 0,
            "low <= open/close <= high holds on every row"
            if len(invalid_dates) == 0
            else f"{len(invalid_dates)} rows violate the OHLC relationship",
            details={"dates": _examples(invalid_dates)} if len(invalid_dates) else None,
        )

    # --- non positive prices ---------------------------------------------
    non_positive = (frame[price_columns] <= 0).any(axis=1)
    non_positive_dates = frame.index[non_positive.fillna(True)]
    add(
        "positive_prices",
        len(non_positive_dates) == 0,
        "all prices are positive"
        if len(non_positive_dates) == 0
        else f"{len(non_positive_dates)} rows contain a non-positive price",
        details={"dates": _examples(non_positive_dates)} if len(non_positive_dates) else None,
    )

    # --- volume -----------------------------------------------------------
    if config.require_non_negative_volume:
        negative_volume_dates = frame.index[frame["volume"] < 0]
        add(
            "non_negative_volume",
            len(negative_volume_dates) == 0,
            "volume is non-negative everywhere"
            if len(negative_volume_dates) == 0
            else f"{len(negative_volume_dates)} rows have negative volume",
            details={"dates": _examples(negative_volume_dates)}
            if len(negative_volume_dates)
            else None,
        )

    if config.zero_volume_warn:
        zero_volume_dates = frame.index[frame["volume"] == 0]
        add(
            "zero_volume_days",
            len(zero_volume_dates) == 0,
            "no zero-volume days"
            if len(zero_volume_dates) == 0
            else f"{len(zero_volume_dates)} zero-volume days",
            severity_on_fail=SEVERITY_WARNING,
            details={"dates": _examples(zero_volume_dates)} if len(zero_volume_dates) else None,
        )

    # --- calendar completeness -------------------------------------------
    expected_index = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    missing_days = expected_index.difference(frame.index)
    missing_ratio = len(missing_days) / len(expected_index)
    add(
        "missing_daily_candles",
        missing_ratio <= config.max_missing_day_ratio,
        f"{len(missing_days)} missing daily candles "
        f"({missing_ratio:.4%} of {len(expected_index)} expected days; "
        f"threshold {config.max_missing_day_ratio:.4%})",
        details={
            "missing_count": int(len(missing_days)),
            "missing_ratio": float(missing_ratio),
            "dates": _examples(missing_days),
        }
        if len(missing_days)
        else None,
    )
    if 0 < len(missing_days) and missing_ratio <= config.max_missing_day_ratio:
        report.add(
            "missing_daily_candles_detail",
            passed=False,
            message=(
                f"{len(missing_days)} missing daily candles are within the configured "
                "tolerance but are reported for visibility"
            ),
            severity_on_fail=SEVERITY_WARNING,
            details={"dates": _examples(missing_days)},
            rows_checked=row_count,
            range_start=range_start,
            range_end=range_end,
        )

    # --- extreme moves (warning only) ------------------------------------
    log_return = np.log(frame["close"] / frame["close"].shift(1))
    extreme_mask = log_return.abs() > config.extreme_daily_log_return_warn
    extreme_dates = frame.index[extreme_mask.fillna(False)]
    add(
        "extreme_daily_move",
        len(extreme_dates) == 0,
        "no daily move exceeds the warning threshold"
        if len(extreme_dates) == 0
        else (
            f"{len(extreme_dates)} days move more than "
            f"{config.extreme_daily_log_return_warn:.0%} in log terms "
            "(reported, never removed)"
        ),
        severity_on_fail=SEVERITY_WARNING,
        details={
            "dates": _examples(extreme_dates),
            "max_abs_log_return": float(log_return.abs().max()),
        }
        if len(extreme_dates)
        else None,
    )

    # --- staleness --------------------------------------------------------
    if expect_current:
        expected_last_ms = last_closed_daily_open_ms(now)
        actual_last_ms = int(frame["open_time_ms"].max())
        lag_days = (expected_last_ms - actual_last_ms) / MS_PER_DAY
        add(
            "series_freshness",
            lag_days <= config.max_staleness_days,
            f"last stored candle is {ms_to_date_str(actual_last_ms)}, "
            f"last closed candle is {ms_to_date_str(expected_last_ms)} "
            f"(lag {lag_days:.0f}d, threshold {config.max_staleness_days}d)",
            details={
                "last_stored_date": ms_to_date_str(actual_last_ms),
                "last_closed_date": ms_to_date_str(expected_last_ms),
                "lag_days": float(lag_days),
            },
        )

    return report


def compare_exchanges(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    *,
    config: DataQualityConfig,
    report: QualityReport,
    primary_label: str = "binance",
    secondary_label: str = "coinbase",
) -> QualityReport:
    """Cross-exchange close-price sanity check.

    Never blocking: DATA_SPEC.md section 7 does not list cross-exchange
    disagreement among the hard failure conditions, and the two venues trade
    different instruments (BTCUSDT vs BTC-USD), so small spreads are normal.
    """
    if primary.empty or secondary.empty:
        report.add(
            "cross_exchange_consistency",
            passed=False,
            message="cross-exchange check skipped: one of the series is empty",
            severity_on_fail=SEVERITY_WARNING,
        )
        return report

    window_start = primary.index.max() - pd.Timedelta(days=config.cross_exchange_check_days)
    joined = (
        primary.loc[primary.index >= window_start, ["close"]]
        .rename(columns={"close": primary_label})
        .join(
            secondary.loc[secondary.index >= window_start, ["close"]].rename(
                columns={"close": secondary_label}
            ),
            how="inner",
        )
    )
    if joined.empty:
        report.add(
            "cross_exchange_consistency",
            passed=False,
            message="cross-exchange check skipped: no overlapping dates in the window",
            severity_on_fail=SEVERITY_WARNING,
        )
        return report

    diff_pct = (
        (joined[primary_label] - joined[secondary_label]).abs()
        / joined[secondary_label]
        * 100
    )
    worst = float(diff_pct.max())
    worst_date = diff_pct.idxmax().strftime("%Y-%m-%d")
    breaches = diff_pct[diff_pct > config.cross_exchange_close_diff_warn_pct]
    report.add(
        "cross_exchange_consistency",
        passed=len(breaches) == 0,
        message=(
            f"compared {len(joined)} overlapping days; median diff "
            f"{float(diff_pct.median()):.3f}%, worst {worst:.3f}% on {worst_date} "
            f"(warn threshold {config.cross_exchange_close_diff_warn_pct}%)"
        ),
        severity_on_fail=SEVERITY_WARNING,
        details={
            "compared_days": int(len(joined)),
            "median_diff_pct": float(diff_pct.median()),
            "max_diff_pct": worst,
            "max_diff_date": worst_date,
            "breach_count": int(len(breaches)),
            "breach_dates": _examples(breaches.index),
        },
        rows_checked=int(len(joined)),
        range_start=joined.index.min().strftime("%Y-%m-%d"),
        range_end=joined.index.max().strftime("%Y-%m-%d"),
    )
    return report
