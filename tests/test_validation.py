"""Data-quality validation tests.

Each test injects one specific defect and asserts that the corresponding check
fails with the right severity -- and, just as importantly, that defects which
DATA_SPEC.md says must NOT block the pipeline (a legitimate crash candle) only
produce a warning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.types import Candle, CandleError
from src.data.validation import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    compare_exchanges,
    new_run_id,
    validate_ohlcv,
)
from src.utils.config import AppConfig


def _validate(frame: pd.DataFrame, config: AppConfig, **kwargs):
    return validate_ohlcv(
        frame,
        source="binance",
        symbol="BTCUSDT",
        timeframe="1d",
        config=config.data_quality,
        run_id=new_run_id("test"),
        expect_current=False,
        **kwargs,
    )


def _check(report, name: str):
    matches = [row for row in report.rows if row.check_name == name]
    assert matches, f"check '{name}' was not run"
    return matches[0]


class TestCleanSeries:
    def test_clean_series_passes_every_blocking_check(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        report = _validate(ohlcv, app_config)
        assert report.ok, [row.message for row in report.errors]
        assert _check(report, "missing_daily_candles").status == "pass"
        assert _check(report, "ohlc_relationship").status == "pass"

    def test_empty_frame_fails_loudly(self, ohlcv: pd.DataFrame, app_config: AppConfig) -> None:
        report = _validate(ohlcv.iloc[0:0], app_config)
        assert not report.ok
        assert _check(report, "non_empty").status == "fail"


class TestBlockingDefects:
    def test_missing_candle_gap_is_detected(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        with_gap = ohlcv.drop(ohlcv.index[100:140])
        report = _validate(with_gap, app_config)
        row = _check(report, "missing_daily_candles")
        assert row.status == "fail"
        assert row.severity == SEVERITY_ERROR
        assert not report.ok
        assert row.details["missing_count"] == 40

    def test_broken_ohlc_relationship_is_detected(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        broken = ohlcv.copy()
        broken.loc[broken.index[50], "high"] = broken.loc[broken.index[50], "low"] * 0.5
        report = _validate(broken, app_config)
        row = _check(report, "ohlc_relationship")
        assert row.status == "fail"
        assert row.severity == SEVERITY_ERROR
        assert not report.ok

    def test_negative_volume_is_detected(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        broken = ohlcv.copy()
        broken.loc[broken.index[10], "volume"] = -1.0
        report = _validate(broken, app_config)
        assert _check(report, "non_negative_volume").status == "fail"
        assert not report.ok

    def test_nan_close_is_detected(self, ohlcv: pd.DataFrame, app_config: AppConfig) -> None:
        broken = ohlcv.copy()
        broken.loc[broken.index[20], "close"] = np.nan
        report = _validate(broken, app_config)
        assert _check(report, "missing_ohlc_values").status == "fail"
        assert _check(report, "non_finite_values").status == "fail"
        assert not report.ok

    def test_duplicate_dates_are_detected(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        duplicated = pd.concat([ohlcv, ohlcv.iloc[[5]]]).sort_index()
        report = _validate(duplicated, app_config)
        assert _check(report, "duplicate_primary_key").status == "fail"
        assert not report.ok

    def test_non_positive_price_is_detected(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        broken = ohlcv.copy()
        broken.loc[broken.index[30], "low"] = 0.0
        report = _validate(broken, app_config)
        assert _check(report, "positive_prices").status == "fail"
        assert not report.ok


class TestNonBlockingWarnings:
    def test_legitimate_crash_warns_but_does_not_block(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        """DATA_SPEC.md section 7: never blindly delete a real crash candle."""
        crashed = ohlcv.copy()
        day = crashed.index[200]
        for column in ("open", "high", "low", "close"):
            crashed.loc[day:, column] = crashed.loc[day:, column] * 0.55

        report = _validate(crashed, app_config)
        row = _check(report, "extreme_daily_move")
        assert row.status == "fail"
        assert row.severity == SEVERITY_WARNING
        assert report.ok, "a large but legitimate price move must not block the pipeline"

    def test_stale_series_is_flagged_when_freshness_is_expected(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        report = validate_ohlcv(
            ohlcv,
            source="binance",
            symbol="BTCUSDT",
            timeframe="1d",
            config=app_config.data_quality,
            expect_current=True,
        )
        row = _check(report, "series_freshness")
        assert row.status == "fail"
        assert row.severity == SEVERITY_ERROR


class TestCrossExchange:
    def test_matching_exchanges_pass(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        report = _validate(ohlcv, app_config)
        secondary = ohlcv.copy()
        secondary["close"] = secondary["close"] * 1.001
        compare_exchanges(ohlcv, secondary, config=app_config.data_quality, report=report)
        assert _check(report, "cross_exchange_consistency").status == "pass"

    def test_diverging_exchanges_warn_without_blocking(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        report = _validate(ohlcv, app_config)
        secondary = ohlcv.copy()
        secondary["close"] = secondary["close"] * 1.25
        compare_exchanges(ohlcv, secondary, config=app_config.data_quality, report=report)
        row = _check(report, "cross_exchange_consistency")
        assert row.status == "fail"
        assert row.severity == SEVERITY_WARNING
        assert report.ok, "cross-exchange divergence is a warning, never a hard failure"


class TestCandleSelfValidation:
    def _candle(self, **overrides) -> Candle:
        base = dict(
            source="binance",
            symbol="BTCUSDT",
            timeframe="1d",
            open_time_ms=1583971200000,
            close_time_ms=1583971200000 + 86_399_999,
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=10.0,
        )
        base.update(overrides)
        return Candle(**base)

    def test_valid_candle_passes(self) -> None:
        self._candle().validate()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"high": 100.0, "close": 105.0},
            {"low": 106.0},
            {"close": -1.0},
            {"volume": -5.0},
            {"close": float("nan")},
            {"open_time_ms": 1583971200001},
            {"close_time_ms": 1583971199999},
        ],
    )
    def test_invalid_candle_is_rejected(self, overrides: dict) -> None:
        with pytest.raises(CandleError):
            self._candle(**overrides).validate()
