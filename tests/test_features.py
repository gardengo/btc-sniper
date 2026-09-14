"""Feature pipeline and indicator correctness tests.

Leakage is covered separately in ``test_leakage.py``; this module checks that
the values are *right*, that the groups are ablatable as MODEL_SPEC.md section 5
requires, and that the pipeline refuses input it cannot safely process.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.features import indicators as ind
from src.features.groups import ALL_GROUPS
from src.features.pipeline import (
    FeaturePipelineError,
    build_features,
    resolve_groups,
)
from src.features.regime import (
    ALL_VOLATILITY_LABELS,
    DIRECTION_LABELS,
    compute_regime_labels,
    regime_summary,
)
from src.storage import repositories as repo
from src.storage.db import transaction
from src.utils.config import AppConfig


class TestIndicatorValues:
    def test_sma_matches_a_manual_mean(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ind.sma(series, 3)
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_log_return_is_the_log_of_the_ratio(self) -> None:
        series = pd.Series([100.0, 110.0, 121.0])
        result = ind.log_return(series, 1)
        assert result.iloc[1] == pytest.approx(np.log(1.1))
        assert ind.log_return(series, 2).iloc[2] == pytest.approx(np.log(1.21))

    def test_rsi_saturates_on_a_pure_uptrend(self) -> None:
        rising = pd.Series(np.linspace(100.0, 200.0, 60))
        assert ind.rsi(rising, 14).iloc[-1] == pytest.approx(100.0)

    def test_rsi_bottoms_on_a_pure_downtrend(self) -> None:
        falling = pd.Series(np.linspace(200.0, 100.0, 60))
        assert ind.rsi(falling, 14).iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_rsi_stays_inside_its_bounds(self, ohlcv: pd.DataFrame) -> None:
        values = ind.rsi(ohlcv["close"], 14).dropna()
        assert values.between(0.0, 100.0).all()

    def test_flat_series_gives_neutral_rsi(self) -> None:
        flat = pd.Series([100.0] * 40)
        assert ind.rsi(flat, 14).iloc[-1] == pytest.approx(50.0)

    def test_macd_histogram_is_line_minus_signal(self, ohlcv: pd.DataFrame) -> None:
        line, signal, histogram = ind.macd(ohlcv["close"])
        pd.testing.assert_series_equal(
            histogram.dropna(), (line - signal).dropna(), check_names=False
        )

    def test_bollinger_bands_are_ordered(self, ohlcv: pd.DataFrame) -> None:
        middle, upper, lower = ind.bollinger(ohlcv["close"], 20, 2.0)
        valid = middle.notna()
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()

    def test_atr_is_non_negative(self, ohlcv: pd.DataFrame) -> None:
        values = ind.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14).dropna()
        assert (values >= 0).all()

    def test_percentile_rank_is_bounded(self, ohlcv: pd.DataFrame) -> None:
        values = ind.rolling_percentile_rank(ohlcv["close"], 90, 30).dropna()
        assert values.between(0.0, 1.0).all()

    def test_percentile_rank_is_one_at_a_new_high(self) -> None:
        rising = pd.Series(np.linspace(100.0, 200.0, 100))
        assert ind.rolling_percentile_rank(rising, 30, 10).iloc[-1] == pytest.approx(1.0)

    def test_safe_log_ratio_maps_non_positive_to_nan(self) -> None:
        numerator = pd.Series([10.0, -1.0, 10.0])
        denominator = pd.Series([5.0, 5.0, 0.0])
        result = ind.safe_log_ratio(numerator, denominator)
        assert result.iloc[0] == pytest.approx(np.log(2.0))
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])

    @pytest.mark.parametrize("window", [0, -5])
    def test_invalid_window_is_rejected(self, window: int) -> None:
        with pytest.raises(ValueError):
            ind.sma(pd.Series([1.0, 2.0]), window)


class TestPipeline:
    def test_all_configured_groups_produce_columns(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        result = build_features(ohlcv, app_config.features)
        assert set(result.column_groups) == set(app_config.features.groups)
        for group, columns in result.column_groups.items():
            assert columns, f"group '{group}' produced no features"

    def test_groups_are_independently_ablatable(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        """MODEL_SPEC.md section 5 ablation order must actually be possible."""
        cumulative: list[str] = []
        previous_columns = 0
        for group in ALL_GROUPS:
            cumulative.append(group)
            result = build_features(
                ohlcv, app_config.features, groups=tuple(cumulative)
            )
            assert result.features.shape[1] > previous_columns
            previous_columns = result.features.shape[1]

    def test_a_single_group_does_not_pull_in_others(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        result = build_features(ohlcv, app_config.features, groups=("price_return",))
        assert set(result.column_groups) == {"price_return"}
        assert not any(c.startswith("rsi") for c in result.features.columns)

    def test_no_duplicate_feature_names(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        result = build_features(ohlcv, app_config.features)
        assert len(set(result.features.columns)) == result.features.shape[1]

    def test_index_is_preserved_exactly(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        result = build_features(ohlcv, app_config.features)
        assert result.features.index.equals(ohlcv.index)

    def test_no_nan_after_the_warmup_period(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        """Every feature must be continuously computable once warmed up."""
        result = build_features(ohlcv, app_config.features)
        usable = result.usable_features()
        assert not usable.empty
        tail = result.features.loc[usable.index.min() :]
        offenders = tail.columns[tail.isna().any()].tolist()
        assert not offenders, f"features with gaps after warmup: {offenders}"

    def test_all_features_are_finite(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        usable = build_features(ohlcv, app_config.features).usable_features()
        assert np.isfinite(usable.to_numpy(dtype="float64")).all()

    def test_result_is_deterministic(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        first = build_features(ohlcv, app_config.features).features
        second = build_features(ohlcv, app_config.features).features
        pd.testing.assert_frame_equal(first, second)

    def test_calendar_gap_is_rejected(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        """Row-based rolling windows are wrong when a day is missing."""
        with_gap = ohlcv.drop(ohlcv.index[300:305])
        with pytest.raises(FeaturePipelineError, match="missing daily rows"):
            build_features(with_gap, app_config.features)

    def test_unsorted_input_is_rejected(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        with pytest.raises(FeaturePipelineError, match="sorted by date"):
            build_features(ohlcv.iloc[::-1], app_config.features)

    def test_empty_input_is_rejected(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        with pytest.raises(FeaturePipelineError, match="empty frame"):
            build_features(ohlcv.iloc[0:0], app_config.features)

    def test_missing_column_is_rejected(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        with pytest.raises(FeaturePipelineError, match="missing columns"):
            build_features(ohlcv.drop(columns=["volume"]), app_config.features)

    def test_unknown_group_is_rejected(self, app_config: AppConfig) -> None:
        from dataclasses import replace

        broken = replace(app_config.features, groups=("price_return", "astrology"))
        with pytest.raises(FeaturePipelineError, match="unknown feature groups"):
            resolve_groups(broken)


class TestRegimeLabels:
    def test_labels_cover_the_required_axes(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        labels = compute_regime_labels(ohlcv, app_config.features.regime)
        assert set(labels["direction"].dropna().unique()) <= set(DIRECTION_LABELS)
        assert set(labels["volatility"].dropna().unique()) <= set(ALL_VOLATILITY_LABELS)

    def test_every_day_past_warmup_is_labelled(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        labels = compute_regime_labels(ohlcv, app_config.features.regime)
        labelled = labels["direction"].dropna()
        assert not labelled.empty
        tail = labels["direction"].loc[labelled.index.min() :]
        assert tail.notna().all(), "direction label has holes after warmup"

    def test_direction_labels_are_mutually_exclusive(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        result = build_features(ohlcv, app_config.features, groups=("regime",))
        one_hot = result.features[[f"regime_is_{name}" for name in DIRECTION_LABELS]]
        rows = one_hot.dropna()
        assert (rows.sum(axis=1) == 1.0).all()

    def test_deep_drawdown_with_a_rebound_is_recovery(self) -> None:
        """A crash followed by a strong bounce is 'recovery', not 'bull'."""
        days = 500
        index = pd.date_range("2020-01-01", periods=days, freq="D")
        close = np.concatenate(
            [
                np.linspace(100.0, 200.0, 200),  # run up, sets the 1y high
                np.linspace(200.0, 80.0, 100),  # crash
                np.linspace(80.0, 140.0, 200),  # rebound, still below the high
            ]
        )
        frame = pd.DataFrame({"close": close}, index=index)
        labels = compute_regime_labels(frame, {})
        assert labels["direction"].iloc[-1] == "recovery"

    def test_summary_counts_sum_to_the_row_count(
        self, ohlcv: pd.DataFrame, app_config: AppConfig
    ) -> None:
        labels = compute_regime_labels(ohlcv, app_config.features.regime)
        summary = regime_summary(labels)
        for axis in ("direction", "volatility"):
            assert summary.loc[summary["axis"] == axis, "days"].sum() == len(labels)


class TestFeaturePersistence:
    def test_feature_round_trip(
        self,
        connection: sqlite3.Connection,
        ohlcv: pd.DataFrame,
        app_config: AppConfig,
    ) -> None:
        result = build_features(ohlcv, app_config.features)
        with transaction(connection):
            written = repo.upsert_features(
                connection,
                result.features,
                feature_version=result.feature_version,
                source="binance",
                symbol="BTCUSDT",
                timeframe="1d",
            )
        assert written == result.features.size

        loaded = repo.load_features(
            connection,
            feature_version=result.feature_version,
            source="binance",
            symbol="BTCUSDT",
        )
        assert set(loaded.columns) == set(result.features.columns)
        pd.testing.assert_frame_equal(
            loaded[result.features.columns].astype("float64"),
            result.features.astype("float64"),
            rtol=1e-9,
            check_freq=False,
        )
