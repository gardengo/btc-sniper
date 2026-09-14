"""Metric, evaluator and baseline-run tests.

The metrics are checked against hand-computed values rather than against
themselves, because a metric that is only tested for self-consistency will
happily agree with its own bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics as m
from src.evaluation.baseline_eval import (
    SCOPE_VALIDATION,
    evaluate_baselines,
    model_version,
    origin_spacing_days,
    space_origins,
)
from src.evaluation.evaluator import (
    ALL_REGIMES,
    EvaluationError,
    ForecastSample,
    evaluate,
    evaluate_sample_table,
    pivot_metrics,
    regime_masks,
)
from src.features.regime import compute_regime_labels
from src.forecast.quantiles import (
    QuantileError,
    assert_ordered,
    enforce_ordering,
    interval_levels,
    quantile_label,
    quantile_labels,
    resolve_interval,
)
from src.utils.config import AppConfig
from src.validation.splits import SplitBoundaries, independent_window_count

QUANTILES: tuple[float, ...] = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)


def _series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"))


class TestQuantileLabels:
    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            (0.025, "q02_5"),
            (0.10, "q10"),
            (0.25, "q25"),
            (0.50, "q50"),
            (0.75, "q75"),
            (0.90, "q90"),
            (0.975, "q97_5"),
        ],
    )
    def test_labels_match_the_model_spec(self, level: float, expected: str) -> None:
        assert quantile_label(level) == expected

    def test_config_quantiles_label_uniquely(self, app_config: AppConfig) -> None:
        labels = quantile_labels(app_config.forecast.quantiles)
        assert len(set(labels)) == len(labels)

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
    def test_out_of_range_levels_are_rejected(self, level: float) -> None:
        with pytest.raises(QuantileError):
            quantile_label(level)

    @pytest.mark.parametrize(
        ("interval", "expected"), [(0.50, (0.25, 0.75)), (0.80, (0.10, 0.90)), (0.95, (0.025, 0.975))]
    )
    def test_intervals_map_to_the_documented_levels(
        self, interval: float, expected: tuple[float, float]
    ) -> None:
        assert interval_levels(interval) == expected
        assert resolve_interval(interval, QUANTILES) == expected

    def test_unavailable_interval_raises_instead_of_approximating(self) -> None:
        """A band labelled 80% that covers 50% is worse than no band at all."""
        with pytest.raises(QuantileError, match="not in"):
            resolve_interval(0.80, (0.25, 0.50, 0.75))

    def test_crossing_is_counted_then_repairable(self) -> None:
        frame = pd.DataFrame({0.25: [0.1, 0.5], 0.75: [0.2, 0.3]})
        assert assert_ordered(frame, (0.25, 0.75)) == 1
        repaired = enforce_ordering(frame, (0.25, 0.75))
        assert assert_ordered(repaired, (0.25, 0.75)) == 0
        assert repaired.loc[1, 0.25] == 0.3


class TestPointMetrics:
    def test_mae_and_rmse_against_hand_values(self) -> None:
        actual = _series([1.0, 2.0, 3.0])
        predicted = _series([1.0, 4.0, 7.0])
        assert m.mae(actual, predicted) == pytest.approx((0 + 2 + 4) / 3)
        assert m.rmse(actual, predicted) == pytest.approx(np.sqrt((0 + 4 + 16) / 3))

    def test_bias_is_signed(self) -> None:
        assert m.bias(_series([1.0, 1.0]), _series([2.0, 3.0])) == pytest.approx(1.5)
        assert m.bias(_series([2.0, 3.0]), _series([1.0, 1.0])) == pytest.approx(-1.5)

    def test_smape_is_symmetric_and_bounded(self) -> None:
        assert m.smape(_series([100.0]), _series([100.0])) == pytest.approx(0.0)
        # 200 * |110-100| / (100+110)
        assert m.smape(_series([100.0]), _series([110.0])) == pytest.approx(
            200.0 * 10.0 / 210.0
        )

    def test_mase_is_one_when_the_forecast_equals_the_reference(self) -> None:
        actual = _series([1.0, -2.0, 3.0])
        reference = _series([0.0, 0.0, 0.0])
        assert m.mase(actual, reference, reference) == pytest.approx(1.0)

    def test_mase_below_one_means_better_than_the_reference(self) -> None:
        actual = _series([1.0, 1.0, 1.0])
        reference = _series([0.0, 0.0, 0.0])
        better = _series([0.5, 0.5, 0.5])
        assert m.mase(actual, better, reference) == pytest.approx(0.5)

    def test_mase_is_nan_when_the_reference_is_perfect(self) -> None:
        actual = _series([1.0, 2.0])
        assert np.isnan(m.mase(actual, _series([0.0, 0.0]), actual))

    def test_nan_pairs_are_dropped_not_imputed(self) -> None:
        actual = _series([1.0, np.nan, 3.0])
        predicted = _series([1.0, 5.0, 5.0])
        assert m.mae(actual, predicted) == pytest.approx(1.0)

    def test_empty_sample_returns_nan_rather_than_raising(self) -> None:
        empty = pd.Series(dtype="float64")
        assert np.isnan(m.mae(empty, empty))
        assert np.isnan(m.rmse(empty, empty))
        assert np.isnan(m.smape(empty, empty))


class TestDirectionMetrics:
    def test_perfect_direction(self) -> None:
        result = m.directional_accuracy(_series([1.0, -1.0]), _series([2.0, -3.0]))
        assert result["direction_accuracy"] == pytest.approx(1.0)
        assert result["direction_calls"] == 2.0

    def test_up_and_down_are_reported_separately(self) -> None:
        actual = _series([1.0, 1.0, -1.0, -1.0])
        predicted = _series([1.0, -1.0, -1.0, -1.0])
        result = m.directional_accuracy(actual, predicted)
        assert result["direction_accuracy_up"] == pytest.approx(0.5)
        assert result["direction_accuracy_down"] == pytest.approx(1.0)

    def test_a_zero_median_abstains_instead_of_scoring_zero(self) -> None:
        """`no_change` makes no directional claim; 0% would misrepresent that."""
        result = m.directional_accuracy(_series([1.0, -1.0]), _series([0.0, 0.0]))
        assert result["direction_calls"] == 0.0
        assert np.isnan(result["direction_accuracy"])


class TestProbabilisticMetrics:
    def test_pinball_at_the_median_is_half_the_absolute_error(self) -> None:
        actual = _series([1.0, -2.0, 3.0])
        predicted = _series([0.0, 0.0, 0.0])
        assert m.pinball_loss(actual, predicted, 0.50) == pytest.approx(
            0.5 * m.mae(actual, predicted)
        )

    def test_pinball_penalises_the_correct_side(self) -> None:
        """At q=0.9 under-prediction should hurt more than over-prediction."""
        under = m.pinball_loss(_series([1.0]), _series([0.0]), 0.90)
        over = m.pinball_loss(_series([0.0]), _series([1.0]), 0.90)
        assert under == pytest.approx(0.9)
        assert over == pytest.approx(0.1)

    def test_pinball_is_zero_for_a_perfect_forecast(self) -> None:
        actual = _series([1.0, 2.0])
        assert m.pinball_loss(actual, actual, 0.25) == pytest.approx(0.0)

    @pytest.mark.parametrize("level", [0.0, 1.0])
    def test_pinball_rejects_degenerate_levels(self, level: float) -> None:
        with pytest.raises(ValueError):
            m.pinball_loss(_series([1.0]), _series([1.0]), level)

    def test_coverage_counts_inclusively(self) -> None:
        actual = _series([0.0, 1.0, 2.0, 3.0])
        lower = _series([0.0, 0.0, 0.0, 0.0])
        upper = _series([2.0, 2.0, 2.0, 2.0])
        assert m.interval_coverage(actual, lower, upper) == pytest.approx(0.75)

    def test_width_and_relative_width(self) -> None:
        lower = _series([-0.1, -0.2])
        upper = _series([0.1, 0.2])
        assert m.interval_width(lower, upper) == pytest.approx(0.30)
        expected = np.mean([np.exp(0.1) - np.exp(-0.1), np.exp(0.2) - np.exp(-0.2)])
        assert m.relative_price_width(lower, upper) == pytest.approx(expected)

    def test_interval_score_equals_width_when_every_actual_is_inside(self) -> None:
        actual = _series([0.0, 0.0])
        lower = _series([-1.0, -1.0])
        upper = _series([1.0, 1.0])
        assert m.interval_score(actual, lower, upper, 0.95) == pytest.approx(2.0)

    def test_interval_score_punishes_a_miss(self) -> None:
        inside = m.interval_score(_series([0.0]), _series([-1.0]), _series([1.0]), 0.95)
        outside = m.interval_score(_series([2.0]), _series([-1.0]), _series([1.0]), 0.95)
        # width 2 + (2/0.05) * 1 = 42
        assert outside == pytest.approx(42.0)
        assert outside > inside

    def test_narrow_but_wrong_loses_to_wide_but_right(self) -> None:
        actual = _series([1.0])
        narrow = m.interval_score(actual, _series([-0.01]), _series([0.01]), 0.80)
        wide = m.interval_score(actual, _series([-2.0]), _series([2.0]), 0.80)
        assert wide < narrow


class TestDispersion:
    def test_reports_worst_alongside_mean(self) -> None:
        result = m.dispersion(pd.Series([0.1, 0.2, 0.9]))
        assert result["mean"] == pytest.approx(0.4)
        assert result["worst"] == pytest.approx(0.9)
        assert result["folds"] == 3.0

    def test_empty_input_is_nan(self) -> None:
        assert np.isnan(m.dispersion(pd.Series(dtype="float64"))["mean"])


def _sample(
    n: int = 60, horizon: int = 7, spacing: int = 1, seed: int = 7
) -> ForecastSample:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    actual = pd.Series(rng.normal(0.0, 0.1, n), index=index)
    quantiles = pd.DataFrame(
        {level: np.full(n, np.quantile(actual, level)) for level in QUANTILES},
        index=index,
    )
    return ForecastSample(
        model_version="test-model",
        horizon_days=horizon,
        origin_close=pd.Series(30_000.0, index=index),
        actual_log_return=actual,
        predicted_quantiles=quantiles,
        origin_spacing_days=spacing,
    )


class TestForecastSample:
    def test_missing_median_level_is_rejected(self) -> None:
        index = pd.date_range("2020-01-01", periods=3, freq="D")
        with pytest.raises(EvaluationError, match="median level"):
            ForecastSample(
                model_version="m",
                horizon_days=1,
                origin_close=pd.Series(1.0, index=index),
                actual_log_return=pd.Series(0.0, index=index),
                predicted_quantiles=pd.DataFrame({0.25: [0.0] * 3}, index=index),
            )

    def test_unresolved_targets_are_not_scoreable(self) -> None:
        sample = _sample(n=10)
        pending = sample.actual_log_return.copy()
        pending.iloc[-3:] = np.nan
        sample = ForecastSample(
            model_version=sample.model_version,
            horizon_days=sample.horizon_days,
            origin_close=sample.origin_close,
            actual_log_return=pending,
            predicted_quantiles=sample.predicted_quantiles,
        )
        assert len(sample.scoreable()) == 7

    def test_price_space_uses_the_origin_close(self) -> None:
        sample = _sample(n=5)
        expected = sample.origin_close * np.exp(sample.actual_log_return)
        pd.testing.assert_series_equal(sample.actual_price(), expected)


class TestEvaluate:
    def test_produces_every_required_metric_family(self) -> None:
        sample = _sample()
        result = evaluate(sample, intervals=(0.50, 0.80, 0.95))
        for key in (
            "return_mae",
            "return_rmse",
            "price_mae",
            "price_smape",
            "direction_accuracy",
            "pinball_q50",
            "pinball_mean",
            "coverage_95",
            "width_95",
            "interval_score_95",
        ):
            assert key in result, key

    def test_perfect_forecast_scores_zero_error(self) -> None:
        sample = _sample(n=30)
        perfect = pd.DataFrame(
            {level: sample.actual_log_return for level in QUANTILES},
            index=sample.origins,
        )
        result = evaluate(
            ForecastSample(
                model_version="perfect",
                horizon_days=sample.horizon_days,
                origin_close=sample.origin_close,
                actual_log_return=sample.actual_log_return,
                predicted_quantiles=perfect,
            )
        )
        assert result["return_mae"] == pytest.approx(0.0)
        assert result["pinball_mean"] == pytest.approx(0.0)
        assert result["coverage_95"] == pytest.approx(1.0)

    def test_mase_against_a_reference_is_reported(self) -> None:
        sample = _sample(n=40)
        zero = pd.Series(0.0, index=sample.origins)
        result = evaluate(sample, reference_median=zero)
        assert "mase" in result and "improvement_vs_no_change" in result
        assert result["improvement_vs_no_change"] == pytest.approx(1.0 - result["mase"])

    def test_independent_windows_account_for_spacing(self) -> None:
        overlapping = evaluate(_sample(n=60, horizon=30, spacing=1))
        spaced = evaluate(_sample(n=60, horizon=30, spacing=30))
        assert overlapping["independent_windows"] == pytest.approx(2.0)
        assert spaced["independent_windows"] == pytest.approx(60.0)

    def test_crossing_count_is_reported_not_raised(self) -> None:
        sample = _sample(n=10)
        crossed = sample.predicted_quantiles.copy()
        crossed[0.975] = crossed[0.025] - 1.0
        result = evaluate(
            ForecastSample(
                model_version="crossed",
                horizon_days=1,
                origin_close=sample.origin_close,
                actual_log_return=sample.actual_log_return,
                predicted_quantiles=crossed,
            )
        )
        assert result["quantile_crossings"] > 0


class TestRegimeTables:
    def test_masks_search_both_axes(self, ohlcv: pd.DataFrame, app_config: AppConfig) -> None:
        labels = compute_regime_labels(ohlcv, app_config.features.regime)
        masks = regime_masks(labels, ("bear", "high_volatility"))
        for label, index in masks.items():
            axis = "direction" if label == "bear" else "volatility"
            assert (labels.loc[index, axis] == label).all()

    def test_absent_regime_is_omitted_not_empty(self, ohlcv: pd.DataFrame) -> None:
        labels = pd.DataFrame(
            {"direction": ["bull"] * len(ohlcv)}, index=ohlcv.index
        )
        assert "bear" not in regime_masks(labels, ("bull", "bear"))

    def test_no_regime_frame_yields_no_masks(self) -> None:
        assert regime_masks(None, ("bull",)) == {}

    def test_sample_table_emits_overall_and_regime_blocks(self) -> None:
        sample = _sample(n=60)
        labels = pd.DataFrame(
            {"direction": ["bull"] * 30 + ["bear"] * 30}, index=sample.origins
        )
        sample = ForecastSample(
            model_version=sample.model_version,
            horizon_days=sample.horizon_days,
            origin_close=sample.origin_close,
            actual_log_return=sample.actual_log_return,
            predicted_quantiles=sample.predicted_quantiles,
            regimes=labels,
        )
        table = evaluate_sample_table(sample, regimes=("bull", "bear"))
        assert set(table["regime"]) == {ALL_REGIMES, "bull", "bear"}
        overall = table[
            (table["regime"] == ALL_REGIMES) & (table["metric_name"] == "sample_size")
        ]
        assert overall["metric_value"].iloc[0] == 60.0

    def test_pivot_metrics_keeps_the_requested_order(self) -> None:
        sample = _sample(n=20)
        table = evaluate_sample_table(sample)
        wide = pivot_metrics(table, ("return_mae", "pinball_mean"))
        assert list(wide.columns) == [
            "horizon_days",
            "model_version",
            "return_mae",
            "pinball_mean",
        ]


class TestOriginSpacing:
    @pytest.mark.parametrize(
        ("horizon", "cap", "expected"), [(1, 30, 1), (7, 30, 7), (30, 30, 30), (365, 30, 30)]
    )
    def test_spacing_is_the_horizon_capped(self, horizon: int, cap: int, expected: int) -> None:
        assert origin_spacing_days(horizon, cap) == expected

    def test_spacing_below_the_cap_keeps_windows_disjoint(self) -> None:
        """At spacing == horizon the effective sample size equals the origin count."""
        spacing = origin_spacing_days(7, 30)
        assert independent_window_count(100, 7, spacing) == pytest.approx(100.0)

    def test_space_origins_uses_the_calendar_not_positions(self) -> None:
        index = pd.DatetimeIndex(
            ["2020-01-01", "2020-01-02", "2020-03-01", "2020-03-02", "2020-06-01"]
        )
        spaced = space_origins(index, 30)
        assert list(spaced.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-03-01", "2020-06-01"]

    def test_spacing_of_one_is_a_no_op(self) -> None:
        index = pd.date_range("2020-01-01", periods=5, freq="D")
        pd.testing.assert_index_equal(space_origins(index, 1), index)


class TestBaselineRun:
    @pytest.fixture
    def boundaries(self) -> SplitBoundaries:
        """A split whose test block sits inside the synthetic date range."""
        return SplitBoundaries(
            split_version="test",
            outer_test_start=pd.Timestamp("2020-01-01"),
            outer_test_end=None,
            inner_validation_end=pd.Timestamp("2019-12-31"),
            embargo_days=0,
            max_outer_test_evaluations=1,
            low_power_horizon_days=180,
        )

    def _run(self, ohlcv, app_config, boundaries, horizons=(1, 30)):
        return evaluate_baselines(
            ohlcv["close"],
            app_config,
            boundaries,
            horizons=horizons,
            candidate_origins=ohlcv.index,
            regimes=compute_regime_labels(ohlcv, app_config.features.regime),
        )

    def test_produces_metrics_for_every_baseline(
        self, ohlcv: pd.DataFrame, app_config: AppConfig, boundaries: SplitBoundaries
    ) -> None:
        result = self._run(ohlcv, app_config, boundaries)
        expected = {
            model_version(name, app_config) for name in app_config.baselines.enabled
        }
        assert set(result.metrics["model_version"]) == expected

    def test_no_origin_reaches_into_the_outer_test(
        self, ohlcv: pd.DataFrame, app_config: AppConfig, boundaries: SplitBoundaries
    ) -> None:
        """The purge rule is the whole point; verify it on the emitted periods."""
        result = self._run(ohlcv, app_config, boundaries, horizons=(30,))
        ends = pd.to_datetime(result.metrics["period_end"].dropna().unique())
        assert (ends + pd.Timedelta(days=30) < boundaries.outer_test_start).all()

    def test_reference_baseline_has_mase_of_one(
        self, ohlcv: pd.DataFrame, app_config: AppConfig, boundaries: SplitBoundaries
    ) -> None:
        result = self._run(ohlcv, app_config, boundaries)
        reference = model_version("no_change", app_config)
        rows = result.metrics[
            (result.metrics["model_version"] == reference)
            & (result.metrics["metric_name"] == "mase")
            & (result.metrics["regime"] == ALL_REGIMES)
        ]
        assert not rows.empty
        assert np.allclose(rows["metric_value"].astype(float), 1.0)

    def test_origin_counts_record_the_spacing_used(
        self, ohlcv: pd.DataFrame, app_config: AppConfig, boundaries: SplitBoundaries
    ) -> None:
        result = self._run(ohlcv, app_config, boundaries, horizons=(1, 30))
        counts = result.origin_counts.set_index("horizon_days")
        assert counts.loc[1, "spacing_days"] == 1
        assert counts.loc[30, "spacing_days"] == 30
        assert counts.loc[1, "scoreable_origins"] > counts.loc[30, "scoreable_origins"]

    def test_low_power_horizons_are_flagged(
        self, ohlcv: pd.DataFrame, app_config: AppConfig, boundaries: SplitBoundaries
    ) -> None:
        result = self._run(ohlcv, app_config, boundaries, horizons=(30, 180))
        flags = result.origin_counts.set_index("horizon_days")["low_power"]
        assert not bool(flags.loc[30])
        assert bool(flags.loc[180])

    def test_outer_test_scope_is_refused(
        self, ohlcv: pd.DataFrame, app_config: AppConfig, boundaries: SplitBoundaries
    ) -> None:
        """Baselines must not be scored on the test block during development."""
        with pytest.raises(ValueError, match="inner block"):
            evaluate_baselines(
                ohlcv["close"],
                app_config,
                boundaries,
                horizons=(30,),
                candidate_origins=ohlcv.index,
                scope="outer_test",
            )

    def test_scope_constant_is_what_the_run_records(
        self, ohlcv: pd.DataFrame, app_config: AppConfig, boundaries: SplitBoundaries
    ) -> None:
        assert self._run(ohlcv, app_config, boundaries).scope == SCOPE_VALIDATION
