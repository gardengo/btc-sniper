"""Target construction and naive baseline tests.

The causality of the baselines is the point of this module: they are the
reference every later model is measured against, so a leak here would silently
raise the bar and make a genuinely worse model look competitive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import (
    BASELINE_CLASSES,
    REFERENCE_BASELINE,
    BaselineError,
    DriftBaseline,
    NoChangeBaseline,
    RollingReturnBaseline,
    build_baselines,
    common_valid_origins,
    empirical_spread,
    observation_counts,
)
from src.models.targets import (
    TargetError,
    assert_disjoint,
    build_targets,
    forward_log_return,
    observable_origins,
    target_column,
    target_dates,
)
from src.utils.config import AppConfig

QUANTILES: tuple[float, ...] = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)


@pytest.fixture
def close(ohlcv: pd.DataFrame) -> pd.Series:
    return ohlcv["close"]


class TestTargets:
    def test_forward_return_matches_the_definition(self, close: pd.Series) -> None:
        target = forward_log_return(close, 30)
        expected = np.log(close.iloc[30] / close.iloc[0])
        assert target.iloc[0] == pytest.approx(expected)

    def test_last_h_rows_are_nan_and_not_filled(self, close: pd.Series) -> None:
        target = forward_log_return(close, 14)
        assert target.iloc[-14:].isna().all()
        assert target.iloc[:-14].notna().all()

    def test_price_reconstruction_round_trips(self, close: pd.Series) -> None:
        """`P(t,h) = Close[t] * exp(Y(t,h))` must land on the actual future close."""
        horizon = 45
        target = forward_log_return(close, horizon)
        reconstructed = close * np.exp(target)
        actual_future = close.shift(-horizon)
        pd.testing.assert_series_equal(
            reconstructed.dropna(),
            actual_future.dropna(),
            check_names=False,
            rtol=1e-12,
        )

    def test_build_targets_has_one_column_per_horizon(self, close: pd.Series) -> None:
        frame = build_targets(close, (1, 7, 30))
        assert list(frame.columns) == [target_column(h) for h in (1, 7, 30)]
        assert len(frame) == len(close)

    def test_duplicate_horizons_collapse(self, close: pd.Series) -> None:
        assert build_targets(close, (7, 7, 1)).shape[1] == 2

    @pytest.mark.parametrize("horizon", [0, -5])
    def test_rejects_non_positive_horizon(self, close: pd.Series, horizon: int) -> None:
        with pytest.raises(TargetError):
            forward_log_return(close, horizon)

    def test_rejects_unsorted_index(self, close: pd.Series) -> None:
        with pytest.raises(TargetError, match="sorted"):
            forward_log_return(close.iloc[::-1], 1)

    def test_rejects_non_positive_prices(self, close: pd.Series) -> None:
        broken = close.copy()
        broken.iloc[10] = 0.0
        with pytest.raises(TargetError, match="non-positive"):
            forward_log_return(broken, 1)

    def test_target_dates_are_calendar_offsets(self, close: pd.Series) -> None:
        dates = target_dates(close.index[:3], 90)
        assert (dates - close.index[:3] == pd.Timedelta(days=90)).all()

    def test_observable_origins_stop_one_horizon_before_the_end(
        self, close: pd.Series
    ) -> None:
        origins = observable_origins(close, 30)
        assert origins.max() == close.index.max() - pd.Timedelta(days=30)

    def test_assert_disjoint_catches_a_target_in_the_feature_matrix(
        self, close: pd.Series
    ) -> None:
        targets = build_targets(close, (7,))
        features = pd.DataFrame({"sma_7": close}, index=close.index)
        assert_disjoint(features, targets)  # clean

        leaked = features.join(targets)
        with pytest.raises(TargetError, match="target columns present"):
            assert_disjoint(leaked, targets)

    def test_assert_disjoint_catches_a_target_shaped_column(
        self, close: pd.Series
    ) -> None:
        targets = build_targets(close, (7,))
        sneaky = pd.DataFrame({target_column(999): close}, index=close.index)
        with pytest.raises(TargetError, match="target-shaped"):
            assert_disjoint(sneaky, targets)


class TestBaselineConstruction:
    def test_config_lists_are_instantiated(self, app_config: AppConfig) -> None:
        built = build_baselines(app_config.baselines)
        assert [b.name for b in built] == list(app_config.baselines.enabled)

    def test_unknown_baseline_is_rejected(self, app_config: AppConfig) -> None:
        broken = type(app_config.baselines)(
            enabled=("no_change", "crystal_ball"),
            drift_min_samples=365,
            rolling_return_window_days=90,
            quantile_lookback_days=None,
            quantile_min_samples=60,
        )
        with pytest.raises(BaselineError, match="crystal_ball"):
            build_baselines(broken)

    def test_reference_baseline_is_registered(self) -> None:
        assert REFERENCE_BASELINE in BASELINE_CLASSES

    def test_rejects_empty_series(self, app_config: AppConfig) -> None:
        baseline = NoChangeBaseline(app_config.baselines)
        with pytest.raises(BaselineError, match="empty"):
            baseline.predict(pd.Series(dtype="float64"), 7, QUANTILES)


class TestEmpiricalSpread:
    def test_median_column_is_exactly_zero(
        self, close: pd.Series, app_config: AppConfig
    ) -> None:
        """The spread carries no location, so the baseline owns the median entirely."""
        spread = empirical_spread(np.log(close), 30, QUANTILES, app_config.baselines)
        defined = spread[0.50].dropna()
        assert not defined.empty
        assert np.allclose(defined.to_numpy(), 0.0, atol=1e-12)

    def test_quantiles_are_monotone_across_levels(
        self, close: pd.Series, app_config: AppConfig
    ) -> None:
        spread = empirical_spread(np.log(close), 30, QUANTILES, app_config.baselines).dropna()
        for low, high in zip(QUANTILES, QUANTILES[1:]):
            assert (spread[high] >= spread[low] - 1e-12).all()

    def test_warmup_needs_horizon_plus_min_samples(
        self, close: pd.Series, app_config: AppConfig
    ) -> None:
        horizon, minimum = 30, app_config.baselines.quantile_min_samples
        spread = empirical_spread(np.log(close), horizon, QUANTILES, app_config.baselines)
        first_defined = spread[0.50].first_valid_index()
        assert first_defined == close.index[horizon + minimum - 1]

    def test_observation_counts_track_the_sample(
        self, close: pd.Series, app_config: AppConfig
    ) -> None:
        counts = observation_counts(np.log(close), 7, app_config.baselines)
        assert counts.dropna().is_monotonic_increasing  # expanding window
        assert counts.max() == len(close) - 7


class TestBaselineBehaviour:
    def test_no_change_median_is_exactly_zero(
        self, close: pd.Series, app_config: AppConfig
    ) -> None:
        prediction = NoChangeBaseline(app_config.baselines).predict(close, 30, QUANTILES)
        median = prediction.median.dropna()
        assert not median.empty
        assert np.allclose(median.to_numpy(), 0.0, atol=1e-12)

    def test_drift_extrapolates_the_mean_daily_return(
        self, close: pd.Series, app_config: AppConfig
    ) -> None:
        horizon = 30
        baseline = DriftBaseline(app_config.baselines)
        location = baseline.location(np.log(close), horizon)
        at = close.index[-1]
        daily = np.log(close).diff(1)
        expected = daily.loc[:at].mean() * horizon
        assert location.loc[at] == pytest.approx(expected)

    def test_rolling_return_uses_only_its_window(
        self, close: pd.Series, app_config: AppConfig
    ) -> None:
        window = app_config.baselines.rolling_return_window_days
        location = RollingReturnBaseline(app_config.baselines).location(np.log(close), 10)
        at = close.index[-1]
        daily = np.log(close).diff(1)
        expected = daily.iloc[-window:].mean() * 10
        assert location.loc[at] == pytest.approx(expected)

    def test_quantiles_never_cross(self, close: pd.Series, app_config: AppConfig) -> None:
        for baseline in build_baselines(app_config.baselines):
            frame = baseline.predict(close, 90, QUANTILES).quantiles.dropna()
            for low, high in zip(QUANTILES, QUANTILES[1:]):
                assert (frame[high] >= frame[low] - 1e-12).all(), baseline.name

    def test_prices_are_the_exponential_of_the_log_return(
        self, close: pd.Series, app_config: AppConfig
    ) -> None:
        prediction = NoChangeBaseline(app_config.baselines).predict(close, 7, QUANTILES)
        prices = prediction.prices(close).dropna()
        # A zero median log return means the median price is today's price.
        pd.testing.assert_series_equal(
            prices[0.50], close.reindex(prices.index), check_names=False
        )

    def test_restrict_narrows_without_recomputing(
        self, close: pd.Series, app_config: AppConfig
    ) -> None:
        prediction = DriftBaseline(app_config.baselines).predict(close, 7, QUANTILES)
        keep = prediction.valid_origins()[:20]
        narrowed = prediction.restrict(keep)
        assert list(narrowed.quantiles.index) == list(keep)
        pd.testing.assert_series_equal(
            narrowed.quantiles[0.50], prediction.quantiles[0.50].reindex(keep)
        )

    def test_common_valid_origins_is_the_intersection(
        self, close: pd.Series, app_config: AppConfig
    ) -> None:
        predictions = [b.predict(close, 30, QUANTILES) for b in build_baselines(app_config.baselines)]
        shared = common_valid_origins(predictions, close.index)
        for prediction in predictions:
            assert shared.isin(prediction.valid_origins()).all()
        # `drift` needs the most history, so it sets the start.
        slowest = min(
            (p.valid_origins().min() for p in predictions if len(p.valid_origins())),
        )
        assert shared.min() >= slowest


class TestBaselineCausality:
    """Truncation equivalence: a forecast must not change when the future is cut off."""

    @pytest.mark.parametrize("horizon", [1, 30, 90])
    def test_predictions_are_identical_on_truncated_history(
        self, close: pd.Series, app_config: AppConfig, horizon: int
    ) -> None:
        cutoff = close.index[600]
        for baseline in build_baselines(app_config.baselines):
            full = baseline.predict(close, horizon, QUANTILES).quantiles.loc[:cutoff]
            truncated = baseline.predict(
                close.loc[:cutoff], horizon, QUANTILES
            ).quantiles.loc[:cutoff]
            pd.testing.assert_frame_equal(
                full, truncated, check_freq=False, obj=baseline.name
            )

    def test_a_future_price_shock_cannot_move_an_earlier_forecast(
        self, close: pd.Series, app_config: AppConfig
    ) -> None:
        cutoff = close.index[500]
        shocked = close.copy()
        shocked.iloc[501:] *= 3.0

        for baseline in build_baselines(app_config.baselines):
            before = baseline.predict(close, 30, QUANTILES).quantiles.loc[:cutoff]
            after = baseline.predict(shocked, 30, QUANTILES).quantiles.loc[:cutoff]
            pd.testing.assert_frame_equal(before, after, obj=baseline.name)
