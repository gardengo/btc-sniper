"""Forecast generation, blending, interpolation and persistence tests."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import build_features
from src.forecast.blending import (
    MODE_LINEAR,
    MODE_STEP,
    BlendError,
    BlendPolicy,
    blend_quantiles,
    blend_source_label,
    to_prices,
)
from src.forecast.generate import (
    ForecastError,
    assert_intervals_contain_median,
    blend_summary,
    generate_forecast,
)
from src.forecast.interpolate import (
    InterpolationError,
    daily_grid,
    enforce_band_order,
    interpolate_curve,
    interpolate_quantiles,
    interpolated_forecast_curve,
)
from src.models.training import TrainingRequest, train_forecaster
from src.storage import repositories as repo
from src.storage.db import transaction
from src.utils.config import AppConfig
from src.validation.folds import EXPANDING
from src.validation.splits import SplitBoundaries

QUANTILES: tuple[float, ...] = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)
TINY_PARAMS: dict[str, object] = {
    "n_estimators": 12,
    "num_leaves": 3,
    "learning_rate": 0.1,
    "min_child_samples": 40,
}


def _policy(full: int = 1, none: int = 30, mode: str = MODE_LINEAR) -> BlendPolicy:
    return BlendPolicy(
        full_model_horizon_days=full, baseline_only_horizon_days=none, mode=mode
    )


class TestBlendPolicy:
    def test_weight_is_one_up_to_the_full_model_horizon(self) -> None:
        assert _policy().weight(1) == pytest.approx(1.0)

    def test_weight_is_zero_from_the_baseline_only_horizon(self) -> None:
        policy = _policy()
        assert policy.weight(30) == pytest.approx(0.0)
        assert policy.weight(365) == pytest.approx(0.0)

    def test_linear_ramp_is_monotone_and_bounded(self) -> None:
        policy = _policy()
        weights = [policy.weight(h) for h in range(1, 31)]
        assert all(0.0 <= w <= 1.0 for w in weights)
        assert all(later <= earlier + 1e-12 for earlier, later in zip(weights, weights[1:]))

    def test_step_mode_switches_without_a_ramp(self) -> None:
        policy = _policy(mode=MODE_STEP)
        assert policy.weight(1) == 1.0
        assert policy.weight(2) == 0.0

    def test_disabled_policy_always_uses_the_model(self) -> None:
        policy = BlendPolicy(1, 30, enabled=False)
        assert policy.weight(365) == 1.0

    def test_overlapping_boundaries_are_rejected(self) -> None:
        with pytest.raises(BlendError, match="greater than"):
            BlendPolicy(30, 30).validate()

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(BlendError, match="mode"):
            BlendPolicy(1, 30, mode="magic").validate()

    def test_config_policy_matches_the_documented_decision(
        self, app_config: AppConfig
    ) -> None:
        """MODEL_SPEC.md 6.5: model only at h=1, baseline only from h=30."""
        policy = BlendPolicy.from_config(app_config)
        assert policy.weight(1) == pytest.approx(1.0)
        assert policy.weight(30) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("weight", "expected"), [(1.0, "model"), (0.0, "baseline"), (0.4, "blend")]
    )
    def test_source_label(self, weight: float, expected: str) -> None:
        assert blend_source_label(weight) == expected


class TestBlendQuantiles:
    def _frames(self):
        index = pd.date_range("2024-01-01", periods=3, freq="D")
        model = pd.DataFrame({0.25: [-0.1] * 3, 0.50: [0.0] * 3, 0.75: [0.1] * 3}, index=index)
        baseline = pd.DataFrame({0.25: [-0.2] * 3, 0.50: [0.1] * 3, 0.75: [0.4] * 3}, index=index)
        return model, baseline

    def test_weight_one_returns_the_model(self) -> None:
        model, baseline = self._frames()
        pd.testing.assert_frame_equal(blend_quantiles(model, baseline, 1.0), model)

    def test_weight_zero_returns_the_baseline(self) -> None:
        model, baseline = self._frames()
        pd.testing.assert_frame_equal(blend_quantiles(model, baseline, 0.0), baseline)

    def test_half_weight_is_the_midpoint(self) -> None:
        model, baseline = self._frames()
        blended = blend_quantiles(model, baseline, 0.5)
        assert blended.loc[blended.index[0], 0.50] == pytest.approx(0.05)

    def test_blending_cannot_create_a_crossing(self) -> None:
        """A convex combination of ordered quantiles stays ordered."""
        model, baseline = self._frames()
        for weight in np.linspace(0.0, 1.0, 11):
            blended = blend_quantiles(model, baseline, float(weight))
            assert (blended[0.75] >= blended[0.50]).all()
            assert (blended[0.50] >= blended[0.25]).all()

    def test_weight_outside_the_unit_interval_is_rejected(self) -> None:
        model, baseline = self._frames()
        with pytest.raises(BlendError, match="in \\[0, 1\\]"):
            blend_quantiles(model, baseline, 1.5)

    def test_missing_baseline_origin_is_an_error_not_a_silent_drop(self) -> None:
        model, baseline = self._frames()
        with pytest.raises(BlendError, match="no baseline"):
            blend_quantiles(model, baseline.iloc[:0], 0.5)

    def test_price_conversion_is_the_exponential(self) -> None:
        model, _ = self._frames()
        prices = to_prices(model, 50_000.0)
        assert prices.loc[prices.index[0], 0.50] == pytest.approx(50_000.0)
        assert prices.loc[prices.index[0], 0.75] == pytest.approx(50_000.0 * np.exp(0.1))

    def test_non_positive_anchor_is_rejected(self) -> None:
        model, _ = self._frames()
        with pytest.raises(BlendError, match="positive"):
            to_prices(model, 0.0)


class TestInterpolation:
    def _knots(self) -> pd.DataFrame:
        return pd.DataFrame(
            {0.25: [-0.05, -0.12, -0.3], 0.50: [0.0, 0.02, 0.05], 0.75: [0.05, 0.16, 0.4]},
            index=pd.Index([1, 30, 365], name="horizon_days"),
        )

    def test_grid_covers_every_day(self) -> None:
        assert daily_grid(365).tolist() == list(range(0, 366))
        assert daily_grid(5, include_origin=False).tolist() == [1, 2, 3, 4, 5]

    def test_curve_starts_at_the_anchor(self) -> None:
        curves = interpolate_quantiles(self._knots())
        assert np.allclose(curves.loc[0].to_numpy(), 0.0)

    def test_knots_are_reproduced_exactly(self) -> None:
        knots = self._knots()
        curves = interpolate_quantiles(knots)
        for horizon in knots.index:
            for level in knots.columns:
                assert curves.loc[horizon, level] == pytest.approx(
                    knots.loc[horizon, level], abs=1e-12
                )

    def test_pchip_does_not_overshoot_the_knots(self) -> None:
        """A natural cubic spline would; that is the whole reason for PCHIP."""
        knots = self._knots()
        curves = interpolate_quantiles(knots)
        for level in knots.columns:
            segment = curves[level].iloc[1:]
            assert segment.min() >= min(knots[level].min(), 0.0) - 1e-9
            assert segment.max() <= max(knots[level].max(), 0.0) + 1e-9

    def test_monotone_knots_produce_a_monotone_curve(self) -> None:
        knots = pd.DataFrame({0.50: [0.0, 0.1, 0.3]}, index=[1, 30, 365])
        curve = interpolate_quantiles(knots)[0.50]
        assert (curve.diff().dropna() >= -1e-12).all()

    def test_bands_never_cross_on_the_drawn_curve(self) -> None:
        curves = interpolate_quantiles(self._knots())
        ordered = sorted(curves.columns)
        for low, high in zip(ordered, ordered[1:]):
            assert (curves[high] >= curves[low] - 1e-12).all()

    def test_band_order_is_repaired_when_inputs_cross(self) -> None:
        frame = pd.DataFrame({0.25: [0.5], 0.75: [0.1]})
        repaired = enforce_band_order(frame)
        assert repaired.loc[0, 0.25] == 0.1
        assert repaired.loc[0, 0.75] == 0.5

    def test_a_single_knot_cannot_be_interpolated(self) -> None:
        with pytest.raises(InterpolationError, match="two knots"):
            interpolate_curve(np.array([1.0]), np.array([0.0]), np.array([1.0]))

    def test_unsorted_knots_are_rejected(self) -> None:
        with pytest.raises(InterpolationError, match="increasing"):
            interpolate_curve(
                np.array([30.0, 1.0]), np.array([0.1, 0.0]), np.array([5.0])
            )

    def test_non_finite_knots_are_rejected(self) -> None:
        with pytest.raises(InterpolationError, match="non-finite"):
            interpolate_curve(
                np.array([1.0, 30.0]), np.array([0.0, np.nan]), np.array([5.0])
            )

    def test_horizon_zero_is_rejected_as_a_knot(self) -> None:
        knots = pd.DataFrame({0.50: [0.0, 0.1]}, index=[0, 30])
        with pytest.raises(InterpolationError, match="1 day or later"):
            interpolate_quantiles(knots)

    def test_price_curve_carries_calendar_dates(self) -> None:
        origin = pd.Timestamp("2026-01-01")
        prices = interpolated_forecast_curve(self._knots(), 50_000.0, origin)
        assert prices.loc[0, "date"] == origin
        assert prices.loc[365, "date"] == origin + pd.Timedelta(days=365)
        assert prices.loc[0, 0.50] == pytest.approx(50_000.0)
        assert (prices[0.50] > 0).all()


class TestGenerate:
    @pytest.fixture
    def boundaries(self) -> SplitBoundaries:
        return SplitBoundaries(
            split_version="test",
            outer_test_start=pd.Timestamp("2020-01-01"),
            outer_test_end=None,
            inner_validation_end=pd.Timestamp("2019-12-31"),
            embargo_days=0,
            max_outer_test_evaluations=1,
            low_power_horizon_days=180,
        )

    @pytest.fixture
    def parts(self, ohlcv: pd.DataFrame, app_config: AppConfig, boundaries):
        features = build_features(ohlcv, app_config.features).usable_features()
        request = TrainingRequest(
            horizons=(1, 7, 30),
            levels=QUANTILES,
            strategy=EXPANDING,
            algorithm="lightgbm_quantile",
            params=TINY_PARAMS,
            seed=42,
        )
        forecaster = train_forecaster(
            features, ohlcv["close"], boundaries, app_config, request
        )
        return features, ohlcv["close"], forecaster

    def test_forecast_is_anchored_on_the_last_closed_candle(
        self, parts, app_config: AppConfig
    ) -> None:
        features, close, forecaster = parts
        forecast = generate_forecast(
            features, close, app_config, forecaster, (1, 7, 30), policy=_policy()
        )
        assert forecast.origin_date == features.dropna(how="any").index.max()
        assert forecast.origin_close == pytest.approx(close.loc[forecast.origin_date])

    def test_median_price_reconstructs_from_the_log_return(
        self, parts, app_config: AppConfig
    ) -> None:
        features, close, forecaster = parts
        forecast = generate_forecast(
            features, close, app_config, forecaster, (1, 7, 30), policy=_policy()
        )
        for row in forecast.points.itertuples(index=False):
            expected = forecast.origin_close * np.exp(row.predicted_log_return)
            assert row.predicted_price == pytest.approx(expected)

    def test_target_dates_are_the_origin_plus_the_horizon(
        self, parts, app_config: AppConfig
    ) -> None:
        features, close, forecaster = parts
        forecast = generate_forecast(
            features, close, app_config, forecaster, (1, 7, 30), policy=_policy()
        )
        for row in forecast.points.itertuples(index=False):
            expected = forecast.origin_date + pd.Timedelta(days=row.horizon_days)
            assert pd.Timestamp(row.target_date) == expected

    def test_every_horizon_records_its_source(
        self, parts, app_config: AppConfig
    ) -> None:
        features, close, forecaster = parts
        forecast = generate_forecast(
            features, close, app_config, forecaster, (1, 7, 30), policy=_policy()
        )
        sources = forecast.points.set_index("horizon_days")["source"]
        assert sources.loc[1] == "model"
        assert sources.loc[7] == "blend"
        assert sources.loc[30] == "baseline"
        assert blend_summary(forecast)["horizons"].sum() == 3

    def test_baseline_only_horizon_needs_no_model(
        self, parts, app_config: AppConfig
    ) -> None:
        """A horizon the model was never trained on is fine at zero weight."""
        features, close, forecaster = parts
        forecast = generate_forecast(
            features, close, app_config, forecaster, (1, 90), policy=_policy()
        )
        assert set(forecast.points["horizon_days"]) == {1, 90}

    def test_a_weighted_horizon_the_model_lacks_is_an_error(
        self, parts, app_config: AppConfig
    ) -> None:
        """Silently substituting the baseline would change what the number means."""
        features, close, forecaster = parts
        with pytest.raises(ForecastError, match="was not trained on it"):
            generate_forecast(
                features,
                close,
                app_config,
                forecaster,
                (90,),
                policy=_policy(full=1, none=365),
            )

    def test_quantiles_are_ordered_and_surround_the_median(
        self, parts, app_config: AppConfig
    ) -> None:
        features, close, forecaster = parts
        forecast = generate_forecast(
            features, close, app_config, forecaster, (1, 7, 30), policy=_policy()
        )
        assert_intervals_contain_median(forecast)

    def test_unknown_origin_is_reported_clearly(
        self, parts, app_config: AppConfig
    ) -> None:
        features, close, forecaster = parts
        with pytest.raises(ForecastError, match="no complete feature row"):
            generate_forecast(
                features,
                close,
                app_config,
                forecaster,
                (1,),
                origin=pd.Timestamp("1999-01-01"),
                policy=_policy(),
            )

    def test_current_price_is_carried_but_never_becomes_the_anchor(
        self, parts, app_config: AppConfig
    ) -> None:
        """DATA_SPEC.md section 2: the two numbers are expected to differ."""
        features, close, forecaster = parts
        live = float(close.iloc[-1]) * 1.05
        forecast = generate_forecast(
            features,
            close,
            app_config,
            forecaster,
            (1,),
            current_price=live,
            policy=_policy(),
        )
        assert forecast.current_price == pytest.approx(live)
        assert forecast.origin_close != pytest.approx(live)

    def test_quantile_matrix_is_interpolatable(
        self, parts, app_config: AppConfig
    ) -> None:
        features, close, forecaster = parts
        forecast = generate_forecast(
            features, close, app_config, forecaster, (1, 7, 30), policy=_policy()
        )
        curves = interpolate_quantiles(forecast.quantile_matrix())
        assert len(curves) == 31
        assert np.allclose(curves.loc[0].to_numpy(), 0.0)


class TestForecastPersistence:
    @pytest.fixture
    def stored(self, connection: sqlite3.Connection, ohlcv, app_config, candles):
        from src.forecast.generate import Forecast

        points = pd.DataFrame(
            [
                {
                    "horizon_days": 1,
                    "target_date": "2024-01-02",
                    "predicted_log_return": 0.01,
                    "predicted_price": 50_502.5,
                    "direction_predicted": 1,
                    "model_weight": 1.0,
                    "source": "model",
                }
            ]
        )
        quantiles = pd.DataFrame(
            [
                {
                    "horizon_days": 1,
                    "quantile": level,
                    "quantile_label": f"q{int(level * 1000)}",
                    "predicted_log_return": value,
                    "predicted_price": 50_000.0 * np.exp(value),
                    "crossing_adjusted": 0,
                }
                for level, value in [(0.025, -0.05), (0.50, 0.01), (0.975, 0.06)]
            ]
        )
        return Forecast(
            forecast_id="abc123",
            origin_date=pd.Timestamp("2024-01-01"),
            origin_close=50_000.0,
            model_version="test-model-v1",
            feature_version="v1",
            horizon_grid_version="v1",
            config_version="v1",
            code_commit="deadbee",
            created_at="2026-09-14T00:00:00Z",
            points=points,
            quantiles=quantiles,
            current_price=50_100.0,
        )

    def test_round_trip(self, connection: sqlite3.Connection, stored) -> None:
        with transaction(connection):
            repo.upsert_forecast(
                connection, stored, source="binance", symbol="BTCUSDT", timeframe="1d"
            )
        row = repo.latest_forecast_row(connection, source="binance", symbol="BTCUSDT")
        assert row["forecast_id"] == "abc123"
        assert row["origin_close"] == pytest.approx(50_000.0)
        assert row["current_price"] == pytest.approx(50_100.0)

        points = repo.load_forecast_points(connection, "abc123")
        assert len(points) == 1
        assert points["predicted_price"].iloc[0] == pytest.approx(50_502.5)

        wide = repo.load_forecast_quantiles(connection, "abc123")
        assert list(wide.columns) == [0.025, 0.50, 0.975]

    def test_rerunning_the_same_origin_replaces_rather_than_duplicates(
        self, connection: sqlite3.Connection, stored
    ) -> None:
        for _ in range(2):
            with transaction(connection):
                repo.upsert_forecast(
                    connection,
                    stored,
                    source="binance",
                    symbol="BTCUSDT",
                    timeframe="1d",
                )
        assert (
            connection.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 1
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM forecast_points").fetchone()[0] == 1
        )

    def test_replacing_a_forecast_leaves_no_orphaned_children(
        self, connection: sqlite3.Connection, stored
    ) -> None:
        """A shorter horizon grid must not leave rows from the longer one."""
        from dataclasses import replace

        with transaction(connection):
            repo.upsert_forecast(
                connection, stored, source="binance", symbol="BTCUSDT", timeframe="1d"
            )
        shorter = replace(
            stored,
            forecast_id="def456",
            quantiles=stored.quantiles.iloc[:1].copy(),
        )
        with transaction(connection):
            repo.upsert_forecast(
                connection, shorter, source="binance", symbol="BTCUSDT", timeframe="1d"
            )
        assert (
            connection.execute("SELECT COUNT(*) FROM forecast_quantiles").fetchone()[0]
            == 1
        )

    def test_history_query_returns_the_horizon(
        self, connection: sqlite3.Connection, stored
    ) -> None:
        with transaction(connection):
            repo.upsert_forecast(
                connection, stored, source="binance", symbol="BTCUSDT", timeframe="1d"
            )
        history = repo.forecast_history(
            connection, source="binance", symbol="BTCUSDT", horizon_days=1
        )
        assert len(history) == 1
        assert history["forecast_origin_date"].iloc[0] == "2024-01-01"
