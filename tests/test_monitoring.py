"""Realization, production metrics and drift tests.

The realization half is about arithmetic and about never inventing an actual.
The drift half is about a subtler failure: a detector that fires every time is
worse than no detector, so the tests pin down the calibration that stops it.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import build_features
from src.monitoring.drift import (
    LEVEL_ALERT,
    LEVEL_STABLE,
    LEVEL_WARN,
    DriftError,
    DriftThresholds,
    feature_drift,
    null_psi_thresholds,
    performance_drift,
    population_stability_index,
    retraining_triggers,
)
from src.monitoring.performance_report import (
    PerformanceContext,
    pending_summary,
    render_markdown,
)
from src.monitoring.realization import (
    STATUS_FULL,
    STATUS_PARTIAL,
    STATUS_PENDING,
    RealizationError,
    drop_in_sample,
    forecast_status,
    forecast_statuses,
    horizons_with_enough_evidence,
    production_metrics,
    realize_points,
    row_status,
    summarise,
)
from src.storage import repositories as repo
from src.storage.db import transaction
from src.utils.config import AppConfig

ORIGIN = pd.Timestamp("2024-01-01")


def _close() -> pd.Series:
    """Deterministic price path: +1% a day from the origin."""
    index = pd.date_range(ORIGIN, periods=40, freq="D")
    return pd.Series(50_000.0 * np.exp(0.01 * np.arange(len(index))), index=index)


def _points(horizons: tuple[int, ...], predicted: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "forecast_id": "f1",
                "forecast_origin_date": ORIGIN.strftime("%Y-%m-%d"),
                "origin_close": 50_000.0,
                "model_version": "m1",
                "horizon_days": horizon,
                "target_date": (ORIGIN + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d"),
                "predicted_log_return": predicted,
                "predicted_price": 50_000.0 * np.exp(predicted),
                "direction_predicted": int(np.sign(predicted)),
            }
            for horizon in horizons
        ]
    )


def _quantiles(horizons: tuple[int, ...], spread: float = 0.2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "forecast_id": "f1",
                "horizon_days": horizon,
                "quantile": level,
                "predicted_log_return": offset,
                "predicted_price": 50_000.0 * np.exp(offset),
            }
            for horizon in horizons
            for level, offset in (
                (0.025, -spread),
                (0.25, -spread / 2),
                (0.50, 0.0),
                (0.75, spread / 2),
                (0.975, spread),
            )
        ]
    )


class TestStatus:
    def test_row_status_depends_on_the_target_date(self) -> None:
        end = pd.Timestamp("2024-02-01")
        assert row_status(pd.Timestamp("2024-01-15"), end) == STATUS_FULL
        assert row_status(pd.Timestamp("2024-03-01"), end) == STATUS_PENDING

    @pytest.mark.parametrize(
        ("resolved", "total", "expected"),
        [(0, 5, STATUS_PENDING), (2, 5, STATUS_PARTIAL), (5, 5, STATUS_FULL)],
    )
    def test_forecast_status(self, resolved: int, total: int, expected: str) -> None:
        assert forecast_status(resolved, total) == expected

    def test_a_forecast_with_no_horizons_has_no_status(self) -> None:
        with pytest.raises(RealizationError):
            forecast_status(0, 0)


class TestRealization:
    def test_resolved_rows_carry_the_actual_close(self) -> None:
        close = _close()
        rows = realize_points(_points((7,)), _quantiles((7,)), close)
        row = rows.iloc[0]
        assert row["evaluation_status"] == STATUS_FULL
        assert row["actual_close"] == pytest.approx(close.loc[ORIGIN + pd.Timedelta(days=7)])

    def test_actual_log_return_matches_the_definition(self) -> None:
        close = _close()
        rows = realize_points(_points((7,)), _quantiles((7,)), close)
        expected = np.log(close.iloc[7] / 50_000.0)
        assert rows.iloc[0]["actual_log_return"] == pytest.approx(expected)

    def test_unarrived_targets_stay_pending_with_no_actual(self) -> None:
        rows = realize_points(_points((365,)), _quantiles((365,)), _close())
        row = rows.iloc[0]
        assert row["evaluation_status"] == STATUS_PENDING
        assert row["actual_close"] is None
        assert row["pinball_loss"] is None

    def test_a_missing_candle_inside_the_range_stays_pending(self) -> None:
        """A gap in the data is a data problem, not a resolved forecast."""
        close = _close().drop(ORIGIN + pd.Timedelta(days=7))
        rows = realize_points(_points((7,)), _quantiles((7,)), close)
        assert rows.iloc[0]["evaluation_status"] == STATUS_PENDING

    def test_direction_is_scored_only_when_a_call_was_made(self) -> None:
        close = _close()
        flat = realize_points(_points((7,), predicted=0.0), _quantiles((7,)), close)
        called = realize_points(_points((7,), predicted=0.05), _quantiles((7,)), close)
        assert flat.iloc[0]["direction_correct"] is None
        assert called.iloc[0]["direction_correct"] == 1

    def test_interval_flags_reflect_containment(self) -> None:
        close = _close()
        # The actual 7-day log return is 0.07. With spread=0.1 the 95% band is
        # +/-0.10 and the 50% band is +/-0.05, so it lands inside one and outside
        # the other -- which is what makes this test say anything.
        rows = realize_points(_points((7,)), _quantiles((7,), spread=0.1), close)
        row = rows.iloc[0]
        assert row["in_interval_95"] == 1
        assert row["in_interval_50"] == 0

    def test_a_wide_band_contains_everything(self) -> None:
        rows = realize_points(_points((7,)), _quantiles((7,), spread=0.5), _close())
        row = rows.iloc[0]
        assert row["in_interval_50"] == 1
        assert row["in_interval_95"] == 1

    def test_pinball_loss_is_zero_for_a_perfect_forecast(self) -> None:
        close = _close()
        actual = float(np.log(close.iloc[7] / 50_000.0))
        quantiles = _quantiles((7,), spread=0.0)
        quantiles["predicted_log_return"] = actual
        rows = realize_points(_points((7,), predicted=actual), quantiles, close)
        assert rows.iloc[0]["pinball_loss"] == pytest.approx(0.0)

    def test_regime_is_taken_at_the_origin(self, ohlcv, app_config: AppConfig) -> None:
        from src.features.regime import compute_regime_labels

        close = _close()
        regimes = pd.DataFrame({"direction": ["bear"]}, index=[ORIGIN])
        rows = realize_points(_points((7,)), _quantiles((7,)), close, regimes=regimes)
        assert rows.iloc[0]["regime"] == "bear"

    def test_summary_counts_resolved_and_pending(self) -> None:
        rows = realize_points(_points((1, 7, 365)), _quantiles((1, 7, 365)), _close())
        result = summarise(rows)
        assert result.forecasts == 1
        assert result.resolved == 2
        assert result.pending == 1

    def test_forecast_level_status_is_partial_when_some_resolved(self) -> None:
        rows = realize_points(_points((1, 365)), _quantiles((1, 365)), _close())
        assert forecast_statuses(rows)["status"].iloc[0] == STATUS_PARTIAL

    def test_empty_input_yields_an_empty_frame(self) -> None:
        assert realize_points(pd.DataFrame(), pd.DataFrame(), _close()).empty


class TestInSampleGuard:
    def test_forecasts_inside_the_training_window_are_dropped(self) -> None:
        """A model that saw the answer produces a flattering, meaningless number."""
        rows = realize_points(_points((1, 7)), _quantiles((1, 7)), _close())
        origins = pd.Series({"f1": ORIGIN.strftime("%Y-%m-%d")})
        kept, dropped = drop_in_sample(rows, origins, {"f1": "2024-06-01"})
        assert dropped == 2
        assert kept.empty

    def test_out_of_sample_forecasts_survive(self) -> None:
        rows = realize_points(_points((1, 7)), _quantiles((1, 7)), _close())
        origins = pd.Series({"f1": ORIGIN.strftime("%Y-%m-%d")})
        kept, dropped = drop_in_sample(rows, origins, {"f1": "2023-06-01"})
        assert dropped == 0
        assert len(kept) == 2

    def test_missing_cutoff_keeps_the_row(self) -> None:
        rows = realize_points(_points((1,)), _quantiles((1,)), _close())
        kept, dropped = drop_in_sample(rows, pd.Series({"f1": "2024-01-01"}), {})
        assert dropped == 0 and len(kept) == 1


class TestProductionMetrics:
    def _metrics(self, app_config: AppConfig) -> pd.DataFrame:
        rows = realize_points(_points((1, 7)), _quantiles((1, 7)), _close())
        return production_metrics(rows, pd.Series({"f1": "m1"}), app_config)

    def test_only_resolved_rows_contribute(self, app_config: AppConfig) -> None:
        rows = realize_points(_points((1, 365)), _quantiles((1, 365)), _close())
        metrics = production_metrics(rows, pd.Series({"f1": "m1"}), app_config)
        assert set(metrics["horizon_days"]) == {1}

    def test_every_required_metric_is_emitted(self, app_config: AppConfig) -> None:
        metrics = self._metrics(app_config)
        names = set(metrics["metric_name"])
        for expected in (
            "price_mae",
            "return_mae",
            "pinball_mean",
            "direction_accuracy",
            "coverage_95",
            "sample_size",
        ):
            assert expected in names

    def test_nothing_resolved_yields_an_empty_frame(self, app_config: AppConfig) -> None:
        rows = realize_points(_points((365,)), _quantiles((365,)), _close())
        assert production_metrics(rows, pd.Series({"f1": "m1"}), app_config).empty

    def test_decision_readiness_uses_the_configured_minimum(
        self, app_config: AppConfig
    ) -> None:
        evidence = horizons_with_enough_evidence(self._metrics(app_config), minimum=20)
        assert not evidence["decision_ready"].any()
        assert horizons_with_enough_evidence(self._metrics(app_config), minimum=1)[
            "decision_ready"
        ].all()


class TestPSI:
    def test_identical_samples_score_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        sample = pd.Series(rng.normal(size=4000))
        assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)

    def test_a_shifted_distribution_scores_high(self) -> None:
        rng = np.random.default_rng(1)
        reference = pd.Series(rng.normal(0, 1, 4000))
        shifted = pd.Series(rng.normal(2.0, 1, 1000))
        assert population_stability_index(reference, shifted) > 0.25

    def test_a_constant_reference_cannot_drift(self) -> None:
        constant = pd.Series([1.0] * 100)
        assert population_stability_index(constant, pd.Series([1.0] * 20)) == 0.0

    def test_empty_input_is_nan(self) -> None:
        assert np.isnan(
            population_stability_index(pd.Series(dtype="float64"), pd.Series([1.0]))
        )

    def test_too_few_bins_is_rejected(self) -> None:
        with pytest.raises(DriftError):
            population_stability_index(pd.Series([1.0, 2.0]), pd.Series([1.0]), bins=1)


class TestDriftThresholds:
    def test_config_round_trip(self, app_config: AppConfig) -> None:
        thresholds = DriftThresholds.from_config(app_config.section("operation"))
        assert thresholds.calibrate
        assert thresholds.warn_quantile < thresholds.alert_quantile

    def test_inverted_quantiles_are_rejected(self) -> None:
        with pytest.raises(DriftError, match="warn_quantile"):
            DriftThresholds(0.1, 0.25, 10, 90, 0.2, 20, warn_quantile=0.99, alert_quantile=0.9).validate()

    def test_calibrated_thresholds_override_the_fixed_ones(self) -> None:
        thresholds = DriftThresholds(0.1, 0.25, 10, 90, 0.2, 20)
        assert thresholds.level_for(0.30) == LEVEL_ALERT
        assert thresholds.level_for(0.30, warn=0.5, alert=1.0) == LEVEL_STABLE
        assert thresholds.level_for(0.60, warn=0.5, alert=1.0) == LEVEL_WARN

    def test_nan_psi_is_stable(self) -> None:
        assert DriftThresholds(0.1, 0.25, 10, 90, 0.2, 20).level_for(float("nan")) == LEVEL_STABLE


class TestFeatureDrift:
    @pytest.fixture
    def features(self, ohlcv: pd.DataFrame, app_config: AppConfig) -> pd.DataFrame:
        return build_features(ohlcv, app_config.features).usable_features()

    @pytest.fixture
    def thresholds(self) -> DriftThresholds:
        return DriftThresholds(
            0.1, 0.25, 10, 90, 0.2, 20, calibration_samples=12, alert_quantile=0.95
        )

    def test_calibration_produces_a_threshold_per_feature(
        self, features: pd.DataFrame, thresholds: DriftThresholds
    ) -> None:
        null = null_psi_thresholds(features, 90, thresholds)
        assert set(null.index) == {str(c) for c in features.columns}
        assert (null["null_alert"] >= null["null_warn"]).all()

    def test_calibration_records_the_joint_alert_count(
        self, features: pd.DataFrame, thresholds: DriftThresholds
    ) -> None:
        """The number that stops 'at least one feature alerted' firing always."""
        null = null_psi_thresholds(features, 90, thresholds)
        assert null.attrs["null_alert_count"] >= 0.0

    def test_a_too_short_reference_cannot_be_calibrated(
        self, features: pd.DataFrame, thresholds: DriftThresholds
    ) -> None:
        assert null_psi_thresholds(features.head(50), 90, thresholds).empty

    def test_drift_frame_reports_excess_over_the_null(
        self, features: pd.DataFrame, thresholds: DriftThresholds
    ) -> None:
        cutoff = features.index[len(features) // 2]
        frame = feature_drift(features, thresholds, training_end=cutoff)
        assert set(frame.columns) >= {"feature", "psi", "null_alert", "excess", "level"}
        assert frame.attrs["calibrated"]

    def test_calibration_flags_far_fewer_features_than_fixed_thresholds(
        self, features: pd.DataFrame, thresholds: DriftThresholds
    ) -> None:
        """The whole point: the fixed thresholds over-report on this kind of data."""
        cutoff = features.index[len(features) // 2]
        calibrated = feature_drift(features, thresholds, training_end=cutoff)
        fixed = feature_drift(
            features,
            DriftThresholds(0.1, 0.25, 10, 90, 0.2, 20, calibrate=False),
            training_end=cutoff,
        )
        assert (calibrated["level"] == LEVEL_ALERT).sum() <= (
            fixed["level"] == LEVEL_ALERT
        ).sum()

    def test_no_reference_yields_an_empty_frame(
        self, features: pd.DataFrame, thresholds: DriftThresholds
    ) -> None:
        early = features.index.min() - pd.Timedelta(days=1)
        assert feature_drift(features, thresholds, training_end=early).empty


class TestPerformanceDrift:
    def _frames(self, production_value: float, sample: int):
        production = pd.DataFrame(
            [
                {
                    "model_version": "m1",
                    "horizon_days": 30,
                    "regime": "all",
                    "metric_name": "pinball_mean",
                    "metric_value": production_value,
                    "sample_size": sample,
                }
            ]
        )
        benchmark = pd.DataFrame(
            [
                {
                    "model_version": "wf-1",
                    "horizon_days": 30,
                    "regime": "all",
                    "metric_name": "pinball_mean",
                    "metric_value": 0.05,
                }
            ]
        )
        return production, benchmark

    def test_degradation_is_relative_to_the_benchmark(self) -> None:
        thresholds = DriftThresholds(0.1, 0.25, 10, 90, 0.2, 20)
        production, benchmark = self._frames(0.06, 50)
        frame = performance_drift(production, benchmark, thresholds)
        assert frame["degradation"].iloc[0] == pytest.approx(0.2)

    def test_a_small_sample_cannot_trigger(self) -> None:
        """OPERATING_SPEC.md 4: no decisions from a handful of forecasts."""
        thresholds = DriftThresholds(0.1, 0.25, 10, 90, 0.2, 20)
        production, benchmark = self._frames(0.50, 3)
        frame = performance_drift(production, benchmark, thresholds)
        assert not frame["decision_ready"].iloc[0]
        assert frame["level"].iloc[0] != LEVEL_ALERT

    def test_a_large_degradation_on_a_real_sample_alerts(self) -> None:
        thresholds = DriftThresholds(0.1, 0.25, 10, 90, 0.2, 20)
        production, benchmark = self._frames(0.50, 100)
        assert performance_drift(production, benchmark, thresholds)["level"].iloc[0] == LEVEL_ALERT

    def test_missing_benchmark_yields_nothing(self) -> None:
        thresholds = DriftThresholds(0.1, 0.25, 10, 90, 0.2, 20)
        production, _ = self._frames(0.06, 50)
        assert performance_drift(production, pd.DataFrame(), thresholds).empty


class TestTriggers:
    def _feature_frame(self, alerts: int, total: int, null_count: float) -> pd.DataFrame:
        frame = pd.DataFrame(
            [
                {
                    "feature": f"f{i}",
                    "psi": 1.0,
                    "level": LEVEL_ALERT if i < alerts else LEVEL_STABLE,
                }
                for i in range(total)
            ]
        )
        frame.attrs["calibrated"] = True
        frame.attrs["null_alert_count"] = null_count
        return frame

    def test_fewer_alerts_than_a_no_drift_window_does_not_fire(self) -> None:
        """This is what stops the detector firing on every single run."""
        thresholds = DriftThresholds(0.1, 0.25, 10, 90, 0.2, 20)
        triggers = retraining_triggers(
            self._feature_frame(alerts=10, total=64, null_count=15.0),
            pd.DataFrame(),
            thresholds,
            new_observations=0,
            min_new_observations=14,
        )
        assert not bool(triggers.iloc[0]["fired"])

    def test_more_alerts_than_the_null_fires(self) -> None:
        thresholds = DriftThresholds(0.1, 0.25, 10, 90, 0.2, 20)
        triggers = retraining_triggers(
            self._feature_frame(alerts=30, total=64, null_count=15.0),
            pd.DataFrame(),
            thresholds,
            new_observations=0,
            min_new_observations=14,
        )
        assert bool(triggers.iloc[0]["fired"])

    def test_new_observations_trigger_is_a_simple_count(self) -> None:
        thresholds = DriftThresholds(0.1, 0.25, 10, 90, 0.2, 20)
        triggers = retraining_triggers(
            pd.DataFrame(),
            pd.DataFrame(),
            thresholds,
            new_observations=20,
            min_new_observations=14,
        ).set_index("trigger")
        assert bool(triggers.loc["new_observations", "fired"])

    def test_all_three_triggers_are_always_reported(self) -> None:
        thresholds = DriftThresholds(0.1, 0.25, 10, 90, 0.2, 20)
        triggers = retraining_triggers(
            pd.DataFrame(), pd.DataFrame(), thresholds, new_observations=0, min_new_observations=14
        )
        assert set(triggers["trigger"]) == {
            "feature_distribution_drift",
            "performance_degradation",
            "new_observations",
        }


class TestReport:
    def _context(self, resolved: int, app_config: AppConfig) -> PerformanceContext:
        rows = realize_points(_points((1, 365)), _quantiles((1, 365)), _close())
        thresholds = DriftThresholds.from_config(app_config.section("operation"))
        return PerformanceContext(
            realization=summarise(rows),
            production=production_metrics(rows, pd.Series({"f1": "m1"}), app_config)
            if resolved
            else pd.DataFrame(),
            evidence=pd.DataFrame(),
            feature_drift=pd.DataFrame(columns=["feature", "psi", "level"]),
            performance_drift=pd.DataFrame(),
            triggers=pd.DataFrame(),
            thresholds=thresholds,
            model_version="m1",
            training_cutoff="2023-12-30",
            data_end="2024-02-09",
        )

    def test_report_renders_every_section(self, app_config: AppConfig) -> None:
        text = render_markdown(self._context(1, app_config))
        for heading in (
            "## 1. How much evidence exists",
            "## 2. Realized performance",
            "## 3. Decision readiness",
            "## 4. Performance drift",
            "## 5. Feature drift",
            "## 6. Retraining triggers",
        ):
            assert heading in text

    def test_report_says_plainly_when_nothing_has_resolved(
        self, app_config: AppConfig
    ) -> None:
        rows = realize_points(_points((365,)), _quantiles((365,)), _close())
        thresholds = DriftThresholds.from_config(app_config.section("operation"))
        context = PerformanceContext(
            realization=summarise(rows),
            production=pd.DataFrame(),
            evidence=pd.DataFrame(),
            feature_drift=pd.DataFrame(columns=["feature", "psi", "level"]),
            performance_drift=pd.DataFrame(),
            triggers=pd.DataFrame(),
            thresholds=thresholds,
            model_version="m1",
            training_cutoff="2023-12-30",
        )
        assert "No forecast has resolved yet" in render_markdown(context)

    def test_pending_summary_splits_by_horizon(self) -> None:
        rows = realize_points(_points((1, 365)), _quantiles((1, 365)), _close())
        summary = pending_summary(rows).set_index("horizon_days")
        assert summary.loc[1, "resolved"] == 1
        assert summary.loc[365, "pending"] == 1


class TestRealizationPersistence:
    def _store_forecast(self, connection: sqlite3.Connection) -> None:
        from src.forecast.generate import Forecast

        forecast = Forecast(
            forecast_id="f1",
            origin_date=ORIGIN,
            origin_close=50_000.0,
            model_version="m1",
            feature_version="v1",
            horizon_grid_version="v1",
            config_version="v1",
            code_commit="abc",
            created_at="2026-09-14T00:00:00Z",
            points=_points((1, 7)).drop(columns=["forecast_id", "forecast_origin_date", "origin_close", "model_version"]),
            quantiles=pd.DataFrame(
                [
                    {
                        "horizon_days": horizon,
                        "quantile": 0.50,
                        "quantile_label": "q50",
                        "predicted_log_return": 0.0,
                        "predicted_price": 50_000.0,
                        "crossing_adjusted": 0,
                    }
                    for horizon in (1, 7)
                ]
            ),
        )
        with transaction(connection):
            repo.upsert_forecast(
                connection, forecast, source="binance", symbol="BTCUSDT", timeframe="1d"
            )

    def test_realizations_round_trip(self, connection: sqlite3.Connection) -> None:
        self._store_forecast(connection)
        rows = realize_points(_points((1, 7)), _quantiles((1, 7)), _close())
        with transaction(connection):
            repo.upsert_realizations(connection, rows)
        stored = repo.load_realizations(connection)
        assert len(stored) == 2
        assert set(stored["evaluation_status"]) == {STATUS_FULL}

    def test_rerunning_updates_rather_than_duplicating(
        self, connection: sqlite3.Connection
    ) -> None:
        """OPERATING_SPEC.md 7: update the realization, never add a prediction."""
        self._store_forecast(connection)
        rows = realize_points(_points((1, 7)), _quantiles((1, 7)), _close())
        for _ in range(3):
            with transaction(connection):
                repo.upsert_realizations(connection, rows)
        assert (
            connection.execute("SELECT COUNT(*) FROM forecast_realizations").fetchone()[0]
            == 2
        )

    def test_a_pending_row_becomes_evaluated_in_place(
        self, connection: sqlite3.Connection
    ) -> None:
        self._store_forecast(connection)
        short = _close().head(3)
        with transaction(connection):
            repo.upsert_realizations(
                connection, realize_points(_points((1, 7)), _quantiles((1, 7)), short)
            )
        assert len(repo.load_realizations(connection, status=STATUS_PENDING)) == 1

        with transaction(connection):
            repo.upsert_realizations(
                connection, realize_points(_points((1, 7)), _quantiles((1, 7)), _close())
            )
        assert repo.load_realizations(connection, status=STATUS_PENDING).empty
        assert len(repo.load_realizations(connection, status=STATUS_FULL)) == 2

    def test_filtering_by_horizon(self, connection: sqlite3.Connection) -> None:
        self._store_forecast(connection)
        with transaction(connection):
            repo.upsert_realizations(
                connection, realize_points(_points((1, 7)), _quantiles((1, 7)), _close())
            )
        assert len(repo.load_realizations(connection, horizon_days=7)) == 1
