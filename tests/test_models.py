"""Model layer tests: datasets, folds, fitting, serialisation, registry.

Two properties matter more than the rest and are tested hardest:

* a fit never sees a label that reaches past its own boundary
* a refit on identical data produces identical predictions
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import build_features
from src.models import registry
from src.models.base import (
    STATUS_CANDIDATE,
    STATUS_PRODUCTION,
    STATUS_REJECTED,
    STATUS_RETIRED,
    ModelError,
    ModelMetadata,
    new_model_id,
    resolve_algorithm,
)
from src.models.dataset import (
    DatasetError,
    align_feature_columns,
    build_inference_matrix,
    build_training_matrix,
)
from src.models.forecaster import MultiHorizonForecaster, fit_horizon_model
from src.models.lightgbm_model import LightGBMQuantileRegressor, resolve_params
from src.models.training import (
    TrainingRequest,
    assert_reproducible,
    build_matrices,
    model_version_name,
    training_origins,
)
from src.storage.db import transaction
from src.utils.config import AppConfig
from src.validation.folds import (
    EXPANDING,
    FoldError,
    FoldPlan,
    assert_fold_is_clean,
    build_folds,
    describe_folds,
    iter_folds,
    window_start,
)
from src.validation.splits import SplitBoundaries, SplitError

LEVELS: tuple[float, ...] = (0.10, 0.50, 0.90)
FAST_PARAMS: dict[str, object] = {
    "n_estimators": 20,
    "num_leaves": 7,
    "learning_rate": 0.1,
    "min_child_samples": 20,
}


@pytest.fixture
def boundaries() -> SplitBoundaries:
    """A split whose outer test starts inside the synthetic date range."""
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
def features(ohlcv: pd.DataFrame, app_config: AppConfig) -> pd.DataFrame:
    return build_features(ohlcv, app_config.features).usable_features()


@pytest.fixture
def close(ohlcv: pd.DataFrame) -> pd.Series:
    return ohlcv["close"]


class TestDataset:
    def test_features_and_target_are_aligned(
        self, features: pd.DataFrame, close: pd.Series
    ) -> None:
        matrix = build_training_matrix(features, close, 7, features.index)
        assert matrix.features.index.equals(matrix.target.index)
        assert matrix.rows > 0

    def test_unresolved_labels_are_dropped(
        self, features: pd.DataFrame, close: pd.Series
    ) -> None:
        """The last h origins have no future price yet and must not be invented."""
        matrix = build_training_matrix(features, close, 30, features.index)
        assert matrix.origins.max() <= close.index.max() - pd.Timedelta(days=30)

    def test_label_values_match_the_definition(
        self, features: pd.DataFrame, close: pd.Series
    ) -> None:
        matrix = build_training_matrix(features, close, 14, features.index)
        origin = matrix.origins[5]
        expected = np.log(close.loc[origin + pd.Timedelta(days=14)] / close.loc[origin])
        assert matrix.target.loc[origin] == pytest.approx(expected)

    def test_purge_is_reasserted_on_build(
        self, features: pd.DataFrame, close: pd.Series, boundaries: SplitBoundaries
    ) -> None:
        contaminated = features.index[features.index < pd.Timestamp("2020-01-01")]
        with pytest.raises(SplitError, match="outer test"):
            build_training_matrix(
                features, close, 30, contaminated, boundaries=boundaries
            )

    def test_inference_matrix_drops_incomplete_rows(
        self, features: pd.DataFrame
    ) -> None:
        frame = features.copy()
        frame.iloc[0, 0] = np.nan
        result = build_inference_matrix(frame, frame.index)
        assert len(result) == len(frame) - 1

    def test_column_alignment_restores_the_fitted_order(
        self, features: pd.DataFrame
    ) -> None:
        expected = tuple(str(c) for c in features.columns)
        shuffled = features[list(reversed(features.columns))]
        assert tuple(align_feature_columns(shuffled, expected).columns) == expected

    def test_missing_column_is_an_error_not_a_silent_reorder(
        self, features: pd.DataFrame
    ) -> None:
        expected = tuple(str(c) for c in features.columns)
        with pytest.raises(DatasetError, match="missing columns"):
            align_feature_columns(features.iloc[:, 1:], expected)

    def test_mismatched_index_is_rejected(self, features: pd.DataFrame) -> None:
        from src.models.dataset import TrainingMatrix

        with pytest.raises(DatasetError, match="not aligned"):
            TrainingMatrix(
                horizon_days=1,
                features=features,
                target=pd.Series(0.0, index=features.index[:-1]),
            )


class TestFolds:
    def test_expanding_window_starts_at_the_data_start(self) -> None:
        start = pd.Timestamp("2018-01-01")
        assert window_start(EXPANDING, pd.Timestamp("2022-06-01"), start) == start

    def test_rolling_window_keeps_only_n_years(self) -> None:
        result = window_start(
            "rolling_4y", pd.Timestamp("2022-06-01"), pd.Timestamp("2010-01-01")
        )
        assert result == pd.Timestamp("2018-06-01")

    def test_rolling_window_cannot_start_before_the_data(self) -> None:
        start = pd.Timestamp("2021-01-01")
        assert window_start("rolling_8y", pd.Timestamp("2022-06-01"), start) == start

    def test_unknown_strategy_names_the_alternatives(self) -> None:
        with pytest.raises(FoldError, match="rolling_4y"):
            window_start("rolling_99y", pd.Timestamp("2022-01-01"), pd.Timestamp("2020-01-01"))

    def test_folds_are_ordered_and_inside_the_inner_block(
        self, features: pd.DataFrame, boundaries: SplitBoundaries
    ) -> None:
        plan = FoldPlan(n_folds=3, validation_days=180, strategy=EXPANDING, min_train_origins=50)
        folds = build_folds(features.index, boundaries, plan, 7)
        assert folds
        for previous, following in zip(folds, folds[1:]):
            assert previous.validation_origins.max() < following.validation_origins.min()
        for fold in folds:
            assert fold.validation_origins.max() <= boundaries.inner_validation_end

    def test_training_stops_a_full_horizon_before_validation(
        self, features: pd.DataFrame, boundaries: SplitBoundaries
    ) -> None:
        """Without the gap the last training labels would read validation prices."""
        horizon = 30
        plan = FoldPlan(n_folds=2, validation_days=180, strategy=EXPANDING, min_train_origins=50)
        for fold in build_folds(features.index, boundaries, plan, horizon):
            gap = fold.validation_origins.min() - fold.train_origins.max()
            assert gap > pd.Timedelta(days=horizon)

    def test_contaminated_fold_is_rejected(
        self, features: pd.DataFrame, boundaries: SplitBoundaries
    ) -> None:
        from src.validation.folds import Fold

        validation_start = pd.Timestamp("2019-06-01")
        bad = Fold(
            index=1,
            horizon_days=30,
            strategy=EXPANDING,
            train_origins=features.index[features.index < validation_start],
            validation_origins=features.index[features.index >= validation_start][:30],
        )
        with pytest.raises(SplitError, match="validation block"):
            assert_fold_is_clean(bad, boundaries)

    def test_iter_folds_checks_every_fold(
        self, features: pd.DataFrame, boundaries: SplitBoundaries
    ) -> None:
        plan = FoldPlan(n_folds=3, validation_days=180, strategy=EXPANDING, min_train_origins=50)
        assert list(iter_folds(features.index, boundaries, plan, 7))

    def test_undersized_folds_are_dropped(
        self, features: pd.DataFrame, boundaries: SplitBoundaries
    ) -> None:
        generous = FoldPlan(n_folds=5, validation_days=120, strategy=EXPANDING, min_train_origins=10)
        strict = FoldPlan(n_folds=5, validation_days=120, strategy=EXPANDING, min_train_origins=400)
        assert len(build_folds(features.index, boundaries, generous, 1)) >= len(
            build_folds(features.index, boundaries, strict, 1)
        )

    def test_rolling_folds_train_on_less_than_expanding(
        self, features: pd.DataFrame, boundaries: SplitBoundaries
    ) -> None:
        base = dict(n_folds=1, validation_days=180, min_train_origins=10)
        expanding = build_folds(
            features.index, boundaries, FoldPlan(strategy=EXPANDING, **base), 7
        )
        rolling = build_folds(
            features.index, boundaries, FoldPlan(strategy="rolling_4y", **base), 7
        )
        assert expanding and rolling
        assert len(rolling[0].train_origins) <= len(expanding[0].train_origins)

    def test_describe_folds_reports_every_fold(
        self, features: pd.DataFrame, boundaries: SplitBoundaries
    ) -> None:
        plan = FoldPlan(n_folds=3, validation_days=180, strategy=EXPANDING, min_train_origins=50)
        folds = build_folds(features.index, boundaries, plan, 7)
        table = describe_folds(folds)
        assert len(table) == len(folds)
        assert set(table["fold"]) == {fold.name for fold in folds}

    @pytest.mark.parametrize("field", ["n_folds", "validation_days", "min_train_origins"])
    def test_invalid_plan_is_rejected(self, field: str) -> None:
        values = {"n_folds": 1, "validation_days": 1, "strategy": EXPANDING, "min_train_origins": 1}
        values[field] = 0
        with pytest.raises(FoldError):
            FoldPlan(**values).validate()


class TestLightGBMRegressor:
    def _fit(self, features: pd.DataFrame, close: pd.Series, level: float = 0.50):
        matrix = build_training_matrix(features, close, 7, features.index)
        model = LightGBMQuantileRegressor(level, FAST_PARAMS, seed=42).fit(
            matrix.features, matrix.target
        )
        return model, matrix

    def test_forced_params_cannot_be_overridden(self) -> None:
        params, _ = resolve_params({"objective": "regression", "num_threads": 8}, level=0.5, seed=1)
        assert params["objective"] == "quantile"
        assert params["num_threads"] == 1

    def test_rounds_are_taken_from_the_param_dict(self) -> None:
        params, rounds = resolve_params({"n_estimators": 42}, level=0.5, seed=1)
        assert rounds == 42
        assert "n_estimators" not in params

    def test_every_seed_slot_is_set(self) -> None:
        params, _ = resolve_params({}, level=0.5, seed=7)
        for key in ("seed", "bagging_seed", "feature_fraction_seed", "data_random_seed"):
            assert params[key] == 7

    @pytest.mark.parametrize("level", [0.0, 1.0, 1.5])
    def test_degenerate_level_is_rejected(self, level: float) -> None:
        with pytest.raises(ModelError):
            resolve_params({}, level=level, seed=1)

    def test_refit_is_bitwise_identical(
        self, features: pd.DataFrame, close: pd.Series
    ) -> None:
        first, matrix = self._fit(features, close)
        second = LightGBMQuantileRegressor(0.50, FAST_PARAMS, seed=42).fit(
            matrix.features, matrix.target
        )
        pd.testing.assert_series_equal(
            first.predict(matrix.features), second.predict(matrix.features)
        )

    def test_the_seed_matters_only_when_sampling_is_on(
        self, features: pd.DataFrame, close: pd.Series
    ) -> None:
        """Without bagging, GBDT is deterministic and the seed changes nothing.

        Worth pinning down: it means a seed-only "reproducibility" check would
        pass vacuously under the default params and prove nothing. The real
        config enables subsampling, which is where the seed starts to matter.
        """
        matrix = build_training_matrix(features, close, 7, features.index)
        sampled = {**FAST_PARAMS, "subsample": 0.7, "subsample_freq": 1}

        def fit(params, seed):
            return (
                LightGBMQuantileRegressor(0.50, params, seed=seed)
                .fit(matrix.features, matrix.target)
                .predict(matrix.features)
            )

        assert fit(FAST_PARAMS, 42).equals(fit(FAST_PARAMS, 99))
        assert not fit(sampled, 42).equals(fit(sampled, 99))
        assert fit(sampled, 42).equals(fit(sampled, 42))

    def test_higher_quantiles_predict_higher_values_on_average(
        self, features: pd.DataFrame, close: pd.Series
    ) -> None:
        matrix = build_training_matrix(features, close, 7, features.index)
        predictions = {
            level: LightGBMQuantileRegressor(level, FAST_PARAMS, seed=42)
            .fit(matrix.features, matrix.target)
            .predict(matrix.features)
            .mean()
            for level in (0.10, 0.50, 0.90)
        }
        assert predictions[0.10] < predictions[0.50] < predictions[0.90]

    def test_text_round_trip_preserves_predictions(
        self, features: pd.DataFrame, close: pd.Series
    ) -> None:
        model, matrix = self._fit(features, close)
        restored = LightGBMQuantileRegressor.from_text(model.to_text(), 0.50, FAST_PARAMS)
        pd.testing.assert_series_equal(
            model.predict(matrix.features), restored.predict(matrix.features)
        )

    def test_unfitted_model_refuses_to_predict_or_serialise(self) -> None:
        model = LightGBMQuantileRegressor(0.50, FAST_PARAMS)
        with pytest.raises(ModelError, match="not fitted"):
            model.predict(pd.DataFrame({"a": [1.0]}))
        with pytest.raises(ModelError, match="unfitted"):
            model.to_text()

    def test_nan_target_is_rejected(self, features: pd.DataFrame) -> None:
        target = pd.Series(np.nan, index=features.index)
        with pytest.raises(ModelError, match="NaN"):
            LightGBMQuantileRegressor(0.50, FAST_PARAMS).fit(features, target)

    def test_reordered_inference_columns_are_refused(
        self, features: pd.DataFrame, close: pd.Series
    ) -> None:
        """Trees index features positionally; a silent reorder predicts nonsense."""
        model, matrix = self._fit(features, close)
        shuffled = matrix.features[list(reversed(matrix.features.columns))]
        with pytest.raises(ModelError, match="column order"):
            model.predict(shuffled)

    def test_feature_importance_covers_the_fitted_columns(
        self, features: pd.DataFrame, close: pd.Series
    ) -> None:
        model, matrix = self._fit(features, close)
        importance = model.feature_importance()
        assert set(importance.index) == set(matrix.feature_names)
        assert importance.is_monotonic_decreasing

    def test_algorithm_resolves_by_name(self) -> None:
        assert resolve_algorithm("lightgbm_quantile") is LightGBMQuantileRegressor

    def test_unknown_algorithm_lists_what_exists(self) -> None:
        with pytest.raises(ModelError, match="lightgbm_quantile"):
            resolve_algorithm("magic_forest")


class TestForecaster:
    @pytest.fixture
    def forecaster(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        boundaries: SplitBoundaries,
        app_config: AppConfig,
    ) -> MultiHorizonForecaster:
        from src.models.training import train_forecaster

        request = TrainingRequest(
            horizons=(1, 7),
            levels=LEVELS,
            strategy=EXPANDING,
            algorithm="lightgbm_quantile",
            params=FAST_PARAMS,
            seed=42,
        )
        return train_forecaster(features, close, boundaries, app_config, request)

    def test_holds_every_requested_horizon(self, forecaster) -> None:
        assert forecaster.horizons == (1, 7)
        assert forecaster.levels == LEVELS

    def test_metadata_records_the_spec_section_10_fields(self, forecaster) -> None:
        metadata = forecaster.metadata
        for value in (
            metadata.model_id,
            metadata.model_version,
            metadata.algorithm,
            metadata.feature_version,
            metadata.horizon_grid_version,
            metadata.training_cutoff,
            metadata.created_at,
        ):
            assert value
        assert metadata.status == STATUS_CANDIDATE
        assert metadata.library_versions["lightgbm"]
        assert metadata.training_rows_by_horizon[1] > 0

    def test_predictions_are_indexed_by_origin(self, forecaster, features) -> None:
        rows = features.tail(5)
        predicted, _ = forecaster.predict_horizon(rows, 1)
        assert predicted.index.equals(rows.index)
        assert list(predicted.columns) == list(LEVELS)

    def test_unknown_horizon_is_refused(self, forecaster, features) -> None:
        with pytest.raises(ModelError, match="no horizon"):
            forecaster.predict_horizon(features.tail(2), 999)

    def test_crossing_is_repaired_and_counted(self, forecaster, features) -> None:
        rows = features.tail(20)
        repaired, crossings = forecaster.predict_horizon(rows, 1, repair_crossing=True)
        assert crossings >= 0
        for low, high in zip(LEVELS, LEVELS[1:]):
            assert (repaired[high] >= repaired[low] - 1e-12).all()

    def test_save_and_load_round_trip(self, forecaster, features, tmp_path) -> None:
        path = forecaster.save(tmp_path)
        assert path.is_file()
        restored = MultiHorizonForecaster.load(path)
        assert restored.metadata.model_version == forecaster.metadata.model_version
        assert restored.horizons == forecaster.horizons
        rows = features.tail(10)
        for horizon in forecaster.horizons:
            pd.testing.assert_frame_equal(
                forecaster.predict_horizon(rows, horizon)[0],
                restored.predict_horizon(rows, horizon)[0],
            )

    def test_load_rejects_an_unknown_artifact_format(self, forecaster, tmp_path) -> None:
        import gzip
        import json

        path = tmp_path / "broken.model.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump({"format_version": 99}, handle)
        with pytest.raises(ModelError, match="format"):
            MultiHorizonForecaster.load(path)

    def test_empty_matrix_cannot_be_fitted(self, features: pd.DataFrame) -> None:
        from src.models.dataset import TrainingMatrix

        empty = TrainingMatrix(
            horizon_days=1,
            features=features.iloc[:0],
            target=pd.Series(dtype="float64", index=features.index[:0]),
        )
        with pytest.raises(ModelError, match="no training rows"):
            fit_horizon_model(
                empty, algorithm="lightgbm_quantile", levels=LEVELS, params=FAST_PARAMS, seed=1
            )


class TestTrainingSelection:
    def test_version_name_encodes_what_changes_the_model(self) -> None:
        name = model_version_name(
            "lightgbm_quantile", "rolling_4y", pd.Timestamp("2023-12-30"), "v1"
        )
        assert name == "lgbmq-rolling_4y-20231230-v1"

    def test_origins_respect_both_window_and_purge(
        self, features: pd.DataFrame, boundaries: SplitBoundaries
    ) -> None:
        horizon = 30
        expanding = training_origins(
            features.index, boundaries, horizon, strategy=EXPANDING
        )
        rolling = training_origins(
            features.index, boundaries, horizon, strategy="rolling_4y"
        )
        assert len(rolling) <= len(expanding)
        assert (expanding + pd.Timedelta(days=horizon) < boundaries.outer_test_start).all()

    def test_cutoff_narrows_but_never_widens(
        self, features: pd.DataFrame, boundaries: SplitBoundaries
    ) -> None:
        wide = training_origins(features.index, boundaries, 7, strategy=EXPANDING)
        cutoff = pd.Timestamp("2019-06-01")
        narrow = training_origins(
            features.index, boundaries, 7, strategy=EXPANDING, cutoff=cutoff
        )
        assert 0 < len(narrow) < len(wide)
        assert narrow.max() + pd.Timedelta(days=7) <= cutoff

    def test_every_horizon_gets_its_own_purged_matrix(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        boundaries: SplitBoundaries,
    ) -> None:
        request = TrainingRequest(
            horizons=(1, 90),
            levels=LEVELS,
            strategy=EXPANDING,
            algorithm="lightgbm_quantile",
            params=FAST_PARAMS,
            seed=42,
        )
        matrices = build_matrices(features, close, boundaries, request)
        assert matrices[1].rows > matrices[90].rows
        assert matrices[90].origins.max() < matrices[1].origins.max()

    def test_reproducibility_check_passes(
        self,
        features: pd.DataFrame,
        close: pd.Series,
        boundaries: SplitBoundaries,
        app_config: AppConfig,
    ) -> None:
        request = TrainingRequest(
            horizons=(7,),
            levels=(0.50,),
            strategy=EXPANDING,
            algorithm="lightgbm_quantile",
            params=FAST_PARAMS,
            seed=42,
        )
        assert assert_reproducible(
            features, close, boundaries, app_config, request, horizon_days=7
        )

    def test_request_from_config_uses_the_configured_defaults(
        self, app_config: AppConfig
    ) -> None:
        request = TrainingRequest.from_config(app_config, horizons=(1, 7))
        assert request.algorithm == app_config.section("models")["primary_algorithm"]
        assert request.levels == app_config.forecast.quantiles
        assert request.seed == app_config.section("models")["random_seed"]


def _metadata(version: str = "test-v1", **changes) -> ModelMetadata:
    base = dict(
        model_id=new_model_id(),
        model_version=version,
        algorithm="lightgbm_quantile",
        feature_version="v1",
        horizon_grid_version="v1",
        training_cutoff="2023-12-30",
        training_rows=1000,
        horizons=(1, 7),
        quantiles=(0.5,),
        created_at="2026-09-14T00:00:00Z",
    )
    base.update(changes)
    return ModelMetadata(**base)


class TestRegistry:
    def test_training_can_only_register_candidates(
        self, connection: sqlite3.Connection
    ) -> None:
        """CLAUDE.md 2.3: a model must not become production as a side effect."""
        with pytest.raises(registry.RegistryError, match="only accepts candidates"):
            with transaction(connection):
                registry.register(connection, _metadata(status=STATUS_PRODUCTION))

    def test_registered_model_is_readable(self, connection: sqlite3.Connection) -> None:
        metadata = _metadata()
        with transaction(connection):
            registry.register(connection, metadata)
        row = registry.get(connection, metadata.model_version)
        assert row["status"] == STATUS_CANDIDATE
        rebuilt = registry.metadata_from_row(row)
        assert rebuilt.model_version == metadata.model_version
        assert rebuilt.horizons == metadata.horizons

    def test_duplicate_version_is_refused(self, connection: sqlite3.Connection) -> None:
        with transaction(connection):
            registry.register(connection, _metadata("dup"))
        with pytest.raises(sqlite3.IntegrityError):
            with transaction(connection):
                registry.register(connection, _metadata("dup"))

    def test_promotion_retires_the_incumbent(
        self, connection: sqlite3.Connection
    ) -> None:
        with transaction(connection):
            registry.register(connection, _metadata("old"))
            registry.register(connection, _metadata("new"))
            registry.promote(connection, "old")
        assert registry.production_model(connection)["model_version"] == "old"

        with transaction(connection):
            registry.promote(connection, "new")
        assert registry.production_model(connection)["model_version"] == "new"
        assert registry.get(connection, "old")["status"] == STATUS_RETIRED

    def test_only_one_production_model_can_exist(
        self, connection: sqlite3.Connection
    ) -> None:
        with transaction(connection):
            registry.register(connection, _metadata("a"))
            registry.register(connection, _metadata("b"))
            registry.promote(connection, "a")
        # Force the inconsistent state the guard exists to catch.
        with transaction(connection):
            connection.execute(
                "UPDATE model_registry SET status = ? WHERE model_version = ?",
                (STATUS_PRODUCTION, "b"),
            )
        with pytest.raises(registry.RegistryError, match="marked production"):
            registry.production_model(connection)

    def test_a_retired_model_cannot_be_repromoted(
        self, connection: sqlite3.Connection
    ) -> None:
        with transaction(connection):
            registry.register(connection, _metadata("x"))
            registry.register(connection, _metadata("y"))
            registry.promote(connection, "x")
            registry.promote(connection, "y")
        with pytest.raises(registry.RegistryError, match="only 'candidate'"):
            with transaction(connection):
                registry.promote(connection, "x")

    def test_rejection_requires_a_reason(self, connection: sqlite3.Connection) -> None:
        with transaction(connection):
            registry.register(connection, _metadata("r"))
        with pytest.raises(registry.RegistryError, match="reason"):
            registry.reject(connection, "r", "   ")
        with transaction(connection):
            registry.reject(connection, "r", "worse interval coverage at 30d")
        assert registry.get(connection, "r")["status"] == STATUS_REJECTED

    def test_outer_test_metrics_can_only_be_written_once(
        self, connection: sqlite3.Connection
    ) -> None:
        """VALIDATION_SPEC.md 4.4: one outer-test evaluation per model version."""
        with transaction(connection):
            registry.register(connection, _metadata("once"))
            registry.record_metrics(connection, "once", test_metrics={"mae": 0.1})
        with pytest.raises(registry.RegistryError, match="already has outer-test"):
            with transaction(connection):
                registry.record_metrics(connection, "once", test_metrics={"mae": 0.05})

    def test_validation_metrics_may_be_rewritten(
        self, connection: sqlite3.Connection
    ) -> None:
        with transaction(connection):
            registry.register(connection, _metadata("v"))
            registry.record_metrics(connection, "v", validation_metrics={"mae": 0.2})
            registry.record_metrics(connection, "v", validation_metrics={"mae": 0.15})
        import json

        stored = json.loads(registry.get(connection, "v")["validation_metrics"])
        assert stored["mae"] == 0.15

    def test_no_production_model_is_a_normal_state(
        self, connection: sqlite3.Connection
    ) -> None:
        assert registry.production_model(connection) is None
        assert registry.load_production(connection) is None

    def test_missing_artifact_is_reported_clearly(
        self, connection: sqlite3.Connection, tmp_path
    ) -> None:
        with transaction(connection):
            registry.register(
                connection, _metadata("gone"), artifact_path=tmp_path / "absent.json.gz"
            )
        with pytest.raises(registry.RegistryError, match="missing"):
            registry.load_artifact(connection, "gone")

    def test_run_lifecycle_is_recorded(self, connection: sqlite3.Connection) -> None:
        metadata = _metadata("run")
        with transaction(connection):
            registry.start_run(connection, "run-1", "training", config_version="v1")
        row = connection.execute(
            "SELECT status, model_id FROM model_runs WHERE run_id = 'run-1'"
        ).fetchone()
        assert row["status"] == "running" and row["model_id"] is None

        with transaction(connection):
            registry.register(connection, metadata)
            registry.finish_run(
                connection, "run-1", status="succeeded", model_id=metadata.model_id
            )
        row = connection.execute(
            "SELECT status, model_id, finished_at FROM model_runs WHERE run_id = 'run-1'"
        ).fetchone()
        assert row["status"] == "succeeded"
        assert row["model_id"] == metadata.model_id
        assert row["finished_at"]

    def test_artifact_version_mismatch_is_caught(
        self, connection: sqlite3.Connection, tmp_path, features, close, boundaries, app_config
    ) -> None:
        """A renamed artifact must not load as a different model version."""
        from src.models.training import train_forecaster

        request = TrainingRequest(
            horizons=(1,),
            levels=(0.5,),
            strategy=EXPANDING,
            algorithm="lightgbm_quantile",
            params=FAST_PARAMS,
            seed=42,
        )
        forecaster = train_forecaster(features, close, boundaries, app_config, request)
        path = forecaster.save(tmp_path)
        with transaction(connection):
            registry.register(connection, _metadata("mislabelled"), artifact_path=path)
        with pytest.raises(registry.RegistryError, match="holds"):
            registry.load_artifact(connection, "mislabelled")
