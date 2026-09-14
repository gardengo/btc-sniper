"""Composing single-quantile regressors into a multi-horizon forecaster.

MODEL_SPEC.md section 6 asks for one model per horizon "for reliability and
debuggability", and section 3 asks for seven quantiles. That is one regressor
per (horizon, quantile) pair: independent, individually inspectable, and
individually replaceable.

Quantile crossing
-----------------
Independently fitted quantiles can cross -- the 75th can land below the 25th for
a particular row. MODEL_SPEC.md section 3 permits repairing this as
post-processing provided the frequency is measured, so `predict` sorts each row
and reports the crossing rate. A rising crossing rate means the quantile models
disagree about the same input, which is a real signal that the fits are too
noisy; hiding the repair would hide that.

Serialisation
-------------
The whole bundle is one gzipped JSON file: metadata plus the booster text for
every (horizon, quantile). One file per model version keeps an artifact
atomic -- a half-written directory of 539 boosters is a model that loads and
silently predicts from the wrong mixture of versions.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.forecast.quantiles import (
    count_crossing_rows,
    count_crossings,
    enforce_ordering,
)
from src.models.base import (
    ModelError,
    ModelMetadata,
    QuantileRegressor,
    resolve_algorithm,
)
from src.models.dataset import TrainingMatrix, align_feature_columns
from src.utils.logging import get_logger
from src.utils.timeutils import utc_now_iso

logger = get_logger(__name__)

ARTIFACT_SUFFIX: str = ".model.json.gz"
ARTIFACT_FORMAT_VERSION: int = 1


@dataclass
class HorizonQuantileModel:
    """Every quantile of one horizon."""

    horizon_days: int
    algorithm: str
    levels: tuple[float, ...]
    regressors: dict[float, QuantileRegressor]

    @property
    def feature_names(self) -> tuple[str, ...]:
        for regressor in self.regressors.values():
            names = getattr(regressor, "feature_names", ())
            if names:
                return tuple(names)
        return ()

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Quantile predictions for ``features``, columns ordered by level."""
        aligned = align_feature_columns(features, self.feature_names)
        return pd.DataFrame(
            {level: self.regressors[level].predict(aligned) for level in self.levels},
            index=aligned.index,
        )

    def feature_importance(self) -> pd.Series:
        """Mean gain across the quantile models, so one level cannot dominate."""
        frames = [
            regressor.feature_importance()
            for regressor in self.regressors.values()
            if not regressor.feature_importance().empty
        ]
        if not frames:
            return pd.Series(dtype="float64")
        return pd.concat(frames, axis=1).mean(axis=1).sort_values(ascending=False)


def fit_horizon_model(
    matrix: TrainingMatrix,
    *,
    algorithm: str,
    levels: tuple[float, ...],
    params: Mapping[str, Any],
    seed: int,
) -> HorizonQuantileModel:
    """Fit one regressor per quantile on the same training matrix."""
    if matrix.rows == 0:
        raise ModelError(f"h={matrix.horizon_days}d: no training rows")
    implementation = resolve_algorithm(algorithm)
    ordered = tuple(sorted(levels))
    regressors: dict[float, QuantileRegressor] = {}
    for level in ordered:
        regressors[level] = implementation(level, params, seed=seed).fit(
            matrix.features, matrix.target
        )
    return HorizonQuantileModel(
        horizon_days=matrix.horizon_days,
        algorithm=algorithm,
        levels=ordered,
        regressors=regressors,
    )


@dataclass
class MultiHorizonForecaster:
    """All horizons of one model version, plus the metadata that identifies it."""

    metadata: ModelMetadata
    models: dict[int, HorizonQuantileModel]
    params: Mapping[str, Any]

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(sorted(self.models))

    @property
    def levels(self) -> tuple[float, ...]:
        for model in self.models.values():
            return model.levels
        return ()

    @property
    def feature_names(self) -> tuple[str, ...]:
        for model in self.models.values():
            return model.feature_names
        return ()

    def predict_horizon(
        self, features: pd.DataFrame, horizon_days: int, *, repair_crossing: bool = True
    ) -> tuple[pd.DataFrame, int]:
        """Quantiles for one horizon, plus the number of crossings found."""
        if horizon_days not in self.models:
            raise ModelError(
                f"model {self.metadata.model_version} has no horizon {horizon_days}d; "
                f"available: {list(self.horizons)}"
            )
        model = self.models[horizon_days]
        predicted = model.predict(features)
        crossings = count_crossings(predicted, model.levels)
        if crossings and repair_crossing:
            affected = count_crossing_rows(predicted, model.levels)
            predicted = enforce_ordering(predicted, model.levels)
            logger.warning(
                "h=%dd: repaired %d quantile crossings affecting %d of %d rows (%.1f%%)",
                horizon_days,
                crossings,
                affected,
                len(predicted),
                100.0 * affected / max(len(predicted), 1),
            )
        return predicted, crossings

    def predict(
        self, features: pd.DataFrame, *, repair_crossing: bool = True
    ) -> dict[int, pd.DataFrame]:
        """Quantiles for every horizon the model holds."""
        return {
            horizon: self.predict_horizon(
                features, horizon, repair_crossing=repair_crossing
            )[0]
            for horizon in self.horizons
        }

    def with_metadata(self, **changes: Any) -> "MultiHorizonForecaster":
        return MultiHorizonForecaster(
            metadata=replace(self.metadata, **changes),
            models=self.models,
            params=self.params,
        )

    def describe(self) -> str:
        return (
            f"{self.metadata.describe()} horizons={len(self.models)} "
            f"quantiles={len(self.levels)} features={len(self.feature_names)}"
        )

    # ------------------------------------------------------------ serialisation

    def artifact_path(self, models_dir: Path) -> Path:
        return models_dir / f"{self.metadata.model_version}{ARTIFACT_SUFFIX}"

    def save(self, models_dir: Path) -> Path:
        """Write the bundle atomically, so a crash cannot leave a partial model."""
        models_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_path(models_dir)
        payload = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "metadata": self.metadata.to_dict(),
            "params": dict(self.params),
            "horizons": {
                str(horizon): {
                    "algorithm": model.algorithm,
                    "levels": [float(level) for level in model.levels],
                    "regressors": {
                        str(level): regressor.to_text()
                        for level, regressor in model.regressors.items()
                    },
                }
                for horizon, model in self.models.items()
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
        temporary.replace(path)
        logger.info(
            "saved %s (%.1f MB)", path.name, path.stat().st_size / (1024 * 1024)
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "MultiHorizonForecaster":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        version = int(payload.get("format_version", 0))
        if version != ARTIFACT_FORMAT_VERSION:
            raise ModelError(
                f"{path.name}: artifact format v{version} is not readable by "
                f"v{ARTIFACT_FORMAT_VERSION}"
            )

        raw_metadata = dict(payload["metadata"])
        raw_metadata["horizons"] = tuple(raw_metadata.get("horizons", ()))
        raw_metadata["quantiles"] = tuple(raw_metadata.get("quantiles", ()))
        raw_metadata["feature_names"] = tuple(raw_metadata.get("feature_names", ()))
        metadata = ModelMetadata(**raw_metadata)

        params = dict(payload.get("params", {}))
        models: dict[int, HorizonQuantileModel] = {}
        for horizon_text, entry in payload["horizons"].items():
            implementation = resolve_algorithm(entry["algorithm"])
            levels = tuple(float(level) for level in entry["levels"])
            regressors = {
                float(level): implementation.from_text(text, float(level), params)
                for level, text in entry["regressors"].items()
            }
            models[int(horizon_text)] = HorizonQuantileModel(
                horizon_days=int(horizon_text),
                algorithm=entry["algorithm"],
                levels=levels,
                regressors=regressors,
            )
        return cls(metadata=metadata, models=models, params=params)


def build_metadata(
    *,
    model_id: str,
    model_version: str,
    algorithm: str,
    config_version: str,
    feature_version: str,
    horizon_grid_version: str,
    training_window_strategy: str,
    training_start: str | None,
    training_cutoff: str,
    training_rows: int,
    training_rows_by_horizon: Mapping[int, int],
    horizons: tuple[int, ...],
    quantiles: tuple[float, ...],
    feature_names: tuple[str, ...],
    hyperparameters: Mapping[str, Any],
    random_seed: int,
    notes: str = "",
) -> ModelMetadata:
    """Assemble the MODEL_SPEC.md section 10 record for a freshly trained model."""
    return ModelMetadata(
        model_id=model_id,
        model_version=model_version,
        algorithm=algorithm,
        feature_version=feature_version,
        horizon_grid_version=horizon_grid_version,
        training_cutoff=training_cutoff,
        config_version=config_version,
        training_window_strategy=training_window_strategy,
        training_start=training_start,
        training_rows=training_rows,
        training_rows_by_horizon=dict(training_rows_by_horizon),
        horizons=horizons,
        quantiles=quantiles,
        feature_names=feature_names,
        hyperparameters=dict(hyperparameters),
        random_seed=random_seed,
        created_at=utc_now_iso(),
        notes=notes,
    )
