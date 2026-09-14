"""Training one model version across every horizon.

This is where the validation rules stop being documentation and start being
code paths. Every fit goes through here, and every fit is purged, window-limited
and recorded before it happens.

Origin selection, in order
--------------------------
1. start from the usable feature rows (post-warmup, no NaN)
2. drop everything the training window excludes (`expanding`, `rolling_Ny`)
3. drop everything the purge excludes: ``origin + horizon + embargo < cutoff``
4. drop origins whose label has not resolved yet
5. assert, again, that nothing left reaches into the outer test

Step 5 is redundant by construction. It stays because the cost of it being
redundant is microseconds and the cost of it being necessary is a worthless
final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from src.models.base import ModelError, new_model_id
from src.models.dataset import TrainingMatrix, build_training_matrix
from src.models.forecaster import (
    MultiHorizonForecaster,
    build_metadata,
    fit_horizon_model,
)
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.validation.folds import EXPANDING, window_start
from src.validation.splits import (
    SplitBoundaries,
    eligible_training_origins,
    last_eligible_training_origin,
)

logger = get_logger(__name__)

ALGORITHM_SHORT_NAMES: dict[str, str] = {"lightgbm_quantile": "lgbmq"}


class TrainingError(RuntimeError):
    """Raised when a model version cannot be trained."""


def model_version_name(
    algorithm: str, strategy: str, cutoff: pd.Timestamp, config_version: str, suffix: str = ""
) -> str:
    """Deterministic, human-readable model version.

    Everything that changes the model is in the name -- algorithm, window,
    training cutoff, config version -- so two versions that differ in any of
    them cannot collide, and one glance says what a version is.
    """
    short = ALGORITHM_SHORT_NAMES.get(algorithm, algorithm)
    base = f"{short}-{strategy}-{cutoff:%Y%m%d}-{config_version}"
    return f"{base}-{suffix}" if suffix else base


@dataclass(frozen=True)
class TrainingRequest:
    """Everything that defines one training run."""

    horizons: tuple[int, ...]
    levels: tuple[float, ...]
    strategy: str
    algorithm: str
    params: Mapping[str, Any]
    seed: int
    cutoff: pd.Timestamp | None = None
    suffix: str = ""

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        horizons: tuple[int, ...],
        strategy: str | None = None,
        cutoff: pd.Timestamp | None = None,
        suffix: str = "",
        params: Mapping[str, Any] | None = None,
    ) -> "TrainingRequest":
        models = config.section("models")
        return cls(
            horizons=tuple(sorted(horizons)),
            levels=config.forecast.quantiles,
            strategy=strategy or str(models.get("default_training_window", EXPANDING)),
            algorithm=str(models.get("primary_algorithm", "lightgbm_quantile")),
            params=dict(params if params is not None else models.get("lightgbm", {})),
            seed=int(models.get("random_seed", 42)),
            cutoff=cutoff,
            suffix=suffix,
        )


def training_origins(
    candidates: pd.DatetimeIndex,
    boundaries: SplitBoundaries,
    horizon_days: int,
    *,
    strategy: str,
    cutoff: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """Origins a model may train on, after the window and the purge."""
    if len(candidates) == 0:
        return candidates
    eligible = eligible_training_origins(
        candidates, boundaries, horizon_days, cutoff=cutoff
    )
    if len(eligible) == 0:
        return eligible
    reference = cutoff if cutoff is not None else boundaries.outer_test_start
    start = window_start(strategy, reference, candidates.min())
    return eligible[eligible >= start]


def build_matrices(
    features: pd.DataFrame,
    close: pd.Series,
    boundaries: SplitBoundaries,
    request: TrainingRequest,
) -> dict[int, TrainingMatrix]:
    """One purged, window-limited training matrix per horizon."""
    candidates = features.dropna(axis=0, how="any").index
    matrices: dict[int, TrainingMatrix] = {}
    for horizon in request.horizons:
        origins = training_origins(
            candidates,
            boundaries,
            horizon,
            strategy=request.strategy,
            cutoff=request.cutoff,
        )
        matrix = build_training_matrix(
            features, close, horizon, origins, boundaries=boundaries
        )
        if matrix.rows == 0:
            logger.warning("h=%dd: no training rows after purge; skipped", horizon)
            continue
        matrices[horizon] = matrix
    if not matrices:
        raise TrainingError("no horizon had any training rows after purging")
    return matrices


def train_forecaster(
    features: pd.DataFrame,
    close: pd.Series,
    boundaries: SplitBoundaries,
    config: AppConfig,
    request: TrainingRequest,
    *,
    notes: str = "",
) -> MultiHorizonForecaster:
    """Fit every (horizon, quantile) model for one model version."""
    matrices = build_matrices(features, close, boundaries, request)

    models = {}
    for horizon, matrix in matrices.items():
        logger.info("fitting h=%dd: %s", horizon, matrix.describe())
        models[horizon] = fit_horizon_model(
            matrix,
            algorithm=request.algorithm,
            levels=request.levels,
            params=request.params,
            seed=request.seed,
        )

    longest = max(matrices)
    shortest = min(matrices)
    reference = matrices[shortest]
    effective_cutoff = (
        request.cutoff
        if request.cutoff is not None
        else last_eligible_training_origin(boundaries, shortest)
    )
    starts = [matrix.origins.min() for matrix in matrices.values()]

    metadata = build_metadata(
        model_id=new_model_id(),
        model_version=model_version_name(
            request.algorithm,
            request.strategy,
            pd.Timestamp(effective_cutoff),
            config.config_version,
            request.suffix,
        ),
        algorithm=request.algorithm,
        config_version=config.config_version,
        feature_version=config.features.version,
        horizon_grid_version=config.forecast.horizon_grid_version,
        training_window_strategy=request.strategy,
        training_start=min(starts).strftime("%Y-%m-%d"),
        training_cutoff=pd.Timestamp(effective_cutoff).strftime("%Y-%m-%d"),
        training_rows=reference.rows,
        training_rows_by_horizon={
            horizon: matrix.rows for horizon, matrix in matrices.items()
        },
        horizons=tuple(sorted(models)),
        quantiles=request.levels,
        feature_names=reference.feature_names,
        hyperparameters=request.params,
        random_seed=request.seed,
        notes=notes,
    )
    forecaster = MultiHorizonForecaster(
        metadata=metadata, models=models, params=dict(request.params)
    )
    logger.info(
        "trained %s: %d horizons (%dd..%dd), %d quantiles, %d..%d rows per horizon",
        forecaster.metadata.model_version,
        len(models),
        shortest,
        longest,
        len(request.levels),
        min(matrix.rows for matrix in matrices.values()),
        max(matrix.rows for matrix in matrices.values()),
    )
    return forecaster


def assert_reproducible(
    features: pd.DataFrame,
    close: pd.Series,
    boundaries: SplitBoundaries,
    config: AppConfig,
    request: TrainingRequest,
    *,
    horizon_days: int,
) -> bool:
    """Refit one horizon and check the predictions are bitwise identical.

    VALIDATION_SPEC.md section 11 requires reproducibility; this makes it
    checkable in a job rather than only in a test, because the thing that breaks
    it in practice is a library upgrade, not a code change.
    """
    single = TrainingRequest(
        horizons=(horizon_days,),
        levels=request.levels,
        strategy=request.strategy,
        algorithm=request.algorithm,
        params=request.params,
        seed=request.seed,
        cutoff=request.cutoff,
    )
    matrices = build_matrices(features, close, boundaries, single)
    matrix = matrices[horizon_days]
    first = fit_horizon_model(
        matrix,
        algorithm=single.algorithm,
        levels=single.levels,
        params=single.params,
        seed=single.seed,
    ).predict(matrix.features)
    second = fit_horizon_model(
        matrix,
        algorithm=single.algorithm,
        levels=single.levels,
        params=single.params,
        seed=single.seed,
    ).predict(matrix.features)
    identical = first.equals(second)
    if not identical:
        raise ModelError(
            f"h={horizon_days}d is not reproducible: two fits on identical data "
            "produced different predictions"
        )
    return identical
