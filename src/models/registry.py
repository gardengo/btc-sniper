"""The model registry: which model exists, and which one is production.

CLAUDE.md section 2.3 is the rule this module enforces. A newly trained model is
a **candidate**. Becoming **production** is a separate, explicit act that
requires a promotion decision (Phase 8), and it can never happen as a side
effect of training. So `register()` always writes `candidate`, and only
`promote()` can write `production`.

There is at most one production model at a time. Promotion retires the incumbent
in the same transaction, because a moment with two production models is a moment
where the daily forecast job cannot say which one produced its output.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.models.base import (
    STATUS_CANDIDATE,
    STATUS_PRODUCTION,
    STATUS_REJECTED,
    STATUS_RETIRED,
    ModelMetadata,
)
from src.models.forecaster import MultiHorizonForecaster
from src.utils.logging import get_logger
from src.utils.timeutils import utc_now_iso

logger = get_logger(__name__)


class RegistryError(RuntimeError):
    """Raised when a registry operation would leave an inconsistent state."""


def _json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def register(
    connection: sqlite3.Connection,
    metadata: ModelMetadata,
    *,
    artifact_path: Path | None = None,
) -> str:
    """Record a freshly trained model as a candidate.

    Refuses any status but `candidate`: training must never be able to promote.
    """
    if metadata.status != STATUS_CANDIDATE:
        raise RegistryError(
            f"register() only accepts candidates, got {metadata.status!r}; "
            "promotion is a separate decision (CLAUDE.md section 2.3)"
        )
    connection.execute(
        "INSERT INTO model_registry (model_id, model_version, algorithm, "
        "library_versions, feature_version, horizon_grid_version, config_version, "
        "code_commit, training_window_strategy, training_start, training_cutoff, "
        "training_rows, hyperparameters, random_seed, validation_metrics, "
        "test_metrics, status, artifact_path, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            metadata.model_id,
            metadata.model_version,
            metadata.algorithm,
            _json(dict(metadata.library_versions)),
            metadata.feature_version,
            metadata.horizon_grid_version,
            metadata.config_version,
            metadata.code_commit,
            metadata.training_window_strategy,
            metadata.training_start,
            metadata.training_cutoff,
            metadata.training_rows,
            _json(
                {
                    "params": dict(metadata.hyperparameters),
                    "horizons": list(metadata.horizons),
                    "quantiles": list(metadata.quantiles),
                    "training_rows_by_horizon": dict(metadata.training_rows_by_horizon),
                    "feature_count": len(metadata.feature_names),
                }
            ),
            metadata.random_seed,
            _json(dict(metadata.validation_metrics)),
            _json(dict(metadata.test_metrics)),
            STATUS_CANDIDATE,
            str(artifact_path) if artifact_path else None,
            metadata.created_at or utc_now_iso(),
        ),
    )
    logger.info("registered candidate %s", metadata.model_version)
    return metadata.model_id


def record_metrics(
    connection: sqlite3.Connection,
    model_version: str,
    *,
    validation_metrics: Mapping[str, Any] | None = None,
    test_metrics: Mapping[str, Any] | None = None,
) -> None:
    """Attach metrics to a registered model.

    Writing outer-test metrics twice for one model version is refused here:
    VALIDATION_SPEC.md section 4.4 allows one evaluation per version, and the
    registry is the only place that can actually enforce it.
    """
    row = get(connection, model_version)
    if row is None:
        raise RegistryError(f"unknown model_version {model_version!r}")

    if test_metrics is not None:
        existing = json.loads(row["test_metrics"] or "{}")
        if existing:
            raise RegistryError(
                f"{model_version} already has outer-test metrics recorded. "
                "A model version may be evaluated on the outer test once "
                "(VALIDATION_SPEC.md section 4.4); a changed design needs a new "
                "model_version."
            )
        connection.execute(
            "UPDATE model_registry SET test_metrics = ? WHERE model_version = ?",
            (_json(dict(test_metrics)), model_version),
        )
        logger.info("recorded the single outer-test evaluation for %s", model_version)

    if validation_metrics is not None:
        connection.execute(
            "UPDATE model_registry SET validation_metrics = ? WHERE model_version = ?",
            (_json(dict(validation_metrics)), model_version),
        )


def promote(
    connection: sqlite3.Connection, model_version: str, *, reason: str = ""
) -> None:
    """Make a candidate the production model, retiring the incumbent.

    Both writes happen in the caller's transaction so there is never a moment
    with two production models or none.
    """
    row = get(connection, model_version)
    if row is None:
        raise RegistryError(f"unknown model_version {model_version!r}")
    if row["status"] == STATUS_PRODUCTION:
        logger.info("%s is already in production", model_version)
        return
    if row["status"] != STATUS_CANDIDATE:
        raise RegistryError(
            f"cannot promote {model_version}: status is {row['status']!r}, "
            f"only {STATUS_CANDIDATE!r} models can be promoted"
        )

    stamp = utc_now_iso()
    incumbent = production_model(connection)
    if incumbent is not None:
        connection.execute(
            "UPDATE model_registry SET status = ?, retired_at = ? WHERE model_version = ?",
            (STATUS_RETIRED, stamp, incumbent["model_version"]),
        )
        logger.info("retired %s", incumbent["model_version"])
    connection.execute(
        "UPDATE model_registry SET status = ?, promoted_at = ?, rejection_reason = ? "
        "WHERE model_version = ?",
        (STATUS_PRODUCTION, stamp, reason or None, model_version),
    )
    logger.info("promoted %s to production", model_version)


def reject(connection: sqlite3.Connection, model_version: str, reason: str) -> None:
    """Mark a candidate rejected, recording why.

    The reason is mandatory. A rejected model with no recorded reason is a
    decision nobody can review later (OPERATING_SPEC.md weekly review).
    """
    if not reason.strip():
        raise RegistryError("a rejection must record a reason")
    updated = connection.execute(
        "UPDATE model_registry SET status = ?, rejection_reason = ? "
        "WHERE model_version = ? AND status = ?",
        (STATUS_REJECTED, reason, model_version, STATUS_CANDIDATE),
    ).rowcount
    if not updated:
        raise RegistryError(
            f"cannot reject {model_version}: it is not a registered candidate"
        )
    logger.info("rejected %s: %s", model_version, reason)


def get(connection: sqlite3.Connection, model_version: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM model_registry WHERE model_version = ?", (model_version,)
    ).fetchone()


def production_model(connection: sqlite3.Connection) -> sqlite3.Row | None:
    """The current production model, or ``None`` when nothing is promoted yet."""
    rows = connection.execute(
        "SELECT * FROM model_registry WHERE status = ? ORDER BY promoted_at DESC",
        (STATUS_PRODUCTION,),
    ).fetchall()
    if len(rows) > 1:
        raise RegistryError(
            f"{len(rows)} models are marked production: "
            f"{[row['model_version'] for row in rows]}. "
            "The daily forecast cannot say which one produced its output."
        )
    return rows[0] if rows else None


def list_models(
    connection: sqlite3.Connection, *, status: str | None = None, limit: int = 50
) -> pd.DataFrame:
    query = (
        "SELECT model_version, algorithm, status, training_window_strategy, "
        "training_start, training_cutoff, training_rows, code_commit, created_at, "
        "promoted_at, retired_at, artifact_path FROM model_registry"
    )
    params: list[Any] = []
    if status is not None:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(int(limit))
    return pd.read_sql_query(query, connection, params=params)


def load_artifact(
    connection: sqlite3.Connection, model_version: str
) -> MultiHorizonForecaster:
    """Load the stored model bundle for a registered version."""
    row = get(connection, model_version)
    if row is None:
        raise RegistryError(f"unknown model_version {model_version!r}")
    path = row["artifact_path"]
    if not path:
        raise RegistryError(f"{model_version} has no stored artifact")
    artifact = Path(path)
    if not artifact.is_file():
        raise RegistryError(f"{model_version} artifact is missing: {artifact}")
    forecaster = MultiHorizonForecaster.load(artifact)
    if forecaster.metadata.model_version != model_version:
        raise RegistryError(
            f"artifact at {artifact} holds {forecaster.metadata.model_version!r}, "
            f"not {model_version!r}"
        )
    return forecaster


def load_production(
    connection: sqlite3.Connection,
) -> MultiHorizonForecaster | None:
    """The production model bundle, or ``None`` when nothing is promoted."""
    row = production_model(connection)
    if row is None:
        return None
    forecaster = load_artifact(connection, row["model_version"])
    return MultiHorizonForecaster(
        metadata=replace(forecaster.metadata, status=STATUS_PRODUCTION),
        models=forecaster.models,
        params=forecaster.params,
    )


def start_run(
    connection: sqlite3.Connection,
    run_id: str,
    run_type: str,
    *,
    config_version: str = "",
    code_commit: str = "",
    random_seed: int | None = None,
    dataset_cutoff: str | None = None,
) -> None:
    """Open a `model_runs` row so a crashed run stays visible as `running`.

    No ``model_id`` yet: the run is opened *before* training, so the model it
    will produce does not exist and the foreign key could not be satisfied.
    `finish_run` attaches it once the model is registered.
    """
    connection.execute(
        "INSERT INTO model_runs (run_id, run_type, started_at, status, "
        "config_version, code_commit, random_seed, dataset_cutoff) "
        "VALUES (?, ?, ?, 'running', ?, ?, ?, ?)",
        (
            run_id,
            run_type,
            utc_now_iso(),
            config_version,
            code_commit,
            random_seed,
            dataset_cutoff,
        ),
    )


def finish_run(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    status: str,
    model_id: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    notes: str = "",
) -> None:
    """Close a run, attaching the model it produced if it produced one."""
    if status not in {"succeeded", "failed"}:
        raise RegistryError(f"invalid run status {status!r}")
    connection.execute(
        "UPDATE model_runs SET finished_at = ?, status = ?, model_id = ?, "
        "metrics = ?, notes = ? WHERE run_id = ?",
        (utc_now_iso(), status, model_id, _json(dict(metrics or {})), notes, run_id),
    )


def metadata_from_row(row: sqlite3.Row) -> ModelMetadata:
    """Rebuild metadata from a registry row (without loading the artifact)."""
    hyperparameters = json.loads(row["hyperparameters"] or "{}")
    return ModelMetadata(
        model_id=row["model_id"],
        model_version=row["model_version"],
        algorithm=row["algorithm"],
        feature_version=row["feature_version"],
        horizon_grid_version=row["horizon_grid_version"],
        training_cutoff=row["training_cutoff"],
        status=row["status"],
        config_version=row["config_version"] or "",
        code_commit=row["code_commit"] or "",
        library_versions=json.loads(row["library_versions"] or "{}"),
        training_window_strategy=row["training_window_strategy"] or "",
        training_start=row["training_start"],
        training_rows=int(row["training_rows"] or 0),
        training_rows_by_horizon={
            int(key): int(value)
            for key, value in hyperparameters.get("training_rows_by_horizon", {}).items()
        },
        horizons=tuple(hyperparameters.get("horizons", ())),
        quantiles=tuple(hyperparameters.get("quantiles", ())),
        hyperparameters=hyperparameters.get("params", {}),
        random_seed=int(row["random_seed"] or 0),
        validation_metrics=json.loads(row["validation_metrics"] or "{}"),
        test_metrics=json.loads(row["test_metrics"] or "{}"),
        created_at=row["created_at"],
    )


def assert_single_production(connection: sqlite3.Connection) -> None:
    """Guard used by jobs before they rely on `production_model()`."""
    production_model(connection)


__all__ = [
    "RegistryError",
    "assert_single_production",
    "finish_run",
    "get",
    "list_models",
    "load_artifact",
    "load_production",
    "metadata_from_row",
    "production_model",
    "promote",
    "record_metrics",
    "register",
    "reject",
    "start_run",
]
