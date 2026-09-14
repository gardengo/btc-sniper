"""Joining stored forecasts to what actually happened.

OPERATING_SPEC.md section 7: when a target close becomes available, **update the
realization row** -- never write a second prediction. A forecast is a statement
made at a point in time; rewriting it later, or storing a second version of it,
destroys the only honest record of what was claimed.

Evaluation status (VALIDATION_SPEC.md section 12)
-------------------------------------------------
A single ``(forecast, horizon)`` row is either `pending` or `fully_evaluated`:
its one target date has arrived or it has not. `partially_evaluable` describes a
whole **forecast**, whose 1-day horizon may have resolved a year before its
365-day horizon. Both live in the same enum because the schema stores the row
status and the dashboard reports the forecast status.

Nothing here invents an actual. A missing target close leaves the row pending.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.evaluation.metrics import pinball_loss
from src.forecast.quantiles import resolve_interval
from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

STATUS_PENDING: str = "pending"
STATUS_PARTIAL: str = "partially_evaluable"
STATUS_FULL: str = "fully_evaluated"


class RealizationError(RuntimeError):
    """Raised when forecasts cannot be reconciled with actuals."""


def row_status(target_date: pd.Timestamp, data_end: pd.Timestamp) -> str:
    """Status of one forecast horizon."""
    return STATUS_FULL if target_date <= data_end else STATUS_PENDING


def forecast_status(resolved: int, total: int) -> str:
    """Status of a whole forecast, from how many of its horizons have resolved."""
    if total <= 0:
        raise RealizationError("a forecast with no horizons has no status")
    if resolved == 0:
        return STATUS_PENDING
    if resolved < total:
        return STATUS_PARTIAL
    return STATUS_FULL


@dataclass(frozen=True)
class RealizationResult:
    """Outcome of one realization pass."""

    rows: pd.DataFrame
    resolved: int
    pending: int
    forecasts: int

    def describe(self) -> str:
        return (
            f"forecasts={self.forecasts} rows={len(self.rows)} "
            f"resolved={self.resolved} pending={self.pending}"
        )


def _interval_flags(
    actual: float, quantiles: dict[float, float], levels: tuple[float, ...]
) -> dict[str, int | None]:
    """Whether the actual fell inside each reported interval."""
    flags: dict[str, int | None] = {
        "in_interval_50": None,
        "in_interval_80": None,
        "in_interval_95": None,
    }
    for interval in (0.50, 0.80, 0.95):
        try:
            low, high = resolve_interval(interval, levels)
        except Exception:  # noqa: BLE001 - an unreported interval is simply absent
            continue
        tag = f"in_interval_{int(round(interval * 100))}"
        flags[tag] = int(quantiles[low] <= actual <= quantiles[high])
    return flags


def realize_points(
    points: pd.DataFrame,
    quantiles: pd.DataFrame,
    close: pd.Series,
    *,
    regimes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach actuals and errors to stored forecast points.

    ``points`` carries one row per (forecast_id, horizon_days); ``quantiles``
    carries the long-format quantile rows for the same forecasts. Rows whose
    target date has not arrived come back `pending` with empty error columns.
    """
    if points.empty:
        return pd.DataFrame()

    data_end = close.index.max()
    grouped = {
        key: group.set_index("quantile")["predicted_log_return"].to_dict()
        for key, group in quantiles.groupby(["forecast_id", "horizon_days"])
    }

    records: list[dict[str, Any]] = []
    for point in points.to_dict("records"):
        forecast_id = point["forecast_id"]
        horizon = int(point["horizon_days"])
        target_date = pd.Timestamp(point["target_date"])
        origin_date = pd.Timestamp(point["forecast_origin_date"])
        origin_close = float(point["origin_close"])
        predicted_log_return = float(point["predicted_log_return"])
        predicted_price = float(point["predicted_price"])

        regime = None
        if regimes is not None and origin_date in regimes.index:
            regime = str(regimes.loc[origin_date, "direction"])

        status = row_status(target_date, data_end)
        record: dict[str, Any] = {
            "forecast_id": forecast_id,
            "horizon_days": horizon,
            "target_date": target_date.strftime("%Y-%m-%d"),
            "evaluation_status": status,
            "actual_close": None,
            "actual_log_return": None,
            "absolute_error": None,
            "percentage_error": None,
            "log_return_error": None,
            "direction_predicted": int(point["direction_predicted"]),
            "direction_actual": None,
            "direction_correct": None,
            "in_interval_50": None,
            "in_interval_80": None,
            "in_interval_95": None,
            "pinball_loss": None,
            "regime": regime,
        }

        if status == STATUS_FULL and target_date in close.index:
            actual_close = float(close.loc[target_date])
            actual_log_return = float(np.log(actual_close / origin_close))
            record.update(
                {
                    "actual_close": actual_close,
                    "actual_log_return": actual_log_return,
                    "absolute_error": abs(predicted_price - actual_close),
                    "percentage_error": (predicted_price - actual_close)
                    / actual_close
                    * 100.0,
                    "log_return_error": predicted_log_return - actual_log_return,
                    "direction_actual": int(np.sign(actual_log_return)),
                }
            )
            record["direction_correct"] = (
                None
                if record["direction_predicted"] == 0 or record["direction_actual"] == 0
                else int(record["direction_predicted"] == record["direction_actual"])
            )
            levels_map = grouped.get((forecast_id, horizon), {})
            if levels_map:
                levels = tuple(sorted(levels_map))
                record.update(_interval_flags(actual_log_return, levels_map, levels))
                losses = [
                    pinball_loss(
                        pd.Series([actual_log_return]),
                        pd.Series([levels_map[level]]),
                        level,
                    )
                    for level in levels
                ]
                record["pinball_loss"] = float(np.mean(losses))
        elif status == STATUS_FULL:
            # The target date is inside the data range but the candle is missing;
            # that is a data problem, not a resolved forecast.
            record["evaluation_status"] = STATUS_PENDING
            logger.warning(
                "forecast %s h=%dd targets %s, which has no stored candle",
                str(forecast_id)[:8],
                horizon,
                target_date.date(),
            )
        records.append(record)

    return pd.DataFrame(records)


def summarise(rows: pd.DataFrame) -> RealizationResult:
    if rows.empty:
        return RealizationResult(rows=rows, resolved=0, pending=0, forecasts=0)
    resolved = int((rows["evaluation_status"] == STATUS_FULL).sum())
    return RealizationResult(
        rows=rows,
        resolved=resolved,
        pending=len(rows) - resolved,
        forecasts=int(rows["forecast_id"].nunique()),
    )


def forecast_statuses(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-forecast status, for the prediction-log view."""
    if rows.empty:
        return pd.DataFrame(columns=["forecast_id", "horizons", "resolved", "status"])
    grouped = rows.groupby("forecast_id")
    summary = pd.DataFrame(
        {
            "horizons": grouped.size(),
            "resolved": grouped["evaluation_status"]
            .apply(lambda values: int((values == STATUS_FULL).sum())),
        }
    ).reset_index()
    summary["status"] = [
        forecast_status(int(row.resolved), int(row.horizons))
        for row in summary.itertuples(index=False)
    ]
    return summary


def drop_in_sample(
    rows: pd.DataFrame, origins: pd.Series, training_cutoffs: Mapping[str, str]
) -> tuple[pd.DataFrame, int]:
    """Remove forecasts whose origin is inside their own model's training data.

    Backfilling forecasts for past dates with a model trained *through* those
    dates produces realizations that look excellent and mean nothing: the model
    saw the answer. This is the single easiest way to accidentally manufacture a
    flattering production metric, so the exclusion happens here rather than
    relying on whoever runs the backfill to remember.

    Returns the surviving rows and how many were dropped.
    """
    if rows.empty or not training_cutoffs:
        return rows, 0
    origin_by_row = rows["forecast_id"].map(origins)
    cutoff_by_row = rows["forecast_id"].map(
        {key: pd.Timestamp(value) for key, value in training_cutoffs.items()}
    )
    in_sample = pd.to_datetime(origin_by_row) <= cutoff_by_row
    in_sample = in_sample.fillna(False)
    dropped = int(in_sample.sum())
    if dropped:
        logger.warning(
            "excluded %d realization rows whose forecast origin is inside the "
            "model's own training data; they are not out-of-sample evidence",
            dropped,
        )
    return rows.loc[~in_sample.to_numpy()], dropped


def production_metrics(
    rows: pd.DataFrame,
    model_versions: pd.Series,
    config: AppConfig,
    *,
    regimes: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Aggregate realized forecasts into tidy `performance_metrics` rows.

    Only `fully_evaluated` rows contribute. OPERATING_SPEC.md section 4 requires
    a minimum realized sample before a horizon's production metric may drive a
    promotion decision, so `sample_size` is carried on every row and
    `min_realized_forecasts_per_horizon` is reported alongside it.
    """
    resolved = rows[rows["evaluation_status"] == STATUS_FULL].copy()
    if resolved.empty:
        return pd.DataFrame(
            columns=[
                "model_version",
                "horizon_days",
                "regime",
                "metric_name",
                "metric_value",
                "sample_size",
                "period_start",
                "period_end",
            ]
        )
    resolved["model_version"] = resolved["forecast_id"].map(model_versions)

    records: list[dict[str, Any]] = []
    for (version, horizon), group in resolved.groupby(
        ["model_version", "horizon_days"], sort=True
    ):
        records.extend(_metric_rows(version, int(horizon), "all", group))
        for label in regimes:
            subset = group[group["regime"] == label]
            if not subset.empty:
                records.extend(_metric_rows(version, int(horizon), label, subset))
    return pd.DataFrame(records)


def _metric_rows(
    model_version: str, horizon: int, regime: str, group: pd.DataFrame
) -> list[dict[str, Any]]:
    directional = group["direction_correct"].dropna()
    values = {
        "sample_size": float(len(group)),
        "price_mae": float(group["absolute_error"].mean()),
        "price_rmse": float(np.sqrt((group["absolute_error"] ** 2).mean())),
        "return_mae": float(group["log_return_error"].abs().mean()),
        "return_bias": float(group["log_return_error"].mean()),
        "pinball_mean": float(group["pinball_loss"].mean()),
        "direction_accuracy": float(directional.mean()) if not directional.empty else float("nan"),
        "direction_calls": float(len(directional)),
        "coverage_50": float(group["in_interval_50"].dropna().mean()),
        "coverage_80": float(group["in_interval_80"].dropna().mean()),
        "coverage_95": float(group["in_interval_95"].dropna().mean()),
    }
    period_start = group["target_date"].min()
    period_end = group["target_date"].max()
    return [
        {
            "model_version": model_version,
            "horizon_days": horizon,
            "regime": regime,
            "metric_name": name,
            "metric_value": None if pd.isna(value) else float(value),
            "sample_size": len(group),
            "period_start": period_start,
            "period_end": period_end,
        }
        for name, value in values.items()
    ]


def horizons_with_enough_evidence(
    metrics: pd.DataFrame, minimum: int
) -> pd.DataFrame:
    """Which horizons have enough realized forecasts to be acted on.

    OPERATING_SPEC.md section 4. Reported rather than filtered: a horizon with
    too few realizations is still worth seeing, it just cannot decide anything.
    """
    if metrics.empty:
        return pd.DataFrame(columns=["horizon_days", "sample_size", "decision_ready"])
    sizes = (
        metrics[metrics["regime"] == "all"]
        .groupby("horizon_days")["sample_size"]
        .max()
        .reset_index()
    )
    sizes["decision_ready"] = sizes["sample_size"] >= int(minimum)
    return sizes


def realizable_query_bounds(close: pd.Series) -> tuple[str, str]:
    """First and last dates for which an actual close exists."""
    if close.empty:
        raise RealizationError("no closes are stored; nothing can be realized")
    return (
        close.index.min().strftime("%Y-%m-%d"),
        close.index.max().strftime("%Y-%m-%d"),
    )


def load_and_realize(
    connection: sqlite3.Connection,
    close: pd.Series,
    *,
    source: str,
    symbol: str,
    timeframe: str = "1d",
    regimes: pd.DataFrame | None = None,
    model_version: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Read every stored forecast point and realize what can be realized."""
    from src.storage import repositories as repo

    points = repo.load_forecast_points_for_realization(
        connection,
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        model_version=model_version,
    )
    if points.empty:
        return pd.DataFrame(), pd.Series(dtype="object")
    quantiles = repo.load_forecast_quantiles_long(
        connection, points["forecast_id"].unique().tolist()
    )
    rows = realize_points(points, quantiles, close, regimes=regimes)
    versions = points.drop_duplicates("forecast_id").set_index("forecast_id")[
        "model_version"
    ]
    return rows, versions
