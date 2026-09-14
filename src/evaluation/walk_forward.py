"""Walk-forward validation: the only place a model is allowed to be judged.

VALIDATION_SPEC.md sections 2, 5 and 8 together define what this has to do:
train and score fold by fold in chronological order, never touch the outer test,
and report every result against the naive baselines on identical origins.

What makes the comparison honest
--------------------------------
**The model and the baselines see the same origins.** Every fold's validation
origins are spaced by `min(horizon, cap)` and then intersected with the origins
all the baselines can score. A model evaluated on a slightly different origin
set than its reference is not being compared to it.

**Every fold refits from scratch.** Nothing is carried between folds, so a fold's
model has never seen its own validation block in any form.

**Per-fold results are kept, not just the average.** VALIDATION_SPEC.md section 7
requires fold dispersion and the worst fold. A model with a good mean and one
catastrophic fold is not a good model, and averaging is exactly the operation
that hides it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.evaluation.baseline_eval import model_version as baseline_version
from src.evaluation.baseline_eval import origin_spacing_days, space_origins
from src.evaluation.evaluator import ALL_REGIMES, ForecastSample, evaluate_sample_table
from src.evaluation.metrics import dispersion
from src.forecast.quantiles import (
    count_crossing_rows,
    count_crossings,
    enforce_ordering,
)
from src.models.baselines import REFERENCE_BASELINE, build_baselines, common_valid_origins
from src.models.dataset import build_training_matrix
from src.models.forecaster import fit_horizon_model
from src.models.targets import forward_log_return
from src.models.training import TrainingRequest
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.validation.folds import Fold, FoldPlan, iter_folds
from src.validation.splits import SplitBoundaries, independent_window_count

logger = get_logger(__name__)

MODEL_TAG: str = "model"


class WalkForwardError(RuntimeError):
    """Raised when a walk-forward run cannot produce any result."""


@dataclass
class WalkForwardResult:
    """Per-fold metrics for one training-window strategy."""

    strategy: str
    model_version: str
    metrics: pd.DataFrame
    folds: pd.DataFrame
    crossings: pd.DataFrame
    horizons: tuple[int, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"strategy={self.strategy} horizons={len(self.horizons)} "
            f"folds={self.folds['fold'].nunique() if not self.folds.empty else 0} "
            f"metric_rows={len(self.metrics)}"
        )


def _validation_origins(
    fold: Fold, spacing_days: int, target: pd.Series, features: pd.DataFrame
) -> pd.DatetimeIndex:
    """Scoreable, spaced validation origins with complete features."""
    spaced = space_origins(fold.validation_origins, spacing_days)
    complete = features.reindex(spaced).dropna(axis=0, how="any").index
    return complete.intersection(target.dropna().index).sort_values()


def run_walk_forward(
    features: pd.DataFrame,
    close: pd.Series,
    boundaries: SplitBoundaries,
    config: AppConfig,
    request: TrainingRequest,
    plan: FoldPlan,
    *,
    regimes: pd.DataFrame | None = None,
    include_baselines: bool = True,
    params_name: str = "",
) -> WalkForwardResult:
    """Train and score one strategy across every fold of every horizon.

    ``params_name`` tags the model identity when comparing hyperparameter
    candidates, so two candidates never collide in one metric table.
    """
    model_identity = request_model_version(request, config, params_name=params_name)
    validation_section = config.section("validation")
    spacing_cap = int(validation_section.get("validation_origin_spacing_days", 30))
    regime_labels = tuple(str(label) for label in validation_section.get("regimes", []))
    intervals = config.forecast.show_intervals
    levels = request.levels

    baselines = build_baselines(config.baselines) if include_baselines else []
    candidates = features.dropna(axis=0, how="any").index

    metric_frames: list[pd.DataFrame] = []
    crossing_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for horizon in request.horizons:
        target = forward_log_return(close, horizon)
        spacing = origin_spacing_days(horizon, spacing_cap)
        folds = list(iter_folds(candidates, boundaries, plan, horizon))
        if not folds:
            logger.warning("h=%dd: no usable folds; skipped", horizon)
            continue

        predictions = {
            baseline.name: baseline.predict(close, horizon, levels)
            for baseline in baselines
        }

        for fold in folds:
            origins = _validation_origins(fold, spacing, target, features)
            if include_baselines and origins.size:
                origins = common_valid_origins(list(predictions.values()), origins)
            if origins.empty:
                logger.warning("%s h=%dd: no scoreable validation origins", fold.name, horizon)
                continue

            train = build_training_matrix(
                features,
                close,
                horizon,
                fold.train_origins,
                boundaries=boundaries,
            )
            if train.rows == 0:
                logger.warning("%s h=%dd: no training rows after purge", fold.name, horizon)
                continue

            model = fit_horizon_model(
                train,
                algorithm=request.algorithm,
                levels=levels,
                params=request.params,
                seed=request.seed,
            )
            predicted = model.predict(features.loc[origins])

            pair_crossings = count_crossings(predicted, levels)
            row_crossings = count_crossing_rows(predicted, levels)
            if pair_crossings:
                predicted = enforce_ordering(predicted, levels)
            crossing_rows.append(
                {
                    "horizon_days": horizon,
                    "fold": fold.name,
                    "rows": len(predicted),
                    "crossing_pairs": pair_crossings,
                    "crossing_rows": row_crossings,
                    "crossing_row_rate": row_crossings / max(len(predicted), 1),
                }
            )

            reference = (
                predictions[REFERENCE_BASELINE].restrict(origins).median
                if include_baselines
                else None
            )
            samples: list[tuple[str, pd.DataFrame]] = [(model_identity, predicted)]
            if include_baselines:
                samples.extend(
                    (baseline_version(name, config), prediction.restrict(origins).quantiles)
                    for name, prediction in predictions.items()
                )

            for version, quantiles in samples:
                sample = ForecastSample(
                    model_version=version,
                    horizon_days=horizon,
                    origin_close=close.reindex(origins),
                    actual_log_return=target.reindex(origins),
                    predicted_quantiles=quantiles,
                    origin_spacing_days=spacing,
                    regimes=None if regimes is None else regimes.reindex(origins),
                )
                table = evaluate_sample_table(
                    sample,
                    intervals=intervals,
                    reference_median=reference,
                    regimes=regime_labels,
                    low_power=boundaries.is_low_power(horizon),
                )
                table["fold"] = fold.name
                table["strategy"] = request.strategy
                table["train_rows"] = train.rows
                metric_frames.append(table)

            # Built here rather than from `describe_folds`, which only sees the
            # raw daily origins: the number that matters is how many origins were
            # actually scored after spacing, and what they are worth once overlap
            # is accounted for.
            fold_rows.append(
                {
                    "fold": fold.name,
                    "horizon_days": horizon,
                    "strategy": request.strategy,
                    "train_origins": train.rows,
                    "train_start": train.origins.min().strftime("%Y-%m-%d"),
                    "train_end": train.origins.max().strftime("%Y-%m-%d"),
                    "spacing_days": spacing,
                    "validation_days": len(fold.validation_origins),
                    "scored_origins": len(origins),
                    "validation_start": origins.min().strftime("%Y-%m-%d"),
                    "validation_end": origins.max().strftime("%Y-%m-%d"),
                    "independent_windows": round(
                        independent_window_count(len(origins), horizon, spacing), 2
                    ),
                }
            )
            logger.info(
                "%s h=%3dd: trained on %d rows, scored %d origins",
                fold.name,
                horizon,
                train.rows,
                len(origins),
            )

    if not metric_frames:
        raise WalkForwardError(
            "walk-forward produced no metrics; every horizon lacked usable folds"
        )

    return WalkForwardResult(
        strategy=request.strategy,
        model_version=model_identity,
        metrics=pd.concat(metric_frames, ignore_index=True),
        folds=pd.DataFrame(fold_rows),
        crossings=pd.DataFrame(crossing_rows),
        horizons=request.horizons,
    )


def request_model_version(
    request: TrainingRequest, config: AppConfig, *, params_name: str = ""
) -> str:
    """Identity a walk-forward model is recorded under.

    Deliberately *not* a trained model version: a walk-forward run produces one
    model per fold, none of which is the model that would be deployed. Recording
    fold models under a deployable-looking version would invite someone to
    promote one.
    """
    short = request.algorithm.split("_")[0]
    base = f"wf-{short}-{request.strategy}-{config.config_version}"
    return f"{base}-{params_name}" if params_name else base


def aggregate_across_folds(
    metrics: pd.DataFrame, metric_names: tuple[str, ...], *, regime: str = ALL_REGIMES
) -> pd.DataFrame:
    """Mean / spread / worst fold per (model, horizon, metric).

    The worst fold is carried alongside the mean because that is the number a
    promotion decision has to survive (VALIDATION_SPEC.md section 10).
    """
    subset = metrics[
        (metrics["regime"] == regime) & (metrics["metric_name"].isin(metric_names))
    ]
    if subset.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    grouped = subset.groupby(["horizon_days", "model_version", "metric_name"], sort=False)
    for (horizon, version, metric), group in grouped:
        stats = dispersion(group["metric_value"])
        values = group["metric_value"].dropna()
        rows.append(
            {
                "horizon_days": int(horizon),
                "model_version": version,
                "metric_name": metric,
                "folds": int(stats["folds"]),
                "mean": stats["mean"],
                "std": stats["std"],
                "worst": stats["worst"],
                "best": float(values.min()) if not values.empty else float("nan"),
                "low_power": bool(group["low_power"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["metric_name", "horizon_days", "model_version"]
    ).reset_index(drop=True)


def fold_consistency(
    metrics: pd.DataFrame,
    metric_name: str,
    model_version: str,
    reference_version: str,
    *,
    regime: str = ALL_REGIMES,
) -> pd.DataFrame:
    """How often the model beats a reference, fold by fold.

    VALIDATION_SPEC.md section 8 asks for "improvement consistency across
    folds", and section 10 makes it a promotion condition. A model that wins on
    average by winning enormously in one fold and losing in the rest is not a
    model anyone should deploy.
    """
    subset = metrics[
        (metrics["regime"] == regime) & (metrics["metric_name"] == metric_name)
    ]
    if subset.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (horizon, fold), group in subset.groupby(["horizon_days", "fold"], sort=True):
        indexed = group.set_index("model_version")["metric_value"]
        if model_version not in indexed or reference_version not in indexed:
            continue
        model_value = indexed[model_version]
        reference_value = indexed[reference_version]
        if pd.isna(model_value) or pd.isna(reference_value) or reference_value == 0:
            continue
        rows.append(
            {
                "horizon_days": int(horizon),
                "fold": fold,
                "model": float(model_value),
                "reference": float(reference_value),
                "improvement": 1.0 - float(model_value) / float(reference_value),
                "beats_reference": bool(model_value < reference_value),
            }
        )
    return pd.DataFrame(rows)


def consistency_summary(consistency: pd.DataFrame) -> pd.DataFrame:
    """Per-horizon win rate and mean improvement over the reference."""
    if consistency.empty:
        return pd.DataFrame()
    grouped = consistency.groupby("horizon_days", sort=True)
    return pd.DataFrame(
        {
            "folds": grouped.size(),
            "folds_won": grouped["beats_reference"].sum(),
            "win_rate": grouped["beats_reference"].mean(),
            "mean_improvement": grouped["improvement"].mean(),
            "worst_improvement": grouped["improvement"].min(),
        }
    ).reset_index()
