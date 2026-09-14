"""Baseline evaluation run.

Assembles the pieces -- purged origins, naive forecasts, realised targets, the
metric tables -- into one reproducible run. Phase 5 reuses the same shape for
tree models, which is why the origin selection and scoring live here rather
than inside the job script.

Two rules this module enforces
------------------------------
**The outer test is not touched.** Baselines are scored on the inner block only.
They have no fitted parameters that could overfit a test set, but looking at
outer-test numbers now would still tell *the developer* what that block looks
like, and CLAUDE.md section 2.2 forbids exactly that kind of repeated
inspection. The baseline's outer-test numbers get computed once, alongside the
final model's single evaluation.

**Every baseline is scored on identical origins.** They warm up at different
speeds, so without an explicit intersection the comparison would silently score
different time periods and attribute the difference to model quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.evaluation.evaluator import ForecastSample, evaluate_sample_table
from src.models.baselines import (
    REFERENCE_BASELINE,
    Baseline,
    BaselinePrediction,
    build_baselines,
    common_valid_origins,
)
from src.models.targets import build_targets, target_column
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.validation.splits import (
    SplitBoundaries,
    assert_no_test_contamination,
    eligible_training_origins,
)

logger = get_logger(__name__)

SCOPE_VALIDATION: str = "validation"
BASELINE_SCOPE: str = "baseline"


def model_version(name: str, config: AppConfig) -> str:
    """Stable identity for a baseline in `performance_metrics`."""
    return f"baseline-{name}-{config.config_version}"


def origin_spacing_days(horizon_days: int, cap: int) -> int:
    """Spacing between evaluation origins for a horizon.

    VALIDATION_SPEC.md section 3 requires origins to be spaced "to avoid
    excessive overlap" with a configurable default of 30 days. Spacing by the
    horizon itself is what actually removes overlap: at ``h`` days apart the
    target windows are exactly disjoint. The configured value is therefore used
    as a **cap**, not a flat rule --

    * ``h <= cap``: spacing ``h``, giving genuinely independent observations.
    * ``h > cap``:  spacing ``cap``, accepting overlap because spacing a
      365-day horizon by 365 days would leave a handful of origins and no
      measurable metric at all. The overlap is then reported honestly through
      ``independent_windows``.

    A flat 30-day spacing would instead throw away 96% of the usable origins at
    ``h=1``, where consecutive origins barely overlap to begin with.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    if cap < 1:
        raise ValueError("origin spacing cap must be >= 1")
    return min(horizon_days, cap)


def space_origins(origins: pd.DatetimeIndex, spacing_days: int) -> pd.DatetimeIndex:
    """Keep origins at least ``spacing_days`` apart, walking forward from the first.

    Calendar-based rather than positional, so a gap in the origin index cannot
    quietly change the real spacing.
    """
    if spacing_days <= 1 or len(origins) == 0:
        return origins
    ordered = origins.sort_values()
    kept = [ordered[0]]
    step = pd.Timedelta(days=spacing_days)
    for origin in ordered[1:]:
        if origin - kept[-1] >= step:
            kept.append(origin)
    return pd.DatetimeIndex(kept, name=origins.name)


@dataclass(frozen=True)
class BaselineEvaluation:
    """Metric tables plus the provenance needed to reproduce the run."""

    scope: str
    split_version: str
    horizons: tuple[int, ...]
    metrics: pd.DataFrame
    origin_counts: pd.DataFrame
    baselines: tuple[str, ...]
    data_start: str
    data_end: str

    def describe(self) -> str:
        return (
            f"scope={self.scope} split={self.split_version} "
            f"baselines={list(self.baselines)} horizons={len(self.horizons)} "
            f"rows={len(self.metrics)}"
        )


def _predictions_for_horizon(
    baselines: list[Baseline],
    close: pd.Series,
    horizon_days: int,
    levels: tuple[float, ...],
) -> dict[str, BaselinePrediction]:
    return {
        baseline.name: baseline.predict(close, horizon_days, levels)
        for baseline in baselines
    }


def evaluate_baselines(
    close: pd.Series,
    config: AppConfig,
    boundaries: SplitBoundaries,
    *,
    horizons: tuple[int, ...],
    candidate_origins: pd.DatetimeIndex,
    regimes: pd.DataFrame | None = None,
    scope: str = SCOPE_VALIDATION,
) -> BaselineEvaluation:
    """Score every enabled baseline over ``horizons`` on the inner block."""
    if scope != SCOPE_VALIDATION:
        raise ValueError(
            f"baselines are only evaluated on the inner block; got scope={scope!r}. "
            "Outer-test baseline numbers are produced once, with the final model."
        )

    validation_section = config.section("validation")
    spacing_cap = int(validation_section.get("validation_origin_spacing_days", 30))
    regime_labels = tuple(str(r) for r in validation_section.get("regimes", []))
    levels = config.forecast.quantiles
    intervals = config.forecast.show_intervals

    baselines = build_baselines(config.baselines)
    targets = build_targets(close, horizons)

    metric_frames: list[pd.DataFrame] = []
    counts: list[dict[str, Any]] = []

    for horizon in horizons:
        spacing = origin_spacing_days(horizon, spacing_cap)
        purged = eligible_training_origins(candidate_origins, boundaries, horizon)
        spaced = space_origins(purged, spacing)
        assert_no_test_contamination(spaced, boundaries, horizon)

        predictions = _predictions_for_horizon(baselines, close, horizon, levels)
        shared = common_valid_origins(list(predictions.values()), spaced)
        actual = targets[target_column(horizon)].reindex(shared)
        # An origin whose target date has not arrived stays unscored rather than
        # being imputed; at this scope the purge already excludes most of them.
        scoreable = actual.dropna().index

        counts.append(
            {
                "horizon_days": horizon,
                "spacing_days": spacing,
                "purged_origins": len(purged),
                "spaced_origins": len(spaced),
                "scoreable_origins": len(scoreable),
                "low_power": boundaries.is_low_power(horizon),
            }
        )
        if len(scoreable) == 0:
            logger.warning("horizon %dd has no scoreable origins; skipped", horizon)
            continue

        reference = predictions[REFERENCE_BASELINE].restrict(scoreable).median
        for name, prediction in predictions.items():
            restricted = prediction.restrict(scoreable)
            sample = ForecastSample(
                model_version=model_version(name, config),
                horizon_days=horizon,
                origin_close=close.reindex(scoreable),
                actual_log_return=actual.reindex(scoreable),
                predicted_quantiles=restricted.quantiles,
                origin_spacing_days=spacing,
                regimes=None if regimes is None else regimes.reindex(scoreable),
            )
            metric_frames.append(
                evaluate_sample_table(
                    sample,
                    intervals=intervals,
                    reference_median=reference,
                    regimes=regime_labels,
                    low_power=boundaries.is_low_power(horizon),
                )
            )

    metrics = (
        pd.concat(metric_frames, ignore_index=True)
        if metric_frames
        else pd.DataFrame(
            columns=[
                "model_version",
                "horizon_days",
                "regime",
                "metric_name",
                "metric_value",
                "sample_size",
                "period_start",
                "period_end",
                "low_power",
            ]
        )
    )
    return BaselineEvaluation(
        scope=scope,
        split_version=boundaries.split_version,
        horizons=tuple(horizons),
        metrics=metrics,
        origin_counts=pd.DataFrame(counts),
        baselines=tuple(b.name for b in baselines),
        data_start=close.index.min().strftime("%Y-%m-%d"),
        data_end=close.index.max().strftime("%Y-%m-%d"),
    )
