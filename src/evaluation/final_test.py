"""The single outer-test evaluation of a frozen design.

VALIDATION_SPEC.md section 4 describes a five-step lifecycle, and step 4 --
"evaluate once on outer test" -- had no implementation until the weekly review
needed it. Everything in the repository protected the outer test; nothing spent
it, so the reserved block sat unused and the production training cutoff could
never move past it.

Why this is the gate on everything downstream
---------------------------------------------
A model that trains through today cannot be evaluated on a block it has already
seen. So there are two models in the lifecycle, and their order is fixed:

1. the **tested model**, trained on eligible pre-test data
   (`origin + horizon + embargo < outer_test_start`), evaluated exactly once
   here. This measures the *design*.
2. the **served model**, trained through the newest resolved label. It can never
   be tested, and it does not need to be: the design it instantiates was
   measured by (1), and from then on the honest record is the logged production
   forecasts (section 4.4).

Running (2) before (1) destroys the only independent verdict this dataset can
produce, silently. So the weekly review refuses to release the purge until a
recorded outer-test evaluation exists for the same configuration.

What "once" is enforced by
--------------------------
`registry.record_metrics` refuses a second outer-test write for a model version
(section 4.4). This module does the measuring; the registry is what makes it
unrepeatable. A design changed after seeing these numbers is a new design and
needs its own reserved block, which this dataset does not have -- that cost is
the reason the discipline exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.evaluation.baseline_eval import model_version as baseline_version
from src.evaluation.baseline_eval import origin_spacing_days, space_origins
from src.evaluation.evaluator import ForecastSample, evaluate_sample_table
from src.models.baselines import REFERENCE_BASELINE, build_baselines, common_valid_origins
from src.models.forecaster import MultiHorizonForecaster
from src.models.targets import forward_log_return
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.validation.splits import (
    SplitBoundaries,
    independent_window_count,
    outer_test_origins,
)

logger = get_logger(__name__)


class FinalTestError(RuntimeError):
    """Raised when the outer test cannot be evaluated honestly."""


@dataclass
class FinalTestResult:
    """Outer-test metrics for one model version, alongside the baselines."""

    model_version: str
    metrics: pd.DataFrame
    coverage: pd.DataFrame
    test_start: str
    test_end: str
    horizons: tuple[int, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"{self.model_version}: {len(self.horizons)} horizons over "
            f"{self.test_start}..{self.test_end}, {len(self.metrics)} metric rows"
        )


def assert_testable(
    forecaster: MultiHorizonForecaster, boundaries: SplitBoundaries
) -> None:
    """Refuse to test a model whose training labels reach into the test block.

    The purge is applied at training time, so this should never fire. It stays
    because the cost of it being redundant is nothing and the cost of it being
    necessary is a final evaluation that means the opposite of what it says.
    """
    cutoff = pd.Timestamp(forecaster.metadata.training_cutoff)
    shortest = min(forecaster.horizons)
    latest_label = cutoff + pd.Timedelta(days=shortest + boundaries.embargo_days)
    if latest_label >= boundaries.outer_test_start:
        raise FinalTestError(
            f"{forecaster.metadata.model_version} trained through {cutoff:%Y-%m-%d}; "
            f"its shortest-horizon label reaches {latest_label:%Y-%m-%d}, inside the "
            f"outer test starting {boundaries.outer_test_start:%Y-%m-%d}. This model "
            "has already seen the answers and cannot be evaluated on them."
        )


def evaluate_final_test(
    forecaster: MultiHorizonForecaster,
    features: pd.DataFrame,
    close: pd.Series,
    boundaries: SplitBoundaries,
    config: AppConfig,
    *,
    horizons: tuple[int, ...] | None = None,
    regimes: pd.DataFrame | None = None,
) -> FinalTestResult:
    """Score the model and every baseline on the outer test, once.

    The model and the baselines see identical origins, spaced by
    `min(horizon, cap)` exactly as inner validation spaces them
    (VALIDATION_SPEC.md section 3.1), so the two sets of numbers are comparable
    to each other and to the inner-block results.
    """
    assert_testable(forecaster, boundaries)

    validation_section = config.section("validation")
    spacing_cap = int(validation_section.get("validation_origin_spacing_days", 30))
    regime_labels = tuple(str(label) for label in validation_section.get("regimes", []))
    intervals = config.forecast.show_intervals
    levels = forecaster.levels
    wanted = tuple(horizons or forecaster.horizons)

    candidates = features.dropna(axis=0, how="any").index
    data_end = close.index.max()
    baselines = build_baselines(config.baselines)

    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    scored_horizons: list[int] = []
    spans: list[pd.Timestamp] = []

    for horizon in wanted:
        if horizon not in forecaster.horizons:
            logger.warning("h=%dd: the model has no such horizon; skipped", horizon)
            continue
        spacing = origin_spacing_days(horizon, spacing_cap)
        block = outer_test_origins(
            candidates,
            boundaries,
            data_end=data_end,
            horizon_days=horizon,
            fully_scoreable_only=True,
        )
        origins = space_origins(block, spacing)
        target = forward_log_return(close, horizon)
        origins = origins.intersection(target.dropna().index).sort_values()
        predictions = {
            baseline.name: baseline.predict(close, horizon, levels)
            for baseline in baselines
        }
        if origins.size:
            origins = common_valid_origins(list(predictions.values()), origins)
        if origins.empty:
            logger.warning("h=%dd: no scoreable outer-test origins", horizon)
            continue

        predicted, _ = forecaster.predict_horizon(features.loc[origins], horizon)
        reference = predictions[REFERENCE_BASELINE].restrict(origins).median
        samples: list[tuple[str, pd.DataFrame]] = [
            (forecaster.metadata.model_version, predicted)
        ]
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
            table["fold"] = "outer_test"
            frames.append(table)

        scored_horizons.append(horizon)
        spans.extend([origins.min(), origins.max()])
        coverage_rows.append(
            {
                "horizon_days": horizon,
                "spacing_days": spacing,
                "block_origins": len(block),
                "scored_origins": len(origins),
                "independent_windows": round(
                    independent_window_count(len(origins), horizon, spacing), 2
                ),
                "test_start": origins.min().strftime("%Y-%m-%d"),
                "test_end": origins.max().strftime("%Y-%m-%d"),
                "low_power": boundaries.is_low_power(horizon),
            }
        )
        logger.info(
            "h=%3dd: scored %d outer-test origins (%.1f independent windows)",
            horizon,
            len(origins),
            independent_window_count(len(origins), horizon, spacing),
        )

    if not frames:
        raise FinalTestError(
            "no horizon had any scoreable outer-test origin; there is nothing to "
            "evaluate"
        )

    return FinalTestResult(
        model_version=forecaster.metadata.model_version,
        metrics=pd.concat(frames, ignore_index=True),
        coverage=pd.DataFrame(coverage_rows),
        test_start=min(spans).strftime("%Y-%m-%d"),
        test_end=max(spans).strftime("%Y-%m-%d"),
        horizons=tuple(scored_horizons),
        notes={
            "split_version": boundaries.split_version,
            "outer_test_start": boundaries.outer_test_start.strftime("%Y-%m-%d"),
            "training_cutoff": forecaster.metadata.training_cutoff,
        },
    )


def test_record(
    result: FinalTestResult, config: AppConfig, *, metric: str = "pinball_mean"
) -> dict[str, Any]:
    """The compact summary written to `model_registry.test_metrics`.

    Recorded with the exact window it was measured on, because section 4.4
    requires the evaluation and its test block to be stored together: a metric
    without its window cannot be checked against a later claim.
    """
    reference = baseline_version(REFERENCE_BASELINE, config)
    overall = result.metrics[result.metrics["regime"] == "all"]
    model = overall[overall["model_version"] == result.model_version]
    baseline = overall[overall["model_version"] == reference]

    def by_horizon(frame: pd.DataFrame, name: str) -> dict[str, float]:
        subset = frame[frame["metric_name"] == name]
        return {
            str(int(row.horizon_days)): round(float(row.metric_value), 10)
            for row in subset.itertuples(index=False)
            if row.metric_value is not None and not pd.isna(row.metric_value)
        }

    return {
        "split_version": result.notes.get("split_version"),
        "test_start": result.test_start,
        "test_end": result.test_end,
        "evaluated_horizons": list(result.horizons),
        "metric": metric,
        "model": by_horizon(model, metric),
        "reference_baseline": reference,
        "baseline": by_horizon(baseline, metric),
        "coverage_95": by_horizon(model, "coverage_95"),
        "sample_size": by_horizon(model, "sample_size"),
    }


def summarise(
    result: FinalTestResult, config: AppConfig, *, metric: str = "pinball_mean"
) -> pd.DataFrame:
    """Model against the reference baseline per horizon, for the report."""
    reference = baseline_version(REFERENCE_BASELINE, config)
    overall = result.metrics[
        (result.metrics["regime"] == "all")
        & (result.metrics["metric_name"].isin({metric, "sample_size", "coverage_95"}))
    ]
    pivoted = overall.pivot_table(
        index="horizon_days",
        columns=["model_version", "metric_name"],
        values="metric_value",
        aggfunc="mean",
    )
    rows: list[dict[str, Any]] = []
    for horizon, row in pivoted.iterrows():
        model_value = row.get((result.model_version, metric))
        baseline_value = row.get((reference, metric))
        if pd.isna(model_value) or pd.isna(baseline_value) or not baseline_value:
            continue
        rows.append(
            {
                "horizon_days": int(horizon),
                "sample_size": int(row.get((result.model_version, "sample_size"), 0)),
                "model": float(model_value),
                "baseline": float(baseline_value),
                "improvement": 1.0 - float(model_value) / float(baseline_value),
                "beats_baseline": bool(model_value < baseline_value),
                "coverage_95": float(row.get((result.model_version, "coverage_95"), float("nan"))),
            }
        )
    return pd.DataFrame(rows).sort_values("horizon_days").reset_index(drop=True)


__all__ = [
    "FinalTestError",
    "FinalTestResult",
    "assert_testable",
    "evaluate_final_test",
    "summarise",
    "test_record",
]
