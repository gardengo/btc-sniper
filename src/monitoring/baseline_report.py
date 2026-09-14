"""Markdown report for a baseline evaluation run.

The point of this report is to establish the bar. Every number here is what a
tree model has to beat in Phase 4-5, and the report is deliberately blunt about
where the data cannot support a verdict at all.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.baseline_eval import BaselineEvaluation
from src.evaluation.evaluator import ALL_REGIMES, pivot_metrics
from src.models.baselines import REFERENCE_BASELINE
from src.monitoring.markdown import markdown_table
from src.utils.timeutils import utc_now_iso

HEADLINE_METRICS: tuple[str, ...] = (
    "sample_size",
    "independent_windows",
    "return_mae",
    "return_bias",
    "mase",
    "pinball_mean",
)
INTERVAL_METRICS: tuple[str, ...] = (
    "coverage_50",
    "coverage_80",
    "coverage_95",
    "rel_width_95",
    "interval_score_95",
)
COUNT_FORMATS: dict[str, str] = {
    "sample_size": "{:,.0f}",
    "direction_calls": "{:,.0f}",
    "quantile_crossings": "{:,.0f}",
    "independent_windows": "{:,.1f}",
    "horizon_days": "{:,.0f}",
}
DIRECTION_METRICS: tuple[str, ...] = (
    "direction_calls",
    "direction_accuracy",
    "direction_accuracy_up",
    "direction_accuracy_down",
)


def _short_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip the `baseline-` prefix so tables stay narrow."""
    if frame.empty or "model_version" not in frame.columns:
        return frame
    trimmed = frame.copy()
    trimmed["model_version"] = trimmed["model_version"].str.replace(
        "^baseline-", "", regex=True
    )
    return trimmed.rename(columns={"model_version": "baseline"})


def _focus(evaluation: BaselineEvaluation, horizons: tuple[int, ...]) -> pd.DataFrame:
    return evaluation.metrics[evaluation.metrics["horizon_days"].isin(horizons)]


def best_per_horizon(
    evaluation: BaselineEvaluation, metric_name: str, horizons: tuple[int, ...]
) -> pd.DataFrame:
    """Which baseline wins each horizon on one metric, and by how much."""
    subset = evaluation.metrics[
        (evaluation.metrics["regime"] == ALL_REGIMES)
        & (evaluation.metrics["metric_name"] == metric_name)
        & (evaluation.metrics["horizon_days"].isin(horizons))
        & evaluation.metrics["metric_value"].notna()
    ]
    if subset.empty:
        return pd.DataFrame()

    rows = []
    for horizon, group in subset.groupby("horizon_days", sort=True):
        ordered = group.sort_values("metric_value")
        winner = ordered.iloc[0]
        reference = group[group["model_version"].str.contains(REFERENCE_BASELINE)]
        reference_value = (
            float(reference["metric_value"].iloc[0]) if not reference.empty else float("nan")
        )
        best_value = float(winner["metric_value"])
        rows.append(
            {
                "horizon_days": int(horizon),
                "best_baseline": str(winner["model_version"]).replace("baseline-", ""),
                metric_name: best_value,
                f"{REFERENCE_BASELINE}_{metric_name}": reference_value,
                "improvement_vs_no_change": (
                    float("nan")
                    if not reference_value
                    else 1.0 - best_value / reference_value
                ),
                "low_power": bool(winner.get("low_power", False)),
            }
        )
    return pd.DataFrame(rows)


def render_markdown(
    evaluation: BaselineEvaluation, *, focus_horizons: tuple[int, ...]
) -> str:
    lines: list[str] = []
    add = lines.append

    add("# BTC Sniper - Baseline Evaluation")
    add("")
    add(f"- generated: `{utc_now_iso()}`")
    add(f"- scope: `{evaluation.scope}` (inner block only; the outer test is untouched)")
    add(f"- split_version: `{evaluation.split_version}`")
    add(f"- data range: `{evaluation.data_start}` .. `{evaluation.data_end}`")
    add(f"- baselines: {', '.join(f'`{name}`' for name in evaluation.baselines)}")
    add(f"- horizons evaluated: {len(evaluation.horizons)}")
    add("")
    add(
        "All baselines share one spread model and differ only in the median they "
        "predict, so differences below are differences in the trend claim, not in "
        "two different interval recipes."
    )
    add("")

    add("## 1. Origin accounting")
    add("")
    add(
        "`purged` applies `origin + horizon < outer_test_start`; `spaced` then thins "
        "origins to `min(horizon, 30)` days apart so target windows overlap as little "
        "as the data allows. `independent_windows` in later tables is the honest "
        "effective sample size."
    )
    add("")
    add(markdown_table(evaluation.origin_counts, "{:,.0f}"))

    add("## 2. Headline comparison")
    add("")
    add(
        "`mase` is the ratio of a baseline's mean absolute log-return error to the "
        f"`{REFERENCE_BASELINE}` baseline's on the same origins: below 1.0 beats it. "
        "`pinball_mean` scores the whole predictive distribution (lower is better)."
    )
    add("")
    headline = pivot_metrics(_focus(evaluation, focus_horizons), HEADLINE_METRICS)
    add(markdown_table(_short_names(headline), column_formats=COUNT_FORMATS))

    add("## 3. Interval calibration")
    add("")
    add(
        "Coverage should sit near its nominal level (0.50 / 0.80 / 0.95). "
        "`rel_width_95` is the 95% band as a fraction of the origin price, and "
        "`interval_score_95` combines width and misses so neither can be gamed."
    )
    add("")
    intervals = pivot_metrics(_focus(evaluation, focus_horizons), INTERVAL_METRICS)
    add(markdown_table(_short_names(intervals), column_formats=COUNT_FORMATS))

    add("## 4. Directional accuracy")
    add("")
    add(
        f"`{REFERENCE_BASELINE}` predicts a median of exactly zero, so it makes no "
        "directional call at all: it shows `0` calls and a blank accuracy rather "
        "than a misleading 0%."
    )
    add("")
    direction = pivot_metrics(_focus(evaluation, focus_horizons), DIRECTION_METRICS)
    add(markdown_table(_short_names(direction), column_formats=COUNT_FORMATS))

    add("## 5. Best baseline per horizon")
    add("")
    for metric in ("return_mae", "pinball_mean"):
        add(f"### by `{metric}`")
        add("")
        add(
            markdown_table(
                best_per_horizon(evaluation, metric, focus_horizons),
                column_formats=COUNT_FORMATS,
            )
        )
    add(
        "A `low_power` horizon carries too few independent windows for these "
        "differences to be meaningful (VALIDATION_SPEC.md section 4.3); it is "
        "reported, not acted on."
    )
    add("")

    add("## 6. Regime breakdown")
    add("")
    add("Regime is the label **at the forecast origin**, which is what is knowable then.")
    add("")
    regime_rows = evaluation.metrics[
        (evaluation.metrics["regime"] != ALL_REGIMES)
        & (evaluation.metrics["metric_name"] == "return_mae")
        & (evaluation.metrics["horizon_days"].isin(focus_horizons))
    ]
    if regime_rows.empty:
        add("_(no regime-tagged origins)_")
        add("")
    else:
        pivoted = regime_rows.pivot_table(
            index=["horizon_days", "model_version"],
            columns="regime",
            values="metric_value",
            sort=False,
        ).reset_index()
        add(markdown_table(_short_names(pivoted), column_formats=COUNT_FORMATS))

    add("## 7. What this establishes")
    add("")
    add(
        "These numbers are the bar for Phase 4. A tree model that does not beat the "
        f"`{REFERENCE_BASELINE}` baseline on `pinball_mean` at the horizons with real "
        "statistical power is not an improvement, however good its headline MAE looks."
    )
    add("")
    return "\n".join(lines)


def write_report(
    evaluation: BaselineEvaluation, path: Path, *, focus_horizons: tuple[int, ...]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_markdown(evaluation, focus_horizons=focus_horizons), encoding="utf-8"
    )
    return path
