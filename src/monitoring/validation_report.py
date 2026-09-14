"""Markdown report for a walk-forward validation run.

Written to be read by someone deciding whether a model is worth anything. That
means the baseline comparison comes before the model's own numbers, the worst
fold is never hidden behind a mean, and low-power horizons are marked everywhere
they appear rather than in a footnote.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.baseline_eval import model_version as baseline_version
from src.evaluation.evaluator import ALL_REGIMES, pivot_metrics
from src.evaluation.walk_forward import (
    WalkForwardResult,
    aggregate_across_folds,
    consistency_summary,
    fold_consistency,
)
from src.models.baselines import REFERENCE_BASELINE
from src.monitoring.markdown import markdown_table
from src.utils.config import AppConfig
from src.utils.provenance import describe_environment
from src.utils.timeutils import utc_now_iso

PRIMARY_METRIC: str = "pinball_mean"
HEADLINE_METRICS: tuple[str, ...] = (
    "return_mae",
    "pinball_mean",
    "coverage_95",
    "interval_score_95",
)
COUNT_FORMATS: dict[str, str] = {
    "horizon_days": "{:,.0f}",
    "folds": "{:,.0f}",
    "folds_won": "{:,.0f}",
    "rows": "{:,.0f}",
    "crossing_pairs": "{:,.0f}",
    "crossing_rows": "{:,.0f}",
    "sample_size": "{:,.0f}",
    "train_origins": "{:,.0f}",
    "validation_origins": "{:,.0f}",
}


def _short(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "model_version" not in frame.columns:
        return frame
    trimmed = frame.copy()
    trimmed["model_version"] = (
        trimmed["model_version"]
        .str.replace("^baseline-", "", regex=True)
        .str.replace("^wf-", "", regex=True)
    )
    return trimmed.rename(columns={"model_version": "model"})


def verdict(
    result: WalkForwardResult, config: AppConfig, *, metric: str = PRIMARY_METRIC
) -> pd.DataFrame:
    """Per-horizon verdict against the reference baseline."""
    reference = baseline_version(REFERENCE_BASELINE, config)
    consistency = fold_consistency(
        result.metrics, metric, result.model_version, reference
    )
    summary = consistency_summary(consistency)
    if summary.empty:
        return summary
    low_power = (
        result.metrics[["horizon_days", "low_power"]]
        .drop_duplicates("horizon_days")
        .set_index("horizon_days")["low_power"]
    )
    summary["low_power"] = summary["horizon_days"].map(low_power).fillna(False)
    summary["verdict"] = [
        _label(row) for row in summary.itertuples(index=False)
    ]
    return summary


def _label(row) -> str:
    """A blunt verdict, so a marginal result cannot read as a win."""
    if row.folds == 0:
        return "no evidence"
    if bool(getattr(row, "low_power", False)):
        return "low power - not decisive"
    if row.win_rate == 1.0 and row.mean_improvement > 0.02:
        return "beats baseline in every fold"
    if row.win_rate >= 0.5 and row.mean_improvement > 0.0:
        return "mixed - not consistent"
    return "does not beat baseline"


def render_markdown(
    result: WalkForwardResult, config: AppConfig, *, focus_horizons: tuple[int, ...]
) -> str:
    reference = baseline_version(REFERENCE_BASELINE, config)
    lines: list[str] = []
    add = lines.append

    add("# BTC Sniper - Walk-Forward Validation")
    add("")
    add(f"- generated: `{utc_now_iso()}`")
    add(f"- environment: `{describe_environment()}`")
    add(f"- training window: `{result.strategy}`")
    add(f"- model identity: `{result.model_version}`")
    add(f"- reference baseline: `{REFERENCE_BASELINE}`")
    add(f"- horizons: {len(result.horizons)}")
    add("")
    add(
        "Inner block only. The outer test is not touched by anything in this "
        "report (VALIDATION_SPEC.md section 4). Every fold refits from scratch, "
        "and the model and the baselines are scored on identical origins."
    )
    add("")

    add("## 1. Verdict")
    add("")
    add(
        f"Judged on `{PRIMARY_METRIC}`, which scores the whole predictive "
        "distribution. `win_rate` is the fraction of folds where the model beat "
        f"`{REFERENCE_BASELINE}`; `worst_improvement` is the fold where it did "
        "worst. A model that wins on average by winning hugely once and losing "
        "the rest is not deployable."
    )
    add("")
    add(markdown_table(verdict(result, config), column_formats=COUNT_FORMATS))

    add("## 2. Fold layout")
    add("")
    add(markdown_table(result.folds, column_formats=COUNT_FORMATS))

    add("## 3. Model vs baselines, averaged across folds")
    add("")
    aggregated = aggregate_across_folds(result.metrics, HEADLINE_METRICS)
    for metric in HEADLINE_METRICS:
        subset = aggregated[aggregated["metric_name"] == metric]
        if subset.empty:
            continue
        add(f"### `{metric}`")
        add("")
        add(
            markdown_table(
                _short(subset.drop(columns=["metric_name"])),
                column_formats=COUNT_FORMATS,
            )
        )
    add(
        "`worst` is the worst fold, not an outlier to be discarded. "
        "For coverage metrics a *higher* number is better, so read `worst` there "
        "as the extreme fold rather than the poorest one."
    )
    add("")

    add("## 4. Per-fold detail")
    add("")
    consistency = fold_consistency(
        result.metrics, PRIMARY_METRIC, result.model_version, reference
    )
    add(markdown_table(consistency, column_formats=COUNT_FORMATS))

    add("## 5. Quantile crossing")
    add("")
    add(
        "Quantiles are fitted independently and can cross; crossings are repaired "
        "by sorting and counted here (MODEL_SPEC.md section 3). A high rate means "
        "the quantile fits disagree about the same input."
    )
    add("")
    add(markdown_table(result.crossings, column_formats=COUNT_FORMATS))

    add("## 6. Regime breakdown")
    add("")
    add("Regime is the label at the forecast origin.")
    add("")
    regime_rows = result.metrics[
        (result.metrics["regime"] != ALL_REGIMES)
        & (result.metrics["metric_name"] == PRIMARY_METRIC)
        & (result.metrics["horizon_days"].isin(focus_horizons))
    ]
    if regime_rows.empty:
        add("_(no regime-tagged validation origins)_")
        add("")
    else:
        pivoted = (
            regime_rows.pivot_table(
                index=["horizon_days", "model_version"],
                columns="regime",
                values="metric_value",
                aggfunc="mean",
                sort=False,
            )
            .reset_index()
        )
        add(markdown_table(_short(pivoted), column_formats=COUNT_FORMATS))

    add("## 7. Raw metric table (overall regime)")
    add("")
    wide = pivot_metrics(
        result.metrics[result.metrics["horizon_days"].isin(focus_horizons)],
        ("sample_size", "independent_windows", *HEADLINE_METRICS, "mase"),
    )
    add(markdown_table(_short(wide), column_formats=COUNT_FORMATS))
    return "\n".join(lines)


def render_comparison(
    results: dict[str, WalkForwardResult],
    config: AppConfig,
    *,
    title: str = "Training Window Comparison",
    preamble: str = (
        "MODEL_SPEC.md section 7 forbids picking a window a priori from the "
        "four-year cycle. The window is chosen here, on inner validation, or not "
        "at all."
    ),
) -> str:
    """Side-by-side comparison of several walk-forward runs.

    Used for both the training-window comparison (VALIDATION_SPEC.md section 9)
    and the pre-declared hyperparameter comparison (section 5).
    """
    lines: list[str] = []
    add = lines.append

    add(f"# BTC Sniper - {title}")
    add("")
    add(f"- generated: `{utc_now_iso()}`")
    add(f"- environment: `{describe_environment()}`")
    add(f"- runs compared: {', '.join(f'`{name}`' for name in results)}")
    add("")
    add(preamble)
    add("")

    rows: list[pd.DataFrame] = []
    for strategy, result in results.items():
        aggregated = aggregate_across_folds(result.metrics, (PRIMARY_METRIC,))
        model_rows = aggregated[aggregated["model_version"] == result.model_version]
        if model_rows.empty:
            continue
        tagged = model_rows.copy()
        tagged["strategy"] = strategy
        rows.append(tagged)

    if not rows:
        add("_(no comparable results)_")
        return "\n".join(lines)

    combined = pd.concat(rows, ignore_index=True)
    add(f"## Mean `{PRIMARY_METRIC}` across folds (lower is better)")
    add("")
    pivoted = combined.pivot_table(
        index="horizon_days", columns="strategy", values="mean", sort=True
    ).reset_index()
    add(markdown_table(pivoted, column_formats=COUNT_FORMATS))

    for group in identical_runs(pivoted):
        add(
            f"> **{' and '.join(f'`{name}`' for name in group)} produced identical "
            "results.** They are not distinguishable on this dataset -- a rolling "
            "window longer than the available history is the expanding window. "
            "Reading a difference into these columns would be reading noise that "
            "is not even there."
        )
        add("")

    add("## Worst fold")
    add("")
    worst = combined.pivot_table(
        index="horizon_days", columns="strategy", values="worst", sort=True
    ).reset_index()
    add(markdown_table(worst, column_formats=COUNT_FORMATS))

    add("## Verdict per run")
    add("")
    verdicts = []
    for strategy, result in results.items():
        summary = verdict(result, config)
        if summary.empty:
            continue
        tagged = summary.copy()
        tagged.insert(0, "run", strategy)
        verdicts.append(tagged)
    if verdicts:
        add(markdown_table(pd.concat(verdicts, ignore_index=True), column_formats=COUNT_FORMATS))
    return "\n".join(lines)


def write_report(
    result: WalkForwardResult,
    config: AppConfig,
    path: Path,
    *,
    focus_horizons: tuple[int, ...],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_markdown(result, config, focus_horizons=focus_horizons), encoding="utf-8"
    )
    return path


def identical_runs(pivoted: pd.DataFrame) -> list[list[str]]:
    """Groups of runs whose columns are numerically identical.

    Four columns of the same numbers invite the conclusion that the choice was
    evaluated and did not matter, when in fact it was never evaluated at all.
    """
    columns = [c for c in pivoted.columns if c != "horizon_days"]
    groups: list[list[str]] = []
    seen: set[str] = set()
    for index, name in enumerate(columns):
        if name in seen:
            continue
        same = [name]
        for other in columns[index + 1 :]:
            if other in seen:
                continue
            if pivoted[name].equals(pivoted[other]):
                same.append(other)
                seen.add(other)
        if len(same) > 1:
            seen.add(name)
            groups.append(same)
    return groups


def write_comparison(
    results: dict[str, WalkForwardResult],
    config: AppConfig,
    path: Path,
    *,
    title: str = "Training Window Comparison",
    preamble: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"title": title}
    if preamble is not None:
        kwargs["preamble"] = preamble
    path.write_text(render_comparison(results, config, **kwargs), encoding="utf-8")
    return path
