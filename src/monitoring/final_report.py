"""Markdown report for the single outer-test evaluation.

Written once per design and never regenerated with different numbers, so it is
also the permanent record of what the design was worth. It states the test
window, the sample size behind every number, and the fact that the block is now
spent -- a reader who wants a better number cannot get one from this dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.evaluator import ALL_REGIMES
from src.evaluation.final_test import FinalTestResult, summarise
from src.models.baselines import REFERENCE_BASELINE
from src.monitoring.markdown import markdown_table
from src.utils.config import AppConfig
from src.utils.provenance import describe_environment
from src.utils.timeutils import utc_now_iso

PRIMARY_METRIC: str = "pinball_mean"
COUNT_FORMATS: dict[str, str] = {
    "horizon_days": "{:,.0f}",
    "sample_size": "{:,.0f}",
    "scored_origins": "{:,.0f}",
    "block_origins": "{:,.0f}",
    "spacing_days": "{:,.0f}",
}


def _short(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "model_version" not in frame.columns:
        return frame
    trimmed = frame.copy()
    trimmed["model_version"] = trimmed["model_version"].str.replace(
        "^baseline-", "", regex=True
    )
    return trimmed.rename(columns={"model_version": "model"})


def render_markdown(result: FinalTestResult, config: AppConfig) -> str:
    lines: list[str] = []
    add = lines.append

    add("# BTC Sniper - Final Outer-Test Evaluation")
    add("")
    add(f"- generated: `{utc_now_iso()}`")
    add(f"- environment: `{describe_environment()}`")
    add(f"- model: `{result.model_version}`")
    add(f"- training cutoff: `{result.notes.get('training_cutoff')}`")
    add(f"- split: `{result.notes.get('split_version')}`")
    add(f"- outer test block: `{result.test_start}` .. `{result.test_end}`")
    add("")
    add(
        "**This is the one evaluation this design gets.** "
        "`registry.record_metrics` refuses a second outer-test write for this "
        "model version (VALIDATION_SPEC.md section 4.4). A design changed in "
        "response to these numbers is a new design and needs its own reserved "
        "block, which this dataset does not have. Nothing below may be used to "
        "select a model, a feature or a hyperparameter."
    )
    add("")

    add("## 1. Model vs baseline")
    add("")
    add(
        f"Judged on `{PRIMARY_METRIC}` against `{REFERENCE_BASELINE}` on identical "
        "origins, the same comparison inner validation made."
    )
    add("")
    add(markdown_table(summarise(result, config), column_formats=COUNT_FORMATS))

    add("## 2. How much evidence each horizon carries")
    add("")
    add(
        "Origins are spaced by `min(horizon, cap)` so target windows overlap as "
        "little as the data allows; `independent_windows` reports what is left "
        "after accounting for the overlap that remains "
        "(VALIDATION_SPEC.md section 3.2)."
    )
    add("")
    add(markdown_table(result.coverage, column_formats=COUNT_FORMATS))

    add("## 3. Full metric table")
    add("")
    overall = result.metrics[result.metrics["regime"] == ALL_REGIMES]
    wide = overall.pivot_table(
        index=["horizon_days", "model_version"],
        columns="metric_name",
        values="metric_value",
        aggfunc="mean",
    ).reset_index()
    wanted = [
        "horizon_days",
        "model_version",
        "sample_size",
        "independent_windows",
        "return_mae",
        PRIMARY_METRIC,
        "coverage_95",
        "interval_score_95",
        "mase",
    ]
    columns = [name for name in wanted if name in wide.columns]
    add(markdown_table(_short(wide[columns]), column_formats=COUNT_FORMATS))

    add("## 4. By regime")
    add("")
    add("Regime is the label at the forecast origin.")
    add("")
    regime_rows = result.metrics[
        (result.metrics["regime"] != ALL_REGIMES)
        & (result.metrics["metric_name"] == PRIMARY_METRIC)
    ]
    if regime_rows.empty:
        add("_(no regime-tagged outer-test origins)_")
        add("")
    else:
        pivoted = regime_rows.pivot_table(
            index=["horizon_days", "model_version"],
            columns="regime",
            values="metric_value",
            aggfunc="mean",
        ).reset_index()
        add(markdown_table(_short(pivoted), column_formats=COUNT_FORMATS))

    add("## 5. What happens next")
    add("")
    add(
        "The design has now been measured. From here the production model is "
        "trained through the newest resolved label rather than stopping at the "
        "purge boundary: it can never be tested, and it no longer needs to be. "
        "Ongoing honest measurement comes from logged production forecasts "
        "(VALIDATION_SPEC.md section 4.4), which is what "
        "`jobs.evaluate_forecasts` accumulates."
    )
    return "\n".join(lines)


def write_report(result: FinalTestResult, config: AppConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(result, config), encoding="utf-8")
    return path


__all__ = ["render_markdown", "write_report"]
