"""Markdown report for production performance and drift.

This is the report the weekly review reads (OPERATING_SPEC.md section 2). It is
written to make two things impossible to miss: how much realized evidence
actually exists, and whether that evidence is enough to decide anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.monitoring.drift import LEVEL_ALERT, LEVEL_WARN, DriftThresholds
from src.monitoring.markdown import markdown_table
from src.monitoring.realization import STATUS_FULL, RealizationResult
from src.utils.provenance import describe_environment
from src.utils.timeutils import utc_now_iso

COUNT_FORMATS: dict[str, str] = {
    "horizon_days": "{:,.0f}",
    "sample_size": "{:,.0f}",
    "horizons": "{:,.0f}",
    "resolved": "{:,.0f}",
    "pending": "{:,.0f}",
    "rows": "{:,.0f}",
    "direction_calls": "{:,.0f}",
}
HEADLINE_METRICS: tuple[str, ...] = (
    "sample_size",
    "price_mae",
    "return_mae",
    "pinball_mean",
    "direction_accuracy",
    "coverage_95",
)


@dataclass(frozen=True)
class PerformanceContext:
    """Everything the renderer needs, gathered by the job."""

    realization: RealizationResult
    production: pd.DataFrame
    evidence: pd.DataFrame
    feature_drift: pd.DataFrame
    performance_drift: pd.DataFrame
    triggers: pd.DataFrame
    thresholds: DriftThresholds
    model_version: str | None
    training_cutoff: str | None
    excluded_in_sample: int = 0
    data_end: str = ""


def _headline(production: pd.DataFrame) -> pd.DataFrame:
    if production.empty:
        return pd.DataFrame()
    overall = production[
        (production["regime"] == "all")
        & (production["metric_name"].isin(HEADLINE_METRICS))
    ]
    if overall.empty:
        return pd.DataFrame()
    wide = overall.pivot_table(
        index="horizon_days", columns="metric_name", values="metric_value", sort=True
    ).reset_index()
    ordered = ["horizon_days"] + [m for m in HEADLINE_METRICS if m in wide.columns]
    return wide[ordered]


def render_markdown(context: PerformanceContext) -> str:
    lines: list[str] = []
    add = lines.append

    add("# BTC Sniper - Production Performance and Drift")
    add("")
    add(f"- generated: `{utc_now_iso()}`")
    add(f"- environment: `{describe_environment()}`")
    add(f"- model: `{context.model_version or 'none'}`")
    add(f"- training cutoff: `{context.training_cutoff or 'unknown'}`")
    add(f"- data through: `{context.data_end}`")
    add("")

    add("## 1. How much evidence exists")
    add("")
    add(f"- {context.realization.describe()}")
    if context.excluded_in_sample:
        add(
            f"- **{context.excluded_in_sample} rows excluded as in-sample.** Their "
            "forecast origin sits inside the model's own training data, so they "
            "are not out-of-sample evidence however good they look."
        )
    add("")
    if context.realization.resolved == 0:
        add(
            "> **No forecast has resolved yet.** Production metrics accumulate from "
            "the first daily run forward; a 30-day horizon needs 30 days before it "
            "says anything, and a 365-day horizon needs a year. This is the normal "
            "state of a newly started system, not a failure. Backfilling forecasts "
            "over past dates would not fix it: a model trained through those dates "
            "has already seen the answers."
        )
        add("")

    add("## 2. Realized performance")
    add("")
    add(markdown_table(_headline(context.production), column_formats=COUNT_FORMATS))

    add("## 3. Decision readiness")
    add("")
    add(
        "OPERATING_SPEC.md section 4: a horizon may not drive a promotion decision "
        f"until it has at least {context.thresholds.min_realized_for_drift} realized "
        "forecasts. Horizons below that are reported, not acted on."
    )
    add("")
    add(markdown_table(context.evidence, column_formats=COUNT_FORMATS))

    add("## 4. Performance drift")
    add("")
    add(
        "Realized production performance against the same model's walk-forward "
        "benchmark. `degradation` above "
        f"{context.thresholds.performance_degradation_fraction:.0%} on a "
        "decision-ready horizon is a retraining trigger."
    )
    add("")
    add(markdown_table(context.performance_drift, column_formats=COUNT_FORMATS))

    add("## 5. Feature drift")
    add("")
    calibrated = bool(context.feature_drift.attrs.get("calibrated", False))
    add(
        "Population Stability Index, training period vs the last "
        f"{context.thresholds.recent_days} days."
    )
    add("")
    if calibrated:
        add(
            "Thresholds are **calibrated per feature** against no-drift windows of "
            "the same length drawn from inside the training period "
            f"(warn at the {context.thresholds.warn_quantile:.0%} quantile, alert "
            f"at {context.thresholds.alert_quantile:.0%}). The fixed 0.1 / 0.25 "
            "values would flag 48 of 64 features on a window with no drift at all, "
            "because daily features are autocorrelated and one window is one "
            "regime. Read `excess`: 1.0 means as different as a quiet stretch of "
            "the training period already was."
        )
    else:
        add(
            f"**Uncalibrated**: falling back to fixed thresholds "
            f"({context.thresholds.feature_psi_warn} / "
            f"{context.thresholds.feature_psi_alert}), which over-report badly on "
            "autocorrelated daily features. Treat the levels as indicative only."
        )
    add("")
    add(
        "Feature drift is an early and unreliable signal either way: it can fire "
        "while the model is still performing perfectly well."
    )
    add("")
    drifted = context.feature_drift
    if drifted.empty:
        add("_(not enough history to compare)_")
        add("")
    else:
        flagged = drifted[drifted["level"].isin((LEVEL_WARN, LEVEL_ALERT))]
        add(
            f"{len(flagged)} of {len(drifted)} features are at or above the warn "
            "threshold. Top 15 by PSI:"
        )
        add("")
        add(markdown_table(drifted.head(15), column_formats=COUNT_FORMATS))

    add("## 6. Retraining triggers")
    add("")
    add(
        "OPERATING_SPEC.md section 3. These report; the weekly review decides. "
        "Nothing retrains on its own, and a fired trigger is a reason to consider "
        "a candidate, not to replace production."
    )
    add("")
    add(markdown_table(context.triggers, column_formats=COUNT_FORMATS))
    return "\n".join(lines)


def write_report(context: PerformanceContext, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(context), encoding="utf-8")
    return path


def pending_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """How many horizons are still waiting, by horizon."""
    if rows.empty:
        return pd.DataFrame(columns=["horizon_days", "resolved", "pending"])
    grouped = rows.groupby("horizon_days")["evaluation_status"]
    return pd.DataFrame(
        {
            "resolved": grouped.apply(lambda s: int((s == STATUS_FULL).sum())),
            "pending": grouped.apply(lambda s: int((s != STATUS_FULL).sum())),
        }
    ).reset_index()
