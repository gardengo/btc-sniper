"""Markdown report for the weekly model review.

OPERATING_SPEC.md section 2 defines the sequence; this renders what each step
found and what was decided. It is written for the person who has to defend the
decision a month later, so every verdict carries the number it was read from and
the checks that did *not* fire are shown alongside the ones that did.

A rejection is as much a result as a promotion. The report says why the
incumbent was kept, in the same detail, because "we looked and kept it" and "we
never looked" are indistinguishable from an unchanged model version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.validation import QualityReport
from src.models.promotion import (
    DECISION_PROMOTE,
    ROUTE_DATA_REFRESH,
    PromotionDecision,
    PromotionPolicy,
)
from src.monitoring.drift import DriftThresholds
from src.monitoring.markdown import markdown_table
from src.monitoring.realization import RealizationResult
from src.utils.provenance import describe_environment
from src.utils.timeutils import utc_now_iso

COUNT_FORMATS: dict[str, str] = {
    "horizon_days": "{:,.0f}",
    "folds": "{:,.0f}",
    "folds_won": "{:,.0f}",
    "sample_size": "{:,.0f}",
    "rows": "{:,.0f}",
    "resolved": "{:,.0f}",
    "pending": "{:,.0f}",
    "horizons": "{:,.0f}",
}
COMPARISON_COLUMNS: tuple[str, ...] = (
    "horizon_days",
    "model_weight",
    "folds",
    "folds_won",
    "win_rate",
    "mean_improvement",
    "worst_improvement",
    "candidate",
    "reference",
    "candidate_coverage",
    "reference_coverage",
    "low_power",
    "decision_horizon",
)


@dataclass(frozen=True)
class ReviewContext:
    """Everything the weekly review found, gathered by the job."""

    quality: QualityReport
    triggers: pd.DataFrame
    thresholds: DriftThresholds
    policy: PromotionPolicy
    blend_weights: pd.DataFrame
    incumbent_version: str | None = None
    candidate_version: str | None = None
    route_detail: str = ""
    decision: PromotionDecision | None = None
    realization: RealizationResult | None = None
    evidence: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_drift: pd.DataFrame = field(default_factory=pd.DataFrame)
    performance_drift: pd.DataFrame = field(default_factory=pd.DataFrame)
    applied: bool = False
    applied_note: str = ""
    skipped_reason: str = ""
    excluded_in_sample: int = 0
    data_end: str = ""
    training_cutoff: str | None = None
    outer_test_released: bool = False
    release_detail: str = ""

    @property
    def any_trigger_fired(self) -> bool:
        return bool(not self.triggers.empty and self.triggers["fired"].any())


def _fired(triggers: pd.DataFrame) -> str:
    if triggers.empty:
        return "none evaluated"
    fired = triggers[triggers["fired"]]
    if fired.empty:
        return "none"
    return ", ".join(f"`{name}`" for name in fired["trigger"])


def _comparison_view(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [name for name in COMPARISON_COLUMNS if name in frame.columns]
    return frame[columns]


def render_markdown(context: ReviewContext) -> str:
    lines: list[str] = []
    add = lines.append
    decision = context.decision

    add("# BTC Sniper - Weekly Model Review")
    add("")
    add(f"- generated: `{utc_now_iso()}`")
    add(f"- environment: `{describe_environment()}`")
    add(f"- data through: `{context.data_end}`")
    add(f"- production model: `{context.incumbent_version or 'none'}`")
    add(f"- candidate: `{context.candidate_version or 'none trained'}`")
    if decision is not None:
        add(f"- route: `{decision.route}` - {context.route_detail}")
        add(f"- **decision: `{decision.action}`**")
    add(f"- applied: `{'yes' if context.applied else 'no'}`{context.applied_note}")
    add(
        "- training cutoff: "
        + (
            "`released` - trained through the newest resolved label"
            if context.outer_test_released
            else "`purged` - stops before the outer test block"
        )
        + f" ({context.release_detail})"
    )
    add("")
    add(
        "Nothing in this review reads the outer test. Every number below comes "
        "from the inner block or from realized production forecasts "
        "(VALIDATION_SPEC.md section 4)."
    )
    add("")

    add("## 1. Data quality")
    add("")
    add(f"- {context.quality.summary()}")
    if not context.quality.ok:
        add("")
        add(
            "> **Blocking data errors are present.** A candidate trained on data "
            "that failed validation would carry the fault into every forecast it "
            "produces, so the review stops before training."
        )
    for row in context.quality.warnings:
        add(f"- warning: `{row.check_name}` - {row.message}")
    add("")

    add("## 2. Realized production performance")
    add("")
    if context.realization is None:
        add("_(no stored forecasts to realize)_")
        add("")
    else:
        add(f"- {context.realization.describe()}")
        if context.excluded_in_sample:
            add(
                f"- {context.excluded_in_sample} rows excluded as in-sample: their "
                "origin sits inside their own model's training window."
            )
        add("")
        if context.realization.resolved == 0:
            add(
                "> No forecast has resolved yet, so production performance cannot "
                "contribute to this decision. It accumulates from the first daily "
                "run forward (OPERATING_SPEC.md section 3.2)."
            )
            add("")
        add(
            "A horizon needs at least "
            f"{context.policy.min_realized_forecasts_per_horizon} realized "
            "forecasts before its production metric may drive anything "
            "(OPERATING_SPEC.md section 4)."
        )
        add("")
        add(markdown_table(context.evidence, column_formats=COUNT_FORMATS))

    add("## 3. Drift")
    add("")
    add("### Performance drift")
    add("")
    add(markdown_table(context.performance_drift, column_formats=COUNT_FORMATS))
    add("### Feature drift")
    add("")
    if context.feature_drift.empty:
        add("_(not enough history to compare)_")
        add("")
    else:
        flagged = int((context.feature_drift["level"] == "alert").sum())
        null_count = float(context.feature_drift.attrs.get("null_alert_count", 0.0))
        add(
            f"{flagged} of {len(context.feature_drift)} features flagged; a "
            f"no-drift window of the same length flags {null_count:.0f} "
            "(OPERATING_SPEC.md section 3.1)."
        )
        add("")
        add(
            markdown_table(context.feature_drift.head(10), column_formats=COUNT_FORMATS)
        )

    add("## 4. Retraining necessity")
    add("")
    add(f"Triggers fired: {_fired(context.triggers)}")
    add("")
    add(markdown_table(context.triggers, column_formats=COUNT_FORMATS))
    add(
        "A fired trigger is a reason to *build* a candidate, never a reason to "
        "ship one. Nothing retrains or promotes on its own "
        "(CLAUDE.md section 2.3)."
    )
    add("")

    if context.skipped_reason:
        add("## 5. No candidate was reviewed")
        add("")
        add(context.skipped_reason)
        add("")
        return "\n".join(lines)

    add("## 5. Candidate vs reference, on inner validation")
    add("")
    add(
        f"Judged on `{context.policy.decision_metric}`, which scores the whole "
        "predictive distribution rather than only its centre."
    )
    add("")
    add(
        "`model_weight` is the share of the served forecast that comes from the "
        "model at that horizon. A horizon at weight 0 is reported but marked "
        "`decision_horizon = False`: the forecast shipped there contains none of "
        "the model, so its number cannot promote or reject anything."
    )
    add("")
    add(markdown_table(context.blend_weights, column_formats=COUNT_FORMATS))

    if decision is None:
        add("_(no comparison was produced)_")
        add("")
        return "\n".join(lines)

    if decision.route == ROUTE_DATA_REFRESH:
        add(
            "> **This route has no comparison table, and that is the finding.** "
            "The candidate is the same configuration as the incumbent trained "
            "through a later cutoff. Folds are laid out backwards from the frozen "
            "`inner_validation_end`, and every feature is causal, so a walk-forward "
            "run over the inner block returns the same numbers it returned before "
            "the new data arrived: the extra data lies entirely after the last "
            "scored origin. Sliding the inner block forward to score it would be "
            "validation overfitting arriving one week at a time "
            "(VALIDATION_SPEC.md section 5). So this candidate is judged on "
            "equivalence and freshness instead, and the report says so rather than "
            "printing a table of differences that are structurally zero."
        )
        add("")
        add(
            "For reference, the candidate against the `no_change` baseline on the "
            "same folds. These are the numbers compared against the incumbent's "
            "recorded walk-forward result by the `reproduces_incumbent_validation` "
            "check -- they are expected to match to the last bit, and a mismatch "
            "means something changed that was not supposed to."
        )
        add("")
        add(
            markdown_table(
                _comparison_view(decision.comparison), column_formats=COUNT_FORMATS
            )
        )
    else:
        add(markdown_table(_comparison_view(decision.comparison), column_formats=COUNT_FORMATS))
        if not decision.regimes.empty:
            add("### By regime")
            add("")
            add(
                "A regime with fewer than "
                f"{context.policy.min_realized_forecasts_per_horizon} scored origins "
                "is reported but cannot veto."
            )
            add("")
            add(markdown_table(decision.regimes, column_formats=COUNT_FORMATS))

    add("## 6. Promotion gate")
    add("")
    add(
        "Blocking checks decide. Advisory checks are recorded because they are "
        "worth knowing and not worth vetoing on."
    )
    add("")
    add(markdown_table(decision.to_frame()))
    add(f"**Decision: `{decision.action}`**")
    add("")
    add(f"> {decision.reason()}")
    add("")
    if decision.action != DECISION_PROMOTE and context.incumbent_version:
        add(
            f"The production model stays `{context.incumbent_version}`. "
            "CLAUDE.md section 2.3: a candidate that is not shown to be better "
            "does not replace what is running."
        )
        add("")
    for advisory in decision.advisories():
        add(f"- advisory `{advisory.name}`: {advisory.detail}")
    if decision.advisories():
        add("")
    return "\n".join(lines)


def write_report(context: ReviewContext, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(context), encoding="utf-8")
    return path


__all__ = ["ReviewContext", "render_markdown", "write_report"]
