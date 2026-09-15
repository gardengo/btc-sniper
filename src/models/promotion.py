"""The promotion gate: may this candidate replace the production model?

CLAUDE.md section 2.3 and VALIDATION_SPEC.md section 10 define the rule. This
module turns it into a decision procedure that produces a *reason* as well as a
verdict, because a promotion nobody can explain later is not reviewable.

Three routes, because there are three genuinely different questions
------------------------------------------------------------------
**bootstrap** -- there is no production model yet. There is nothing to beat, so
the question is whether serving this model is better than serving the baseline
alone.

**configuration_change** -- the candidate differs from the incumbent in
algorithm, hyperparameters, training window, feature version or seed. Inner
validation can separate the two, so it must: the candidate has to beat the
incumbent on identical folds by the configured margin.

**data_refresh** -- the candidate is the same configuration trained through a
later cutoff. This is the weekly case, and it is the one where the usual gate
does not work.

Why a data refresh cannot be judged on inner validation
-------------------------------------------------------
Folds are laid out backwards from `validation.inner_validation_end`, a frozen
date (VALIDATION_SPEC.md section 2.1). Every feature in the repository is
causal. So a walk-forward run over the inner block produces **the same numbers
today as it did a year ago**: the extra data a refreshed candidate was trained
on lies entirely after the last fold, where nothing is scored.

That is not a gap to be patched. It is what "the validation block is frozen"
means, and the alternative -- sliding the inner block forward so the newest data
gets scored -- is the validation overfitting VALIDATION_SPEC.md section 5
prohibits, arriving one week at a time.

So a data refresh is not promoted on a claim of improvement. It is promoted on a
claim of **equivalence plus freshness**: the configuration is provably unchanged,
the cutoff has genuinely advanced, the refit reproduces the incumbent's recorded
validation numbers, and the new model's live forecast is sane. Nothing here
asserts the refreshed model is better, because nothing available can show that.
The evidence that it is better is the argument for retraining at all, and it is a
prior, not a measurement: a model fitted through last week has seen the market
that produced the price it is forecasting from.

What the gate judges, and where
-------------------------------
Only the horizons where the model actually carries weight. `forecast.blend`
gives the model a weight that falls to zero by 30 days (MODEL_SPEC.md section
6.5), so at 90 days the served forecast contains none of the model at all.
Rejecting a candidate for being worse at 90 days would be rejecting it on
evidence about a number nobody ships.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.evaluator import ALL_REGIMES
from src.evaluation.walk_forward import (
    aggregate_across_folds,
    consistency_summary,
    fold_consistency,
)
from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

ROUTE_BOOTSTRAP: str = "bootstrap"
ROUTE_CONFIGURATION: str = "configuration_change"
ROUTE_DATA_REFRESH: str = "data_refresh"

DECISION_PROMOTE: str = "promote"
DECISION_REJECT: str = "reject"
DECISION_KEEP: str = "keep_incumbent"


class PromotionError(RuntimeError):
    """Raised when a promotion decision cannot be formed."""


def nominal_coverage(metric_name: str) -> float:
    """`coverage_95` -> 0.95. The nominal level a coverage metric is aiming at."""
    tail = metric_name.rsplit("_", 1)[-1]
    if not tail.isdigit():
        raise PromotionError(f"cannot read a nominal level from {metric_name!r}")
    return int(tail) / 100.0


@dataclass(frozen=True)
class PromotionPolicy:
    """The configured bars a candidate has to clear."""

    require_validation_improvement: bool = True
    min_improvement_fraction: float = 0.02
    require_stability_across_folds: bool = True
    require_interval_coverage_not_materially_worse: bool = True
    max_coverage_shortfall: float = 0.05
    max_regime_degradation_fraction: float = 0.25
    min_realized_forecasts_per_horizon: int = 20
    decision_metric: str = "pinball_mean"
    coverage_metric: str = "coverage_95"

    @classmethod
    def from_config(cls, config: AppConfig) -> "PromotionPolicy":
        section: Mapping[str, Any] = dict(
            config.section("operation").get("production_promotion", {}) or {}
        )
        policy = cls(
            require_validation_improvement=bool(
                section.get("require_validation_improvement", True)
            ),
            min_improvement_fraction=float(
                section.get("min_improvement_fraction", 0.02)
            ),
            require_stability_across_folds=bool(
                section.get("require_stability_across_folds", True)
            ),
            require_interval_coverage_not_materially_worse=bool(
                section.get("require_interval_coverage_not_materially_worse", True)
            ),
            max_coverage_shortfall=float(section.get("max_coverage_shortfall", 0.05)),
            max_regime_degradation_fraction=float(
                section.get("max_regime_degradation_fraction", 0.25)
            ),
            min_realized_forecasts_per_horizon=int(
                section.get("min_realized_forecasts_per_horizon", 20)
            ),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if self.min_improvement_fraction < 0:
            raise PromotionError(
                "operation.production_promotion.min_improvement_fraction must be >= 0"
            )
        if self.max_coverage_shortfall < 0:
            raise PromotionError(
                "operation.production_promotion.max_coverage_shortfall must be >= 0"
            )
        if self.max_regime_degradation_fraction <= 0:
            raise PromotionError(
                "operation.production_promotion.max_regime_degradation_fraction "
                "must be > 0"
            )


@dataclass(frozen=True)
class GateCheck:
    """One condition, its verdict, and enough detail to argue with it."""

    name: str
    passed: bool
    detail: str
    blocking: bool = True

    @property
    def vetoes(self) -> bool:
        return self.blocking and not self.passed


# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

# Deliberately excluded: `training_cutoff`, `training_start` and `training_rows`
# (how much data the model saw, which is the whole point of a refresh),
# `config_version` (which moves when a logging setting changes), and the trained
# horizon set (`model_version_name` does not encode it either, so two models
# differing only there already share an identity). Everything that actually
# changes the fitted function is listed.
CONFIGURATION_KEYS: tuple[str, ...] = (
    "algorithm",
    "training_window_strategy",
    "feature_version",
    "horizon_grid_version",
    "random_seed",
)


def configuration_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """The configuration half of a registry row, cutoff excluded."""
    raw = row["hyperparameters"] if "hyperparameters" in _keys(row) else "{}"
    hyperparameters = json.loads(raw or "{}") if isinstance(raw, str) else dict(raw)
    payload: dict[str, Any] = {key: row[key] for key in CONFIGURATION_KEYS}
    payload["params"] = hyperparameters.get("params", {})
    return payload


def _keys(row: Mapping[str, Any]) -> Sequence[str]:
    keys = getattr(row, "keys", None)
    return list(keys()) if callable(keys) else list(row)


def configuration_fingerprint(row: Mapping[str, Any], length: int = 12) -> str:
    """Stable hash of what a model *is*, independent of how much data it saw.

    Two models with the same fingerprint and different cutoffs are the same
    configuration refreshed. That is the distinction the whole route selection
    rests on, so it is computed from named fields rather than from the version
    string, which is only a rendering of them.
    """
    payload = json.dumps(configuration_payload(row), sort_keys=True, default=str)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()[:length]


def configuration_row(
    config: AppConfig, *, strategy: str, params: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    """A registry-shaped configuration for a model that has not been trained yet.

    Lets a caller ask "has *this* design already been evaluated on the outer
    test?" without training anything first, which is the question
    `jobs.weekly_model_review` has to answer before it decides what cutoff to
    train at.
    """
    return {
        "algorithm": str(config.section("models").get("primary_algorithm", "")),
        "training_window_strategy": strategy,
        "feature_version": config.features.version,
        "horizon_grid_version": config.forecast.horizon_grid_version,
        "random_seed": int(seed),
        "hyperparameters": json.dumps({"params": dict(params)}),
    }


def evaluated_designs(models: pd.DataFrame) -> set[str]:
    """Configuration fingerprints that have a recorded outer-test evaluation."""
    if models.empty:
        return set()
    return {
        configuration_fingerprint(row)
        for row in models.to_dict("records")
        if row.get("test_metrics") and row["test_metrics"] not in ("{}", "null")
    }


def classify_route(
    candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None
) -> tuple[str, str]:
    """Which of the three questions this review is actually asking."""
    if incumbent is None:
        return ROUTE_BOOTSTRAP, "no production model is registered"
    candidate_print = configuration_fingerprint(candidate)
    incumbent_print = configuration_fingerprint(incumbent)
    if candidate_print == incumbent_print:
        return (
            ROUTE_DATA_REFRESH,
            f"identical configuration ({candidate_print}), cutoff "
            f"{incumbent['training_cutoff']} -> {candidate['training_cutoff']}",
        )
    return (
        ROUTE_CONFIGURATION,
        f"configuration differs: {incumbent_print} -> {candidate_print} "
        f"({', '.join(configuration_diff(candidate, incumbent)) or 'hyperparameters'})",
    )


def configuration_diff(
    candidate: Mapping[str, Any], incumbent: Mapping[str, Any]
) -> list[str]:
    """Which configuration fields differ, for the decision record."""
    left, right = configuration_payload(candidate), configuration_payload(incumbent)
    return [key for key in sorted(left) if left[key] != right.get(key)]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(
    metrics: pd.DataFrame,
    *,
    candidate_version: str,
    reference_version: str,
    policy: PromotionPolicy,
    weights: Mapping[int, float],
) -> pd.DataFrame:
    """Per-horizon candidate-vs-reference table on identical folds.

    ``metrics`` must contain both versions scored on the same folds and origins;
    that is what `run_walk_forward` produces, and comparing rows from two runs
    that saw different origins would not be a comparison.

    ``weights`` is the blend weight per horizon. A horizon with zero weight is
    reported and marked ``decision_horizon = False``: the served forecast does
    not contain the model there, so its number cannot promote or reject anything.
    """
    consistency = fold_consistency(
        metrics, policy.decision_metric, candidate_version, reference_version
    )
    summary = consistency_summary(consistency)
    if summary.empty:
        return pd.DataFrame()

    aggregated = aggregate_across_folds(
        metrics, (policy.decision_metric, policy.coverage_metric)
    )

    def pick(metric: str, version: str, column: str = "mean") -> pd.Series:
        subset = aggregated[
            (aggregated["metric_name"] == metric)
            & (aggregated["model_version"] == version)
        ]
        if subset.empty:
            return pd.Series(dtype=float)
        return subset.set_index("horizon_days")[column]

    low_power = (
        aggregated.drop_duplicates("horizon_days")
        .set_index("horizon_days")["low_power"]
    )
    nominal = nominal_coverage(policy.coverage_metric)

    frame = summary.copy()
    frame["model_weight"] = frame["horizon_days"].map(
        lambda horizon: float(weights.get(int(horizon), 0.0))
    )
    frame["candidate"] = frame["horizon_days"].map(
        pick(policy.decision_metric, candidate_version)
    )
    frame["reference"] = frame["horizon_days"].map(
        pick(policy.decision_metric, reference_version)
    )
    frame["candidate_coverage"] = frame["horizon_days"].map(
        pick(policy.coverage_metric, candidate_version)
    )
    frame["reference_coverage"] = frame["horizon_days"].map(
        pick(policy.coverage_metric, reference_version)
    )
    # Coverage is judged by distance from nominal, not by being larger. A 95%
    # interval that covers 99% of outcomes is miscalibrated in the same way one
    # covering 90% is; it is simply wrong in the direction that feels safe.
    frame["coverage_penalty"] = (frame["candidate_coverage"] - nominal).abs() - (
        frame["reference_coverage"] - nominal
    ).abs()
    frame["low_power"] = frame["horizon_days"].map(low_power).fillna(False).astype(bool)
    frame["decision_horizon"] = (frame["model_weight"] > 0.0) & (~frame["low_power"])
    return frame.sort_values("horizon_days").reset_index(drop=True)


def regime_comparison(
    metrics: pd.DataFrame,
    *,
    candidate_version: str,
    reference_version: str,
    policy: PromotionPolicy,
    horizons: Sequence[int],
) -> pd.DataFrame:
    """Candidate vs reference inside each market regime, at the decision horizons.

    VALIDATION_SPEC.md section 10 makes "no catastrophic degradation in a key
    regime" a promotion condition. Regimes recur, so a candidate that is much
    worse in one of them will be much worse again.

    A regime with fewer scored origins than `min_realized_forecasts_per_horizon`
    is reported but cannot veto -- the project's standing rule that a tiny sample
    does not decide anything (OPERATING_SPEC.md section 4), applied here too.
    """
    wanted = set(int(h) for h in horizons)
    subset = metrics[
        (metrics["regime"] != ALL_REGIMES)
        & (metrics["horizon_days"].isin(wanted))
        & (metrics["model_version"].isin({candidate_version, reference_version}))
        & (metrics["metric_name"].isin({policy.decision_metric, "sample_size"}))
    ]
    if subset.empty:
        return pd.DataFrame()

    pivoted = subset.pivot_table(
        index=["horizon_days", "regime"],
        columns=["model_version", "metric_name"],
        values="metric_value",
        aggfunc="mean",
    )
    rows: list[dict[str, Any]] = []
    for (horizon, regime), row in pivoted.iterrows():
        candidate = row.get((candidate_version, policy.decision_metric))
        reference = row.get((reference_version, policy.decision_metric))
        sample = row.get((candidate_version, "sample_size"))
        if pd.isna(candidate) or pd.isna(reference) or not reference:
            continue
        degradation = float(candidate) / float(reference) - 1.0
        decisive = bool(
            not pd.isna(sample)
            and int(sample) >= policy.min_realized_forecasts_per_horizon
        )
        rows.append(
            {
                "horizon_days": int(horizon),
                "regime": str(regime),
                "sample_size": 0 if pd.isna(sample) else int(sample),
                "candidate": float(candidate),
                "reference": float(reference),
                "degradation": degradation,
                "decisive": decisive,
                "catastrophic": bool(
                    decisive and degradation > policy.max_regime_degradation_fraction
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon_days", "regime"]).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateInputs:
    """Everything the gate needs, gathered by the weekly review job."""

    route: str
    candidate_version: str
    incumbent_version: str | None = None
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    regimes: pd.DataFrame = field(default_factory=pd.DataFrame)
    reproducible: bool | None = None
    cutoff_advance_days: int | None = None
    new_observations: int = 0
    min_new_observations: int = 0
    equivalent: bool | None = None
    equivalence_detail: str = ""
    sanity_violations: tuple[str, ...] = ()
    sanity_detail: str = ""


@dataclass(frozen=True)
class PromotionDecision:
    """The verdict, the reason, and the table it was read from."""

    route: str
    action: str
    candidate_version: str
    incumbent_version: str | None
    checks: tuple[GateCheck, ...]
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    regimes: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def promote(self) -> bool:
        return self.action == DECISION_PROMOTE

    def failed(self) -> tuple[GateCheck, ...]:
        return tuple(check for check in self.checks if check.vetoes)

    def advisories(self) -> tuple[GateCheck, ...]:
        return tuple(
            check for check in self.checks if not check.blocking and not check.passed
        )

    def reason(self) -> str:
        """One line, recorded in the registry with the promotion or rejection."""
        if self.promote:
            passed = ", ".join(check.name for check in self.checks if check.blocking)
            return f"{self.route}: passed {passed or 'the gate'}"
        failures = "; ".join(f"{c.name} ({c.detail})" for c in self.failed())
        return f"{self.route}: {failures or 'no candidate to promote'}"

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "check": check.name,
                    "result": "pass" if check.passed else "FAIL",
                    "blocking": check.blocking,
                    "detail": check.detail,
                }
                for check in self.checks
            ]
        )

    def describe(self) -> str:
        return (
            f"route={self.route} action={self.action} "
            f"candidate={self.candidate_version} "
            f"blocking_failures={len(self.failed())}"
        )


def _decision_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty or "decision_horizon" not in comparison.columns:
        return pd.DataFrame()
    return comparison[comparison["decision_horizon"]]


def _fmt(rows: pd.DataFrame, column: str) -> str:
    return ", ".join(
        f"h={int(row.horizon_days)}d {getattr(row, column):+.1%}"
        for row in rows.itertuples(index=False)
    )


def _check_earns_weight(rows: pd.DataFrame, policy: PromotionPolicy) -> GateCheck:
    """Somewhere it is used, the model must be consistently better than nothing.

    Without this a model that merely ties the baseline everywhere would be
    promoted, and serving a tree that reproduces the baseline is strictly worse
    than serving the baseline: same forecast, more moving parts.
    """
    if rows.empty:
        return GateCheck(
            "earns_its_weight", False, "no horizon where the model carries weight"
        )

    earned = rows[
        (rows["win_rate"] >= 1.0)
        & (rows["mean_improvement"] >= policy.min_improvement_fraction)
    ]
    if earned.empty:
        best = rows.loc[rows["mean_improvement"].idxmax()]
        return GateCheck(
            "earns_its_weight",
            False,
            f"no weighted horizon improves by {policy.min_improvement_fraction:.0%} "
            f"in every fold; best is h={int(best.horizon_days)}d at "
            f"{best.mean_improvement:+.1%} in {int(best.folds_won)}/{int(best.folds)} folds",
        )
    return GateCheck(
        "earns_its_weight",
        True,
        f"{_fmt(earned, 'mean_improvement')} in every fold",
    )


def _check_no_damage(rows: pd.DataFrame, policy: PromotionPolicy) -> GateCheck:
    """Nowhere it is used may the model be materially worse than the baseline."""
    if rows.empty:
        return GateCheck("does_no_damage", True, "no weighted horizon to check")
    hurt = rows[rows["mean_improvement"] < -policy.min_improvement_fraction]
    if not hurt.empty:
        return GateCheck(
            "does_no_damage",
            False,
            f"worse than the reference where it is served: "
            f"{_fmt(hurt, 'mean_improvement')}",
        )
    return GateCheck(
        "does_no_damage",
        True,
        f"no weighted horizon worse than {policy.min_improvement_fraction:.0%}",
    )


def _check_improvement(rows: pd.DataFrame, policy: PromotionPolicy) -> GateCheck:
    if rows.empty:
        return GateCheck(
            "beats_incumbent", False, "no comparable horizon at which to judge"
        )
    short = rows[rows["mean_improvement"] < policy.min_improvement_fraction]
    if not short.empty:
        return GateCheck(
            "beats_incumbent",
            False,
            f"below the {policy.min_improvement_fraction:.0%} margin at "
            f"{_fmt(short, 'mean_improvement')}",
        )
    return GateCheck(
        "beats_incumbent", True, f"{_fmt(rows, 'mean_improvement')} over the incumbent"
    )


def _check_fold_stability(rows: pd.DataFrame, policy: PromotionPolicy) -> GateCheck:
    """Winning on average by winning once is not winning (VALIDATION_SPEC 10.1)."""
    if rows.empty:
        return GateCheck("stable_across_folds", False, "no folds to check")
    unstable = rows[rows["win_rate"] < 1.0]
    if not unstable.empty:
        detail = ", ".join(
            f"h={int(row.horizon_days)}d {int(row.folds_won)}/{int(row.folds)} folds"
            for row in unstable.itertuples(index=False)
        )
        return GateCheck(
            "stable_across_folds", False, f"loses in at least one fold: {detail}"
        )
    return GateCheck("stable_across_folds", True, "wins in every fold at every horizon")


def _check_coverage(rows: pd.DataFrame, policy: PromotionPolicy) -> GateCheck:
    if rows.empty or rows["coverage_penalty"].isna().all():
        return GateCheck(
            "interval_coverage", True, "no coverage measurement available", blocking=False
        )
    worse = rows[rows["coverage_penalty"] > policy.max_coverage_shortfall]
    if not worse.empty:
        detail = ", ".join(
            f"h={int(row.horizon_days)}d {row.candidate_coverage:.2f} vs "
            f"{row.reference_coverage:.2f}"
            for row in worse.itertuples(index=False)
        )
        return GateCheck(
            "interval_coverage",
            False,
            f"further from nominal than the reference by more than "
            f"{policy.max_coverage_shortfall:.2f}: {detail}",
        )
    return GateCheck(
        "interval_coverage",
        True,
        f"no horizon further from nominal by more than "
        f"{policy.max_coverage_shortfall:.2f}",
    )


def _check_regimes(regimes: pd.DataFrame, policy: PromotionPolicy) -> GateCheck:
    if regimes.empty:
        return GateCheck(
            "regime_stability",
            True,
            "no regime-tagged origins to compare",
            blocking=False,
        )
    catastrophic = regimes[regimes["catastrophic"]]
    if not catastrophic.empty:
        detail = ", ".join(
            f"{row.regime} h={int(row.horizon_days)}d {row.degradation:+.0%}"
            for row in catastrophic.itertuples(index=False)
        )
        return GateCheck(
            "regime_stability",
            False,
            f"worse than the reference by more than "
            f"{policy.max_regime_degradation_fraction:.0%} in: {detail}",
        )
    decisive = int(regimes["decisive"].sum())
    return GateCheck(
        "regime_stability",
        True,
        f"no catastrophic regime among {decisive} with enough sample "
        f"({len(regimes) - decisive} too small to decide)",
    )


def _check_catastrophe(comparison: pd.DataFrame, policy: PromotionPolicy) -> GateCheck:
    """A low-power horizon cannot justify a promotion, but it can veto one.

    VALIDATION_SPEC.md section 10 says exactly that. The horizon is excluded from
    the decision metric yet still served, so a catastrophic result there is a
    reason not to ship even though a good result there is not a reason to ship.
    """
    served = comparison[comparison["model_weight"] > 0.0] if not comparison.empty else comparison
    if served.empty:
        return GateCheck("no_catastrophic_horizon", True, "nothing served from the model")
    ruined = served[
        served["mean_improvement"] < -policy.max_regime_degradation_fraction
    ]
    if not ruined.empty:
        return GateCheck(
            "no_catastrophic_horizon",
            False,
            f"catastrophically worse where it is served: {_fmt(ruined, 'mean_improvement')}",
        )
    return GateCheck(
        "no_catastrophic_horizon",
        True,
        f"no served horizon worse than {policy.max_regime_degradation_fraction:.0%}",
    )


def _check_reproducible(flag: bool | None) -> GateCheck:
    if flag is None:
        return GateCheck(
            "reproducible",
            False,
            "not checked; VALIDATION_SPEC.md section 11 requires it before promotion",
        )
    return GateCheck(
        "reproducible",
        bool(flag),
        "refit produced identical predictions"
        if flag
        else "a refit on identical data produced different predictions",
    )


def _check_weight_support(comparison: pd.DataFrame) -> GateCheck:
    """Advisory: horizons trusted more than the evidence supports.

    The blend ramp was chosen so the forecast curve has no step in it, not from a
    per-horizon measurement, so it can hand a horizon substantial weight where
    the model is indistinguishable from the baseline. That is worth seeing. It is
    not a veto: a weight on a model that merely ties does no damage.
    """
    if comparison.empty:
        return GateCheck("blend_weight_supported", True, "nothing to check", blocking=False)
    trusted = comparison[
        (comparison["model_weight"] >= 0.5) & (comparison["win_rate"] < 1.0)
    ]
    if trusted.empty:
        return GateCheck(
            "blend_weight_supported",
            True,
            "every horizon weighted at 50% or more wins in every fold",
            blocking=False,
        )
    detail = ", ".join(
        f"h={int(row.horizon_days)}d weight {row.model_weight:.2f} but "
        f"{int(row.folds_won)}/{int(row.folds)} folds"
        for row in trusted.itertuples(index=False)
    )
    return GateCheck("blend_weight_supported", False, detail, blocking=False)


def _check_cutoff_advance(days: int | None, route: str) -> GateCheck:
    if days is None:
        return GateCheck("cutoff_advances", False, "training cutoff unknown")
    if days <= 0:
        return GateCheck(
            "cutoff_advances",
            False,
            f"the candidate's cutoff is {abs(days)} days earlier than the "
            "incumbent's; a refresh that sees less data is not a refresh",
        )
    return GateCheck(
        "cutoff_advances", True, f"trained through {days} more days than the incumbent"
    )


def _check_new_observations(inputs: GateInputs) -> GateCheck:
    enough = inputs.new_observations >= inputs.min_new_observations
    return GateCheck(
        "enough_new_data",
        enough,
        f"{inputs.new_observations} new daily candles since the incumbent's cutoff "
        f"(threshold {inputs.min_new_observations})",
    )


def _check_equivalence(inputs: GateInputs) -> GateCheck:
    """The refit must reproduce the incumbent's recorded validation numbers.

    On a frozen inner block with causal features this is expected to hold
    exactly, which is what makes it a useful integrity check rather than a
    tautology: if it fails, something changed that was not supposed to -- revised
    history, a feature-pipeline change, or non-determinism.
    """
    if inputs.equivalent is None:
        return GateCheck(
            "reproduces_incumbent_validation",
            True,
            inputs.equivalence_detail
            or "the incumbent has no recorded validation numbers to compare against; "
            "the candidate's are recorded now for the next review",
            blocking=False,
        )
    return GateCheck(
        "reproduces_incumbent_validation",
        bool(inputs.equivalent),
        inputs.equivalence_detail,
    )


def _check_sanity(inputs: GateInputs) -> GateCheck:
    if inputs.sanity_violations:
        return GateCheck(
            "forecast_is_sane",
            False,
            f"{len(inputs.sanity_violations)} implausible predictions: "
            + "; ".join(inputs.sanity_violations[:3]),
        )
    return GateCheck(
        "forecast_is_sane",
        True,
        inputs.sanity_detail or "predictions are finite, ordered and within "
        "historically observed moves",
    )


def _inner_validation_note() -> GateCheck:
    return GateCheck(
        "inner_validation_cannot_separate",
        True,
        "folds end at the frozen inner_validation_end, so the extra training data "
        "lies entirely after the last scored origin; this route is judged on "
        "equivalence and freshness, not on a measured improvement",
        blocking=False,
    )


def evaluate_gate(inputs: GateInputs, policy: PromotionPolicy) -> PromotionDecision:
    """Run the checks for this route and return the decision with its reasons."""
    rows = _decision_rows(inputs.comparison)
    checks: list[GateCheck] = []

    if inputs.route == ROUTE_BOOTSTRAP:
        checks.append(_check_earns_weight(rows, policy))
        checks.append(_check_no_damage(rows, policy))
        checks.append(_check_catastrophe(inputs.comparison, policy))
        if policy.require_interval_coverage_not_materially_worse:
            checks.append(_check_coverage(rows, policy))
        checks.append(_check_regimes(inputs.regimes, policy))
        checks.append(_check_weight_support(inputs.comparison))
    elif inputs.route == ROUTE_CONFIGURATION:
        if policy.require_validation_improvement:
            checks.append(_check_improvement(rows, policy))
        if policy.require_stability_across_folds:
            checks.append(_check_fold_stability(rows, policy))
        checks.append(_check_catastrophe(inputs.comparison, policy))
        if policy.require_interval_coverage_not_materially_worse:
            checks.append(_check_coverage(rows, policy))
        checks.append(_check_regimes(inputs.regimes, policy))
        checks.append(_check_weight_support(inputs.comparison))
    elif inputs.route == ROUTE_DATA_REFRESH:
        checks.append(_check_cutoff_advance(inputs.cutoff_advance_days, inputs.route))
        checks.append(_check_new_observations(inputs))
        checks.append(_check_equivalence(inputs))
        checks.append(_check_sanity(inputs))
        checks.append(_inner_validation_note())
    else:
        raise PromotionError(f"unknown promotion route {inputs.route!r}")

    checks.append(_check_reproducible(inputs.reproducible))

    ordered = tuple(checks)
    blocked = any(check.vetoes for check in ordered)
    action = (
        DECISION_KEEP
        if blocked and inputs.incumbent_version
        else DECISION_REJECT
        if blocked
        else DECISION_PROMOTE
    )
    return PromotionDecision(
        route=inputs.route,
        action=action,
        candidate_version=inputs.candidate_version,
        incumbent_version=inputs.incumbent_version,
        checks=ordered,
        comparison=inputs.comparison,
        regimes=inputs.regimes,
    )


def forecast_sanity(
    predicted: pd.DataFrame, close: pd.Series, *, extreme_multiple: float = 3.0
) -> tuple[tuple[str, ...], str]:
    """Is this model's live forecast plausible at all?

    The data-refresh route cannot measure the candidate against anything, so this
    is the check that catches a broken refit: a fit that diverged, a feature
    column that arrived as NaN, quantiles that came back out of order. The bar is
    deliberately loose -- it asks whether the forecast is *possible*, not whether
    it is good -- because a tight bar here would be a model selected on the most
    recent data, which is the thing the whole project is built to avoid.

    ``predicted`` is indexed by horizon in days, with one column per quantile
    level, in predicted log-return space.
    """
    if predicted.empty:
        return ("no predictions produced",), ""

    log_close = np.log(close.astype(float))
    levels = sorted(float(column) for column in predicted.columns)
    median_level = min(levels, key=lambda level: abs(level - 0.50))
    violations: list[str] = []
    worst_seen: dict[int, float] = {}

    for horizon, row in predicted.iterrows():
        horizon = int(horizon)
        values = row.reindex(levels).astype(float)
        if not np.isfinite(values.to_numpy()).all():
            violations.append(f"h={horizon}d has non-finite quantiles")
            continue
        if (values.diff().dropna() < 0).any():
            violations.append(f"h={horizon}d quantiles are out of order")
        history = log_close.diff(horizon).dropna()
        if history.empty:
            continue
        extreme = float(history.abs().max())
        worst_seen[horizon] = extreme
        median = float(values[median_level])
        if abs(median) > extreme:
            violations.append(
                f"h={horizon}d median move {median:+.2f} exceeds the largest "
                f"{horizon}-day move ever observed ({extreme:.2f})"
            )
        span = float(values.iloc[-1] - values.iloc[0])
        if span > extreme_multiple * 2.0 * extreme:
            violations.append(
                f"h={horizon}d interval spans {span:.2f} in log space, more than "
                f"{extreme_multiple:.0f}x the historical range"
            )

    detail = (
        f"{len(predicted)} horizons checked against the largest move ever observed "
        f"at each ({min(worst_seen.values()):.2f}-{max(worst_seen.values()):.2f} in "
        "log space)"
        if worst_seen
        else f"{len(predicted)} horizons checked"
    )
    return tuple(violations), detail


# ---------------------------------------------------------------------------
# The recorded validation fingerprint
# ---------------------------------------------------------------------------


def validation_record(
    comparison: pd.DataFrame, *, policy: PromotionPolicy, model_version: str
) -> dict[str, Any]:
    """The walk-forward numbers stored on a registry row.

    Kept small and per-horizon on purpose: it exists so the *next* review can ask
    whether a refit of the same configuration still produces the same inner-block
    result. `model_registry.validation_metrics` was empty before this -- training
    cannot fill it, because training does not validate.
    """
    if comparison.empty:
        return {"metric": policy.decision_metric, "walk_forward": model_version}
    return {
        "metric": policy.decision_metric,
        "walk_forward": model_version,
        "by_horizon": {
            str(int(row.horizon_days)): round(float(row.candidate), 10)
            for row in comparison.itertuples(index=False)
            if not pd.isna(row.candidate)
        },
        "coverage_metric": policy.coverage_metric,
        "coverage_by_horizon": {
            str(int(row.horizon_days)): round(float(row.candidate_coverage), 10)
            for row in comparison.itertuples(index=False)
            if not pd.isna(row.candidate_coverage)
        },
    }


def compare_validation_records(
    candidate: Mapping[str, Any] | None,
    incumbent: Mapping[str, Any] | None,
    *,
    tolerance: float = 1e-9,
) -> tuple[bool | None, str]:
    """Does a refit reproduce the incumbent's recorded inner-block numbers?

    Returns ``None`` when the incumbent has nothing recorded -- the honest answer
    the first time, and not a pass. On a frozen inner block with causal features
    the numbers should match to the last bit; a mismatch means history was
    revised, the feature pipeline changed, or a fit is not deterministic.
    """
    if not candidate or not incumbent:
        return None, "no recorded walk-forward numbers to compare against"
    left = dict(candidate.get("by_horizon", {}) or {})
    right = dict(incumbent.get("by_horizon", {}) or {})
    if not left or not right:
        return None, "no recorded walk-forward numbers to compare against"
    shared = sorted(set(left) & set(right), key=int)
    if not shared:
        return None, "the recorded horizons do not overlap"

    drifted = [
        f"h={horizon}d {right[horizon]:.6f} -> {left[horizon]:.6f}"
        for horizon in shared
        if abs(float(left[horizon]) - float(right[horizon]))
        > tolerance * max(1.0, abs(float(right[horizon])))
    ]
    if drifted:
        return False, (
            f"{len(drifted)} of {len(shared)} horizons changed on a frozen inner "
            f"block: {'; '.join(drifted[:3])}"
        )
    return True, (
        f"all {len(shared)} shared horizons reproduce the incumbent's recorded "
        "inner-block result exactly"
    )


def assert_comparable_folds(candidate: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Refuse to compare two runs whose fold numbering does not line up.

    Fold names are positional (`fold01`, `fold02`, ...) and a fold is dropped
    when its training set is too small, so two strategies can end up with the
    same names over different validation blocks. Comparing those would silently
    score a candidate's 2021 against an incumbent's 2022 and report the market
    difference as a model difference.
    """
    def layout(frame: pd.DataFrame) -> dict[tuple[int, str], str]:
        if frame.empty:
            return {}
        return {
            (int(row.horizon_days), str(row.fold)): str(row.validation_start)
            for row in frame.itertuples(index=False)
        }

    left, right = layout(candidate), layout(reference)
    shared = set(left) & set(right)
    if not shared:
        raise PromotionError(
            "the two walk-forward runs share no (horizon, fold) key; they cannot "
            "be compared"
        )
    mismatched = [key for key in sorted(shared) if left[key] != right[key]]
    if mismatched:
        horizon, fold = mismatched[0]
        raise PromotionError(
            f"{len(mismatched)} folds cover different periods between the two runs "
            f"(h={horizon}d {fold}: {right[(horizon, fold)]} vs "
            f"{left[(horizon, fold)]}); a comparison would report a market "
            "difference as a model difference"
        )


__all__ = [
    "DECISION_KEEP",
    "assert_comparable_folds",
    "DECISION_PROMOTE",
    "DECISION_REJECT",
    "GateCheck",
    "GateInputs",
    "PromotionDecision",
    "PromotionError",
    "PromotionPolicy",
    "ROUTE_BOOTSTRAP",
    "ROUTE_CONFIGURATION",
    "ROUTE_DATA_REFRESH",
    "classify_route",
    "compare_validation_records",
    "compare",
    "configuration_diff",
    "configuration_fingerprint",
    "configuration_payload",
    "configuration_row",
    "evaluated_designs",
    "evaluate_gate",
    "forecast_sanity",
    "nominal_coverage",
    "regime_comparison",
    "validation_record",
]
