"""Drift detection: has the world moved away from what the model was trained on?

OPERATING_SPEC.md section 3 makes two of the retraining triggers drift-based:
recent performance degradation over a minimum sample, and feature distribution
drift past a threshold. This module measures both. It never retrains anything --
it reports, and the weekly review decides (section 2).

Two kinds of drift, deliberately kept apart
-------------------------------------------
**Feature drift** says the inputs look different from the training period. It can
fire while the model is still performing fine, and on its own it is a warning,
not a verdict.

**Performance drift** says the realized forecasts have got worse. It is the one
that matters, and it is also the one that arrives late -- a 365-day forecast
cannot degrade visibly until a year has passed. That asymmetry is why both are
measured: feature drift is the early, unreliable signal; performance drift is the
late, reliable one.

Population Stability Index, and why its usual thresholds do not apply
--------------------------------------------------------------------
PSI compares two distributions over fixed bins::

    PSI = sum( (current_share - reference_share) * ln(current_share / reference_share) )

The conventional 0.1 / 0.25 thresholds come from credit scoring, where the
current sample is roughly an independent draw. Daily market features are nothing
like that: they are strongly autocorrelated, so a contiguous 90-day window is one
regime, not a random sample of five years. Measured on this dataset, a window
drawn from *inside the reference period itself* -- by construction no drift at
all -- puts **48 of 64 features past 0.25**. A trigger that fires every time
teaches the reader to ignore it, which is worse than having no trigger.

So the thresholds are calibrated instead. `null_psi_thresholds` samples
contiguous windows from the reference period, computes the PSI each one produces
against the rest of the reference, and takes high quantiles of that distribution.
A feature is flagged only when it exceeds what a no-drift window of the same
length already produces. That cut the real-data alert count from 48 features to
about 12.

Per-feature calibration is still not enough, and the measurement says so: on
no-drift windows it flags about **16% of features**, not the nominal 5%. Two
reasons, both structural rather than fixable by moving a number. The recent
window is always at the edge of the reference rather than inside it, and markets
trend. And the features are strongly correlated with each other, so flags arrive
in clusters instead of independently.

The trigger therefore asks a different question: **does this window flag more
features than a no-drift window does?** `null_psi_thresholds` records the joint
alert count of each calibration window, and `retraining_triggers` compares
against its high quantile. That is the multiple-testing-aware version of the same
calibration, it costs no extra computation, and it brings the trigger's own false
positive rate down to 1 in 4 on the same no-drift checks.

Feature drift stays a weak signal even so. It is one of several triggers and the
weekly review adjudicates; it must never be the sole reason to replace anything.

The fixed thresholds remain as a fallback for when there is not enough reference
history to calibrate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

EPSILON: float = 1e-6
LEVEL_STABLE: str = "stable"
LEVEL_WARN: str = "warn"
LEVEL_ALERT: str = "alert"


class DriftError(ValueError):
    """Raised when drift cannot be computed from the given samples."""


@dataclass(frozen=True)
class DriftThresholds:
    """Configured thresholds for both drift kinds."""

    feature_psi_warn: float
    feature_psi_alert: float
    psi_bins: int
    recent_days: int
    performance_degradation_fraction: float
    min_realized_for_drift: int
    calibrate: bool = True
    calibration_samples: int = 24
    warn_quantile: float = 0.90
    alert_quantile: float = 0.99
    calibration_seed: int = 20260914

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "DriftThresholds":
        drift = dict(section.get("drift", {}) or {})
        calibration = dict(drift.get("calibration", {}) or {})
        thresholds = cls(
            feature_psi_warn=float(drift.get("feature_psi_warn", 0.10)),
            feature_psi_alert=float(drift.get("feature_psi_alert", 0.25)),
            psi_bins=int(drift.get("psi_bins", 10)),
            recent_days=int(drift.get("recent_days", 90)),
            performance_degradation_fraction=float(
                drift.get("performance_degradation_fraction", 0.20)
            ),
            min_realized_for_drift=int(drift.get("min_realized_for_drift", 20)),
            calibrate=bool(calibration.get("enabled", True)),
            calibration_samples=int(calibration.get("samples", 24)),
            warn_quantile=float(calibration.get("warn_quantile", 0.90)),
            alert_quantile=float(calibration.get("alert_quantile", 0.99)),
            calibration_seed=int(calibration.get("seed", 20260914)),
        )
        thresholds.validate()
        return thresholds

    def validate(self) -> None:
        if self.feature_psi_alert <= self.feature_psi_warn:
            raise DriftError("feature_psi_alert must be greater than feature_psi_warn")
        if self.psi_bins < 2:
            raise DriftError("psi_bins must be >= 2")
        if self.recent_days < 1:
            raise DriftError("recent_days must be >= 1")
        if not 0.0 < self.warn_quantile < self.alert_quantile < 1.0:
            raise DriftError(
                "drift calibration needs 0 < warn_quantile < alert_quantile < 1"
            )
        if self.calibration_samples < 2:
            raise DriftError("drift calibration needs at least 2 samples")

    def level_for(
        self, psi: float, *, warn: float | None = None, alert: float | None = None
    ) -> str:
        """Severity of one PSI value, against calibrated thresholds when given."""
        if np.isnan(psi):
            return LEVEL_STABLE
        high = self.feature_psi_alert if alert is None or np.isnan(alert) else alert
        low = self.feature_psi_warn if warn is None or np.isnan(warn) else warn
        if psi >= high:
            return LEVEL_ALERT
        if psi >= low:
            return LEVEL_WARN
        return LEVEL_STABLE


def population_stability_index(
    reference: pd.Series, current: pd.Series, *, bins: int = 10
) -> float:
    """PSI between a reference and a current sample.

    Bin edges come from the **reference** quantiles, which is what makes the
    comparison meaningful: the question is how much of the current sample has
    moved out of the regions the reference occupied.
    """
    left = pd.Series(reference).dropna().astype("float64")
    right = pd.Series(current).dropna().astype("float64")
    if left.empty or right.empty:
        return float("nan")
    if bins < 2:
        raise DriftError("bins must be >= 2")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(left.to_numpy(), quantiles))
    if edges.size < 2:
        # A constant reference feature cannot drift in any measurable way.
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    reference_share = np.histogram(left.to_numpy(), bins=edges)[0] / len(left)
    current_share = np.histogram(right.to_numpy(), bins=edges)[0] / len(right)
    reference_share = np.clip(reference_share, EPSILON, None)
    current_share = np.clip(current_share, EPSILON, None)
    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def null_psi_thresholds(
    reference: pd.DataFrame, window_rows: int, thresholds: DriftThresholds
) -> pd.DataFrame:
    """How much PSI a *no-drift* window of the same length produces, per feature.

    Contiguous windows are drawn from inside the reference period and scored
    against the rest of it. Contiguous, not random rows: the whole reason the
    conventional thresholds fail here is serial correlation, and sampling rows
    independently would destroy exactly the property being calibrated for.

    Returns a frame indexed by feature with the warn and alert quantiles of that
    null distribution. An empty frame means the reference is too short to
    calibrate, and the caller falls back to the fixed thresholds.
    """
    usable = len(reference) - window_rows
    if window_rows < 2 or usable < window_rows:
        logger.warning(
            "reference has %d rows; too short to calibrate a %d-row window",
            len(reference),
            window_rows,
        )
        return pd.DataFrame()

    rng = np.random.default_rng(thresholds.calibration_seed)
    starts = rng.integers(0, usable, size=thresholds.calibration_samples)
    samples: dict[str, list[float]] = {str(c): [] for c in reference.columns}
    for start in starts:
        held = reference.iloc[int(start) : int(start) + window_rows]
        rest = reference.drop(held.index)
        if rest.empty:
            continue
        for column in reference.columns:
            samples[str(column)].append(
                population_stability_index(
                    rest[column], held[column], bins=thresholds.psi_bins
                )
            )

    rows = []
    limits: dict[str, float] = {}
    for column, values in samples.items():
        clean = [value for value in values if not np.isnan(value)]
        if not clean:
            continue
        alert = float(np.quantile(clean, thresholds.alert_quantile))
        limits[column] = alert
        rows.append(
            {
                "feature": column,
                "null_warn": float(np.quantile(clean, thresholds.warn_quantile)),
                "null_alert": alert,
                "null_median": float(np.median(clean)),
                "null_samples": len(clean),
            }
        )
    frame = pd.DataFrame(rows).set_index("feature")

    # How many features a no-drift window flags *at once*. Per-feature thresholds
    # alone are not enough: the features are strongly correlated with each other,
    # so flags arrive in clusters and "at least one feature alerted" is a
    # condition that fires on essentially every window. This is the joint,
    # multiple-testing-aware version of the same calibration, and it costs nothing
    # extra because the per-window values are already computed.
    per_window = [
        sum(
            1
            for column, values in samples.items()
            if column in limits
            and index < len(values)
            and not np.isnan(values[index])
            and values[index] >= limits[column]
        )
        for index in range(thresholds.calibration_samples)
    ]
    frame.attrs["null_alert_count"] = (
        float(np.quantile(per_window, thresholds.alert_quantile)) if per_window else 0.0
    )
    frame.attrs["null_alert_count_median"] = (
        float(np.median(per_window)) if per_window else 0.0
    )
    return frame


def feature_drift(
    features: pd.DataFrame,
    thresholds: DriftThresholds,
    *,
    training_end: pd.Timestamp,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """PSI per feature, training period vs the recent window.

    Thresholds are calibrated against no-drift windows of the same length unless
    `drift.calibration.enabled` is off or the reference is too short.
    """
    columns = ["feature", "psi", "null_warn", "null_alert", "excess", "level"]
    if features.empty:
        return pd.DataFrame(columns=columns)
    end = as_of or features.index.max()
    start = end - pd.Timedelta(days=thresholds.recent_days)

    reference = features.loc[features.index <= training_end]
    current = features.loc[(features.index > start) & (features.index <= end)]
    if reference.empty or current.empty:
        logger.warning(
            "cannot measure feature drift: reference rows=%d, recent rows=%d",
            len(reference),
            len(current),
        )
        return pd.DataFrame(columns=columns)

    null = (
        null_psi_thresholds(reference, len(current), thresholds)
        if thresholds.calibrate
        else pd.DataFrame()
    )
    if thresholds.calibrate and null.empty:
        logger.warning(
            "falling back to the fixed PSI thresholds; they are known to "
            "over-report on autocorrelated daily features"
        )

    rows = []
    for column in features.columns:
        name = str(column)
        psi = population_stability_index(
            reference[column], current[column], bins=thresholds.psi_bins
        )
        warn = float(null.loc[name, "null_warn"]) if name in null.index else float("nan")
        alert = float(null.loc[name, "null_alert"]) if name in null.index else float("nan")
        rows.append(
            {
                "feature": name,
                "psi": psi,
                "null_warn": warn,
                "null_alert": alert,
                # How far past a no-drift window this feature actually is. This is
                # the number to read: 1.0 means "exactly as different as a quiet
                # stretch of the training period already was".
                "excess": psi / alert if alert and not np.isnan(alert) else float("nan"),
                "level": thresholds.level_for(psi, warn=warn, alert=alert),
            }
        )
    frame = (
        pd.DataFrame(rows)
        .sort_values(["excess", "psi"], ascending=False)
        .reset_index(drop=True)
    )
    frame.attrs["reference_rows"] = len(reference)
    frame.attrs["recent_rows"] = len(current)
    frame.attrs["calibrated"] = not null.empty
    frame.attrs["null_alert_count"] = float(null.attrs.get("null_alert_count", 0.0))
    return frame


def performance_drift(
    production: pd.DataFrame,
    benchmark: pd.DataFrame,
    thresholds: DriftThresholds,
    *,
    metric_name: str = "pinball_mean",
) -> pd.DataFrame:
    """Realized production performance against its validation benchmark.

    Degradation is relative: ``production / benchmark - 1``. A horizon with fewer
    than ``min_realized_for_drift`` realized forecasts is reported but marked not
    decision-ready, because OPERATING_SPEC.md section 4 forbids acting on a tiny
    sample and a metric computed from three forecasts will swing wildly.
    """
    if production.empty or benchmark.empty:
        return pd.DataFrame()

    live = production[
        (production["metric_name"] == metric_name) & (production["regime"] == "all")
    ].set_index("horizon_days")
    reference = benchmark[
        (benchmark["metric_name"] == metric_name) & (benchmark["regime"] == "all")
    ]
    if live.empty or reference.empty:
        return pd.DataFrame()
    reference_by_horizon = reference.groupby("horizon_days")["metric_value"].mean()

    rows = []
    for horizon, row in live.iterrows():
        expected = reference_by_horizon.get(horizon)
        observed = row["metric_value"]
        if expected is None or pd.isna(expected) or pd.isna(observed) or expected == 0:
            continue
        degradation = float(observed) / float(expected) - 1.0
        sample = int(row["sample_size"])
        decision_ready = sample >= thresholds.min_realized_for_drift
        rows.append(
            {
                "horizon_days": int(horizon),
                "metric_name": metric_name,
                "validation": float(expected),
                "production": float(observed),
                "degradation": degradation,
                "sample_size": sample,
                "decision_ready": decision_ready,
                "level": (
                    LEVEL_ALERT
                    if decision_ready
                    and degradation > thresholds.performance_degradation_fraction
                    else LEVEL_STABLE
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("horizon_days").reset_index(drop=True)


def retraining_triggers(
    feature_frame: pd.DataFrame,
    performance_frame: pd.DataFrame,
    thresholds: DriftThresholds,
    *,
    new_observations: int,
    min_new_observations: int,
) -> pd.DataFrame:
    """The OPERATING_SPEC.md section 3 conditions, evaluated.

    Reported as a table rather than a boolean: "retrain" and "why" are different
    questions, and the weekly review needs the second one.
    """
    alerts = (
        feature_frame[feature_frame["level"] == LEVEL_ALERT]
        if not feature_frame.empty
        else pd.DataFrame()
    )
    # A no-drift window already flags this many features at once on this data, so
    # the trigger asks whether the current window flags *more* than that -- not
    # whether any single feature crossed a line.
    null_alerts = float(feature_frame.attrs.get("null_alert_count", 0.0))
    feature_fired = len(alerts) > null_alerts
    degraded = (
        performance_frame[performance_frame["level"] == LEVEL_ALERT]
        if not performance_frame.empty
        else pd.DataFrame()
    )
    return pd.DataFrame(
        [
            {
                "trigger": "feature_distribution_drift",
                "fired": feature_fired,
                "detail": (
                    f"{len(alerts)} of {len(feature_frame)} features flagged; "
                    f"a no-drift window of the same length flags {null_alerts:.0f}"
                    if bool(feature_frame.attrs.get("calibrated", False))
                    else (
                        f"{len(alerts)} of {len(feature_frame)} features past the "
                        f"fixed PSI level {thresholds.feature_psi_alert} "
                        "(uncalibrated; over-reports badly)"
                    )
                ),
            },
            {
                "trigger": "performance_degradation",
                "fired": not degraded.empty,
                "detail": (
                    f"{len(degraded)} horizons worse than validation by more than "
                    f"{thresholds.performance_degradation_fraction:.0%}"
                    if not degraded.empty
                    else "no horizon past the degradation threshold with enough sample"
                ),
            },
            {
                "trigger": "new_observations",
                "fired": new_observations >= min_new_observations,
                "detail": (
                    f"{new_observations} new daily candles since the training cutoff "
                    f"(threshold {min_new_observations})"
                ),
            },
        ]
    )
