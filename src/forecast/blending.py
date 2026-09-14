"""Combining the model forecast with the baseline forecast, per horizon.

Why this module exists
----------------------
Walk-forward validation found that the tree model beats the `no_change` baseline
in every fold at h=1, ties at h=7, and is *worse* from h=30 outward, with the gap
widening (MODEL_SPEC.md section 6.5). Serving the model everywhere would mean
knowingly shipping a worse forecast at most horizons because it came out of a
model. That inverts the point of measuring anything.

Serving the model at h=1 and the baseline from h=30 with a hard switch is also
wrong, for a different reason: the forecast curve would jump at the boundary, and
a reader cannot tell a modelling artefact from a market view.

So the two are blended with a weight that falls from 1 to 0 across the range
where the evidence runs out.

Quantile averaging
------------------
The blend is a weighted average of the two quantile *functions*
(Vincentization), not of two densities::

    q_blend(p, h) = w(h) * q_model(p, h) + (1 - w(h)) * q_baseline(p, h)

A convex combination of monotone functions is monotone, so this cannot introduce
quantile crossing. It also degrades gracefully: at ``w=0`` it is exactly the
baseline, at ``w=1`` exactly the model.

The weights are frozen
----------------------
`forecast.blend` in `config.yaml` holds them, set from inner validation and not
re-tuned per forecast. A weight re-derived from recent performance would be a
model selected on data the forecast is then scored against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODE_LINEAR: str = "linear"
MODE_STEP: str = "step"
VALID_MODES: tuple[str, ...] = (MODE_LINEAR, MODE_STEP)


class BlendError(ValueError):
    """Raised when the blend is misconfigured or cannot be applied."""


@dataclass(frozen=True)
class BlendPolicy:
    """Per-horizon weight given to the model rather than the baseline."""

    full_model_horizon_days: int
    baseline_only_horizon_days: int
    mode: str = MODE_LINEAR
    enabled: bool = True

    @classmethod
    def from_config(cls, config: AppConfig) -> "BlendPolicy":
        section: Mapping[str, Any] = config.forecast_blend
        policy = cls(
            full_model_horizon_days=int(section.get("full_model_horizon_days", 1)),
            baseline_only_horizon_days=int(
                section.get("baseline_only_horizon_days", 30)
            ),
            mode=str(section.get("mode", MODE_LINEAR)),
            enabled=bool(section.get("enabled", True)),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise BlendError(f"forecast.blend.mode must be one of {VALID_MODES}")
        if self.full_model_horizon_days < 1:
            raise BlendError("forecast.blend.full_model_horizon_days must be >= 1")
        if self.baseline_only_horizon_days <= self.full_model_horizon_days:
            raise BlendError(
                "forecast.blend.baseline_only_horizon_days must be greater than "
                "full_model_horizon_days"
            )

    def weight(self, horizon_days: int) -> float:
        """Weight on the model at ``horizon_days``, in [0, 1]."""
        if not self.enabled:
            return 1.0
        if horizon_days <= self.full_model_horizon_days:
            return 1.0
        if horizon_days >= self.baseline_only_horizon_days:
            return 0.0
        if self.mode == MODE_STEP:
            return 0.0
        span = self.baseline_only_horizon_days - self.full_model_horizon_days
        travelled = horizon_days - self.full_model_horizon_days
        return float(1.0 - travelled / span)

    def describe(self) -> str:
        if not self.enabled:
            return "blending disabled: model only"
        return (
            f"model only to {self.full_model_horizon_days}d, "
            f"{self.mode} ramp, baseline only from "
            f"{self.baseline_only_horizon_days}d"
        )

    def weights_table(self, horizons: tuple[int, ...]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "horizon_days": list(horizons),
                "model_weight": [round(self.weight(h), 4) for h in horizons],
                "baseline_weight": [round(1.0 - self.weight(h), 4) for h in horizons],
            }
        )


def blend_quantiles(
    model: pd.DataFrame, baseline: pd.DataFrame, weight: float
) -> pd.DataFrame:
    """Weighted average of two quantile frames on the same index and levels."""
    if not 0.0 <= weight <= 1.0:
        raise BlendError(f"blend weight must be in [0, 1], got {weight}")
    if weight == 1.0:
        return model.copy()
    if weight == 0.0:
        return baseline.reindex(model.index).copy()

    aligned = baseline.reindex(index=model.index, columns=model.columns)
    missing = aligned.isna().all(axis=1)
    if missing.any():
        raise BlendError(
            f"{int(missing.sum())} origins have a model forecast but no baseline "
            "forecast; they cannot be blended"
        )
    return model * weight + aligned * (1.0 - weight)


def blend_source_label(weight: float) -> str:
    """Short provenance label recorded with each forecast point."""
    if weight >= 1.0:
        return "model"
    if weight <= 0.0:
        return "baseline"
    return "blend"


def to_prices(quantiles: pd.DataFrame, origin_close: float | pd.Series) -> pd.DataFrame:
    """Convert log-return quantiles to prices (MODEL_SPEC.md section 1)."""
    if isinstance(origin_close, pd.Series):
        close = origin_close.reindex(quantiles.index)
        return quantiles.apply(lambda column: close * np.exp(column))
    if origin_close <= 0:
        raise BlendError("origin close must be positive")
    return quantiles.apply(lambda column: float(origin_close) * np.exp(column))
