"""Model interfaces and the metadata every trained model must carry.

MODEL_SPEC.md section 10 lists what has to be recorded for a model to be
reproducible and auditable. Putting it in a frozen dataclass means a model
cannot reach the registry with half of it missing.

The abstraction is deliberately thin: fit a single quantile, predict a single
quantile. Multi-quantile and multi-horizon behaviour is composed on top in
`src/models/forecaster.py`, so adding a second algorithm means implementing one
small class rather than reproducing the composition logic.
"""

from __future__ import annotations

import importlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import pandas as pd

from src.utils.provenance import code_commit, library_versions

STATUS_CANDIDATE: str = "candidate"
STATUS_PRODUCTION: str = "production"
STATUS_RETIRED: str = "retired"
STATUS_REJECTED: str = "rejected"
VALID_STATUSES: tuple[str, ...] = (
    STATUS_CANDIDATE,
    STATUS_PRODUCTION,
    STATUS_RETIRED,
    STATUS_REJECTED,
)


class ModelError(RuntimeError):
    """Raised when a model cannot be fitted, used or described."""


def new_model_id() -> str:
    """Opaque unique id for one trained artifact."""
    return uuid.uuid4().hex


@dataclass(frozen=True)
class ModelMetadata:
    """Everything MODEL_SPEC.md section 10 requires a trained model to save."""

    model_id: str
    model_version: str
    algorithm: str
    feature_version: str
    horizon_grid_version: str
    training_cutoff: str
    status: str = STATUS_CANDIDATE
    config_version: str = ""
    code_commit: str = field(default_factory=code_commit)
    library_versions: Mapping[str, str] = field(default_factory=library_versions)
    training_window_strategy: str = ""
    training_start: str | None = None
    training_rows: int = 0
    # Per-horizon counts, because the purge costs more data at longer
    # horizons: `training_rows` alone would misreport every horizon but one.
    training_rows_by_horizon: Mapping[int, int] = field(default_factory=dict)
    horizons: tuple[int, ...] = ()
    quantiles: tuple[float, ...] = ()
    feature_names: tuple[str, ...] = ()
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    random_seed: int = 0
    validation_metrics: Mapping[str, Any] = field(default_factory=dict)
    test_metrics: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ModelError(
                f"invalid model status {self.status!r}; expected one of {VALID_STATUSES}"
            )
        if not self.model_version:
            raise ModelError("model_version must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        return (
            f"{self.model_version} ({self.algorithm}, {self.status}) "
            f"cutoff={self.training_cutoff} rows={self.training_rows} "
            f"window={self.training_window_strategy} commit={self.code_commit}"
        )


class QuantileRegressor(ABC):
    """Predicts one quantile of the target distribution.

    Implementations must be deterministic given the same seed, data and library
    versions: VALIDATION_SPEC.md section 11 requires a run to be reproducible,
    and `tests/test_models.py` asserts bitwise-identical refits.
    """

    algorithm: str

    @abstractmethod
    def fit(self, features: pd.DataFrame, target: pd.Series) -> "QuantileRegressor":
        """Fit on training rows only. Never sees validation or test data."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predicted quantile of the log return, indexed like ``features``."""

    @abstractmethod
    def to_text(self) -> str:
        """Serialise to a portable text form."""

    @classmethod
    @abstractmethod
    def from_text(cls, payload: str, level: float, params: Mapping[str, Any]) -> "QuantileRegressor":
        """Rebuild from :meth:`to_text`."""

    def feature_importance(self) -> pd.Series:
        """Per-feature importance; empty when the algorithm has no notion of it."""
        return pd.Series(dtype="float64")


ALGORITHM_REGISTRY: dict[str, type[QuantileRegressor]] = {}

# Implementations register themselves on import. Resolving a name imports its
# module on demand, so `src.models.targets` does not drag LightGBM in and a
# caller never has to remember an import purely for its side effect.
LAZY_IMPLEMENTATION_MODULES: dict[str, str] = {
    "lightgbm_quantile": "src.models.lightgbm_model",
}


def register_algorithm(cls: type[QuantileRegressor]) -> type[QuantileRegressor]:
    """Register an implementation under its ``algorithm`` name."""
    name = getattr(cls, "algorithm", "")
    if not name:
        raise ModelError(f"{cls.__name__} must define an `algorithm` name")
    ALGORITHM_REGISTRY[name] = cls
    return cls


def resolve_algorithm(name: str) -> type[QuantileRegressor]:
    """Look up an algorithm, importing its module if it is not registered yet."""
    if name not in ALGORITHM_REGISTRY and name in LAZY_IMPLEMENTATION_MODULES:
        importlib.import_module(LAZY_IMPLEMENTATION_MODULES[name])
    try:
        return ALGORITHM_REGISTRY[name]
    except KeyError as exc:
        known = sorted(set(ALGORITHM_REGISTRY) | set(LAZY_IMPLEMENTATION_MODULES))
        raise ModelError(f"unknown algorithm {name!r}; available: {known}") from exc
