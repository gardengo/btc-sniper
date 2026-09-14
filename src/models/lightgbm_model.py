"""LightGBM quantile regression.

The primary candidate of MODEL_SPEC.md section 6. One booster per quantile,
trained with LightGBM's `quantile` objective so the interval is learned from the
target distribution rather than manufactured from fixed percentage bands
(MODEL_SPEC.md section 8).

Why trees first
---------------
CLAUDE.md section 6 forbids reaching for complexity before a simple model has
been shown insufficient. Gradient-boosted trees also need no scaler, which
removes a whole class of leakage: there is no fitted transform that could
accidentally be fitted across a fold boundary (CLAUDE.md section 2.1).

The native `lgb.train` API is used rather than the sklearn wrapper. The wrapper
would pull in scikit-learn purely to hold hyperparameters, and reloading a saved
model through it requires poking at private attributes. A `Booster` is the thing
LightGBM actually saves and loads.

Determinism
-----------
`deterministic`, `force_row_wise` and single-threading are forced on. Without
them LightGBM's histogram construction varies with thread scheduling, so two
runs of the same code on the same data produce different models and
VALIDATION_SPEC.md section 11 becomes unenforceable. The cost is some speed, on
a dataset small enough not to care.
"""

from __future__ import annotations

from typing import Any, Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.base import ModelError, QuantileRegressor, register_algorithm
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_ROUNDS: int = 300

# Settings the caller may not override: they are what makes a run reproducible.
FORCED_PARAMS: dict[str, Any] = {
    "objective": "quantile",
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
    "verbosity": -1,
}
# Number of boosting rounds is passed to `train()`, not through the param dict.
ROUND_KEYS: tuple[str, ...] = ("n_estimators", "num_iterations", "num_boost_round")


def resolve_params(
    params: Mapping[str, Any], *, level: float, seed: int
) -> tuple[dict[str, Any], int]:
    """Split configured settings into LightGBM params and a boosting-round count.

    Sklearn-style names (`min_child_samples`, `subsample`, `colsample_bytree`,
    `reg_lambda`, ...) are accepted: LightGBM's own parameter parser resolves
    them to their canonical names, so the config stays in the vocabulary most
    readers know.
    """
    if not 0.0 < level < 1.0:
        raise ModelError(f"quantile level must be in (0, 1), got {level}")

    resolved = {str(key): value for key, value in params.items()}
    rounds = DEFAULT_ROUNDS
    for key in ROUND_KEYS:
        if key in resolved:
            rounds = int(resolved.pop(key))
    if rounds < 1:
        raise ModelError("the number of boosting rounds must be >= 1")

    overridden = sorted(set(resolved) & set(FORCED_PARAMS))
    if overridden:
        logger.warning(
            "ignoring configured LightGBM params that would break reproducibility: %s",
            overridden,
        )
    resolved.update(FORCED_PARAMS)
    resolved["alpha"] = float(level)
    for seed_key in ("seed", "bagging_seed", "feature_fraction_seed", "data_random_seed"):
        resolved[seed_key] = int(seed)
    return resolved, rounds


@register_algorithm
class LightGBMQuantileRegressor(QuantileRegressor):
    """One LightGBM booster predicting one quantile of the future log return."""

    algorithm: str = "lightgbm_quantile"

    def __init__(
        self, level: float, params: Mapping[str, Any], *, seed: int = 42
    ) -> None:
        self.level = float(level)
        self.params = dict(params)
        self.seed = int(seed)
        self._booster: lgb.Booster | None = None
        self._feature_names: tuple[str, ...] = ()

    # ------------------------------------------------------------------ fitting

    def fit(
        self, features: pd.DataFrame, target: pd.Series
    ) -> "LightGBMQuantileRegressor":
        if features.empty:
            raise ModelError("cannot fit on an empty feature matrix")
        if not features.index.equals(target.index):
            raise ModelError("features and target must share an index")
        if target.isna().any():
            raise ModelError(
                "target contains NaN; unresolved labels must be dropped before fitting"
            )
        matrix = features.to_numpy(dtype="float64")
        if not np.isfinite(matrix).all():
            raise ModelError("feature matrix contains non-finite values")

        params, rounds = resolve_params(self.params, level=self.level, seed=self.seed)
        self._feature_names = tuple(str(column) for column in features.columns)
        dataset = lgb.Dataset(
            matrix,
            label=target.to_numpy(dtype="float64"),
            feature_name=list(self._feature_names),
            free_raw_data=False,
        )
        self._booster = lgb.train(params, dataset, num_boost_round=rounds)
        return self

    # --------------------------------------------------------------- prediction

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self._booster is None:
            raise ModelError("model is not fitted")
        if features.empty:
            return pd.Series(dtype="float64", index=features.index)
        if tuple(str(column) for column in features.columns) != self._feature_names:
            raise ModelError(
                "inference features do not match the fitted column order; "
                "use `dataset.align_feature_columns` first"
            )
        values = self._booster.predict(features.to_numpy(dtype="float64"))
        return pd.Series(np.asarray(values, dtype="float64"), index=features.index)

    def feature_importance(self) -> pd.Series:
        """Total split gain per feature, which is the interpretable one.

        Split *count* rewards features used for many cheap splits; gain reflects
        how much the objective actually improved.
        """
        if self._booster is None or not self._feature_names:
            return pd.Series(dtype="float64")
        gains = self._booster.feature_importance(importance_type="gain")
        return pd.Series(
            np.asarray(gains, dtype="float64"), index=list(self._feature_names)
        ).sort_values(ascending=False)

    # ------------------------------------------------------------ serialisation

    def to_text(self) -> str:
        """LightGBM's own text format.

        Chosen over pickle deliberately: a pickle is executable, ties the
        artifact to one Python version, and would make loading a stored model a
        code-execution decision. The text format is inspectable and portable.
        """
        if self._booster is None:
            raise ModelError("cannot serialise an unfitted model")
        return self._booster.model_to_string()

    @classmethod
    def from_text(
        cls, payload: str, level: float, params: Mapping[str, Any]
    ) -> "LightGBMQuantileRegressor":
        instance = cls(level, params)
        instance._booster = lgb.Booster(model_str=payload)
        instance._feature_names = tuple(instance._booster.feature_name())
        return instance

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def is_fitted(self) -> bool:
        return self._booster is not None
