"""Assembling the (features, target) matrix a model is fitted on.

Everything that can leak does so at this seam: the target is the only forward-
looking object in the project, and this is where it meets the features. So the
checks live here rather than in the caller, and they run on every build rather
than in a test only.

What a build guarantees
-----------------------
1. ``X`` and ``y`` share exactly one index, aligned by forecast origin.
2. No target column is present in ``X`` (`targets.assert_disjoint`).
3. Every row has a resolved label -- rows whose target date has not arrived are
   dropped, never imputed.
4. The origin set was purged for this horizon before it got here; the build
   re-asserts it rather than trusting the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.models.targets import assert_disjoint, forward_log_return, target_column
from src.utils.logging import get_logger
from src.validation.splits import SplitBoundaries, assert_no_test_contamination

logger = get_logger(__name__)


class DatasetError(RuntimeError):
    """Raised when a training matrix cannot be built safely."""


@dataclass(frozen=True)
class TrainingMatrix:
    """Aligned features and labels for one horizon."""

    horizon_days: int
    features: pd.DataFrame
    target: pd.Series

    def __post_init__(self) -> None:
        if not self.features.index.equals(self.target.index):
            raise DatasetError("features and target are not aligned on the same index")

    @property
    def origins(self) -> pd.DatetimeIndex:
        return self.features.index

    @property
    def rows(self) -> int:
        return len(self.features)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(str(c) for c in self.features.columns)

    def describe(self) -> str:
        if self.rows == 0:
            return f"h={self.horizon_days}d rows=0 <empty>"
        return (
            f"h={self.horizon_days}d rows={self.rows} "
            f"features={self.features.shape[1]} "
            f"{self.origins.min():%Y-%m-%d}..{self.origins.max():%Y-%m-%d}"
        )


def build_training_matrix(
    features: pd.DataFrame,
    close: pd.Series,
    horizon_days: int,
    origins: pd.DatetimeIndex,
    *,
    boundaries: SplitBoundaries | None = None,
    require_labels: bool = True,
) -> TrainingMatrix:
    """Features and labels for ``origins`` at ``horizon_days``.

    ``boundaries`` re-asserts the outer-test purge; pass it for every fit.
    ``require_labels=False`` builds an inference matrix, where the label is
    expected to be missing because the target date is in the future.
    """
    if horizon_days < 1:
        raise DatasetError("horizon_days must be >= 1")

    target = forward_log_return(close, horizon_days)
    frame = features.reindex(origins)
    assert_disjoint(frame, target.to_frame())

    aligned_target = target.reindex(frame.index)
    usable = frame.dropna(axis=0, how="any").index
    if require_labels:
        usable = usable.intersection(aligned_target.dropna().index)

    if boundaries is not None and len(usable):
        assert_no_test_contamination(usable, boundaries, horizon_days)

    matrix = TrainingMatrix(
        horizon_days=horizon_days,
        features=frame.loc[usable],
        target=aligned_target.loc[usable].rename(target_column(horizon_days)),
    )
    dropped = len(origins) - matrix.rows
    if dropped:
        logger.debug(
            "h=%dd: dropped %d of %d origins (missing features or unresolved label)",
            horizon_days,
            dropped,
            len(origins),
        )
    return matrix


def build_inference_matrix(
    features: pd.DataFrame, origins: pd.DatetimeIndex
) -> pd.DataFrame:
    """Feature rows to predict from, with incomplete rows removed.

    No target is involved, so nothing here can leak; the only failure mode is
    predicting from a half-warm feature row, which this prevents.
    """
    frame = features.reindex(origins)
    complete = frame.dropna(axis=0, how="any")
    missing = len(frame) - len(complete)
    if missing:
        logger.warning(
            "%d of %d inference origins have incomplete features and were dropped",
            missing,
            len(frame),
        )
    return complete


def align_feature_columns(
    frame: pd.DataFrame, expected: tuple[str, ...]
) -> pd.DataFrame:
    """Reorder inference columns to the order the model was fitted on.

    Gradient-boosted trees index features positionally. A reordered or renamed
    column would not raise -- it would silently predict from the wrong feature,
    which is the worst possible failure because the output still looks like a
    forecast.
    """
    missing = [name for name in expected if name not in frame.columns]
    if missing:
        raise DatasetError(f"inference features are missing columns: {missing}")
    extra = [str(c) for c in frame.columns if str(c) not in set(expected)]
    if extra:
        logger.debug("ignoring %d feature columns the model was not fitted on", len(extra))
    return frame.loc[:, list(expected)]
