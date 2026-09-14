"""Feature pipeline.

Assembles the enabled feature groups into one point-in-time feature matrix.

The pipeline enforces two invariants that the rest of the system depends on:

1. **Complete calendar.** Rolling windows here are row-based. If a daily candle
   were missing, a "30-day" window would silently span 31 calendar days, so a
   gap is an error rather than something to paper over.
2. **No future information.** Feature columns are derived only from OHLCV rows
   at or before the row being computed. Targets are built separately in the
   model layer; they never enter this matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from src.features.groups import ALL_GROUPS, GROUP_BUILDERS
from src.utils.config import FeaturesConfig
from src.utils.logging import get_logger
from src.utils.timeutils import utc_now_iso

logger = get_logger(__name__)

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class FeaturePipelineError(RuntimeError):
    """Raised when the input series cannot support point-in-time features."""


@dataclass(frozen=True)
class FeatureBuildResult:
    """Feature matrix plus the metadata needed to reproduce it."""

    features: pd.DataFrame
    feature_version: str
    groups: tuple[str, ...]
    built_at: str
    warmup_days: int
    rows_total: int
    rows_usable: int
    column_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def first_usable_date(self) -> str | None:
        usable = self.usable_features()
        return None if usable.empty else usable.index.min().strftime("%Y-%m-%d")

    def usable_features(self) -> pd.DataFrame:
        """Rows where every feature is present (warmup rows dropped)."""
        return self.features.dropna(axis=0, how="any")

    def describe(self) -> str:
        return (
            f"feature_version={self.feature_version} groups={list(self.groups)} "
            f"columns={self.features.shape[1]} rows={self.rows_total} "
            f"usable={self.rows_usable} first_usable={self.first_usable_date}"
        )


def _validate_input(frame: pd.DataFrame, *, require_complete_calendar: bool) -> None:
    if frame.empty:
        raise FeaturePipelineError("cannot build features from an empty frame")
    missing_columns = [c for c in REQUIRED_INPUT_COLUMNS if c not in frame.columns]
    if missing_columns:
        raise FeaturePipelineError(f"input frame is missing columns: {missing_columns}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise FeaturePipelineError("input frame must be indexed by a DatetimeIndex")
    if not frame.index.is_monotonic_increasing:
        raise FeaturePipelineError("input frame must be sorted by date ascending")
    if frame.index.has_duplicates:
        raise FeaturePipelineError("input frame contains duplicate dates")
    if require_complete_calendar:
        expected = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
        missing = expected.difference(frame.index)
        if len(missing):
            sample = [d.strftime("%Y-%m-%d") for d in missing[:10]]
            raise FeaturePipelineError(
                f"input frame has {len(missing)} missing daily rows "
                f"(row-based rolling windows would be wrong); first: {sample}"
            )


def resolve_groups(config: FeaturesConfig) -> tuple[str, ...]:
    """Configured groups, validated against the known builders."""
    requested = tuple(config.groups) or ALL_GROUPS
    unknown = [name for name in requested if name not in GROUP_BUILDERS]
    if unknown:
        raise FeaturePipelineError(f"unknown feature groups in config: {unknown}")
    return requested


def build_features(
    frame: pd.DataFrame,
    config: FeaturesConfig,
    *,
    groups: tuple[str, ...] | None = None,
    require_complete_calendar: bool = True,
) -> FeatureBuildResult:
    """Build the point-in-time feature matrix for an OHLCV frame."""
    _validate_input(frame, require_complete_calendar=require_complete_calendar)
    selected = groups if groups is not None else resolve_groups(config)

    params: dict[str, Any] = dict(config.params)
    column_groups: dict[str, tuple[str, ...]] = {}
    built: list[pd.DataFrame] = []

    for name in selected:
        builder = GROUP_BUILDERS[name]
        group_params = dict(config.regime) if name == "regime" else params
        group_frame = builder(frame, group_params)
        if group_frame.index.equals(frame.index) is False:
            raise FeaturePipelineError(f"group '{name}' changed the index alignment")
        column_groups[name] = tuple(group_frame.columns)
        built.append(group_frame)
        logger.debug("built %d %s features", group_frame.shape[1], name)

    features = pd.concat(built, axis=1)
    duplicated = features.columns[features.columns.duplicated()].tolist()
    if duplicated:
        raise FeaturePipelineError(f"duplicate feature names across groups: {duplicated}")

    usable = features.dropna(axis=0, how="any")
    result = FeatureBuildResult(
        features=features,
        feature_version=config.version,
        groups=tuple(selected),
        built_at=utc_now_iso(),
        warmup_days=config.min_warmup_days,
        rows_total=len(features),
        rows_usable=len(usable),
        column_groups=column_groups,
    )
    logger.info("features built: %s", result.describe())
    return result


def assert_no_lookahead(
    frame: pd.DataFrame,
    config: FeaturesConfig,
    *,
    cutoff: str | pd.Timestamp,
    groups: tuple[str, ...] | None = None,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Prove that features up to ``cutoff`` do not depend on data after it.

    Builds the matrix twice -- once on the full series, once on the series
    truncated at ``cutoff`` -- and compares the overlapping rows. Any difference
    means a feature reached forward in time.

    Returns the frame of mismatching cells (empty when the check passes).
    """
    cutoff_ts = pd.Timestamp(cutoff)
    full = build_features(frame, config, groups=groups).features.loc[:cutoff_ts]
    truncated = build_features(
        frame.loc[:cutoff_ts], config, groups=groups
    ).features.loc[:cutoff_ts]

    aligned_full, aligned_truncated = full.align(truncated, join="inner")
    difference = (aligned_full - aligned_truncated).abs()
    both_nan = aligned_full.isna() & aligned_truncated.isna()
    mismatched = (difference > tolerance) | (
        aligned_full.isna() ^ aligned_truncated.isna()
    )
    mismatched = mismatched & ~both_nan
    columns_with_issue = mismatched.any(axis=0)
    return mismatched.loc[mismatched.any(axis=1), columns_with_issue[columns_with_issue].index]
