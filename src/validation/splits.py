"""Outer-test isolation and purged training-origin selection.

This module makes the frozen split in `config.yaml` executable instead of
merely documented. It answers one question: **given a forecast horizon, which
origins is a model allowed to train on?**

The rule that matters
---------------------
A training example at origin ``t`` for horizon ``h`` has label
``log(Close[t+h] / Close[t])``. Its label therefore *reads a price at* ``t+h``.
If ``t + h`` falls inside the outer test block, the model has been trained on
the answer it is about to be tested on -- even though the feature row at ``t``
is entirely in the past.

So the eligibility rule is not "origin before the test start". It is::

    origin + horizon + embargo < outer_test_start

This purge is horizon-dependent: at h=1 it costs one day of training data, at
h=365 it costs a full year. `VALIDATION_SPEC.md` section 4.1 records the rule.

Everything here is pure and deterministic; it takes a DatetimeIndex of candidate
origins and returns subsets of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SplitError(RuntimeError):
    """Raised when a split is misconfigured or a contamination check fails."""


@dataclass(frozen=True)
class SplitBoundaries:
    """The frozen train / validation / outer-test boundaries."""

    split_version: str
    outer_test_start: pd.Timestamp
    outer_test_end: pd.Timestamp | None
    inner_validation_end: pd.Timestamp
    embargo_days: int
    max_outer_test_evaluations: int
    low_power_horizon_days: int

    @classmethod
    def from_config(cls, config: AppConfig) -> "SplitBoundaries":
        section: Mapping[str, Any] = config.section("validation")
        start = section.get("outer_test_start")
        if not start:
            raise SplitError("validation.outer_test_start is not configured")
        raw_end = section.get("outer_test_end")
        inner_end = section.get("inner_validation_end")
        boundaries = cls(
            split_version=str(section.get("split_version", "v1")),
            outer_test_start=pd.Timestamp(str(start)),
            outer_test_end=pd.Timestamp(str(raw_end)) if raw_end else None,
            inner_validation_end=pd.Timestamp(str(inner_end))
            if inner_end
            else pd.Timestamp(str(start)) - pd.Timedelta(days=1),
            embargo_days=int(section.get("embargo_days", 0) or 0),
            max_outer_test_evaluations=int(
                section.get("max_outer_test_evaluations_per_model_version", 1)
            ),
            low_power_horizon_days=int(section.get("low_power_horizon_days", 180)),
        )
        boundaries.validate()
        return boundaries

    def validate(self) -> None:
        if self.embargo_days < 0:
            raise SplitError("validation.embargo_days must be >= 0")
        if self.inner_validation_end >= self.outer_test_start:
            raise SplitError(
                f"inner_validation_end ({self.inner_validation_end:%Y-%m-%d}) must be "
                f"before outer_test_start ({self.outer_test_start:%Y-%m-%d})"
            )
        if self.outer_test_end is not None and self.outer_test_end <= self.outer_test_start:
            raise SplitError("outer_test_end must be after outer_test_start")

    def is_low_power(self, horizon_days: int) -> bool:
        """Whether this horizon is flagged as statistically under-powered."""
        return horizon_days >= self.low_power_horizon_days

    def resolved_test_end(self, data_end: pd.Timestamp) -> pd.Timestamp:
        """Outer test end, resolving a rolling (``null``) configuration."""
        if self.outer_test_end is None:
            return data_end
        return min(self.outer_test_end, data_end)


def last_eligible_training_origin(
    boundaries: SplitBoundaries, horizon_days: int
) -> pd.Timestamp:
    """Latest origin whose label stays strictly outside the outer test block."""
    if horizon_days < 1:
        raise SplitError("horizon_days must be >= 1")
    offset = horizon_days + boundaries.embargo_days + 1
    return boundaries.outer_test_start - pd.Timedelta(days=offset)


def eligible_training_origins(
    origins: pd.DatetimeIndex,
    boundaries: SplitBoundaries,
    horizon_days: int,
    *,
    cutoff: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """Origins a model may train on for ``horizon_days``.

    ``cutoff`` narrows the selection further (used by inner walk-forward folds,
    where each fold has its own training cutoff). The outer-test purge is always
    applied on top of it, never replaced by it.
    """
    limit = last_eligible_training_origin(boundaries, horizon_days)
    if cutoff is not None:
        fold_limit = cutoff - pd.Timedelta(days=horizon_days + boundaries.embargo_days)
        limit = min(limit, fold_limit)
    return origins[origins <= limit]


def outer_test_origins(
    origins: pd.DatetimeIndex,
    boundaries: SplitBoundaries,
    *,
    data_end: pd.Timestamp,
    horizon_days: int | None = None,
    fully_scoreable_only: bool = False,
) -> pd.DatetimeIndex:
    """Origins that belong to the outer test block.

    With ``fully_scoreable_only`` the result is restricted to origins whose
    ``horizon_days`` target date has already occurred, i.e. those that can reach
    ``fully_evaluated`` rather than staying ``pending``.
    """
    end = boundaries.resolved_test_end(data_end)
    selected = origins[(origins >= boundaries.outer_test_start) & (origins <= end)]
    if fully_scoreable_only:
        if horizon_days is None:
            raise SplitError("horizon_days is required when fully_scoreable_only is set")
        selected = selected[selected <= data_end - pd.Timedelta(days=horizon_days)]
    return selected


def assert_no_test_contamination(
    train_origins: pd.DatetimeIndex,
    boundaries: SplitBoundaries,
    horizon_days: int,
) -> None:
    """Fail loudly if any training label reaches into the outer test block.

    Call this immediately before fitting. It is the last line of defence for
    CLAUDE.md section 2.2, and it is cheap enough to run every time.
    """
    if len(train_origins) == 0:
        return
    target_dates = train_origins + pd.Timedelta(days=horizon_days)
    contaminated = train_origins[target_dates >= boundaries.outer_test_start]
    if len(contaminated):
        first = contaminated.min()
        raise SplitError(
            f"{len(contaminated)} training origins at horizon {horizon_days}d have "
            f"labels inside the outer test block starting "
            f"{boundaries.outer_test_start:%Y-%m-%d}; earliest offender "
            f"{first:%Y-%m-%d} targets {first + pd.Timedelta(days=horizon_days):%Y-%m-%d}"
        )


def independent_window_count(
    origin_count: int, horizon_days: int, spacing_days: int = 1
) -> float:
    """Roughly how many non-overlapping target windows a set of origins holds.

    Two origins ``s`` days apart at horizon ``h`` share ``h - s`` days of their
    target window, so the effective sample size is::

        origin_count * min(spacing_days, horizon_days) / horizon_days

    Once origins are spaced at least a horizon apart the windows are disjoint and
    this is just the origin count. The default ``spacing_days=1`` is the daily
    case, where consecutive origins overlap almost completely and the effective
    sample size collapses to ``origin_count / horizon_days`` -- the number that
    decides whether a horizon can support a verdict at all.
    """
    if horizon_days < 1:
        raise SplitError("horizon_days must be >= 1")
    if spacing_days < 1:
        raise SplitError("spacing_days must be >= 1")
    return origin_count * min(spacing_days, horizon_days) / horizon_days


def describe_split(
    origins: pd.DatetimeIndex,
    boundaries: SplitBoundaries,
    horizons: tuple[int, ...],
    *,
    data_end: pd.Timestamp,
) -> pd.DataFrame:
    """Per-horizon training/test sizes and effective sample sizes."""
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        train = eligible_training_origins(origins, boundaries, horizon)
        test_all = outer_test_origins(origins, boundaries, data_end=data_end)
        test_full = outer_test_origins(
            origins,
            boundaries,
            data_end=data_end,
            horizon_days=horizon,
            fully_scoreable_only=True,
        )
        rows.append(
            {
                "horizon_days": horizon,
                "train_origins": len(train),
                "train_last": train.max().strftime("%Y-%m-%d") if len(train) else "",
                "train_independent_windows": round(
                    independent_window_count(len(train), horizon), 1
                ),
                "test_origins": len(test_all),
                "test_scoreable_now": len(test_full),
                "test_independent_windows": round(
                    independent_window_count(len(test_full), horizon), 1
                ),
                "low_power": boundaries.is_low_power(horizon),
            }
        )
    return pd.DataFrame(rows)
