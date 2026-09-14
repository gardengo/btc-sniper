"""Walk-forward fold generation and training-window selection.

VALIDATION_SPEC.md section 2 requires expanding- or rolling-window folds, and
section 9 requires comparing several window lengths. This module produces both
and makes the purge rule non-optional inside every fold.

The shape of a fold
-------------------
For a horizon ``h``, a fold is a pair of origin sets::

    train:      [window_start, validation_start - h - embargo)
    validation: [validation_start, validation_start + validation_days)

The gap of ``h + embargo`` days between them is not cosmetic. A training example
at origin ``t`` carries the label ``log(Close[t+h] / Close[t])``, so without the
gap the last training labels would read prices from inside the validation block.
This is the same rule that protects the outer test (VALIDATION_SPEC.md section
4.1), applied one level in.

Why folds are anchored to the end
---------------------------------
Folds are laid out backwards from `inner_validation_end`, so the most recent
validation block is always complete and always present. Laying them forward from
the start of the data leaves a ragged remainder at the end -- the most
market-relevant period -- and quietly changes which data every fold sees as soon
as one more day of history arrives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.validation.splits import (
    SplitBoundaries,
    SplitError,
    assert_no_test_contamination,
    independent_window_count,
)

logger = get_logger(__name__)

EXPANDING: str = "expanding"
ROLLING_PREFIX: str = "rolling_"

# MODEL_SPEC.md section 7: the window candidates to compare.
WINDOW_YEARS: dict[str, int] = {"rolling_4y": 4, "rolling_5y": 5, "rolling_8y": 8}


class FoldError(RuntimeError):
    """Raised when folds cannot be generated from the given configuration."""


def window_start(
    strategy: str, validation_start: pd.Timestamp, data_start: pd.Timestamp
) -> pd.Timestamp:
    """First training origin allowed by a training-window strategy.

    ``expanding`` uses every available day; a ``rolling_Ny`` window keeps the
    last ``N`` years before the fold. MODEL_SPEC.md section 7 is explicit that
    the four-year Bitcoin cycle must not be assumed when choosing between them --
    the choice is made empirically on inner validation, never a priori.
    """
    if strategy == EXPANDING:
        return data_start
    if strategy in WINDOW_YEARS:
        years = WINDOW_YEARS[strategy]
        return max(data_start, validation_start - pd.DateOffset(years=years))
    raise FoldError(
        f"unknown training window strategy {strategy!r}; "
        f"available: {[EXPANDING, *sorted(WINDOW_YEARS)]}"
    )


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold at one horizon."""

    index: int
    horizon_days: int
    strategy: str
    train_origins: pd.DatetimeIndex
    validation_origins: pd.DatetimeIndex

    @property
    def name(self) -> str:
        return f"fold{self.index:02d}"

    @property
    def training_cutoff(self) -> pd.Timestamp | None:
        return self.train_origins.max() if len(self.train_origins) else None

    def describe(self) -> str:
        train_range = (
            f"{self.train_origins.min():%Y-%m-%d}..{self.train_origins.max():%Y-%m-%d}"
            if len(self.train_origins)
            else "<empty>"
        )
        validation_range = (
            f"{self.validation_origins.min():%Y-%m-%d}.."
            f"{self.validation_origins.max():%Y-%m-%d}"
            if len(self.validation_origins)
            else "<empty>"
        )
        return (
            f"{self.name} h={self.horizon_days}d {self.strategy} "
            f"train[{len(self.train_origins)}]={train_range} "
            f"val[{len(self.validation_origins)}]={validation_range}"
        )

    def independent_windows(self, spacing_days: int = 1) -> float:
        return independent_window_count(
            len(self.validation_origins), self.horizon_days, spacing_days
        )


@dataclass(frozen=True)
class FoldPlan:
    """Fold layout parameters, resolved from config once and reused."""

    n_folds: int
    validation_days: int
    strategy: str
    min_train_origins: int

    @classmethod
    def from_config(cls, config: AppConfig, *, strategy: str | None = None) -> "FoldPlan":
        section = config.section("validation")
        walk_forward = dict(section.get("walk_forward", {}) or {})
        models = config.section("models")
        return cls(
            n_folds=int(walk_forward.get("n_folds", 5)),
            validation_days=int(walk_forward.get("validation_days", 365)),
            strategy=strategy or str(models.get("default_training_window", EXPANDING)),
            min_train_origins=int(walk_forward.get("min_train_origins", 250)),
        )

    def validate(self) -> None:
        if self.n_folds < 1:
            raise FoldError("validation.walk_forward.n_folds must be >= 1")
        if self.validation_days < 1:
            raise FoldError("validation.walk_forward.validation_days must be >= 1")
        if self.min_train_origins < 1:
            raise FoldError("validation.walk_forward.min_train_origins must be >= 1")


def build_folds(
    origins: pd.DatetimeIndex,
    boundaries: SplitBoundaries,
    plan: FoldPlan,
    horizon_days: int,
) -> list[Fold]:
    """Walk-forward folds for one horizon, newest fold last.

    Folds whose training set is smaller than ``plan.min_train_origins`` are
    dropped rather than trained on: a model fitted on a hundred overlapping rows
    produces a metric that looks real and is not.
    """
    plan.validate()
    if horizon_days < 1:
        raise FoldError("horizon_days must be >= 1")
    if len(origins) == 0:
        return []

    ordered = origins.sort_values()
    data_start = ordered.min()
    # Everything happens strictly inside the inner block; the outer test is never
    # a validation fold.
    inner = ordered[ordered <= boundaries.inner_validation_end]
    if len(inner) == 0:
        return []

    gap = pd.Timedelta(days=horizon_days + boundaries.embargo_days)
    validation_end = boundaries.inner_validation_end
    folds: list[Fold] = []

    for position in range(plan.n_folds):
        block_end = validation_end - pd.Timedelta(days=position * plan.validation_days)
        block_start = block_end - pd.Timedelta(days=plan.validation_days - 1)
        if block_start <= data_start:
            break

        validation_origins = inner[(inner >= block_start) & (inner <= block_end)]
        if len(validation_origins) == 0:
            continue

        train_limit = block_start - gap
        start = window_start(plan.strategy, block_start, data_start)
        train_origins = inner[(inner >= start) & (inner < train_limit)]
        if len(train_origins) < plan.min_train_origins:
            logger.debug(
                "dropping fold ending %s at h=%dd: only %d training origins",
                block_end.date(),
                horizon_days,
                len(train_origins),
            )
            continue

        folds.append(
            Fold(
                index=position,
                horizon_days=horizon_days,
                strategy=plan.strategy,
                train_origins=train_origins,
                validation_origins=validation_origins,
            )
        )

    folds.sort(key=lambda fold: fold.validation_origins.min())
    return [
        Fold(
            index=number,
            horizon_days=fold.horizon_days,
            strategy=fold.strategy,
            train_origins=fold.train_origins,
            validation_origins=fold.validation_origins,
        )
        for number, fold in enumerate(folds, start=1)
    ]


def assert_fold_is_clean(fold: Fold, boundaries: SplitBoundaries) -> None:
    """Fail if a fold's training labels reach into its validation block or the test.

    Called immediately before fitting. Two separate leaks are possible and both
    are silent: a training label landing in the validation block (which inflates
    the fold metric) and one landing in the outer test (which destroys the final
    evaluation).
    """
    assert_no_test_contamination(fold.train_origins, boundaries, fold.horizon_days)
    if len(fold.train_origins) == 0 or len(fold.validation_origins) == 0:
        return

    gap = pd.Timedelta(days=fold.horizon_days + boundaries.embargo_days)
    validation_start = fold.validation_origins.min()
    targets = fold.train_origins + pd.Timedelta(days=fold.horizon_days)
    offenders = fold.train_origins[targets >= validation_start]
    if len(offenders):
        raise SplitError(
            f"{fold.name}: {len(offenders)} training origins at horizon "
            f"{fold.horizon_days}d have labels inside the validation block starting "
            f"{validation_start:%Y-%m-%d} (required gap {gap.days}d); earliest "
            f"offender {offenders.min():%Y-%m-%d}"
        )


def iter_folds(
    origins: pd.DatetimeIndex,
    boundaries: SplitBoundaries,
    plan: FoldPlan,
    horizon_days: int,
) -> Iterator[Fold]:
    """Folds for a horizon, each checked for contamination before being yielded."""
    for fold in build_folds(origins, boundaries, plan, horizon_days):
        assert_fold_is_clean(fold, boundaries)
        yield fold


def describe_folds(folds: list[Fold], *, spacing_days: int = 1) -> pd.DataFrame:
    """Fold sizes as a table, for the validation report."""
    return pd.DataFrame(
        [
            {
                "fold": fold.name,
                "horizon_days": fold.horizon_days,
                "strategy": fold.strategy,
                "train_origins": len(fold.train_origins),
                "train_start": fold.train_origins.min().strftime("%Y-%m-%d")
                if len(fold.train_origins)
                else "",
                "train_end": fold.train_origins.max().strftime("%Y-%m-%d")
                if len(fold.train_origins)
                else "",
                "validation_origins": len(fold.validation_origins),
                "validation_start": fold.validation_origins.min().strftime("%Y-%m-%d"),
                "validation_end": fold.validation_origins.max().strftime("%Y-%m-%d"),
                "validation_independent_windows": round(
                    fold.independent_windows(spacing_days), 2
                ),
            }
            for fold in folds
        ]
    )
