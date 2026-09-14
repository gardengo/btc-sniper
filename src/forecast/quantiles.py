"""Quantile labelling and prediction-interval helpers.

MODEL_SPEC.md section 3 fixes both the seven required quantiles and the three
intervals derived from them. The labels defined here are what
`forecast_quantiles.quantile_label` stores, so they must stay stable: a renamed
label silently orphans every forecast already on disk.
"""

from __future__ import annotations

import pandas as pd


class QuantileError(ValueError):
    """Raised for an unusable quantile level or interval request."""


def quantile_label(level: float) -> str:
    """Storage label for a quantile level (``0.025`` -> ``q02_5``)."""
    if not 0.0 < level < 1.0:
        raise QuantileError(f"quantile level must be in (0, 1), got {level}")
    percent = level * 100.0
    rounded = round(percent, 1)
    if abs(rounded - round(rounded)) < 1e-9:
        return f"q{int(round(rounded)):02d}"
    return f"q{rounded:04.1f}".replace(".", "_")


def quantile_labels(levels: tuple[float, ...]) -> tuple[str, ...]:
    """Labels for a list of levels, rejecting duplicates after labelling."""
    labels = tuple(quantile_label(level) for level in levels)
    if len(set(labels)) != len(labels):
        raise QuantileError(f"quantile levels collide after labelling: {labels}")
    return labels


def interval_levels(interval: float) -> tuple[float, float]:
    """Lower/upper quantile levels of a central interval (``0.80`` -> 0.10, 0.90)."""
    if not 0.0 < interval < 1.0:
        raise QuantileError(f"interval must be in (0, 1), got {interval}")
    tail = (1.0 - interval) / 2.0
    return round(tail, 10), round(1.0 - tail, 10)


def resolve_interval(
    interval: float, available: tuple[float, ...], *, tolerance: float = 1e-9
) -> tuple[float, float]:
    """Map an interval onto two configured quantile levels.

    Raises rather than approximating: an 80% band drawn from the 25/75 quantiles
    would be labelled 80% on the chart while covering 50% of the distribution,
    which is worse than not drawing it.
    """
    lower, upper = interval_levels(interval)
    resolved = []
    for wanted in (lower, upper):
        match = [level for level in available if abs(level - wanted) <= tolerance]
        if not match:
            raise QuantileError(
                f"interval {interval} needs quantile {wanted}, which is not in "
                f"forecast.quantiles={list(available)}"
            )
        resolved.append(match[0])
    return resolved[0], resolved[1]


def count_crossings(frame: pd.DataFrame, levels: tuple[float, ...]) -> int:
    """Total adjacent-level violations, i.e. a higher quantile predicting less.

    Counts *pairs*, not rows: one row can violate several adjacent pairs at
    once, so this can exceed the row count. Use :func:`count_crossing_rows` for
    a rate. MODEL_SPEC.md section 3 allows enforcing order as post-processing
    but requires the frequency to be measured, so this counts instead of raising.
    """
    ordered = sorted(levels)
    crossings = 0
    for low, high in zip(ordered, ordered[1:]):
        crossings += int((frame[high] < frame[low]).sum())
    return crossings


def count_crossing_rows(frame: pd.DataFrame, levels: tuple[float, ...]) -> int:
    """Rows containing at least one crossing -- the number a rate divides by."""
    ordered = sorted(levels)
    if frame.empty or len(ordered) < 2:
        return 0
    affected = pd.Series(False, index=frame.index)
    for low, high in zip(ordered, ordered[1:]):
        affected |= frame[high] < frame[low]
    return int(affected.sum())


def enforce_ordering(frame: pd.DataFrame, levels: tuple[float, ...]) -> pd.DataFrame:
    """Sort each row's quantiles ascending, repairing any crossing."""
    ordered = sorted(levels)
    values = frame[ordered].to_numpy(copy=True)
    values.sort(axis=1)
    repaired = frame.copy()
    repaired[ordered] = values
    return repaired
