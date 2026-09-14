"""Forecast horizon grid construction.

MODEL_SPEC.md section 2 defines four horizon bands:

===============  ========
band             spacing
===============  ========
1..30 days       1 day
31..90 days      3 days
91..180 days     7 days
181..365 days    14 days
===============  ========

The band widths are not exact multiples of their spacing (``180 - 90 = 90`` is
not divisible by 7, ``365 - 180 = 185`` is not divisible by 14), so a naive
``range(start, end, step)`` would drop the band endpoints and the grid would
never contain 180 or 365 -- both of which VALIDATION_SPEC.md and the project
brief require as evaluation horizons.

Each band is therefore anchored on its **end** day and stepped backwards until
it would reach the previous band. That keeps the spacing inside a band exactly
as specified, always yields 30 / 90 / 180 / 365, and confines the irregularity
to a single shorter gap at each band boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.config import ForecastConfig, HorizonGridConfig


@dataclass(frozen=True)
class HorizonBand:
    """One contiguous horizon band with a fixed spacing."""

    name: str
    start_day: int
    end_day: int
    step_days: int

    def horizons(self, exclusive_lower_bound: int) -> list[int]:
        """Horizons in this band, anchored on ``end_day`` and stepped backwards."""
        if self.step_days < 1:
            raise ValueError(f"band '{self.name}' step_days must be >= 1")
        lower = max(exclusive_lower_bound, self.start_day - 1)
        values = []
        day = self.end_day
        while day > lower:
            values.append(day)
            day -= self.step_days
        return sorted(values)


def build_bands(grid: HorizonGridConfig, max_horizon_days: int) -> list[HorizonBand]:
    """Translate the configured grid into explicit bands."""
    daily_until = grid.daily_until
    return [
        HorizonBand("daily", 1, daily_until, 1),
        HorizonBand("every_3d", daily_until + 1, 90, grid.every_n_days_31_90),
        HorizonBand("every_7d", 91, 180, grid.every_n_days_91_180),
        HorizonBand("every_14d", 181, max_horizon_days, grid.every_n_days_181_365),
    ]


def build_horizon_grid(
    grid: HorizonGridConfig,
    max_horizon_days: int = 365,
    required: tuple[int, ...] = (),
) -> tuple[int, ...]:
    """Build the full ascending horizon grid in days.

    ``required`` horizons are asserted to be present; they are never silently
    injected, because a horizon that the model does not actually fit must not
    appear in the grid.
    """
    if max_horizon_days < 1:
        raise ValueError("max_horizon_days must be >= 1")

    horizons: list[int] = []
    previous_end = 0
    for band in build_bands(grid, max_horizon_days):
        if band.start_day > max_horizon_days:
            break
        capped = HorizonBand(
            band.name, band.start_day, min(band.end_day, max_horizon_days), band.step_days
        )
        band_horizons = capped.horizons(previous_end)
        horizons.extend(band_horizons)
        if band_horizons:
            previous_end = band_horizons[-1]

    result = tuple(sorted(set(horizons)))
    if not result:
        raise ValueError("horizon grid is empty")
    if result[-1] != max_horizon_days:
        raise ValueError(
            f"horizon grid must end at max_horizon_days={max_horizon_days}, got {result[-1]}"
        )
    missing = [h for h in required if h not in result]
    if missing:
        raise ValueError(f"horizon grid is missing required evaluation horizons: {missing}")
    return result


def horizon_grid_from_config(config: ForecastConfig) -> tuple[int, ...]:
    """Horizon grid for the configured forecast section."""
    return build_horizon_grid(
        config.horizon_grid,
        max_horizon_days=config.max_horizon_days,
        required=config.required_evaluation_horizons,
    )


def describe_grid(horizons: tuple[int, ...]) -> str:
    """Human-readable one-line summary used in logs and reports."""
    if not horizons:
        return "<empty>"
    gaps = {horizons[i + 1] - horizons[i] for i in range(len(horizons) - 1)}
    return (
        f"{len(horizons)} horizons, {horizons[0]}d..{horizons[-1]}d, "
        f"gaps={sorted(gaps)}"
    )
