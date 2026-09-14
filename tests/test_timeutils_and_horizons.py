"""UTC time handling and forecast horizon grid tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.forecast.horizons import build_horizon_grid, describe_grid, horizon_grid_from_config
from src.utils.config import AppConfig, HorizonGridConfig
from src.utils.timeutils import (
    MS_PER_DAY,
    UTC,
    date_to_ms,
    ensure_utc,
    last_closed_daily_open_ms,
    ms_to_date_str,
    parse_date,
    to_iso,
)


class TestTimeUtils:
    def test_date_to_ms_is_utc_midnight(self) -> None:
        assert date_to_ms("2020-03-12") == 1583971200000
        assert ms_to_date_str(1583971200000) == "2020-03-12"

    def test_round_trip_over_a_long_range(self) -> None:
        start = date_to_ms("2017-08-17")
        for offset in range(0, 3400, 37):
            ms = start + offset * MS_PER_DAY
            assert date_to_ms(ms_to_date_str(ms)) == ms

    def test_naive_datetime_is_treated_as_utc(self) -> None:
        naive = datetime(2024, 5, 1, 12, 0)
        assert ensure_utc(naive).tzinfo is UTC
        assert to_iso(naive) == "2024-05-01T12:00:00Z"

    def test_non_utc_datetime_is_converted(self) -> None:
        seoul = timezone(timedelta(hours=9))
        # 2024-05-02 08:00 KST is 2024-05-01 23:00 UTC, still the previous UTC day.
        aware = datetime(2024, 5, 2, 8, 0, tzinfo=seoul)
        assert to_iso(aware) == "2024-05-01T23:00:00Z"
        assert parse_date(aware).isoformat() == "2024-05-01"

    @pytest.mark.parametrize(
        "now_iso, expected",
        [
            ("2026-09-14T00:00:01+00:00", "2026-09-13"),
            ("2026-09-14T23:59:59+00:00", "2026-09-13"),
            ("2026-01-01T00:00:00+00:00", "2025-12-31"),
        ],
    )
    def test_last_closed_candle_excludes_today(self, now_iso: str, expected: str) -> None:
        """The in-progress UTC day must never be treated as closed."""
        now = datetime.fromisoformat(now_iso)
        assert ms_to_date_str(last_closed_daily_open_ms(now)) == expected

    def test_last_closed_candle_is_stable_within_a_day(self) -> None:
        morning = datetime(2026, 9, 14, 1, 0, tzinfo=UTC)
        evening = datetime(2026, 9, 14, 22, 0, tzinfo=UTC)
        assert last_closed_daily_open_ms(morning) == last_closed_daily_open_ms(evening)


class TestHorizonGrid:
    def test_configured_grid_contains_every_required_horizon(
        self, app_config: AppConfig
    ) -> None:
        grid = horizon_grid_from_config(app_config.forecast)
        for horizon in app_config.forecast.required_evaluation_horizons:
            assert horizon in grid, f"required evaluation horizon {horizon} missing"

    def test_grid_spans_exactly_one_year(self, app_config: AppConfig) -> None:
        grid = horizon_grid_from_config(app_config.forecast)
        assert grid[0] == 1
        assert grid[-1] == app_config.forecast.max_horizon_days == 365

    def test_grid_is_strictly_increasing_and_unique(self, app_config: AppConfig) -> None:
        grid = horizon_grid_from_config(app_config.forecast)
        assert list(grid) == sorted(set(grid))

    def test_band_spacing_matches_the_spec(self, app_config: AppConfig) -> None:
        """MODEL_SPEC.md section 2: 1d / 3d / 7d / 14d inside the four bands."""
        grid = horizon_grid_from_config(app_config.forecast)
        daily = [h for h in grid if h <= 30]
        assert daily == list(range(1, 31))

        def gaps(values: list[int]) -> set[int]:
            return {b - a for a, b in zip(values, values[1:])}

        assert gaps([h for h in grid if 31 <= h <= 90]) == {3}
        assert gaps([h for h in grid if 91 <= h <= 180]) == {7}
        assert gaps([h for h in grid if 181 <= h <= 365]) == {14}

    def test_band_endpoints_are_present(self, app_config: AppConfig) -> None:
        """The endpoint anchoring exists precisely so these survive."""
        grid = horizon_grid_from_config(app_config.forecast)
        for endpoint in (30, 90, 180, 365):
            assert endpoint in grid

    def test_grid_is_deterministic(self, app_config: AppConfig) -> None:
        first = horizon_grid_from_config(app_config.forecast)
        second = horizon_grid_from_config(app_config.forecast)
        assert first == second

    def test_missing_required_horizon_raises(self) -> None:
        grid = HorizonGridConfig(
            daily_until=30, every_n_days_31_90=3, every_n_days_91_180=7, every_n_days_181_365=14
        )
        with pytest.raises(ValueError, match="missing required evaluation horizons"):
            build_horizon_grid(grid, max_horizon_days=365, required=(200,))

    def test_shorter_max_horizon_is_respected(self) -> None:
        grid = HorizonGridConfig(
            daily_until=30, every_n_days_31_90=3, every_n_days_91_180=7, every_n_days_181_365=14
        )
        horizons = build_horizon_grid(grid, max_horizon_days=90, required=(1, 7, 30, 90))
        assert horizons[-1] == 90
        assert max(horizons) == 90

    def test_describe_grid_reports_size_and_gaps(self, app_config: AppConfig) -> None:
        grid = horizon_grid_from_config(app_config.forecast)
        description = describe_grid(grid)
        assert f"{len(grid)} horizons" in description
        assert "1d..365d" in description
