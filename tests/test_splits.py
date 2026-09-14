"""Outer-test isolation tests.

The purge rule these tests cover is the difference between an honest final test
and a meaningless one, so they check the arithmetic exactly rather than
approximately.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecast.horizons import horizon_grid_from_config
from src.utils.config import AppConfig
from src.validation.splits import (
    SplitBoundaries,
    SplitError,
    assert_no_test_contamination,
    describe_split,
    eligible_training_origins,
    independent_window_count,
    last_eligible_training_origin,
    outer_test_origins,
)

ORIGINS = pd.date_range("2018-08-16", "2026-09-13", freq="D")
DATA_END = pd.Timestamp("2026-09-13")


@pytest.fixture
def boundaries(app_config: AppConfig) -> SplitBoundaries:
    return SplitBoundaries.from_config(app_config)


class TestConfiguredSplit:
    def test_frozen_values_are_what_the_spec_records(
        self, boundaries: SplitBoundaries
    ) -> None:
        assert boundaries.split_version == "v1"
        assert boundaries.outer_test_start == pd.Timestamp("2024-01-01")
        assert boundaries.inner_validation_end == pd.Timestamp("2023-12-31")
        assert boundaries.embargo_days == 0

    def test_rolling_test_end_follows_the_data(self, boundaries: SplitBoundaries) -> None:
        assert boundaries.outer_test_end is None
        assert boundaries.resolved_test_end(DATA_END) == DATA_END

    def test_inner_end_after_test_start_is_rejected(self) -> None:
        with pytest.raises(SplitError, match="must be before"):
            SplitBoundaries(
                split_version="bad",
                outer_test_start=pd.Timestamp("2024-01-01"),
                outer_test_end=None,
                inner_validation_end=pd.Timestamp("2024-06-01"),
                embargo_days=0,
                max_outer_test_evaluations=1,
                low_power_horizon_days=180,
            ).validate()

    def test_negative_embargo_is_rejected(self) -> None:
        with pytest.raises(SplitError, match="embargo_days"):
            SplitBoundaries(
                split_version="bad",
                outer_test_start=pd.Timestamp("2024-01-01"),
                outer_test_end=None,
                inner_validation_end=pd.Timestamp("2023-12-31"),
                embargo_days=-1,
                max_outer_test_evaluations=1,
                low_power_horizon_days=180,
            ).validate()


class TestPurge:
    @pytest.mark.parametrize(
        "horizon, expected_last",
        [
            (1, "2023-12-30"),
            (7, "2023-12-24"),
            (30, "2023-12-01"),
            (90, "2023-10-02"),
            (365, "2022-12-31"),
        ],
    )
    def test_last_eligible_origin_is_exact(
        self, boundaries: SplitBoundaries, horizon: int, expected_last: str
    ) -> None:
        """origin + horizon must land strictly before the test start."""
        last = last_eligible_training_origin(boundaries, horizon)
        assert last.strftime("%Y-%m-%d") == expected_last
        assert last + pd.Timedelta(days=horizon) < boundaries.outer_test_start

    def test_purge_cost_grows_with_the_horizon(self, boundaries: SplitBoundaries) -> None:
        short = eligible_training_origins(ORIGINS, boundaries, 1)
        long = eligible_training_origins(ORIGINS, boundaries, 365)
        assert len(short) - len(long) == 364

    @pytest.mark.parametrize("horizon", [1, 7, 30, 90, 180, 365])
    def test_no_training_label_reaches_the_test_block(
        self, boundaries: SplitBoundaries, horizon: int
    ) -> None:
        train = eligible_training_origins(ORIGINS, boundaries, horizon)
        targets = train + pd.Timedelta(days=horizon)
        assert (targets < boundaries.outer_test_start).all()
        assert_no_test_contamination(train, boundaries, horizon)

    def test_embargo_adds_extra_distance(self) -> None:
        with_embargo = SplitBoundaries(
            split_version="v1",
            outer_test_start=pd.Timestamp("2024-01-01"),
            outer_test_end=None,
            inner_validation_end=pd.Timestamp("2023-12-31"),
            embargo_days=10,
            max_outer_test_evaluations=1,
            low_power_horizon_days=180,
        )
        assert last_eligible_training_origin(with_embargo, 30) == pd.Timestamp("2023-11-21")

    def test_fold_cutoff_narrows_but_never_widens(
        self, boundaries: SplitBoundaries
    ) -> None:
        """An inner fold cutoff must not be able to bypass the outer purge."""
        far_future = eligible_training_origins(
            ORIGINS, boundaries, 90, cutoff=pd.Timestamp("2030-01-01")
        )
        unrestricted = eligible_training_origins(ORIGINS, boundaries, 90)
        assert far_future.equals(unrestricted)

        earlier = eligible_training_origins(
            ORIGINS, boundaries, 90, cutoff=pd.Timestamp("2021-01-01")
        )
        assert earlier.max() == pd.Timestamp("2020-10-03")
        assert len(earlier) < len(unrestricted)


class TestContaminationGuard:
    def test_contaminated_origins_are_rejected(self, boundaries: SplitBoundaries) -> None:
        bad = pd.DatetimeIndex(["2023-12-20"])  # +30d lands in 2024
        with pytest.raises(SplitError, match="inside the outer test block"):
            assert_no_test_contamination(bad, boundaries, 30)

    def test_error_names_the_offending_dates(self, boundaries: SplitBoundaries) -> None:
        bad = pd.DatetimeIndex(["2023-12-20", "2023-12-25"])
        with pytest.raises(SplitError, match="2023-12-20"):
            assert_no_test_contamination(bad, boundaries, 30)

    def test_boundary_case_is_excluded(self, boundaries: SplitBoundaries) -> None:
        """An origin whose target lands exactly on the test start is contaminated."""
        exact = pd.DatetimeIndex(["2023-12-02"])  # +30d == 2024-01-01
        with pytest.raises(SplitError):
            assert_no_test_contamination(exact, boundaries, 30)

        safe = pd.DatetimeIndex(["2023-12-01"])  # +30d == 2023-12-31
        assert_no_test_contamination(safe, boundaries, 30)

    def test_empty_training_set_is_not_an_error(self, boundaries: SplitBoundaries) -> None:
        assert_no_test_contamination(pd.DatetimeIndex([]), boundaries, 365)


class TestOuterTestBlock:
    def test_test_origins_start_at_the_boundary(self, boundaries: SplitBoundaries) -> None:
        test = outer_test_origins(ORIGINS, boundaries, data_end=DATA_END)
        assert test.min() == pd.Timestamp("2024-01-01")
        assert test.max() == DATA_END

    def test_train_and_test_never_overlap(self, boundaries: SplitBoundaries) -> None:
        for horizon in (1, 30, 365):
            train = eligible_training_origins(ORIGINS, boundaries, horizon)
            test = outer_test_origins(ORIGINS, boundaries, data_end=DATA_END)
            assert train.intersection(test).empty

    def test_fully_scoreable_subset_shrinks_with_the_horizon(
        self, boundaries: SplitBoundaries
    ) -> None:
        short = outer_test_origins(
            ORIGINS, boundaries, data_end=DATA_END, horizon_days=1, fully_scoreable_only=True
        )
        long = outer_test_origins(
            ORIGINS, boundaries, data_end=DATA_END, horizon_days=365, fully_scoreable_only=True
        )
        assert len(short) > len(long)
        assert long.max() == pd.Timestamp("2025-09-13")

    def test_fully_scoreable_requires_a_horizon(self, boundaries: SplitBoundaries) -> None:
        with pytest.raises(SplitError, match="horizon_days is required"):
            outer_test_origins(
                ORIGINS, boundaries, data_end=DATA_END, fully_scoreable_only=True
            )


class TestPowerAccounting:
    def test_independent_windows_account_for_overlap(self) -> None:
        assert independent_window_count(365, 365) == pytest.approx(1.0)
        assert independent_window_count(1000, 1) == pytest.approx(1000.0)

    def test_long_horizons_are_flagged_low_power(self, boundaries: SplitBoundaries) -> None:
        assert not boundaries.is_low_power(90)
        assert boundaries.is_low_power(180)
        assert boundaries.is_low_power(365)

    def test_describe_split_covers_the_whole_grid(
        self, boundaries: SplitBoundaries, app_config: AppConfig
    ) -> None:
        horizons = horizon_grid_from_config(app_config.forecast)
        table = describe_split(ORIGINS, boundaries, horizons, data_end=DATA_END)
        assert len(table) == len(horizons)
        assert (table["train_origins"] > 0).all()
        # Training set must shrink monotonically as the purge grows.
        assert table["train_origins"].is_monotonic_decreasing
