"""Promotion gate, route selection and the single outer-test evaluation.

These are the tests TASKS.md Phase 10 calls "promotion gate tests". They matter
more than most: every other check in the project stops a bad number from being
computed, and this one stops a bad model from being served.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.final_test import (
    FinalTestError,
    assert_testable,
    evaluate_final_test,
    summarise,
)
# Aliased: pytest would collect the imported `test_record` as a test function.
from src.evaluation.final_test import test_record as build_test_record
from src.features.pipeline import build_features
from src.models import registry
from src.models.base import STATUS_CANDIDATE
from src.models.promotion import (
    DECISION_KEEP,
    DECISION_PROMOTE,
    DECISION_REJECT,
    ROUTE_BOOTSTRAP,
    ROUTE_CONFIGURATION,
    ROUTE_DATA_REFRESH,
    GateInputs,
    PromotionError,
    PromotionPolicy,
    assert_comparable_folds,
    classify_route,
    compare,
    compare_validation_records,
    configuration_diff,
    configuration_fingerprint,
    configuration_row,
    evaluate_gate,
    evaluated_designs,
    forecast_sanity,
    nominal_coverage,
    regime_comparison,
    validation_record,
)
from src.models.training import TrainingError, TrainingRequest, train_forecaster
from tests.conftest import make_ohlcv
from src.validation.splits import SplitBoundaries, post_test_training_origins

CANDIDATE = "cand"
INCUMBENT = "inc"
POLICY = PromotionPolicy()


@pytest.fixture(scope="module")
def long_ohlcv() -> pd.DataFrame:
    """Synthetic history long enough to reach past `outer_test_start`.

    The shared `ohlcv` fixture stops in 2020, so the outer-test block of the
    frozen split is empty on it and nothing here could be exercised at all.
    """
    return make_ohlcv(days=3_200, start="2017-08-17")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def registry_row(**overrides) -> dict:
    row = {
        "model_id": "id-1",
        "model_version": "lgbmq-expanding-20231230-v1-aaaaaa",
        "algorithm": "lightgbm_quantile",
        "training_window_strategy": "expanding",
        "feature_version": "v1",
        "horizon_grid_version": "v1",
        "random_seed": 42,
        "training_cutoff": "2023-12-30",
        "validation_metrics": "{}",
        "test_metrics": "{}",
        "hyperparameters": json.dumps({"params": {"num_leaves": 3}}),
    }
    row.update(overrides)
    return row


def metric_rows(
    version: str,
    *,
    horizon: int,
    fold: str,
    pinball: float,
    coverage: float = 0.95,
    regime: str = "all",
    sample_size: int = 100,
    low_power: bool = False,
) -> list[dict]:
    return [
        {
            "model_version": version,
            "horizon_days": horizon,
            "regime": regime,
            "metric_name": name,
            "metric_value": value,
            "sample_size": sample_size,
            "period_start": "2022-01-01",
            "period_end": "2022-12-31",
            "low_power": low_power,
            "fold": fold,
        }
        for name, value in (
            ("pinball_mean", pinball),
            ("coverage_95", coverage),
            ("sample_size", float(sample_size)),
        )
    ]


def metrics_frame(
    candidate_values: dict[int, list[float]],
    reference_values: dict[int, list[float]],
    *,
    candidate_coverage: float = 0.95,
    reference_coverage: float = 0.95,
    low_power: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Tidy metric rows for two model versions over matching folds."""
    rows: list[dict] = []
    for horizon, values in candidate_values.items():
        for index, value in enumerate(values, start=1):
            rows += metric_rows(
                CANDIDATE,
                horizon=horizon,
                fold=f"fold{index:02d}",
                pinball=value,
                coverage=candidate_coverage,
                low_power=horizon in low_power,
            )
    for horizon, values in reference_values.items():
        for index, value in enumerate(values, start=1):
            rows += metric_rows(
                INCUMBENT,
                horizon=horizon,
                fold=f"fold{index:02d}",
                pinball=value,
                coverage=reference_coverage,
                low_power=horizon in low_power,
            )
    return pd.DataFrame(rows)


WEIGHTS = {1: 1.0, 7: 0.79, 30: 0.0}


def comparison_of(
    candidate_values: dict[int, list[float]],
    reference_values: dict[int, list[float]],
    **kwargs,
) -> pd.DataFrame:
    return compare(
        metrics_frame(candidate_values, reference_values, **kwargs),
        candidate_version=CANDIDATE,
        reference_version=INCUMBENT,
        policy=POLICY,
        weights=kwargs.pop("weights", WEIGHTS),
    )


# ---------------------------------------------------------------------------
# identity and route
# ---------------------------------------------------------------------------


class TestConfigurationIdentity:
    def test_the_cutoff_is_not_part_of_the_identity(self):
        early = registry_row(training_cutoff="2023-12-30")
        late = registry_row(training_cutoff="2026-09-13")
        assert configuration_fingerprint(early) == configuration_fingerprint(late)

    def test_hyperparameters_are_part_of_the_identity(self):
        other = registry_row(hyperparameters=json.dumps({"params": {"num_leaves": 7}}))
        assert configuration_fingerprint(other) != configuration_fingerprint(registry_row())
        assert configuration_diff(other, registry_row()) == ["params"]

    @pytest.mark.parametrize(
        "field, value",
        [
            ("algorithm", "xgboost"),
            ("training_window_strategy", "rolling_4y"),
            ("feature_version", "v2"),
            ("horizon_grid_version", "v2"),
            ("random_seed", 7),
        ],
    )
    def test_every_configuration_field_changes_the_fingerprint(self, field, value):
        changed = registry_row(**{field: value})
        assert configuration_fingerprint(changed) != configuration_fingerprint(registry_row())

    def test_the_trained_horizon_set_is_not_part_of_the_identity(self):
        # `model_version_name` does not encode it either, so two models differing
        # only there would already collide on the registry's UNIQUE constraint.
        many = registry_row(
            hyperparameters=json.dumps({"params": {"num_leaves": 3}, "horizons": [1, 7]})
        )
        assert configuration_fingerprint(many) == configuration_fingerprint(registry_row())

    def test_a_configuration_can_be_fingerprinted_before_training(self, app_config):
        row = configuration_row(
            app_config,
            strategy="expanding",
            params=dict(app_config.section("models")["lightgbm"]),
            seed=int(app_config.section("models")["random_seed"]),
        )
        trained = registry_row(
            algorithm=app_config.section("models")["primary_algorithm"],
            feature_version=app_config.features.version,
            horizon_grid_version=app_config.forecast.horizon_grid_version,
            random_seed=int(app_config.section("models")["random_seed"]),
            hyperparameters=json.dumps(
                {"params": dict(app_config.section("models")["lightgbm"])}
            ),
        )
        assert configuration_fingerprint(row) == configuration_fingerprint(trained)


class TestRoute:
    def test_no_incumbent_is_bootstrap(self):
        route, _ = classify_route(registry_row(), None)
        assert route == ROUTE_BOOTSTRAP

    def test_same_configuration_later_cutoff_is_a_data_refresh(self):
        route, detail = classify_route(
            registry_row(training_cutoff="2026-09-13"), registry_row()
        )
        assert route == ROUTE_DATA_REFRESH
        assert "2023-12-30 -> 2026-09-13" in detail

    def test_different_hyperparameters_is_a_configuration_change(self):
        route, detail = classify_route(
            registry_row(hyperparameters=json.dumps({"params": {"num_leaves": 31}})),
            registry_row(),
        )
        assert route == ROUTE_CONFIGURATION
        assert "params" in detail


class TestEvaluatedDesigns:
    def test_only_rows_with_recorded_test_metrics_count(self):
        frame = pd.DataFrame(
            [
                registry_row(test_metrics="{}"),
                registry_row(test_metrics=""),
                registry_row(test_metrics=json.dumps({"model": {"1": 0.01}})),
            ]
        )
        assert evaluated_designs(frame) == {configuration_fingerprint(registry_row())}

    def test_an_empty_registry_has_evaluated_nothing(self):
        assert evaluated_designs(pd.DataFrame()) == set()


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


class TestCompare:
    def test_improvement_is_relative_to_the_reference(self):
        frame = comparison_of({1: [0.9, 0.9]}, {1: [1.0, 1.0]})
        row = frame.iloc[0]
        assert row["win_rate"] == 1.0
        assert row["mean_improvement"] == pytest.approx(0.10)
        assert row["candidate"] == pytest.approx(0.9)
        assert row["reference"] == pytest.approx(1.0)

    def test_an_unweighted_horizon_is_reported_but_not_decisive(self):
        frame = comparison_of({1: [0.9], 30: [2.0]}, {1: [1.0], 30: [1.0]})
        indexed = frame.set_index("horizon_days")
        assert bool(indexed.loc[1, "decision_horizon"]) is True
        assert bool(indexed.loc[30, "decision_horizon"]) is False

    def test_a_low_power_horizon_is_not_decisive_even_when_weighted(self):
        frame = compare(
            metrics_frame({7: [0.9]}, {7: [1.0]}, low_power=(7,)),
            candidate_version=CANDIDATE,
            reference_version=INCUMBENT,
            policy=POLICY,
            weights={7: 1.0},
        )
        assert bool(frame.iloc[0]["low_power"]) is True
        assert bool(frame.iloc[0]["decision_horizon"]) is False

    def test_coverage_is_penalised_by_distance_from_nominal_in_both_directions(self):
        # 0.99 against a nominal 0.95 is as miscalibrated as 0.91, just in the
        # direction that feels safe.
        over = comparison_of(
            {1: [1.0]}, {1: [1.0]}, candidate_coverage=0.99, reference_coverage=0.95
        )
        under = comparison_of(
            {1: [1.0]}, {1: [1.0]}, candidate_coverage=0.91, reference_coverage=0.95
        )
        assert over.iloc[0]["coverage_penalty"] == pytest.approx(0.04)
        assert under.iloc[0]["coverage_penalty"] == pytest.approx(0.04)

    def test_nominal_is_read_from_the_metric_name(self):
        assert nominal_coverage("coverage_95") == 0.95
        assert nominal_coverage("coverage_50") == 0.50
        with pytest.raises(PromotionError):
            nominal_coverage("coverage_all")

    def test_a_missing_reference_produces_no_comparison(self):
        frame = compare(
            metrics_frame({1: [0.9]}, {}),
            candidate_version=CANDIDATE,
            reference_version=INCUMBENT,
            policy=POLICY,
            weights=WEIGHTS,
        )
        assert frame.empty


class TestFoldComparability:
    def test_matching_layouts_are_accepted(self):
        layout = pd.DataFrame(
            [{"horizon_days": 1, "fold": "fold01", "validation_start": "2021-01-01"}]
        )
        assert_comparable_folds(layout, layout.copy())

    def test_folds_covering_different_periods_are_refused(self):
        left = pd.DataFrame(
            [{"horizon_days": 1, "fold": "fold01", "validation_start": "2021-01-01"}]
        )
        right = pd.DataFrame(
            [{"horizon_days": 1, "fold": "fold01", "validation_start": "2022-01-01"}]
        )
        with pytest.raises(PromotionError, match="different periods"):
            assert_comparable_folds(left, right)

    def test_runs_with_nothing_in_common_are_refused(self):
        left = pd.DataFrame(
            [{"horizon_days": 1, "fold": "fold01", "validation_start": "2021-01-01"}]
        )
        right = pd.DataFrame(
            [{"horizon_days": 7, "fold": "fold01", "validation_start": "2021-01-01"}]
        )
        with pytest.raises(PromotionError, match="share no"):
            assert_comparable_folds(left, right)


class TestRegimeComparison:
    def test_a_small_regime_cannot_veto(self):
        rows = metric_rows(
            CANDIDATE, horizon=1, fold="fold01", pinball=5.0, regime="bear",
            sample_size=3,
        ) + metric_rows(
            INCUMBENT, horizon=1, fold="fold01", pinball=1.0, regime="bear",
            sample_size=3,
        )
        frame = regime_comparison(
            pd.DataFrame(rows),
            candidate_version=CANDIDATE,
            reference_version=INCUMBENT,
            policy=POLICY,
            horizons=[1],
        )
        assert frame.iloc[0]["degradation"] == pytest.approx(4.0)
        assert bool(frame.iloc[0]["decisive"]) is False
        assert bool(frame.iloc[0]["catastrophic"]) is False

    def test_a_large_regime_flags_a_catastrophe(self):
        rows = metric_rows(
            CANDIDATE, horizon=1, fold="fold01", pinball=2.0, regime="bear",
            sample_size=100,
        ) + metric_rows(
            INCUMBENT, horizon=1, fold="fold01", pinball=1.0, regime="bear",
            sample_size=100,
        )
        frame = regime_comparison(
            pd.DataFrame(rows),
            candidate_version=CANDIDATE,
            reference_version=INCUMBENT,
            policy=POLICY,
            horizons=[1],
        )
        assert bool(frame.iloc[0]["catastrophic"]) is True


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def gate(route: str, comparison: pd.DataFrame, **kwargs):
    defaults = {
        "route": route,
        "candidate_version": CANDIDATE,
        "incumbent_version": None if route == ROUTE_BOOTSTRAP else INCUMBENT,
        "comparison": comparison,
        "reproducible": True,
    }
    defaults.update(kwargs)
    return evaluate_gate(GateInputs(**defaults), POLICY)


def check(decision, name: str):
    return next(item for item in decision.checks if item.name == name)


class TestBootstrapGate:
    def test_a_model_that_wins_where_it_is_served_is_promoted(self):
        decision = gate(
            ROUTE_BOOTSTRAP,
            comparison_of({1: [0.95, 0.96], 7: [0.999, 0.998]}, {1: [1.0, 1.0], 7: [1.0, 1.0]}),
        )
        assert decision.action == DECISION_PROMOTE
        assert check(decision, "earns_its_weight").passed

    def test_a_model_that_only_ties_earns_nothing(self):
        # Serving a tree that reproduces the baseline is strictly worse than
        # serving the baseline: same forecast, more moving parts.
        decision = gate(
            ROUTE_BOOTSTRAP,
            comparison_of({1: [1.0, 1.0], 7: [1.0, 1.0]}, {1: [1.0, 1.0], 7: [1.0, 1.0]}),
        )
        assert decision.action == DECISION_REJECT
        assert not check(decision, "earns_its_weight").passed

    def test_winning_on_average_but_not_in_every_fold_earns_nothing(self):
        decision = gate(
            ROUTE_BOOTSTRAP,
            comparison_of({1: [0.5, 1.4]}, {1: [1.0, 1.0]}),
        )
        assert decision.action == DECISION_REJECT
        assert "1/2 folds" in check(decision, "earns_its_weight").detail

    def test_damage_at_a_served_horizon_blocks_it(self):
        decision = gate(
            ROUTE_BOOTSTRAP,
            comparison_of({1: [0.9, 0.9], 7: [1.2, 1.2]}, {1: [1.0, 1.0], 7: [1.0, 1.0]}),
        )
        assert decision.action == DECISION_REJECT
        assert not check(decision, "does_no_damage").passed

    def test_a_disaster_at_an_unserved_horizon_does_not_block_it(self):
        # The served forecast at h=30 contains none of the model, so its number
        # cannot reject anything (VALIDATION_SPEC.md section 10).
        decision = gate(
            ROUTE_BOOTSTRAP,
            comparison_of(
                {1: [0.9, 0.9], 7: [1.0, 1.0], 30: [5.0, 5.0]},
                {1: [1.0, 1.0], 7: [1.0, 1.0], 30: [1.0, 1.0]},
            ),
        )
        assert decision.action == DECISION_PROMOTE
        assert check(decision, "no_catastrophic_horizon").passed

    def test_a_catastrophe_at_a_weighted_low_power_horizon_still_vetoes(self):
        comparison = compare(
            metrics_frame({1: [0.9, 0.9], 365: [3.0, 3.0]}, {1: [1.0, 1.0], 365: [1.0, 1.0]},
                          low_power=(365,)),
            candidate_version=CANDIDATE,
            reference_version=INCUMBENT,
            policy=POLICY,
            weights={1: 1.0, 365: 1.0},
        )
        decision = gate(ROUTE_BOOTSTRAP, comparison)
        assert decision.action == DECISION_REJECT
        assert not check(decision, "no_catastrophic_horizon").passed

    def test_materially_worse_coverage_blocks_it(self):
        decision = gate(
            ROUTE_BOOTSTRAP,
            comparison_of(
                {1: [0.9, 0.9]},
                {1: [1.0, 1.0]},
                candidate_coverage=0.60,
                reference_coverage=0.95,
            ),
        )
        assert decision.action == DECISION_REJECT
        assert not check(decision, "interval_coverage").passed

    def test_the_blend_weight_advisory_reports_without_vetoing(self):
        decision = gate(
            ROUTE_BOOTSTRAP,
            comparison_of({1: [0.9, 0.9], 7: [0.99, 1.01]}, {1: [1.0, 1.0], 7: [1.0, 1.0]}),
        )
        advisory = check(decision, "blend_weight_supported")
        assert advisory.blocking is False
        assert advisory.passed is False
        assert decision.action == DECISION_PROMOTE
        assert advisory in decision.advisories()


class TestConfigurationChangeGate:
    def test_beating_the_incumbent_everywhere_promotes(self):
        decision = gate(
            ROUTE_CONFIGURATION,
            comparison_of({1: [0.9, 0.9], 7: [0.9, 0.9]}, {1: [1.0, 1.0], 7: [1.0, 1.0]}),
        )
        assert decision.action == DECISION_PROMOTE

    def test_a_mixed_result_keeps_the_incumbent(self):
        decision = gate(
            ROUTE_CONFIGURATION,
            comparison_of({1: [0.9, 0.9], 7: [1.0, 1.0]}, {1: [1.0, 1.0], 7: [1.0, 1.0]}),
        )
        assert decision.action == DECISION_KEEP
        assert not check(decision, "beats_incumbent").passed

    def test_losing_one_fold_keeps_the_incumbent(self):
        decision = gate(
            ROUTE_CONFIGURATION,
            comparison_of({1: [0.5, 1.1], 7: [0.5, 1.1]}, {1: [1.0, 1.0], 7: [1.0, 1.0]}),
        )
        assert decision.action == DECISION_KEEP
        assert not check(decision, "stable_across_folds").passed

    def test_a_failed_gate_names_the_check_and_the_number(self):
        decision = gate(
            ROUTE_CONFIGURATION,
            comparison_of({1: [1.0, 1.0], 7: [1.0, 1.0]}, {1: [1.0, 1.0], 7: [1.0, 1.0]}),
        )
        assert "beats_incumbent" in decision.reason()
        assert "+0.0%" in decision.reason()

    def test_policy_switches_turn_checks_off(self):
        relaxed = PromotionPolicy(
            require_validation_improvement=False,
            require_stability_across_folds=False,
            require_interval_coverage_not_materially_worse=False,
        )
        comparison = comparison_of({1: [1.0, 1.0], 7: [1.0, 1.0]}, {1: [1.0, 1.0], 7: [1.0, 1.0]})
        decision = evaluate_gate(
            GateInputs(
                route=ROUTE_CONFIGURATION,
                candidate_version=CANDIDATE,
                incumbent_version=INCUMBENT,
                comparison=comparison,
                reproducible=True,
            ),
            relaxed,
        )
        # regime_stability is advisory when there are no regime-tagged origins.
        assert [item.name for item in decision.checks if item.blocking] == [
            "no_catastrophic_horizon",
            "reproducible",
        ]
        assert decision.action == DECISION_PROMOTE


class TestDataRefreshGate:
    def _inputs(self, **kwargs):
        defaults = {
            "route": ROUTE_DATA_REFRESH,
            "candidate_version": CANDIDATE,
            "incumbent_version": INCUMBENT,
            "reproducible": True,
            "cutoff_advance_days": 90,
            "new_observations": 90,
            "min_new_observations": 14,
            "equivalent": True,
            "equivalence_detail": "reproduced",
        }
        defaults.update(kwargs)
        return GateInputs(**defaults)

    def test_a_fresh_equivalent_refit_is_promoted(self):
        decision = evaluate_gate(self._inputs(), POLICY)
        assert decision.action == DECISION_PROMOTE

    def test_it_states_that_inner_validation_cannot_separate_the_two(self):
        note = check(evaluate_gate(self._inputs(), POLICY), "inner_validation_cannot_separate")
        assert note.blocking is False
        assert "frozen inner_validation_end" in note.detail

    def test_a_cutoff_that_goes_backwards_keeps_the_incumbent(self):
        decision = evaluate_gate(self._inputs(cutoff_advance_days=-5), POLICY)
        assert decision.action == DECISION_KEEP
        assert not check(decision, "cutoff_advances").passed

    def test_too_little_new_data_keeps_the_incumbent(self):
        decision = evaluate_gate(self._inputs(new_observations=3), POLICY)
        assert decision.action == DECISION_KEEP
        assert not check(decision, "enough_new_data").passed

    def test_a_changed_inner_block_result_keeps_the_incumbent(self):
        decision = evaluate_gate(
            self._inputs(equivalent=False, equivalence_detail="h=1d 0.007 -> 0.009"),
            POLICY,
        )
        assert decision.action == DECISION_KEEP
        assert not check(decision, "reproduces_incumbent_validation").passed

    def test_a_missing_incumbent_record_is_noted_not_failed(self):
        decision = evaluate_gate(self._inputs(equivalent=None), POLICY)
        assert decision.action == DECISION_PROMOTE
        assert check(decision, "reproduces_incumbent_validation").blocking is False

    def test_an_implausible_forecast_keeps_the_incumbent(self):
        decision = evaluate_gate(
            self._inputs(sanity_violations=("h=1d median move +3.10 exceeds",)), POLICY
        )
        assert decision.action == DECISION_KEEP
        assert not check(decision, "forecast_is_sane").passed


class TestAlwaysChecked:
    def test_an_unchecked_reproducibility_is_a_failure_not_a_pass(self):
        decision = gate(
            ROUTE_BOOTSTRAP,
            comparison_of({1: [0.9, 0.9], 7: [1.0, 1.0]}, {1: [1.0, 1.0], 7: [1.0, 1.0]}),
            reproducible=None,
        )
        assert decision.action == DECISION_REJECT
        assert not check(decision, "reproducible").passed

    def test_an_unknown_route_is_refused(self):
        with pytest.raises(PromotionError, match="unknown promotion route"):
            gate("whatever", pd.DataFrame())

    def test_a_promotion_records_which_checks_it_passed(self):
        decision = gate(
            ROUTE_BOOTSTRAP,
            comparison_of({1: [0.9, 0.9], 7: [1.0, 1.0]}, {1: [1.0, 1.0], 7: [1.0, 1.0]}),
        )
        assert decision.promote
        assert "earns_its_weight" in decision.reason()
        assert set(decision.to_frame()["check"]) == {c.name for c in decision.checks}


# ---------------------------------------------------------------------------
# sanity and the recorded validation fingerprint
# ---------------------------------------------------------------------------


class TestForecastSanity:
    @pytest.fixture
    def close(self) -> pd.Series:
        return make_ohlcv(days=500)["close"]

    def _predicted(self, values: list[float]) -> pd.DataFrame:
        return pd.DataFrame([values], index=[1], columns=[0.025, 0.5, 0.975])

    def test_an_ordinary_forecast_passes(self, close):
        violations, detail = forecast_sanity(self._predicted([-0.02, 0.0, 0.02]), close)
        assert violations == ()
        assert "1 horizons checked" in detail

    def test_a_move_larger_than_anything_ever_observed_is_flagged(self, close):
        violations, _ = forecast_sanity(self._predicted([0.9, 1.0, 1.1]), close)
        assert len(violations) >= 1
        assert any("exceeds the largest" in item for item in violations)

    def test_non_finite_quantiles_are_flagged(self, close):
        violations, _ = forecast_sanity(
            self._predicted([float("nan"), 0.0, 0.02]), close
        )
        assert any("non-finite" in item for item in violations)

    def test_out_of_order_quantiles_are_flagged(self, close):
        violations, _ = forecast_sanity(self._predicted([0.02, 0.0, -0.02]), close)
        assert any("out of order" in item for item in violations)

    def test_no_predictions_at_all_is_a_violation(self, close):
        violations, _ = forecast_sanity(pd.DataFrame(), close)
        assert violations == ("no predictions produced",)


class TestValidationRecord:
    def test_a_record_round_trips_through_json(self):
        comparison = comparison_of({1: [0.9, 0.9]}, {1: [1.0, 1.0]})
        record = validation_record(comparison, policy=POLICY, model_version="wf-x")
        restored = json.loads(json.dumps(record))
        assert restored["by_horizon"]["1"] == pytest.approx(0.9)
        assert restored["metric"] == "pinball_mean"

    def test_identical_records_reproduce(self):
        record = {"by_horizon": {"1": 0.0071, "7": 0.0211}}
        ok, detail = compare_validation_records(record, dict(record))
        assert ok is True
        assert "2 shared horizons" in detail

    def test_a_changed_number_on_a_frozen_block_is_a_failure(self):
        ok, detail = compare_validation_records(
            {"by_horizon": {"1": 0.0091}}, {"by_horizon": {"1": 0.0071}}
        )
        assert ok is False
        assert "0.007100 -> 0.009100" in detail

    def test_nothing_recorded_yet_is_neither_pass_nor_fail(self):
        assert compare_validation_records({"by_horizon": {"1": 0.1}}, {})[0] is None
        assert compare_validation_records(None, None)[0] is None

    def test_non_overlapping_horizons_cannot_be_compared(self):
        ok, detail = compare_validation_records(
            {"by_horizon": {"1": 0.1}}, {"by_horizon": {"7": 0.2}}
        )
        assert ok is None
        assert "do not overlap" in detail


# ---------------------------------------------------------------------------
# releasing the purge
# ---------------------------------------------------------------------------


class TestOuterTestRelease:
    def test_post_test_origins_are_limited_only_by_the_resolved_label(self):
        origins = pd.date_range("2024-01-01", periods=100, freq="D")
        kept = post_test_training_origins(
            origins, data_end=origins.max(), horizon_days=7
        )
        assert kept.max() == origins.max() - pd.Timedelta(days=7)
        assert kept.min() == origins.min()

    def test_releasing_without_a_data_end_is_refused(self, app_config):
        with pytest.raises(TrainingError, match="data_end"):
            TrainingRequest.from_config(
                app_config, horizons=(1,), release_outer_test=True
            )

    def test_a_released_model_trains_past_the_outer_test_start(
        self, long_ohlcv, app_config
    ):
        boundaries = SplitBoundaries.from_config(app_config)
        features = build_features(long_ohlcv, app_config.features).usable_features()
        close = long_ohlcv["close"]
        assert close.index.max() > boundaries.outer_test_start

        purged = train_forecaster(
            features,
            close,
            boundaries,
            app_config,
            TrainingRequest.from_config(app_config, horizons=(1,)),
        )
        released = train_forecaster(
            features,
            close,
            boundaries,
            app_config,
            TrainingRequest.from_config(
                app_config,
                horizons=(1,),
                release_outer_test=True,
                data_end=close.index.max(),
            ),
        )
        assert pd.Timestamp(purged.metadata.training_cutoff) < boundaries.outer_test_start
        assert pd.Timestamp(released.metadata.training_cutoff) > boundaries.outer_test_start
        assert released.metadata.training_rows > purged.metadata.training_rows
        # Same configuration, later cutoff: exactly the data_refresh route.
        assert classify_route(
            {
                "algorithm": released.metadata.algorithm,
                "training_window_strategy": released.metadata.training_window_strategy,
                "feature_version": released.metadata.feature_version,
                "horizon_grid_version": released.metadata.horizon_grid_version,
                "random_seed": released.metadata.random_seed,
                "training_cutoff": released.metadata.training_cutoff,
                "hyperparameters": json.dumps(
                    {"params": dict(released.metadata.hyperparameters)}
                ),
            },
            {
                "algorithm": purged.metadata.algorithm,
                "training_window_strategy": purged.metadata.training_window_strategy,
                "feature_version": purged.metadata.feature_version,
                "horizon_grid_version": purged.metadata.horizon_grid_version,
                "random_seed": purged.metadata.random_seed,
                "training_cutoff": purged.metadata.training_cutoff,
                "hyperparameters": json.dumps(
                    {"params": dict(purged.metadata.hyperparameters)}
                ),
            },
        )[0] == ROUTE_DATA_REFRESH


# ---------------------------------------------------------------------------
# the single outer-test evaluation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tested_model(long_ohlcv, app_config):
    boundaries = SplitBoundaries.from_config(app_config)
    features = build_features(long_ohlcv, app_config.features).usable_features()
    forecaster = train_forecaster(
        features,
        long_ohlcv["close"],
        boundaries,
        app_config,
        TrainingRequest.from_config(app_config, horizons=(1, 7)),
    )
    return forecaster, features, boundaries


@pytest.fixture(scope="module")
def final_result(tested_model, long_ohlcv, app_config):
    forecaster, features, boundaries = tested_model
    return evaluate_final_test(
        forecaster, features, long_ohlcv["close"], boundaries, app_config
    )


class TestFinalEvaluation:
    def test_a_model_trained_past_the_boundary_cannot_be_tested(self, tested_model):
        forecaster, _, boundaries = tested_model
        contaminated = forecaster.with_metadata(training_cutoff="2025-01-01")
        with pytest.raises(FinalTestError, match="already seen the answers"):
            assert_testable(contaminated, boundaries)

    def test_the_model_and_the_baselines_are_scored_on_the_same_origins(
        self, final_result
    ):
        sizes = final_result.metrics[
            (final_result.metrics["metric_name"] == "sample_size")
            & (final_result.metrics["regime"] == "all")
        ]
        for _, group in sizes.groupby("horizon_days"):
            assert group["metric_value"].nunique() == 1

    def test_every_scored_origin_lies_inside_the_test_block(
        self, final_result, tested_model, long_ohlcv
    ):
        _, _, boundaries = tested_model
        assert pd.Timestamp(final_result.test_start) >= boundaries.outer_test_start
        assert pd.Timestamp(final_result.test_end) <= long_ohlcv.index.max()
        assert final_result.coverage["scored_origins"].gt(0).all()

    def test_the_record_carries_the_window_it_was_measured_on(
        self, final_result, tested_model, app_config
    ):
        _, _, boundaries = tested_model
        record = build_test_record(final_result, app_config)
        assert record["test_start"] == final_result.test_start
        assert record["test_end"] == final_result.test_end
        assert record["split_version"] == boundaries.split_version
        assert set(record["model"]) == {str(h) for h in final_result.horizons}

    def test_the_summary_compares_against_the_baseline(self, final_result, app_config):
        table = summarise(final_result, app_config)
        assert set(table.columns) >= {"model", "baseline", "improvement", "coverage_95"}
        recomputed = 1.0 - table["model"] / table["baseline"]
        assert table["improvement"].to_numpy() == pytest.approx(recomputed.to_numpy())

    def test_a_second_outer_test_write_is_refused(
        self, connection, tested_model, final_result, app_config
    ):
        forecaster, _, _ = tested_model
        registry.register(connection, forecaster.metadata)
        record = build_test_record(final_result, app_config)
        version = forecaster.metadata.model_version
        registry.record_metrics(connection, version, test_metrics=record)
        with pytest.raises(registry.RegistryError, match="already has outer-test"):
            registry.record_metrics(connection, version, test_metrics=record)

    def test_a_recorded_evaluation_releases_that_design(
        self, connection, tested_model, final_result, app_config
    ):
        forecaster, _, _ = tested_model
        registry.register(connection, forecaster.metadata)
        version = forecaster.metadata.model_version
        assert evaluated_designs(registry.evaluated_models(connection)) == set()

        registry.record_metrics(
            connection, version, test_metrics=build_test_record(final_result, app_config)
        )
        released = evaluated_designs(registry.evaluated_models(connection))
        assert configuration_fingerprint(registry.get(connection, version)) in released

    def test_registration_alone_does_not_release_anything(
        self, connection, tested_model
    ):
        forecaster, _, _ = tested_model
        registry.register(connection, forecaster.metadata)
        assert registry.get(connection, forecaster.metadata.model_version)["status"] == (
            STATUS_CANDIDATE
        )
        assert evaluated_designs(registry.evaluated_models(connection)) == set()
