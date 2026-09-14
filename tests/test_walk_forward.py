"""Walk-forward validation tests.

The run itself trains real models, so these use a tiny parameter set and few
horizons. What is being tested is the *protocol* -- which origins each fold sees,
that the model and baselines are compared on identical ones, and that the
aggregation cannot hide a bad fold -- not whether LightGBM is any good.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.baseline_eval import model_version as baseline_version
from src.evaluation.evaluator import ALL_REGIMES
from src.evaluation.walk_forward import (
    WalkForwardError,
    aggregate_across_folds,
    consistency_summary,
    fold_consistency,
    request_model_version,
    run_walk_forward,
)
from src.features.pipeline import build_features
from src.features.regime import compute_regime_labels
from src.models.baselines import REFERENCE_BASELINE
from src.models.training import TrainingRequest
from src.monitoring.validation_report import (
    identical_runs,
    render_comparison,
    render_markdown,
    verdict,
)
from src.utils.config import AppConfig
from src.validation.folds import EXPANDING, FoldPlan
from src.validation.splits import SplitBoundaries

LEVELS: tuple[float, ...] = (0.10, 0.50, 0.90)
TINY_PARAMS: dict[str, object] = {
    "n_estimators": 12,
    "num_leaves": 3,
    "learning_rate": 0.1,
    "min_child_samples": 40,
}


# The run fixture trains real models, so it is built once per module rather than
# once per test: every test below reads the same run and none of them mutate it.
@pytest.fixture(scope="module")
def market() -> pd.DataFrame:
    from tests.conftest import make_ohlcv

    return make_ohlcv()


@pytest.fixture(scope="module")
def config() -> AppConfig:
    from src.utils.config import load_config

    return load_config()


@pytest.fixture(scope="module")
def boundaries() -> SplitBoundaries:
    return SplitBoundaries(
        split_version="test",
        outer_test_start=pd.Timestamp("2020-01-01"),
        outer_test_end=None,
        inner_validation_end=pd.Timestamp("2019-12-31"),
        embargo_days=0,
        max_outer_test_evaluations=1,
        low_power_horizon_days=180,
    )


@pytest.fixture(scope="module")
def plan() -> FoldPlan:
    return FoldPlan(
        n_folds=2, validation_days=150, strategy=EXPANDING, min_train_origins=60
    )


@pytest.fixture(scope="module")
def request_() -> TrainingRequest:
    return TrainingRequest(
        horizons=(1, 7),
        levels=LEVELS,
        strategy=EXPANDING,
        algorithm="lightgbm_quantile",
        params=TINY_PARAMS,
        seed=42,
    )


@pytest.fixture(scope="module")
def run(
    market: pd.DataFrame,
    config: AppConfig,
    boundaries: SplitBoundaries,
    plan: FoldPlan,
    request_: TrainingRequest,
):
    features = build_features(market, config.features).usable_features()
    regimes = compute_regime_labels(market, config.features.regime)
    return run_walk_forward(
        features,
        market["close"],
        boundaries,
        config,
        request_,
        plan,
        regimes=regimes,
    )


class TestRunProtocol:
    def test_model_and_every_baseline_are_scored(
        self, run, config: AppConfig
    ) -> None:
        versions = set(run.metrics["model_version"])
        assert run.model_version in versions
        for name in config.baselines.enabled:
            assert baseline_version(name, config) in versions

    def test_model_and_baselines_share_the_same_sample(self, run) -> None:
        """Different origin sets would make the comparison meaningless."""
        sizes = run.metrics[
            (run.metrics["metric_name"] == "sample_size")
            & (run.metrics["regime"] == ALL_REGIMES)
        ]
        for _, group in sizes.groupby(["horizon_days", "fold"]):
            assert group["metric_value"].nunique() == 1

    def test_folds_never_reach_into_the_outer_test(
        self, run, boundaries: SplitBoundaries
    ) -> None:
        for row in run.folds.itertuples(index=False):
            target_end = pd.Timestamp(row.train_end) + pd.Timedelta(days=row.horizon_days)
            assert target_end < boundaries.outer_test_start
            assert pd.Timestamp(row.validation_end) <= boundaries.inner_validation_end

    def test_training_ends_before_its_own_validation_block(self, run) -> None:
        for row in run.folds.itertuples(index=False):
            gap = pd.Timestamp(row.validation_start) - pd.Timestamp(row.train_end)
            assert gap > pd.Timedelta(days=row.horizon_days)

    def test_fold_table_counts_the_origins_actually_scored(self, run) -> None:
        """Not the raw daily origins: spacing removes most of them at h=7."""
        for row in run.folds.itertuples(index=False):
            assert row.scored_origins <= row.validation_days
            if row.spacing_days > 1:
                assert row.scored_origins < row.validation_days

    def test_independent_windows_match_the_spacing(self, run) -> None:
        for row in run.folds.itertuples(index=False):
            expected = (
                row.scored_origins
                * min(row.spacing_days, row.horizon_days)
                / row.horizon_days
            )
            assert row.independent_windows == pytest.approx(expected, abs=0.01)

    def test_model_identity_is_not_a_deployable_version(self, run) -> None:
        """A fold model must not look like something that could be promoted."""
        assert run.model_version.startswith("wf-")

    def test_params_name_separates_candidates(
        self, request_: TrainingRequest, config: AppConfig
    ) -> None:
        plain = request_model_version(request_, config)
        tagged = request_model_version(request_, config, params_name="strong")
        assert plain != tagged
        assert tagged.endswith("-strong")

    def test_crossing_table_covers_every_fold(self, run) -> None:
        assert len(run.crossings) == len(run.folds)
        assert (run.crossings["crossing_rows"] <= run.crossings["rows"]).all()
        assert (run.crossings["crossing_pairs"] >= run.crossings["crossing_rows"]).all()

    def test_regime_rows_are_produced(self, run) -> None:
        assert (run.metrics["regime"] != ALL_REGIMES).any()

    def test_impossible_plan_raises_rather_than_returning_nothing(
        self,
        market: pd.DataFrame,
        config: AppConfig,
        boundaries: SplitBoundaries,
        request_: TrainingRequest,
    ) -> None:
        features = build_features(market, config.features).usable_features()
        impossible = FoldPlan(
            n_folds=1, validation_days=30, strategy=EXPANDING, min_train_origins=100_000
        )
        with pytest.raises(WalkForwardError):
            run_walk_forward(
                features,
                market["close"],
                boundaries,
                config,
                request_,
                impossible,
            )


class TestAggregation:
    def test_worst_fold_is_kept_beside_the_mean(self, run) -> None:
        aggregated = aggregate_across_folds(run.metrics, ("pinball_mean",))
        assert not aggregated.empty
        assert (aggregated["worst"] >= aggregated["mean"] - 1e-12).all()
        assert (aggregated["best"] <= aggregated["mean"] + 1e-12).all()

    def test_aggregation_counts_the_folds_it_used(self, run) -> None:
        aggregated = aggregate_across_folds(run.metrics, ("pinball_mean",))
        expected = run.metrics["fold"].nunique()
        assert aggregated["folds"].max() == expected

    def test_unknown_metric_yields_an_empty_frame(self, run) -> None:
        assert aggregate_across_folds(run.metrics, ("not_a_metric",)).empty

    def test_reference_has_a_perfect_win_rate_against_itself(
        self, run, config: AppConfig
    ) -> None:
        reference = baseline_version(REFERENCE_BASELINE, config)
        consistency = fold_consistency(
            run.metrics, "pinball_mean", reference, reference
        )
        assert not consistency.empty
        assert np.allclose(consistency["improvement"], 0.0)
        assert not consistency["beats_reference"].any()

    def test_consistency_summary_counts_wins_per_horizon(
        self, run, config: AppConfig
    ) -> None:
        reference = baseline_version(REFERENCE_BASELINE, config)
        consistency = fold_consistency(
            run.metrics, "pinball_mean", run.model_version, reference
        )
        summary = consistency_summary(consistency)
        assert set(summary["horizon_days"]) <= set(run.horizons)
        for row in summary.itertuples(index=False):
            assert 0 <= row.win_rate <= 1
            assert row.folds_won <= row.folds
            assert row.worst_improvement <= row.mean_improvement + 1e-12

    def test_missing_reference_yields_no_comparison(self, run) -> None:
        assert fold_consistency(
            run.metrics, "pinball_mean", run.model_version, "not-a-model"
        ).empty


class TestVerdict:
    def _summary(self, win_rate: float, improvement: float, low_power: bool = False):
        return pd.DataFrame(
            [
                {
                    "horizon_days": 1,
                    "folds": 4,
                    "folds_won": int(win_rate * 4),
                    "win_rate": win_rate,
                    "mean_improvement": improvement,
                    "worst_improvement": improvement,
                    "low_power": low_power,
                }
            ]
        )

    def test_verdict_is_produced_for_a_real_run(
        self, run, config: AppConfig
    ) -> None:
        summary = verdict(run, config)
        assert not summary.empty
        assert "verdict" in summary.columns

    def test_a_marginal_win_is_not_called_a_win(self, run, config: AppConfig) -> None:
        """One fold won out of four must never read as a success."""
        from src.monitoring.validation_report import _label

        assert _label(next(self._summary(0.25, -0.01).itertuples(index=False))) == (
            "does not beat baseline"
        )
        assert _label(next(self._summary(0.50, 0.001).itertuples(index=False))) == (
            "mixed - not consistent"
        )
        assert _label(next(self._summary(1.0, 0.05).itertuples(index=False))) == (
            "beats baseline in every fold"
        )

    def test_low_power_overrides_any_apparent_win(self) -> None:
        """VALIDATION_SPEC.md 4.3: a long horizon can never justify promotion."""
        from src.monitoring.validation_report import _label

        row = next(self._summary(1.0, 0.5, low_power=True).itertuples(index=False))
        assert _label(row) == "low power - not decisive"


class TestReports:
    def test_validation_report_covers_every_section(
        self, run, config: AppConfig
    ) -> None:
        text = render_markdown(run, config, focus_horizons=(1, 7))
        for heading in (
            "## 1. Verdict",
            "## 2. Fold layout",
            "## 3. Model vs baselines",
            "## 4. Per-fold detail",
            "## 5. Quantile crossing",
            "## 6. Regime breakdown",
        ):
            assert heading in text

    def test_comparison_report_takes_a_title(self, run, config: AppConfig) -> None:
        text = render_comparison(
            {"expanding": run}, config, title="Hyperparameter Comparison"
        )
        assert "# BTC Sniper - Hyperparameter Comparison" in text

    def test_identical_runs_are_grouped(self) -> None:
        frame = pd.DataFrame(
            {
                "horizon_days": [1, 7],
                "expanding": [0.1, 0.2],
                "rolling_4y": [0.11, 0.21],
                "rolling_5y": [0.1, 0.2],
            }
        )
        assert identical_runs(frame) == [["expanding", "rolling_5y"]]

    def test_distinct_runs_are_not_grouped(self) -> None:
        frame = pd.DataFrame(
            {"horizon_days": [1], "a": [0.1], "b": [0.2]}
        )
        assert identical_runs(frame) == []
