"""The weekly model review: decide whether production should change.

Usage::

    python -m jobs.weekly_model_review                 # review, report, change nothing
    python -m jobs.weekly_model_review --apply         # apply the decision
    python -m jobs.weekly_model_review --dry-run       # no candidate is registered
    python -m jobs.weekly_model_review --candidate <v> # review a candidate already registered
    python -m jobs.weekly_model_review --force         # build a candidate even with no trigger

Implements the OPERATING_SPEC.md section 2 sequence:

    data quality -> performance update -> drift review -> retraining necessity
    -> candidate training -> inner validation -> compare -> promote or reject

Three deliberate boundaries
---------------------------
**Building a candidate is not promoting one.** The default run trains and
registers a candidate and writes the decision to a report; it does not touch the
production model. Applying the decision needs `--apply`, and that is the only
mode that writes to `model_registry.status` (CLAUDE.md section 2.3).

**A rejection is recorded, not discarded.** `registry.reject` requires a reason
and the gate produces one, so a month later the record says which check failed
and at which number.

**Nothing here reads the outer test.** Walk-forward runs are inner-block only,
production metrics come from realized forecasts, and the data-quality step reads
the stored candles without writing a second set of check rows -- the daily
`jobs.data_quality_report` owns those.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.data.binance import SOURCE as BINANCE_SOURCE
from src.data.validation import validate_ohlcv
from src.evaluation.baseline_eval import model_version as baseline_version
from src.evaluation.walk_forward import WalkForwardResult, run_walk_forward
from src.features.pipeline import build_features
from src.features.regime import compute_regime_labels
from src.forecast.blending import BlendPolicy
from src.forecast.horizons import horizon_grid_from_config
from src.models import registry
from src.models.baselines import REFERENCE_BASELINE
from src.models.promotion import (
    ROUTE_BOOTSTRAP,
    ROUTE_DATA_REFRESH,
    GateInputs,
    PromotionPolicy,
    assert_comparable_folds,
    classify_route,
    compare,
    compare_validation_records,
    configuration_fingerprint,
    configuration_row,
    evaluate_gate,
    evaluated_designs,
    forecast_sanity,
    regime_comparison,
    validation_record,
)
from src.models.training import TrainingRequest, assert_reproducible, train_forecaster
from src.monitoring.drift import (
    DriftThresholds,
    feature_drift,
    performance_drift,
    retraining_triggers,
)
from src.monitoring.realization import (
    drop_in_sample,
    horizons_with_enough_evidence,
    load_and_realize,
    production_metrics,
    summarise,
)
from src.monitoring.review_report import ReviewContext, write_report
from src.storage import repositories as repo
from src.storage.db import open_connection, transaction
from src.utils.config import AppConfig, load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.provenance import code_commit, describe_environment
from src.validation.folds import FoldPlan
from src.validation.splits import SplitBoundaries

logger = get_logger("jobs.weekly_model_review")

REPORT_NAME: str = "weekly_model_review.md"
CANDIDATE_TAG: str = "candidate"
INCUMBENT_TAG: str = "incumbent"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly model review.")
    parser.add_argument(
        "--horizons",
        default="required",
        help="'required' (default), 'all', or a comma-separated list of days",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="apply the decision: promote or reject the candidate in the registry",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="review without registering a candidate or writing anything",
    )
    parser.add_argument(
        "--candidate",
        default=None,
        help="review this already-registered candidate instead of training a new one",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="build a candidate even when no retraining trigger fired",
    )
    parser.add_argument("--window", default=None, help="training window strategy")
    parser.add_argument("--output", default=None, help="report path")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


def resolve_horizons(spec: str, config: AppConfig) -> tuple[int, ...]:
    if spec == "required":
        return config.forecast.required_evaluation_horizons
    grid = horizon_grid_from_config(config.forecast)
    if spec == "all":
        return grid
    wanted = tuple(sorted({int(part) for part in spec.split(",") if part.strip()}))
    unknown = [horizon for horizon in wanted if horizon not in grid]
    if unknown:
        raise SystemExit(f"horizons not in the configured grid: {unknown}")
    return wanted


def _row_params(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    hyperparameters = json.loads(row["hyperparameters"] or "{}")
    return dict(hyperparameters.get("params", {}))


def _stored_validation(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    return json.loads(row["validation_metrics"] or "{}")


def _sanity_frame(forecaster, features: pd.DataFrame) -> pd.DataFrame:
    """The candidate's live quantiles at the newest usable origin, by horizon."""
    if features.empty:
        return pd.DataFrame()
    latest = features.iloc[[-1]]
    rows: dict[int, pd.Series] = {}
    for horizon in forecaster.horizons:
        predicted, _ = forecaster.predict_horizon(latest, horizon)
        rows[int(horizon)] = predicted.iloc[0]
    return pd.DataFrame(rows).T.sort_index()


def _measure_production(
    connection, config: AppConfig, frame: pd.DataFrame, regimes: pd.DataFrame
) -> tuple[Any, pd.DataFrame, int]:
    """Realize stored forecasts read-only; the daily job owns persistence."""
    market = config.market.primary
    rows, versions = load_and_realize(
        connection,
        frame["close"],
        source=BINANCE_SOURCE,
        symbol=market.symbol,
        timeframe=market.timeframe,
        regimes=regimes,
    )
    if rows.empty:
        return None, pd.DataFrame(), 0

    points = repo.load_forecast_points_for_realization(
        connection,
        source=BINANCE_SOURCE,
        symbol=market.symbol,
        timeframe=market.timeframe,
        model_version=None,
    )
    origins = points.drop_duplicates("forecast_id").set_index("forecast_id")[
        "forecast_origin_date"
    ]
    cutoffs: dict[str, str] = {}
    cache: dict[str, str | None] = {}
    for forecast_id, version in versions.items():
        if version not in cache:
            row = registry.get(connection, version)
            cache[version] = row["training_cutoff"] if row else None
        if cache[version]:
            cutoffs[forecast_id] = cache[version]

    out_of_sample, excluded = drop_in_sample(rows, origins, cutoffs)
    regime_labels = tuple(
        str(label) for label in config.section("validation").get("regimes", [])
    )
    production = production_metrics(
        out_of_sample, versions, config, regimes=regime_labels
    )
    return summarise(rows), production, excluded


def _benchmark(connection, strategy: str | None) -> pd.DataFrame:
    metrics = repo.load_performance_metrics(connection, scope="validation")
    if metrics.empty or strategy is None:
        return metrics
    return metrics[metrics["model_version"].str.contains(strategy, regex=False)]


def _walk_forward(
    features: pd.DataFrame,
    close: pd.Series,
    boundaries: SplitBoundaries,
    config: AppConfig,
    regimes: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    strategy: str,
    params: Mapping[str, Any] | None,
    tag: str,
    include_baselines: bool,
) -> WalkForwardResult:
    request = TrainingRequest.from_config(
        config, horizons=horizons, strategy=strategy, params=params or None
    )
    plan = FoldPlan.from_config(config, strategy=strategy)
    logger.info("inner validation for the %s configuration (%s)", tag, plan)
    return run_walk_forward(
        features,
        close,
        boundaries,
        config,
        request,
        plan,
        regimes=regimes,
        include_baselines=include_baselines,
        params_name=tag,
    )


def run(config: AppConfig, args: argparse.Namespace) -> int:
    boundaries = SplitBoundaries.from_config(config)
    horizons = resolve_horizons(args.horizons, config)
    policy = PromotionPolicy.from_config(config)
    thresholds = DriftThresholds.from_config(config.section("operation"))
    blend = BlendPolicy.from_config(config)
    weights = {int(h): blend.weight(int(h)) for h in horizons}
    market = config.market.primary
    persist = not args.dry_run
    review_id = f"review-{uuid.uuid4().hex[:12]}"
    logger.info("environment: %s", describe_environment())
    logger.info("review %s (apply=%s, dry_run=%s)", review_id, args.apply, args.dry_run)

    with open_connection(config) as connection:
        frame = repo.load_ohlcv(
            connection, BINANCE_SOURCE, market.symbol, market.timeframe
        )
        if frame.empty:
            logger.error(
                "no OHLCV rows stored; run `python -m jobs.update_market_data` first"
            )
            return 1
        close = frame["close"]

        # 1. Data quality --------------------------------------------------
        quality = validate_ohlcv(
            frame,
            source=BINANCE_SOURCE,
            symbol=market.symbol,
            timeframe=market.timeframe,
            config=config.data_quality,
            run_id=review_id,
        )
        logger.info("data quality: %s", quality.summary())

        features_result = build_features(frame, config.features)
        usable = features_result.usable_features()
        regimes = compute_regime_labels(frame, config.features.regime)

        # 2. Performance update -------------------------------------------
        realization, production, excluded = _measure_production(
            connection, config, frame, regimes
        )
        evidence = horizons_with_enough_evidence(
            production, policy.min_realized_forecasts_per_horizon
        )

        # 3. Drift review --------------------------------------------------
        incumbent_row = registry.production_model(connection)
        incumbent_version = (
            incumbent_row["model_version"] if incumbent_row is not None else None
        )
        reference_row = incumbent_row
        if reference_row is None:
            latest = registry.list_models(connection, limit=1)
            if not latest.empty:
                reference_row = registry.get(connection, latest.iloc[0]["model_version"])
        training_cutoff = (
            reference_row["training_cutoff"] if reference_row is not None else None
        )
        strategy = (
            reference_row["training_window_strategy"] if reference_row is not None else None
        )

        drift_performance = performance_drift(
            production, _benchmark(connection, strategy), thresholds
        )
        drift_features = (
            feature_drift(usable, thresholds, training_end=pd.Timestamp(training_cutoff))
            if training_cutoff
            else pd.DataFrame(columns=["feature", "psi", "level"])
        )
        new_observations = (
            int((close.index > pd.Timestamp(training_cutoff)).sum())
            if training_cutoff
            else 0
        )
        min_new = int(
            config.section("operation").get("min_new_observations_since_training", 14)
        )

        # 4. Retraining necessity -----------------------------------------
        triggers = retraining_triggers(
            drift_features,
            drift_performance,
            thresholds,
            new_observations=new_observations,
            min_new_observations=min_new,
        )
        for row in triggers.itertuples(index=False):
            logger.info(
                "trigger %-28s %-5s %s",
                row.trigger,
                "FIRED" if row.fired else "-",
                row.detail,
            )
        fired = bool(triggers["fired"].any())

        context = ReviewContext(
            quality=quality,
            triggers=triggers,
            thresholds=thresholds,
            policy=policy,
            blend_weights=blend.weights_table(horizons),
            incumbent_version=incumbent_version,
            realization=realization,
            evidence=evidence,
            feature_drift=drift_features,
            performance_drift=drift_performance,
            excluded_in_sample=excluded,
            data_end=close.index.max().strftime("%Y-%m-%d"),
            training_cutoff=training_cutoff,
        )

        strategy_name = args.window or str(
            config.section("models").get("default_training_window", "expanding")
        )
        released, release_detail = _release_status(connection, config, strategy_name)
        logger.info("training cutoff: %s", release_detail)
        context = replace(
            context, outer_test_released=released, release_detail=release_detail
        )

        skip = _should_skip(quality, fired, args)
        if skip:
            logger.info("no candidate reviewed: %s", skip)
            return _finish(config, args, context, skip_reason=skip)

        # 5. Candidate ------------------------------------------------------
        if persist:
            with transaction(connection):
                registry.start_run(
                    connection,
                    review_id,
                    "weekly_review",
                    config_version=config.config_version,
                    code_commit=code_commit(),
                    random_seed=int(config.section("models").get("random_seed", 0)),
                )
        try:
            candidate_row, forecaster = _obtain_candidate(
                connection,
                config,
                usable,
                close,
                boundaries,
                horizons=horizons,
                window=args.window,
                existing=args.candidate,
                persist=persist,
                release_outer_test=released,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            if persist:
                with transaction(connection):
                    registry.finish_run(
                        connection, review_id, status="failed", notes=str(exc)
                    )
            raise
        candidate_version = candidate_row["model_version"]
        logger.info("candidate under review: %s", candidate_version)

        route, route_detail = classify_route(candidate_row, incumbent_row)
        logger.info("route: %s - %s", route, route_detail)

        # 6-7. Inner validation and comparison ------------------------------
        comparison = pd.DataFrame()
        regimes_frame = pd.DataFrame()
        record: dict[str, Any] = {}
        equivalent: bool | None = None
        equivalence_detail = ""

        if route == ROUTE_DATA_REFRESH:
            # Re-running the folds would reproduce the incumbent's numbers by
            # construction (see src/models/promotion.py). The useful question is
            # whether it still does, so the run happens and is compared against
            # what the incumbent recorded rather than against itself.
            candidate_result = _walk_forward(
                usable, close, boundaries, config, regimes,
                horizons=horizons,
                strategy=candidate_row["training_window_strategy"],
                params=_row_params(candidate_row),
                tag=CANDIDATE_TAG,
                include_baselines=True,
            )
            comparison = compare(
                candidate_result.metrics,
                candidate_version=candidate_result.model_version,
                reference_version=baseline_version(REFERENCE_BASELINE, config),
                policy=policy,
                weights=weights,
            )
            record = validation_record(
                comparison, policy=policy, model_version=candidate_result.model_version
            )
            equivalent, equivalence_detail = compare_validation_records(
                record, _stored_validation(incumbent_row)
            )
            logger.info("equivalence: %s - %s", equivalent, equivalence_detail)
        else:
            candidate_result = _walk_forward(
                usable, close, boundaries, config, regimes,
                horizons=horizons,
                strategy=candidate_row["training_window_strategy"],
                params=_row_params(candidate_row),
                tag=CANDIDATE_TAG,
                include_baselines=True,
            )
            if route == ROUTE_BOOTSTRAP:
                metrics = candidate_result.metrics
                reference_version = baseline_version(REFERENCE_BASELINE, config)
            else:
                incumbent_result = _walk_forward(
                    usable, close, boundaries, config, regimes,
                    horizons=horizons,
                    strategy=incumbent_row["training_window_strategy"],
                    params=_row_params(incumbent_row),
                    tag=INCUMBENT_TAG,
                    include_baselines=False,
                )
                assert_comparable_folds(candidate_result.folds, incumbent_result.folds)
                metrics = pd.concat(
                    [candidate_result.metrics, incumbent_result.metrics],
                    ignore_index=True,
                )
                reference_version = incumbent_result.model_version

            comparison = compare(
                metrics,
                candidate_version=candidate_result.model_version,
                reference_version=reference_version,
                policy=policy,
                weights=weights,
            )
            decision_horizons = (
                comparison.loc[comparison["decision_horizon"], "horizon_days"].tolist()
                if not comparison.empty
                else []
            )
            regimes_frame = regime_comparison(
                metrics,
                candidate_version=candidate_result.model_version,
                reference_version=reference_version,
                policy=policy,
                horizons=decision_horizons,
            )
            record = validation_record(
                comparison, policy=policy, model_version=candidate_result.model_version
            )

        # Reproducibility and sanity ---------------------------------------
        reproducible = _check_reproducible(
            usable, close, boundaries, config, candidate_row, horizons, released
        )
        violations, sanity_detail = (
            forecast_sanity(_sanity_frame(forecaster, usable), close)
            if forecaster is not None
            else ((), "artifact not loaded; sanity not checked")
        )
        if violations:
            logger.warning("forecast sanity: %s", "; ".join(violations))

        cutoff_advance = (
            (
                pd.Timestamp(candidate_row["training_cutoff"])
                - pd.Timestamp(incumbent_row["training_cutoff"])
            ).days
            if incumbent_row is not None
            else None
        )

        # 8. Gate ------------------------------------------------------------
        decision = evaluate_gate(
            GateInputs(
                route=route,
                candidate_version=candidate_version,
                incumbent_version=incumbent_version,
                comparison=comparison,
                regimes=regimes_frame,
                reproducible=reproducible,
                cutoff_advance_days=cutoff_advance,
                new_observations=new_observations,
                min_new_observations=min_new,
                equivalent=equivalent,
                equivalence_detail=equivalence_detail,
                sanity_violations=violations,
                sanity_detail=sanity_detail,
            ),
            policy,
        )
        logger.info("gate: %s", decision.describe())
        for check in decision.checks:
            logger.info(
                "  %-34s %-9s %s",
                check.name,
                "pass" if check.passed else ("FAIL" if check.blocking else "note"),
                check.detail,
            )

        applied_note = ""
        if args.apply and persist:
            with transaction(connection):
                if record:
                    registry.record_metrics(
                        connection, candidate_version, validation_metrics=record
                    )
                if decision.promote:
                    registry.promote(
                        connection, candidate_version, reason=decision.reason()
                    )
                else:
                    registry.reject(connection, candidate_version, decision.reason())
                registry.finish_run(
                    connection,
                    review_id,
                    status="succeeded",
                    model_id=candidate_row["model_id"],
                    metrics={"route": route, "action": decision.action},
                    notes=decision.reason(),
                )
            applied_note = f" ({decision.action})"
            logger.info("applied: %s -> %s", candidate_version, decision.action)
        elif persist:
            with transaction(connection):
                registry.finish_run(
                    connection,
                    review_id,
                    status="succeeded",
                    model_id=candidate_row["model_id"],
                    metrics={"route": route, "action": decision.action, "applied": False},
                    notes=f"reported only: {decision.reason()}",
                )
            applied_note = " (re-run with --apply to act on this decision)"
            logger.info(
                "decision recorded but NOT applied; the production model is unchanged"
            )
        else:
            applied_note = " (--dry-run: nothing was written)"

        context = replace(
            context,
            candidate_version=candidate_version,
            route_detail=route_detail,
            decision=decision,
            applied=bool(args.apply and persist),
            applied_note=applied_note,
        )

    return _finish(config, args, context)



def _release_status(connection, config: AppConfig, strategy: str) -> tuple[bool, str]:
    """May a new model train past the outer-test purge boundary?

    Only once this exact design has had its single outer-test evaluation. Before
    that, training through today would destroy the reserved block without ever
    reading it -- silently, because the resulting model looks completely normal.

    The consequence of answering "no" is visible and worth stating plainly: the
    production model stays pinned at the purge cutoff, so however much new data
    arrives, retraining produces the identical model version. That is the correct
    behaviour, and it is also the reason `jobs.final_evaluation` exists.
    """
    seed = int(config.section("models").get("random_seed", 42))
    params = dict(config.section("models").get("lightgbm", {}))
    fingerprint = configuration_fingerprint(
        configuration_row(config, strategy=strategy, params=params, seed=seed)
    )
    evaluated = evaluated_designs(registry.evaluated_models(connection))
    if fingerprint in evaluated:
        return True, (
            f"design {fingerprint} has a recorded outer-test evaluation, so the "
            "purge is spent and the model trains through the newest resolved label"
        )
    return False, (
        f"design {fingerprint} has no recorded outer-test evaluation; training "
        "stops at the purge boundary so the reserved block stays usable. Run "
        "`python -m jobs.final_evaluation --confirm` to spend it and release the "
        "cutoff."
    )


def _should_skip(quality, fired: bool, args: argparse.Namespace) -> str:
    if not quality.ok:
        return (
            f"{len(quality.errors)} blocking data-quality errors: "
            f"{', '.join(row.check_name for row in quality.errors)}. "
            "A candidate trained on data that failed validation would carry the "
            "fault into every forecast it produces."
        )
    if args.candidate:
        return ""
    if not fired and not args.force:
        return (
            "no retraining trigger fired, so no candidate was built. "
            "OPERATING_SPEC.md section 3: retraining is not automatic just "
            "because a week passed. Use `--force` to build one anyway."
        )
    return ""


def _obtain_candidate(
    connection,
    config: AppConfig,
    features: pd.DataFrame,
    close: pd.Series,
    boundaries: SplitBoundaries,
    *,
    horizons: tuple[int, ...],
    window: str | None,
    existing: str | None,
    persist: bool,
    release_outer_test: bool = False,
):
    """The candidate this review judges: an existing one, or a newly trained one."""
    if existing:
        row = registry.get(connection, existing)
        if row is None:
            raise SystemExit(f"unknown model_version {existing!r}")
        if row["status"] != "candidate":
            raise SystemExit(
                f"{existing} has status {row['status']!r}; only a candidate can be "
                "reviewed for promotion"
            )
        forecaster = registry.load_artifact(connection, existing)
        return row, forecaster

    request = TrainingRequest.from_config(
        config,
        horizons=horizons,
        strategy=window,
        cutoff=None,
        release_outer_test=release_outer_test,
        data_end=close.index.max() if release_outer_test else None,
    )
    forecaster = train_forecaster(features, close, boundaries, config, request)
    logger.info("trained: %s", forecaster.describe())
    if not persist:
        return _row_from_metadata(forecaster.metadata), forecaster

    existing_row = registry.get(connection, forecaster.metadata.model_version)
    if existing_row is not None:
        logger.info(
            "an identical candidate is already registered (%s); reviewing it",
            forecaster.metadata.model_version,
        )
        return existing_row, forecaster

    path = forecaster.save(config.paths.models_dir)
    with transaction(connection):
        registry.register(connection, forecaster.metadata, artifact_path=path)
    return registry.get(connection, forecaster.metadata.model_version), forecaster


def _row_from_metadata(metadata) -> dict[str, Any]:
    """A registry-shaped mapping for a candidate that was never registered."""
    return {
        "model_id": metadata.model_id,
        "model_version": metadata.model_version,
        "algorithm": metadata.algorithm,
        "feature_version": metadata.feature_version,
        "horizon_grid_version": metadata.horizon_grid_version,
        "random_seed": metadata.random_seed,
        "training_window_strategy": metadata.training_window_strategy,
        "training_cutoff": metadata.training_cutoff,
        "validation_metrics": "{}",
        "hyperparameters": json.dumps(
            {
                "params": dict(metadata.hyperparameters),
                "horizons": list(metadata.horizons),
                "quantiles": list(metadata.quantiles),
            }
        ),
    }


def _check_reproducible(
    features: pd.DataFrame,
    close: pd.Series,
    boundaries: SplitBoundaries,
    config: AppConfig,
    candidate_row: Mapping[str, Any],
    horizons: tuple[int, ...],
    release_outer_test: bool = False,
) -> bool:
    """Refit one horizon and confirm the predictions are identical."""
    probe = min(horizons)
    request = TrainingRequest.from_config(
        config,
        horizons=horizons,
        strategy=candidate_row["training_window_strategy"],
        params=_row_params(candidate_row) or None,
        release_outer_test=release_outer_test,
        data_end=close.index.max() if release_outer_test else None,
    )
    try:
        assert_reproducible(
            features, close, boundaries, config, request, horizon_days=probe
        )
    except Exception as exc:  # noqa: BLE001 - reported as a failed gate check
        logger.error("reproducibility check failed at h=%dd: %s", probe, exc)
        return False
    logger.info("reproducibility check passed at h=%dd", probe)
    return True


def _finish(
    config: AppConfig,
    args: argparse.Namespace,
    context: ReviewContext,
    *,
    skip_reason: str = "",
) -> int:
    if skip_reason:
        context = replace(context, skipped_reason=skip_reason)
    path = (
        config.paths.reports_dir / REPORT_NAME
        if args.output is None
        else Path(args.output)
    )
    write_report(context, path)
    logger.info("report written: %s", path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    setup_logging(config)
    return run(config, args)


if __name__ == "__main__":
    sys.exit(main())
