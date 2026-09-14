"""Resolve forecast targets, update production metrics, check for drift.

Usage::

    python -m jobs.evaluate_forecasts
    python -m jobs.evaluate_forecasts --model-version <version>
    python -m jobs.evaluate_forecasts --no-persist

Runs daily after the forecast job (OPERATING_SPEC.md section 1, steps 5-6). It
resolves any forecast whose target close has arrived, updates the realization
rows in place rather than writing a second prediction (section 7), and reports
the retraining triggers of section 3 without acting on any of them.

Production metrics are honest by construction here: forecasts whose origin sits
inside their own model's training window are excluded, because a model that has
already seen the answer produces a flattering number and no information.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data.binance import SOURCE as BINANCE_SOURCE
from src.features.pipeline import build_features
from src.features.regime import compute_regime_labels
from src.models import registry
from src.monitoring.drift import (
    DriftThresholds,
    feature_drift,
    performance_drift,
    retraining_triggers,
)
from src.monitoring.performance_report import PerformanceContext, write_report
from src.monitoring.realization import (
    drop_in_sample,
    horizons_with_enough_evidence,
    load_and_realize,
    production_metrics,
    summarise,
)
from src.storage import repositories as repo
from src.storage.db import open_connection, transaction
from src.utils.config import AppConfig, load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.provenance import describe_environment

logger = get_logger("jobs.evaluate_forecasts")

REPORT_NAME: str = "production_performance.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realize forecasts and report drift.")
    parser.add_argument(
        "--model-version", default=None, help="restrict to one model version"
    )
    parser.add_argument("--output", default=None, help="report path")
    parser.add_argument(
        "--no-persist", action="store_true", help="compute without storing"
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


def _benchmark(connection, strategy: str | None) -> pd.DataFrame:
    """Walk-forward metrics for the strategy the production model was built with.

    Loosely matched by strategy name rather than by model version: a walk-forward
    run trains one model per fold and none of them is the deployed model, so
    there is no exact identity to join on. The match is logged so a reader can
    see which benchmark a degradation number was measured against.
    """
    metrics = repo.load_performance_metrics(connection, scope="validation")
    if metrics.empty or strategy is None:
        return metrics
    matched = metrics[metrics["model_version"].str.contains(strategy, regex=False)]
    if matched.empty:
        logger.warning(
            "no walk-forward benchmark found for strategy %r; performance drift "
            "cannot be measured",
            strategy,
        )
    return matched


def run(config: AppConfig, args: argparse.Namespace) -> int:
    market = config.market.primary
    thresholds = DriftThresholds.from_config(config.section("operation"))
    logger.info("environment: %s", describe_environment())

    with open_connection(config) as connection:
        frame = repo.load_ohlcv(connection, BINANCE_SOURCE, market.symbol, market.timeframe)
        if frame.empty:
            logger.error("no OHLCV rows stored; run `python -m jobs.update_market_data` first")
            return 1
        close = frame["close"]
        regimes = compute_regime_labels(frame, config.features.regime)

        rows, versions = load_and_realize(
            connection,
            close,
            source=BINANCE_SOURCE,
            symbol=market.symbol,
            timeframe=market.timeframe,
            regimes=regimes,
            model_version=args.model_version,
        )
        if rows.empty:
            logger.warning(
                "no stored forecasts; run `python -m jobs.generate_forecast` first"
            )
            return 0

        result = summarise(rows)
        logger.info("realization: %s", result.describe())

        if not args.no_persist:
            with transaction(connection):
                repo.upsert_realizations(connection, rows)

        points = repo.load_forecast_points_for_realization(
            connection,
            source=BINANCE_SOURCE,
            symbol=market.symbol,
            timeframe=market.timeframe,
            model_version=args.model_version,
        )
        origins = points.drop_duplicates("forecast_id").set_index("forecast_id")[
            "forecast_origin_date"
        ]
        cutoffs = _training_cutoffs(connection, versions)
        out_of_sample, excluded = drop_in_sample(rows, origins, cutoffs)

        regime_labels = tuple(
            str(label) for label in config.section("validation").get("regimes", [])
        )
        production = production_metrics(
            out_of_sample, versions, config, regimes=regime_labels
        )
        if not production.empty and not args.no_persist:
            with transaction(connection):
                repo.upsert_performance_metrics(
                    connection, production, scope="production", run_id="production"
                )

        model_version = args.model_version or _primary_version(versions)
        model_row = registry.get(connection, model_version) if model_version else None
        strategy = model_row["training_window_strategy"] if model_row else None

        evidence = horizons_with_enough_evidence(
            production, config.section("operation").get(
                "production_promotion", {}
            ).get("min_realized_forecasts_per_horizon", 20)
        )
        drift_performance = performance_drift(
            production, _benchmark(connection, strategy), thresholds
        )

        features = build_features(frame, config.features).usable_features()
        training_cutoff = model_row["training_cutoff"] if model_row else None
        drift_features = (
            feature_drift(
                features, thresholds, training_end=pd.Timestamp(training_cutoff)
            )
            if training_cutoff
            else pd.DataFrame(columns=["feature", "psi", "level"])
        )
        new_observations = (
            int((close.index > pd.Timestamp(training_cutoff)).sum())
            if training_cutoff
            else 0
        )
        triggers = retraining_triggers(
            drift_features,
            drift_performance,
            thresholds,
            new_observations=new_observations,
            min_new_observations=int(
                config.section("operation").get("min_new_observations_since_training", 14)
            ),
        )

        context = PerformanceContext(
            realization=result,
            production=production,
            evidence=evidence,
            feature_drift=drift_features,
            performance_drift=drift_performance,
            triggers=triggers,
            thresholds=thresholds,
            model_version=model_version,
            training_cutoff=training_cutoff,
            excluded_in_sample=excluded,
            data_end=close.index.max().strftime("%Y-%m-%d"),
        )

    path = (
        config.paths.reports_dir / REPORT_NAME
        if args.output is None
        else Path(args.output)
    )
    write_report(context, path)
    logger.info("report written: %s", path)
    _log_triggers(triggers)
    return 0


def _training_cutoffs(connection, versions: pd.Series) -> dict[str, str]:
    """Training cutoff per forecast_id, for the in-sample exclusion."""
    cutoffs: dict[str, str] = {}
    cache: dict[str, str | None] = {}
    for forecast_id, version in versions.items():
        if version not in cache:
            row = registry.get(connection, version)
            cache[version] = row["training_cutoff"] if row else None
        cutoff = cache[version]
        if cutoff:
            cutoffs[forecast_id] = cutoff
    return cutoffs


def _primary_version(versions: pd.Series) -> str | None:
    if versions.empty:
        return None
    return str(versions.value_counts().idxmax())


def _log_triggers(triggers: pd.DataFrame) -> None:
    for row in triggers.itertuples(index=False):
        logger.info(
            "trigger %-28s %-5s %s",
            row.trigger,
            "FIRED" if row.fired else "-",
            row.detail,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    setup_logging(config)
    return run(config, args)


if __name__ == "__main__":
    sys.exit(main())
