"""Score the naive baselines and write the reference report.

Usage::

    python -m jobs.evaluate_baselines
    python -m jobs.evaluate_baselines --all-horizons     # full 77-horizon grid
    python -m jobs.evaluate_baselines --no-persist       # report only

Runs on the **inner block only**. The outer test stays untouched until a frozen
design is evaluated once (VALIDATION_SPEC.md section 4.4), so nothing here can
leak knowledge of the test block into later design choices.

Origins are taken from the usable feature index, not from the raw price index,
so the baselines are scored on exactly the origins a feature-based model will be
able to use. Otherwise the baselines would get a year of extra early history and
the comparison would be unfair in their favour.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.binance import SOURCE as BINANCE_SOURCE
from src.evaluation.baseline_eval import evaluate_baselines
from src.features.pipeline import build_features
from src.features.regime import compute_regime_labels
from src.forecast.horizons import horizon_grid_from_config
from src.monitoring.baseline_report import write_report
from src.storage import repositories as repo
from src.storage.db import open_connection, transaction
from src.utils.config import AppConfig, load_config
from src.utils.logging import get_logger, setup_logging
from src.validation.splits import SplitBoundaries

logger = get_logger("jobs.evaluate_baselines")

DEFAULT_REPORT_NAME: str = "baseline_evaluation.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the naive baselines.")
    parser.add_argument(
        "--all-horizons",
        action="store_true",
        help="score the full horizon grid instead of the required evaluation horizons",
    )
    parser.add_argument("--output", default=None, help="report path")
    parser.add_argument(
        "--no-persist", action="store_true", help="skip writing performance_metrics"
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


def run(config: AppConfig, *, all_horizons: bool, output: str | None, persist: bool) -> int:
    market = config.market.primary
    boundaries = SplitBoundaries.from_config(config)
    grid = horizon_grid_from_config(config.forecast)
    focus = config.forecast.required_evaluation_horizons
    horizons = grid if all_horizons else focus

    with open_connection(config) as connection:
        frame = repo.load_ohlcv(connection, BINANCE_SOURCE, market.symbol, market.timeframe)
        if frame.empty:
            logger.error("no OHLCV rows stored; run `python -m jobs.update_market_data` first")
            return 1

        features = build_features(frame, config.features)
        usable = features.usable_features().index
        if usable.empty:
            logger.error("no usable feature rows; cannot select evaluation origins")
            return 1

        regimes = compute_regime_labels(frame, config.features.regime)
        logger.info(
            "candidate origins: %d (%s..%s), horizons: %d",
            len(usable),
            usable.min().date(),
            usable.max().date(),
            len(horizons),
        )

        evaluation = evaluate_baselines(
            frame["close"],
            config,
            boundaries,
            horizons=horizons,
            candidate_origins=usable,
            regimes=regimes,
        )
        logger.info("evaluated: %s", evaluation.describe())

        if evaluation.metrics.empty:
            logger.error("no metrics produced; every horizon was unscoreable")
            return 1

        if persist:
            with transaction(connection):
                written = repo.upsert_performance_metrics(
                    connection,
                    evaluation.metrics,
                    scope="baseline",
                    run_id=f"{evaluation.scope}-{evaluation.split_version}",
                )
            logger.info("persisted %d metric rows", written)
        else:
            logger.info("persistence skipped (--no-persist)")

    path = config.paths.reports_dir / DEFAULT_REPORT_NAME if output is None else Path(output)
    write_report(evaluation, path, focus_horizons=focus)
    logger.info("report written: %s", path)

    _log_headline(evaluation, focus)
    return 0


def _log_headline(evaluation, focus: tuple[int, ...]) -> None:
    """Print the one thing a reader needs: who wins where, and where it is noise."""
    metrics = evaluation.metrics
    for horizon in focus:
        rows = metrics[
            (metrics["horizon_days"] == horizon)
            & (metrics["regime"] == "all")
            & (metrics["metric_name"] == "pinball_mean")
            & metrics["metric_value"].notna()
        ].sort_values("metric_value")
        if rows.empty:
            continue
        best = rows.iloc[0]
        flag = " [low power]" if bool(best.get("low_power", False)) else ""
        logger.info(
            "h=%3dd best pinball_mean: %-16s %.6f (n=%d)%s",
            horizon,
            str(best["model_version"]).replace("baseline-", ""),
            float(best["metric_value"]),
            int(best["sample_size"]),
            flag,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    setup_logging(config)
    return run(
        config,
        all_horizons=args.all_horizons,
        output=args.output,
        persist=not args.no_persist,
    )


if __name__ == "__main__":
    sys.exit(main())
