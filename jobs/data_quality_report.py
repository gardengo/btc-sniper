"""Generate the baseline data-quality report.

Usage::

    python -m jobs.data_quality_report
    python -m jobs.data_quality_report --output reports/custom.md

Reads only what is already stored; it performs no network I/O, so it can be run
repeatedly without touching the exchanges.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.binance import SOURCE as BINANCE_SOURCE
from src.data.coinbase import SOURCE as COINBASE_SOURCE
from src.data.validation import compare_exchanges, new_run_id, validate_ohlcv
from src.features.pipeline import build_features
from src.forecast.horizons import horizon_grid_from_config
from src.monitoring.data_report import ReportContext, write_report
from src.storage import repositories as repo
from src.storage.db import open_connection, transaction
from src.utils.config import AppConfig, load_config
from src.utils.logging import get_logger, setup_logging
from src.validation.splits import SplitBoundaries

logger = get_logger("jobs.data_quality_report")

DEFAULT_REPORT_NAME: str = "data_quality_report.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the BTC data-quality report.")
    parser.add_argument("--output", default=None, help="output Markdown path")
    parser.add_argument(
        "--no-features", action="store_true", help="skip the feature-matrix sections"
    )
    parser.add_argument(
        "--no-persist-checks",
        action="store_true",
        help="do not write the check results into data_quality_checks",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


def run(
    config: AppConfig,
    *,
    output: Path,
    with_features: bool,
    persist_checks: bool,
) -> int:
    primary = config.market.primary
    secondary = config.market.secondary

    with open_connection(config, read_only=True) as connection:
        frame = repo.load_ohlcv(connection, BINANCE_SOURCE, primary.symbol, primary.timeframe)
        secondary_frame = repo.load_ohlcv(
            connection, COINBASE_SOURCE, secondary.symbol, secondary.timeframe
        )

    if frame.empty:
        logger.error("no OHLCV rows stored; run `python -m jobs.update_market_data` first")
        return 1

    run_id = new_run_id("report")
    report = validate_ohlcv(
        frame,
        source=BINANCE_SOURCE,
        symbol=primary.symbol,
        timeframe=primary.timeframe,
        config=config.data_quality,
        run_id=run_id,
    )
    if not secondary_frame.empty:
        compare_exchanges(
            frame,
            secondary_frame,
            config=config.data_quality,
            report=report,
            primary_label=primary.symbol,
            secondary_label=secondary.symbol,
        )

    features = None
    if with_features:
        features = build_features(frame, config.features)

    context = ReportContext(
        source=BINANCE_SOURCE,
        symbol=primary.symbol,
        timeframe=primary.timeframe,
        ohlcv=frame,
        quality=report,
        features=features,
        regime_params=dict(config.features.regime),
        secondary_label=f"{COINBASE_SOURCE}/{secondary.symbol}",
        secondary_ohlcv=secondary_frame,
        split=SplitBoundaries.from_config(config),
        horizons=horizon_grid_from_config(config.forecast),
    )
    written = write_report(context, output)
    logger.info("report written to %s", written)

    if persist_checks:
        with open_connection(config) as connection, transaction(connection):
            repo.record_quality_checks(connection, report.rows)

    logger.info("data quality (run %s): %s", run_id, report.summary())
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    setup_logging(config)
    config.paths.ensure()
    output = Path(args.output) if args.output else config.paths.reports_dir / DEFAULT_REPORT_NAME
    return run(
        config,
        output=output,
        with_features=not args.no_features,
        persist_checks=not args.no_persist_checks,
    )


if __name__ == "__main__":
    sys.exit(main())
