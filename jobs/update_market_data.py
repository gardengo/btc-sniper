"""Ingest market data, validate it, and record the quality outcome.

Usage::

    python -m jobs.update_market_data                 # incremental update
    python -m jobs.update_market_data --full-refresh  # full backfill
    python -m jobs.update_market_data --no-coinbase   # skip the cross-check

Exit code 1 means a blocking data-quality error was recorded; per
OPERATING_SPEC.md section 9 no forecast should be generated in that state.
"""

from __future__ import annotations

import argparse
import sys

from src.data.binance import SOURCE as BINANCE_SOURCE
from src.data.coinbase import SOURCE as COINBASE_SOURCE
from src.data.ingest import ingest_binance_ohlcv, ingest_coinbase_ohlcv
from src.data.validation import compare_exchanges, new_run_id, validate_ohlcv
from src.storage import repositories as repo
from src.storage.db import init_db_from_config, open_connection, transaction
from src.utils.config import AppConfig, load_config
from src.utils.logging import get_logger, setup_logging

logger = get_logger("jobs.update_market_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update BTC market data in SQLite.")
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="re-fetch the whole history instead of only new candles",
    )
    parser.add_argument(
        "--no-coinbase",
        action="store_true",
        help="skip the Coinbase cross-exchange sanity check",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


def run(config: AppConfig, *, full_refresh: bool, with_coinbase: bool) -> int:
    """Run ingestion + validation. Returns a process exit code."""
    init_db_from_config(config)
    primary = config.market.primary
    secondary = config.market.secondary
    run_id = new_run_id("ingest")

    with open_connection(config) as connection:
        binance_result = ingest_binance_ohlcv(
            connection, config, full_refresh=full_refresh
        )

        coinbase_result = None
        if with_coinbase and secondary.enabled:
            try:
                coinbase_result = ingest_coinbase_ohlcv(connection, config)
            except Exception as exc:  # noqa: BLE001 - sanity check must not break ingestion
                logger.warning("coinbase cross-check ingestion failed: %s", exc)

        primary_frame = repo.load_ohlcv(
            connection, BINANCE_SOURCE, primary.symbol, primary.timeframe
        )
        report = validate_ohlcv(
            primary_frame,
            source=BINANCE_SOURCE,
            symbol=primary.symbol,
            timeframe=primary.timeframe,
            config=config.data_quality,
            run_id=run_id,
        )

        if coinbase_result is not None:
            secondary_frame = repo.load_ohlcv(
                connection, COINBASE_SOURCE, secondary.symbol, secondary.timeframe
            )
            compare_exchanges(
                primary_frame,
                secondary_frame,
                config=config.data_quality,
                report=report,
                primary_label=primary.symbol,
                secondary_label=secondary.symbol,
            )

        with transaction(connection):
            repo.record_quality_checks(connection, report.rows)

        coverage = repo.ohlcv_coverage(
            connection, BINANCE_SOURCE, primary.symbol, primary.timeframe
        )

    logger.info("ingestion: %s", binance_result.describe())
    if coinbase_result is not None:
        logger.info("cross-check: %s", coinbase_result.describe())
    logger.info(
        "coverage: %d rows, %s..%s",
        coverage["rows"],
        coverage["first_date"],
        coverage["last_date"],
    )
    logger.info("data quality (run %s): %s", run_id, report.summary())

    if not report.ok:
        for row in report.errors:
            logger.error("blocking: %s - %s", row.check_name, row.message)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    setup_logging(config)
    return run(
        config, full_refresh=args.full_refresh, with_coinbase=not args.no_coinbase
    )


if __name__ == "__main__":
    sys.exit(main())
