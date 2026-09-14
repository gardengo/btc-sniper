"""Build the point-in-time feature matrix and persist it.

Usage::

    python -m jobs.build_features
    python -m jobs.build_features --check-leakage   # also run the cutoff proof

Features are rebuilt from scratch every run. They are a pure function of the
stored OHLCV series and the configured feature version, so rebuilding is cheap
and keeps the table consistent with the code that produced it.
"""

from __future__ import annotations

import argparse
import sys

from src.data.binance import SOURCE as BINANCE_SOURCE
from src.features.pipeline import assert_no_lookahead, build_features
from src.storage import repositories as repo
from src.storage.db import init_db_from_config, open_connection, transaction
from src.utils.config import AppConfig, load_config
from src.utils.logging import get_logger, setup_logging

logger = get_logger("jobs.build_features")

DEFAULT_LEAKAGE_CUTOFFS: tuple[str, ...] = (
    "2019-06-30",
    "2020-03-12",
    "2021-11-09",
    "2023-01-01",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and store BTC features.")
    parser.add_argument(
        "--check-leakage",
        action="store_true",
        help="rebuild features on truncated history and assert the values match",
    )
    parser.add_argument(
        "--no-persist", action="store_true", help="compute features without writing them"
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


def run(config: AppConfig, *, check_leakage: bool, persist: bool) -> int:
    init_db_from_config(config)
    market = config.market.primary

    with open_connection(config) as connection:
        frame = repo.load_ohlcv(connection, BINANCE_SOURCE, market.symbol, market.timeframe)
        if frame.empty:
            logger.error("no OHLCV rows stored; run `python -m jobs.update_market_data` first")
            return 1

        result = build_features(frame, config.features)

        if check_leakage:
            for cutoff in DEFAULT_LEAKAGE_CUTOFFS:
                if cutoff < frame.index.min().strftime("%Y-%m-%d"):
                    continue
                mismatched = assert_no_lookahead(frame, config.features, cutoff=cutoff)
                if mismatched.size:
                    logger.error(
                        "LEAKAGE at cutoff %s in columns: %s",
                        cutoff,
                        list(mismatched.columns),
                    )
                    return 1
                logger.info("leakage check passed at cutoff %s", cutoff)

        if persist:
            with transaction(connection):
                written = repo.upsert_features(
                    connection,
                    result.features,
                    feature_version=result.feature_version,
                    source=BINANCE_SOURCE,
                    symbol=market.symbol,
                    timeframe=market.timeframe,
                )
            coverage = repo.feature_coverage(
                connection,
                feature_version=result.feature_version,
                source=BINANCE_SOURCE,
                symbol=market.symbol,
            )
            logger.info("wrote %d feature cells", written)
            logger.info(
                "feature coverage: %d features x %d days (%s..%s)",
                coverage["features"],
                coverage["days"],
                coverage["first_date"],
                coverage["last_date"],
            )
        else:
            logger.info("persistence skipped (--no-persist)")

    logger.info("done: %s", result.describe())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    setup_logging(config)
    return run(config, check_leakage=args.check_leakage, persist=not args.no_persist)


if __name__ == "__main__":
    sys.exit(main())
