"""Long-running Binance realtime price consumer.

Usage::

    python -m jobs.stream_realtime_price                 # run until interrupted
    python -m jobs.stream_realtime_price --duration 60   # run for 60 seconds
    python -m jobs.stream_realtime_price --once          # one REST price, no socket

This process only ever writes to `realtime_price`. It never touches
`market_ohlcv`, so it cannot disturb the data the model is trained on
(DATA_SPEC.md section 1). It is safe to run alongside the daily jobs.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys

from src.data.binance_stream import BinancePriceStream
from src.data.realtime import get_current_price, prune_realtime_prices
from src.storage.db import init_db_from_config, open_connection
from src.utils.config import AppConfig, load_config
from src.utils.logging import get_logger, setup_logging

logger = get_logger("jobs.stream_realtime_price")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream the BTC current price.")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop after this many seconds (default: run until interrupted)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="fetch a single current price over REST and exit",
    )
    parser.add_argument("--symbol", default=None, help="override the streamed symbol")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


async def _run_stream(
    config: AppConfig, *, symbol: str | None, duration: float | None
) -> int:
    init_db_from_config(config)
    with open_connection(config) as connection:
        stream = BinancePriceStream(connection, config, symbol=symbol)
        reference = stream.refresh_reference_close()
        logger.info(
            "plausibility reference (last daily close): %s",
            f"{reference:,.2f}" if reference else "unavailable",
        )
        prune_realtime_prices(connection, config, symbol=stream.symbol)

        task = asyncio.create_task(stream.run())
        try:
            if duration is None:
                await task
            else:
                await asyncio.wait_for(asyncio.shield(task), timeout=duration)
        except (TimeoutError, asyncio.CancelledError):
            logger.info("stopping after %.0fs", duration or 0.0)
        except KeyboardInterrupt:  # pragma: no cover - interactive only
            logger.info("interrupted by user")
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        logger.info("stream stopped: %s", stream.stats.describe())
        if stream.stats.last_tick is not None:
            tick = stream.stats.last_tick
            logger.info("last price: %s %.2f", tick.symbol, tick.price)
        return 0 if stream.stats.persisted or stream.stats.received else 1


def _run_once(config: AppConfig, *, symbol: str | None) -> int:
    init_db_from_config(config)
    with open_connection(config) as connection:
        snapshot = get_current_price(connection, config, symbol=symbol)
        logger.info("current price: %s", snapshot.describe())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    setup_logging(config)

    if args.once:
        return _run_once(config, symbol=args.symbol)
    try:
        return asyncio.run(_run_stream(config, symbol=args.symbol, duration=args.duration))
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        logger.info("interrupted by user")
        return 0


if __name__ == "__main__":
    sys.exit(main())
