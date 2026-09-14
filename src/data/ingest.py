"""Market-data ingestion orchestration.

Supports the two modes required by DATA_SPEC.md section 3: an initial full
backfill and an incremental update. Both write through the same idempotent
upsert, so re-running over an overlapping range is safe.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from src.data.binance import SOURCE as BINANCE_SOURCE
from src.data.binance import BinanceClient
from src.data.coinbase import SOURCE as COINBASE_SOURCE
from src.data.coinbase import CoinbaseClient
from src.data.types import Candle, CandleError
from src.storage import repositories as repo
from src.storage.db import transaction
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.utils.timeutils import MS_PER_DAY, ms_to_date, ms_to_date_str, parse_date

logger = get_logger(__name__)

DEFAULT_OVERLAP_DAYS: int = 3


@dataclass(frozen=True)
class IngestionResult:
    """Outcome of one ingestion pass over a single source."""

    source: str
    symbol: str
    timeframe: str
    mode: str
    requested_start: str
    fetched: int
    written: int
    rejected: int
    first_date: str | None
    last_date: str | None

    def describe(self) -> str:
        span = (
            f"{self.first_date}..{self.last_date}"
            if self.first_date and self.last_date
            else "no new candles"
        )
        return (
            f"{self.source}/{self.symbol} [{self.mode}] fetched={self.fetched} "
            f"written={self.written} rejected={self.rejected} span={span}"
        )


def _validated(candles: list[Candle]) -> tuple[list[Candle], list[str]]:
    """Split candles into structurally valid ones and rejection messages."""
    accepted: list[Candle] = []
    rejected: list[str] = []
    for candle in candles:
        try:
            candle.validate()
        except CandleError as exc:
            rejected.append(str(exc))
        else:
            accepted.append(candle)
    return accepted, rejected


def resolve_incremental_start(
    connection: sqlite3.Connection,
    *,
    source: str,
    symbol: str,
    timeframe: str,
    history_start: str,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
) -> tuple[str, str]:
    """Decide where an update should start.

    Returns ``(start_date, mode)``. When the table is empty the mode is
    ``backfill``; otherwise the start is rewound by ``overlap_days`` so a candle
    that was revised after ingestion gets corrected.
    """
    last_ms = repo.latest_open_time_ms(connection, source, symbol, timeframe)
    if last_ms is None:
        return history_start, "backfill"
    start = ms_to_date(last_ms) - timedelta(days=max(0, overlap_days))
    floor = parse_date(history_start)
    return max(start, floor).isoformat(), "incremental"


def ingest_binance_ohlcv(
    connection: sqlite3.Connection,
    config: AppConfig,
    *,
    client: BinanceClient | None = None,
    full_refresh: bool = False,
    start: str | date | datetime | None = None,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
    now: datetime | None = None,
) -> IngestionResult:
    """Fetch Binance daily candles and upsert them into ``market_ohlcv``."""
    market = config.market.primary
    settings = config.ingestion.binance

    if start is not None:
        start_date, mode = parse_date(start).isoformat(), "explicit"
    elif full_refresh:
        start_date, mode = settings.history_start, "full_refresh"
    else:
        start_date, mode = resolve_incremental_start(
            connection,
            source=BINANCE_SOURCE,
            symbol=market.symbol,
            timeframe=market.timeframe,
            history_start=settings.history_start,
            overlap_days=overlap_days,
        )

    logger.info(
        "ingesting %s %s from %s (mode=%s)",
        BINANCE_SOURCE,
        market.symbol,
        start_date,
        mode,
    )
    owned_client = client is None
    active = client or BinanceClient(settings)
    try:
        candles = active.fetch_klines(
            market.symbol, market.timeframe, start=start_date, now=now
        )
    finally:
        if owned_client:
            active.close()

    accepted, rejected = _validated(candles)
    for message in rejected:
        logger.error("rejected candle: %s", message)

    with transaction(connection):
        written = repo.upsert_candles(connection, accepted)

    result = IngestionResult(
        source=BINANCE_SOURCE,
        symbol=market.symbol,
        timeframe=market.timeframe,
        mode=mode,
        requested_start=start_date,
        fetched=len(candles),
        written=written,
        rejected=len(rejected),
        first_date=accepted[0].date_str if accepted else None,
        last_date=accepted[-1].date_str if accepted else None,
    )
    logger.info(result.describe())
    return result


def ingest_coinbase_ohlcv(
    connection: sqlite3.Connection,
    config: AppConfig,
    *,
    client: CoinbaseClient | None = None,
    days: int | None = None,
    now: datetime | None = None,
) -> IngestionResult:
    """Fetch the Coinbase cross-check window and upsert it.

    Only a trailing window is stored: Coinbase is a sanity-check source, not a
    training source, so a full history is unnecessary.
    """
    market = config.market.secondary
    settings = config.ingestion.coinbase
    window_days = days if days is not None else config.data_quality.cross_exchange_check_days

    last_ms = repo.latest_open_time_ms(connection, BINANCE_SOURCE, config.market.primary.symbol)
    anchor = ms_to_date(last_ms) if last_ms is not None else (now or datetime.now()).date()
    start_date = (anchor - timedelta(days=window_days)).isoformat()

    logger.info("ingesting %s %s from %s", COINBASE_SOURCE, market.symbol, start_date)
    owned_client = client is None
    active = client or CoinbaseClient(settings)
    try:
        candles = active.fetch_daily_candles(market.symbol, start=start_date, now=now)
    finally:
        if owned_client:
            active.close()

    accepted, rejected = _validated(candles)
    for message in rejected:
        logger.warning("rejected coinbase candle: %s", message)

    with transaction(connection):
        written = repo.upsert_candles(connection, accepted)

    result = IngestionResult(
        source=COINBASE_SOURCE,
        symbol=market.symbol,
        timeframe=market.timeframe,
        mode="cross_check_window",
        requested_start=start_date,
        fetched=len(candles),
        written=written,
        rejected=len(rejected),
        first_date=accepted[0].date_str if accepted else None,
        last_date=accepted[-1].date_str if accepted else None,
    )
    logger.info(result.describe())
    return result


def missing_candle_dates(connection: sqlite3.Connection, config: AppConfig) -> list[str]:
    """Calendar days with no stored Binance candle inside the stored range."""
    market = config.market.primary
    coverage = repo.ohlcv_coverage(connection, BINANCE_SOURCE, market.symbol, market.timeframe)
    if not coverage["rows"]:
        return []
    stored = {
        int(row["open_time_ms"])
        for row in connection.execute(
            "SELECT open_time_ms FROM market_ohlcv "
            "WHERE source = ? AND symbol = ? AND timeframe = ?",
            (BINANCE_SOURCE, market.symbol, market.timeframe),
        )
    }
    first = int(coverage["first_open_time_ms"])
    last = int(coverage["last_open_time_ms"])
    return [
        ms_to_date_str(ms)
        for ms in range(first, last + MS_PER_DAY, MS_PER_DAY)
        if ms not in stored
    ]
