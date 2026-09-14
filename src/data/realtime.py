"""Realtime price storage and current-price resolution.

This module is the read side of the realtime path and is deliberately
**synchronous and websocket-free**, so the Streamlit app can import it without
pulling in an event loop.

The hard rule it enforces (DATA_SPEC.md section 1): a realtime tick is UI
freshness only. It is written to `realtime_price`, never to `market_ohlcv`, and
it can never become the anchor a forecast is built from. The dashboard shows two
different numbers on purpose -- a live `current_price` and the `origin_close` of
the last fully closed daily candle -- and DATA_SPEC.md section 2 says they are
expected to differ.

Resolution order for "what is BTC worth right now":

1. newest stored tick, if younger than ``realtime.max_price_age_seconds``
2. Binance REST ticker, if ``realtime.rest_fallback`` is enabled
3. the stale tick, clearly flagged as stale
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from src.data.binance import SOURCE as BINANCE_SOURCE
from src.data.binance import BinanceClient
from src.data.types import RealtimeTick
from src.storage import repositories as repo
from src.storage.db import transaction
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.utils.timeutils import ensure_utc, ms_to_datetime, utc_now

logger = get_logger(__name__)

TRANSPORT_WEBSOCKET: str = "websocket"
TRANSPORT_REST: str = "rest"


class RealtimePriceError(RuntimeError):
    """Raised when no current price can be produced by any route."""


@dataclass(frozen=True)
class PriceSnapshot:
    """A current-price answer together with how much it can be trusted."""

    symbol: str
    price: float
    event_time: datetime
    transport: str
    age_seconds: float
    is_stale: bool
    source: str = BINANCE_SOURCE

    def describe(self) -> str:
        state = "STALE" if self.is_stale else "fresh"
        return (
            f"{self.symbol} {self.price:,.2f} via {self.transport} "
            f"({self.age_seconds:.1f}s old, {state})"
        )


def store_tick(connection: sqlite3.Connection, tick: RealtimeTick) -> None:
    """Validate and persist one tick."""
    tick.validate()
    with transaction(connection):
        repo.insert_realtime_tick(connection, tick)


def is_plausible(
    price: float, last_daily_close: float | None, max_deviation_pct: float
) -> bool:
    """Whether a live price is close enough to the last daily close to be real.

    A corrupted feed is far more likely than a 50% intraday move -- the worst day
    in BTC's recorded history, 2020-03-12, moved 39.6%. Rejecting an implausible
    tick keeps a nonsense number off the dashboard; because the price then goes
    stale, the failure is visible rather than silent.
    """
    if last_daily_close is None or last_daily_close <= 0:
        return True
    deviation = abs(price - last_daily_close) / last_daily_close * 100.0
    return deviation <= max_deviation_pct


def latest_snapshot(
    connection: sqlite3.Connection,
    config: AppConfig,
    *,
    symbol: str | None = None,
    now: datetime | None = None,
) -> PriceSnapshot | None:
    """Newest stored tick as a snapshot, or ``None`` when nothing is stored."""
    target = symbol or config.market.primary.symbol
    row = repo.latest_realtime_tick(connection, BINANCE_SOURCE, target)
    if row is None:
        return None

    reference = ensure_utc(now) if now is not None else utc_now()
    event_time = ms_to_datetime(int(row["event_time_ms"]))
    age = (reference - event_time).total_seconds()
    return PriceSnapshot(
        symbol=target,
        price=float(row["price"]),
        event_time=event_time,
        transport=str(row["transport"]),
        age_seconds=age,
        is_stale=age > config.realtime.max_price_age_seconds,
    )


def fetch_rest_snapshot(
    config: AppConfig,
    *,
    symbol: str | None = None,
    client: BinanceClient | None = None,
    now: datetime | None = None,
) -> PriceSnapshot:
    """Current price straight from the Binance REST ticker.

    This is the WebSocket fallback of DATA_SPEC.md section 4 and
    OPERATING_SPEC.md section 9.
    """
    target = symbol or config.market.primary.symbol
    owned = client is None
    active = client or BinanceClient(config.ingestion.binance)
    try:
        price = active.fetch_current_price(target)
    finally:
        if owned:
            active.close()

    reference = ensure_utc(now) if now is not None else utc_now()
    return PriceSnapshot(
        symbol=target,
        price=price,
        event_time=reference,
        transport=TRANSPORT_REST,
        age_seconds=0.0,
        is_stale=False,
    )


def get_current_price(
    connection: sqlite3.Connection,
    config: AppConfig,
    *,
    symbol: str | None = None,
    client: BinanceClient | None = None,
    persist_fallback: bool = True,
    now: datetime | None = None,
) -> PriceSnapshot:
    """Best available current price, with its freshness attached.

    Never raises for a merely stale price -- it returns it flagged -- so the
    dashboard can show a last-known value and say that it is old, which beats
    showing nothing.
    """
    target = symbol or config.market.primary.symbol
    snapshot = latest_snapshot(connection, config, symbol=target, now=now)
    if snapshot is not None and not snapshot.is_stale:
        return snapshot

    if not config.realtime.rest_fallback:
        if snapshot is None:
            raise RealtimePriceError(
                f"no realtime price stored for {target} and REST fallback is disabled"
            )
        logger.warning("serving stale price: %s", snapshot.describe())
        return snapshot

    reason = "no stored tick" if snapshot is None else f"{snapshot.age_seconds:.0f}s old"
    logger.info("falling back to REST current price for %s (%s)", target, reason)
    try:
        fallback = fetch_rest_snapshot(config, symbol=target, client=client, now=now)
    except Exception as exc:  # noqa: BLE001 - any transport failure is recoverable here
        if snapshot is None:
            raise RealtimePriceError(
                f"no realtime price available for {target}: {exc}"
            ) from exc
        logger.warning("REST fallback failed (%s); serving stale price", exc)
        return snapshot

    if persist_fallback:
        tick = RealtimeTick(
            source=BINANCE_SOURCE,
            symbol=target,
            event_time_ms=int(fallback.event_time.timestamp() * 1000),
            price=fallback.price,
            transport=TRANSPORT_REST,
        )
        try:
            store_tick(connection, tick)
        except sqlite3.Error as exc:
            logger.warning("could not persist REST fallback tick: %s", exc)
    return fallback


def prune_realtime_prices(
    connection: sqlite3.Connection,
    config: AppConfig,
    *,
    symbol: str | None = None,
    now: datetime | None = None,
) -> int:
    """Delete ticks older than ``realtime.retention_hours``.

    A tick every couple of seconds is roughly 43k rows per day, so the table
    needs a bound. Only display history is discarded; nothing the model uses
    lives here.
    """
    retention = config.realtime.retention_hours
    if retention <= 0:
        return 0
    target = symbol or config.market.primary.symbol
    reference = ensure_utc(now) if now is not None else utc_now()
    cutoff_ms = int((reference.timestamp() - retention * 3600) * 1000)

    with transaction(connection):
        cursor = connection.execute(
            "DELETE FROM realtime_price WHERE source = ? AND symbol = ? "
            "AND event_time_ms < ?",
            (BINANCE_SOURCE, target, cutoff_ms),
        )
        deleted = cursor.rowcount
    if deleted:
        logger.info("pruned %d realtime rows older than %dh", deleted, retention)
    return deleted


def recent_prices(
    connection: sqlite3.Connection,
    config: AppConfig,
    *,
    symbol: str | None = None,
    limit: int = 500,
):
    """Recent ticks, newest first, for an intraday sparkline."""
    target = symbol or config.market.primary.symbol
    return connection.execute(
        "SELECT event_time_ms, price, transport FROM realtime_price "
        "WHERE source = ? AND symbol = ? ORDER BY event_time_ms DESC LIMIT ?",
        (BINANCE_SOURCE, target, int(limit)),
    ).fetchall()
