"""Binance Spot WebSocket consumer for the current price.

DATA_SPEC.md section 4 defines the pipeline this implements::

    WebSocket tick -> parse -> validate -> latest price store -> Streamlit reads

and the failure handling it must provide: reconnect after disconnect, heartbeat
/ connection health monitoring, and a REST fallback when the socket is
unavailable (the fallback itself lives in :mod:`src.data.realtime`, because the
dashboard needs it without an event loop).

Design notes
------------
**Throttled persistence.** Ticks arrive far faster than a price display needs.
The newest tick is always kept in memory; SQLite is written at most once per
``realtime.persist_interval_seconds``. The store is what Streamlit reads, so
that interval is the real UI latency floor, not the stream rate.

**Heartbeat by silence.** A TCP connection can stay open while the feed is dead.
If no message arrives for ``realtime.heartbeat_timeout_seconds`` the read times
out and the connection is torn down and rebuilt, rather than waiting forever.

**Plausibility gate.** A price wildly far from the last closed daily candle is
rejected instead of displayed. See :func:`src.data.realtime.is_plausible`.
"""

from __future__ import annotations

import asyncio
import json
import random
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from src.data.binance import SOURCE as BINANCE_SOURCE
from src.data.realtime import (
    TRANSPORT_WEBSOCKET,
    is_plausible,
    prune_realtime_prices,
    store_tick,
)
from src.data.types import CandleError, RealtimeTick
from src.storage import repositories as repo
from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Event types that carry a usable last price, mapped to their price field.
PRICE_FIELD_BY_EVENT: dict[str, str] = {
    "trade": "p",
    "24hrMiniTicker": "c",
    "24hrTicker": "c",
}
# Trade messages carry a dedicated trade time; tickers only carry the event time.
TIME_FIELD_BY_EVENT: dict[str, str] = {
    "trade": "T",
    "24hrMiniTicker": "E",
    "24hrTicker": "E",
}


class StreamParseError(ValueError):
    """Raised when a stream message cannot be interpreted as a price update."""


def parse_message(payload: str | bytes, *, symbol: str) -> RealtimeTick:
    """Turn one raw stream message into a :class:`RealtimeTick`.

    Handles raw streams and the combined-stream envelope
    (``{"stream": ..., "data": {...}}``).
    """
    try:
        message: Any = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise StreamParseError(f"message is not valid JSON: {exc}") from exc

    if not isinstance(message, dict):
        raise StreamParseError(f"unexpected message shape: {type(message).__name__}")
    if "data" in message and isinstance(message["data"], dict):
        message = message["data"]

    event = message.get("e")
    if event not in PRICE_FIELD_BY_EVENT:
        raise StreamParseError(f"unsupported event type: {event!r}")

    try:
        price = float(message[PRICE_FIELD_BY_EVENT[event]])
        event_time_ms = int(message[TIME_FIELD_BY_EVENT[event]])
    except (KeyError, TypeError, ValueError) as exc:
        raise StreamParseError(f"malformed {event} message: {exc}") from exc

    return RealtimeTick(
        source=BINANCE_SOURCE,
        symbol=str(message.get("s") or symbol).upper(),
        event_time_ms=event_time_ms,
        price=price,
        transport=TRANSPORT_WEBSOCKET,
    )


@dataclass
class StreamStats:
    """Counters exposed for logging and tests."""

    received: int = 0
    persisted: int = 0
    rejected_unparseable: int = 0
    rejected_implausible: int = 0
    reconnects: int = 0
    heartbeat_timeouts: int = 0
    last_tick: RealtimeTick | None = field(default=None, repr=False)

    def describe(self) -> str:
        return (
            f"received={self.received} persisted={self.persisted} "
            f"rejected(parse)={self.rejected_unparseable} "
            f"rejected(implausible)={self.rejected_implausible} "
            f"reconnects={self.reconnects} heartbeat_timeouts={self.heartbeat_timeouts}"
        )


class BinancePriceStream:
    """Consumes a Binance price stream into the ``realtime_price`` table."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        config: AppConfig,
        *,
        symbol: str | None = None,
        connect_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.connection = connection
        self.config = config
        self.symbol = (symbol or config.market.primary.symbol).upper()
        self.settings = config.realtime
        self.stats = StreamStats()
        self._connect = connect_factory or ws_connect
        self._last_persist_at: float = 0.0
        self._last_prune_at: float = 0.0
        self._reference_close: float | None = None

    # ---------------------------------------------------------------- helpers

    def refresh_reference_close(self) -> float | None:
        """Last closed daily close, used as the plausibility reference."""
        frame = repo.load_ohlcv(
            self.connection,
            BINANCE_SOURCE,
            self.config.market.primary.symbol,
            self.config.market.primary.timeframe,
        )
        self._reference_close = float(frame["close"].iloc[-1]) if not frame.empty else None
        return self._reference_close

    def handle_message(self, payload: str | bytes, *, force_persist: bool = False) -> bool:
        """Process one message. Returns whether it was persisted."""
        self.stats.received += 1
        try:
            tick = parse_message(payload, symbol=self.symbol)
            tick.validate()
        except (StreamParseError, CandleError) as exc:
            self.stats.rejected_unparseable += 1
            logger.warning("discarding stream message: %s", exc)
            return False

        if not is_plausible(
            tick.price,
            self._reference_close,
            self.settings.max_deviation_from_last_close_pct,
        ):
            self.stats.rejected_implausible += 1
            logger.error(
                "rejected implausible price %.2f for %s: more than %.0f%% from the "
                "last daily close %.2f; the feed may be corrupted",
                tick.price,
                tick.symbol,
                self.settings.max_deviation_from_last_close_pct,
                self._reference_close or float("nan"),
            )
            return False

        self.stats.last_tick = tick

        elapsed = _monotonic()
        if not force_persist and (
            elapsed - self._last_persist_at < self.settings.persist_interval_seconds
        ):
            return False

        store_tick(self.connection, tick)
        self._last_persist_at = elapsed
        self.stats.persisted += 1
        self._maybe_prune(elapsed)
        return True

    def _maybe_prune(self, elapsed: float) -> None:
        if self.settings.retention_hours <= 0:
            return
        if elapsed - self._last_prune_at < 3600:
            return
        self._last_prune_at = elapsed
        prune_realtime_prices(self.connection, self.config, symbol=self.symbol)

    # ------------------------------------------------------------------- loop

    async def _consume_once(self) -> None:
        """Open one connection and read until it fails or goes silent."""
        url = self.settings.stream_path(self.symbol)
        logger.info("connecting to %s", url)
        async with self._connect(
            url,
            ping_interval=self.settings.ping_interval_seconds,
            ping_timeout=self.settings.ping_timeout_seconds,
            open_timeout=self.settings.heartbeat_timeout_seconds,
            close_timeout=self.settings.close_timeout_seconds,
        ) as socket:
            logger.info("connected; streaming %s", self.symbol)
            while True:
                try:
                    payload = await asyncio.wait_for(
                        socket.recv(), timeout=self.settings.heartbeat_timeout_seconds
                    )
                except TimeoutError:
                    self.stats.heartbeat_timeouts += 1
                    logger.warning(
                        "no message for %.0fs; treating the connection as dead",
                        self.settings.heartbeat_timeout_seconds,
                    )
                    return
                self.handle_message(payload)

    async def run(self, *, max_reconnects: int | None = None) -> StreamStats:
        """Stream forever, reconnecting with exponential backoff and jitter.

        ``max_reconnects`` bounds the loop for tests; production passes ``None``.
        """
        self.refresh_reference_close()
        attempt = 0
        while True:
            try:
                await self._consume_once()
                attempt = 0  # a clean session resets the backoff
            except asyncio.CancelledError:
                logger.info("stream cancelled; %s", self.stats.describe())
                raise
            except (ConnectionClosed, OSError, TimeoutError) as exc:
                logger.warning("stream connection lost: %s", exc)
            except Exception as exc:  # noqa: BLE001 - never let the consumer die
                logger.exception("unexpected stream error: %s", exc)

            if max_reconnects is not None and self.stats.reconnects >= max_reconnects:
                logger.info("reconnect limit reached; %s", self.stats.describe())
                return self.stats

            attempt += 1
            self.stats.reconnects += 1
            delay = self.settings.reconnect_delay(attempt) + random.uniform(
                0.0, self.settings.reconnect_jitter_seconds
            )
            logger.info(
                "reconnecting in %.2fs (attempt %d); %s",
                delay,
                attempt,
                self.stats.describe(),
            )
            await asyncio.sleep(delay)
            # The daily candle may have rolled over while we were disconnected.
            self.refresh_reference_close()


def _monotonic() -> float:
    """Monotonic clock, isolated so tests can control persistence throttling."""
    return time.monotonic()


async def stream_prices(
    connection: sqlite3.Connection,
    config: AppConfig,
    *,
    symbol: str | None = None,
    max_reconnects: int | None = None,
    connect_factory: Callable[..., Any] | None = None,
) -> StreamStats:
    """Convenience entry point used by the job."""
    stream = BinancePriceStream(
        connection, config, symbol=symbol, connect_factory=connect_factory
    )
    return await stream.run(max_reconnects=max_reconnects)
