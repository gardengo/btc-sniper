"""Binance Spot REST client (canonical market data source).

DATA_SPEC.md section 1: `GET /api/v3/klines` returns klines keyed by open time.
The response is a list of arrays::

    [ open_time, open, high, low, close, volume, close_time, quote_volume,
      trade_count, taker_buy_base, taker_buy_quote, ignore ]

Only *closed* candles are returned by :meth:`BinanceClient.fetch_klines`; the
still-forming candle of the current UTC day is dropped, because DATA_SPEC.md
section 5 forbids treating it as a closed-day observation.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime

from src.data.http import RestClient
from src.data.types import Candle
from src.utils.config import BinanceIngestionConfig
from src.utils.logging import get_logger
from src.utils.timeutils import (
    MS_PER_DAY,
    date_to_ms,
    last_closed_daily_open_ms,
    ms_to_date_str,
)

logger = get_logger(__name__)

SOURCE: str = "binance"
KLINE_FIELD_COUNT: int = 12


class BinanceClient:
    """Thin wrapper over the Binance Spot public REST endpoints."""

    def __init__(self, config: BinanceIngestionConfig, *, client: RestClient | None = None) -> None:
        self.config = config
        self._client = client or RestClient(
            config.base_url,
            retry=config.retry,
            timeout_seconds=config.request_timeout_seconds,
            min_request_interval_seconds=config.min_request_interval_seconds,
        )

    # ------------------------------------------------------------------ klines

    def _parse_kline(self, raw: list, symbol: str, timeframe: str) -> Candle:
        if len(raw) < KLINE_FIELD_COUNT:
            raise ValueError(f"unexpected kline payload with {len(raw)} fields: {raw!r}")
        return Candle(
            source=SOURCE,
            symbol=symbol,
            timeframe=timeframe,
            open_time_ms=int(raw[0]),
            close_time_ms=int(raw[6]),
            open=float(raw[1]),
            high=float(raw[2]),
            low=float(raw[3]),
            close=float(raw[4]),
            volume=float(raw[5]),
            quote_volume=float(raw[7]),
            trade_count=int(raw[8]),
            taker_buy_base=float(raw[9]),
            taker_buy_quote=float(raw[10]),
            is_closed=True,
        )

    def iter_klines(
        self,
        symbol: str,
        timeframe: str = "1d",
        *,
        start: str | date | datetime | None = None,
        end_open_time_ms: int | None = None,
        now: datetime | None = None,
    ) -> Iterator[Candle]:
        """Yield closed candles from ``start`` forward, paging through the API.

        ``end_open_time_ms`` defaults to the last fully closed daily candle, so
        the in-progress candle is never yielded.
        """
        if timeframe != "1d":
            raise ValueError(f"only the 1d timeframe is supported, got '{timeframe}'")

        cursor = date_to_ms(start or self.config.history_start)
        last_allowed = (
            end_open_time_ms
            if end_open_time_ms is not None
            else last_closed_daily_open_ms(now)
        )
        if cursor > last_allowed:
            logger.info(
                "nothing to fetch: start %s is after the last closed candle %s",
                ms_to_date_str(cursor),
                ms_to_date_str(last_allowed),
            )
            return

        while cursor <= last_allowed:
            payload = self._client.get_json(
                self.config.klines_endpoint,
                params={
                    "symbol": symbol,
                    "interval": timeframe,
                    "startTime": cursor,
                    "endTime": last_allowed,
                    "limit": self.config.max_limit,
                },
            )
            if not payload:
                logger.info(
                    "no klines returned for %s from %s; stopping",
                    symbol,
                    ms_to_date_str(cursor),
                )
                return

            newest_open_time = cursor
            for raw in payload:
                candle = self._parse_kline(raw, symbol, timeframe)
                if candle.open_time_ms > last_allowed:
                    continue
                newest_open_time = max(newest_open_time, candle.open_time_ms)
                yield candle

            if len(payload) < self.config.max_limit:
                return
            next_cursor = newest_open_time + MS_PER_DAY
            if next_cursor <= cursor:
                # Defensive: a non-advancing cursor would loop forever.
                logger.error(
                    "kline cursor failed to advance at %s; aborting", ms_to_date_str(cursor)
                )
                return
            cursor = next_cursor

    def fetch_klines(
        self,
        symbol: str,
        timeframe: str = "1d",
        *,
        start: str | date | datetime | None = None,
        end_open_time_ms: int | None = None,
        now: datetime | None = None,
    ) -> list[Candle]:
        """Materialised :meth:`iter_klines`, sorted by open time."""
        candles = list(
            self.iter_klines(
                symbol,
                timeframe,
                start=start,
                end_open_time_ms=end_open_time_ms,
                now=now,
            )
        )
        candles.sort(key=lambda candle: candle.open_time_ms)
        return candles

    # ----------------------------------------------------------- current price

    def fetch_current_price(self, symbol: str) -> float:
        """Latest traded price via REST.

        This is the WebSocket fallback described in DATA_SPEC.md section 4; it is
        for UI freshness only and never feeds the model.
        """
        payload = self._client.get_json(
            self.config.ticker_price_endpoint, params={"symbol": symbol}
        )
        return float(payload["price"])

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BinanceClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
