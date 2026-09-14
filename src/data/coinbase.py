"""Coinbase Exchange client used strictly for cross-exchange sanity checks.

DATA_SPEC.md section 1: Coinbase BTC-USD is never merged into the canonical
Binance training series. It exists to detect a Binance series that has gone
wrong (stale, mis-scaled, or corrupted), so it is stored under its own
``source`` value and only ever compared against Binance.

``GET /products/{product_id}/candles`` returns newest-first arrays of
``[time_seconds, low, high, open, close, volume]``.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from src.data.http import RestClient
from src.data.types import Candle
from src.utils.config import CoinbaseIngestionConfig
from src.utils.logging import get_logger
from src.utils.timeutils import UTC, MS_PER_DAY, last_closed_daily_open_ms, parse_date

logger = get_logger(__name__)

SOURCE: str = "coinbase"
CANDLE_FIELD_COUNT: int = 6


class CoinbaseClient:
    """Read-only client for Coinbase Exchange daily candles."""

    def __init__(
        self, config: CoinbaseIngestionConfig, *, client: RestClient | None = None
    ) -> None:
        self.config = config
        self._client = client or RestClient(
            config.base_url,
            retry=config.retry,
            timeout_seconds=config.request_timeout_seconds,
            min_request_interval_seconds=config.min_request_interval_seconds,
        )

    def _parse_candle(self, raw: list, product_id: str) -> Candle:
        if len(raw) < CANDLE_FIELD_COUNT:
            raise ValueError(f"unexpected candle payload with {len(raw)} fields: {raw!r}")
        open_time_ms = int(raw[0]) * 1000
        return Candle(
            source=SOURCE,
            symbol=product_id,
            timeframe="1d",
            open_time_ms=open_time_ms,
            close_time_ms=open_time_ms + MS_PER_DAY - 1,
            low=float(raw[1]),
            high=float(raw[2]),
            open=float(raw[3]),
            close=float(raw[4]),
            volume=float(raw[5]),
            is_closed=True,
        )

    def fetch_daily_candles(
        self,
        product_id: str,
        *,
        start: str | date | datetime,
        end: str | date | datetime | None = None,
        now: datetime | None = None,
    ) -> list[Candle]:
        """Fetch daily candles in ``[start, end]``, paging backwards as needed.

        The still-forming current UTC day is excluded, matching the Binance
        client so the two series are directly comparable.
        """
        last_allowed_ms = last_closed_daily_open_ms(now)
        start_date = parse_date(start)
        end_date = parse_date(end) if end is not None else None
        last_allowed_date = datetime.fromtimestamp(last_allowed_ms / 1000, tz=UTC).date()
        if end_date is None or end_date > last_allowed_date:
            end_date = last_allowed_date
        if start_date > end_date:
            return []

        window_days = max(1, self.config.max_candles_per_request)
        collected: dict[int, Candle] = {}
        window_start = start_date
        while window_start <= end_date:
            window_end = min(window_start + timedelta(days=window_days - 1), end_date)
            payload = self._client.get_json(
                self.config.candles_endpoint.format(product_id=product_id),
                params={
                    "granularity": self.config.granularity_seconds,
                    "start": f"{window_start.isoformat()}T00:00:00Z",
                    "end": f"{window_end.isoformat()}T00:00:00Z",
                },
            )
            for raw in payload or []:
                candle = self._parse_candle(raw, product_id)
                if candle.open_time_ms <= last_allowed_ms:
                    collected[candle.open_time_ms] = candle
            window_start = window_end + timedelta(days=1)

        candles = sorted(collected.values(), key=lambda candle: candle.open_time_ms)
        logger.info(
            "fetched %d coinbase daily candles for %s (%s..%s)",
            len(candles),
            product_id,
            start_date.isoformat(),
            end_date.isoformat(),
        )
        return candles

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "CoinbaseClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
