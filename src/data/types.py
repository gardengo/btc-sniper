"""Value objects shared by the ingestion clients and the storage layer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from src.utils.timeutils import MS_PER_DAY, ms_to_date, ms_to_date_str

TIMEFRAME_DURATION_MS: dict[str, int] = {"1d": MS_PER_DAY}


class CandleError(ValueError):
    """Raised when a candle payload cannot be parsed into a valid candle."""


@dataclass(frozen=True, slots=True)
class Candle:
    """One closed OHLCV bar from a single exchange.

    The primary key is ``(source, symbol, timeframe, open_time_ms)`` as required
    by DATA_SPEC.md section 3.
    """

    source: str
    symbol: str
    timeframe: str
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float | None = None
    trade_count: int | None = None
    taker_buy_base: float | None = None
    taker_buy_quote: float | None = None
    is_closed: bool = True

    @property
    def date_utc(self) -> date:
        """UTC calendar date the candle opens on."""
        return ms_to_date(self.open_time_ms)

    @property
    def date_str(self) -> str:
        """``YYYY-MM-DD`` UTC date string."""
        return ms_to_date_str(self.open_time_ms)

    def validate(self) -> None:
        """Structural self-check.

        This guards the *shape* of a single candle only. Series-level rules
        (gaps, ordering, duplicates) belong to :mod:`src.data.validation`.
        """
        prices = {"open": self.open, "high": self.high, "low": self.low, "close": self.close}
        for name, value in prices.items():
            if not math.isfinite(value):
                raise CandleError(f"{self.symbol} {self.date_str}: {name} is not finite")
            if value <= 0:
                raise CandleError(f"{self.symbol} {self.date_str}: {name} must be > 0, got {value}")
        if not math.isfinite(self.volume) or self.volume < 0:
            raise CandleError(f"{self.symbol} {self.date_str}: invalid volume {self.volume}")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise CandleError(
                f"{self.symbol} {self.date_str}: OHLC relationship violated "
                f"(o={self.open} h={self.high} l={self.low} c={self.close})"
            )
        if self.high < self.low:
            raise CandleError(f"{self.symbol} {self.date_str}: high < low")
        if self.close_time_ms <= self.open_time_ms:
            raise CandleError(f"{self.symbol} {self.date_str}: close_time <= open_time")
        expected = TIMEFRAME_DURATION_MS.get(self.timeframe)
        if expected is not None and self.open_time_ms % expected != 0:
            raise CandleError(
                f"{self.symbol} {self.date_str}: open_time is not aligned to a {self.timeframe} boundary"
            )


@dataclass(frozen=True, slots=True)
class RealtimeTick:
    """A realtime price observation used for UI freshness only.

    DATA_SPEC.md section 1: realtime ticks never overwrite the historical daily
    close that the model is anchored on.
    """

    source: str
    symbol: str
    event_time_ms: int
    price: float
    transport: str

    def validate(self) -> None:
        if not math.isfinite(self.price) or self.price <= 0:
            raise CandleError(f"{self.symbol}: invalid realtime price {self.price}")
        if self.event_time_ms <= 0:
            raise CandleError(f"{self.symbol}: invalid event time {self.event_time_ms}")
