"""Shared pytest fixtures.

Tests run entirely offline against synthetic data. Nothing here touches the
exchanges or the project database.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.types import Candle
from src.storage.db import connect, init_db
from src.utils.config import AppConfig, load_config
from src.utils.timeutils import MS_PER_DAY, date_to_ms

SYNTHETIC_START: str = "2018-01-01"
SYNTHETIC_DAYS: int = 900
SYNTHETIC_SEED: int = 20260914


def make_ohlcv(
    days: int = SYNTHETIC_DAYS,
    start: str = SYNTHETIC_START,
    seed: int = SYNTHETIC_SEED,
) -> pd.DataFrame:
    """Deterministic synthetic daily OHLCV with a realistic shape.

    A geometric random walk with a volatility regime shift, so both the
    volatility features and the regime labels have something to react to.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=days, freq="D")

    volatility = np.where(np.arange(days) < days // 2, 0.02, 0.05)
    drift = np.where(np.arange(days) < days // 3, 0.0015, -0.0010)
    log_returns = rng.normal(drift, volatility)
    close = 10_000.0 * np.exp(np.cumsum(log_returns))

    previous_close = np.concatenate([[10_000.0], close[:-1]])
    open_price = previous_close * np.exp(rng.normal(0.0, 0.002, days))
    intraday = np.abs(rng.normal(0.0, 0.01, days))
    high = np.maximum(open_price, close) * (1.0 + intraday)
    low = np.minimum(open_price, close) * (1.0 - intraday)
    volume = np.abs(rng.normal(50_000.0, 12_000.0, days)) + 1_000.0

    open_time_ms = np.array([date_to_ms(day.date()) for day in index], dtype="int64")
    frame = pd.DataFrame(
        {
            "open_time_ms": open_time_ms,
            "close_time_ms": open_time_ms + MS_PER_DAY - 1,
            "date_utc": index.strftime("%Y-%m-%d"),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trade_count": (volume / 10.0).astype("int64"),
            "taker_buy_base": volume * rng.uniform(0.35, 0.65, days),
            "taker_buy_quote": volume * close * 0.5,
            "is_closed": 1,
        },
        index=index,
    )
    frame.index.name = "date_utc"
    return frame


def frame_to_candles(
    frame: pd.DataFrame, source: str = "binance", symbol: str = "BTCUSDT"
) -> list[Candle]:
    """Convert a synthetic frame into :class:`Candle` objects."""
    return [
        Candle(
            source=source,
            symbol=symbol,
            timeframe="1d",
            open_time_ms=int(row.open_time_ms),
            close_time_ms=int(row.close_time_ms),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
            quote_volume=float(row.quote_volume),
            trade_count=int(row.trade_count),
            taker_buy_base=float(row.taker_buy_base),
            taker_buy_quote=float(row.taker_buy_quote),
        )
        for row in frame.itertuples()
    ]


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    """The repository's real configuration (read-only use)."""
    return load_config()


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """Synthetic OHLCV frame indexed by UTC date."""
    return make_ohlcv()


@pytest.fixture
def candles(ohlcv: pd.DataFrame) -> list[Candle]:
    return frame_to_candles(ohlcv)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """A freshly initialised database in a temporary directory."""
    return init_db(tmp_path / "test.db")


@pytest.fixture
def connection(temp_db: Path):
    conn = connect(temp_db)
    try:
        yield conn
    finally:
        conn.close()
