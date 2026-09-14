"""Read/write helpers for the SQLite tables.

Every write is an idempotent upsert keyed on the natural primary key, so an
ingestion job can be re-run over an overlapping range without creating
duplicates (DATA_SPEC.md section 3).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.data.types import Candle, RealtimeTick
from src.utils.logging import get_logger
from src.utils.timeutils import date_to_ms, utc_now_iso

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.data.validation import QualityCheckRow

logger = get_logger(__name__)

OHLCV_COLUMNS: tuple[str, ...] = (
    "source",
    "symbol",
    "timeframe",
    "open_time_ms",
    "close_time_ms",
    "date_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "is_closed",
    "ingested_at",
)

_UPSERT_OHLCV = f"""
INSERT INTO market_ohlcv ({", ".join(OHLCV_COLUMNS)})
VALUES ({", ".join("?" for _ in OHLCV_COLUMNS)})
ON CONFLICT(source, symbol, timeframe, open_time_ms) DO UPDATE SET
    close_time_ms   = excluded.close_time_ms,
    date_utc        = excluded.date_utc,
    open            = excluded.open,
    high            = excluded.high,
    low             = excluded.low,
    close           = excluded.close,
    volume          = excluded.volume,
    quote_volume    = excluded.quote_volume,
    trade_count     = excluded.trade_count,
    taker_buy_base  = excluded.taker_buy_base,
    taker_buy_quote = excluded.taker_buy_quote,
    is_closed       = excluded.is_closed,
    ingested_at     = excluded.ingested_at
"""


def _candle_row(candle: Candle, ingested_at: str) -> tuple[Any, ...]:
    return (
        candle.source,
        candle.symbol,
        candle.timeframe,
        candle.open_time_ms,
        candle.close_time_ms,
        candle.date_str,
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.volume,
        candle.quote_volume,
        candle.trade_count,
        candle.taker_buy_base,
        candle.taker_buy_quote,
        int(candle.is_closed),
        ingested_at,
    )


def upsert_candles(connection: sqlite3.Connection, candles: Iterable[Candle]) -> int:
    """Insert or update candles. Returns the number of rows written."""
    ingested_at = utc_now_iso()
    rows = [_candle_row(candle, ingested_at) for candle in candles]
    if not rows:
        return 0
    connection.executemany(_UPSERT_OHLCV, rows)
    return len(rows)


def load_ohlcv(
    connection: sqlite3.Connection,
    source: str,
    symbol: str,
    timeframe: str = "1d",
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    closed_only: bool = True,
) -> pd.DataFrame:
    """Load an OHLCV series ordered by ``open_time_ms`` ascending.

    The returned frame is indexed by ``date_utc`` (``datetime64[ns]``, UTC
    calendar days) so downstream feature code never re-parses timestamps.
    """
    clauses = ["source = ?", "symbol = ?", "timeframe = ?"]
    params: list[Any] = [source, symbol, timeframe]
    if closed_only:
        clauses.append("is_closed = 1")
    if start_date is not None:
        clauses.append("open_time_ms >= ?")
        params.append(date_to_ms(start_date))
    if end_date is not None:
        clauses.append("open_time_ms <= ?")
        params.append(date_to_ms(end_date))

    query = (
        "SELECT open_time_ms, close_time_ms, date_utc, open, high, low, close, volume, "
        "quote_volume, trade_count, taker_buy_base, taker_buy_quote, is_closed "
        f"FROM market_ohlcv WHERE {' AND '.join(clauses)} ORDER BY open_time_ms ASC"
    )
    frame = pd.read_sql_query(query, connection, params=params)
    if frame.empty:
        frame["date_utc"] = pd.Series(dtype="datetime64[ns]")
        return frame.set_index("date_utc")
    frame["date_utc"] = pd.to_datetime(frame["date_utc"], format="%Y-%m-%d")
    return frame.set_index("date_utc")


def ohlcv_coverage(
    connection: sqlite3.Connection, source: str, symbol: str, timeframe: str = "1d"
) -> dict[str, Any]:
    """Row count and stored date range for one series."""
    row = connection.execute(
        "SELECT COUNT(*) AS n, MIN(open_time_ms) AS first_ms, MAX(open_time_ms) AS last_ms, "
        "MIN(date_utc) AS first_date, MAX(date_utc) AS last_date "
        "FROM market_ohlcv WHERE source = ? AND symbol = ? AND timeframe = ?",
        (source, symbol, timeframe),
    ).fetchone()
    return {
        "rows": int(row["n"]),
        "first_open_time_ms": row["first_ms"],
        "last_open_time_ms": row["last_ms"],
        "first_date": row["first_date"],
        "last_date": row["last_date"],
    }


def latest_open_time_ms(
    connection: sqlite3.Connection, source: str, symbol: str, timeframe: str = "1d"
) -> int | None:
    """Open time of the newest stored candle, or ``None`` when empty."""
    row = connection.execute(
        "SELECT MAX(open_time_ms) AS last_ms FROM market_ohlcv "
        "WHERE source = ? AND symbol = ? AND timeframe = ? AND is_closed = 1",
        (source, symbol, timeframe),
    ).fetchone()
    return None if row is None or row["last_ms"] is None else int(row["last_ms"])


def insert_realtime_tick(connection: sqlite3.Connection, tick: RealtimeTick) -> None:
    """Store a realtime price observation (UI freshness only)."""
    connection.execute(
        "INSERT INTO realtime_price (source, symbol, event_time_ms, price, transport, received_at) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(source, symbol, event_time_ms) DO UPDATE SET "
        "price = excluded.price, transport = excluded.transport, "
        "received_at = excluded.received_at",
        (
            tick.source,
            tick.symbol,
            tick.event_time_ms,
            tick.price,
            tick.transport,
            utc_now_iso(),
        ),
    )


def latest_realtime_tick(
    connection: sqlite3.Connection, source: str, symbol: str
) -> sqlite3.Row | None:
    """Most recent realtime tick for a symbol."""
    return connection.execute(
        "SELECT * FROM realtime_price WHERE source = ? AND symbol = ? "
        "ORDER BY event_time_ms DESC LIMIT 1",
        (source, symbol),
    ).fetchone()


def record_quality_checks(
    connection: sqlite3.Connection, results: Sequence["QualityCheckRow"]
) -> int:
    """Persist data-quality check outcomes."""
    if not results:
        return 0
    connection.executemany(
        "INSERT INTO data_quality_checks (run_id, checked_at, source, symbol, timeframe, "
        "check_name, severity, status, message, details, rows_checked, range_start, range_end) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                row.run_id,
                row.checked_at,
                row.source,
                row.symbol,
                row.timeframe,
                row.check_name,
                row.severity,
                row.status,
                row.message,
                json.dumps(row.details, default=str) if row.details else None,
                row.rows_checked,
                row.range_start,
                row.range_end,
            )
            for row in results
        ],
    )
    return len(results)


def upsert_features(
    connection: sqlite3.Connection,
    frame: pd.DataFrame,
    *,
    feature_version: str,
    source: str,
    symbol: str,
    timeframe: str,
) -> int:
    """Persist a wide feature frame in long format.

    ``frame`` is indexed by UTC date and holds one column per feature. NaN
    values are stored as SQL NULL rather than dropped, so a missing warmup value
    stays visible instead of silently disappearing.
    """
    if frame.empty:
        return 0
    computed_at = utc_now_iso()
    rows: list[tuple[Any, ...]] = []
    for timestamp, record in frame.iterrows():
        open_time_ms = date_to_ms(timestamp.date())
        date_str = timestamp.strftime("%Y-%m-%d")
        for name, value in record.items():
            stored = None if pd.isna(value) else float(value)
            rows.append(
                (
                    feature_version,
                    source,
                    symbol,
                    timeframe,
                    open_time_ms,
                    date_str,
                    str(name),
                    stored,
                    computed_at,
                )
            )
    connection.executemany(
        "INSERT INTO features (feature_version, source, symbol, timeframe, open_time_ms, "
        "date_utc, feature_name, value, computed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(feature_version, source, symbol, timeframe, open_time_ms, feature_name) "
        "DO UPDATE SET value = excluded.value, computed_at = excluded.computed_at",
        rows,
    )
    return len(rows)


def load_features(
    connection: sqlite3.Connection,
    *,
    feature_version: str,
    source: str,
    symbol: str,
    timeframe: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load features back as a wide frame indexed by UTC date."""
    clauses = ["feature_version = ?", "source = ?", "symbol = ?", "timeframe = ?"]
    params: list[Any] = [feature_version, source, symbol, timeframe]
    if start_date is not None:
        clauses.append("open_time_ms >= ?")
        params.append(date_to_ms(start_date))
    if end_date is not None:
        clauses.append("open_time_ms <= ?")
        params.append(date_to_ms(end_date))

    frame = pd.read_sql_query(
        "SELECT date_utc, feature_name, value FROM features "
        f"WHERE {' AND '.join(clauses)} ORDER BY open_time_ms ASC",
        connection,
        params=params,
    )
    if frame.empty:
        return pd.DataFrame(index=pd.Index([], name="date_utc", dtype="datetime64[ns]"))
    frame["date_utc"] = pd.to_datetime(frame["date_utc"], format="%Y-%m-%d")
    wide = frame.pivot(index="date_utc", columns="feature_name", values="value")
    wide.columns.name = None
    return wide.sort_index()


def feature_coverage(
    connection: sqlite3.Connection, *, feature_version: str, source: str, symbol: str
) -> dict[str, Any]:
    """Row/feature counts and date range for a stored feature version."""
    row = connection.execute(
        "SELECT COUNT(*) AS n, COUNT(DISTINCT feature_name) AS n_features, "
        "COUNT(DISTINCT date_utc) AS n_days, MIN(date_utc) AS first_date, "
        "MAX(date_utc) AS last_date FROM features "
        "WHERE feature_version = ? AND source = ? AND symbol = ?",
        (feature_version, source, symbol),
    ).fetchone()
    return {
        "rows": int(row["n"]),
        "features": int(row["n_features"]),
        "days": int(row["n_days"]),
        "first_date": row["first_date"],
        "last_date": row["last_date"],
    }
