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

import numpy as np
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


def upsert_performance_metrics(
    connection: sqlite3.Connection,
    frame: pd.DataFrame,
    *,
    scope: str,
    run_id: str = "",
    computed_at: str | None = None,
) -> int:
    """Persist a tidy metric table (one row per metric) into `performance_metrics`.

    Expects the columns produced by `src.evaluation.evaluator`: ``model_version``,
    ``horizon_days``, ``regime``, ``metric_name``, ``metric_value``,
    ``sample_size`` and the optional ``period_start`` / ``period_end`` /
    ``fold``. Upserts on the natural key so re-running an evaluation replaces
    its own rows instead of accumulating duplicates.
    """
    if frame.empty:
        return 0
    stamp = computed_at or utc_now_iso()
    rows = []
    for record in frame.to_dict("records"):
        value = record.get("metric_value")
        rows.append(
            (
                stamp,
                scope,
                str(record["model_version"]),
                run_id,
                int(record.get("horizon_days", -1)),
                str(record.get("regime", "all")),
                str(record.get("fold", "all")),
                str(record["metric_name"]),
                None if value is None or pd.isna(value) else float(value),
                int(record.get("sample_size", 0) or 0),
                record.get("period_start"),
                record.get("period_end"),
            )
        )
    connection.executemany(
        "INSERT INTO performance_metrics (computed_at, scope, model_version, run_id, "
        "horizon_days, regime, fold, metric_name, metric_value, sample_size, "
        "period_start, period_end) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (scope, model_version, run_id, horizon_days, regime, fold, metric_name) "
        "DO UPDATE SET computed_at = excluded.computed_at, "
        "metric_value = excluded.metric_value, sample_size = excluded.sample_size, "
        "period_start = excluded.period_start, period_end = excluded.period_end",
        rows,
    )
    logger.info("recorded %d %s metric rows", len(rows), scope)
    return len(rows)


def load_performance_metrics(
    connection: sqlite3.Connection,
    *,
    scope: str,
    model_version: str | None = None,
    regime: str = "all",
) -> pd.DataFrame:
    """Read metrics back for reporting and model comparison."""
    query = (
        "SELECT model_version, horizon_days, regime, fold, metric_name, metric_value, "
        "sample_size, period_start, period_end, computed_at FROM performance_metrics "
        "WHERE scope = ? AND regime = ?"
    )
    params: list[Any] = [scope, regime]
    if model_version is not None:
        query += " AND model_version = ?"
        params.append(model_version)
    query += " ORDER BY horizon_days, model_version, metric_name"
    return pd.read_sql_query(query, connection, params=params)


def upsert_forecast(
    connection: sqlite3.Connection,
    forecast: Any,
    *,
    source: str,
    symbol: str,
    timeframe: str,
    run_id: str | None = None,
    model_id: str | None = None,
) -> int:
    """Persist one :class:`src.forecast.generate.Forecast` and its points.

    Idempotent on ``(source, symbol, timeframe, origin_date, model_version)``:
    re-running the daily job for the same origin replaces that forecast rather
    than accumulating near-duplicates. Points and quantiles cascade on delete,
    so the replacement cannot leave orphaned rows from a longer horizon grid.
    """
    origin_date = forecast.origin_date.strftime("%Y-%m-%d")
    existing = connection.execute(
        "SELECT forecast_id FROM forecasts WHERE source = ? AND symbol = ? "
        "AND timeframe = ? AND forecast_origin_date = ? AND model_version = ?",
        (source, symbol, timeframe, origin_date, forecast.model_version),
    ).fetchone()
    if existing is not None:
        connection.execute(
            "DELETE FROM forecasts WHERE forecast_id = ?", (existing["forecast_id"],)
        )

    connection.execute(
        "INSERT INTO forecasts (forecast_id, run_id, created_at, source, symbol, "
        "timeframe, forecast_origin_date, origin_open_time_ms, origin_close, "
        "current_price, model_id, model_version, feature_version, "
        "horizon_grid_version, config_version, code_commit) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            forecast.forecast_id,
            run_id,
            forecast.created_at,
            source,
            symbol,
            timeframe,
            origin_date,
            date_to_ms(forecast.origin_date.date()),
            float(forecast.origin_close),
            None if forecast.current_price is None else float(forecast.current_price),
            model_id,
            forecast.model_version,
            forecast.feature_version,
            forecast.horizon_grid_version,
            forecast.config_version,
            forecast.code_commit,
        ),
    )
    connection.executemany(
        "INSERT INTO forecast_points (forecast_id, horizon_days, target_date, "
        "predicted_log_return, predicted_price, direction_predicted, "
        "model_weight, blend_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                forecast.forecast_id,
                int(row["horizon_days"]),
                str(row["target_date"]),
                float(row["predicted_log_return"]),
                float(row["predicted_price"]),
                int(row["direction_predicted"]),
                None if row.get("model_weight") is None else float(row["model_weight"]),
                None if row.get("source") is None else str(row["source"]),
            )
            for row in forecast.points.to_dict("records")
        ],
    )
    connection.executemany(
        "INSERT INTO forecast_quantiles (forecast_id, horizon_days, quantile_label, "
        "quantile, predicted_log_return, predicted_price, crossing_adjusted) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                forecast.forecast_id,
                int(row["horizon_days"]),
                str(row["quantile_label"]),
                float(row["quantile"]),
                float(row["predicted_log_return"]),
                float(row["predicted_price"]),
                int(row["crossing_adjusted"]),
            )
            for row in forecast.quantiles.to_dict("records")
        ],
    )
    logger.info(
        "stored forecast %s (%s, %d horizons)",
        forecast.forecast_id[:8],
        origin_date,
        len(forecast.points),
    )
    return len(forecast.points)


def latest_forecast_row(
    connection: sqlite3.Connection,
    *,
    source: str,
    symbol: str,
    timeframe: str = "1d",
    model_version: str | None = None,
) -> sqlite3.Row | None:
    """Newest stored forecast, optionally for one model version."""
    query = (
        "SELECT * FROM forecasts WHERE source = ? AND symbol = ? AND timeframe = ?"
    )
    params: list[Any] = [source, symbol, timeframe]
    if model_version is not None:
        query += " AND model_version = ?"
        params.append(model_version)
    query += " ORDER BY forecast_origin_date DESC, created_at DESC LIMIT 1"
    return connection.execute(query, params).fetchone()


def load_forecast_points(
    connection: sqlite3.Connection, forecast_id: str
) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT horizon_days, target_date, predicted_log_return, predicted_price, "
        "direction_predicted, model_weight, blend_source FROM forecast_points "
        "WHERE forecast_id = ? ORDER BY horizon_days",
        connection,
        params=[forecast_id],
    )


def load_forecast_quantiles(
    connection: sqlite3.Connection,
    forecast_id: str,
    *,
    value_column: str = "predicted_price",
) -> pd.DataFrame:
    """Wide quantile matrix indexed by horizon, in prices or in log returns.

    The chart needs log returns because interpolation happens in return space
    (MODEL_SPEC.md section 9); the tables need prices. Same rows either way.
    """
    if value_column not in {"predicted_price", "predicted_log_return"}:
        raise ValueError(f"unknown value_column {value_column!r}")
    long = pd.read_sql_query(
        "SELECT horizon_days, quantile, predicted_log_return, predicted_price "
        "FROM forecast_quantiles WHERE forecast_id = ? ORDER BY horizon_days, quantile",
        connection,
        params=[forecast_id],
    )
    if long.empty:
        return long
    return long.pivot(
        index="horizon_days", columns="quantile", values=value_column
    ).sort_index()


def forecast_history(
    connection: sqlite3.Connection,
    *,
    source: str,
    symbol: str,
    horizon_days: int,
    limit: int = 60,
) -> pd.DataFrame:
    """One horizon's median forecast over recent origins, for the revision chart."""
    return pd.read_sql_query(
        "SELECT f.forecast_origin_date, f.origin_close, f.model_version, "
        "p.target_date, p.predicted_price, p.predicted_log_return "
        "FROM forecasts f JOIN forecast_points p ON p.forecast_id = f.forecast_id "
        "WHERE f.source = ? AND f.symbol = ? AND p.horizon_days = ? "
        "ORDER BY f.forecast_origin_date DESC LIMIT ?",
        connection,
        params=[source, symbol, int(horizon_days), int(limit)],
    )


def load_forecast_points_for_realization(
    connection: sqlite3.Connection,
    *,
    source: str,
    symbol: str,
    timeframe: str = "1d",
    model_version: str | None = None,
) -> pd.DataFrame:
    """Every stored forecast point with the origin context realization needs."""
    query = (
        "SELECT f.forecast_id, f.forecast_origin_date, f.origin_close, "
        "f.model_version, p.horizon_days, p.target_date, p.predicted_log_return, "
        "p.predicted_price, p.direction_predicted "
        "FROM forecasts f JOIN forecast_points p ON p.forecast_id = f.forecast_id "
        "WHERE f.source = ? AND f.symbol = ? AND f.timeframe = ?"
    )
    params: list[Any] = [source, symbol, timeframe]
    if model_version is not None:
        query += " AND f.model_version = ?"
        params.append(model_version)
    query += " ORDER BY f.forecast_origin_date, p.horizon_days"
    return pd.read_sql_query(query, connection, params=params)


def load_forecast_quantiles_long(
    connection: sqlite3.Connection, forecast_ids: Sequence[str]
) -> pd.DataFrame:
    """Long-format quantile rows for a set of forecasts."""
    if not forecast_ids:
        return pd.DataFrame(
            columns=["forecast_id", "horizon_days", "quantile", "predicted_log_return"]
        )
    placeholders = ",".join("?" for _ in forecast_ids)
    return pd.read_sql_query(
        "SELECT forecast_id, horizon_days, quantile, predicted_log_return, "
        f"predicted_price FROM forecast_quantiles WHERE forecast_id IN ({placeholders})",
        connection,
        params=list(forecast_ids),
    )


def upsert_realizations(connection: sqlite3.Connection, rows: pd.DataFrame) -> int:
    """Update realization rows in place.

    OPERATING_SPEC.md section 7: a target that resolves updates its row rather
    than creating a second prediction. The upsert is on
    ``(forecast_id, horizon_days)``, so re-running the job is idempotent and a
    pending row becomes evaluated without leaving its earlier state behind.
    """
    if rows.empty:
        return 0
    stamp = utc_now_iso()
    payload = [
        (
            record["forecast_id"],
            int(record["horizon_days"]),
            str(record["target_date"]),
            str(record["evaluation_status"]),
            record.get("actual_close"),
            record.get("actual_log_return"),
            record.get("absolute_error"),
            record.get("percentage_error"),
            record.get("log_return_error"),
            record.get("direction_predicted"),
            record.get("direction_actual"),
            record.get("direction_correct"),
            record.get("in_interval_50"),
            record.get("in_interval_80"),
            record.get("in_interval_95"),
            record.get("pinball_loss"),
            record.get("regime"),
            stamp,
        )
        for record in rows.replace({np.nan: None}).to_dict("records")
    ]
    connection.executemany(
        "INSERT INTO forecast_realizations (forecast_id, horizon_days, target_date, "
        "evaluation_status, actual_close, actual_log_return, absolute_error, "
        "percentage_error, log_return_error, direction_predicted, direction_actual, "
        "direction_correct, in_interval_50, in_interval_80, in_interval_95, "
        "pinball_loss, regime, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (forecast_id, horizon_days) DO UPDATE SET "
        "target_date = excluded.target_date, "
        "evaluation_status = excluded.evaluation_status, "
        "actual_close = excluded.actual_close, "
        "actual_log_return = excluded.actual_log_return, "
        "absolute_error = excluded.absolute_error, "
        "percentage_error = excluded.percentage_error, "
        "log_return_error = excluded.log_return_error, "
        "direction_actual = excluded.direction_actual, "
        "direction_correct = excluded.direction_correct, "
        "in_interval_50 = excluded.in_interval_50, "
        "in_interval_80 = excluded.in_interval_80, "
        "in_interval_95 = excluded.in_interval_95, "
        "pinball_loss = excluded.pinball_loss, "
        "regime = excluded.regime, "
        "updated_at = excluded.updated_at",
        payload,
    )
    logger.info("updated %d realization rows", len(payload))
    return len(payload)


def load_realizations(
    connection: sqlite3.Connection,
    *,
    status: str | None = None,
    horizon_days: int | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Realization rows joined to their forecast origin, for the prediction log."""
    query = (
        "SELECT r.*, f.forecast_origin_date, f.origin_close, f.model_version "
        "FROM forecast_realizations r "
        "JOIN forecasts f ON f.forecast_id = r.forecast_id WHERE 1 = 1"
    )
    params: list[Any] = []
    if status is not None:
        query += " AND r.evaluation_status = ?"
        params.append(status)
    if horizon_days is not None:
        query += " AND r.horizon_days = ?"
        params.append(int(horizon_days))
    query += " ORDER BY f.forecast_origin_date DESC, r.horizon_days"
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))
    return pd.read_sql_query(query, connection, params=params)
