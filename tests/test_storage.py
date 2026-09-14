"""Storage layer tests: schema, idempotent upserts and round-tripping."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.data.types import Candle
from src.storage import repositories as repo
from src.storage.db import (
    connect,
    get_schema_version,
    init_db,
    table_names,
    transaction,
)
from src.utils.config import AppConfig

EXPECTED_TABLES: set[str] = {
    "data_quality_checks",
    "features",
    "forecast_points",
    "forecast_quantiles",
    "forecast_realizations",
    "forecasts",
    "market_ohlcv",
    "model_registry",
    "model_runs",
    "performance_metrics",
    "realtime_price",
    "schema_meta",
}


class TestSchema:
    def test_every_documented_table_exists(self, connection: sqlite3.Connection) -> None:
        assert EXPECTED_TABLES.issubset(set(table_names(connection)))

    def test_schema_version_is_recorded(self, connection: sqlite3.Connection) -> None:
        assert get_schema_version(connection) == "1"

    def test_init_is_idempotent(self, tmp_path: Path, candles: list[Candle]) -> None:
        path = init_db(tmp_path / "idem.db")
        first = connect(path)
        with transaction(first):
            repo.upsert_candles(first, candles[:10])
        first.close()

        init_db(path)  # re-running must not drop anything
        second = connect(path)
        try:
            assert second.execute("SELECT COUNT(*) FROM market_ohlcv").fetchone()[0] == 10
        finally:
            second.close()

    def test_foreign_keys_are_enforced(self, connection: sqlite3.Connection) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            with transaction(connection):
                connection.execute(
                    "INSERT INTO forecast_points (forecast_id, horizon_days, target_date, "
                    "predicted_log_return, predicted_price) VALUES (?, ?, ?, ?, ?)",
                    ("does-not-exist", 30, "2026-10-13", 0.01, 80000.0),
                )

    def test_read_only_connection_rejects_writes(self, temp_db: Path) -> None:
        reader = connect(temp_db, read_only=True)
        try:
            with pytest.raises(sqlite3.OperationalError):
                reader.execute(
                    "INSERT INTO schema_meta (key, value, updated_at) VALUES ('x','y','z')"
                )
        finally:
            reader.close()


class TestOhlcvRepository:
    def test_upsert_is_idempotent(
        self, connection: sqlite3.Connection, candles: list[Candle]
    ) -> None:
        with transaction(connection):
            repo.upsert_candles(connection, candles)
        with transaction(connection):
            repo.upsert_candles(connection, candles)

        coverage = repo.ohlcv_coverage(connection, "binance", "BTCUSDT")
        assert coverage["rows"] == len(candles)

    def test_upsert_updates_a_revised_candle(
        self, connection: sqlite3.Connection, candles: list[Candle]
    ) -> None:
        with transaction(connection):
            repo.upsert_candles(connection, candles[:5])

        revised = replace(candles[0], close=12_345.0)
        with transaction(connection):
            repo.upsert_candles(connection, [revised])

        frame = repo.load_ohlcv(connection, "binance", "BTCUSDT")
        assert len(frame) == 5
        assert float(frame["close"].iloc[0]) == pytest.approx(12_345.0)

    def test_round_trip_preserves_values_and_order(
        self, connection: sqlite3.Connection, candles: list[Candle], ohlcv: pd.DataFrame
    ) -> None:
        with transaction(connection):
            repo.upsert_candles(connection, candles)

        loaded = repo.load_ohlcv(connection, "binance", "BTCUSDT")
        assert loaded.index.is_monotonic_increasing
        assert len(loaded) == len(ohlcv)
        pd.testing.assert_series_equal(
            loaded["close"].reset_index(drop=True),
            ohlcv["close"].reset_index(drop=True),
            check_names=False,
            rtol=1e-12,
        )

    def test_date_filters_are_inclusive(
        self, connection: sqlite3.Connection, candles: list[Candle]
    ) -> None:
        with transaction(connection):
            repo.upsert_candles(connection, candles)

        window = repo.load_ohlcv(
            connection, "binance", "BTCUSDT", start_date="2018-02-01", end_date="2018-02-28"
        )
        assert window.index.min().strftime("%Y-%m-%d") == "2018-02-01"
        assert window.index.max().strftime("%Y-%m-%d") == "2018-02-28"
        assert len(window) == 28

    def test_latest_open_time_tracks_the_newest_candle(
        self, connection: sqlite3.Connection, candles: list[Candle]
    ) -> None:
        assert repo.latest_open_time_ms(connection, "binance", "BTCUSDT") is None
        with transaction(connection):
            repo.upsert_candles(connection, candles)
        assert repo.latest_open_time_ms(connection, "binance", "BTCUSDT") == (
            candles[-1].open_time_ms
        )

    def test_sources_are_isolated(
        self, connection: sqlite3.Connection, candles: list[Candle]
    ) -> None:
        """Coinbase rows must never appear in a Binance query."""
        coinbase = [
            replace(candle, source="coinbase", symbol="BTC-USD")
            for candle in candles[:50]
        ]
        with transaction(connection):
            repo.upsert_candles(connection, candles)
            repo.upsert_candles(connection, coinbase)

        binance_frame = repo.load_ohlcv(connection, "binance", "BTCUSDT")
        coinbase_frame = repo.load_ohlcv(connection, "coinbase", "BTC-USD")
        assert len(binance_frame) == len(candles)
        assert len(coinbase_frame) == 50


class TestFeatureRepository:
    def test_nan_features_round_trip_as_null(
        self, connection: sqlite3.Connection
    ) -> None:
        """A warmup NaN must stay NaN, not become 0."""
        frame = pd.DataFrame(
            {"feature_a": [float("nan"), 1.5]},
            index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        )
        with transaction(connection):
            repo.upsert_features(
                connection,
                frame,
                feature_version="test",
                source="binance",
                symbol="BTCUSDT",
                timeframe="1d",
            )
        loaded = repo.load_features(
            connection, feature_version="test", source="binance", symbol="BTCUSDT"
        )
        assert pd.isna(loaded["feature_a"].iloc[0])
        assert loaded["feature_a"].iloc[1] == pytest.approx(1.5)

    def test_feature_versions_do_not_collide(
        self, connection: sqlite3.Connection
    ) -> None:
        index = pd.to_datetime(["2020-01-01"])
        with transaction(connection):
            repo.upsert_features(
                connection,
                pd.DataFrame({"x": [1.0]}, index=index),
                feature_version="v1",
                source="binance",
                symbol="BTCUSDT",
                timeframe="1d",
            )
            repo.upsert_features(
                connection,
                pd.DataFrame({"x": [2.0]}, index=index),
                feature_version="v2",
                source="binance",
                symbol="BTCUSDT",
                timeframe="1d",
            )
        v1 = repo.load_features(
            connection, feature_version="v1", source="binance", symbol="BTCUSDT"
        )
        v2 = repo.load_features(
            connection, feature_version="v2", source="binance", symbol="BTCUSDT"
        )
        assert v1["x"].iloc[0] == pytest.approx(1.0)
        assert v2["x"].iloc[0] == pytest.approx(2.0)


class TestTransaction:
    def test_rollback_on_error_leaves_no_partial_write(
        self, connection: sqlite3.Connection, candles: list[Candle]
    ) -> None:
        with pytest.raises(RuntimeError):
            with transaction(connection):
                repo.upsert_candles(connection, candles[:10])
                raise RuntimeError("boom")
        assert connection.execute("SELECT COUNT(*) FROM market_ohlcv").fetchone()[0] == 0
