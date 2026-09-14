"""Ingestion orchestration and configuration tests."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from src.data.ingest import (
    ingest_binance_ohlcv,
    missing_candle_dates,
    resolve_incremental_start,
)
from src.data.types import Candle
from src.storage import repositories as repo
from src.storage.db import transaction
from src.utils.config import AppConfig, ConfigError, load_config
from src.utils.timeutils import UTC, date_to_ms

NOW = datetime(2018, 1, 11, 6, 0, tzinfo=UTC)


class StubBinanceClient:
    """Returns a fixed candle list and records the start it was asked for."""

    def __init__(self, candles: list[Candle]) -> None:
        self.candles = candles
        self.requested_start: str | None = None

    def fetch_klines(self, symbol, timeframe="1d", *, start=None, now=None, **_):
        self.requested_start = str(start)
        return list(self.candles)

    def close(self) -> None:
        return None


class TestIncrementalStart:
    def test_empty_table_triggers_a_backfill(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        start, mode = resolve_incremental_start(
            connection,
            source="binance",
            symbol="BTCUSDT",
            timeframe="1d",
            history_start="2017-08-17",
        )
        assert mode == "backfill"
        assert start == "2017-08-17"

    def test_populated_table_rewinds_by_the_overlap(
        self, connection: sqlite3.Connection, candles: list[Candle]
    ) -> None:
        with transaction(connection):
            repo.upsert_candles(connection, candles[:100])
        last_date = candles[99].date_str

        start, mode = resolve_incremental_start(
            connection,
            source="binance",
            symbol="BTCUSDT",
            timeframe="1d",
            history_start="2017-08-17",
            overlap_days=3,
        )
        assert mode == "incremental"
        assert start < last_date
        assert (date_to_ms(last_date) - date_to_ms(start)) == 3 * 86_400_000

    def test_start_never_precedes_the_configured_history_start(
        self, connection: sqlite3.Connection, candles: list[Candle]
    ) -> None:
        with transaction(connection):
            repo.upsert_candles(connection, candles[:2])
        start, _ = resolve_incremental_start(
            connection,
            source="binance",
            symbol="BTCUSDT",
            timeframe="1d",
            history_start="2018-01-01",
            overlap_days=365,
        )
        assert start == "2018-01-01"


class TestIngestion:
    def test_ingestion_writes_and_reports(
        self, connection: sqlite3.Connection, app_config: AppConfig, candles: list[Candle]
    ) -> None:
        client = StubBinanceClient(candles[:50])
        result = ingest_binance_ohlcv(
            connection, app_config, client=client, full_refresh=True, now=NOW
        )
        assert result.fetched == 50
        assert result.written == 50
        assert result.rejected == 0
        assert repo.ohlcv_coverage(connection, "binance", "BTCUSDT")["rows"] == 50

    def test_rerunning_ingestion_does_not_duplicate(
        self, connection: sqlite3.Connection, app_config: AppConfig, candles: list[Candle]
    ) -> None:
        client = StubBinanceClient(candles[:50])
        ingest_binance_ohlcv(connection, app_config, client=client, full_refresh=True, now=NOW)
        ingest_binance_ohlcv(connection, app_config, client=client, full_refresh=True, now=NOW)
        assert repo.ohlcv_coverage(connection, "binance", "BTCUSDT")["rows"] == 50

    def test_structurally_invalid_candles_are_rejected_not_stored(
        self, connection: sqlite3.Connection, app_config: AppConfig, candles: list[Candle]
    ) -> None:
        from dataclasses import replace

        corrupted = list(candles[:10])
        corrupted[3] = replace(corrupted[3], high=1.0, low=100.0)

        result = ingest_binance_ohlcv(
            connection,
            app_config,
            client=StubBinanceClient(corrupted),
            full_refresh=True,
            now=NOW,
        )
        assert result.rejected == 1
        assert result.written == 9
        assert repo.ohlcv_coverage(connection, "binance", "BTCUSDT")["rows"] == 9

    def test_explicit_start_is_passed_through(
        self, connection: sqlite3.Connection, app_config: AppConfig, candles: list[Candle]
    ) -> None:
        client = StubBinanceClient(candles[:5])
        result = ingest_binance_ohlcv(
            connection, app_config, client=client, start="2019-05-01", now=NOW
        )
        assert client.requested_start == "2019-05-01"
        assert result.mode == "explicit"

    def test_missing_candle_dates_finds_an_internal_gap(
        self, connection: sqlite3.Connection, app_config: AppConfig, candles: list[Candle]
    ) -> None:
        subset = candles[:20] + candles[25:40]
        with transaction(connection):
            repo.upsert_candles(connection, subset)
        missing = missing_candle_dates(connection, app_config)
        assert len(missing) == 5
        assert missing[0] == candles[20].date_str

    def test_no_gap_reported_for_a_complete_series(
        self, connection: sqlite3.Connection, app_config: AppConfig, candles: list[Candle]
    ) -> None:
        with transaction(connection):
            repo.upsert_candles(connection, candles)
        assert missing_candle_dates(connection, app_config) == []


class TestConfig:
    def test_repository_config_loads(self, app_config: AppConfig) -> None:
        assert app_config.project.timezone == "UTC"
        assert app_config.market.primary.exchange == "binance"
        assert app_config.market.primary.symbol == "BTCUSDT"
        assert app_config.market.secondary.exchange == "coinbase"

    def test_quantiles_cover_the_required_intervals(self, app_config: AppConfig) -> None:
        """MODEL_SPEC.md section 3 needs these seven to form 50/80/95% bands."""
        assert app_config.forecast.quantiles == (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)

    def test_database_path_is_inside_the_project(self, app_config: AppConfig) -> None:
        assert app_config.storage.database_path.is_relative_to(app_config.paths.root)

    def test_no_api_credentials_in_config(self, app_config: AppConfig) -> None:
        """CLAUDE.md section 9 forbids hard-coded keys."""
        text = app_config.source_path.read_text(encoding="utf-8").lower()
        for banned in ("api_key", "apikey", "secret", "password", "token"):
            assert banned not in text, f"config.yaml appears to contain '{banned}'"

    def test_non_utc_timezone_is_rejected(self, tmp_path: Path, app_config: AppConfig) -> None:
        raw = yaml.safe_load(app_config.source_path.read_text(encoding="utf-8"))
        raw["project"]["timezone"] = "Asia/Seoul"
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ConfigError, match="must be UTC"):
            load_config(path)

    def test_unordered_quantiles_are_rejected(
        self, tmp_path: Path, app_config: AppConfig
    ) -> None:
        raw = yaml.safe_load(app_config.source_path.read_text(encoding="utf-8"))
        raw["forecast"]["quantiles"] = [0.5, 0.1, 0.9]
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ConfigError, match="ascending order"):
            load_config(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_notifications_stay_disabled(self, app_config: AppConfig) -> None:
        """OPERATING_SPEC.md section 10: no notification system in scope."""
        assert app_config.raw["notifications"]["enabled"] is False
