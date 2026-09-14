"""Realtime price path tests.

Covers the two things that can go quietly wrong here:

1. realtime data leaking into the model's data (it must not, ever)
2. a stale price being presented as if it were live

Everything runs offline; the WebSocket is replaced by a scripted fake.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import timedelta

import pytest

from src.data.binance_stream import (
    BinancePriceStream,
    StreamParseError,
    parse_message,
)
from src.data.realtime import (
    TRANSPORT_REST,
    TRANSPORT_WEBSOCKET,
    PriceSnapshot,
    RealtimePriceError,
    get_current_price,
    is_plausible,
    latest_snapshot,
    prune_realtime_prices,
    store_tick,
)
from src.data.types import Candle, CandleError, RealtimeTick
from src.storage import repositories as repo
from src.storage.db import transaction
from src.utils.config import AppConfig
from src.utils.timeutils import UTC, utc_now

NOW = utc_now().replace(microsecond=0)


def mini_ticker(price: float, event_time_ms: int, symbol: str = "BTCUSDT") -> str:
    return json.dumps(
        {"e": "24hrMiniTicker", "E": event_time_ms, "s": symbol, "c": str(price)}
    )


def make_tick(price: float, *, offset_seconds: float = 0.0, transport: str = TRANSPORT_WEBSOCKET):
    moment = NOW - timedelta(seconds=offset_seconds)
    return RealtimeTick(
        source="binance",
        symbol="BTCUSDT",
        event_time_ms=int(moment.timestamp() * 1000),
        price=price,
        transport=transport,
    )


class StubBinanceClient:
    """Returns a fixed REST price, or raises to simulate an outage."""

    def __init__(self, price: float | None = 80_000.0, error: Exception | None = None) -> None:
        self.price = price
        self.error = error
        self.calls = 0

    def fetch_current_price(self, symbol: str) -> float:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return float(self.price)

    def close(self) -> None:
        return None


class TestMessageParsing:
    def test_mini_ticker_uses_the_close_field(self) -> None:
        tick = parse_message(mini_ticker(77_790.67, 1789344000000), symbol="BTCUSDT")
        assert tick.price == pytest.approx(77_790.67)
        assert tick.event_time_ms == 1789344000000
        assert tick.transport == TRANSPORT_WEBSOCKET

    def test_trade_uses_the_trade_time_not_the_event_time(self) -> None:
        """Trade messages carry both; the trade time is the truthful one."""
        payload = json.dumps(
            {
                "e": "trade",
                "E": 1789344000000,
                "T": 1789343999000,
                "s": "BTCUSDT",
                "p": "77791.01",
            }
        )
        tick = parse_message(payload, symbol="BTCUSDT")
        assert tick.price == pytest.approx(77_791.01)
        assert tick.event_time_ms == 1789343999000

    def test_combined_stream_envelope_is_unwrapped(self) -> None:
        payload = json.dumps(
            {
                "stream": "btcusdt@miniTicker",
                "data": json.loads(mini_ticker(100.0, 1789344000000)),
            }
        )
        assert parse_message(payload, symbol="BTCUSDT").price == pytest.approx(100.0)

    def test_symbol_is_normalised_to_upper_case(self) -> None:
        tick = parse_message(mini_ticker(1.0, 1, symbol="btcusdt"), symbol="btcusdt")
        assert tick.symbol == "BTCUSDT"

    @pytest.mark.parametrize(
        "payload",
        [
            "not json",
            json.dumps([1, 2, 3]),
            json.dumps({"e": "depthUpdate", "E": 1}),
            json.dumps({"e": "24hrMiniTicker", "E": 1}),
            json.dumps({"e": "24hrMiniTicker", "E": 1, "c": "abc"}),
            json.dumps({"result": None, "id": 1}),
        ],
    )
    def test_unusable_messages_raise(self, payload: str) -> None:
        with pytest.raises(StreamParseError):
            parse_message(payload, symbol="BTCUSDT")

    def test_non_positive_price_fails_validation(self) -> None:
        tick = parse_message(mini_ticker(-1.0, 1789344000000), symbol="BTCUSDT")
        with pytest.raises(CandleError):
            tick.validate()


class TestPlausibility:
    @pytest.mark.parametrize("price", [76_000.0, 100_000.0, 50_000.0])
    def test_prices_near_the_last_close_are_accepted(self, price: float) -> None:
        assert is_plausible(price, 77_000.0, 50.0)

    @pytest.mark.parametrize("price", [1.0, 770_000.0, 0.01])
    def test_absurd_prices_are_rejected(self, price: float) -> None:
        assert not is_plausible(price, 77_000.0, 50.0)

    def test_worst_real_crash_would_still_be_accepted(self) -> None:
        """2020-03-12 moved -39.6%; the gate must not fire on real moves."""
        assert is_plausible(77_000.0 * 0.604, 77_000.0, 50.0)

    def test_missing_reference_accepts_everything(self) -> None:
        assert is_plausible(1.0, None, 50.0)


class TestStoreAndSnapshot:
    def test_tick_round_trips(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        store_tick(connection, make_tick(77_000.0))
        snapshot = latest_snapshot(connection, app_config, now=NOW)
        assert snapshot is not None
        assert snapshot.price == pytest.approx(77_000.0)
        assert not snapshot.is_stale

    def test_latest_wins_over_older_ticks(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        store_tick(connection, make_tick(1.0, offset_seconds=30))
        store_tick(connection, make_tick(2.0, offset_seconds=10))
        store_tick(connection, make_tick(3.0, offset_seconds=20))
        snapshot = latest_snapshot(connection, app_config, now=NOW)
        assert snapshot.price == pytest.approx(2.0)

    def test_old_tick_is_flagged_stale(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        store_tick(connection, make_tick(77_000.0, offset_seconds=600))
        snapshot = latest_snapshot(connection, app_config, now=NOW)
        assert snapshot.is_stale
        assert snapshot.age_seconds == pytest.approx(600, abs=2)

    def test_empty_store_returns_none(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        assert latest_snapshot(connection, app_config, now=NOW) is None

    def test_invalid_tick_is_never_stored(
        self, connection: sqlite3.Connection
    ) -> None:
        with pytest.raises(CandleError):
            store_tick(connection, make_tick(-5.0))
        assert connection.execute("SELECT COUNT(*) FROM realtime_price").fetchone()[0] == 0


class TestCurrentPriceResolution:
    def test_fresh_tick_is_served_without_touching_rest(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        store_tick(connection, make_tick(77_000.0))
        client = StubBinanceClient(80_000.0)
        snapshot = get_current_price(connection, app_config, client=client, now=NOW)
        assert snapshot.price == pytest.approx(77_000.0)
        assert client.calls == 0

    def test_stale_tick_triggers_the_rest_fallback(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        store_tick(connection, make_tick(77_000.0, offset_seconds=600))
        client = StubBinanceClient(80_000.0)
        snapshot = get_current_price(connection, app_config, client=client, now=NOW)
        assert snapshot.price == pytest.approx(80_000.0)
        assert snapshot.transport == TRANSPORT_REST
        assert client.calls == 1

    def test_empty_store_triggers_the_rest_fallback(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        client = StubBinanceClient(80_000.0)
        snapshot = get_current_price(connection, app_config, client=client, now=NOW)
        assert snapshot.price == pytest.approx(80_000.0)
        assert client.calls == 1

    def test_fallback_price_is_persisted_for_the_next_read(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        client = StubBinanceClient(80_000.0)
        get_current_price(connection, app_config, client=client, now=NOW)
        row = repo.latest_realtime_tick(connection, "binance", "BTCUSDT")
        assert row is not None
        assert row["transport"] == TRANSPORT_REST

    def test_rest_outage_falls_back_to_the_stale_tick(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        """A stale number the user can see is better than a blank dashboard."""
        store_tick(connection, make_tick(77_000.0, offset_seconds=600))
        client = StubBinanceClient(error=ConnectionError("binance unreachable"))
        snapshot = get_current_price(connection, app_config, client=client, now=NOW)
        assert snapshot.price == pytest.approx(77_000.0)
        assert snapshot.is_stale

    def test_total_outage_with_no_history_raises(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        client = StubBinanceClient(error=ConnectionError("binance unreachable"))
        with pytest.raises(RealtimePriceError, match="no realtime price available"):
            get_current_price(connection, app_config, client=client, now=NOW)


class TestRetention:
    def test_old_rows_are_pruned(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        store_tick(connection, make_tick(1.0, offset_seconds=100 * 3600))
        store_tick(connection, make_tick(2.0, offset_seconds=10))
        deleted = prune_realtime_prices(connection, app_config, now=NOW)
        assert deleted == 1
        assert connection.execute("SELECT COUNT(*) FROM realtime_price").fetchone()[0] == 1

    def test_recent_rows_survive(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        store_tick(connection, make_tick(2.0, offset_seconds=60))
        assert prune_realtime_prices(connection, app_config, now=NOW) == 0


class TestStreamConsumer:
    def _stream(
        self, connection: sqlite3.Connection, app_config: AppConfig, reference: float | None
    ) -> BinancePriceStream:
        stream = BinancePriceStream(connection, app_config)
        stream._reference_close = reference
        return stream

    def test_first_message_is_persisted_immediately(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        stream = self._stream(connection, app_config, 77_000.0)
        assert stream.handle_message(mini_ticker(77_100.0, 1789344000000)) is True
        assert stream.stats.persisted == 1

    def test_persistence_is_throttled(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        """A flood of ticks must not become a flood of SQLite writes."""
        stream = self._stream(connection, app_config, 77_000.0)
        for index in range(50):
            stream.handle_message(mini_ticker(77_100.0 + index, 1789344000000 + index))
        assert stream.stats.received == 50
        assert stream.stats.persisted == 1
        assert stream.stats.last_tick.price == pytest.approx(77_149.0)

    def test_throttle_can_be_overridden(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        stream = self._stream(connection, app_config, 77_000.0)
        stream.handle_message(mini_ticker(77_100.0, 1789344000000))
        assert stream.handle_message(mini_ticker(77_200.0, 1789344001000), force_persist=True)
        assert stream.stats.persisted == 2
        assert stream.stats.rejected_implausible == 0

    def test_unparseable_messages_are_counted_not_raised(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        stream = self._stream(connection, app_config, 77_000.0)
        assert stream.handle_message('{"result":null,"id":1}') is False
        assert stream.stats.rejected_unparseable == 1
        assert stream.stats.persisted == 0

    def test_implausible_price_is_rejected_and_not_stored(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        stream = self._stream(connection, app_config, 77_000.0)
        assert stream.handle_message(mini_ticker(1.0, 1789344000000)) is False
        assert stream.stats.rejected_implausible == 1
        assert connection.execute("SELECT COUNT(*) FROM realtime_price").fetchone()[0] == 0

    def test_reference_close_comes_from_the_last_daily_candle(
        self, connection: sqlite3.Connection, app_config: AppConfig, candles: list[Candle]
    ) -> None:
        with transaction(connection):
            repo.upsert_candles(connection, candles)
        stream = BinancePriceStream(connection, app_config)
        assert stream.refresh_reference_close() == pytest.approx(float(candles[-1].close))

    def test_reference_close_is_none_without_candles(
        self, connection: sqlite3.Connection, app_config: AppConfig
    ) -> None:
        assert BinancePriceStream(connection, app_config).refresh_reference_close() is None


class TestModelDataIsolation:
    def test_streaming_never_writes_to_market_ohlcv(
        self, connection: sqlite3.Connection, app_config: AppConfig, candles: list[Candle]
    ) -> None:
        """DATA_SPEC.md section 1: a tick must never disturb a closed candle."""
        with transaction(connection):
            repo.upsert_candles(connection, candles)
        before = repo.load_ohlcv(connection, "binance", "BTCUSDT")

        stream = BinancePriceStream(connection, app_config)
        stream.refresh_reference_close()
        for index in range(20):
            stream.handle_message(
                mini_ticker(float(candles[-1].close) * 1.02, 1789344000000 + index * 5000),
                force_persist=True,
            )

        after = repo.load_ohlcv(connection, "binance", "BTCUSDT")
        assert len(after) == len(before)
        assert after["close"].equals(before["close"])
        assert connection.execute("SELECT COUNT(*) FROM realtime_price").fetchone()[0] == 20

    def test_realtime_price_and_daily_close_are_allowed_to_differ(
        self, connection: sqlite3.Connection, app_config: AppConfig, candles: list[Candle]
    ) -> None:
        """DATA_SPEC.md section 2 expects the live price to differ from the anchor."""
        with transaction(connection):
            repo.upsert_candles(connection, candles)
        anchor = float(candles[-1].close)
        store_tick(connection, make_tick(anchor * 1.03))

        snapshot = latest_snapshot(connection, app_config, now=NOW)
        assert snapshot.price != pytest.approx(anchor)


class FakeSocket:
    """Yields scripted messages, then raises to simulate a dropped connection."""

    def __init__(self, messages: list[str], error: Exception | None = None) -> None:
        self.messages = list(messages)
        self.error = error or ConnectionError("socket closed")

    async def recv(self) -> str:
        if self.messages:
            return self.messages.pop(0)
        raise self.error

    async def __aenter__(self) -> "FakeSocket":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class TestReconnect:
    def test_stream_reconnects_after_a_dropped_connection(
        self, connection: sqlite3.Connection, app_config: AppConfig, monkeypatch
    ) -> None:
        sessions = [
            FakeSocket([mini_ticker(77_100.0, 1789344000000)]),
            FakeSocket([mini_ticker(77_200.0, 1789344002000)]),
        ]
        opened: list[str] = []

        def factory(url: str, **kwargs: object) -> FakeSocket:
            opened.append(url)
            return sessions.pop(0) if sessions else FakeSocket([])

        stream = BinancePriceStream(connection, app_config, connect_factory=factory)
        stream._reference_close = 77_000.0
        monkeypatch.setattr(
            "src.data.binance_stream.BinancePriceStream.refresh_reference_close",
            lambda self: 77_000.0,
        )
        monkeypatch.setattr(app_config.realtime.__class__, "reconnect_delay", lambda self, n: 0.0)

        stats = asyncio.run(stream.run(max_reconnects=2))
        assert stats.reconnects == 2
        assert len(opened) >= 2
        assert stats.received >= 2

    def test_backoff_grows_then_caps(self, app_config: AppConfig) -> None:
        settings = app_config.realtime
        delays = [settings.reconnect_delay(n) for n in range(1, 9)]
        assert delays[0] == pytest.approx(settings.reconnect_initial_seconds)
        assert delays == sorted(delays)
        assert delays[-1] == pytest.approx(settings.reconnect_max_seconds)

    def test_stream_url_is_lower_case(self, app_config: AppConfig) -> None:
        url = app_config.realtime.stream_path("BTCUSDT")
        assert url.endswith("/btcusdt@miniTicker")
