"""Exchange client tests.

No network access: a fake transport replays recorded-shape payloads, which lets
the pagination, retry and in-progress-candle rules be tested deterministically.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
import requests

from src.data.binance import BinanceClient
from src.data.coinbase import CoinbaseClient
from src.data.http import HttpError, RestClient
from src.utils.config import AppConfig, RetryPolicy
from src.utils.timeutils import MS_PER_DAY, UTC, date_to_ms, ms_to_date_str

NOW = datetime(2018, 1, 11, 6, 0, tzinfo=UTC)


def make_kline(open_time_ms: int, close_price: float = 100.0) -> list[Any]:
    """One Binance kline array in the documented field order."""
    return [
        open_time_ms,
        "99.0",
        "105.0",
        "95.0",
        str(close_price),
        "10.0",
        open_time_ms + MS_PER_DAY - 1,
        "1000.0",
        50,
        "5.0",
        "500.0",
        "0",
    ]


class FakeRestClient:
    """Stands in for :class:`RestClient`, recording the params it was called with."""

    def __init__(self, pages: list[list[Any]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append(dict(params or {}))
        if not self.pages:
            return []
        return self.pages.pop(0)

    def close(self) -> None:
        return None


class TestBinanceClient:
    def _client(self, app_config: AppConfig, pages: list[list[Any]]) -> tuple:
        fake = FakeRestClient(pages)
        return BinanceClient(app_config.ingestion.binance, client=fake), fake

    def test_parses_every_kline_field(self, app_config: AppConfig) -> None:
        open_time = date_to_ms("2018-01-05")
        client, _ = self._client(app_config, [[make_kline(open_time, 101.5)]])
        candles = client.fetch_klines("BTCUSDT", start="2018-01-05", now=NOW)

        assert len(candles) == 1
        candle = candles[0]
        assert candle.source == "binance"
        assert candle.open_time_ms == open_time
        assert candle.close == pytest.approx(101.5)
        assert candle.high == pytest.approx(105.0)
        assert candle.quote_volume == pytest.approx(1000.0)
        assert candle.trade_count == 50
        assert candle.taker_buy_base == pytest.approx(5.0)
        assert candle.date_str == "2018-01-05"
        candle.validate()

    def test_in_progress_candle_is_never_returned(self, app_config: AppConfig) -> None:
        """DATA_SPEC.md section 5: today's forming candle is not a closed day."""
        today = date_to_ms("2018-01-11")
        yesterday = date_to_ms("2018-01-10")
        client, _ = self._client(
            app_config, [[make_kline(yesterday), make_kline(today)]]
        )
        candles = client.fetch_klines("BTCUSDT", start="2018-01-10", now=NOW)

        returned = {candle.date_str for candle in candles}
        assert returned == {"2018-01-10"}
        assert "2018-01-11" not in returned

    def test_request_end_time_is_the_last_closed_candle(self, app_config: AppConfig) -> None:
        client, fake = self._client(app_config, [[make_kline(date_to_ms("2018-01-05"))]])
        client.fetch_klines("BTCUSDT", start="2018-01-05", now=NOW)
        assert ms_to_date_str(fake.calls[0]["endTime"]) == "2018-01-10"

    def test_pagination_advances_the_cursor(self, app_config: AppConfig) -> None:
        limit = app_config.ingestion.binance.max_limit
        first_page = [
            make_kline(date_to_ms("2018-01-01") + i * MS_PER_DAY) for i in range(limit)
        ]
        second_page = [make_kline(date_to_ms("2018-01-01") + limit * MS_PER_DAY)]
        client, fake = self._client(app_config, [first_page, second_page])

        candles = client.fetch_klines(
            "BTCUSDT",
            start="2018-01-01",
            end_open_time_ms=date_to_ms("2018-01-01") + limit * MS_PER_DAY,
            now=NOW,
        )
        assert len(candles) == limit + 1
        assert len(fake.calls) == 2
        assert fake.calls[1]["startTime"] > fake.calls[0]["startTime"]

    def test_results_are_sorted_and_unique(self, app_config: AppConfig) -> None:
        base = date_to_ms("2018-01-01")
        shuffled = [make_kline(base + i * MS_PER_DAY) for i in (3, 1, 0, 2)]
        client, _ = self._client(app_config, [shuffled])
        candles = client.fetch_klines("BTCUSDT", start="2018-01-01", now=NOW)

        open_times = [candle.open_time_ms for candle in candles]
        assert open_times == sorted(open_times)

    def test_empty_response_stops_cleanly(self, app_config: AppConfig) -> None:
        client, _ = self._client(app_config, [[]])
        assert client.fetch_klines("BTCUSDT", start="2018-01-01", now=NOW) == []

    def test_start_after_last_closed_candle_fetches_nothing(
        self, app_config: AppConfig
    ) -> None:
        client, fake = self._client(app_config, [[make_kline(date_to_ms("2018-01-05"))]])
        assert client.fetch_klines("BTCUSDT", start="2018-02-01", now=NOW) == []
        assert fake.calls == []

    def test_malformed_payload_raises(self, app_config: AppConfig) -> None:
        client, _ = self._client(app_config, [[[1, 2, 3]]])
        with pytest.raises(ValueError, match="unexpected kline payload"):
            client.fetch_klines("BTCUSDT", start="2018-01-05", now=NOW)

    def test_non_daily_timeframe_is_rejected(self, app_config: AppConfig) -> None:
        client, _ = self._client(app_config, [])
        with pytest.raises(ValueError, match="only the 1d timeframe"):
            client.fetch_klines("BTCUSDT", "4h", start="2018-01-01", now=NOW)


class TestCoinbaseClient:
    def test_parses_the_candle_field_order(self, app_config: AppConfig) -> None:
        """Coinbase order is [time, low, high, open, close, volume]."""
        seconds = date_to_ms("2018-01-05") // 1000
        fake = FakeRestClient([[[seconds, 95.0, 105.0, 99.0, 101.0, 10.0]]])
        client = CoinbaseClient(app_config.ingestion.coinbase, client=fake)

        candles = client.fetch_daily_candles("BTC-USD", start="2018-01-05", now=NOW)
        assert len(candles) == 1
        candle = candles[0]
        assert candle.source == "coinbase"
        assert candle.low == pytest.approx(95.0)
        assert candle.high == pytest.approx(105.0)
        assert candle.open == pytest.approx(99.0)
        assert candle.close == pytest.approx(101.0)
        candle.validate()

    def test_in_progress_candle_is_excluded(self, app_config: AppConfig) -> None:
        today = date_to_ms("2018-01-11") // 1000
        yesterday = date_to_ms("2018-01-10") // 1000
        fake = FakeRestClient(
            [
                [
                    [today, 95.0, 105.0, 99.0, 101.0, 10.0],
                    [yesterday, 95.0, 105.0, 99.0, 100.0, 10.0],
                ]
            ]
        )
        client = CoinbaseClient(app_config.ingestion.coinbase, client=fake)
        candles = client.fetch_daily_candles("BTC-USD", start="2018-01-10", now=NOW)
        assert [candle.date_str for candle in candles] == ["2018-01-10"]


class FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.headers: dict[str, str] = {}
        self.request_count = 0

    def get(self, url: str, params: Any = None, timeout: float | None = None) -> Any:
        self.request_count += 1
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self) -> None:
        return None


class TestRestClientRetry:
    def _client(self, session: FakeSession, max_retries: int = 3) -> RestClient:
        slept: list[float] = []
        client = RestClient(
            "https://example.test",
            retry=RetryPolicy(max_retries=max_retries, initial_seconds=0.0, max_seconds=0.0),
            session=session,
            sleep=slept.append,
        )
        client.slept = slept  # type: ignore[attr-defined]
        return client

    def test_retries_a_500_then_succeeds(self) -> None:
        session = FakeSession([FakeResponse(500, text="boom"), FakeResponse(200, {"ok": 1})])
        client = self._client(session)
        assert client.get_json("/x") == {"ok": 1}
        assert session.request_count == 2

    def test_retries_a_rate_limit(self) -> None:
        session = FakeSession([FakeResponse(429, text="slow down"), FakeResponse(200, [])])
        client = self._client(session)
        assert client.get_json("/x") == []
        assert session.request_count == 2

    def test_retries_a_transport_error(self) -> None:
        session = FakeSession(
            [requests.ConnectionError("reset"), FakeResponse(200, {"ok": 1})]
        )
        client = self._client(session)
        assert client.get_json("/x") == {"ok": 1}

    def test_gives_up_after_max_retries(self) -> None:
        session = FakeSession([FakeResponse(503, text="down") for _ in range(3)])
        client = self._client(session, max_retries=3)
        with pytest.raises(HttpError, match="giving up"):
            client.get_json("/x")
        assert session.request_count == 3

    def test_client_error_is_not_retried(self) -> None:
        """A 400 is a bug, not a blip; retrying it just delays the report."""
        session = FakeSession([FakeResponse(400, text="bad symbol")])
        client = self._client(session)
        with pytest.raises(HttpError, match="HTTP 400"):
            client.get_json("/x")
        assert session.request_count == 1

    def test_backoff_grows_exponentially(self) -> None:
        policy = RetryPolicy(max_retries=6, initial_seconds=1.0, multiplier=2.0, max_seconds=10.0)
        assert [policy.delay_for(n) for n in range(1, 6)] == [1.0, 2.0, 4.0, 8.0, 10.0]
