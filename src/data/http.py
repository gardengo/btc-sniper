"""Shared HTTP client behaviour for the exchange REST clients.

DATA_SPEC.md section 3 requires retry with exponential backoff. Only transport
errors, HTTP 5xx and rate-limit responses are retried; a 4xx that is not a rate
limit is a programming or symbol error and is raised immediately so it surfaces
in the logs instead of being retried into a timeout.
"""

from __future__ import annotations

import time
from typing import Any, Mapping
from urllib.parse import urljoin

import requests

from src.utils.config import RetryPolicy
from src.utils.logging import get_logger

logger = get_logger(__name__)

RETRYABLE_STATUS: frozenset[int] = frozenset({408, 418, 429, 500, 502, 503, 504})
USER_AGENT: str = "btc-sniper/0.1 (research; contact via repository)"


class HttpError(RuntimeError):
    """Raised when a request fails after exhausting the retry policy."""


class RestClient:
    """Minimal JSON REST client with rate limiting and exponential backoff."""

    def __init__(
        self,
        base_url: str,
        *,
        retry: RetryPolicy,
        timeout_seconds: float = 20.0,
        min_request_interval_seconds: float = 0.0,
        session: requests.Session | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.retry = retry
        self.timeout_seconds = timeout_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self._sleep = sleep
        self._last_request_at: float = 0.0

    def _throttle(self) -> None:
        if self.min_request_interval_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.min_request_interval_seconds - elapsed
        if remaining > 0:
            self._sleep(remaining)

    def get_json(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        """GET a JSON document, retrying transient failures.

        Raises :class:`HttpError` when every attempt fails.
        """
        url = urljoin(self.base_url, path.lstrip("/"))
        last_error: Exception | None = None

        for attempt in range(1, self.retry.max_retries + 1):
            self._throttle()
            try:
                response = self._session.get(
                    url, params=dict(params or {}), timeout=self.timeout_seconds
                )
                self._last_request_at = time.monotonic()
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "request error on %s (attempt %d/%d): %s",
                    url,
                    attempt,
                    self.retry.max_retries,
                    exc,
                )
            else:
                if response.status_code == 200:
                    return response.json()
                message = (
                    f"HTTP {response.status_code} from {url}: {response.text[:200]}"
                )
                if response.status_code not in RETRYABLE_STATUS:
                    raise HttpError(message)
                last_error = HttpError(message)
                logger.warning(
                    "retryable response on %s (attempt %d/%d): %s",
                    url,
                    attempt,
                    self.retry.max_retries,
                    message,
                )

            if attempt < self.retry.max_retries:
                delay = self.retry.delay_for(attempt)
                logger.info("backing off %.2fs before retrying %s", delay, url)
                self._sleep(delay)

        raise HttpError(
            f"giving up on {url} after {self.retry.max_retries} attempts: {last_error}"
        ) from last_error

    def close(self) -> None:
        """Release the underlying session."""
        self._session.close()

    def __enter__(self) -> "RestClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
