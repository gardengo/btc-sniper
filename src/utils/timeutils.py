"""UTC time helpers.

CLAUDE.md section 9: every timestamp handled internally is UTC. Conversions to a
local timezone are a presentation concern and never happen in this module.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

MS_PER_DAY: int = 86_400_000
UTC = timezone.utc


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(tz=UTC)


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a trailing ``Z``."""
    return to_iso(utc_now())


def to_iso(dt: datetime) -> str:
    """Serialise a datetime as an ISO-8601 UTC string (second resolution)."""
    return ensure_utc(dt).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_utc(dt: datetime) -> datetime:
    """Attach UTC to a naive datetime, or convert an aware one to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def parse_date(value: str | date | datetime) -> date:
    """Parse ``YYYY-MM-DD`` (or a date/datetime) into a UTC calendar date."""
    if isinstance(value, datetime):
        return ensure_utc(value).date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip()[:10])


def date_to_ms(value: str | date | datetime) -> int:
    """Epoch milliseconds of 00:00:00 UTC on the given calendar date."""
    d = parse_date(value)
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)


def ms_to_datetime(ms: int) -> datetime:
    """Epoch milliseconds -> aware UTC datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def ms_to_date(ms: int) -> date:
    """Epoch milliseconds -> UTC calendar date."""
    return ms_to_datetime(ms).date()


def ms_to_date_str(ms: int) -> str:
    """Epoch milliseconds -> ``YYYY-MM-DD`` UTC string."""
    return ms_to_date(ms).isoformat()


def daily_open_ms(ms: int) -> int:
    """Floor epoch milliseconds to the start of its UTC day."""
    return (ms // MS_PER_DAY) * MS_PER_DAY


def last_closed_daily_open_ms(now: datetime | None = None) -> int:
    """Open time of the most recent *fully closed* UTC daily candle.

    DATA_SPEC.md section 5: today's still-forming candle must never be used as a
    closed-day observation, so the current UTC day is excluded.
    """
    reference = ensure_utc(now) if now is not None else utc_now()
    today_open = date_to_ms(reference.date())
    return today_open - MS_PER_DAY


def add_days(value: str | date | datetime, days: int) -> date:
    """Calendar date ``days`` after the given date (UTC)."""
    return parse_date(value) + timedelta(days=days)


def date_range_days(start: str | date, end: str | date) -> int:
    """Inclusive day count between two UTC calendar dates."""
    return (parse_date(end) - parse_date(start)).days + 1
