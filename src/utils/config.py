"""Typed access to ``config.yaml``.

Configuration is centralised (CLAUDE.md section 9). Modules receive a parsed
:class:`AppConfig` instead of reading YAML or environment variables themselves.
Sections that later phases will consume are still reachable as raw mappings via
:meth:`AppConfig.section`, so adding a key never requires touching this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"
CONFIG_PATH_ENV_VAR: str = "BTC_SNIPER_CONFIG"


class ConfigError(RuntimeError):
    """Raised when the configuration file is missing or structurally invalid."""


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required config key '{where}.{key}'")
    return mapping[key]


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    timezone: str
    base_currency: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectConfig":
        tz = str(data.get("timezone", "UTC"))
        if tz.upper() != "UTC":
            raise ConfigError(f"project.timezone must be UTC, got '{tz}'")
        return cls(
            name=str(_require(data, "name", "project")),
            timezone="UTC",
            base_currency=str(data.get("base_currency", "USD")),
        )


@dataclass(frozen=True)
class PathsConfig:
    root: Path
    data_dir: Path
    logs_dir: Path
    reports_dir: Path
    models_dir: Path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], root: Path) -> "PathsConfig":
        def resolve(key: str, default: str) -> Path:
            raw = Path(str(data.get(key, default)))
            return raw if raw.is_absolute() else root / raw

        return cls(
            root=root,
            data_dir=resolve("data_dir", "data"),
            logs_dir=resolve("logs_dir", "logs"),
            reports_dir=resolve("reports_dir", "reports"),
            models_dir=resolve("models_dir", "artifacts/models"),
        )

    def ensure(self) -> None:
        """Create the runtime directories if they do not exist yet."""
        for path in (self.data_dir, self.logs_dir, self.reports_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    file: Path | None
    console: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], root: Path) -> "LoggingConfig":
        raw_file = data.get("file")
        path: Path | None = None
        if raw_file:
            candidate = Path(str(raw_file))
            path = candidate if candidate.is_absolute() else root / candidate
        return cls(
            level=str(data.get("level", "INFO")).upper(),
            file=path,
            console=bool(data.get("console", True)),
        )


@dataclass(frozen=True)
class MarketSource:
    exchange: str
    symbol: str
    timeframe: str = "1d"
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> "MarketSource":
        return cls(
            exchange=str(_require(data, "exchange", where)),
            symbol=str(_require(data, "symbol", where)),
            timeframe=str(data.get("timeframe", "1d")),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass(frozen=True)
class MarketConfig:
    primary: MarketSource
    secondary: MarketSource

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarketConfig":
        return cls(
            primary=MarketSource.from_dict(
                _require(data, "primary", "market"), "market.primary"
            ),
            secondary=MarketSource.from_dict(
                _require(data, "secondary", "market"), "market.secondary"
            ),
        )


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff parameters for a single HTTP client."""

    max_retries: int = 5
    initial_seconds: float = 1.0
    multiplier: float = 2.0
    max_seconds: float = 30.0

    def delay_for(self, attempt: int) -> float:
        """Backoff delay in seconds before retry ``attempt`` (1-based)."""
        if attempt < 1:
            raise ValueError("attempt must be >= 1")
        delay = self.initial_seconds * (self.multiplier ** (attempt - 1))
        return min(delay, self.max_seconds)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RetryPolicy":
        return cls(
            max_retries=int(data.get("max_retries", 5)),
            initial_seconds=float(data.get("backoff_initial_seconds", 1.0)),
            multiplier=float(data.get("backoff_multiplier", 2.0)),
            max_seconds=float(data.get("backoff_max_seconds", 30.0)),
        )


@dataclass(frozen=True)
class BinanceIngestionConfig:
    base_url: str
    klines_endpoint: str
    ticker_price_endpoint: str
    max_limit: int
    history_start: str
    request_timeout_seconds: float
    min_request_interval_seconds: float
    retry: RetryPolicy

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BinanceIngestionConfig":
        return cls(
            base_url=str(data.get("base_url", "https://api.binance.com")).rstrip("/"),
            klines_endpoint=str(data.get("klines_endpoint", "/api/v3/klines")),
            ticker_price_endpoint=str(
                data.get("ticker_price_endpoint", "/api/v3/ticker/price")
            ),
            max_limit=int(data.get("max_limit", 1000)),
            history_start=str(data.get("history_start", "2017-08-17")),
            request_timeout_seconds=float(data.get("request_timeout_seconds", 20)),
            min_request_interval_seconds=float(
                data.get("min_request_interval_seconds", 0.25)
            ),
            retry=RetryPolicy.from_dict(data),
        )


@dataclass(frozen=True)
class CoinbaseIngestionConfig:
    base_url: str
    candles_endpoint: str
    granularity_seconds: int
    max_candles_per_request: int
    request_timeout_seconds: float
    min_request_interval_seconds: float
    retry: RetryPolicy

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CoinbaseIngestionConfig":
        return cls(
            base_url=str(
                data.get("base_url", "https://api.exchange.coinbase.com")
            ).rstrip("/"),
            candles_endpoint=str(
                data.get("candles_endpoint", "/products/{product_id}/candles")
            ),
            granularity_seconds=int(data.get("granularity_seconds", 86400)),
            max_candles_per_request=int(data.get("max_candles_per_request", 300)),
            request_timeout_seconds=float(data.get("request_timeout_seconds", 20)),
            min_request_interval_seconds=float(
                data.get("min_request_interval_seconds", 0.25)
            ),
            retry=RetryPolicy.from_dict(data),
        )


@dataclass(frozen=True)
class IngestionConfig:
    binance: BinanceIngestionConfig
    coinbase: CoinbaseIngestionConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IngestionConfig":
        return cls(
            binance=BinanceIngestionConfig.from_dict(data.get("binance", {})),
            coinbase=CoinbaseIngestionConfig.from_dict(data.get("coinbase", {})),
        )


@dataclass(frozen=True)
class RealtimeConfig:
    """Realtime price stream settings.

    Realtime data exists for UI freshness only (DATA_SPEC.md section 1). None of
    these values can influence the model or a stored daily candle.
    """

    source: str
    rest_fallback: bool
    dashboard_refresh_seconds: float
    websocket_url: str
    stream_suffix: str
    reconnect_initial_seconds: float
    reconnect_multiplier: float
    reconnect_max_seconds: float
    reconnect_jitter_seconds: float
    heartbeat_timeout_seconds: float
    ping_interval_seconds: float
    ping_timeout_seconds: float
    close_timeout_seconds: float
    max_price_age_seconds: float
    persist_interval_seconds: float
    retention_hours: int
    max_deviation_from_last_close_pct: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RealtimeConfig":
        return cls(
            source=str(data.get("source", "binance_websocket")),
            rest_fallback=bool(data.get("rest_fallback", True)),
            dashboard_refresh_seconds=float(data.get("dashboard_refresh_seconds", 5)),
            websocket_url=str(
                data.get("websocket_url", "wss://stream.binance.com:9443/ws")
            ).rstrip("/"),
            stream_suffix=str(data.get("stream_suffix", "@miniTicker")),
            reconnect_initial_seconds=float(data.get("reconnect_initial_seconds", 1.0)),
            reconnect_multiplier=float(data.get("reconnect_multiplier", 2.0)),
            reconnect_max_seconds=float(data.get("reconnect_max_seconds", 60.0)),
            reconnect_jitter_seconds=float(data.get("reconnect_jitter_seconds", 0.5)),
            heartbeat_timeout_seconds=float(data.get("heartbeat_timeout_seconds", 30)),
            ping_interval_seconds=float(data.get("ping_interval_seconds", 20)),
            ping_timeout_seconds=float(data.get("ping_timeout_seconds", 20)),
            close_timeout_seconds=float(data.get("close_timeout_seconds", 3)),
            max_price_age_seconds=float(data.get("max_price_age_seconds", 60)),
            persist_interval_seconds=float(data.get("persist_interval_seconds", 2.0)),
            retention_hours=int(data.get("retention_hours", 48)),
            max_deviation_from_last_close_pct=float(
                data.get("max_deviation_from_last_close_pct", 50.0)
            ),
        )

    def stream_path(self, symbol: str) -> str:
        """Full raw-stream URL for a symbol (Binance requires lowercase)."""
        return f"{self.websocket_url}/{symbol.lower()}{self.stream_suffix}"

    def reconnect_delay(self, attempt: int) -> float:
        """Backoff before reconnect ``attempt`` (1-based), before jitter."""
        if attempt < 1:
            raise ValueError("attempt must be >= 1")
        delay = self.reconnect_initial_seconds * (
            self.reconnect_multiplier ** (attempt - 1)
        )
        return min(delay, self.reconnect_max_seconds)


@dataclass(frozen=True)
class DataQualityConfig:
    max_missing_day_ratio: float
    require_no_duplicate_keys: bool
    require_monotonic_open_time: bool
    require_valid_ohlc_relationship: bool
    require_non_negative_volume: bool
    extreme_daily_log_return_warn: float
    zero_volume_warn: bool
    cross_exchange_close_diff_warn_pct: float
    cross_exchange_check_days: int
    max_staleness_days: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataQualityConfig":
        return cls(
            max_missing_day_ratio=float(data.get("max_missing_day_ratio", 0.005)),
            require_no_duplicate_keys=bool(data.get("require_no_duplicate_keys", True)),
            require_monotonic_open_time=bool(
                data.get("require_monotonic_open_time", True)
            ),
            require_valid_ohlc_relationship=bool(
                data.get("require_valid_ohlc_relationship", True)
            ),
            require_non_negative_volume=bool(
                data.get("require_non_negative_volume", True)
            ),
            extreme_daily_log_return_warn=float(
                data.get("extreme_daily_log_return_warn", 0.25)
            ),
            zero_volume_warn=bool(data.get("zero_volume_warn", True)),
            cross_exchange_close_diff_warn_pct=float(
                data.get("cross_exchange_close_diff_warn_pct", 1.0)
            ),
            cross_exchange_check_days=int(data.get("cross_exchange_check_days", 120)),
            max_staleness_days=int(data.get("max_staleness_days", 2)),
        )


@dataclass(frozen=True)
class HorizonGridConfig:
    daily_until: int
    every_n_days_31_90: int
    every_n_days_91_180: int
    every_n_days_181_365: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HorizonGridConfig":
        return cls(
            daily_until=int(data.get("daily_until", 30)),
            every_n_days_31_90=int(data.get("every_n_days_31_90", 3)),
            every_n_days_91_180=int(data.get("every_n_days_91_180", 7)),
            every_n_days_181_365=int(data.get("every_n_days_181_365", 14)),
        )


@dataclass(frozen=True)
class ForecastConfig:
    anchor: str
    target: str
    max_horizon_days: int
    horizon_grid_version: str
    horizon_grid: HorizonGridConfig
    required_evaluation_horizons: tuple[int, ...]
    quantiles: tuple[float, ...]
    interpolation: str
    show_intervals: tuple[float, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ForecastConfig":
        visualization = data.get("visualization", {})
        quantiles = tuple(float(q) for q in data.get("quantiles", []))
        if not quantiles:
            raise ConfigError("forecast.quantiles must not be empty")
        if sorted(quantiles) != list(quantiles):
            raise ConfigError("forecast.quantiles must be listed in ascending order")
        return cls(
            anchor=str(data.get("anchor", "last_closed_daily_candle")),
            target=str(data.get("target", "future_log_return")),
            max_horizon_days=int(data.get("max_horizon_days", 365)),
            horizon_grid_version=str(data.get("horizon_grid_version", "v1")),
            horizon_grid=HorizonGridConfig.from_dict(data.get("horizon_grid", {})),
            required_evaluation_horizons=tuple(
                int(h)
                for h in data.get("required_evaluation_horizons", [1, 7, 30, 90, 180, 365])
            ),
            quantiles=quantiles,
            interpolation=str(visualization.get("interpolation", "pchip")),
            show_intervals=tuple(
                float(i) for i in visualization.get("show_intervals", [])
            ),
        )


@dataclass(frozen=True)
class FeaturesConfig:
    version: str
    groups: tuple[str, ...]
    min_warmup_days: int
    params: Mapping[str, Any]
    regime: Mapping[str, Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeaturesConfig":
        return cls(
            version=str(data.get("version", "v1")),
            groups=tuple(str(g) for g in data.get("groups", [])),
            min_warmup_days=int(data.get("min_warmup_days", 365)),
            params=dict(data.get("params", {})),
            regime=dict(data.get("regime", {})),
        )


@dataclass(frozen=True)
class StorageConfig:
    database_path: Path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], data_dir: Path) -> "StorageConfig":
        raw = Path(str(data.get("database", "btc_forecast.db")))
        return cls(database_path=raw if raw.is_absolute() else data_dir / raw)


@dataclass(frozen=True)
class AppConfig:
    """Parsed configuration plus the raw mapping for not-yet-typed sections."""

    config_version: str
    source_path: Path
    project: ProjectConfig
    paths: PathsConfig
    logging: LoggingConfig
    market: MarketConfig
    ingestion: IngestionConfig
    realtime: RealtimeConfig
    data_quality: DataQualityConfig
    forecast: ForecastConfig
    features: FeaturesConfig
    storage: StorageConfig
    raw: Mapping[str, Any] = field(repr=False, default_factory=dict)

    def section(self, name: str) -> Mapping[str, Any]:
        """Raw mapping for a config section (used by not-yet-typed sections)."""
        value = self.raw.get(name, {})
        if not isinstance(value, Mapping):
            raise ConfigError(f"config section '{name}' is not a mapping")
        return value


def resolve_config_path(path: str | Path | None = None) -> Path:
    """Resolve the config file path: explicit argument > env var > repo default."""
    if path is not None:
        return Path(path).resolve()
    env_value = os.environ.get(CONFIG_PATH_ENV_VAR)
    if env_value:
        return Path(env_value).resolve()
    return DEFAULT_CONFIG_PATH


def load_config(path: str | Path | None = None) -> AppConfig:
    """Parse ``config.yaml`` into an :class:`AppConfig`."""
    config_path = resolve_config_path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"config file must contain a mapping: {config_path}")

    root = config_path.parent
    paths = PathsConfig.from_dict(raw.get("paths", {}), root)
    return AppConfig(
        config_version=str(raw.get("config_version", "v1")),
        source_path=config_path,
        project=ProjectConfig.from_dict(_require(raw, "project", "<root>")),
        paths=paths,
        logging=LoggingConfig.from_dict(raw.get("logging", {}), root),
        market=MarketConfig.from_dict(_require(raw, "market", "<root>")),
        ingestion=IngestionConfig.from_dict(raw.get("ingestion", {})),
        realtime=RealtimeConfig.from_dict(raw.get("realtime", {})),
        data_quality=DataQualityConfig.from_dict(raw.get("data_quality", {})),
        forecast=ForecastConfig.from_dict(_require(raw, "forecast", "<root>")),
        features=FeaturesConfig.from_dict(raw.get("features", {})),
        storage=StorageConfig.from_dict(raw.get("storage", {}), paths.data_dir),
        raw=raw,
    )


@lru_cache(maxsize=4)
def _cached_config(resolved: Path) -> AppConfig:
    return load_config(resolved)


def get_config(path: str | Path | None = None) -> AppConfig:
    """Cached :func:`load_config` for processes that read config repeatedly."""
    return _cached_config(resolve_config_path(path))
