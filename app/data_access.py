"""Everything the dashboard reads, with no Streamlit in it.

Two reasons this is a separate module. The app opens the database **read-only**,
so no amount of clicking can mutate stored forecasts or promote a model; and
every question the pages ask is answered by a plain function over a connection,
which means the answers can be tested without rendering anything.

The pages are deliberately thin on top of this. A number that needs explaining
gets explained here, next to the query that produced it, rather than in a caption
that drifts away from the data.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from src.data.binance import SOURCE as BINANCE_SOURCE
from src.data.realtime import PriceSnapshot, latest_snapshot
from src.forecast.interpolate import interpolated_forecast_curve
from src.models import registry
from src.storage import repositories as repo
from src.utils.config import AppConfig
from src.utils.timeutils import utc_now

MEDIAN_LEVEL: float = 0.50
STATUS_FULL: str = "fully_evaluated"
# 1M / 3M / 6M / 12M, the summary OPERATING_SPEC.md section 8 asks the dashboard
# for. They are grid horizons, so no interpolated value is ever shown as a number.
SUMMARY_HORIZONS: tuple[int, ...] = (30, 90, 180, 365)


@dataclass(frozen=True)
class ForecastView:
    """One stored forecast, assembled for display."""

    forecast_id: str
    origin_date: pd.Timestamp
    origin_close: float
    current_price: float | None
    model_version: str
    created_at: str
    is_production: bool
    points: pd.DataFrame
    prices: pd.DataFrame = field(repr=False)
    curve: pd.DataFrame = field(repr=False)

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(int(h) for h in self.points["horizon_days"])

    @property
    def max_horizon(self) -> int:
        return int(self.points["horizon_days"].max())

    def provenance(self) -> pd.DataFrame:
        """How much of each horizon came from the model rather than the baseline."""
        if "blend_source" not in self.points.columns:
            return pd.DataFrame()
        frame = self.points[["horizon_days", "model_weight", "blend_source"]].copy()
        return frame.dropna(subset=["blend_source"])


@dataclass(frozen=True)
class PerformanceView:
    """What is known about how the model performs, and from where."""

    production: pd.DataFrame
    validation: pd.DataFrame
    outer_test: pd.DataFrame
    realized: int
    pending: int

    @property
    def has_production_evidence(self) -> bool:
        return self.realized > 0


def read_only_connection(config: AppConfig) -> sqlite3.Connection:
    """A connection the dashboard cannot write through."""
    from src.storage.db import connect

    return connect(config.storage.database_path, read_only=True)


# ---------------------------------------------------------------------------
# prices
# ---------------------------------------------------------------------------


def price_history(
    connection: sqlite3.Connection, config: AppConfig, *, days: int = 365
) -> pd.DataFrame:
    """Recent closed daily candles, newest last."""
    market = config.market.primary
    frame = repo.load_ohlcv(
        connection, BINANCE_SOURCE, market.symbol, market.timeframe
    )
    if frame.empty:
        return frame
    return frame.tail(max(int(days), 1))[["open", "high", "low", "close", "volume"]]


def current_price(
    connection: sqlite3.Connection, config: AppConfig, *, now: datetime | None = None
) -> PriceSnapshot | None:
    """The newest stored realtime tick, or ``None`` when the stream never ran.

    Read-only on purpose: the dashboard displays what `jobs.stream_realtime_price`
    has collected and never fetches on its own. A page that called the exchange on
    every rerun would hammer the API from every open browser tab, and an outage
    would look like a broken dashboard rather than a stale price. When the tick is
    old the page says so -- CLAUDE.md section 2.4 wants the current price live, and
    a number that is quietly 40 minutes old is worse than one labelled stale.
    """
    return latest_snapshot(connection, config, now=now or utc_now())


# ---------------------------------------------------------------------------
# forecasts
# ---------------------------------------------------------------------------


def production_version(connection: sqlite3.Connection) -> str | None:
    row = registry.production_model(connection)
    return None if row is None else str(row["model_version"])


def load_forecast_view(
    connection: sqlite3.Connection,
    config: AppConfig,
    *,
    model_version: str | None = None,
) -> ForecastView | None:
    """The newest stored forecast, with its chart curve already built."""
    market = config.market.primary
    row = repo.latest_forecast_row(
        connection,
        source=BINANCE_SOURCE,
        symbol=market.symbol,
        timeframe=market.timeframe,
        model_version=model_version,
    )
    if row is None:
        return None

    forecast_id = str(row["forecast_id"])
    points = repo.load_forecast_points(connection, forecast_id)
    prices = repo.load_forecast_quantiles(connection, forecast_id)
    returns = repo.load_forecast_quantiles(
        connection, forecast_id, value_column="predicted_log_return"
    )
    origin_date = pd.Timestamp(str(row["forecast_origin_date"]))
    origin_close = float(row["origin_close"])
    curve = (
        interpolated_forecast_curve(returns, origin_close, origin_date)
        if not returns.empty
        else pd.DataFrame()
    )
    return ForecastView(
        forecast_id=forecast_id,
        origin_date=origin_date,
        origin_close=origin_close,
        current_price=None if row["current_price"] is None else float(row["current_price"]),
        model_version=str(row["model_version"]),
        created_at=str(row["created_at"]),
        is_production=str(row["model_version"]) == production_version(connection),
        points=points,
        prices=prices,
        curve=curve,
    )


def horizon_summary(
    view: ForecastView, *, horizons: tuple[int, ...] = SUMMARY_HORIZONS
) -> pd.DataFrame:
    """The 1M / 3M / 6M / 12M table, in prices and in percent of today's close."""
    if view.points.empty:
        return pd.DataFrame()
    points = view.points.set_index("horizon_days")
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        if horizon not in points.index or horizon not in view.prices.index:
            continue
        band = view.prices.loc[horizon]
        median = float(points.loc[horizon, "predicted_price"])
        rows.append(
            {
                "horizon": _label(horizon),
                "horizon_days": int(horizon),
                "target_date": str(points.loc[horizon, "target_date"]),
                "median": median,
                "change_pct": 100.0 * (median / view.origin_close - 1.0),
                "low_95": float(band.min()),
                "high_95": float(band.max()),
                "source": str(points.loc[horizon].get("blend_source") or "unknown"),
            }
        )
    return pd.DataFrame(rows)


def _label(horizon_days: int) -> str:
    months = {30: "1M", 90: "3M", 180: "6M", 365: "12M"}
    return months.get(horizon_days, f"{horizon_days}D")


# ---------------------------------------------------------------------------
# prediction log
# ---------------------------------------------------------------------------


LOG_COLUMNS: tuple[str, ...] = (
    "forecast_origin_date",
    "target_date",
    "horizon_days",
    "evaluation_status",
    "predicted_median",
    "actual_close",
    "absolute_error",
    "percentage_error",
    "in_interval_50",
    "in_interval_80",
    "in_interval_95",
    "direction_correct",
    "regime",
    "model_version",
)


def prediction_log(
    connection: sqlite3.Connection,
    config: AppConfig,
    *,
    status: str | None = None,
    horizon_days: int | None = None,
    origin_from: str | None = None,
    origin_to: str | None = None,
    limit: int | None = 5_000,
) -> pd.DataFrame:
    """Realization rows for the prediction log, filtered as the page asks.

    The predicted median is joined in from `forecast_points`: a log that shows an
    actual price without the prediction beside it is not a log of anything.
    """
    frame = repo.load_realizations(
        connection, status=status, horizon_days=horizon_days, limit=limit
    )
    if frame.empty:
        return pd.DataFrame(columns=list(LOG_COLUMNS))

    if origin_from:
        frame = frame[frame["forecast_origin_date"] >= str(origin_from)]
    if origin_to:
        frame = frame[frame["forecast_origin_date"] <= str(origin_to)]
    if frame.empty:
        return pd.DataFrame(columns=list(LOG_COLUMNS))

    market = config.market.primary
    predicted = repo.load_forecast_points_for_realization(
        connection,
        source=BINANCE_SOURCE,
        symbol=market.symbol,
        timeframe=market.timeframe,
    )
    frame = _attach_predicted(frame, predicted)
    columns = [name for name in LOG_COLUMNS if name in frame.columns]
    return frame[columns].reset_index(drop=True)


def _attach_predicted(frame: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    if predicted.empty:
        frame["predicted_median"] = np.nan
        return frame
    keyed = predicted.set_index(["forecast_id", "horizon_days"])["predicted_price"]
    index = pd.MultiIndex.from_arrays(
        [frame["forecast_id"], frame["horizon_days"].astype(int)]
    )
    frame = frame.copy()
    frame["predicted_median"] = keyed.reindex(index).to_numpy()
    return frame


def log_filter_options(connection: sqlite3.Connection) -> dict[str, list]:
    """The values the prediction-log filters can actually take.

    Built from the stored rows rather than from config: offering a filter for a
    horizon that has never been forecast produces an empty table and no
    explanation of why.
    """
    frame = repo.load_realizations(connection, limit=None)
    if frame.empty:
        return {"statuses": [], "horizons": [], "origins": []}
    return {
        "statuses": sorted(frame["evaluation_status"].dropna().unique().tolist()),
        "horizons": sorted(int(h) for h in frame["horizon_days"].dropna().unique()),
        "origins": sorted(frame["forecast_origin_date"].dropna().unique().tolist()),
    }


# ---------------------------------------------------------------------------
# performance
# ---------------------------------------------------------------------------


def load_performance(connection: sqlite3.Connection) -> PerformanceView:
    """Every scope of metric the project records, kept apart.

    Production, validation and outer-test numbers answer different questions and
    are never mixed into one table. Validation says what the design was worth on
    held-out inner folds; production says what the deployed model has actually
    done since it started running; the outer test is the single independent
    verdict and is absent until `jobs.final_evaluation` has been run.
    """
    realizations = repo.load_realizations(connection, limit=None)
    realized = (
        int((realizations["evaluation_status"] == STATUS_FULL).sum())
        if not realizations.empty
        else 0
    )
    pending = len(realizations) - realized if not realizations.empty else 0
    return PerformanceView(
        production=repo.load_performance_metrics(connection, scope="production"),
        validation=repo.load_performance_metrics(connection, scope="validation"),
        outer_test=repo.load_performance_metrics(connection, scope="outer_test"),
        realized=realized,
        pending=pending,
    )


HEADLINE_METRICS: tuple[str, ...] = (
    "sample_size",
    "return_mae",
    "price_mae",
    "pinball_mean",
    "direction_accuracy",
    "coverage_50",
    "coverage_80",
    "coverage_95",
)


def metric_table(
    metrics: pd.DataFrame,
    *,
    regime: str = "all",
    names: tuple[str, ...] = HEADLINE_METRICS,
) -> pd.DataFrame:
    """Tidy metric rows reshaped to one row per horizon."""
    if metrics.empty:
        return pd.DataFrame()
    subset = metrics[
        (metrics["regime"] == regime) & (metrics["metric_name"].isin(names))
    ]
    if subset.empty:
        return pd.DataFrame()
    wide = subset.pivot_table(
        index="horizon_days", columns="metric_name", values="metric_value", aggfunc="mean"
    ).reset_index()
    ordered = ["horizon_days"] + [name for name in names if name in wide.columns]
    return wide[ordered].sort_values("horizon_days").reset_index(drop=True)


def regime_table(
    metrics: pd.DataFrame, *, metric_name: str = "pinball_mean"
) -> pd.DataFrame:
    """One metric broken out by market regime at the forecast origin."""
    if metrics.empty:
        return pd.DataFrame()
    subset = metrics[
        (metrics["regime"] != "all") & (metrics["metric_name"] == metric_name)
    ]
    if subset.empty:
        return pd.DataFrame()
    return (
        subset.pivot_table(
            index="horizon_days", columns="regime", values="metric_value", aggfunc="mean"
        )
        .reset_index()
        .sort_values("horizon_days")
        .reset_index(drop=True)
    )


def model_history(connection: sqlite3.Connection, *, limit: int = 50) -> pd.DataFrame:
    """The registry, newest first, with the fields a reader needs to tell them apart."""
    frame = registry.list_models(connection, limit=limit)
    if frame.empty:
        return frame
    columns = [
        "model_version",
        "status",
        "training_window_strategy",
        "training_cutoff",
        "training_rows",
        "created_at",
        "promoted_at",
        "retired_at",
    ]
    return frame[[name for name in columns if name in frame.columns]]


def revision_history(
    connection: sqlite3.Connection,
    config: AppConfig,
    *,
    horizon_days: int = 365,
    limit: int = 60,
) -> pd.DataFrame:
    """How one horizon's forecast has moved as new origins arrived.

    The point of the chart is drift in the *view*, not accuracy: a 365-day median
    that swings wildly from one day to the next is telling you the model is
    reacting to noise, and that is visible long before any of those forecasts can
    be scored.
    """
    market = config.market.primary
    frame = repo.forecast_history(
        connection,
        source=BINANCE_SOURCE,
        symbol=market.symbol,
        horizon_days=horizon_days,
        limit=limit,
    )
    if frame.empty:
        return frame
    frame = frame.sort_values("forecast_origin_date").reset_index(drop=True)
    frame["change_pct"] = 100.0 * (frame["predicted_price"] / frame["origin_close"] - 1.0)
    return frame


__all__ = [
    "ForecastView",
    "HEADLINE_METRICS",
    "LOG_COLUMNS",
    "PerformanceView",
    "SUMMARY_HORIZONS",
    "current_price",
    "horizon_summary",
    "load_forecast_view",
    "load_performance",
    "log_filter_options",
    "metric_table",
    "model_history",
    "prediction_log",
    "price_history",
    "production_version",
    "read_only_connection",
    "regime_table",
    "revision_history",
]
