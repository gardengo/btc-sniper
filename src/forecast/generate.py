"""Producing one dated forecast across the whole horizon grid.

The anchor rule (DATA_SPEC.md section 2, CLAUDE.md section 2.4)
--------------------------------------------------------------
A forecast is anchored on the **last closed daily candle**, never on the live
price. The live price is carried alongside it as `current_price` for display, and
the two are expected to differ. Anchoring on the live price would mean the whole
one-year forecast moves on every tick, which is exactly what CLAUDE.md section
2.4 forbids.

Composition
-----------
Each horizon's quantiles are a blend of the model and the baseline, weighted by
`forecast.blend` (see `src/forecast/blending.py`). Horizons where the model has
no validated advantage are served as the baseline, and every point records which
it was, so a stored forecast can always be traced back to what produced it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.forecast.blending import (
    BlendError,
    BlendPolicy,
    blend_quantiles,
    blend_source_label,
    to_prices,
)
from src.forecast.quantiles import (
    count_crossing_rows,
    count_crossings,
    enforce_ordering,
    quantile_label,
)
from src.models.baselines import REFERENCE_BASELINE, BASELINE_CLASSES
from src.models.dataset import build_inference_matrix
from src.models.forecaster import MultiHorizonForecaster
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.utils.provenance import code_commit
from src.utils.timeutils import utc_now_iso

logger = get_logger(__name__)


class ForecastError(RuntimeError):
    """Raised when a forecast cannot be produced for an origin."""


@dataclass(frozen=True)
class Forecast:
    """One origin's forecast over the whole horizon grid."""

    forecast_id: str
    origin_date: pd.Timestamp
    origin_close: float
    model_version: str
    feature_version: str
    horizon_grid_version: str
    config_version: str
    code_commit: str
    created_at: str
    points: pd.DataFrame
    quantiles: pd.DataFrame
    current_price: float | None = None
    blend: str = ""
    crossings: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(sorted(self.points["horizon_days"].unique()))

    def quantile_matrix(self) -> pd.DataFrame:
        """Wide log-return quantiles indexed by horizon, for interpolation."""
        return self.quantiles.pivot(
            index="horizon_days", columns="quantile", values="predicted_log_return"
        ).sort_index()

    def describe(self) -> str:
        median = self.points.set_index("horizon_days")["predicted_price"]
        parts = [
            f"{h}d={median.loc[h]:,.0f}" for h in (30, 90, 365) if h in median.index
        ]
        return (
            f"{self.origin_date:%Y-%m-%d} anchor={self.origin_close:,.2f} "
            f"model={self.model_version} horizons={len(self.horizons)} "
            + (" ".join(parts))
        )


def _baseline_quantiles(
    close: pd.Series, horizon: int, levels: tuple[float, ...], config: AppConfig
) -> pd.DataFrame:
    """Reference-baseline quantiles for every origin, at one horizon."""
    baseline = BASELINE_CLASSES[REFERENCE_BASELINE](config.baselines)
    return baseline.predict(close, horizon, levels).quantiles


def generate_forecast(
    features: pd.DataFrame,
    close: pd.Series,
    config: AppConfig,
    forecaster: MultiHorizonForecaster,
    horizons: tuple[int, ...],
    *,
    origin: pd.Timestamp | None = None,
    current_price: float | None = None,
    policy: BlendPolicy | None = None,
    run_id: str | None = None,
) -> Forecast:
    """Forecast every horizon from one origin."""
    if not horizons:
        raise ForecastError("no horizons requested")

    usable = features.dropna(axis=0, how="any")
    if usable.empty:
        raise ForecastError("no complete feature row is available to forecast from")
    anchor = pd.Timestamp(origin) if origin is not None else usable.index.max()
    if anchor not in usable.index:
        raise ForecastError(
            f"origin {anchor:%Y-%m-%d} has no complete feature row; "
            f"the latest usable origin is {usable.index.max():%Y-%m-%d}"
        )
    if anchor not in close.index:
        raise ForecastError(f"origin {anchor:%Y-%m-%d} has no closed daily candle")

    origin_close = float(close.loc[anchor])
    blend = policy or BlendPolicy.from_config(config)
    levels = config.forecast.quantiles
    row = build_inference_matrix(usable, pd.DatetimeIndex([anchor]))
    if row.empty:
        raise ForecastError(f"origin {anchor:%Y-%m-%d} has an incomplete feature row")

    point_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    total_crossings = 0

    for horizon in sorted(horizons):
        weight = blend.weight(horizon)
        baseline = _baseline_quantiles(close, horizon, levels, config).loc[[anchor]]
        if baseline.isna().to_numpy().any():
            raise ForecastError(
                f"h={horizon}d: the baseline has no forecast at {anchor:%Y-%m-%d}; "
                "not enough history to estimate its spread"
            )

        if weight > 0.0:
            if horizon not in forecaster.models:
                raise ForecastError(
                    f"h={horizon}d carries model weight {weight:.2f} but "
                    f"{forecaster.metadata.model_version} was not trained on it. "
                    "Train the full horizon grid, or set the blend so this horizon "
                    "is baseline-only."
                )
            predicted, crossings = forecaster.predict_horizon(row, horizon)
            total_crossings += crossings
            combined = blend_quantiles(predicted, baseline, weight)
        else:
            combined = baseline.copy()

        # A convex combination of ordered quantiles stays ordered, but the model
        # half may have been repaired; re-check rather than assume.
        if count_crossings(combined, levels):
            combined = enforce_ordering(combined, levels)

        prices = to_prices(combined, origin_close)
        target_date = anchor + pd.Timedelta(days=horizon)
        median = float(combined.loc[anchor, 0.50])
        point_rows.append(
            {
                "horizon_days": horizon,
                "target_date": target_date.strftime("%Y-%m-%d"),
                "predicted_log_return": median,
                "predicted_price": float(prices.loc[anchor, 0.50]),
                "direction_predicted": int(np.sign(median)),
                "model_weight": round(weight, 4),
                "source": blend_source_label(weight),
            }
        )
        for level in levels:
            quantile_rows.append(
                {
                    "horizon_days": horizon,
                    "quantile": float(level),
                    "quantile_label": quantile_label(level),
                    "predicted_log_return": float(combined.loc[anchor, level]),
                    "predicted_price": float(prices.loc[anchor, level]),
                    "crossing_adjusted": 0,
                }
            )

    forecast = Forecast(
        forecast_id=run_id or uuid.uuid4().hex,
        origin_date=anchor,
        origin_close=origin_close,
        model_version=forecaster.metadata.model_version,
        feature_version=forecaster.metadata.feature_version,
        horizon_grid_version=config.forecast.horizon_grid_version,
        config_version=config.config_version,
        code_commit=code_commit(),
        created_at=utc_now_iso(),
        points=pd.DataFrame(point_rows),
        quantiles=pd.DataFrame(quantile_rows),
        current_price=current_price,
        blend=blend.describe(),
        crossings=total_crossings,
    )
    logger.info("generated forecast: %s", forecast.describe())
    if total_crossings:
        logger.warning(
            "repaired %d quantile crossings while generating this forecast",
            total_crossings,
        )
    return forecast


def assert_intervals_contain_median(forecast: Forecast) -> None:
    """Every interval must contain the median at every horizon.

    Cheap, and it catches the one failure that would make a chart actively
    misleading: a band drawn away from the line it is supposed to surround.
    """
    wide = forecast.quantile_matrix()
    crossings = count_crossing_rows(wide, tuple(wide.columns))
    if crossings:
        raise ForecastError(
            f"{crossings} horizons have quantiles out of order after generation"
        )
    if 0.50 not in wide.columns:
        raise ForecastError("the forecast has no median quantile")
    median = wide[0.50]
    for level in wide.columns:
        if level < 0.50 and (wide[level] > median + 1e-12).any():
            raise ForecastError(f"quantile {level} sits above the median")
        if level > 0.50 and (wide[level] < median - 1e-12).any():
            raise ForecastError(f"quantile {level} sits below the median")


def blend_summary(forecast: Forecast) -> pd.DataFrame:
    """How many horizons came from the model, the baseline, or a blend."""
    counts = forecast.points["source"].value_counts()
    return pd.DataFrame(
        {
            "source": counts.index,
            "horizons": counts.to_numpy(),
            "share": (counts / len(forecast.points)).round(3).to_numpy(),
        }
    )
