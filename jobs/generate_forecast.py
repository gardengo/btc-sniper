"""Produce and store the daily forecast.

Usage::

    python -m jobs.generate_forecast
    python -m jobs.generate_forecast --model-version lgbmq-expanding-20231230-v1
    python -m jobs.generate_forecast --origin 2025-06-30      # backfill one date
    python -m jobs.generate_forecast --no-persist

Runs once per day, after the daily candle has closed. The forecast is anchored on
the **last closed daily candle**, not on the live price: CLAUDE.md section 2.4
separates the two, so intraday movement never regenerates a one-year forecast.
The live price is fetched and stored alongside it for display only.

By default the forecast comes from the **production** model. `--model-version`
targets a specific registered model instead, which is a research path and is
logged as such -- it does not make that model production.
"""

from __future__ import annotations

import argparse
import sys
import uuid

import pandas as pd

from src.data.binance import SOURCE as BINANCE_SOURCE
from src.data.realtime import RealtimePriceError, get_current_price
from src.features.pipeline import build_features
from src.forecast.blending import BlendPolicy
from src.forecast.generate import (
    assert_intervals_contain_median,
    blend_summary,
    generate_forecast,
)
from src.forecast.horizons import horizon_grid_from_config
from src.models import registry
from src.models.base import STATUS_PRODUCTION
from src.storage import repositories as repo
from src.storage.db import open_connection, transaction
from src.utils.config import AppConfig, load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.provenance import code_commit, describe_environment

logger = get_logger("jobs.generate_forecast")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the daily BTC forecast.")
    parser.add_argument(
        "--model-version",
        default=None,
        help="use this registered model instead of the production one (research)",
    )
    parser.add_argument("--origin", default=None, help="forecast origin date (YYYY-MM-DD)")
    parser.add_argument(
        "--no-current-price",
        action="store_true",
        help="skip the live price lookup (offline runs)",
    )
    parser.add_argument(
        "--no-persist", action="store_true", help="generate without storing"
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


def _resolve_model(connection, args: argparse.Namespace):
    if args.model_version:
        forecaster = registry.load_artifact(connection, args.model_version)
        logger.warning(
            "using %s explicitly; this is a research run and does not make it "
            "the production model",
            args.model_version,
        )
        return forecaster
    forecaster = registry.load_production(connection)
    if forecaster is None:
        logger.error(
            "no production model is registered. Train a candidate with "
            "`python -m jobs.train_model`, then promote it through the weekly "
            "review; or pass --model-version for a research run."
        )
    return forecaster


def _current_price(connection, config: AppConfig, skip: bool) -> float | None:
    if skip:
        return None
    try:
        snapshot = get_current_price(connection, config)
    except RealtimePriceError as exc:
        logger.warning("no current price available (%s); forecast is unaffected", exc)
        return None
    logger.info("current price: %s", snapshot.describe())
    return snapshot.price


def run(config: AppConfig, args: argparse.Namespace) -> int:
    market = config.market.primary
    horizons = horizon_grid_from_config(config.forecast)
    policy = BlendPolicy.from_config(config)
    logger.info("environment: %s", describe_environment())
    logger.info("blend policy: %s", policy.describe())

    with open_connection(config) as connection:
        frame = repo.load_ohlcv(connection, BINANCE_SOURCE, market.symbol, market.timeframe)
        if frame.empty:
            logger.error("no OHLCV rows stored; run `python -m jobs.update_market_data` first")
            return 1

        forecaster = _resolve_model(connection, args)
        if forecaster is None:
            return 1
        logger.info("model: %s", forecaster.describe())

        features = build_features(frame, config.features).usable_features()
        current_price = _current_price(connection, config, args.no_current_price)
        run_id = uuid.uuid4().hex

        forecast = generate_forecast(
            features,
            frame["close"],
            config,
            forecaster,
            horizons,
            origin=pd.Timestamp(args.origin) if args.origin else None,
            current_price=current_price,
            policy=policy,
        )
        assert_intervals_contain_median(forecast)
        logger.info(
            "composition: %s",
            ", ".join(
                f"{row.source}={row.horizons}"
                for row in blend_summary(forecast).itertuples(index=False)
            ),
        )
        _log_headline(forecast)

        if args.no_persist:
            logger.info("persistence skipped (--no-persist)")
            return 0

        model_row = registry.get(connection, forecast.model_version)
        with transaction(connection):
            registry.start_run(
                connection,
                run_id,
                "forecast",
                config_version=config.config_version,
                code_commit=code_commit(),
                dataset_cutoff=forecast.origin_date.strftime("%Y-%m-%d"),
            )
            written = repo.upsert_forecast(
                connection,
                forecast,
                source=BINANCE_SOURCE,
                symbol=market.symbol,
                timeframe=market.timeframe,
                run_id=run_id,
                model_id=model_row["model_id"] if model_row else None,
            )
            registry.finish_run(
                connection,
                run_id,
                status="succeeded",
                model_id=model_row["model_id"] if model_row else None,
                metrics={"horizons": written},
            )
        if model_row is not None and model_row["status"] != STATUS_PRODUCTION:
            logger.warning(
                "stored a forecast from a %s model; the dashboard should label it "
                "as non-production",
                model_row["status"],
            )
    return 0


def _log_headline(forecast) -> None:
    """The few numbers a person actually reads off a daily run."""
    points = forecast.points.set_index("horizon_days")
    prices = forecast.quantiles.pivot(
        index="horizon_days", columns="quantile", values="predicted_price"
    )
    logger.info(
        "anchor %s close %.2f",
        forecast.origin_date.strftime("%Y-%m-%d"),
        forecast.origin_close,
    )
    for horizon in (30, 90, 180, 365):
        if horizon not in points.index:
            continue
        logger.info(
            "h=%3dd %s  median %10.0f  95%% [%9.0f, %10.0f]  (%s)",
            horizon,
            points.loc[horizon, "target_date"],
            points.loc[horizon, "predicted_price"],
            prices.loc[horizon, 0.025],
            prices.loc[horizon, 0.975],
            points.loc[horizon, "source"],
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    setup_logging(config)
    return run(config, args)


if __name__ == "__main__":
    sys.exit(main())
