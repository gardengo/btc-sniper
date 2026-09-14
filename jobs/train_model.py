"""Train one model version and register it as a candidate.

Usage::

    python -m jobs.train_model
    python -m jobs.train_model --window rolling_4y
    python -m jobs.train_model --horizons all          # full 77-horizon grid
    python -m jobs.train_model --horizons 1,7,30
    python -m jobs.train_model --check-reproducible

Training never promotes. A model produced here is a `candidate`; becoming the
production model is a separate decision made by the weekly review against the
promotion gate (CLAUDE.md section 2.3, Phase 8).

The training cutoff defaults to the purge limit implied by the frozen split, so
no label can reach into the outer test. Passing `--cutoff` narrows it further;
it can never widen it.
"""

from __future__ import annotations

import argparse
import sys
import uuid

import pandas as pd

from src.data.binance import SOURCE as BINANCE_SOURCE
from src.features.pipeline import build_features
from src.forecast.horizons import horizon_grid_from_config
from src.models import registry
from src.models.training import TrainingRequest, assert_reproducible, train_forecaster
from src.storage import repositories as repo
from src.storage.db import open_connection, transaction
from src.utils.config import AppConfig, load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.provenance import code_commit, describe_environment
from src.validation.splits import SplitBoundaries

logger = get_logger("jobs.train_model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a candidate forecast model.")
    parser.add_argument(
        "--horizons",
        default="required",
        help="'required' (default), 'all', or a comma-separated list of days",
    )
    parser.add_argument("--window", default=None, help="training window strategy")
    parser.add_argument("--cutoff", default=None, help="training cutoff date (YYYY-MM-DD)")
    parser.add_argument("--suffix", default="", help="suffix for the model version")
    parser.add_argument(
        "--check-reproducible",
        action="store_true",
        help="refit one horizon and assert the predictions are identical",
    )
    parser.add_argument(
        "--no-persist", action="store_true", help="train without saving or registering"
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


def resolve_horizons(spec: str, config: AppConfig) -> tuple[int, ...]:
    if spec == "required":
        return config.forecast.required_evaluation_horizons
    grid = horizon_grid_from_config(config.forecast)
    if spec == "all":
        return grid
    wanted = tuple(sorted({int(part) for part in spec.split(",") if part.strip()}))
    unknown = [horizon for horizon in wanted if horizon not in grid]
    if unknown:
        raise SystemExit(f"horizons not in the configured grid: {unknown}")
    return wanted


def run(config: AppConfig, args: argparse.Namespace) -> int:
    boundaries = SplitBoundaries.from_config(config)
    horizons = resolve_horizons(args.horizons, config)
    cutoff = pd.Timestamp(args.cutoff) if args.cutoff else None
    market = config.market.primary
    logger.info("environment: %s", describe_environment())

    with open_connection(config) as connection:
        frame = repo.load_ohlcv(connection, BINANCE_SOURCE, market.symbol, market.timeframe)
        if frame.empty:
            logger.error("no OHLCV rows stored; run `python -m jobs.update_market_data` first")
            return 1

        features = build_features(frame, config.features)
        usable = features.usable_features()
        request = TrainingRequest.from_config(
            config,
            horizons=horizons,
            strategy=args.window,
            cutoff=cutoff,
            suffix=args.suffix,
        )
        logger.info(
            "training %s on %d usable feature rows, window=%s, horizons=%d",
            request.algorithm,
            len(usable),
            request.strategy,
            len(horizons),
        )

        if args.check_reproducible:
            probe = min(horizons)
            assert_reproducible(
                usable, frame["close"], boundaries, config, request, horizon_days=probe
            )
            logger.info("reproducibility check passed at h=%dd", probe)

        run_id = uuid.uuid4().hex
        persist = not args.no_persist
        if persist:
            # Opened before training so a crash leaves a visible `running` row
            # rather than no trace of the attempt at all.
            with transaction(connection):
                registry.start_run(
                    connection,
                    run_id,
                    "training",
                    config_version=config.config_version,
                    code_commit=code_commit(),
                    random_seed=request.seed,
                    dataset_cutoff=str(cutoff.date()) if cutoff else None,
                )

        try:
            forecaster = train_forecaster(
                usable, frame["close"], boundaries, config, request
            )
        except Exception as exc:
            if persist:
                with transaction(connection):
                    registry.finish_run(
                        connection, run_id, status="failed", notes=str(exc)
                    )
            raise
        logger.info("model: %s", forecaster.describe())

        if not persist:
            logger.info("persistence skipped (--no-persist)")
            return 0

        path = forecaster.save(config.paths.models_dir)
        with transaction(connection):
            registry.register(connection, forecaster.metadata, artifact_path=path)
            registry.finish_run(
                connection,
                run_id,
                status="succeeded",
                model_id=forecaster.metadata.model_id,
                metrics={
                    "horizons": len(forecaster.models),
                    "training_rows": forecaster.metadata.training_rows,
                },
            )
        logger.info(
            "registered candidate %s (artifact: %s)",
            forecaster.metadata.model_version,
            path.name,
        )
        logger.info(
            "not promoted: promotion is a separate decision (CLAUDE.md section 2.3)"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    setup_logging(config)
    return run(config, args)


if __name__ == "__main__":
    sys.exit(main())
