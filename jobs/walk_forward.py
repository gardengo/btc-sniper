"""Run walk-forward validation and write the validation report.

Usage::

    python -m jobs.walk_forward
    python -m jobs.walk_forward --window rolling_4y
    python -m jobs.walk_forward --compare-windows       # all configured windows
    python -m jobs.walk_forward --compare-params        # pre-declared param sets
    python -m jobs.walk_forward --horizons 1,7,30

Inner block only. Nothing here reads the outer test, and the models trained here
are fold models, not deployable ones -- they are recorded under a `wf-` identity
precisely so nobody can promote one.

`--compare-windows` implements VALIDATION_SPEC.md section 9: the training window
is selected from inner validation, never assumed from the four-year cycle.

`--compare-params` runs the candidate sets declared in `models.lightgbm_candidates`.
They are declared in config up front on purpose -- repeatedly adjusting one number
and re-checking the same folds is the validation overfitting VALIDATION_SPEC.md
section 5 prohibits. Run it, record the choice, freeze it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.binance import SOURCE as BINANCE_SOURCE
from src.evaluation.walk_forward import WalkForwardResult, run_walk_forward
from src.features.pipeline import build_features
from src.features.regime import compute_regime_labels
from src.forecast.horizons import horizon_grid_from_config
from src.models.training import TrainingRequest
from src.monitoring.validation_report import write_comparison, write_report
from src.storage import repositories as repo
from src.storage.db import open_connection, transaction
from src.utils.config import AppConfig, load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.provenance import describe_environment
from src.validation.folds import FoldPlan
from src.validation.splits import SplitBoundaries

logger = get_logger("jobs.walk_forward")

REPORT_NAME: str = "walk_forward_validation.md"
COMPARISON_NAME: str = "training_window_comparison.md"
PARAMS_COMPARISON_NAME: str = "hyperparameter_comparison.md"
# The candidate name whose settings are frozen into `models.lightgbm`.
SELECTED_PARAMS: str = "strong"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward validation run.")
    parser.add_argument(
        "--horizons",
        default="required",
        help="'required' (default), 'all', or a comma-separated list of days",
    )
    parser.add_argument("--window", default=None, help="training window strategy")
    parser.add_argument(
        "--compare-windows",
        action="store_true",
        help="run every window in models.training_windows and compare them",
    )
    parser.add_argument(
        "--compare-params",
        action="store_true",
        help="run every pre-declared set in models.lightgbm_candidates and compare them",
    )
    parser.add_argument("--output", default=None, help="report path")
    parser.add_argument(
        "--no-persist", action="store_true", help="skip writing performance_metrics"
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


def _persist(config: AppConfig, result: WalkForwardResult) -> int:
    """Store per-fold metrics under the validation scope."""
    with open_connection(config) as connection:
        with transaction(connection):
            return repo.upsert_performance_metrics(
                connection,
                result.metrics,
                scope="validation",
                run_id=f"wf-{result.strategy}",
            )


def run(config: AppConfig, args: argparse.Namespace) -> int:
    boundaries = SplitBoundaries.from_config(config)
    horizons = resolve_horizons(args.horizons, config)
    focus = config.forecast.required_evaluation_horizons
    market = config.market.primary
    logger.info("environment: %s", describe_environment())

    with open_connection(config) as connection:
        frame = repo.load_ohlcv(connection, BINANCE_SOURCE, market.symbol, market.timeframe)
    if frame.empty:
        logger.error("no OHLCV rows stored; run `python -m jobs.update_market_data` first")
        return 1

    features = build_features(frame, config.features).usable_features()
    regimes = compute_regime_labels(frame, config.features.regime)
    close = frame["close"]

    strategies = (
        tuple(str(name) for name in config.section("models").get("training_windows", []))
        if args.compare_windows
        else (args.window or str(config.section("models").get("default_training_window")),)
    )
    logger.info(
        "walk-forward over %d horizons, windows: %s", len(horizons), list(strategies)
    )

    candidates: dict[str, dict] = (
        {
            str(name): dict(params)
            for name, params in config.section("models")
            .get("lightgbm_candidates", {})
            .items()
        }
        if args.compare_params
        else {"": {}}
    )
    if args.compare_params and not candidates:
        logger.error("--compare-params needs models.lightgbm_candidates in config")
        return 1

    results: dict[str, WalkForwardResult] = {}
    for strategy in strategies:
        plan = FoldPlan.from_config(config, strategy=strategy)
        for params_name, params in candidates.items():
            request = TrainingRequest.from_config(
                config,
                horizons=horizons,
                strategy=strategy,
                params=params or None,
            )
            label = f"{strategy}/{params_name}" if params_name else strategy
            logger.info("=== %s (%s) ===", label, plan)
            result = run_walk_forward(
                features,
                close,
                boundaries,
                config,
                request,
                plan,
                regimes=regimes,
                params_name=params_name,
            )
            logger.info("done: %s", result.describe())
            results[label] = result

            if not args.no_persist:
                logger.info("persisted %d metric rows", _persist(config, result))

    # The headline report must describe the configuration that is actually in
    # use, not whichever candidate happened to be listed first -- the candidate
    # list deliberately starts with a negative control.
    active = config.section("models").get("default_training_window")
    primary_key = next(
        (key for key in results if key == active or key.endswith(f"/{SELECTED_PARAMS}")),
        next(iter(results)),
    )
    primary = results[primary_key]
    logger.info("headline report: %s", primary_key)
    path = config.paths.reports_dir / REPORT_NAME if args.output is None else Path(args.output)
    write_report(primary, config, path, focus_horizons=focus)
    logger.info("report written: %s", path)

    if len(results) > 1:
        if args.compare_params:
            name, title = PARAMS_COMPARISON_NAME, "Hyperparameter Comparison"
            preamble = (
                "Candidate sets are declared in `models.lightgbm_candidates` up "
                "front and run once. Adjusting one number and re-checking the same "
                "folds is the validation overfitting VALIDATION_SPEC.md section 5 "
                "prohibits; the choice made here is recorded and frozen."
            )
        else:
            name, title, preamble = COMPARISON_NAME, "Training Window Comparison", None
        comparison = config.paths.reports_dir / name
        write_comparison(results, config, comparison, title=title, preamble=preamble)
        logger.info("comparison written: %s", comparison)

    _log_verdict(primary, config)
    return 0


def _log_verdict(result: WalkForwardResult, config: AppConfig) -> None:
    from src.monitoring.validation_report import verdict

    summary = verdict(result, config)
    if summary.empty:
        logger.warning("no verdict could be formed")
        return
    for row in summary.itertuples(index=False):
        logger.info(
            "h=%3dd  win_rate=%.2f (%d/%d folds)  mean_improvement=%+.3f  -> %s",
            int(row.horizon_days),
            row.win_rate,
            int(row.folds_won),
            int(row.folds),
            row.mean_improvement,
            row.verdict,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    setup_logging(config)
    return run(config, args)


if __name__ == "__main__":
    sys.exit(main())
