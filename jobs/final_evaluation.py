"""Evaluate a frozen design once on the outer test.

Usage::

    python -m jobs.final_evaluation --dry-run            # see the numbers, record nothing
    python -m jobs.final_evaluation --confirm            # spend the outer test
    python -m jobs.final_evaluation --model-version <v> --confirm

VALIDATION_SPEC.md section 4, step 4. This is the only place the reserved block
is read, and for a given `model_version` it can happen once: `record_metrics`
refuses a second write.

`--confirm` is required, and `--dry-run` exists for a reason
------------------------------------------------------------
Recording the evaluation is irreversible, and so, in a subtler way, is *looking*
at it. CLAUDE.md section 2.2 forbids inspecting the test repeatedly and forbids
changing the design because the result was disappointing. A `--dry-run` that
prints the numbers without recording them does not restore that innocence -- it
is still an inspection -- so it exists only to check that the machinery runs, and
the report it writes says so.

Running this releases the training cutoff
-----------------------------------------
Until a design has been evaluated here, the production model must stop at the
purge boundary (`origin + horizon + embargo < outer_test_start`), because a model
trained past it destroys the block before it has been used. Afterwards
`jobs.weekly_model_review` may train through the newest resolved label instead.
That is the one-way trade described in
`src/validation/splits.py::post_test_training_origins`.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import pandas as pd

from src.data.binance import SOURCE as BINANCE_SOURCE
from src.evaluation.final_test import evaluate_final_test, summarise, test_record
from src.features.pipeline import build_features
from src.features.regime import compute_regime_labels
from src.models import registry
from src.models.base import STATUS_CANDIDATE
from src.monitoring.final_report import write_report
from src.storage import repositories as repo
from src.storage.db import open_connection, transaction
from src.utils.config import AppConfig, load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.provenance import code_commit, describe_environment
from src.validation.splits import SplitBoundaries

logger = get_logger("jobs.final_evaluation")

REPORT_NAME: str = "final_evaluation.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen design once on the outer test."
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="the model to evaluate; defaults to the newest testable candidate",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="record the evaluation. Irreversible: one per model version, ever.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and report without recording (still an inspection)",
    )
    parser.add_argument("--horizons", default="required", help="'required', 'all', or a list")
    parser.add_argument("--output", default=None, help="report path")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args(argv)


def resolve_horizons(spec: str, config: AppConfig) -> tuple[int, ...] | None:
    if spec == "all":
        return None
    if spec == "required":
        return config.forecast.required_evaluation_horizons
    return tuple(sorted({int(part) for part in spec.split(",") if part.strip()}))


def _pick_model(connection, boundaries: SplitBoundaries) -> str | None:
    """The newest candidate whose training cutoff is still behind the test block."""
    models = registry.list_models(connection, status=STATUS_CANDIDATE, limit=50)
    if models.empty:
        return None
    testable = models[
        pd.to_datetime(models["training_cutoff"]) < boundaries.outer_test_start
    ]
    if testable.empty:
        return None
    return str(testable.iloc[0]["model_version"])


def run(config: AppConfig, args: argparse.Namespace) -> int:
    if not args.confirm and not args.dry_run:
        logger.error(
            "refusing to read the outer test without --confirm. This evaluation can "
            "happen once per model version (VALIDATION_SPEC.md section 4.4); use "
            "--dry-run to check the machinery without recording."
        )
        return 2

    boundaries = SplitBoundaries.from_config(config)
    horizons = resolve_horizons(args.horizons, config)
    market = config.market.primary
    logger.info("environment: %s", describe_environment())

    with open_connection(config) as connection:
        frame = repo.load_ohlcv(
            connection, BINANCE_SOURCE, market.symbol, market.timeframe
        )
        if frame.empty:
            logger.error(
                "no OHLCV rows stored; run `python -m jobs.update_market_data` first"
            )
            return 1

        version = args.model_version or _pick_model(connection, boundaries)
        if version is None:
            logger.error(
                "no testable candidate is registered. Train one with "
                "`python -m jobs.train_model` -- its cutoff must be before %s.",
                boundaries.outer_test_start.strftime("%Y-%m-%d"),
            )
            return 1

        row = registry.get(connection, version)
        if row is None:
            logger.error("unknown model_version %r", version)
            return 1
        if row["test_metrics"] and row["test_metrics"] != "{}":
            logger.error(
                "%s already has outer-test metrics recorded. A model version is "
                "evaluated once (VALIDATION_SPEC.md section 4.4); a changed design "
                "needs a new model_version.",
                version,
            )
            return 1

        forecaster = registry.load_artifact(connection, version)
        features = build_features(frame, config.features).usable_features()
        regimes = compute_regime_labels(frame, config.features.regime)

        run_id = f"final-{uuid.uuid4().hex[:12]}"
        if args.confirm and not args.dry_run:
            with transaction(connection):
                registry.start_run(
                    connection,
                    run_id,
                    "final_evaluation",
                    config_version=config.config_version,
                    code_commit=code_commit(),
                    random_seed=int(row["random_seed"] or 0),
                    dataset_cutoff=frame.index.max().strftime("%Y-%m-%d"),
                )

        logger.warning(
            "reading the outer test for %s (block from %s). This is the one "
            "evaluation this design gets.",
            version,
            boundaries.outer_test_start.strftime("%Y-%m-%d"),
        )
        result = evaluate_final_test(
            forecaster,
            features,
            frame["close"],
            boundaries,
            config,
            horizons=horizons,
            regimes=regimes,
        )
        logger.info("done: %s", result.describe())
        for row_ in summarise(result, config).itertuples(index=False):
            logger.info(
                "h=%3dd  model=%.4f  baseline=%.4f  improvement=%+.1f%%  "
                "coverage_95=%.2f  n=%d",
                row_.horizon_days,
                row_.model,
                row_.baseline,
                100.0 * row_.improvement,
                row_.coverage_95,
                row_.sample_size,
            )

        if args.confirm and not args.dry_run:
            record = test_record(result, config)
            with transaction(connection):
                registry.record_metrics(connection, version, test_metrics=record)
                repo.upsert_performance_metrics(
                    connection, result.metrics, scope="outer_test", run_id=run_id
                )
                registry.finish_run(
                    connection,
                    run_id,
                    status="succeeded",
                    model_id=row["model_id"],
                    metrics={"horizons": len(result.horizons)},
                    notes=f"outer test {result.test_start}..{result.test_end}",
                )
            logger.info("recorded the single outer-test evaluation for %s", version)
            logger.info(
                "the production training cutoff is now released: "
                "`python -m jobs.weekly_model_review --force` can train through "
                "the newest resolved label"
            )
        else:
            logger.warning("--dry-run: nothing recorded, but the test has been read")

    path = (
        config.paths.reports_dir / REPORT_NAME
        if args.output is None
        else Path(args.output)
    )
    write_report(result, config, path)
    logger.info("report written: %s", path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    setup_logging(config)
    return run(config, args)


if __name__ == "__main__":
    sys.exit(main())
