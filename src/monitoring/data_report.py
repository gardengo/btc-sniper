"""Baseline data-quality report.

Renders everything known about the stored market data and the derived feature
matrix into a Markdown document: coverage, validation outcomes, per-year
statistics, regime composition and feature warmup/NaN behaviour.

The report is descriptive only. It makes no modelling claim, and in particular
it does not assert any cyclical structure in the price history
(CLAUDE.md section 7).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data.validation import QualityReport
from src.features.pipeline import FeatureBuildResult
from src.features.regime import compute_regime_labels, regime_summary
from src.validation.splits import SplitBoundaries, describe_split
from src.utils.timeutils import utc_now_iso


@dataclass(frozen=True)
class ReportContext:
    """Everything the renderer needs, gathered by the calling job."""

    source: str
    symbol: str
    timeframe: str
    ohlcv: pd.DataFrame
    quality: QualityReport
    features: FeatureBuildResult | None
    regime_params: dict[str, Any]
    secondary_label: str | None = None
    secondary_ohlcv: pd.DataFrame | None = None
    split: SplitBoundaries | None = None
    horizons: tuple[int, ...] = ()


def _markdown_table(frame: pd.DataFrame, float_format: str = "{:,.4f}") -> str:
    if frame.empty:
        return "_(no rows)_\n"
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda v: "" if pd.isna(v) else float_format.format(v)
            )
        else:
            formatted[column] = formatted[column].astype(str)
    header = "| " + " | ".join(str(c) for c in formatted.columns) + " |"
    divider = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in formatted.itertuples(index=False)
    ]
    return "\n".join([header, divider, *body]) + "\n"


def yearly_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-calendar-year price/volume/return statistics."""
    close = frame["close"]
    daily_log_return = np.log(close / close.shift(1))
    grouped = frame.assign(_year=frame.index.year, _ret=daily_log_return).groupby("_year")

    rows: list[dict[str, Any]] = []
    for year, block in grouped:
        returns = block["_ret"].dropna()
        first_close = float(block["close"].iloc[0])
        last_close = float(block["close"].iloc[-1])
        rows.append(
            {
                "year": int(year),
                "days": int(len(block)),
                "first_close": first_close,
                "last_close": last_close,
                "min_close": float(block["close"].min()),
                "max_close": float(block["close"].max()),
                "year_return_pct": (last_close / first_close - 1.0) * 100.0,
                "ann_vol_pct": float(returns.std(ddof=1) * np.sqrt(365) * 100.0)
                if len(returns) > 1
                else np.nan,
                "max_abs_daily_move_pct": float(returns.abs().max() * 100.0)
                if len(returns)
                else np.nan,
                "median_volume": float(block["volume"].median()),
            }
        )
    return pd.DataFrame(rows)


def feature_warmup_table(result: FeatureBuildResult) -> pd.DataFrame:
    """First date each feature becomes available, and its NaN count."""
    rows: list[dict[str, Any]] = []
    group_of = {
        column: group
        for group, columns in result.column_groups.items()
        for column in columns
    }
    for column in result.features.columns:
        series = result.features[column]
        valid = series.dropna()
        rows.append(
            {
                "feature": column,
                "group": group_of.get(column, "?"),
                "first_valid": valid.index.min().strftime("%Y-%m-%d")
                if not valid.empty
                else "",
                "nan_rows": int(series.isna().sum()),
                "nan_after_warmup": int(
                    series.loc[result.usable_features().index.min() :].isna().sum()
                )
                if result.rows_usable
                else int(series.isna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "feature"]).reset_index(drop=True)


def feature_statistics(result: FeatureBuildResult) -> pd.DataFrame:
    """Descriptive statistics over the usable (post-warmup) feature rows."""
    usable = result.usable_features()
    if usable.empty:
        return pd.DataFrame()
    described = usable.describe().T[["mean", "std", "min", "50%", "max"]]
    described.insert(0, "feature", described.index)
    return described.reset_index(drop=True)


def _quality_rows_table(quality: QualityReport) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": row.check_name,
                "status": row.status.upper(),
                "severity": row.severity,
                "message": row.message,
            }
            for row in quality.rows
        ]
    )


def render_markdown(context: ReportContext) -> str:
    """Render the full report."""
    frame = context.ohlcv
    quality = context.quality
    lines: list[str] = []

    lines.append("# BTC Sniper - Data Quality Report")
    lines.append("")
    lines.append(f"- Generated at: `{utc_now_iso()}`")
    lines.append(f"- Source: `{context.source}` / `{context.symbol}` / `{context.timeframe}`")
    lines.append(f"- Quality run id: `{quality.run_id}`")
    lines.append("- All timestamps are UTC.")
    lines.append("")

    lines.append("## 1. Coverage")
    lines.append("")
    if frame.empty:
        lines.append("_No market data stored._")
        return "\n".join(lines) + "\n"

    expected_days = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    missing_days = expected_days.difference(frame.index)
    coverage = pd.DataFrame(
        [
            {"metric": "rows stored", "value": f"{len(frame):,}"},
            {"metric": "first candle", "value": frame.index.min().strftime("%Y-%m-%d")},
            {"metric": "last candle", "value": frame.index.max().strftime("%Y-%m-%d")},
            {"metric": "calendar days spanned", "value": f"{len(expected_days):,}"},
            {"metric": "missing daily candles", "value": f"{len(missing_days):,}"},
            {
                "metric": "coverage",
                "value": f"{(1 - len(missing_days) / len(expected_days)) * 100:.4f}%",
            },
            {"metric": "first close", "value": f"{float(frame['close'].iloc[0]):,.2f}"},
            {"metric": "last close", "value": f"{float(frame['close'].iloc[-1]):,.2f}"},
            {"metric": "min close", "value": f"{float(frame['close'].min()):,.2f}"},
            {"metric": "max close", "value": f"{float(frame['close'].max()):,.2f}"},
        ]
    )
    lines.append(_markdown_table(coverage))
    if len(missing_days):
        listed = ", ".join(d.strftime("%Y-%m-%d") for d in missing_days[:30])
        lines.append(f"Missing dates (first 30): {listed}")
        lines.append("")

    lines.append("## 2. Validation checks")
    lines.append("")
    lines.append(f"Result: **{quality.summary()}**")
    lines.append("")
    lines.append(_markdown_table(_quality_rows_table(quality)))
    if quality.errors:
        lines.append("> Blocking errors present: the daily pipeline must not")
        lines.append("> generate a forecast in this state (OPERATING_SPEC.md section 9).")
        lines.append("")

    lines.append("## 3. Yearly statistics")
    lines.append("")
    lines.append(
        "Descriptive only. `year_return_pct` compares the first and last stored "
        "close of each calendar year and is not a model result."
    )
    lines.append("")
    lines.append(_markdown_table(yearly_statistics(frame), float_format="{:,.2f}"))

    lines.append("## 4. Market regime composition")
    lines.append("")
    lines.append(
        "Labels are computed causally (each day uses only data up to that day), "
        "so they are safe both as model features and as an evaluation breakdown."
    )
    lines.append("")
    labels = compute_regime_labels(frame, context.regime_params)
    lines.append(_markdown_table(regime_summary(labels), float_format="{:.4f}"))

    by_year = (
        labels.assign(year=labels.index.year)
        .pivot_table(index="year", columns="direction", aggfunc="size", fill_value=0)
        .reset_index()
    )
    lines.append("Direction regime days per year:")
    lines.append("")
    lines.append(_markdown_table(by_year, float_format="{:,.0f}"))

    if context.secondary_ohlcv is not None and not context.secondary_ohlcv.empty:
        lines.append("## 5. Cross-exchange sanity check")
        lines.append("")
        joined = frame[["close"]].join(
            context.secondary_ohlcv[["close"]], how="inner", rsuffix="_secondary"
        )
        if joined.empty:
            lines.append("_No overlapping dates._")
        else:
            diff_pct = (
                (joined["close"] - joined["close_secondary"]).abs()
                / joined["close_secondary"]
                * 100
            )
            summary = pd.DataFrame(
                [
                    {"metric": "overlapping days", "value": f"{len(joined):,}"},
                    {"metric": "median abs diff", "value": f"{diff_pct.median():.4f}%"},
                    {"metric": "p95 abs diff", "value": f"{diff_pct.quantile(0.95):.4f}%"},
                    {"metric": "max abs diff", "value": f"{diff_pct.max():.4f}%"},
                    {
                        "metric": "max abs diff date",
                        "value": diff_pct.idxmax().strftime("%Y-%m-%d"),
                    },
                ]
            )
            lines.append(
                f"Comparing `{context.symbol}` against `{context.secondary_label}`. "
                "The secondary series is a sanity check only and is never merged "
                "into the training series (DATA_SPEC.md section 1)."
            )
            lines.append("")
            lines.append(_markdown_table(summary))
        lines.append("")

    if context.split is not None and context.features is not None and context.horizons:
        boundaries = context.split
        lines.append("## 6. Validation split (frozen)")
        lines.append("")
        lines.append(
            f"`split_version={boundaries.split_version}`, frozen before any model was "
            "trained. See VALIDATION_SPEC.md section 4.2. Moving `outer_test_start` "
            "invalidates every recorded outer-test metric."
        )
        lines.append("")
        boundary_table = pd.DataFrame(
            [
                {"block": "inner (train + validation)", "range": f"..{boundaries.inner_validation_end:%Y-%m-%d}"},
                {"block": "outer test", "range": f"{boundaries.outer_test_start:%Y-%m-%d}.."
                 + (f"{boundaries.outer_test_end:%Y-%m-%d}" if boundaries.outer_test_end else "latest data")},
                {"block": "embargo (days, on top of the horizon purge)", "range": str(boundaries.embargo_days)},
            ]
        )
        lines.append(_markdown_table(boundary_table))
        lines.append(
            "Per-horizon sizes. `independent windows` divides the origin count by the "
            "horizon, because consecutive daily origins share almost all of their "
            "target window; it is the number that decides whether a horizon can "
            "support a verdict."
        )
        lines.append("")
        key_horizons = tuple(
            h for h in context.horizons if h in (1, 7, 30, 90, 180, 365)
        )
        # Warmup rows hold NaN features and can never be trained on, so the
        # split is described over the usable index only.
        split_table = describe_split(
            context.features.usable_features().index,
            boundaries,
            key_horizons,
            data_end=frame.index.max(),
        )
        lines.append(_markdown_table(split_table, float_format="{:,.1f}"))

    if context.features is not None:
        result = context.features
        lines.append("## 7. Feature matrix")
        lines.append("")
        feature_summary = pd.DataFrame(
            [
                {"metric": "feature_version", "value": result.feature_version},
                {"metric": "groups", "value": ", ".join(result.groups)},
                {"metric": "feature columns", "value": f"{result.features.shape[1]:,}"},
                {"metric": "rows (all)", "value": f"{result.rows_total:,}"},
                {"metric": "rows (no NaN)", "value": f"{result.rows_usable:,}"},
                {"metric": "first fully usable date", "value": result.first_usable_date or ""},
                {"metric": "built at", "value": result.built_at},
            ]
        )
        lines.append(_markdown_table(feature_summary))

        lines.append("### 7.1 Columns per group")
        lines.append("")
        group_counts = pd.DataFrame(
            [
                {"group": group, "columns": len(columns), "names": ", ".join(columns)}
                for group, columns in result.column_groups.items()
            ]
        )
        lines.append(_markdown_table(group_counts))

        lines.append("### 7.2 Warmup and missing values")
        lines.append("")
        lines.append(
            "`nan_after_warmup` must be 0 for every feature; a non-zero value means "
            "the feature is intermittently uncomputable and needs attention before "
            "it is used for training."
        )
        lines.append("")
        lines.append(_markdown_table(feature_warmup_table(result), float_format="{:,.0f}"))

        lines.append("### 7.3 Feature statistics (post-warmup)")
        lines.append("")
        lines.append(_markdown_table(feature_statistics(result), float_format="{:,.4f}"))

    return "\n".join(lines) + "\n"


def write_report(context: ReportContext, path: Path) -> Path:
    """Render and write the report, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(context), encoding="utf-8")
    return path
