"""Prediction Log: every forecast the system made, and what happened to it.

The page exists so a prediction cannot quietly disappear. A forecast that has
not resolved is shown as pending rather than omitted, because a log that only
lists the resolved rows is a log of the results you already know.
"""

from __future__ import annotations

import sqlite3

import streamlit as st

from app import data_access as data
from src.utils.config import AppConfig

STATUS_LABELS: dict[str, str] = {
    "pending": "pending - the target date has not arrived",
    "partially_evaluable": "partially evaluable",
    "fully_evaluated": "fully evaluated",
}
INTERVAL_COLUMNS: tuple[str, ...] = (
    "in_interval_50",
    "in_interval_80",
    "in_interval_95",
)


def render(connection: sqlite3.Connection, config: AppConfig) -> None:
    st.subheader("Prediction Log")

    options = data.log_filter_options(connection)
    if not options["origins"]:
        st.warning(
            "No forecast has been logged yet. Run "
            "`python -m jobs.generate_forecast`, then "
            "`python -m jobs.evaluate_forecasts` to create the realization rows."
        )
        return

    status, horizon, origin_from, origin_to = _filters(options)
    frame = data.prediction_log(
        connection,
        config,
        status=status,
        horizon_days=horizon,
        origin_from=origin_from,
        origin_to=origin_to,
    )
    if frame.empty:
        st.info("No rows match these filters.")
        return

    resolved = int((frame["evaluation_status"] == data.STATUS_FULL).sum())
    st.caption(
        f"{len(frame):,} rows - {resolved:,} resolved, {len(frame) - resolved:,} "
        "still waiting for their target date."
    )
    if resolved == 0:
        st.info(
            "Nothing has resolved yet, so the error columns are empty. A 30-day "
            "horizon needs 30 days before it says anything and a 365-day horizon "
            "needs a year (OPERATING_SPEC.md section 3.2)."
        )

    st.dataframe(
        _displayable(frame),
        width="stretch",
        hide_index=True,
        column_config={
            "forecast_origin_date": st.column_config.TextColumn("Forecast date"),
            "target_date": st.column_config.TextColumn("Target date"),
            "horizon_days": st.column_config.NumberColumn("Horizon", format="%d"),
            "evaluation_status": st.column_config.TextColumn("Status"),
            "predicted_median": st.column_config.NumberColumn(
                "Predicted", format="$%,.0f"
            ),
            "actual_close": st.column_config.NumberColumn("Actual", format="$%,.0f"),
            "absolute_error": st.column_config.NumberColumn("Error", format="$%,.0f"),
            "percentage_error": st.column_config.NumberColumn("Error %", format="%+.2f%%"),
            "in_interval": st.column_config.TextColumn("In interval"),
            "direction_correct": st.column_config.TextColumn("Direction"),
            "regime": st.column_config.TextColumn("Regime at origin"),
            "model_version": st.column_config.TextColumn("Model"),
        },
    )
    st.caption(
        "`In interval` reads 50/80/95 and marks which prediction bands actually "
        "contained the outcome. Regime is the label **at the forecast origin**, "
        "which is the only one knowable at forecast time "
        "(VALIDATION_SPEC.md section 6)."
    )


def _filters(options: dict) -> tuple[str | None, int | None, str | None, str | None]:
    left, middle, right = st.columns([2, 1, 2])
    with left:
        chosen = st.selectbox(
            "Evaluation status",
            ["all", *options["statuses"]],
            format_func=lambda value: "all"
            if value == "all"
            else STATUS_LABELS.get(value, value),
            key="log_status",
        )
    with middle:
        horizon = st.selectbox(
            "Horizon (days)", ["all", *options["horizons"]], key="log_horizon"
        )
    with right:
        origins = options["origins"]
        start, end = st.select_slider(
            "Forecast date",
            options=origins,
            value=(origins[0], origins[-1]),
            key="log_origins",
        )
    return (
        None if chosen == "all" else str(chosen),
        None if horizon == "all" else int(horizon),
        str(start),
        str(end),
    )


def _displayable(frame):
    """Collapse the three interval flags into one readable column."""
    display = frame.copy()
    present = [name for name in INTERVAL_COLUMNS if name in display.columns]
    if present:
        display["in_interval"] = [
            _interval_label(row) for row in display[present].itertuples(index=False)
        ]
        display = display.drop(columns=present)
    if "direction_correct" in display.columns:
        display["direction_correct"] = display["direction_correct"].map(
            {1: "correct", 0: "wrong", 1.0: "correct", 0.0: "wrong"}
        ).fillna("-")
    ordered = [
        "forecast_origin_date",
        "target_date",
        "horizon_days",
        "evaluation_status",
        "predicted_median",
        "actual_close",
        "absolute_error",
        "percentage_error",
        "in_interval",
        "direction_correct",
        "regime",
        "model_version",
    ]
    return display[[name for name in ordered if name in display.columns]]


def _interval_label(row) -> str:
    hits = [
        label
        for label, value in zip(("50", "80", "95"), row)
        if value == 1 or value is True
    ]
    if hits:
        return "/".join(hits)
    return "-" if all(value is None or value != value for value in row) else "none"


__all__ = ["render"]
