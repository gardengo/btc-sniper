"""Model Performance: what is actually known about how well this forecasts.

The page keeps three kinds of evidence apart, because merging them would answer
a question nobody asked:

- **production** - realized forecasts since the model started running. The only
  evidence about the deployed model, and it starts empty.
- **validation** - walk-forward folds on the inner block. Says what the *design*
  was worth; already used to select it, so it flatters.
- **outer test** - the single independent verdict, absent until
  `jobs.final_evaluation` has been run.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from app import charts, data_access as data
from src.utils.config import AppConfig

METRIC_FORMATS: dict[str, str] = {
    "horizon_days": "%d",
    "sample_size": "%d",
    "return_mae": "%.4f",
    "price_mae": "$%,.0f",
    "pinball_mean": "%.4f",
    "direction_accuracy": "%.3f",
    "coverage_50": "%.3f",
    "coverage_80": "%.3f",
    "coverage_95": "%.3f",
}


def render(connection: sqlite3.Connection, config: AppConfig) -> None:
    st.subheader("Model Performance")
    performance = data.load_performance(connection)

    production_tab, validation_tab, models_tab, revision_tab = st.tabs(
        ["Production", "Validation", "Model history", "Forecast revision"]
    )
    with production_tab:
        _production(performance)
    with validation_tab:
        _validation(performance)
    with models_tab:
        _models(connection)
    with revision_tab:
        _revision(connection, config)


def _numbers(frame: pd.DataFrame) -> None:
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            name: st.column_config.NumberColumn(name, format=fmt)
            for name, fmt in METRIC_FORMATS.items()
            if name in frame.columns
        },
    )


def _production(performance: data.PerformanceView) -> None:
    st.caption(
        "Realized forecasts from the daily run. Forecasts whose origin sits "
        "inside their own model's training window are excluded: a model that has "
        "already seen the answer produces a flattering number and no information "
        "(OPERATING_SPEC.md section 3.2)."
    )
    if not performance.has_production_evidence:
        waiting = (
            f"{performance.pending:,} forecast rows are waiting for their target date"
            if performance.pending
            else "no forecast has been logged yet either"
        )
        st.info(
            f"**No forecast has resolved yet** - {waiting}. "
            "Production performance accumulates from the first daily run forward: "
            "a 30-day horizon says nothing for 30 days, a 365-day horizon nothing "
            "for a year. This is the normal state of a newly started system.\n\n"
            "Backfilling forecasts over past dates would not shortcut it - a "
            "model trained through those dates has already seen the answers, and "
            "`realization.drop_in_sample()` excludes them in code rather than "
            "trusting anyone to remember."
        )
        return

    table = data.metric_table(performance.production)
    _numbers(table)
    st.plotly_chart(charts.coverage_chart(table), width="stretch")

    regimes = data.regime_table(performance.production)
    if not regimes.empty:
        st.markdown("#### By market regime at the forecast origin")
        _numbers(regimes)


def _validation(performance: data.PerformanceView) -> None:
    st.caption(
        "Walk-forward folds on the inner block. These numbers chose the design, "
        "so they are the most optimistic honest estimate available and must not "
        "be read as production performance."
    )
    if performance.validation.empty:
        st.info(
            "No validation metrics stored. Run `python -m jobs.walk_forward`."
        )
    else:
        versions = sorted(performance.validation["model_version"].unique())
        chosen = st.selectbox("Model identity", versions, key="perf_validation_version")
        subset = performance.validation[
            performance.validation["model_version"] == chosen
        ]
        table = data.metric_table(subset)
        _numbers(table)
        st.plotly_chart(charts.coverage_chart(table), width="stretch")
        regimes = data.regime_table(subset)
        if not regimes.empty:
            st.markdown("#### By market regime")
            _numbers(regimes)

    st.markdown("#### Outer test")
    if performance.outer_test.empty:
        st.info(
            "**The outer test has not been evaluated.** The block reserved from "
            "2024-01-01 is still untouched, which is why the production training "
            "cutoff is pinned at the purge boundary. "
            "`python -m jobs.final_evaluation --confirm` spends it - once per "
            "design, permanently (VALIDATION_SPEC.md sections 4.4 and 4.5)."
        )
    else:
        st.caption(
            "The single independent verdict on this design. It was evaluated "
            "once and cannot be re-run; nothing here may be used to select a "
            "model, a feature or a hyperparameter."
        )
        _numbers(data.metric_table(performance.outer_test))


def _models(connection: sqlite3.Connection) -> None:
    history = data.model_history(connection)
    if history.empty:
        st.info("No model has been registered yet.")
        return
    st.dataframe(history, width="stretch", hide_index=True)
    st.caption(
        "`candidate` models are trained but not serving. Exactly one model can be "
        "`production` at a time, and promotion retires the incumbent in the same "
        "transaction (CLAUDE.md section 2.3). A `rejected` row carries the reason "
        "the weekly review recorded."
    )


def _revision(connection: sqlite3.Connection, config: AppConfig) -> None:
    horizons = config.forecast.required_evaluation_horizons
    default = len(horizons) - 1 if horizons else 0
    horizon = st.selectbox(
        "Horizon (days)", list(horizons), index=default, key="perf_revision_horizon"
    )
    revisions = data.revision_history(
        connection,
        config,
        horizon_days=int(horizon),
        limit=int(config.section("streamlit").get("forecast_revision_days", 30)),
    )
    st.plotly_chart(
        charts.revision_chart(revisions, horizon_days=int(horizon)), width="stretch"
    )
    st.caption(
        "How the same horizon's forecast moved as new origins arrived. This is "
        "visible long before any of those forecasts can be scored: a long-horizon "
        "median that swings from day to day is reacting to noise, and a line that "
        "only tracks the anchor price is saying nothing beyond today's close."
    )
    if not revisions.empty:
        st.dataframe(
            revisions[
                [
                    "forecast_origin_date",
                    "target_date",
                    "origin_close",
                    "predicted_price",
                    "change_pct",
                    "model_version",
                ]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "origin_close": st.column_config.NumberColumn(
                    "Anchor", format="$%,.0f"
                ),
                "predicted_price": st.column_config.NumberColumn(
                    "Median", format="$%,.0f"
                ),
                "change_pct": st.column_config.NumberColumn(
                    "Change", format="%+.1f%%"
                ),
            },
        )


__all__ = ["render"]
