"""Dashboard: what the system currently believes, and how much to trust it."""

from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from app import charts, data_access as data
from src.forecast.blending import BlendPolicy
from src.utils.config import AppConfig

HISTORY_CHOICES: dict[str, int] = {
    "6 months": 182,
    "1 year": 365,
    "2 years": 730,
    "All": 10_000,
}


def render(connection: sqlite3.Connection, config: AppConfig) -> None:
    st.subheader("Dashboard")

    view = data.load_forecast_view(connection, config)
    if view is None:
        st.warning(
            "No forecast is stored yet. Run `python -m jobs.generate_forecast` "
            "after the daily candle has closed."
        )
        return

    _header(connection, config, view)
    _provenance_notice(view, config)

    label = st.radio(
        "History shown",
        list(HISTORY_CHOICES),
        index=1,
        horizontal=True,
        key="dashboard_history",
    )
    history = data.price_history(connection, config, days=HISTORY_CHOICES[label])
    st.plotly_chart(
        charts.forecast_chart(
            history,
            view.curve,
            origin_date=view.origin_date,
            origin_close=view.origin_close,
            current_price=view.current_price,
            intervals=config.forecast.show_intervals,
        ),
        width="stretch",
    )
    st.caption(
        "The bands are PCHIP curves through the forecast grid, drawn so the eye "
        "can follow them between horizons. They are never stored and never "
        "scored; every number in the table below comes from a real grid horizon "
        "(CLAUDE.md section 5)."
    )

    _summary(view)


def _header(
    connection: sqlite3.Connection, config: AppConfig, view: data.ForecastView
) -> None:
    snapshot = data.current_price(connection, config)
    production = data.production_version(connection)

    left, middle, right = st.columns(3)
    with left:
        if snapshot is None:
            st.metric("Current price", "-")
            st.caption(
                "No realtime tick stored. Start `python -m jobs.stream_realtime_price`."
            )
        else:
            delta = 100.0 * (snapshot.price / view.origin_close - 1.0)
            st.metric(
                "Current price",
                f"${snapshot.price:,.0f}",
                delta=f"{delta:+.2f}% vs last close",
            )
            age = f"{snapshot.age_seconds:.0f}s old via {snapshot.transport}"
            st.caption(f":red[STALE] - {age}" if snapshot.is_stale else age)
    with middle:
        st.metric("Forecast anchor", view.origin_date.strftime("%Y-%m-%d"))
        st.caption(f"generated {view.created_at}")
    with right:
        st.metric("Production model", production or "none")
        st.caption(f"forecast from `{view.model_version}`")

    if production is None:
        st.warning(
            "**No model has been promoted to production.** This forecast came "
            "from a candidate, so it is research output, not a production "
            "forecast. Promotion is a separate decision made by "
            "`python -m jobs.weekly_model_review --apply` (CLAUDE.md section 2.3)."
        )
    elif not view.is_production:
        st.warning(
            f"This forecast came from `{view.model_version}`, which is **not** "
            f"the production model (`{production}`). It was generated with "
            "`--model-version`, a research path."
        )

    st.caption(
        "The forecast is anchored to the last closed daily candle and refreshes "
        "once a day. The current price above updates continuously and never "
        "moves the forecast (CLAUDE.md section 2.4)."
    )


def _provenance_notice(view: data.ForecastView, config: AppConfig) -> None:
    provenance = view.provenance()
    if provenance.empty:
        return
    counts = provenance["blend_source"].value_counts()
    policy = BlendPolicy.from_config(config)
    with st.expander(
        f"What produced this forecast - {policy.describe()}", expanded=False
    ):
        st.markdown(
            "Walk-forward validation found the tree beats the `no_change` "
            "baseline in every fold at 1 day, ties at 7 days and is **worse** "
            "from 30 days outward (MODEL_SPEC.md section 6.5). Serving the model "
            "everywhere would knowingly ship a worse forecast, so the weight "
            "falls to zero and most of the curve above is the unconditional "
            "baseline distribution."
        )
        st.write(
            {str(source): int(count) for source, count in counts.items()}
        )
        st.plotly_chart(charts.provenance_chart(provenance), width="stretch")


def _summary(view: data.ForecastView) -> None:
    summary = data.horizon_summary(view)
    if summary.empty:
        st.info("None of the 1M / 3M / 6M / 12M horizons is in this forecast grid.")
        return

    st.markdown("#### Key horizons")
    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
        column_config={
            "horizon": st.column_config.TextColumn("Horizon"),
            "horizon_days": st.column_config.NumberColumn("Days", format="%d"),
            "target_date": st.column_config.TextColumn("Target date"),
            "median": st.column_config.NumberColumn("Median", format="$%,.0f"),
            "change_pct": st.column_config.NumberColumn("Change", format="%+.1f%%"),
            "low_95": st.column_config.NumberColumn("95% low", format="$%,.0f"),
            "high_95": st.column_config.NumberColumn("95% high", format="$%,.0f"),
            "source": st.column_config.TextColumn("Produced by"),
        },
    )
    if (summary["change_pct"].abs() < 1e-9).all():
        st.caption(
            "Every median equals today's close. That is the baseline's answer, "
            "not a bug: on this data the historical median multi-day return is "
            "indistinguishable from zero, and the model has no demonstrated edge "
            "at these horizons. The interval, not the median, is what these rows "
            "are for."
        )


__all__ = ["render"]
