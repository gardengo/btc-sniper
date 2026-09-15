"""Plotly figures for the dashboard. No Streamlit here either, so they can be
built and inspected in a test.

Two rules run through all of these, both from CLAUDE.md section 5:

**The boundary between what happened and what is guessed must be obvious.** Past
prices and forecast bands are never drawn in the same colour, and a NOW marker
sits on the last closed candle.

**Interpolated points are drawn, never quoted.** The bands are PCHIP curves
through the grid horizons so the eye can follow them; every number the pages put
in a table comes from a real grid horizon instead.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.forecast.quantiles import resolve_interval

TEMPLATE: str = "plotly_white"
HISTORY_COLOUR: str = "#111827"
MEDIAN_COLOUR: str = "#2563eb"
# Widest band palest: the 95% interval is the least surprising place for the
# price to land, so it should be the least visually insistent thing on the chart.
BAND_COLOURS: dict[float, str] = {
    0.50: "rgba(37, 99, 235, 0.32)",
    0.80: "rgba(37, 99, 235, 0.18)",
    0.95: "rgba(37, 99, 235, 0.09)",
}
BASELINE_COLOUR: str = "#9ca3af"


def _empty(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5
    )
    figure.update_layout(
        template=TEMPLATE,
        height=380,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def forecast_chart(
    history: pd.DataFrame,
    curve: pd.DataFrame,
    *,
    origin_date: pd.Timestamp,
    origin_close: float,
    current_price: float | None = None,
    intervals: tuple[float, ...] = (0.50, 0.80, 0.95),
    title: str = "BTC-USD: history and forecast",
) -> go.Figure:
    """Past closes, the forecast median, and its prediction bands on one axis.

    ``curve`` is the daily interpolated frame from
    `src.forecast.interpolate.interpolated_forecast_curve`: a ``date`` column
    plus one column per quantile level.
    """
    if curve.empty:
        return _empty("No forecast is stored yet.")

    levels = sorted(float(c) for c in curve.columns if c != "date")
    dates = pd.to_datetime(curve["date"])
    figure = go.Figure()

    # Bands widest-first so the narrow ones draw on top of the wide ones.
    for interval in sorted(intervals, reverse=True):
        try:
            low, high = resolve_interval(interval, tuple(levels))
        except Exception:  # noqa: BLE001 - an unavailable band is skipped, not fatal
            continue
        colour = BAND_COLOURS.get(interval, "rgba(37, 99, 235, 0.12)")
        figure.add_trace(
            go.Scatter(
                x=dates, y=curve[low], mode="lines", line={"width": 0},
                hoverinfo="skip", showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=dates, y=curve[high], mode="lines", line={"width": 0},
                fill="tonexty", fillcolor=colour,
                name=f"{int(interval * 100)}% interval",
                hovertemplate="%{y:,.0f}<extra></extra>",
            )
        )

    median = min(levels, key=lambda level: abs(level - 0.50))
    figure.add_trace(
        go.Scatter(
            x=dates, y=curve[median], mode="lines",
            line={"color": MEDIAN_COLOUR, "width": 2},
            name="median forecast",
            hovertemplate="%{x|%Y-%m-%d} %{y:,.0f}<extra>median</extra>",
        )
    )

    if not history.empty:
        figure.add_trace(
            go.Scatter(
                x=history.index, y=history["close"], mode="lines",
                line={"color": HISTORY_COLOUR, "width": 1.6},
                name="actual close",
                hovertemplate="%{x|%Y-%m-%d} %{y:,.0f}<extra>actual</extra>",
            )
        )

    figure.add_vline(
        x=origin_date, line={"color": "#6b7280", "width": 1, "dash": "dot"}
    )
    figure.add_annotation(
        x=origin_date, y=1.0, yref="paper", yanchor="bottom",
        text="NOW", showarrow=False, font={"size": 11, "color": "#6b7280"},
    )
    figure.add_trace(
        go.Scatter(
            x=[origin_date], y=[origin_close], mode="markers",
            marker={"color": HISTORY_COLOUR, "size": 7},
            name="last closed candle",
            hovertemplate="anchor %{y:,.0f}<extra></extra>",
        )
    )
    if current_price is not None:
        figure.add_trace(
            go.Scatter(
                x=[origin_date], y=[current_price], mode="markers",
                marker={"color": "#dc2626", "size": 7, "symbol": "diamond"},
                name="current price",
                hovertemplate="live %{y:,.0f}<extra></extra>",
            )
        )

    figure.update_layout(
        template=TEMPLATE,
        title=title,
        height=520,
        hovermode="x unified",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        yaxis={"title": "USD", "tickformat": ",.0f"},
        xaxis={"title": None},
    )
    return figure


def provenance_chart(provenance: pd.DataFrame) -> go.Figure:
    """How much of each horizon's forecast came from the model.

    Worth a chart rather than a footnote: the weight falls to zero by 30 days, so
    most of the curve a reader is looking at is the unconditional baseline
    distribution, not a model output (MODEL_SPEC.md section 6.5).
    """
    if provenance.empty:
        return _empty("This forecast has no recorded provenance.")
    frame = provenance.dropna(subset=["model_weight"])
    if frame.empty:
        return _empty("This forecast predates provenance recording.")
    figure = go.Figure(
        go.Bar(
            x=frame["horizon_days"],
            y=frame["model_weight"],
            marker={"color": MEDIAN_COLOUR},
            hovertemplate="h=%{x}d model weight %{y:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        template=TEMPLATE,
        height=260,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis={"title": "model weight", "range": [0, 1.05]},
        xaxis={"title": "horizon (days)"},
        showlegend=False,
    )
    return figure


def revision_chart(revisions: pd.DataFrame, *, horizon_days: int) -> go.Figure:
    """The same horizon's forecast, as it was made on each origin.

    Plotted against the origin date, with the anchor price beside it. The gap
    between the two lines is the model's actual claim; if the forecast line only
    tracks the anchor, the model is saying nothing beyond today's price.
    """
    if revisions.empty:
        return _empty("No forecast history yet.")
    if len(revisions) < 2:
        return _empty(
            f"Only one {horizon_days}-day forecast is stored. "
            "The revision chart needs at least two origins to show movement."
        )

    origins = pd.to_datetime(revisions["forecast_origin_date"])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=origins, y=revisions["origin_close"], mode="lines",
            line={"color": BASELINE_COLOUR, "width": 1.5},
            name="anchor price on that day",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=origins, y=revisions["predicted_price"], mode="lines+markers",
            line={"color": MEDIAN_COLOUR, "width": 2},
            name=f"{horizon_days}-day median forecast",
        )
    )
    figure.update_layout(
        template=TEMPLATE,
        title=f"{horizon_days}-day forecast revision history",
        height=400,
        hovermode="x unified",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        yaxis={"title": "USD", "tickformat": ",.0f"},
        xaxis={"title": "forecast origin"},
    )
    return figure


def coverage_chart(table: pd.DataFrame) -> go.Figure:
    """Realized interval coverage against the nominal level it is aiming at.

    The dashed nominal lines are the point of the chart. A 95% band covering 99%
    of outcomes is miscalibrated exactly as much as one covering 91%, and only a
    reference line makes that readable.
    """
    if table.empty:
        return _empty("No realized forecast has been scored yet.")
    columns = [name for name in table.columns if name.startswith("coverage_")]
    if not columns:
        return _empty("No coverage metric is stored.")

    figure = go.Figure()
    for column in sorted(columns):
        nominal = int(column.rsplit("_", 1)[-1]) / 100.0
        figure.add_trace(
            go.Scatter(
                x=table["horizon_days"], y=table[column], mode="lines+markers",
                name=f"{column} (nominal {nominal:.0%})",
            )
        )
        figure.add_hline(
            y=nominal, line={"width": 1, "dash": "dot", "color": "#9ca3af"}
        )
    figure.update_layout(
        template=TEMPLATE,
        title="Interval coverage vs nominal",
        height=380,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        yaxis={"title": "realized coverage", "range": [0, 1.05], "tickformat": ".0%"},
        xaxis={"title": "horizon (days)"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return figure


__all__ = [
    "coverage_chart",
    "forecast_chart",
    "provenance_chart",
    "revision_chart",
]
