"""Markdown rendering helpers shared by the report modules.

Extracted so the data-quality report and the evaluation reports format tables
identically; two implementations would drift and produce reports that look
subtly different for no reason.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def markdown_table(
    frame: pd.DataFrame,
    float_format: str = "{:,.4f}",
    column_formats: Mapping[str, str] | None = None,
) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table.

    ``column_formats`` overrides ``float_format`` per column, so a count column
    can print as ``1,962`` instead of ``1,962.0000`` in a table that is otherwise
    four-decimal metrics.

    Cell values are escaped for ``|`` because a pipe inside a metric name or a
    message silently breaks the column layout.
    """
    if frame.empty:
        return "_(no rows)_\n"
    overrides = dict(column_formats or {})
    formatted = frame.copy()
    for column in formatted.columns:
        # Floats get the shared format; integers stay verbatim unless the caller
        # asks otherwise, so adding this parameter cannot reformat existing reports.
        spec = overrides.get(str(column))
        if spec is None and pd.api.types.is_float_dtype(formatted[column]):
            spec = float_format
        if spec is not None:
            formatted[column] = formatted[column].map(
                lambda v, spec=spec: "" if pd.isna(v) else spec.format(v)
            )
        else:
            formatted[column] = formatted[column].astype(str)
    header = "| " + " | ".join(_cell(c) for c in formatted.columns) + " |"
    divider = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    body = [
        "| " + " | ".join(_cell(value) for value in row) + " |"
        for row in formatted.itertuples(index=False)
    ]
    return "\n".join([header, divider, *body]) + "\n"


def _cell(value: object) -> str:
    return str(value).replace("|", r"\|")
