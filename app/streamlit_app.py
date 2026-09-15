"""BTC Sniper dashboard entry point.

Run with::

    .venv/Scripts/python.exe -m streamlit run app/streamlit_app.py

Pages come from `streamlit.pages` in `config.yaml` rather than from a hard-coded
list, so turning one off is a config change. The database is opened read-only for
every rerun: this is a window onto what the jobs produced, and a viewer must not
be able to promote a model or overwrite a forecast by clicking something.

The page deliberately does not auto-refresh on a timer. The forecast changes once
a day and the realtime price comes from whatever `jobs.stream_realtime_price` has
stored, so a timer would spend CPU redrawing an unchanged chart; the Refresh
button re-reads on demand and says when it last did.
"""

from __future__ import annotations

import os
import sqlite3

import streamlit as st

from src.utils.config import AppConfig, load_config
from src.utils.timeutils import utc_now_iso

PAGE_TITLE: str = "BTC Sniper"
PAGES: dict[str, str] = {
    "dashboard": "Dashboard",
    "prediction_log": "Prediction Log",
    "model_performance": "Model Performance",
}


CONFIG_ENV: str = "BTC_SNIPER_CONFIG"


def _config() -> AppConfig:
    """The configuration this app reads.

    `BTC_SNIPER_CONFIG` points it at a different `config.yaml`, which is how the
    smoke test runs the real pages against a synthetic database instead of
    whatever happens to be in the developer's `data/` directory.
    """
    return load_config(os.environ.get(CONFIG_ENV) or None)


def _render_page(
    name: str, connection: sqlite3.Connection, config: AppConfig
) -> None:
    if name == "dashboard":
        from app.views import dashboard

        dashboard.render(connection, config)
    elif name == "prediction_log":
        from app.views import prediction_log

        prediction_log.render(connection, config)
    elif name == "model_performance":
        from app.views import model_performance

        model_performance.render(connection, config)
    else:
        st.error(f"Unknown page {name!r}. Check `streamlit.pages` in config.yaml.")


def enabled_pages(config: AppConfig) -> list[str]:
    """Pages listed in config that this app actually implements."""
    configured = [
        str(name) for name in config.section("streamlit").get("pages", []) or []
    ]
    known = [name for name in configured if name in PAGES]
    return known or list(PAGES)


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="*", layout="wide")
    config = _config()

    st.title(PAGE_TITLE)
    st.caption(
        "A research and monitoring system for BTC price forecasts. It places no "
        "orders, optimises no portfolio and sends no alerts (CLAUDE.md section 1)."
    )

    pages = enabled_pages(config)
    with st.sidebar:
        st.header(PAGE_TITLE)
        choice = st.radio(
            "Page", pages, format_func=lambda name: PAGES[name], key="page"
        )
        st.divider()
        if st.button("Refresh data", width="stretch"):
            st.session_state["refreshed_at"] = utc_now_iso()
            st.rerun()
        refreshed = st.session_state.get("refreshed_at")
        st.caption(f"last manual refresh: {refreshed}" if refreshed else "read on load")
        st.divider()
        st.caption(f"config `{config.config_version}`")
        st.caption(f"database `{config.storage.database_path.name}` (read-only)")

    try:
        connection = _open(config)
    except FileNotFoundError:
        st.error(
            f"No database at `{config.storage.database_path}`. Run "
            "`python -m jobs.update_market_data --full-refresh` to create it."
        )
        return

    try:
        _render_page(choice, connection, config)
    finally:
        connection.close()


def _open(config: AppConfig) -> sqlite3.Connection:
    from app.data_access import read_only_connection

    return read_only_connection(config)


if __name__ == "__main__":
    main()
