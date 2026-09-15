"""Streamlit dashboard for BTC Sniper (OPERATING_SPEC.md section 8).

Read-only by construction: `app.data_access` opens the database through a
read-only URI, so nothing a viewer clicks can change a stored forecast or
promote a model. Those are job decisions, not page decisions.
"""
