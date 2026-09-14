"""Logging setup shared by jobs, library code and the Streamlit app.

CLAUDE.md section 9 requires that log output makes the cause of a failure
explicit, so the formatter always carries module and line number.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from src.utils.config import AppConfig

LOG_FORMAT: str = "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(message)s"
DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%SZ"  # converter is gmtime, so Z is literal
_CONFIGURED: bool = False


class _UTCFormatter(logging.Formatter):
    """Formatter that emits UTC timestamps regardless of machine locale."""

    converter = time.gmtime


def setup_logging(
    config: AppConfig | None = None,
    *,
    level: str | None = None,
    log_file: Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger once per process and return it.

    Repeated calls are no-ops unless ``force`` is set, which keeps Streamlit
    re-runs from stacking duplicate handlers.
    """
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED and not force:
        return root

    resolved_level = (level or (config.logging.level if config else "INFO")).upper()
    to_console = config.logging.console if config else True
    target_file = log_file if log_file is not None else (config.logging.file if config else None)

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(resolved_level)
    formatter = _UTCFormatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if to_console:
        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    if target_file is not None:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Module-level logger accessor."""
    return logging.getLogger(name)
