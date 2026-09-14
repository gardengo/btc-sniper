"""SQLite connection handling and schema initialisation.

The database is the single persistence layer (ARCHITECTURE.md section 3).
Repository functions live in :mod:`src.storage.repositories`; this module only
owns connections, pragmas and the schema itself.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.utils.timeutils import utc_now_iso

logger = get_logger(__name__)

SCHEMA_PATH: Path = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION: str = "1"


def _apply_pragmas(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA busy_timeout = 5000")


def connect(database_path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    """Open a SQLite connection with the project pragmas applied.

    ``read_only`` opens the file via a URI so a reader (for example Streamlit)
    cannot mutate production data by accident.
    """
    path = Path(database_path)
    if read_only:
        if not path.is_file():
            raise FileNotFoundError(f"database does not exist: {path}")
        connection = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro", uri=True, timeout=10.0
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    _apply_pragmas(connection)
    return connection


@contextmanager
def open_connection(
    config: AppConfig, *, read_only: bool = False
) -> Iterator[sqlite3.Connection]:
    """Context-managed connection to the configured database."""
    connection = connect(config.storage.database_path, read_only=read_only)
    try:
        yield connection
    finally:
        connection.close()


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Explicit transaction: commit on success, roll back on any exception."""
    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()


def read_schema_sql() -> str:
    """The DDL shipped with the package."""
    return SCHEMA_PATH.read_text(encoding="utf-8")


def init_db(database_path: Path | str) -> Path:
    """Create the schema if missing and record the schema version.

    Idempotent: every statement in ``schema.sql`` uses ``IF NOT EXISTS``.
    """
    path = Path(database_path)
    connection = connect(path)
    try:
        with transaction(connection):
            connection.executescript(read_schema_sql())
            connection.execute(
                "INSERT INTO schema_meta (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                ("schema_version", SCHEMA_VERSION, utc_now_iso()),
            )
    finally:
        connection.close()
    logger.info("database ready at %s (schema version %s)", path, SCHEMA_VERSION)
    return path


def init_db_from_config(config: AppConfig) -> Path:
    """Initialise the database referenced by ``config``."""
    config.paths.ensure()
    return init_db(config.storage.database_path)


def table_names(connection: sqlite3.Connection) -> list[str]:
    """Names of the user tables present in the database."""
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [row["name"] for row in rows]


def get_schema_version(connection: sqlite3.Connection) -> str | None:
    """Schema version recorded in ``schema_meta``, if any."""
    row = connection.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    return None if row is None else str(row["value"])
