"""Database connection helper for the provider registry.

Provides WAL-mode aiosqlite connections with busy timeout.
"""
from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite

from api.provider_registry.dao.schema import initialize_database

logger = logging.getLogger(__name__)

# Default database path; callers can override via get_connection(path=...)
_DEFAULT_DB_DIR = Path.home() / ".hermes" / "webui"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "providers.db"


async def get_connection(
    path: str | Path | None = None,
    *,
    run_migrations: bool = False,
) -> aiosqlite.Connection:
    """Open (or create) the providers.db with WAL + busy_timeout.

    Args:
        path: Override database file path. Defaults to ~/.hermes/webui/providers.db.
        run_migrations: If True, run initialize_database() after connecting.

    Returns:
        An aiosqlite.Connection with row_factory set to aiosqlite.Row.
    """
    db_path = Path(path) if path else _DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row

    # Connection-level pragmas (redundant with schema init but safe)
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA busy_timeout=5000")
    await db.execute("PRAGMA foreign_keys=ON")

    if run_migrations:
        await initialize_database(db)

    logger.debug("Connected to %s", db_path)
    return db
