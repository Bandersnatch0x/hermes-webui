"""DAO for schema_migrations table.

Standard migration bookkeeping. Tracks which schema versions have been applied.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    return dict(row)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SchemaMigrationsDAO:
    """Async operations on the schema_migrations table."""

    @staticmethod
    async def get_applied(db: aiosqlite.Connection) -> list[dict[str, Any]]:
        """Return all applied migrations, ordered by version."""
        cursor = await db.execute(
            "SELECT version, applied_at, description FROM schema_migrations ORDER BY version"
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]

    @staticmethod
    async def get_current_version(db: aiosqlite.Connection) -> int:
        """Return the highest applied schema version, or 0 if none."""
        cursor = await db.execute(
            "SELECT MAX(version) FROM schema_migrations"
        )
        row = await cursor.fetchone()
        return row[0] if row and row[0] is not None else 0

    @staticmethod
    async def is_applied(db: aiosqlite.Connection, version: int) -> bool:
        """Check if a specific version has been applied."""
        cursor = await db.execute(
            "SELECT 1 FROM schema_migrations WHERE version = ?",
            (version,),
        )
        row = await cursor.fetchone()
        return row is not None

    @staticmethod
    async def record(
        db: aiosqlite.Connection,
        version: int,
        description: str | None = None,
    ) -> None:
        """Record a migration as applied."""
        await db.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at, description) "
            "VALUES (?, ?, ?)",
            (version, _now(), description),
        )
        await db.commit()
        logger.info("Recorded schema migration v%d: %s", version, description)

    @staticmethod
    async def delete(db: aiosqlite.Connection, version: int) -> bool:
        """Remove a migration record. Use with caution (rollback)."""
        cursor = await db.execute(
            "DELETE FROM schema_migrations WHERE version = ?",
            (version,),
        )
        await db.commit()
        return cursor.rowcount > 0
