"""DAO for provider_usage_cache table.

Caches usage/quota status per provider. Status values:
available, unsupported, error, unknown.
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


class ProviderUsageCacheDAO:
    """Async CRUD for the provider_usage_cache table."""

    @staticmethod
    async def get(db: aiosqlite.Connection, provider_id: str) -> dict[str, Any] | None:
        """Return cached usage for a provider, or None."""
        cursor = await db.execute(
            "SELECT * FROM provider_usage_cache WHERE provider_id = ?",
            (provider_id,),
        )
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    @staticmethod
    async def upsert(
        db: aiosqlite.Connection,
        provider_id: str,
        *,
        status: str,
        payload_json: str | None = None,
        message: str | None = None,
    ) -> dict[str, Any]:
        """Insert or update usage cache for a provider.

        Args:
            provider_id: The provider instance id.
            status: One of 'available', 'unsupported', 'error', 'unknown'.
            payload_json: Optional JSON string with usage details.
            message: Optional human-readable status message.
        """
        now = _now()
        await db.execute(
            "INSERT OR REPLACE INTO provider_usage_cache "
            "(provider_id, status, payload_json, message, fetched_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (provider_id, status, payload_json, message, now),
        )
        await db.commit()
        return await ProviderUsageCacheDAO.get(db, provider_id)

    @staticmethod
    async def delete(db: aiosqlite.Connection, provider_id: str) -> bool:
        """Delete usage cache for a provider. Returns True if found."""
        cursor = await db.execute(
            "DELETE FROM provider_usage_cache WHERE provider_id = ?",
            (provider_id,),
        )
        await db.commit()
        return cursor.rowcount > 0

    @staticmethod
    async def list_all(db: aiosqlite.Connection) -> list[dict[str, Any]]:
        """Return all cached usage entries."""
        cursor = await db.execute(
            "SELECT * FROM provider_usage_cache ORDER BY provider_id"
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
