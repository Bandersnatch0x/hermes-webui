"""DAO for provider_sync_state table.

Tracks DB-to-live projection and reconcile state for each provider.
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


class ProviderSyncStateDAO:
    """Async CRUD for the provider_sync_state table."""

    @staticmethod
    async def get(db: aiosqlite.Connection, provider_id: str) -> dict[str, Any] | None:
        """Return sync state for a provider, or None."""
        cursor = await db.execute(
            "SELECT * FROM provider_sync_state WHERE provider_id = ?",
            (provider_id,),
        )
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    @staticmethod
    async def upsert(db: aiosqlite.Connection, provider_id: str, **fields: Any) -> dict[str, Any]:
        """Insert or update sync state fields.

        Accepted keyword args: last_projected_at, last_projected_hash,
        last_imported_at, last_credential_sync_at, last_error, retry_count.
        Only provided fields are set; others retain their current value.
        """
        existing = await ProviderSyncStateDAO.get(db, provider_id)
        if existing is None:
            # Insert with defaults
            defaults = {
                "last_projected_at": None,
                "last_projected_hash": None,
                "last_imported_at": None,
                "last_credential_sync_at": None,
                "last_error": None,
                "retry_count": 0,
            }
            defaults.update(fields)
            cols = ["provider_id"] + list(defaults.keys())
            placeholders = ", ".join(["?"] * len(cols))
            col_names = ", ".join(cols)
            await db.execute(
                f"INSERT INTO provider_sync_state ({col_names}) VALUES ({placeholders})",
                [provider_id] + list(defaults.values()),
            )
        else:
            if not fields:
                return existing
            set_clause = ", ".join(f"{k} = ?" for k in fields)
            values = list(fields.values()) + [provider_id]
            await db.execute(
                f"UPDATE provider_sync_state SET {set_clause} WHERE provider_id = ?",
                values,
            )

        await db.commit()
        return await ProviderSyncStateDAO.get(db, provider_id)

    @staticmethod
    async def set_projected(db: aiosqlite.Connection, provider_id: str, *, hash_value: str) -> dict[str, Any]:
        """Mark a provider as projected with current timestamp and hash."""
        return await ProviderSyncStateDAO.upsert(
            db, provider_id,
            last_projected_at=_now(),
            last_projected_hash=hash_value,
        )

    @staticmethod
    async def set_error(db: aiosqlite.Connection, provider_id: str, error: str) -> dict[str, Any]:
        """Record an error and increment retry count."""
        state = await ProviderSyncStateDAO.get(db, provider_id)
        retry = (state["retry_count"] + 1) if state else 1
        return await ProviderSyncStateDAO.upsert(
            db, provider_id,
            last_error=error,
            retry_count=retry,
        )

    @staticmethod
    async def clear_error(db: aiosqlite.Connection, provider_id: str) -> dict[str, Any]:
        """Clear error state and reset retry count."""
        return await ProviderSyncStateDAO.upsert(
            db, provider_id,
            last_error=None,
            retry_count=0,
        )

    @staticmethod
    async def delete(db: aiosqlite.Connection, provider_id: str) -> bool:
        """Delete sync state for a provider. Returns True if found."""
        cursor = await db.execute(
            "DELETE FROM provider_sync_state WHERE provider_id = ?",
            (provider_id,),
        )
        await db.commit()
        return cursor.rowcount > 0

    @staticmethod
    async def list_all(db: aiosqlite.Connection) -> list[dict[str, Any]]:
        """Return all sync state entries."""
        cursor = await db.execute(
            "SELECT * FROM provider_sync_state ORDER BY provider_id"
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
