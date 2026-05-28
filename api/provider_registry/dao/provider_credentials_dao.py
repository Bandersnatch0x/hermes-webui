"""DAO for provider_credentials table.

Stores encrypted API keys. The api_key_ciphertext field must contain
actual encrypted payload (AES-GCM in phase 1), not plain text.
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


class ProviderCredentialsDAO:
    """Async CRUD for the provider_credentials table."""

    @staticmethod
    async def get(db: aiosqlite.Connection, provider_id: str) -> dict[str, Any] | None:
        """Return credential record for a provider, or None."""
        cursor = await db.execute(
            "SELECT provider_id, auth_type, api_key_hint, updated_at, version "
            "FROM provider_credentials WHERE provider_id = ?",
            (provider_id,),
        )
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    @staticmethod
    async def get_with_ciphertext(db: aiosqlite.Connection, provider_id: str) -> dict[str, Any] | None:
        """Return credential record INCLUDING ciphertext, or None.

        Use this only when you need to decrypt the key (e.g. for API calls).
        Do NOT expose ciphertext to the frontend.
        """
        cursor = await db.execute(
            "SELECT * FROM provider_credentials WHERE provider_id = ?",
            (provider_id,),
        )
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    @staticmethod
    async def upsert(
        db: aiosqlite.Connection,
        provider_id: str,
        *,
        auth_type: str = "bearer",
        api_key_ciphertext: str,
        api_key_hint: str | None = None,
    ) -> dict[str, Any]:
        """Insert or replace a credential record.

        Args:
            provider_id: The provider instance id (FK to provider_instances).
            auth_type: Authentication type (phase 1: 'bearer').
            api_key_ciphertext: AES-GCM encrypted payload.
            api_key_hint: Masked hint for UI display (e.g. 'sk-...abc').
        """
        now = _now()
        await db.execute(
            "INSERT OR REPLACE INTO provider_credentials "
            "(provider_id, auth_type, api_key_ciphertext, api_key_hint, updated_at, version) "
            "VALUES (?, ?, ?, ?, ?, COALESCE("
            "  (SELECT version + 1 FROM provider_credentials WHERE provider_id = ?), 1))",
            (provider_id, auth_type, api_key_ciphertext, api_key_hint, now, provider_id),
        )
        await db.commit()
        logger.info("Upserted credential for provider %s", provider_id)
        return await ProviderCredentialsDAO.get(db, provider_id)

    @staticmethod
    async def delete(db: aiosqlite.Connection, provider_id: str) -> bool:
        """Delete credential record. Returns True if found."""
        cursor = await db.execute(
            "DELETE FROM provider_credentials WHERE provider_id = ?",
            (provider_id,),
        )
        await db.commit()
        return cursor.rowcount > 0

    @staticmethod
    async def list_all(db: aiosqlite.Connection) -> list[dict[str, Any]]:
        """List all credential records (without ciphertext)."""
        cursor = await db.execute(
            "SELECT provider_id, auth_type, api_key_hint, updated_at, version "
            "FROM provider_credentials ORDER BY provider_id"
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
