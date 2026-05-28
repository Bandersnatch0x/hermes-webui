"""DAO for provider_models_cache table.

Composite primary key: (provider_id, model_id).
Used to cache model inventories fetched from provider APIs.
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


class ProviderModelsCacheDAO:
    """Async CRUD for the provider_models_cache table."""

    @staticmethod
    async def get_models(
        db: aiosqlite.Connection, provider_id: str
    ) -> list[dict[str, Any]]:
        """Return all cached models for a provider, ordered by model_id."""
        cursor = await db.execute(
            "SELECT provider_id, model_id, label, raw_json, fetched_at "
            "FROM provider_models_cache WHERE provider_id = ? ORDER BY model_id",
            (provider_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]

    @staticmethod
    async def get_model(
        db: aiosqlite.Connection, provider_id: str, model_id: str
    ) -> dict[str, Any] | None:
        """Return a single cached model entry."""
        cursor = await db.execute(
            "SELECT * FROM provider_models_cache WHERE provider_id = ? AND model_id = ?",
            (provider_id, model_id),
        )
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    @staticmethod
    async def replace_all(
        db: aiosqlite.Connection,
        provider_id: str,
        models: list[dict[str, Any]],
    ) -> int:
        """Replace the entire model cache for a provider.

        Deletes existing entries and inserts new ones atomically.

        Args:
            provider_id: The provider instance id.
            models: List of dicts with keys: model_id, label (optional), raw_json (optional).

        Returns:
            Number of models inserted.
        """
        now = _now()
        await db.execute(
            "DELETE FROM provider_models_cache WHERE provider_id = ?",
            (provider_id,),
        )
        for m in models:
            await db.execute(
                "INSERT INTO provider_models_cache "
                "(provider_id, model_id, label, raw_json, fetched_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (provider_id, m["model_id"], m.get("label"), m.get("raw_json"), now),
            )
        await db.commit()
        logger.info("Replaced models cache for %s: %d models", provider_id, len(models))
        return len(models)

    @staticmethod
    async def delete_for_provider(db: aiosqlite.Connection, provider_id: str) -> int:
        """Delete all cached models for a provider. Returns count deleted."""
        cursor = await db.execute(
            "DELETE FROM provider_models_cache WHERE provider_id = ?",
            (provider_id,),
        )
        await db.commit()
        return cursor.rowcount

    @staticmethod
    async def count(db: aiosqlite.Connection, provider_id: str) -> int:
        """Count cached models for a provider."""
        cursor = await db.execute(
            "SELECT COUNT(*) FROM provider_models_cache WHERE provider_id = ?",
            (provider_id,),
        )
        row = await cursor.fetchone()
        return row[0]
