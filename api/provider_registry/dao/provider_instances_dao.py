"""DAO for provider_instances table.

Supports CRUD with soft delete. List operations exclude soft-deleted rows
by default (deleted_at IS NULL).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_SELECT_COLS = (
    "id, kind, provider_key, display_name, adapter_type, base_url, "
    "enabled, response_format, default_model, models_endpoint, "
    "usage_strategy, usage_endpoint_url, usage_parser_type, "
    "is_builtin_locked, sort_order, created_at, updated_at, deleted_at"
)


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    return dict(row)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProviderInstancesDAO:
    """Async CRUD for the provider_instances table."""

    # ---- Read ----

    @staticmethod
    async def get_by_id(db: aiosqlite.Connection, instance_id: str) -> dict[str, Any] | None:
        """Return a single instance by id, or None if not found / soft-deleted."""
        cursor = await db.execute(
            f"SELECT {_SELECT_COLS} FROM provider_instances WHERE id = ? AND deleted_at IS NULL",
            (instance_id,),
        )
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    @staticmethod
    async def get_by_id_include_deleted(db: aiosqlite.Connection, instance_id: str) -> dict[str, Any] | None:
        """Return a single instance by id regardless of soft-delete status."""
        cursor = await db.execute(
            f"SELECT {_SELECT_COLS} FROM provider_instances WHERE id = ?",
            (instance_id,),
        )
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    @staticmethod
    async def get_by_provider_key(db: aiosqlite.Connection, provider_key: str) -> dict[str, Any] | None:
        """Return instance by unique provider_key (excludes soft-deleted)."""
        cursor = await db.execute(
            f"SELECT {_SELECT_COLS} FROM provider_instances WHERE provider_key = ? AND deleted_at IS NULL",
            (provider_key,),
        )
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    @staticmethod
    async def list_all(
        db: aiosqlite.Connection,
        *,
        include_deleted: bool = False,
        kind: str | None = None,
        enabled_only: bool = False,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List provider instances with optional filters.

        Default ordering: sort_order ASC, display_name COLLATE NOCASE ASC, id ASC.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if not include_deleted:
            conditions.append("deleted_at IS NULL")
        if kind is not None:
            conditions.append("kind = ?")
            params.append(kind)
        if enabled_only:
            conditions.append("enabled = 1")

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = (
            f"SELECT {_SELECT_COLS} FROM provider_instances{where}"
            f" ORDER BY sort_order ASC, display_name COLLATE NOCASE ASC, id ASC"
            f" LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]

    @staticmethod
    async def count(db: aiosqlite.Connection, *, include_deleted: bool = False) -> int:
        """Return total count of provider instances."""
        where = "" if include_deleted else " WHERE deleted_at IS NULL"
        cursor = await db.execute(f"SELECT COUNT(*) FROM provider_instances{where}")
        row = await cursor.fetchone()
        return row[0]

    # ---- Create ----

    @staticmethod
    async def create(db: aiosqlite.Connection, data: dict[str, Any]) -> dict[str, Any]:
        """Insert a new provider instance. Returns the created row as dict.

        Required fields: id, kind, provider_key, display_name, adapter_type.
        created_at/updated_at default to now if not provided.
        """
        now = _now()
        data.setdefault("created_at", now)
        data.setdefault("updated_at", now)
        data.setdefault("enabled", 1)
        data.setdefault("is_builtin_locked", 0)
        data.setdefault("sort_order", 0)
        data.setdefault("usage_strategy", "auto")

        cols = list(data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)

        await db.execute(
            f"INSERT INTO provider_instances ({col_names}) VALUES ({placeholders})",
            list(data.values()),
        )
        await db.commit()

        logger.info("Created provider instance %s (%s)", data.get("id"), data.get("provider_key"))
        return data

    # ---- Update ----

    @staticmethod
    async def update(db: aiosqlite.Connection, instance_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update fields on an existing instance. Returns updated row or None.

        Automatically sets updated_at to now.
        Raises ValueError if instance is soft-deleted or not found.
        """
        existing = await ProviderInstancesDAO.get_by_id(db, instance_id)
        if existing is None:
            # Check if it exists but is soft-deleted
            deleted = await ProviderInstancesDAO.get_by_id_include_deleted(db, instance_id)
            if deleted and deleted["deleted_at"] is not None:
                raise ValueError(f"Cannot update soft-deleted instance {instance_id}")
            return None

        # Prevent overwriting immutable fields
        for immutable in ("id", "created_at"):
            updates.pop(immutable, None)

        updates["updated_at"] = _now()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [instance_id]

        await db.execute(
            f"UPDATE provider_instances SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        return await ProviderInstancesDAO.get_by_id(db, instance_id)

    # ---- Soft Delete ----

    @staticmethod
    async def soft_delete(db: aiosqlite.Connection, instance_id: str) -> bool:
        """Soft-delete an instance by setting deleted_at. Returns True if found.

        Idempotent: returns True even if already soft-deleted.
        """
        now = _now()
        cursor = await db.execute(
            "UPDATE provider_instances SET deleted_at = ?, updated_at = ? "
            "WHERE id = ? AND deleted_at IS NULL",
            (now, now, instance_id),
        )
        await db.commit()
        affected = cursor.rowcount
        if affected:
            logger.info("Soft-deleted provider instance %s", instance_id)
        return affected > 0

    @staticmethod
    async def restore(db: aiosqlite.Connection, instance_id: str) -> bool:
        """Restore a soft-deleted instance. Returns True if restored."""
        now = _now()
        cursor = await db.execute(
            "UPDATE provider_instances SET deleted_at = NULL, updated_at = ? "
            "WHERE id = ? AND deleted_at IS NOT NULL",
            (now, instance_id),
        )
        await db.commit()
        return cursor.rowcount > 0

    # ---- Hard Delete ----

    @staticmethod
    async def hard_delete(db: aiosqlite.Connection, instance_id: str) -> bool:
        """Permanently remove an instance. Also cascades to credentials etc.

        Use with caution — prefer soft_delete for most operations.
        """
        cursor = await db.execute("DELETE FROM provider_instances WHERE id = ?", (instance_id,))
        await db.commit()
        if cursor.rowcount:
            logger.warning("Hard-deleted provider instance %s", instance_id)
        return cursor.rowcount > 0
