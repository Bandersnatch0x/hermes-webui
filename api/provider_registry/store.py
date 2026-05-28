"""SQLite DAO layer for the provider registry.

All provider instance, credential, model cache, usage cache, and sync state
operations go through this module.  The database lives at
<hermes_home>/webui/providers.db.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _db_path() -> Path:
    """Resolve the providers.db path from the active Hermes home."""
    from api.profiles import get_active_hermes_home
    home = get_active_hermes_home()
    db_dir = home / "webui"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "providers.db"


def _connect() -> sqlite3.Connection:
    """Return a new connection with WAL mode and row_factory."""
    path = _db_path()
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def transaction():
    """Context manager yielding a connection inside a transaction."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Schema & Migrations
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 1

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS provider_instances (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL CHECK(kind IN ('official', 'custom')),
    provider_key    TEXT NOT NULL UNIQUE,
    display_name    TEXT NOT NULL,
    adapter_type    TEXT NOT NULL CHECK(adapter_type IN ('openai', 'anthropic')),
    base_url        TEXT,
    enabled         INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
    response_format TEXT,
    default_model   TEXT,
    models_endpoint TEXT,
    usage_strategy  TEXT NOT NULL DEFAULT 'auto'
                    CHECK(usage_strategy IN ('auto', 'endpoint', 'auto+endpoint', 'none')),
    usage_endpoint_url  TEXT,
    usage_parser_type   TEXT,
    is_builtin_locked   INTEGER NOT NULL DEFAULT 0 CHECK(is_builtin_locked IN (0, 1)),
    sort_order      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    deleted_at      TEXT
);

CREATE TABLE IF NOT EXISTS provider_credentials (
    provider_id         TEXT PRIMARY KEY REFERENCES provider_instances(id),
    auth_type           TEXT NOT NULL DEFAULT 'bearer',
    api_key_ciphertext  TEXT,
    api_key_hint        TEXT,
    updated_at          TEXT NOT NULL,
    version             INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS provider_models_cache (
    provider_id TEXT NOT NULL REFERENCES provider_instances(id),
    model_id    TEXT NOT NULL,
    label       TEXT,
    raw_json    TEXT,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (provider_id, model_id)
);

CREATE TABLE IF NOT EXISTS provider_usage_cache (
    provider_id TEXT PRIMARY KEY REFERENCES provider_instances(id),
    status      TEXT NOT NULL CHECK(status IN ('available', 'unsupported', 'error', 'unknown')),
    payload_json TEXT,
    message     TEXT,
    fetched_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_sync_state (
    provider_id             TEXT PRIMARY KEY REFERENCES provider_instances(id),
    last_projected_at       TEXT,
    last_projected_hash     TEXT,
    last_imported_at        TEXT,
    last_credential_sync_at TEXT,
    last_error              TEXT,
    retry_count             INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
"""


def init_db() -> None:
    """Create tables if they don't exist and apply pending migrations."""
    with transaction() as conn:
        conn.executescript(_CREATE_TABLES)
        # Record migration
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (_SCHEMA_VERSION, _now()),
        )


def _ensure_db(conn: sqlite3.Connection) -> None:
    """Lazily init if tables are missing."""
    try:
        conn.execute("SELECT COUNT(*) FROM provider_instances")
    except sqlite3.OperationalError:
        conn.close()
        init_db()


# ---------------------------------------------------------------------------
# Provider Instances CRUD
# ---------------------------------------------------------------------------

def list_providers(include_disabled: bool = False) -> list[dict[str, Any]]:
    """List all non-deleted provider instances."""
    with _connect() as conn:
        where = "WHERE deleted_at IS NULL"
        if not include_disabled:
            where += " AND enabled = 1"
        rows = conn.execute(
            f"SELECT * FROM provider_instances {where} ORDER BY sort_order ASC, display_name COLLATE NOCASE ASC, id ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_provider(provider_id: str) -> dict[str, Any] | None:
    """Get a single provider instance by id."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM provider_instances WHERE id = ? AND deleted_at IS NULL",
            (provider_id,),
        ).fetchone()
        return dict(row) if row else None


def get_provider_by_key(provider_key: str) -> dict[str, Any] | None:
    """Get a single provider instance by its unique provider_key (e.g. custom:xiaomi-cn)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM provider_instances WHERE provider_key = ? AND deleted_at IS NULL",
            (provider_key,),
        ).fetchone()
        return dict(row) if row else None


def create_provider(data: dict[str, Any]) -> dict[str, Any]:
    """Create a new provider instance. Returns the created row."""
    now = _now()
    pid = data.get("id") or _uuid()
    with transaction() as conn:
        conn.execute(
            """INSERT INTO provider_instances
               (id, kind, provider_key, display_name, adapter_type, base_url,
                enabled, response_format, default_model, models_endpoint,
                usage_strategy, usage_endpoint_url, usage_parser_type,
                is_builtin_locked, sort_order, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pid,
                data.get("kind", "custom"),
                data["provider_key"],
                data["display_name"],
                data["adapter_type"],
                data.get("base_url"),
                int(data.get("enabled", True)),
                data.get("response_format"),
                data.get("default_model"),
                data.get("models_endpoint"),
                data.get("usage_strategy", "auto"),
                data.get("usage_endpoint_url"),
                data.get("usage_parser_type"),
                int(data.get("is_builtin_locked", False)),
                data.get("sort_order", 0),
                now,
                now,
            ),
        )
        # Init sync state
        conn.execute(
            "INSERT OR IGNORE INTO provider_sync_state (provider_id) VALUES (?)",
            (pid,),
        )
    return get_provider(pid)  # type: ignore[return-value]


def update_provider(provider_id: str, data: dict[str, Any], expected_updated_at: str | None = None) -> dict[str, Any] | None:
    """Update a provider instance. Returns updated row or None.

    If expected_updated_at is provided, uses optimistic concurrency —
    rejects the update if the current updated_at doesn't match.
    """
    allowed = {
        "kind", "provider_key", "display_name", "adapter_type", "base_url",
        "enabled", "response_format", "default_model", "models_endpoint",
        "usage_strategy", "usage_endpoint_url", "usage_parser_type",
        "is_builtin_locked", "sort_order",
    }
    updates = {k: v for k, v in data.items() if k in allowed}
    if not updates:
        return get_provider(provider_id)

    now = _now()
    with transaction() as conn:
        # Optimistic concurrency check
        if expected_updated_at:
            current = conn.execute(
                "SELECT updated_at FROM provider_instances WHERE id = ? AND deleted_at IS NULL",
                (provider_id,),
            ).fetchone()
            if not current:
                raise ValueError("Provider not found")
            if current["updated_at"] != expected_updated_at:
                raise ValueError("Conflict: provider has been modified since you loaded it")

        # Build SET clause
        set_parts = []
        values = []
        for k, v in updates.items():
            if k == "enabled":
                v = int(bool(v))
            elif k == "is_builtin_locked":
                v = int(bool(v))
            set_parts.append(f"{k} = ?")
            values.append(v)
        set_parts.append("updated_at = ?")
        values.append(now)
        values.append(provider_id)

        conn.execute(
            f"UPDATE provider_instances SET {', '.join(set_parts)} WHERE id = ? AND deleted_at IS NULL",
            values,
        )
    return get_provider(provider_id)


def delete_provider(provider_id: str) -> bool:
    """Soft-delete a provider instance. Returns True if found and deleted."""
    now = _now()
    with transaction() as conn:
        cur = conn.execute(
            "UPDATE provider_instances SET deleted_at = ?, updated_at = ? WHERE id = ? AND deleted_at IS NULL",
            (now, now, provider_id),
        )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def _simple_encrypt(plaintext: str) -> str:
    """Encrypt API key for storage.

    Phase 1: base64 encoding with a clear upgrade path to AES-GCM.
    TODO: upgrade to AES-GCM with machine-secret key before production.
    """
    import base64
    return base64.b64encode(plaintext.encode("utf-8")).decode("ascii")


def _simple_decrypt(ciphertext: str) -> str:
    """Decrypt API key from storage (Phase 1 base64)."""
    import base64
    return base64.b64decode(ciphertext.encode("ascii")).decode("utf-8")


def set_credential(provider_id: str, api_key: str, auth_type: str = "bearer") -> dict[str, Any]:
    """Set or replace the credential for a provider."""
    now = _now()
    hint = api_key[:4] + "..." + api_key[-4:] if len(api_key) >= 12 else "***"
    ciphertext = _simple_encrypt(api_key)
    with transaction() as conn:
        conn.execute(
            """INSERT INTO provider_credentials (provider_id, auth_type, api_key_ciphertext, api_key_hint, updated_at, version)
               VALUES (?, ?, ?, ?, ?, 1)
               ON CONFLICT(provider_id) DO UPDATE SET
                   auth_type = excluded.auth_type,
                   api_key_ciphertext = excluded.api_key_ciphertext,
                   api_key_hint = excluded.api_key_hint,
                   updated_at = excluded.updated_at,
                   version = version + 1""",
            (provider_id, auth_type, ciphertext, hint, now),
        )
        # Update sync state
        conn.execute(
            "UPDATE provider_sync_state SET last_credential_sync_at = ? WHERE provider_id = ?",
            (now, provider_id),
        )
    row = conn.execute(
        "SELECT * FROM provider_credentials WHERE provider_id = ?",
        (provider_id,),
    ).fetchone() if False else None  # re-open after commit
    return get_credential(provider_id) or {}


def get_credential(provider_id: str) -> dict[str, Any] | None:
    """Get credential metadata (without decrypting the key)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT provider_id, auth_type, api_key_hint, updated_at, version FROM provider_credentials WHERE provider_id = ?",
            (provider_id,),
        ).fetchone()
        return dict(row) if row else None


def get_decrypted_key(provider_id: str) -> str | None:
    """Get the decrypted API key for a provider (internal use only)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT api_key_ciphertext FROM provider_credentials WHERE provider_id = ?",
            (provider_id,),
        ).fetchone()
        if not row or not row["api_key_ciphertext"]:
            return None
        return _simple_decrypt(row["api_key_ciphertext"])


def delete_credential(provider_id: str) -> bool:
    """Delete the credential for a provider. Returns True if found."""
    now = _now()
    with transaction() as conn:
        cur = conn.execute(
            "DELETE FROM provider_credentials WHERE provider_id = ?",
            (provider_id,),
        )
        if cur.rowcount > 0:
            conn.execute(
                "UPDATE provider_sync_state SET last_credential_sync_at = ? WHERE provider_id = ?",
                (now, provider_id),
            )
            return True
        return False


# ---------------------------------------------------------------------------
# Models Cache
# ---------------------------------------------------------------------------

def refresh_models_cache(provider_id: str, models: list[dict[str, Any]]) -> None:
    """Replace the cached models for a provider."""
    now = _now()
    with transaction() as conn:
        conn.execute(
            "DELETE FROM provider_models_cache WHERE provider_id = ?",
            (provider_id,),
        )
        for m in models:
            conn.execute(
                """INSERT INTO provider_models_cache (provider_id, model_id, label, raw_json, fetched_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (provider_id, m["id"], m.get("label"), json.dumps(m.get("raw", {})), now),
            )


def get_cached_models(provider_id: str) -> list[dict[str, Any]]:
    """Get cached models for a provider."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM provider_models_cache WHERE provider_id = ? ORDER BY model_id",
            (provider_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("raw_json"):
                try:
                    d["raw"] = json.loads(d["raw_json"])
                except (json.JSONDecodeError, TypeError):
                    d["raw"] = {}
            del d["raw_json"]
            result.append(d)
        return result


# ---------------------------------------------------------------------------
# Usage Cache
# ---------------------------------------------------------------------------

def refresh_usage_cache(provider_id: str, status: str, payload: dict | None = None, message: str | None = None) -> None:
    """Replace the cached usage for a provider."""
    now = _now()
    with transaction() as conn:
        conn.execute(
            """INSERT INTO provider_usage_cache (provider_id, status, payload_json, message, fetched_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(provider_id) DO UPDATE SET
                   status = excluded.status,
                   payload_json = excluded.payload_json,
                   message = excluded.message,
                   fetched_at = excluded.fetched_at""",
            (provider_id, status, json.dumps(payload) if payload else None, message, now),
        )


def get_cached_usage(provider_id: str) -> dict[str, Any] | None:
    """Get cached usage for a provider."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM provider_usage_cache WHERE provider_id = ?",
            (provider_id,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("payload_json"):
            try:
                d["payload"] = json.loads(d["payload_json"])
            except (json.JSONDecodeError, TypeError):
                d["payload"] = {}
        del d["payload_json"]
        return d


# ---------------------------------------------------------------------------
# Sync State
# ---------------------------------------------------------------------------

def get_sync_state(provider_id: str) -> dict[str, Any] | None:
    """Get sync state for a provider."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM provider_sync_state WHERE provider_id = ?",
            (provider_id,),
        ).fetchone()
        return dict(row) if row else None


def update_sync_state(provider_id: str, **kwargs: Any) -> None:
    """Update fields on provider_sync_state."""
    allowed = {
        "last_projected_at", "last_projected_hash", "last_imported_at",
        "last_credential_sync_at", "last_error", "retry_count",
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    set_parts = [f"{k} = ?" for k in updates]
    values = list(updates.values())
    values.append(provider_id)
    with transaction() as conn:
        conn.execute(
            f"UPDATE provider_sync_state SET {', '.join(set_parts)} WHERE provider_id = ?",
            values,
        )
