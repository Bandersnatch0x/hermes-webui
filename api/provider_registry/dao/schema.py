"""SQLite schema for the provider registry database.

Defines all tables and the initialize_database() entry point.
Schema version is tracked via schema_migrations.
"""
from __future__ import annotations

import logging

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_DDL = [
    # --- provider_instances ---
    """CREATE TABLE IF NOT EXISTS provider_instances (
        id              TEXT PRIMARY KEY,
        kind            TEXT NOT NULL,
        provider_key    TEXT NOT NULL,
        display_name    TEXT NOT NULL,
        adapter_type    TEXT NOT NULL,
        base_url        TEXT,
        enabled         INTEGER NOT NULL DEFAULT 1,
        response_format TEXT,
        default_model   TEXT,
        models_endpoint TEXT,
        usage_strategy  TEXT NOT NULL DEFAULT 'auto',
        usage_endpoint_url  TEXT,
        usage_parser_type   TEXT,
        is_builtin_locked   INTEGER NOT NULL DEFAULT 0,
        sort_order      INTEGER NOT NULL DEFAULT 0,
        created_at      TEXT NOT NULL,
        updated_at      TEXT NOT NULL,
        deleted_at      TEXT,
        UNIQUE(provider_key),
        CHECK(kind IN ('official', 'custom')),
        CHECK(adapter_type IN ('openai', 'anthropic')),
        CHECK(enabled IN (0, 1)),
        CHECK(is_builtin_locked IN (0, 1)),
        CHECK(usage_strategy IN ('auto', 'endpoint', 'auto+endpoint', 'none')),
        CHECK(
            (adapter_type = 'openai' AND response_format IN ('completions', 'messages', 'responses'))
            OR (adapter_type = 'anthropic' AND response_format IS NULL)
        )
    )""",

    # --- provider_credentials ---
    """CREATE TABLE IF NOT EXISTS provider_credentials (
        provider_id         TEXT PRIMARY KEY,
        auth_type           TEXT NOT NULL,
        api_key_ciphertext  TEXT,
        api_key_hint        TEXT,
        updated_at          TEXT NOT NULL,
        version             INTEGER NOT NULL DEFAULT 1
    )""",

    # --- provider_models_cache ---
    """CREATE TABLE IF NOT EXISTS provider_models_cache (
        provider_id TEXT NOT NULL,
        model_id    TEXT NOT NULL,
        label       TEXT,
        raw_json    TEXT,
        fetched_at  TEXT NOT NULL,
        PRIMARY KEY (provider_id, model_id)
    )""",

    # --- provider_usage_cache ---
    """CREATE TABLE IF NOT EXISTS provider_usage_cache (
        provider_id TEXT PRIMARY KEY,
        status      TEXT NOT NULL,
        payload_json TEXT,
        message     TEXT,
        fetched_at  TEXT NOT NULL,
        CHECK(status IN ('available', 'unsupported', 'error', 'unknown'))
    )""",

    # --- provider_sync_state ---
    """CREATE TABLE IF NOT EXISTS provider_sync_state (
        provider_id             TEXT PRIMARY KEY,
        last_projected_at       TEXT,
        last_projected_hash     TEXT,
        last_imported_at        TEXT,
        last_credential_sync_at TEXT,
        last_error              TEXT,
        retry_count             INTEGER NOT NULL DEFAULT 0
    )""",

    # --- schema_migrations ---
    """CREATE TABLE IF NOT EXISTS schema_migrations (
        version     INTEGER PRIMARY KEY,
        applied_at  TEXT NOT NULL,
        description TEXT
    )""",
]

# Indexes for common query patterns
_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_provider_instances_kind ON provider_instances(kind)",
    "CREATE INDEX IF NOT EXISTS idx_provider_instances_deleted ON provider_instances(deleted_at)",
    "CREATE INDEX IF NOT EXISTS idx_provider_instances_sort ON provider_instances(sort_order, display_name COLLATE NOCASE, id)",
    "CREATE INDEX IF NOT EXISTS idx_provider_models_cache_provider ON provider_models_cache(provider_id)",
]


async def initialize_database(db: aiosqlite.Connection) -> None:
    """Create all tables and indexes. Idempotent."""
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA busy_timeout=5000")
    await db.execute("PRAGMA foreign_keys=ON")

    for ddl in _DDL:
        await db.execute(ddl)
    for idx in _INDEXES:
        await db.execute(idx)

    # Record schema version if not already present
    await db.execute(
        "INSERT OR IGNORE INTO schema_migrations (version, applied_at, description) "
        "VALUES (?, datetime('now'), ?)",
        (SCHEMA_VERSION, "initial schema"),
    )
    await db.commit()
    logger.info("provider_registry schema v%d initialized", SCHEMA_VERSION)
