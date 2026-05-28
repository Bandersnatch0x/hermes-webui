"""Tests for DB schema bootstrap, table creation, and constraints in the provider registry.

These tests target the WORKING synchronous sqlite3 implementation in
``api/provider_registry/store.py``.  Every test uses ``tmp_path`` +
``monkeypatch`` to isolate from the real DB.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect store._db_path() to a fresh tmp directory for every test."""
    from api.provider_registry import store

    db_dir = tmp_path / "webui"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_file = db_dir / "providers.db"

    monkeypatch.setattr(store, "_db_path", lambda: db_file)
    # Also clear any cached module-level state if present
    yield db_file


def _table_names(db_path: Path) -> set[str]:
    """Return the set of user-created table names in the DB."""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 1. test_bootstrap_creates_expected_tables
# ---------------------------------------------------------------------------

def test_bootstrap_creates_expected_tables(_isolate_db: Path):
    """After init_db(), all expected tables should exist."""
    from api.provider_registry.store import init_db

    init_db()

    expected = {
        "provider_instances",
        "provider_credentials",
        "provider_models_cache",
        "provider_usage_cache",
        "provider_sync_state",
        "schema_migrations",
    }
    actual = _table_names(_isolate_db)
    missing = expected - actual
    assert not missing, f"Missing tables after bootstrap: {missing}"
    # Also verify no unexpected tables appeared
    assert expected == actual, f"Unexpected extra tables: {actual - expected}"


# ---------------------------------------------------------------------------
# 2. test_provider_key_is_unique_within_profile_db
# ---------------------------------------------------------------------------

def test_provider_key_is_unique_within_profile_db(_isolate_db: Path):
    """INSERTing a duplicate provider_key must raise a UNIQUE constraint error."""
    from api.provider_registry.store import init_db, create_provider

    init_db()

    base = {
        "provider_key": "openai-1",
        "display_name": "OpenAI",
        "adapter_type": "openai",
    }
    create_provider(base)

    # Second insert with the same provider_key must fail
    with pytest.raises(sqlite3.IntegrityError):
        create_provider({
            "provider_key": "openai-1",  # duplicate
            "display_name": "OpenAI Copy",
            "adapter_type": "openai",
        })


# ---------------------------------------------------------------------------
# 3. test_schema_has_wal_mode
# ---------------------------------------------------------------------------

def test_schema_has_wal_mode(_isolate_db: Path):
    """The connection should use WAL journal mode."""
    from api.provider_registry.store import init_db, _connect

    init_db()

    conn = _connect()
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal", f"Expected WAL journal mode, got: {mode}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 4. test_bootstrap_is_idempotent
# ---------------------------------------------------------------------------

def test_bootstrap_is_idempotent(_isolate_db: Path):
    """Calling init_db() twice must not fail or duplicate schema_migrations rows."""
    from api.provider_registry.store import init_db

    init_db()
    init_db()  # second call should be a no-op

    conn = sqlite3.connect(str(_isolate_db))
    try:
        migrations = conn.execute("SELECT * FROM schema_migrations").fetchall()
        assert len(migrations) == 1, (
            f"Expected 1 schema_migrations row, got {len(migrations)}"
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 5. test_provider_instances_required_fields
# ---------------------------------------------------------------------------

def test_provider_instances_required_fields(_isolate_db: Path):
    """Inserting without required fields must fail with an IntegrityError."""
    from api.provider_registry.store import init_db, transaction, _now

    init_db()

    # Missing provider_key, display_name, adapter_type (all NOT NULL)
    with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError)):
        with transaction() as conn:
            conn.execute(
                """INSERT INTO provider_instances
                   (id, kind, created_at, updated_at)
                   VALUES (?, ?, ?, ?)""",
                ("test-id", "custom", _now(), _now()),
            )
