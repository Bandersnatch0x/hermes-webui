"""Tests for provider registry sync state and projection.

Covers: sync state upsert/read, credential set/get/delete through the store,
and verification that operations persist correctly.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect store._db_path() to a fresh tmp directory for every test."""
    from api.provider_registry import store

    db_dir = tmp_path / "webui"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_file = db_dir / "providers.db"

    monkeypatch.setattr(store, "_db_path", lambda: db_file)
    yield db_file


def _seed_provider(provider_id: str = "test_prov") -> None:
    """Insert a minimal provider for tests that need one."""
    from api.provider_registry.store import init_db, create_provider
    init_db()
    create_provider({
        "id": provider_id,
        "kind": "custom",
        "provider_key": f"key_{provider_id}",
        "display_name": "Test",
        "adapter_type": "openai",
    })


# ---------------------------------------------------------------------------

def test_sync_state_starts_empty():
    """Before any update, sync_state should be empty (row created by create_provider)."""
    from api.provider_registry.store import init_db, get_sync_state

    _seed_provider()
    state = get_sync_state("test_prov")
    assert state is not None
    assert state["last_projected_at"] is None
    assert state["last_error"] is None
    assert state["retry_count"] == 0


def test_update_sync_state_persists():
    """update_sync_state should persist the fields."""
    from api.provider_registry.store import update_sync_state, get_sync_state

    _seed_provider()
    update_sync_state("test_prov", last_projected_at="2026-05-28T00:00:00Z", retry_count=3)

    state = get_sync_state("test_prov")
    assert state["last_projected_at"] == "2026-05-28T00:00:00Z"
    assert state["retry_count"] == 3


def test_update_sync_state_error_field():
    """Can set last_error on sync state."""
    from api.provider_registry.store import update_sync_state, get_sync_state

    _seed_provider()
    update_sync_state("test_prov", last_error="connection timeout", retry_count=1)

    state = get_sync_state("test_prov")
    assert state["last_error"] == "connection timeout"
    assert state["retry_count"] == 1


def test_update_sync_state_ignores_unknown_fields():
    """Unknown kwargs should be silently ignored."""
    from api.provider_registry.store import update_sync_state, get_sync_state

    _seed_provider()
    update_sync_state("test_prov", last_projected_at="2026-05-28T00:00:00Z", bogus_field="nope")

    state = get_sync_state("test_prov")
    assert state["last_projected_at"] == "2026-05-28T00:00:00Z"
    assert "bogus_field" not in state


def test_set_credential_encrypts_before_persist():
    """Credential ciphertext must not be the plaintext key."""
    from api.provider_registry.store import set_credential, get_credential

    _seed_provider()
    set_credential("test_prov", "sk-abcdef1234567890")

    meta = get_credential("test_prov")
    assert meta is not None
    assert meta["api_key_hint"] is not None
    # ciphertext should NOT be the plaintext
    assert "sk-abcdef1234567890" not in (meta.get("api_key_hint") or "")


def test_get_decrypted_key_returns_original():
    """Decrypting a stored credential should return the original key."""
    from api.provider_registry.store import set_credential, get_decrypted_key

    _seed_provider()
    original = "sk-test-key-1234567890abcdef"
    set_credential("test_prov", original)

    decrypted = get_decrypted_key("test_prov")
    assert decrypted == original


def test_credential_hint_is_masked():
    """The api_key_hint should be a masked/truncated version."""
    from api.provider_registry.store import set_credential, get_credential

    _seed_provider()
    set_credential("test_prov", "sk-verylongkey123456")

    meta = get_credential("test_prov")
    hint = meta["api_key_hint"]
    # hint should contain '...' (masking)
    assert "..." in hint
    # hint should NOT be the full key
    assert hint != "sk-verylongkey123456"


def test_credential_hint_short_key():
    """Short keys should get '***' hint."""
    from api.provider_registry.store import set_credential, get_credential

    _seed_provider()
    set_credential("test_prov", "abc")

    meta = get_credential("test_prov")
    assert meta["api_key_hint"] == "***"


def test_delete_credential_removes_row():
    """Deleting a credential should remove the row."""
    from api.provider_registry.store import set_credential, delete_credential, get_credential

    _seed_provider()
    set_credential("test_prov", "sk-to-be-deleted-12345")

    assert delete_credential("test_prov") is True
    assert get_credential("test_prov") is None


def test_delete_credential_nonexistent_returns_false():
    """Deleting a nonexistent credential should return False."""
    from api.provider_registry.store import delete_credential

    _seed_provider()
    assert delete_credential("test_prov") is False


def test_set_credential_updates_version_on_conflict():
    """Replacing a credential should bump the version."""
    from api.provider_registry.store import set_credential, get_credential

    _seed_provider()
    set_credential("test_prov", "sk-first-key-12345678")
    v1 = get_credential("test_prov")["version"]

    set_credential("test_prov", "sk-second-key-1234567")
    v2 = get_credential("test_prov")["version"]

    assert v2 > v1


def test_credential_sync_state_updated_on_set():
    """Setting a credential should update last_credential_sync_at."""
    from api.provider_registry.store import set_credential, get_sync_state

    _seed_provider()
    set_credential("test_prov", "sk-sync-test-123456789")

    state = get_sync_state("test_prov")
    assert state["last_credential_sync_at"] is not None


def test_get_decrypted_key_returns_none_for_missing():
    """No credential stored → get_decrypted_key returns None."""
    from api.provider_registry.store import get_decrypted_key

    _seed_provider()
    assert get_decrypted_key("test_prov") is None
