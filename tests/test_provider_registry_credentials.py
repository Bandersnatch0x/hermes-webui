"""Tests for provider registry credential encryption and service-level operations.

Covers: service.set_credential, service.delete_credential, encryption roundtrip,
hint masking, and credential lifecycle through the service layer.
"""
from __future__ import annotations

import base64
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
    # Clear bootstrap guard so each test starts fresh
    from api.provider_registry import services
    monkeypatch.setattr(services, "_bootstrap_ran", False)
    yield db_file


def _create_test_provider() -> dict:
    """Create a test provider via the service layer and return it."""
    from api.provider_registry import services
    return services.create_provider({
        "kind": "custom",
        "provider_key": "cred_test_key",
        "display_name": "Cred Test",
        "adapter_type": "openai",
        "base_url": "https://test.example/v1",
    })


# ---------------------------------------------------------------------------

def test_set_credential_encrypts_before_persist():
    """Ciphertext stored in DB must not be the raw plaintext."""
    from api.provider_registry import store

    prov = _create_test_provider()
    store.set_credential(prov["id"], "sk-secret-api-key-12345")

    # Read raw ciphertext from DB
    from api.provider_registry.store import _connect
    with _connect() as conn:
        row = conn.execute(
            "SELECT api_key_ciphertext FROM provider_credentials WHERE provider_id = ?",
            (prov["id"],),
        ).fetchone()
    assert row is not None
    ciphertext = row["api_key_ciphertext"]
    assert ciphertext != "sk-secret-api-key-12345"
    # It should be base64 (phase 1)
    decoded = base64.b64decode(ciphertext.encode("ascii")).decode("utf-8")
    assert decoded == "sk-secret-api-key-12345"


def test_get_decrypted_key_roundtrip():
    """Encrypt → store → decrypt should return original key."""
    from api.provider_registry.store import set_credential, get_decrypted_key

    prov = _create_test_provider()
    key = "sk-round...trip"
    set_credential(prov["id"], key)

    assert get_decrypted_key(prov["id"]) == key


def test_hint_contains_dots_for_long_keys():
    """Long keys should produce a hint with '...' separator."""
    from api.provider_registry.store import set_credential, get_credential

    prov = _create_test_provider()
    set_credential(prov["id"], "sk-1234567890abcdef")

    meta = get_credential(prov["id"])
    assert "..." in meta["api_key_hint"]


def test_hint_is_triple_star_for_short_keys():
    """Keys shorter than 12 chars get '***' hint."""
    from api.provider_registry.store import set_credential, get_credential

    prov = _create_test_provider()
    set_credential(prov["id"], "short")

    meta = get_credential(prov["id"])
    assert meta["api_key_hint"] == "***"


def test_service_set_credential():
    """services.set_credential should work end-to-end."""
    from api.provider_registry import services

    prov = _create_test_provider()
    result = services.set_credential(prov["id"], "sk-serv...test")

    assert result["provider_id"] == prov["id"]
    assert result["api_key_hint"] is not None


def test_service_set_credential_missing_provider():
    """Setting credential on nonexistent provider should raise."""
    from api.provider_registry import services

    with pytest.raises(ValueError, match="Provider not found"):
        services.set_credential("nonexistent_id", "sk-xxx")


def test_service_set_credential_empty_key():
    """Empty api_key should raise ValueError."""
    from api.provider_registry import services

    prov = _create_test_provider()
    with pytest.raises(ValueError, match="api_key"):
        services.set_credential(prov["id"], "")


def test_service_delete_credential():
    """services.delete_credential should remove the credential."""
    from api.provider_registry import services

    prov = _create_test_provider()
    services.set_credential(prov["id"], "sk-del...test")

    result = services.delete_credential(prov["id"])
    assert result is True


def test_service_delete_credential_missing_provider():
    """Deleting credential on nonexistent provider should raise."""
    from api.provider_registry import services

    with pytest.raises(ValueError, match="Provider not found"):
        services.delete_credential("nonexistent_id")


def test_credential_overwrite_bumps_version():
    """Overwriting a credential should increment version."""
    from api.provider_registry.store import set_credential, get_credential

    prov = _create_test_provider()
    set_credential(prov["id"], "sk-ver1...abcd")
    v1 = get_credential(prov["id"])["version"]

    set_credential(prov["id"], "sk-ver2...efgh")
    v2 = get_credential(prov["id"])["version"]

    assert v2 > v1
