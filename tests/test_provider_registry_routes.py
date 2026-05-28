"""Tests for provider registry API route integration.

Covers: route registration in api/routes.py, route handler functions,
GET/POST/PATCH/DELETE dispatch, and error handling.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import pytest


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect store._db_path() to a fresh tmp directory for every test."""
    from api.provider_registry import store

    db_dir = tmp_path / "webui"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_file = db_dir / "providers.db"

    monkeypatch.setattr(store, "_db_path", lambda: db_file)
    from api.provider_registry import services
    monkeypatch.setattr(services, "_bootstrap_ran", False)
    yield db_file


# ---------------------------------------------------------------------------
# Route registration (source-level)
# ---------------------------------------------------------------------------

def test_registry_route_registered_in_routes_py():
    """api/routes.py must reference the registry route handlers."""
    src = Path("api/routes.py").read_text(encoding="utf-8")
    assert "/api/providers/registry" in src
    assert "handle_registry_list" in src
    assert "handle_registry_create" in src


# ---------------------------------------------------------------------------
# Route handler unit tests
# ---------------------------------------------------------------------------

def test_handle_registry_list_returns_providers():
    """handle_registry_list should return a dict with providers list."""
    from api.provider_registry.routes import handle_registry_list

    result = handle_registry_list()
    assert "providers" in result
    assert "count" in result
    assert isinstance(result["providers"], list)


def test_handle_registry_list_includes_official():
    """After bootstrap, official providers should appear in the list."""
    from api.provider_registry.routes import handle_registry_list

    result = handle_registry_list()
    keys = {p["provider_key"] for p in result["providers"]}
    # Should have at least openai and anthropic from bootstrap
    assert "openai" in keys or any("openai" in k for k in keys)


def test_handle_registry_create_custom_provider():
    """Creating a custom provider via route handler should succeed."""
    from api.provider_registry.routes import handle_registry_create

    result = handle_registry_create({
        "kind": "custom",
        "provider_key": "route_test_key",
        "display_name": "Route Test",
        "adapter_type": "openai",
        "base_url": "https://route.test/v1",
    })
    assert result["provider_key"] == "route_test_key"
    assert result["adapter_type"] == "openai"


def test_handle_registry_create_validation_error():
    """Invalid create payload should raise ValueError."""
    from api.provider_registry.routes import handle_registry_create

    with pytest.raises(ValueError):
        handle_registry_create({"display_name": "Missing key"})


def test_handle_registry_get_one():
    """handle_registry_get_one should return a provider with credential info."""
    from api.provider_registry.routes import handle_registry_create, handle_registry_get_one

    created = handle_registry_create({
        "kind": "custom",
        "provider_key": "get_one_key",
        "display_name": "Get One",
        "adapter_type": "openai",
    })
    pid = created["id"]

    result = handle_registry_get_one(f"/api/providers/registry/{pid}")
    assert result is not None
    assert result["id"] == pid
    assert "credential" in result


def test_handle_registry_get_one_not_found():
    """Requesting a nonexistent provider should raise ValueError."""
    from api.provider_registry.routes import handle_registry_get_one

    # Use a valid hex ID that doesn't exist in the DB
    with pytest.raises(ValueError, match="not found"):
        handle_registry_get_one("/api/providers/registry/abcdef0123456789")


def test_handle_registry_get_one_bad_path():
    """A path that doesn't match the ID pattern should return None."""
    from api.provider_registry.routes import handle_registry_get_one

    result = handle_registry_get_one("/api/providers/registry/")
    assert result is None


def test_handle_registry_models_get():
    """handle_registry_models_get should return models dict."""
    from api.provider_registry.routes import handle_registry_create, handle_registry_models_get

    created = handle_registry_create({
        "kind": "custom",
        "provider_key": "models_route_key",
        "display_name": "Models Route",
        "adapter_type": "openai",
    })
    pid = created["id"]

    result = handle_registry_models_get(f"/api/providers/registry/{pid}/models")
    assert result is not None
    assert "models" in result


def test_handle_registry_usage_get():
    """handle_registry_usage_get should return usage dict."""
    from api.provider_registry.routes import handle_registry_create, handle_registry_usage_get

    created = handle_registry_create({
        "kind": "custom",
        "provider_key": "usage_route_key",
        "display_name": "Usage Route",
        "adapter_type": "openai",
    })
    pid = created["id"]

    result = handle_registry_usage_get(f"/api/providers/registry/{pid}/usage")
    assert result is not None


def test_handle_registry_update():
    """handle_registry_update should modify provider fields."""
    from api.provider_registry.routes import handle_registry_create, handle_registry_update

    created = handle_registry_create({
        "kind": "custom",
        "provider_key": "update_route_key",
        "display_name": "Update Route",
        "adapter_type": "openai",
    })
    pid = created["id"]

    result = handle_registry_update(f"/api/providers/registry/{pid}", {"display_name": "Updated"})
    assert result["display_name"] == "Updated"


def test_handle_registry_delete():
    """handle_registry_delete should soft-delete the provider."""
    from api.provider_registry.routes import handle_registry_create, handle_registry_delete

    created = handle_registry_create({
        "kind": "custom",
        "provider_key": "delete_route_key",
        "display_name": "Delete Route",
        "adapter_type": "openai",
    })
    pid = created["id"]

    result = handle_registry_delete(f"/api/providers/registry/{pid}")
    assert result["ok"] is True


def test_handle_registry_activate():
    """handle_registry_activate should enable and set sort_order."""
    from api.provider_registry.routes import handle_registry_create, handle_registry_activate

    created = handle_registry_create({
        "kind": "custom",
        "provider_key": "activate_route_key",
        "display_name": "Activate Route",
        "adapter_type": "openai",
    })
    pid = created["id"]

    result = handle_registry_activate(f"/api/providers/registry/{pid}/activate")
    assert result["ok"] is True


def test_handle_registry_credential_put():
    """handle_registry_credential_put should set API key."""
    from api.provider_registry.routes import handle_registry_create, handle_registry_credential_put

    created = handle_registry_create({
        "kind": "custom",
        "provider_key": "cred_put_key",
        "display_name": "Cred Put",
        "adapter_type": "openai",
    })
    pid = created["id"]

    result = handle_registry_credential_put(
        f"/api/providers/registry/{pid}/credential",
        {"api_key": "sk-put...test"},
    )
    assert result["provider_id"] == pid


def test_handle_registry_credential_delete():
    """handle_registry_credential_delete should remove the credential."""
    from api.provider_registry.routes import (
        handle_registry_create,
        handle_registry_credential_put,
        handle_registry_credential_delete,
    )

    created = handle_registry_create({
        "kind": "custom",
        "provider_key": "cred_del_key",
        "display_name": "Cred Del",
        "adapter_type": "openai",
    })
    pid = created["id"]

    handle_registry_credential_put(
        f"/api/providers/registry/{pid}/credential",
        {"api_key": "sk-del...test"},
    )
    result = handle_registry_credential_delete(f"/api/providers/registry/{pid}/credential")
    assert result["ok"] is True


def test_id_regex_matches_hex_ids():
    """Route ID regex should match 16-char hex IDs."""
    from api.provider_registry.routes import _REGISTRY_ID_RE

    assert _REGISTRY_ID_RE.match("/api/providers/registry/abc123def4567890")
    assert not _REGISTRY_ID_RE.match("/api/providers/registry/not-hex!")
    assert not _REGISTRY_ID_RE.match("/api/providers/registry/")
