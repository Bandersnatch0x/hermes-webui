"""Route handler functions for the provider registry API.

These functions are called from api/routes.py to handle registry endpoints.
Each returns a dict with {"_status": int, "_data": dict} for the caller to
serialize, or raises to let the caller handle errors.
"""

from __future__ import annotations

import json
import re
from typing import Any

from api.provider_registry import services

# Regex patterns for :id extraction
_REGISTRY_ID_RE = re.compile(r"^/api/providers/registry/([a-f0-9]+)$")
_REGISTRY_ID_MODELS_RE = re.compile(r"^/api/providers/registry/([a-f0-9]+)/models$")
_REGISTRY_ID_MODELS_REFRESH_RE = re.compile(r"^/api/providers/registry/([a-f0-9]+)/models/refresh$")
_REGISTRY_ID_USAGE_RE = re.compile(r"^/api/providers/registry/([a-f0-9]+)/usage$")
_REGISTRY_ID_USAGE_REFRESH_RE = re.compile(r"^/api/providers/registry/([a-f0-9]+)/usage/refresh$")
_REGISTRY_ID_ACTIVATE_RE = re.compile(r"^/api/providers/registry/([a-f0-9]+)/activate$")
_REGISTRY_ID_CREDENTIAL_RE = re.compile(r"^/api/providers/registry/([a-f0-9]+)/credential$")


def _extract_id(path: str, pattern: re.Pattern) -> str | None:
    """Extract provider id from path using pattern."""
    m = pattern.match(path)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# GET handlers
# ---------------------------------------------------------------------------

def handle_registry_list() -> dict[str, Any]:
    """GET /api/providers/registry — list all providers."""
    providers = services.list_providers(include_disabled=True)
    return {"providers": providers, "count": len(providers)}


def handle_registry_get_one(path: str) -> dict[str, Any] | None:
    """GET /api/providers/registry/:id — get single provider."""
    pid = _extract_id(path, _REGISTRY_ID_RE)
    if not pid:
        return None
    provider = services.get_provider(pid)
    if not provider:
        raise ValueError("Provider not found")
    # Include credential hint
    from api.provider_registry import store
    store.init_db()
    cred = store.get_credential(pid)
    provider["credential"] = cred
    return provider


def handle_registry_models_get(path: str) -> dict[str, Any] | None:
    """GET /api/providers/registry/:id/models — get cached models."""
    pid = _extract_id(path, _REGISTRY_ID_MODELS_RE)
    if not pid:
        return None
    return services.get_models(pid)


def handle_registry_usage_get(path: str) -> dict[str, Any] | None:
    """GET /api/providers/registry/:id/usage — get cached usage."""
    pid = _extract_id(path, _REGISTRY_ID_USAGE_RE)
    if not pid:
        return None
    return services.get_usage(pid)


# ---------------------------------------------------------------------------
# POST handlers
# ---------------------------------------------------------------------------

def handle_registry_create(body: dict[str, Any]) -> dict[str, Any]:
    """POST /api/providers/registry — create a new provider."""
    return services.create_provider(body)


def handle_registry_models_refresh(path: str) -> dict[str, Any] | None:
    """POST /api/providers/registry/:id/models/refresh — refresh models."""
    pid = _extract_id(path, _REGISTRY_ID_MODELS_REFRESH_RE)
    if not pid:
        return None
    return services.refresh_models(pid)


def handle_registry_usage_refresh(path: str) -> dict[str, Any] | None:
    """POST /api/providers/registry/:id/usage/refresh — refresh usage."""
    pid = _extract_id(path, _REGISTRY_ID_USAGE_REFRESH_RE)
    if not pid:
        return None
    return services.refresh_usage(pid)


def handle_registry_activate(path: str) -> dict[str, Any] | None:
    """POST /api/providers/registry/:id/activate — activate a provider."""
    pid = _extract_id(path, _REGISTRY_ID_ACTIVATE_RE)
    if not pid:
        return None
    return services.activate_provider(pid)


def handle_reconcile() -> dict[str, Any]:
    """POST /api/providers/reconcile — reconcile DB with live config."""
    return services.reconcile()


# ---------------------------------------------------------------------------
# PATCH handler
# ---------------------------------------------------------------------------

def handle_registry_update(path: str, body: dict[str, Any]) -> dict[str, Any] | None:
    """PATCH /api/providers/registry/:id — update a provider."""
    pid = _extract_id(path, _REGISTRY_ID_RE)
    if not pid:
        return None
    return services.update_provider(pid, body)


# ---------------------------------------------------------------------------
# DELETE handlers
# ---------------------------------------------------------------------------

def handle_registry_delete(path: str) -> dict[str, Any] | None:
    """DELETE /api/providers/registry/:id — soft-delete a provider."""
    pid = _extract_id(path, _REGISTRY_ID_RE)
    if not pid:
        return None
    ok = services.delete_provider(pid)
    if not ok:
        raise ValueError("Provider not found")
    return {"ok": True}


def handle_registry_credential_delete(path: str) -> dict[str, Any] | None:
    """DELETE /api/providers/registry/:id/credential — delete API key."""
    pid = _extract_id(path, _REGISTRY_ID_CREDENTIAL_RE)
    if not pid:
        return None
    ok = services.delete_credential(pid)
    return {"ok": ok}


# ---------------------------------------------------------------------------
# PUT handler
# ---------------------------------------------------------------------------

def handle_registry_credential_put(path: str, body: dict[str, Any]) -> dict[str, Any] | None:
    """PUT /api/providers/registry/:id/credential — set API key."""
    pid = _extract_id(path, _REGISTRY_ID_CREDENTIAL_RE)
    if not pid:
        return None
    api_key = body.get("api_key", "")
    return services.set_credential(pid, api_key)
