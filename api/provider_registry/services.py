"""Service layer for the provider registry.

Orchestrates store operations, adapter calls, and validation.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from api.provider_registry import store
from api.provider_registry.adapters.resolution import resolve_adapter

logger = logging.getLogger(__name__)

# Bootstrap guard — run once per process lifetime.
_bootstrap_ran = False


def _ensure_bootstrapped() -> None:
    """Trigger bootstrap import on first access if not already done."""
    global _bootstrap_ran
    if _bootstrap_ran:
        return
    _bootstrap_ran = True
    try:
        from api.provider_registry.bootstrap import bootstrap_on_startup
        bootstrap_on_startup()
    except Exception as exc:
        logger.warning("Provider registry bootstrap failed: %s", exc)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALID_KINDS = {"official", "custom"}
_VALID_ADAPTER_TYPES = {"openai", "anthropic"}
_VALID_RESPONSE_FORMATS = {"completions", "messages", "responses"}
_VALID_USAGE_STRATEGIES = {"auto", "endpoint", "auto+endpoint", "none"}


def _validate_create(data: dict[str, Any]) -> None:
    """Raise ValueError on invalid create payload."""
    if not data.get("provider_key"):
        raise ValueError("provider_key is required")
    if not data.get("display_name"):
        raise ValueError("display_name is required")
    if not data.get("adapter_type"):
        raise ValueError("adapter_type is required")
    kind = data.get("kind", "custom")
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of: {', '.join(sorted(_VALID_KINDS))}")
    at = data["adapter_type"]
    if at not in _VALID_ADAPTER_TYPES:
        raise ValueError(f"adapter_type must be one of: {', '.join(sorted(_VALID_ADAPTER_TYPES))}")
    rf = data.get("response_format")
    if at == "openai":
        if rf and rf not in _VALID_RESPONSE_FORMATS:
            raise ValueError(f"response_format must be one of: {', '.join(sorted(_VALID_RESPONSE_FORMATS))}")
    elif at == "anthropic":
        if rf is not None:
            raise ValueError("response_format must be null for anthropic adapter")
    us = data.get("usage_strategy", "auto")
    if us not in _VALID_USAGE_STRATEGIES:
        raise ValueError(f"usage_strategy must be one of: {', '.join(sorted(_VALID_USAGE_STRATEGIES))}")


def _validate_update(data: dict[str, Any]) -> None:
    """Raise ValueError on invalid update payload."""
    if "kind" in data and data["kind"] not in _VALID_KINDS:
        raise ValueError(f"kind must be one of: {', '.join(sorted(_VALID_KINDS))}")
    if "adapter_type" in data and data["adapter_type"] not in _VALID_ADAPTER_TYPES:
        raise ValueError(f"adapter_type must be one of: {', '.join(sorted(_VALID_ADAPTER_TYPES))}")
    if "usage_strategy" in data and data["usage_strategy"] not in _VALID_USAGE_STRATEGIES:
        raise ValueError(f"usage_strategy must be one of: {', '.join(sorted(_VALID_USAGE_STRATEGIES))}")


# ---------------------------------------------------------------------------
# Registry CRUD
# ---------------------------------------------------------------------------

def list_providers(include_disabled: bool = False) -> list[dict[str, Any]]:
    """List all non-deleted providers."""
    store.init_db()
    _ensure_bootstrapped()
    return store.list_providers(include_disabled=include_disabled)


def get_provider(provider_id: str) -> dict[str, Any] | None:
    """Get a provider by id."""
    store.init_db()
    return store.get_provider(provider_id)


def create_provider(data: dict[str, Any]) -> dict[str, Any]:
    """Create a new provider instance."""
    store.init_db()
    _validate_create(data)
    # Default response_format for openai if not specified
    if data.get("adapter_type") == "openai" and not data.get("response_format"):
        data["response_format"] = "completions"
    return store.create_provider(data)


def update_provider(provider_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """Update a provider instance with optimistic concurrency."""
    store.init_db()
    _validate_update(data)
    expected_ts = data.pop("_expected_updated_at", None)
    result = store.update_provider(provider_id, data, expected_updated_at=expected_ts)
    if result is None:
        raise ValueError("Provider not found")
    return result


def delete_provider(provider_id: str) -> bool:
    """Soft-delete a provider."""
    store.init_db()
    provider = store.get_provider(provider_id)
    if not provider:
        return False
    if provider.get("is_builtin_locked"):
        raise ValueError("Cannot delete a built-in locked provider")
    return store.delete_provider(provider_id)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def set_credential(provider_id: str, api_key: str) -> dict[str, Any]:
    """Set or replace the API key for a provider."""
    store.init_db()
    provider = store.get_provider(provider_id)
    if not provider:
        raise ValueError("Provider not found")
    if not api_key:
        raise ValueError("api_key is required")
    return store.set_credential(provider_id, api_key)


def delete_credential(provider_id: str) -> bool:
    """Delete the credential for a provider."""
    store.init_db()
    provider = store.get_provider(provider_id)
    if not provider:
        raise ValueError("Provider not found")
    return store.delete_credential(provider_id)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def refresh_models(provider_id: str) -> dict[str, Any]:
    """Fetch models from the provider API and cache them."""
    store.init_db()
    provider = store.get_provider(provider_id)
    if not provider:
        raise ValueError("Provider not found")

    adapter = resolve_adapter(
        provider["adapter_type"],
        response_format=provider.get("response_format"),
    )
    models_req = adapter.build_models_request()
    if models_req is None:
        # Adapter doesn't support models endpoint
        return {"status": "unsupported", "models": [], "message": "Adapter has no models endpoint"}

    api_key = store.get_decrypted_key(provider_id)
    base_url = provider.get("base_url") or ""

    try:
        url = base_url.rstrip("/") + "/" + models_req.path.lstrip("/")
        if models_req.query:
            from urllib.parse import urlencode
            url += "?" + urlencode(models_req.query)
        req = urllib.request.Request(url, method=models_req.method)
        for k, v in models_req.headers.items():
            req.add_header(k, v)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")

        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_data = json.loads(resp.read())

        models = adapter.parse_models_response(raw_data)
        model_dicts = [
            {"id": m.id, "label": m.label, "raw": m.raw}
            for m in models
        ]
        store.refresh_models_cache(provider_id, model_dicts)
        return {"status": "ok", "models": model_dicts, "count": len(model_dicts)}
    except Exception as exc:
        logger.warning("refresh_models failed for %s: %s", provider_id, exc)
        return {"status": "error", "models": [], "message": str(exc)}


def get_models(provider_id: str) -> dict[str, Any]:
    """Get cached models for a provider."""
    store.init_db()
    provider = store.get_provider(provider_id)
    if not provider:
        raise ValueError("Provider not found")
    models = store.get_cached_models(provider_id)
    return {"models": models, "count": len(models)}


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

def refresh_usage(provider_id: str) -> dict[str, Any]:
    """Probe provider usage and cache the result."""
    store.init_db()
    provider = store.get_provider(provider_id)
    if not provider:
        raise ValueError("Provider not found")

    strategy = provider.get("usage_strategy", "auto")
    if strategy == "none":
        return {"status": "unsupported", "message": "Usage probing disabled"}

    adapter = resolve_adapter(
        provider["adapter_type"],
        response_format=provider.get("response_format"),
    )

    api_key = store.get_decrypted_key(provider_id)
    base_url = provider.get("base_url") or ""
    result_status = "unknown"
    result_payload = None
    result_message = None

    # Try adapter-native probe
    if strategy in ("auto", "auto+endpoint"):
        if api_key and adapter.native_usage_probe_supported():
            try:
                usage = adapter.probe_usage(api_key, base_url)
                if usage:
                    result_status = usage.status
                    result_payload = usage.payload
                    result_message = usage.message
            except Exception as exc:
                logger.warning("native usage probe failed for %s: %s", provider_id, exc)
                result_status = "error"
                result_message = str(exc)

    # Try custom endpoint
    if strategy in ("endpoint", "auto+endpoint") and provider.get("usage_endpoint_url"):
        try:
            url = provider["usage_endpoint_url"]
            req = urllib.request.Request(url)
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read())
            result_status = "available"
            result_payload = raw
        except Exception as exc:
            if result_status not in ("available",):  # don't overwrite good result
                result_status = "error"
                result_message = str(exc)

    store.refresh_usage_cache(provider_id, result_status, result_payload, result_message)
    return {
        "status": result_status,
        "payload": result_payload,
        "message": result_message,
    }


def get_usage(provider_id: str) -> dict[str, Any]:
    """Get cached usage for a provider."""
    store.init_db()
    provider = store.get_provider(provider_id)
    if not provider:
        raise ValueError("Provider not found")
    usage = store.get_cached_usage(provider_id)
    return usage or {"status": "unknown", "message": "No cached usage"}


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------

def activate_provider(provider_id: str) -> dict[str, Any]:
    """Set a provider as the active/default provider.

    Enables the provider and updates sort_order to make it first.
    """
    store.init_db()
    provider = store.get_provider(provider_id)
    if not provider:
        raise ValueError("Provider not found")
    # Enable if disabled and set sort_order to 0 (first)
    store.update_provider(provider_id, {"enabled": True, "sort_order": 0})
    store.update_sync_state(provider_id, last_projected_at=store._now())
    return {"ok": True, "provider_id": provider_id}


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------

def reconcile() -> dict[str, Any]:
    """Import live config drift into the registry DB.

    Phase 1: imports custom_providers[] from config.yaml into the registry.
    """
    store.init_db()
    from api.config import get_config
    config = get_config()
    custom = config.get("custom_providers", []) or []
    imported = 0
    skipped = 0
    errors = []

    for cp in custom:
        name = cp.get("name") or cp.get("provider") or ""
        if not name:
            skipped += 1
            continue
        provider_key = f"custom:{name.lower().replace(' ', '-')}"
        # Check if already exists
        existing = None
        for p in store.list_providers(include_disabled=True):
            if p["provider_key"] == provider_key:
                existing = p
                break
        if existing:
            skipped += 1
            continue
        try:
            store.create_provider({
                "kind": "custom",
                "provider_key": provider_key,
                "display_name": name,
                "adapter_type": cp.get("adapter_type", "openai"),
                "base_url": cp.get("base_url"),
                "response_format": cp.get("response_format"),
                "default_model": cp.get("default_model"),
            })
            imported += 1
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    return {
        "ok": True,
        "imported": imported,
        "skipped": skipped,
        "errors": errors,
    }
