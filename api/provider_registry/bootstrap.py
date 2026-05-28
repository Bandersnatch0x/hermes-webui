"""Bootstrap import for the provider registry.

On first run when the registry is empty, seeds official provider instances
and imports custom_providers[] from config.yaml.  Tracks completion via
schema_migrations (version 2 = bootstrap_complete).

Phase 1.6 of the provider registry design spec.
"""

from __future__ import annotations

import logging
from typing import Any

from api.provider_registry import store

logger = logging.getLogger(__name__)

# Migration version that marks bootstrap as complete.
_BOOTSTRAP_MIGRATION_VERSION = 2


def _is_bootstrap_done(conn) -> bool:
    """Check whether the bootstrap migration has already been applied."""
    try:
        row = conn.execute(
            "SELECT 1 FROM schema_migrations WHERE version = ?",
            (_BOOTSTRAP_MIGRATION_VERSION,),
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _mark_bootstrap_done(conn) -> None:
    """Record that bootstrap completed successfully."""
    from api.provider_registry.store import _now
    conn.execute(
        "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
        (_BOOTSTRAP_MIGRATION_VERSION, _now()),
    )


# ---------------------------------------------------------------------------
# Official provider seeding
# ---------------------------------------------------------------------------

# Adapter type mapping per official provider slug.
_ANTHROPIC_FAMILY = {"anthropic"}
# All others default to 'openai' adapter type.

# Known base URLs for official providers that use non-default endpoints.
_OFFICIAL_BASE_URLS: dict[str, str] = {
    "nous": "https://api.nousresearch.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai-codex": "https://api.openai.com/v1",
    "xai-oauth": "https://api.x.ai/v1",
    "zai": "https://open.bigmodel.cn/api/paas/v4",
    "minimax": "https://api.minimax.chat/v1",
    "minimax-cn": "https://api.minimax.chat/v1",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "opencode-zen": "https://api.opencode.ai/v1",
    "opencode-go": "https://go.opencode.ai/v1",
    "ollama-cloud": "https://api.ollama.com/v1",
}

# Response format overrides for official providers that need non-default format.
_OFFICIAL_RESPONSE_FORMATS: dict[str, str] = {
    "openai-codex": "responses",
}


def _seed_official_providers() -> int:
    """Seed official provider instances from _PROVIDER_DISPLAY / _PROVIDER_MODELS.

    Returns the number of providers created.
    """
    from api.config import _PROVIDER_DISPLAY, _PROVIDER_MODELS

    created = 0
    for sort_idx, (slug, display_name) in enumerate(_PROVIDER_DISPLAY.items()):
        provider_key = slug
        # Check if already exists
        existing = None
        for p in store.list_providers(include_disabled=True):
            if p["provider_key"] == provider_key:
                existing = p
                break
        if existing:
            continue

        adapter_type = "anthropic" if slug in _ANTHROPIC_FAMILY else "openai"
        base_url = _OFFICIAL_BASE_URLS.get(slug)
        response_format = _OFFICIAL_RESPONSE_FORMATS.get(slug)

        # Determine default model from the first entry in _PROVIDER_MODELS
        default_model = None
        models = _PROVIDER_MODELS.get(slug, [])
        if models:
            default_model = models[0]["id"]

        try:
            store.create_provider({
                "kind": "official",
                "provider_key": provider_key,
                "display_name": display_name,
                "adapter_type": adapter_type,
                "base_url": base_url,
                "enabled": True,
                "response_format": response_format,
                "default_model": default_model,
                "is_builtin_locked": True,
                "sort_order": sort_idx + 1,  # 1-indexed; active provider gets 0
            })
            created += 1
            logger.debug("Seeded official provider: %s (%s)", slug, display_name)
        except Exception as exc:
            logger.warning("Failed to seed official provider %s: %s", slug, exc)

    return created


# ---------------------------------------------------------------------------
# Custom provider import from config.yaml
# ---------------------------------------------------------------------------

def _import_custom_providers() -> dict[str, Any]:
    """Import config.yaml.custom_providers[] into the registry.

    Returns {imported, skipped, errors}.
    """
    from api.config import get_config

    config = get_config()
    custom = config.get("custom_providers", []) or []
    imported = 0
    skipped = 0
    errors: list[str] = []

    # Build a set of existing provider_keys for fast lookup
    existing_keys = {p["provider_key"] for p in store.list_providers(include_disabled=True)}

    for cp in custom:
        if not isinstance(cp, dict):
            skipped += 1
            continue
        name = cp.get("name") or cp.get("provider") or ""
        if not name:
            skipped += 1
            continue

        provider_key = f"custom:{name.lower().replace(' ', '-')}"
        if provider_key in existing_keys:
            skipped += 1
            continue

        adapter_type = cp.get("adapter_type", "openai")
        if adapter_type not in ("openai", "anthropic"):
            adapter_type = "openai"

        try:
            store.create_provider({
                "kind": "custom",
                "provider_key": provider_key,
                "display_name": name,
                "adapter_type": adapter_type,
                "base_url": cp.get("base_url"),
                "enabled": True,
                "response_format": cp.get("response_format"),
                "default_model": cp.get("default_model"),
                "is_builtin_locked": False,
                "sort_order": 1000 + imported,  # after official providers
            })
            imported += 1
            existing_keys.add(provider_key)
            logger.debug("Imported custom provider: %s", name)
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    return {"imported": imported, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# Active provider / default model import
# ---------------------------------------------------------------------------

def _import_active_provider() -> str | None:
    """Import the current active/default provider into the registry.

    Sets the active provider's sort_order to 0 (first in list) and returns
    its provider_key, or None if not found.
    """
    from api.config import get_config, _resolve_provider_alias

    config = get_config()
    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, dict):
        return None

    raw_provider = str(model_cfg.get("provider") or "").strip().lower()
    if not raw_provider:
        return None

    # Resolve aliases
    canonical = _resolve_provider_alias(raw_provider)
    provider_key = canonical

    # Find the provider in the registry
    for p in store.list_providers(include_disabled=True):
        if p["provider_key"] == provider_key:
            # Activate: set sort_order to 0 and enable
            store.update_provider(p["id"], {
                "enabled": True,
                "sort_order": 0,
            })
            logger.debug("Activated provider: %s (sort_order=0)", provider_key)
            return provider_key

    logger.info("Active provider %s not found in registry", provider_key)
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def bootstrap_on_startup() -> dict[str, Any]:
    """Run bootstrap import if the registry hasn't been bootstrapped yet.

    Idempotent — safe to call on every startup.  Returns a summary dict:
    {bootstrapped: bool, official_created, custom_imported, custom_skipped,
     custom_errors, active_provider}
    """
    # Ensure tables exist
    store.init_db()

    with store.transaction() as conn:
        if _is_bootstrap_done(conn):
            logger.debug("Bootstrap already complete, skipping")
            return {"bootstrapped": False, "reason": "already_done"}

    logger.info("Running provider registry bootstrap...")

    # 1. Seed official providers
    official_created = _seed_official_providers()
    logger.info("Seeded %d official providers", official_created)

    # 2. Import custom_providers[]
    custom_result = _import_custom_providers()
    logger.info(
        "Imported %d custom providers (%d skipped, %d errors)",
        custom_result["imported"],
        custom_result["skipped"],
        len(custom_result["errors"]),
    )

    # 3. Import active provider
    active_key = _import_active_provider()
    if active_key:
        logger.info("Active provider imported: %s", active_key)

    # 4. Mark bootstrap complete
    with store.transaction() as conn:
        _mark_bootstrap_done(conn)

    logger.info("Provider registry bootstrap complete")

    return {
        "bootstrapped": True,
        "official_created": official_created,
        "custom_imported": custom_result["imported"],
        "custom_skipped": custom_result["skipped"],
        "custom_errors": custom_result["errors"],
        "active_provider": active_key,
    }
