"""Adapter resolution — dispatch to the right adapter class by type.

Usage::

    from api.provider_registry.adapters.resolution import resolve_adapter

    adapter = resolve_adapter("openai", response_format="responses")
    req = adapter.build_chat_request(messages, model="gpt-4o")
"""

from __future__ import annotations

from typing import Any

from api.provider_registry.adapters.anthropic import AnthropicAdapter
from api.provider_registry.adapters.base import ProviderAdapter
from api.provider_registry.adapters.openai import OpenAIAdapter

#: Registry of adapter type names to their concrete classes.
ADAPTER_TYPES: dict[str, type[ProviderAdapter]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
}

#: Valid response_format values for each adapter type.
_RESPONSE_FORMAT_BY_TYPE: dict[str, frozenset[str] | None] = {
    "openai": frozenset({"completions", "messages", "responses"}),
    "anthropic": None,  # must be None
}


def resolve_adapter(
    adapter_type: str,
    response_format: str | None = None,
) -> ProviderAdapter:
    """Return a concrete adapter instance for the given type.

    Parameters
    ----------
    adapter_type:
        One of the keys in :data:`ADAPTER_TYPES` (``openai``,
        ``anthropic``).
    response_format:
        Required for ``openai`` (defaults to ``completions``).
        Must be ``None`` for ``anthropic``.

    Raises
    ------
    ValueError
        If *adapter_type* is unknown, or *response_format* is invalid
        for the given type.
    """
    if adapter_type not in ADAPTER_TYPES:
        valid = ", ".join(sorted(ADAPTER_TYPES))
        raise ValueError(
            f"Unknown adapter_type {adapter_type!r}; expected one of: {valid}"
        )

    allowed_formats = _RESPONSE_FORMAT_BY_TYPE.get(adapter_type)

    # Anthropic (and other non-OpenAI types): response_format must be None.
    if allowed_formats is None:
        if response_format is not None:
            raise ValueError(
                f"adapter_type={adapter_type!r} does not support "
                f"response_format (got {response_format!r})"
            )
        return ADAPTER_TYPES[adapter_type]()

    # OpenAI: validate response_format, default to 'completions'.
    fmt = response_format or "completions"
    if fmt not in allowed_formats:
        raise ValueError(
            f"Invalid response_format {fmt!r} for adapter_type={adapter_type!r}; "
            f"expected one of: {', '.join(sorted(allowed_formats))}"
        )
    return ADAPTER_TYPES[adapter_type](response_format=fmt)
