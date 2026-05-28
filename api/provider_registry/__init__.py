"""Provider registry for Hermes WebUI.

SQLite-backed provider management that unifies official and third-party
providers under one runtime model.
"""

from api.provider_registry.adapters import (
    ADAPTER_TYPES,
    AnthropicAdapter,
    ChatResult,
    ModelInfo,
    OpenAIAdapter,
    ProviderAdapter,
    ProviderRequest,
    UsageResult,
    resolve_adapter,
)
from api.provider_registry import services, store

__all__ = [
    "ADAPTER_TYPES",
    "AnthropicAdapter",
    "ChatResult",
    "ModelInfo",
    "OpenAIAdapter",
    "ProviderAdapter",
    "ProviderRequest",
    "UsageResult",
    "resolve_adapter",
    "services",
    "store",
]
