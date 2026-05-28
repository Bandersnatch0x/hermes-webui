"""Provider adapters for the registry system.

Each adapter translates between a provider-specific API and the unified
internal interface used by the registry services.
"""

from api.provider_registry.adapters.base import (
    ChatResult,
    ModelInfo,
    ProviderAdapter,
    ProviderRequest,
    UsageResult,
)
from api.provider_registry.adapters.anthropic import AnthropicAdapter
from api.provider_registry.adapters.openai import OpenAIAdapter
from api.provider_registry.adapters.resolution import (
    ADAPTER_TYPES,
    resolve_adapter,
)

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
]
