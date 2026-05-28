"""Base types and abstract interface for provider adapters.

Each adapter translates between a provider-specific API shape and the
unified internal types used by the registry services.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProviderRequest:
    """A fully-built provider API request, ready for HTTP execution.

    The adapter owns request shape; HTTP execution is the caller's
    responsibility.
    """

    method: str
    path: str
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] | None = None
    query: dict[str, str] | None = None


@dataclass(frozen=True)
class ChatResult:
    """Unified chat completion result across all adapter types."""

    content: str
    model: str
    usage: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelInfo:
    """A single model entry from a provider's model list."""

    id: str
    label: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UsageResult:
    """Provider usage/quota probe result.

    ``status`` is one of: ``available``, ``unsupported``, ``error``, ``unknown``.
    """

    status: str
    payload: dict[str, Any] | None = None
    message: str | None = None


class ProviderAdapter(ABC):
    """Abstract base class for provider adapters.

    Each concrete adapter (OpenAI, Anthropic, etc.) implements request
    building and response parsing for its provider family.
    """

    @abstractmethod
    def build_chat_request(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> ProviderRequest:
        """Build a provider-specific chat completion request.

        Parameters
        ----------
        messages:
            Conversation messages in the provider's expected format.
        model:
            Model identifier.
        **kwargs:
            Provider-specific options (temperature, max_tokens, etc.).
        """

    @abstractmethod
    def parse_chat_response(self, response_data: dict[str, Any]) -> ChatResult:
        """Parse a provider chat response into a unified ChatResult."""

    @abstractmethod
    def build_models_request(self) -> ProviderRequest | None:
        """Build a request to list available models.

        Returns ``None`` when the provider has no public models endpoint
        and the caller should fall back to static catalogs.
        """

    @abstractmethod
    def parse_models_response(self, response_data: dict[str, Any]) -> list[ModelInfo]:
        """Parse a provider models list response."""

    @abstractmethod
    def probe_usage(self, api_key: str, base_url: str) -> UsageResult | None:
        """Attempt to probe provider usage/quota.

        Returns ``None`` when the adapter has no native usage probing
        capability.
        """

    @abstractmethod
    def native_usage_probe_supported(self) -> bool:
        """Whether this adapter has built-in usage probing."""
