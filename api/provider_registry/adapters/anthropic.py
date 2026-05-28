"""Anthropic provider adapter.

Implements the Anthropic Messages API (``POST /v1/messages``).
"""

from __future__ import annotations

import json
from typing import Any

from api.provider_registry.adapters.base import (
    ChatResult,
    ModelInfo,
    ProviderAdapter,
    ProviderRequest,
    UsageResult,
)

_ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicAdapter(ProviderAdapter):
    """Adapter for the Anthropic Messages API.

    Uses ``x-api-key`` authentication and the ``anthropic-version`` header.
    """

    # -- Chat request building ------------------------------------------------

    def build_chat_request(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> ProviderRequest:
        """POST /v1/messages — Anthropic Messages API.

        Parameters
        ----------
        messages:
            Conversation messages.  Each message must have ``role`` and
            ``content`` keys, matching the Anthropic messages format.
        model:
            Anthropic model identifier (e.g. ``claude-sonnet-4-20250514``).
        **kwargs:
            Optional parameters: ``max_tokens``, ``system``, ``temperature``,
            ``top_p``, ``stop_sequences``, ``stream``.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "stream": False,
        }
        if "system" in kwargs and kwargs["system"] is not None:
            body["system"] = kwargs["system"]
        for key in ("temperature", "top_p", "stop_sequences"):
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]

        headers = {
            "Content-Type": "application/json",
            "x-api-key": "",  # caller must inject
            "anthropic-version": _ANTHROPIC_API_VERSION,
        }
        return ProviderRequest(
            method="POST",
            path="/v1/messages",
            headers=headers,
            body=body,
        )

    # -- Chat response parsing ------------------------------------------------

    def parse_chat_response(self, response_data: dict[str, Any]) -> ChatResult:
        """Parse an Anthropic Messages API response.

        Expected shape::

            {
                "id": "msg_...",
                "model": "claude-sonnet-4-20250514",
                "content": [{"type": "text", "text": "..."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": N, "output_tokens": N}
            }
        """
        model = str(response_data.get("model", ""))
        content_blocks = response_data.get("content") or []
        text_parts = [
            block.get("text", "")
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        content = "\n".join(text_parts)

        usage_raw = response_data.get("usage")
        usage = None
        if isinstance(usage_raw, dict):
            input_tokens = usage_raw.get("input_tokens", 0)
            output_tokens = usage_raw.get("output_tokens", 0)
            usage = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        return ChatResult(
            content=content,
            model=model,
            usage=usage,
            raw=response_data,
        )

    # -- Models ---------------------------------------------------------------

    def build_models_request(self) -> ProviderRequest | None:
        """Anthropic has no public models endpoint.

        Returns ``None`` so the caller falls back to static catalogs.
        """
        return None

    def parse_models_response(self, response_data: dict[str, Any]) -> list[ModelInfo]:
        """Anthropic has no public models endpoint — returns empty list."""
        return []

    # -- Usage probing --------------------------------------------------------

    def native_usage_probe_supported(self) -> bool:
        """Anthropic has no standard billing/usage endpoint."""
        return False

    def probe_usage(self, api_key: str, base_url: str) -> UsageResult | None:
        """Anthropic has no native usage probing — returns None."""
        return None
