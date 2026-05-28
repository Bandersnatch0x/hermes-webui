"""OpenAI provider adapter.

Supports three response formats:
- ``completions``: POST /v1/chat/completions (standard OpenAI Chat Completions API)
- ``messages``:    POST /v1/messages (Anthropic-compatible via LiteLLM/proxy)
- ``responses``:   POST /v1/responses (OpenAI Responses API)
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

_VALID_RESPONSE_FORMATS = frozenset({"completions", "messages", "responses"})


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI-compatible providers.

    Parameters
    ----------
    response_format:
        One of ``completions``, ``messages``, ``responses``.
        Determines which API shape is used for chat requests.
    """

    def __init__(self, response_format: str = "completions") -> None:
        if response_format not in _VALID_RESPONSE_FORMATS:
            raise ValueError(
                f"Invalid response_format {response_format!r}; "
                f"expected one of: {', '.join(sorted(_VALID_RESPONSE_FORMATS))}"
            )
        self._response_format = response_format

    @property
    def response_format(self) -> str:
        return self._response_format

    # -- Chat request building ------------------------------------------------

    def build_chat_request(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> ProviderRequest:
        """Build an OpenAI chat request dispatched by response_format."""
        if self._response_format == "completions":
            return self._build_completions_request(messages, model, **kwargs)
        if self._response_format == "messages":
            return self._build_messages_request(messages, model, **kwargs)
        return self._build_responses_request(messages, model, **kwargs)

    def _build_completions_request(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> ProviderRequest:
        """POST /v1/chat/completions — standard OpenAI Chat Completions."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        for key in ("temperature", "max_tokens", "top_p", "frequency_penalty",
                     "presence_penalty", "stop", "n", "logprobs", "tools",
                     "tool_choice", "response_format"):
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]
        return ProviderRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            body=body,
        )

    def _build_messages_request(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> ProviderRequest:
        """POST /v1/messages — Anthropic-compatible messages via proxy."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "stream": False,
        }
        if "system" in kwargs:
            body["system"] = kwargs["system"]
        for key in ("temperature", "top_p", "stop_sequences"):
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]
        return ProviderRequest(
            method="POST",
            path="/v1/messages",
            headers={"Content-Type": "application/json"},
            body=body,
        )

    def _build_responses_request(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> ProviderRequest:
        """POST /v1/responses — OpenAI Responses API."""
        # Convert messages list into a single input string or structured input.
        # The Responses API uses 'input' instead of 'messages'.
        input_items: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            input_items.append({"role": role, "content": content})

        body: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": False,
        }
        if "instructions" in kwargs:
            body["instructions"] = kwargs["instructions"]
        for key in ("temperature", "max_output_tokens", "top_p",
                     "tools", "tool_choice"):
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]
        return ProviderRequest(
            method="POST",
            path="/v1/responses",
            headers={"Content-Type": "application/json"},
            body=body,
        )

    # -- Chat response parsing ------------------------------------------------

    def parse_chat_response(self, response_data: dict[str, Any]) -> ChatResult:
        """Parse an OpenAI chat response dispatched by response_format."""
        if self._response_format == "completions":
            return self._parse_completions_response(response_data)
        if self._response_format == "messages":
            return self._parse_messages_response(response_data)
        return self._parse_responses_response(response_data)

    def _parse_completions_response(self, data: dict[str, Any]) -> ChatResult:
        """Parse /v1/chat/completions response."""
        model = str(data.get("model", ""))
        choices = data.get("choices") or []
        content = ""
        if choices:
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "")
        usage = data.get("usage") or None
        if usage is not None:
            usage = dict(usage)
        return ChatResult(content=content, model=model, usage=usage, raw=data)

    def _parse_messages_response(self, data: dict[str, Any]) -> ChatResult:
        """Parse /v1/messages response (Anthropic-compatible shape)."""
        model = str(data.get("model", ""))
        content_blocks = data.get("content") or []
        text_parts = [
            block.get("text", "")
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        content = "\n".join(text_parts)
        usage_raw = data.get("usage") or None
        usage = None
        if usage_raw is not None:
            usage = {
                "prompt_tokens": usage_raw.get("input_tokens", 0),
                "completion_tokens": usage_raw.get("output_tokens", 0),
                "total_tokens": (
                    usage_raw.get("input_tokens", 0)
                    + usage_raw.get("output_tokens", 0)
                ),
            }
        return ChatResult(content=content, model=model, usage=usage, raw=data)

    def _parse_responses_response(self, data: dict[str, Any]) -> ChatResult:
        """Parse /v1/responses response."""
        model = str(data.get("model", ""))
        # The Responses API returns output as a list of items.
        output_items = data.get("output") or []
        text_parts: list[str] = []
        for item in output_items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text_parts.append(str(part.get("text", "")))
        content = "\n".join(text_parts)
        usage_raw = data.get("usage") or None
        usage = None
        if usage_raw is not None:
            usage = dict(usage_raw)
        return ChatResult(content=content, model=model, usage=usage, raw=data)

    # -- Models ---------------------------------------------------------------

    def build_models_request(self) -> ProviderRequest:
        """GET /v1/models."""
        return ProviderRequest(
            method="GET",
            path="/v1/models",
            headers={},
        )

    def parse_models_response(self, response_data: dict[str, Any]) -> list[ModelInfo]:
        """Parse standard OpenAI {data: [{id, owned_by, ...}]} models list."""
        data = response_data.get("data") or []
        results: list[ModelInfo] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            if not model_id:
                continue
            label = entry.get("owned_by")
            results.append(ModelInfo(
                id=str(model_id),
                label=str(label) if label else None,
                raw=entry,
            ))
        return results

    # -- Usage probing --------------------------------------------------------

    def native_usage_probe_supported(self) -> bool:
        """OpenAI has a billing/subscription endpoint."""
        return True

    def probe_usage(self, api_key: str, base_url: str) -> UsageResult | None:
        """Probe OpenAI billing/subscription for usage info.

        This method builds and returns a *request* is not how we do it —
        instead it actually performs the HTTP call using urllib, since usage
        probing is a side-effect operation that lives at the adapter boundary.

        Returns a UsageResult or None on network/parse failure.
        """
        import urllib.request
        import urllib.error

        url = base_url.rstrip("/") + "/v1/dashboard/billing/subscription"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            return UsageResult(status="available", payload=raw, message=None)
        except urllib.error.HTTPError as exc:
            return UsageResult(
                status="error",
                payload=None,
                message=f"HTTP {exc.code}: {exc.reason}",
            )
        except Exception as exc:
            return UsageResult(
                status="error",
                payload=None,
                message=str(exc),
            )
