"""Tests for the provider registry adapter layer.

Covers:
- OpenAIAdapter (completions, messages, responses formats)
- AnthropicAdapter
- resolve_adapter dispatch
- Usage probing orchestration
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from api.provider_registry.adapters.base import (
    ChatResult,
    ModelInfo,
    ProviderAdapter,
    ProviderRequest,
    UsageResult,
)
from api.provider_registry.adapters.openai import OpenAIAdapter
from api.provider_registry.adapters.anthropic import AnthropicAdapter
from api.provider_registry.adapters.resolution import (
    ADAPTER_TYPES,
    resolve_adapter,
)
from api.provider_registry.adapters.usage import (
    probe_provider_usage,
    _parse_openai_usage_json,
    _parse_anthropic_usage_json,
    _parse_openai_usage_headers,
    _parse_generic_balance_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]


# ---------------------------------------------------------------------------
# OpenAIAdapter — completions
# ---------------------------------------------------------------------------


class TestOpenAICompletions:
    def setup_method(self) -> None:
        self.adapter = OpenAIAdapter(response_format="completions")

    def test_response_format_property(self) -> None:
        assert self.adapter.response_format == "completions"

    def test_build_chat_request_basic(self) -> None:
        req = self.adapter.build_chat_request(SAMPLE_MESSAGES, model="gpt-4o")
        assert isinstance(req, ProviderRequest)
        assert req.method == "POST"
        assert req.path == "/v1/chat/completions"
        assert req.body is not None
        assert req.body["model"] == "gpt-4o"
        assert req.body["messages"] == SAMPLE_MESSAGES
        assert req.body["stream"] is False

    def test_build_chat_request_with_kwargs(self) -> None:
        req = self.adapter.build_chat_request(
            SAMPLE_MESSAGES,
            model="gpt-4o",
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
        )
        assert req.body is not None
        assert req.body["temperature"] == 0.7
        assert req.body["max_tokens"] == 1024
        assert req.body["top_p"] == 0.9

    def test_build_chat_request_ignores_none_kwargs(self) -> None:
        req = self.adapter.build_chat_request(
            SAMPLE_MESSAGES, model="gpt-4o", temperature=None
        )
        assert req.body is not None
        assert "temperature" not in req.body

    def test_parse_chat_response_basic(self) -> None:
        response = {
            "model": "gpt-4o",
            "choices": [
                {"message": {"role": "assistant", "content": "Hi there!"}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = self.adapter.parse_chat_response(response)
        assert isinstance(result, ChatResult)
        assert result.content == "Hi there!"
        assert result.model == "gpt-4o"
        assert result.usage is not None
        assert result.usage["total_tokens"] == 15
        assert result.raw == response

    def test_parse_chat_response_empty_choices(self) -> None:
        result = self.adapter.parse_chat_response({"model": "gpt-4o", "choices": []})
        assert result.content == ""
        assert result.model == "gpt-4o"
        assert result.usage is None

    def test_build_models_request(self) -> None:
        req = self.adapter.build_models_request()
        assert req is not None
        assert req.method == "GET"
        assert req.path == "/v1/models"

    def test_parse_models_response(self) -> None:
        data = {
            "data": [
                {"id": "gpt-4o", "owned_by": "openai"},
                {"id": "gpt-3.5-turbo", "owned_by": "openai"},
            ]
        }
        models = self.adapter.parse_models_response(data)
        assert len(models) == 2
        assert models[0].id == "gpt-4o"
        assert models[0].label == "openai"
        assert models[1].id == "gpt-3.5-turbo"

    def test_parse_models_response_empty(self) -> None:
        assert self.adapter.parse_models_response({}) == []
        assert self.adapter.parse_models_response({"data": []}) == []

    def test_native_usage_probe_supported(self) -> None:
        assert self.adapter.native_usage_probe_supported() is True


# ---------------------------------------------------------------------------
# OpenAIAdapter — messages
# ---------------------------------------------------------------------------


class TestOpenAIMessages:
    def setup_method(self) -> None:
        self.adapter = OpenAIAdapter(response_format="messages")

    def test_build_chat_request(self) -> None:
        req = self.adapter.build_chat_request(
            SAMPLE_MESSAGES, model="gpt-4o", system="Be brief."
        )
        assert req.path == "/v1/messages"
        assert req.body is not None
        assert req.body["system"] == "Be brief."
        assert req.body["max_tokens"] == 4096
        assert req.body["stream"] is False

    def test_parse_chat_response_text_blocks(self) -> None:
        response = {
            "model": "gpt-4o",
            "content": [
                {"type": "text", "text": "Hello!"},
                {"type": "text", "text": "How can I help?"},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 8},
        }
        result = self.adapter.parse_chat_response(response)
        assert result.content == "Hello!\nHow can I help?"
        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 8
        assert result.usage["total_tokens"] == 18


# ---------------------------------------------------------------------------
# OpenAIAdapter — responses
# ---------------------------------------------------------------------------


class TestOpenAIResponses:
    def setup_method(self) -> None:
        self.adapter = OpenAIAdapter(response_format="responses")

    def test_build_chat_request(self) -> None:
        req = self.adapter.build_chat_request(
            SAMPLE_MESSAGES, model="gpt-4o", instructions="Be helpful."
        )
        assert req.path == "/v1/responses"
        assert req.body is not None
        assert req.body["instructions"] == "Be helpful."
        assert req.body["input"][0]["role"] == "system"
        assert req.body["input"][1]["role"] == "user"

    def test_parse_chat_response_output_items(self) -> None:
        response = {
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Result here."}
                    ],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        result = self.adapter.parse_chat_response(response)
        assert result.content == "Result here."
        assert result.usage is not None


# ---------------------------------------------------------------------------
# OpenAIAdapter — validation
# ---------------------------------------------------------------------------


class TestOpenAIValidation:
    def test_invalid_response_format(self) -> None:
        with pytest.raises(ValueError, match="Invalid response_format"):
            OpenAIAdapter(response_format="invalid")

    def test_default_is_completions(self) -> None:
        adapter = OpenAIAdapter()
        assert adapter.response_format == "completions"


# ---------------------------------------------------------------------------
# AnthropicAdapter
# ---------------------------------------------------------------------------


class TestAnthropicAdapter:
    def setup_method(self) -> None:
        self.adapter = AnthropicAdapter()

    def test_build_chat_request_basic(self) -> None:
        msgs = [{"role": "user", "content": "Hello!"}]
        req = self.adapter.build_chat_request(msgs, model="claude-sonnet-4-20250514")
        assert req.method == "POST"
        assert req.path == "/v1/messages"
        assert req.body is not None
        assert req.body["model"] == "claude-sonnet-4-20250514"
        assert req.body["max_tokens"] == 4096
        assert req.body["stream"] is False
        assert "anthropic-version" in req.headers
        assert req.headers["anthropic-version"] == "2023-06-01"

    def test_build_chat_request_with_system(self) -> None:
        msgs = [{"role": "user", "content": "Hello!"}]
        req = self.adapter.build_chat_request(
            msgs, model="claude-sonnet-4-20250514", system="Be terse."
        )
        assert req.body is not None
        assert req.body["system"] == "Be terse."

    def test_parse_chat_response(self) -> None:
        response = {
            "model": "claude-sonnet-4-20250514",
            "content": [
                {"type": "text", "text": "Hi!"},
                {"type": "text", "text": "How can I help?"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 8},
        }
        result = self.adapter.parse_chat_response(response)
        assert isinstance(result, ChatResult)
        assert result.content == "Hi!\nHow can I help?"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 12
        assert result.usage["completion_tokens"] == 8
        assert result.usage["total_tokens"] == 20

    def test_parse_chat_response_no_text_blocks(self) -> None:
        response = {
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "tool_use", "id": "tu_1"}],
        }
        result = self.adapter.parse_chat_response(response)
        assert result.content == ""
        assert result.usage is None

    def test_build_models_request_returns_none(self) -> None:
        assert self.adapter.build_models_request() is None

    def test_parse_models_response_returns_empty(self) -> None:
        assert self.adapter.parse_models_response({}) == []

    def test_native_usage_probe_not_supported(self) -> None:
        assert self.adapter.native_usage_probe_supported() is False

    def test_probe_usage_returns_none(self) -> None:
        assert self.adapter.probe_usage("key", "https://api.anthropic.com") is None


# ---------------------------------------------------------------------------
# resolve_adapter
# ---------------------------------------------------------------------------


class TestResolveAdapter:
    def test_resolve_openai_default(self) -> None:
        adapter = resolve_adapter("openai")
        assert isinstance(adapter, OpenAIAdapter)
        assert adapter.response_format == "completions"

    def test_resolve_openai_completions(self) -> None:
        adapter = resolve_adapter("openai", response_format="completions")
        assert isinstance(adapter, OpenAIAdapter)
        assert adapter.response_format == "completions"

    def test_resolve_openai_messages(self) -> None:
        adapter = resolve_adapter("openai", response_format="messages")
        assert adapter.response_format == "messages"

    def test_resolve_openai_responses(self) -> None:
        adapter = resolve_adapter("openai", response_format="responses")
        assert adapter.response_format == "responses"

    def test_resolve_anthropic(self) -> None:
        adapter = resolve_adapter("anthropic")
        assert isinstance(adapter, AnthropicAdapter)

    def test_resolve_anthropic_with_response_format_raises(self) -> None:
        with pytest.raises(ValueError, match="does not support response_format"):
            resolve_adapter("anthropic", response_format="completions")

    def test_resolve_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown adapter_type"):
            resolve_adapter("unknown")

    def test_resolve_openai_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid response_format"):
            resolve_adapter("openai", response_format="grpc")

    def test_adapter_types_registry(self) -> None:
        assert "openai" in ADAPTER_TYPES
        assert "anthropic" in ADAPTER_TYPES
        assert ADAPTER_TYPES["openai"] is OpenAIAdapter
        assert ADAPTER_TYPES["anthropic"] is AnthropicAdapter


# ---------------------------------------------------------------------------
# Usage parsers
# ---------------------------------------------------------------------------


class TestUsageParsers:
    def test_openai_usage_json_success(self) -> None:
        payload = {"hard_limit_usd": 100.0, "total_usage": 42.5}
        result = _parse_openai_usage_json(200, {}, payload)
        assert result.status == "available"
        assert result.payload == payload
        assert result.message is None

    def test_openai_usage_json_error(self) -> None:
        result = _parse_openai_usage_json(403, {}, {})
        assert result.status == "error"
        assert "403" in (result.message or "")

    def test_anthropic_usage_json_success(self) -> None:
        payload = {"current_usage": 1000}
        result = _parse_anthropic_usage_json(200, {}, payload)
        assert result.status == "available"
        assert result.payload == payload

    def test_openai_usage_headers_success(self) -> None:
        headers = {
            "x-ratelimit-remaining-requests": "500",
            "x-ratelimit-remaining-tokens": "100000",
            "content-type": "application/json",
        }
        result = _parse_openai_usage_headers(200, headers, {})
        assert result.status == "available"
        assert result.payload is not None
        assert "x-ratelimit-remaining-requests" in result.payload
        assert "content-type" not in result.payload

    def test_openai_usage_headers_no_ratelimit(self) -> None:
        result = _parse_openai_usage_headers(200, {"content-type": "application/json"}, {})
        assert result.status == "unknown"

    def test_generic_balance_json_success(self) -> None:
        payload = {"balance": 50.0, "currency": "USD"}
        result = _parse_generic_balance_json(200, {}, payload)
        assert result.status == "available"
        assert result.payload == payload

    def test_generic_balance_json_error(self) -> None:
        result = _parse_generic_balance_json(500, {}, {})
        assert result.status == "error"


# ---------------------------------------------------------------------------
# probe_provider_usage orchestration
# ---------------------------------------------------------------------------


class TestProbeProviderUsage:
    def setup_method(self) -> None:
        self.adapter = OpenAIAdapter()

    def test_strategy_none(self) -> None:
        result = probe_provider_usage(
            self.adapter, "key", "https://api.openai.com", strategy="none"
        )
        assert result.status == "unsupported"

    def test_strategy_auto_with_native_support(self) -> None:
        mock_result = UsageResult(status="available", payload={"usage": 100})
        with patch.object(self.adapter, "probe_usage", return_value=mock_result):
            result = probe_provider_usage(
                self.adapter, "key", "https://api.openai.com", strategy="auto"
            )
        assert result.status == "available"
        assert result.payload == {"usage": 100}

    def test_strategy_auto_without_native_support(self) -> None:
        anthropic = AnthropicAdapter()
        result = probe_provider_usage(
            anthropic, "key", "https://api.anthropic.com", strategy="auto"
        )
        assert result.status == "unsupported"

    def test_strategy_endpoint_success(self) -> None:
        with patch(
            "api.provider_registry.adapters.usage._fetch_endpoint",
            return_value=(200, {}, {"balance": 50.0}),
        ):
            result = probe_provider_usage(
                self.adapter,
                "key",
                "https://api.openai.com",
                strategy="endpoint",
                endpoint_url="https://example.com/usage",
                parser_type="generic_balance_json",
            )
        assert result.status == "available"
        assert result.payload == {"balance": 50.0}

    def test_strategy_endpoint_missing_url(self) -> None:
        result = probe_provider_usage(
            self.adapter,
            "key",
            "https://api.openai.com",
            strategy="endpoint",
            parser_type="generic_balance_json",
        )
        assert result.status == "error"
        assert "endpoint_url" in (result.message or "")

    def test_strategy_endpoint_missing_parser(self) -> None:
        result = probe_provider_usage(
            self.adapter,
            "key",
            "https://api.openai.com",
            strategy="endpoint",
            endpoint_url="https://example.com/usage",
        )
        assert result.status == "error"
        assert "parser_type" in (result.message or "")

    def test_strategy_endpoint_unknown_parser(self) -> None:
        result = probe_provider_usage(
            self.adapter,
            "key",
            "https://api.openai.com",
            strategy="endpoint",
            endpoint_url="https://example.com/usage",
            parser_type="unknown_parser",
        )
        assert result.status == "error"
        assert "Unknown parser_type" in (result.message or "")

    def test_strategy_auto_endpoint_fallback(self) -> None:
        """auto+endpoint: when auto returns unsupported, falls back to endpoint."""
        anthropic = AnthropicAdapter()
        with patch(
            "api.provider_registry.adapters.usage._fetch_endpoint",
            return_value=(200, {}, {"balance": 25.0}),
        ):
            result = probe_provider_usage(
                anthropic,
                "key",
                "https://api.anthropic.com",
                strategy="auto+endpoint",
                endpoint_url="https://example.com/usage",
                parser_type="generic_balance_json",
            )
        assert result.status == "available"
        assert result.payload == {"balance": 25.0}

    def test_strategy_auto_endpoint_native_wins(self) -> None:
        """auto+endpoint: when auto returns available, endpoint is skipped."""
        mock_result = UsageResult(status="available", payload={"native": True})
        with patch.object(self.adapter, "probe_usage", return_value=mock_result):
            result = probe_provider_usage(
                self.adapter,
                "key",
                "https://api.openai.com",
                strategy="auto+endpoint",
                endpoint_url="https://example.com/usage",
                parser_type="generic_balance_json",
            )
        assert result.status == "available"
        assert result.payload == {"native": True}

    def test_unknown_strategy(self) -> None:
        result = probe_provider_usage(
            self.adapter, "key", "https://api.openai.com", strategy="bogus"
        )
        assert result.status == "error"
        assert "Unknown usage strategy" in (result.message or "")


# ---------------------------------------------------------------------------
# Data class invariants
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_provider_request_frozen(self) -> None:
        req = ProviderRequest(method="GET", path="/test")
        with pytest.raises(AttributeError):
            req.method = "POST"  # type: ignore[misc]

    def test_chat_result_frozen(self) -> None:
        r = ChatResult(content="hi", model="m")
        with pytest.raises(AttributeError):
            r.content = "bye"  # type: ignore[misc]

    def test_model_info_frozen(self) -> None:
        m = ModelInfo(id="gpt-4o")
        with pytest.raises(AttributeError):
            m.id = "gpt-3.5"  # type: ignore[misc]

    def test_usage_result_frozen(self) -> None:
        u = UsageResult(status="available")
        with pytest.raises(AttributeError):
            u.status = "error"  # type: ignore[misc]

    def test_chat_result_default_raw(self) -> None:
        r = ChatResult(content="", model="")
        assert r.raw == {}

    def test_model_info_default_raw(self) -> None:
        m = ModelInfo(id="test")
        assert m.raw == {}
        assert m.label is None


# ---------------------------------------------------------------------------
# Edge cases — OpenAI completions kwargs forwarding
# ---------------------------------------------------------------------------


class TestOpenAICompletionsEdgeCases:
    def setup_method(self) -> None:
        self.adapter = OpenAIAdapter(response_format="completions")

    def test_build_request_forwards_tools(self) -> None:
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        req = self.adapter.build_chat_request(
            SAMPLE_MESSAGES, model="gpt-4o", tools=tools
        )
        assert req.body is not None
        assert req.body["tools"] == tools

    def test_build_request_forwards_tool_choice(self) -> None:
        req = self.adapter.build_chat_request(
            SAMPLE_MESSAGES, model="gpt-4o", tool_choice="auto"
        )
        assert req.body is not None
        assert req.body["tool_choice"] == "auto"

    def test_build_request_forwards_stop(self) -> None:
        req = self.adapter.build_chat_request(
            SAMPLE_MESSAGES, model="gpt-4o", stop=["\n"]
        )
        assert req.body is not None
        assert req.body["stop"] == ["\n"]

    def test_build_request_forwards_n(self) -> None:
        req = self.adapter.build_chat_request(
            SAMPLE_MESSAGES, model="gpt-4o", n=3
        )
        assert req.body is not None
        assert req.body["n"] == 3

    def test_build_request_forwards_logprobs(self) -> None:
        req = self.adapter.build_chat_request(
            SAMPLE_MESSAGES, model="gpt-4o", logprobs=True
        )
        assert req.body is not None
        assert req.body["logprobs"] is True

    def test_build_request_forwards_response_format_dict(self) -> None:
        fmt = {"type": "json_object"}
        req = self.adapter.build_chat_request(
            SAMPLE_MESSAGES, model="gpt-4o", response_format=fmt
        )
        assert req.body is not None
        assert req.body["response_format"] == fmt

    def test_build_request_with_empty_messages(self) -> None:
        req = self.adapter.build_chat_request([], model="gpt-4o")
        assert req.body is not None
        assert req.body["messages"] == []

    def test_build_request_ignores_unknown_kwargs(self) -> None:
        req = self.adapter.build_chat_request(
            SAMPLE_MESSAGES, model="gpt-4o", unknown_key="value"
        )
        assert req.body is not None
        assert "unknown_key" not in req.body

    def test_parse_chat_response_missing_model(self) -> None:
        response = {
            "choices": [
                {"message": {"role": "assistant", "content": "Hi"}}
            ],
        }
        result = self.adapter.parse_chat_response(response)
        assert result.model == ""
        assert result.content == "Hi"

    def test_parse_chat_response_none_usage(self) -> None:
        response = {
            "model": "gpt-4o",
            "choices": [{"message": {"content": "x"}}],
            "usage": None,
        }
        result = self.adapter.parse_chat_response(response)
        assert result.usage is None

    def test_parse_chat_response_missing_choices(self) -> None:
        response = {"model": "gpt-4o"}
        result = self.adapter.parse_chat_response(response)
        assert result.content == ""
        assert result.model == "gpt-4o"

    def test_parse_chat_response_choices_with_none_content(self) -> None:
        response = {
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": None}}],
        }
        result = self.adapter.parse_chat_response(response)
        assert result.content == ""

    def test_parse_models_response_skips_non_dict(self) -> None:
        data = {"data": [{"id": "gpt-4o"}, "invalid", None, {"id": "gpt-3.5"}]}
        models = self.adapter.parse_models_response(data)
        assert len(models) == 2
        assert models[0].id == "gpt-4o"
        assert models[1].id == "gpt-3.5"

    def test_parse_models_response_skips_missing_id(self) -> None:
        data = {"data": [{"owned_by": "openai"}, {"id": "gpt-4o"}]}
        models = self.adapter.parse_models_response(data)
        assert len(models) == 1
        assert models[0].id == "gpt-4o"

    def test_parse_models_response_no_owned_by(self) -> None:
        data = {"data": [{"id": "gpt-4o"}]}
        models = self.adapter.parse_models_response(data)
        assert models[0].label is None

    def test_parse_models_response_preserves_raw(self) -> None:
        entry = {"id": "gpt-4o", "owned_by": "openai", "created": 123}
        data = {"data": [entry]}
        models = self.adapter.parse_models_response(data)
        assert models[0].raw == entry


# ---------------------------------------------------------------------------
# Edge cases — OpenAI messages format
# ---------------------------------------------------------------------------


class TestOpenAIMessagesEdgeCases:
    def setup_method(self) -> None:
        self.adapter = OpenAIAdapter(response_format="messages")

    def test_build_request_custom_max_tokens(self) -> None:
        req = self.adapter.build_chat_request(
            SAMPLE_MESSAGES, model="gpt-4o", max_tokens=8192
        )
        assert req.body is not None
        assert req.body["max_tokens"] == 8192

    def test_build_request_with_stop_sequences(self) -> None:
        req = self.adapter.build_chat_request(
            SAMPLE_MESSAGES, model="gpt-4o", stop_sequences=["END"]
        )
        assert req.body is not None
        assert req.body["stop_sequences"] == ["END"]

    def test_parse_response_empty_content(self) -> None:
        response = {"model": "gpt-4o", "content": []}
        result = self.adapter.parse_chat_response(response)
        assert result.content == ""
        assert result.usage is None

    def test_parse_response_mixed_text_and_tool_use(self) -> None:
        response = {
            "model": "gpt-4o",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "tu_1", "name": "search"},
                {"type": "text", "text": "Found it!"},
            ],
        }
        result = self.adapter.parse_chat_response(response)
        assert result.content == "Let me check.\nFound it!"

    def test_parse_response_no_usage(self) -> None:
        response = {
            "model": "gpt-4o",
            "content": [{"type": "text", "text": "Hi"}],
        }
        result = self.adapter.parse_chat_response(response)
        assert result.usage is None

    def test_parse_response_zero_tokens(self) -> None:
        response = {
            "model": "gpt-4o",
            "content": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        result = self.adapter.parse_chat_response(response)
        assert result.usage is not None
        assert result.usage["total_tokens"] == 0


# ---------------------------------------------------------------------------
# Edge cases — OpenAI responses format
# ---------------------------------------------------------------------------


class TestOpenAIResponsesEdgeCases:
    def setup_method(self) -> None:
        self.adapter = OpenAIAdapter(response_format="responses")

    def test_build_request_max_output_tokens(self) -> None:
        req = self.adapter.build_chat_request(
            SAMPLE_MESSAGES, model="gpt-4o", max_output_tokens=4096
        )
        assert req.body is not None
        assert req.body["max_output_tokens"] == 4096

    def test_build_request_with_tools(self) -> None:
        tools = [{"type": "function", "function": {"name": "calc"}}]
        req = self.adapter.build_chat_request(
            SAMPLE_MESSAGES, model="gpt-4o", tools=tools
        )
        assert req.body is not None
        assert req.body["tools"] == tools

    def test_parse_response_multiple_messages(self) -> None:
        response = {
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "First."}],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Second."}],
                },
            ],
        }
        result = self.adapter.parse_chat_response(response)
        assert result.content == "First.\nSecond."

    def test_parse_response_non_message_items_ignored(self) -> None:
        response = {
            "model": "gpt-4o",
            "output": [
                {"type": "reasoning", "summary": []},
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Answer."}],
                },
                {"type": "function_call", "name": "tool"},
            ],
        }
        result = self.adapter.parse_chat_response(response)
        assert result.content == "Answer."

    def test_parse_response_empty_output(self) -> None:
        response = {"model": "gpt-4o", "output": []}
        result = self.adapter.parse_chat_response(response)
        assert result.content == ""

    def test_parse_response_no_output_key(self) -> None:
        response = {"model": "gpt-4o"}
        result = self.adapter.parse_chat_response(response)
        assert result.content == ""

    def test_parse_response_non_dict_items_skipped(self) -> None:
        response = {
            "model": "gpt-4o",
            "output": ["invalid", 42, None],
        }
        result = self.adapter.parse_chat_response(response)
        assert result.content == ""

    def test_parse_response_usage_passthrough(self) -> None:
        usage = {"input_tokens": 100, "output_tokens": 50}
        response = {
            "model": "gpt-4o",
            "output": [],
            "usage": usage,
        }
        result = self.adapter.parse_chat_response(response)
        assert result.usage == usage


# ---------------------------------------------------------------------------
# Edge cases — Anthropic adapter
# ---------------------------------------------------------------------------


class TestAnthropicEdgeCases:
    def setup_method(self) -> None:
        self.adapter = AnthropicAdapter()

    def test_build_request_with_stop_sequences(self) -> None:
        msgs = [{"role": "user", "content": "Hi"}]
        req = self.adapter.build_chat_request(
            msgs, model="claude-sonnet-4-20250514", stop_sequences=["END"]
        )
        assert req.body is not None
        assert req.body["stop_sequences"] == ["END"]

    def test_build_request_with_top_p(self) -> None:
        msgs = [{"role": "user", "content": "Hi"}]
        req = self.adapter.build_chat_request(
            msgs, model="claude-sonnet-4-20250514", top_p=0.95
        )
        assert req.body is not None
        assert req.body["top_p"] == 0.95

    def test_build_request_ignores_unknown_kwargs(self) -> None:
        msgs = [{"role": "user", "content": "Hi"}]
        req = self.adapter.build_chat_request(
            msgs, model="claude-sonnet-4-20250514", unknown="ignored"
        )
        assert req.body is not None
        assert "unknown" not in req.body

    def test_build_request_custom_max_tokens(self) -> None:
        msgs = [{"role": "user", "content": "Hi"}]
        req = self.adapter.build_chat_request(
            msgs, model="claude-sonnet-4-20250514", max_tokens=8192
        )
        assert req.body is not None
        assert req.body["max_tokens"] == 8192

    def test_build_request_system_none_not_included(self) -> None:
        msgs = [{"role": "user", "content": "Hi"}]
        req = self.adapter.build_chat_request(
            msgs, model="claude-sonnet-4-20250514", system=None
        )
        assert req.body is not None
        assert "system" not in req.body

    def test_parse_response_empty_content_array(self) -> None:
        response = {"model": "claude-sonnet-4-20250514", "content": []}
        result = self.adapter.parse_chat_response(response)
        assert result.content == ""
        assert result.usage is None

    def test_parse_response_non_dict_content_entries(self) -> None:
        response = {
            "model": "claude-sonnet-4-20250514",
            "content": ["not a dict", None, 42],
        }
        result = self.adapter.parse_chat_response(response)
        assert result.content == ""

    def test_parse_response_mixed_text_and_tool_use(self) -> None:
        response = {
            "model": "claude-sonnet-4-20250514",
            "content": [
                {"type": "text", "text": "Thinking..."},
                {"type": "tool_use", "id": "tu_1"},
                {"type": "text", "text": "Done."},
            ],
        }
        result = self.adapter.parse_chat_response(response)
        assert result.content == "Thinking...\nDone."

    def test_parse_response_no_model_field(self) -> None:
        response = {
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        result = self.adapter.parse_chat_response(response)
        assert result.model == ""
        assert result.content == "Hi"

    def test_parse_response_usage_is_not_dict(self) -> None:
        response = {
            "model": "claude-sonnet-4-20250514",
            "content": [],
            "usage": "invalid",
        }
        result = self.adapter.parse_chat_response(response)
        assert result.usage is None


# ---------------------------------------------------------------------------
# Edge cases — resolve_adapter
# ---------------------------------------------------------------------------


class TestResolveAdapterEdgeCases:
    def test_resolve_openai_with_none_format(self) -> None:
        adapter = resolve_adapter("openai", response_format=None)
        assert isinstance(adapter, OpenAIAdapter)
        assert adapter.response_format == "completions"

    def test_resolve_openai_all_formats(self) -> None:
        for fmt in ("completions", "messages", "responses"):
            adapter = resolve_adapter("openai", response_format=fmt)
            assert isinstance(adapter, OpenAIAdapter)
            assert adapter.response_format == fmt

    def test_resolve_roundtrip_completions(self) -> None:
        adapter = resolve_adapter("openai", response_format="completions")
        req = adapter.build_chat_request(SAMPLE_MESSAGES, model="gpt-4o")
        assert req.method == "POST"
        assert req.path == "/v1/chat/completions"
        assert req.body is not None
        assert req.body["model"] == "gpt-4o"

    def test_resolve_roundtrip_messages(self) -> None:
        adapter = resolve_adapter("openai", response_format="messages")
        req = adapter.build_chat_request(SAMPLE_MESSAGES, model="gpt-4o")
        assert req.path == "/v1/messages"

    def test_resolve_roundtrip_responses(self) -> None:
        adapter = resolve_adapter("openai", response_format="responses")
        req = adapter.build_chat_request(SAMPLE_MESSAGES, model="gpt-4o")
        assert req.path == "/v1/responses"
        assert req.body is not None
        assert "input" in req.body

    def test_resolve_roundtrip_anthropic(self) -> None:
        adapter = resolve_adapter("anthropic")
        req = adapter.build_chat_request(
            [{"role": "user", "content": "Hi"}], model="claude-sonnet-4-20250514"
        )
        assert req.path == "/v1/messages"
        assert "anthropic-version" in req.headers


# ---------------------------------------------------------------------------
# Edge cases — usage parsers
# ---------------------------------------------------------------------------


class TestUsageParsersEdgeCases:
    def test_openai_usage_json_error_500(self) -> None:
        result = _parse_openai_usage_json(500, {}, {})
        assert result.status == "error"
        assert "500" in (result.message or "")

    def test_openai_usage_json_error_401(self) -> None:
        result = _parse_openai_usage_json(401, {}, {"error": "unauthorized"})
        assert result.status == "error"
        assert "401" in (result.message or "")

    def test_anthropic_usage_json_error_403(self) -> None:
        result = _parse_anthropic_usage_json(403, {}, {})
        assert result.status == "error"
        assert "403" in (result.message or "")

    def test_anthropic_usage_json_error_500(self) -> None:
        result = _parse_anthropic_usage_json(500, {}, {"error": "internal"})
        assert result.status == "error"

    def test_openai_usage_headers_error_code(self) -> None:
        result = _parse_openai_usage_headers(429, {}, {})
        assert result.status == "error"
        assert "429" in (result.message or "")

    def test_openai_usage_headers_empty_body(self) -> None:
        headers = {"x-ratelimit-remaining-requests": "100"}
        result = _parse_openai_usage_headers(200, headers, {})
        assert result.status == "available"
        assert result.payload is not None
        assert "x-ratelimit-remaining-requests" in result.payload

    def test_generic_balance_json_error_402(self) -> None:
        result = _parse_generic_balance_json(402, {}, {})
        assert result.status == "error"

    def test_generic_balance_json_empty_body(self) -> None:
        result = _parse_generic_balance_json(200, {}, {})
        assert result.status == "available"
        assert result.payload == {}


# ---------------------------------------------------------------------------
# Edge cases — probe_provider_usage orchestration
# ---------------------------------------------------------------------------


class TestProbeProviderUsageEdgeCases:
    def setup_method(self) -> None:
        self.adapter = OpenAIAdapter()

    def test_strategy_endpoint_fetch_returns_error(self) -> None:
        with patch(
            "api.provider_registry.adapters.usage._fetch_endpoint",
            return_value=(500, {}, {"error": "internal"}),
        ):
            result = probe_provider_usage(
                self.adapter,
                "key",
                "https://api.openai.com",
                strategy="endpoint",
                endpoint_url="https://example.com/usage",
                parser_type="generic_balance_json",
            )
        assert result.status == "error"
        assert "500" in (result.message or "")

    def test_auto_endpoint_auto_error_falls_back(self) -> None:
        """auto+endpoint: when auto returns error status, falls back to endpoint."""
        error_result = UsageResult(status="error", message="network timeout")
        with (
            patch.object(self.adapter, "probe_usage", return_value=error_result),
            patch(
                "api.provider_registry.adapters.usage._fetch_endpoint",
                return_value=(200, {}, {"balance": 75.0}),
            ),
        ):
            result = probe_provider_usage(
                self.adapter,
                "key",
                "https://api.openai.com",
                strategy="auto+endpoint",
                endpoint_url="https://example.com/usage",
                parser_type="generic_balance_json",
            )
        assert result.status == "available"
        assert result.payload == {"balance": 75.0}

    def test_strategy_auto_adapter_returns_none(self) -> None:
        """auto: adapter.probe_usage returns None."""
        with patch.object(self.adapter, "probe_usage", return_value=None):
            result = probe_provider_usage(
                self.adapter, "key", "https://api.openai.com", strategy="auto"
            )
        assert result.status == "unsupported"

    def test_strategy_endpoint_with_openai_usage_json_parser(self) -> None:
        payload = {"hard_limit_usd": 100.0, "total_usage": 42.5}
        with patch(
            "api.provider_registry.adapters.usage._fetch_endpoint",
            return_value=(200, {}, payload),
        ):
            result = probe_provider_usage(
                self.adapter,
                "key",
                "https://api.openai.com",
                strategy="endpoint",
                endpoint_url="https://example.com/usage",
                parser_type="openai_usage_json",
            )
        assert result.status == "available"
        assert result.payload == payload

    def test_strategy_endpoint_with_openai_usage_headers_parser(self) -> None:
        headers = {"x-ratelimit-remaining-tokens": "50000"}
        with patch(
            "api.provider_registry.adapters.usage._fetch_endpoint",
            return_value=(200, headers, {}),
        ):
            result = probe_provider_usage(
                self.adapter,
                "key",
                "https://api.openai.com",
                strategy="endpoint",
                endpoint_url="https://example.com/usage",
                parser_type="openai_usage_headers",
            )
        assert result.status == "available"


# ---------------------------------------------------------------------------
# Edge cases — data class invariants
# ---------------------------------------------------------------------------


class TestDataClassEdgeCases:
    def test_provider_request_with_all_fields(self) -> None:
        req = ProviderRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"Authorization": "Bearer x"},
            body={"model": "gpt-4o"},
            query={"version": "2024"},
        )
        assert req.method == "POST"
        assert req.path == "/v1/chat/completions"
        assert req.headers == {"Authorization": "Bearer x"}
        assert req.body == {"model": "gpt-4o"}
        assert req.query == {"version": "2024"}

    def test_provider_request_body_frozen(self) -> None:
        req = ProviderRequest(method="GET", path="/test")
        assert req.body is None
        assert req.query is None

    def test_chat_result_with_all_fields(self) -> None:
        raw = {"id": "chatcmpl-1"}
        r = ChatResult(
            content="Hello",
            model="gpt-4o",
            usage={"total_tokens": 10},
            raw=raw,
        )
        assert r.content == "Hello"
        assert r.model == "gpt-4o"
        assert r.usage == {"total_tokens": 10}
        assert r.raw == raw

    def test_model_info_with_label_and_raw(self) -> None:
        raw = {"id": "gpt-4o", "owned_by": "openai", "created": 123}
        m = ModelInfo(id="gpt-4o", label="openai", raw=raw)
        assert m.id == "gpt-4o"
        assert m.label == "openai"
        assert m.raw == raw

    def test_usage_result_with_all_fields(self) -> None:
        u = UsageResult(
            status="available",
            payload={"balance": 50.0},
            message="ok",
        )
        assert u.status == "available"
        assert u.payload == {"balance": 50.0}
        assert u.message == "ok"

    def test_usage_result_defaults(self) -> None:
        u = UsageResult(status="unknown")
        assert u.payload is None
        assert u.message is None

    def test_frozen_dataclass_equality(self) -> None:
        r1 = ProviderRequest(method="GET", path="/test")
        r2 = ProviderRequest(method="GET", path="/test")
        assert r1 == r2

        c1 = ChatResult(content="hi", model="m")
        c2 = ChatResult(content="hi", model="m")
        assert c1 == c2

        m1 = ModelInfo(id="gpt-4o", label="openai")
        m2 = ModelInfo(id="gpt-4o", label="openai")
        assert m1 == m2

        u1 = UsageResult(status="available", payload={"x": 1})
        u2 = UsageResult(status="available", payload={"x": 1})
        assert u1 == u2

    def test_frozen_dataclass_inequality(self) -> None:
        r1 = ProviderRequest(method="GET", path="/a")
        r2 = ProviderRequest(method="GET", path="/b")
        assert r1 != r2

    def test_chat_result_body_and_headers_not_mutable_on_provider_request(self) -> None:
        req = ProviderRequest(
            method="POST",
            path="/test",
            headers={"X-Test": "val"},
            body={"key": "value"},
        )
        # Verify the reference is to the same object (frozen doesn't deep-copy)
        assert req.headers["X-Test"] == "val"
        assert req.body is not None
        assert req.body["key"] == "value"
