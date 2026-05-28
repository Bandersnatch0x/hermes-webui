"""Usage probing orchestration for provider adapters.

Supports four strategies:
- ``auto``: adapter-native probing only
- ``endpoint``: configured endpoint + parser only
- ``auto+endpoint``: try adapter-native first, fall back to configured endpoint
- ``none``: no probing
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable

from api.provider_registry.adapters.base import ProviderAdapter, UsageResult

# Parser type constants
_PARSER_OPENAI_USAGE_JSON = "openai_usage_json"
_PARSER_ANTHROPIC_USAGE_JSON = "anthropic_usage_json"
_PARSER_OPENAI_USAGE_HEADERS = "openai_usage_headers"
_PARSER_GENERIC_BALANCE_JSON = "generic_balance_json"
_PARSER_NONE = "none"

_VALID_STRATEGIES = frozenset({"auto", "endpoint", "auto+endpoint", "none"})
_VALID_PARSER_TYPES = frozenset({
    _PARSER_OPENAI_USAGE_JSON,
    _PARSER_ANTHROPIC_USAGE_JSON,
    _PARSER_OPENAI_USAGE_HEADERS,
    _PARSER_GENERIC_BALANCE_JSON,
    _PARSER_NONE,
})

# Maps parser type name to a callable that takes (status_code, headers, body_json)
# and returns a UsageResult.
_PARSER_MAP: dict[str, Callable[[int, dict[str, str], dict[str, Any]], UsageResult]] = {}


def _register_parser(
    name: str,
) -> Callable[[Callable[[int, dict[str, str], dict[str, Any]], UsageResult]], Callable[[int, dict[str, str], dict[str, Any]], UsageResult]]:
    def decorator(fn: Callable[[int, dict[str, str], dict[str, Any]], UsageResult]) -> Callable[[int, dict[str, str], dict[str, Any]], UsageResult]:
        _PARSER_MAP[name] = fn
        return fn
    return decorator


@_register_parser(_PARSER_OPENAI_USAGE_JSON)
def _parse_openai_usage_json(
    status_code: int,
    headers: dict[str, str],
    body: dict[str, Any],
) -> UsageResult:
    """Parse an OpenAI-style usage/billing JSON payload.

    Looks for common fields like ``hard_limit_usd``, ``system_hard_limit``,
    ``total_usage``, etc.
    """
    if status_code >= 400:
        return UsageResult(
            status="error",
            message=f"HTTP {status_code}",
        )
    return UsageResult(status="available", payload=body, message=None)


@_register_parser(_PARSER_ANTHROPIC_USAGE_JSON)
def _parse_anthropic_usage_json(
    status_code: int,
    headers: dict[str, str],
    body: dict[str, Any],
) -> UsageResult:
    """Parse an Anthropic-style usage JSON payload.

    Looks for fields like ``current_usage``, ``limit``, etc.
    """
    if status_code >= 400:
        return UsageResult(status="error", message=f"HTTP {status_code}")
    return UsageResult(status="available", payload=body, message=None)


@_register_parser(_PARSER_OPENAI_USAGE_HEADERS)
def _parse_openai_usage_headers(
    status_code: int,
    headers: dict[str, str],
    body: dict[str, Any],
) -> UsageResult:
    """Extract usage information from OpenAI-style response headers.

    Looks for ``x-ratelimit-*`` headers.
    """
    if status_code >= 400:
        return UsageResult(status="error", message=f"HTTP {status_code}")

    lower_headers = {k.lower(): v for k, v in headers.items()}
    usage_headers: dict[str, str] = {}
    for key, val in lower_headers.items():
        if key.startswith("x-ratelimit-"):
            usage_headers[key] = val

    if not usage_headers:
        return UsageResult(
            status="unknown",
            message="No rate-limit headers found",
        )
    return UsageResult(status="available", payload=usage_headers, message=None)


@_register_parser(_PARSER_GENERIC_BALANCE_JSON)
def _parse_generic_balance_json(
    status_code: int,
    headers: dict[str, str],
    body: dict[str, Any],
) -> UsageResult:
    """Parse a generic balance/credit JSON response.

    Passes through the entire payload for the caller to interpret.
    """
    if status_code >= 400:
        return UsageResult(status="error", message=f"HTTP {status_code}")
    return UsageResult(status="available", payload=body, message=None)


def _fetch_endpoint(
    url: str,
    api_key: str,
    timeout: float = 10.0,
) -> tuple[int, dict[str, str], dict[str, Any]]:
    """Fetch a URL with Bearer auth.

    Returns (status_code, response_headers, parsed_json_body).
    Falls back to empty dict on non-JSON responses.
    """
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_headers = dict(resp.headers)
            raw_body = resp.read().decode("utf-8")
            try:
                body = json.loads(raw_body)
            except (json.JSONDecodeError, ValueError):
                body = {}
            return resp.status, raw_headers, body
    except urllib.error.HTTPError as exc:
        raw_headers = dict(exc.headers) if hasattr(exc, "headers") else {}
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:
            body = {}
        return exc.code, raw_headers, body
    except Exception as exc:
        return 0, {}, {"error": str(exc)}


def probe_provider_usage(
    adapter: ProviderAdapter,
    api_key: str,
    base_url: str,
    strategy: str = "auto",
    endpoint_url: str | None = None,
    parser_type: str | None = None,
) -> UsageResult:
    """Orchestrate usage probing according to the configured strategy.

    Parameters
    ----------
    adapter:
        The provider adapter instance.
    api_key:
        API key for authentication.
    base_url:
        Provider base URL.
    strategy:
        One of ``auto``, ``endpoint``, ``auto+endpoint``, ``none``.
    endpoint_url:
        Explicit usage endpoint URL (required for ``endpoint`` and
        ``auto+endpoint`` strategies).
    parser_type:
        Parser type for endpoint responses (required for ``endpoint``
        and ``auto+endpoint`` strategies).

    Returns
    -------
    UsageResult
        The probe result.
    """
    if strategy not in _VALID_STRATEGIES:
        return UsageResult(
            status="error",
            message=f"Unknown usage strategy: {strategy!r}",
        )

    if strategy == "none":
        return UsageResult(status="unsupported", message="Usage probing disabled")

    if strategy == "auto":
        return _probe_auto(adapter, api_key, base_url)

    if strategy == "endpoint":
        return _probe_endpoint(api_key, endpoint_url, parser_type)

    # strategy == "auto+endpoint"
    auto_result = _probe_auto(adapter, api_key, base_url)
    if auto_result is not None and auto_result.status == "available":
        return auto_result
    endpoint_result = _probe_endpoint(api_key, endpoint_url, parser_type)
    return endpoint_result


def _probe_auto(
    adapter: ProviderAdapter,
    api_key: str,
    base_url: str,
) -> UsageResult:
    """Try adapter-native usage probing."""
    if not adapter.native_usage_probe_supported():
        return UsageResult(
            status="unsupported",
            message="Adapter has no native usage probing",
        )
    result = adapter.probe_usage(api_key, base_url)
    if result is None:
        return UsageResult(
            status="unsupported",
            message="Adapter returned no usage data",
        )
    return result


def _probe_endpoint(
    api_key: str,
    endpoint_url: str | None,
    parser_type: str | None,
) -> UsageResult:
    """Probe a configured usage endpoint with the specified parser."""
    if not endpoint_url:
        return UsageResult(
            status="error",
            message="endpoint_url is required for endpoint strategy",
        )
    if not parser_type or parser_type == _PARSER_NONE:
        return UsageResult(
            status="error",
            message="parser_type is required for endpoint strategy",
        )
    if parser_type not in _VALID_PARSER_TYPES:
        return UsageResult(
            status="error",
            message=f"Unknown parser_type: {parser_type!r}",
        )
    parser_fn = _PARSER_MAP.get(parser_type)
    if parser_fn is None:
        return UsageResult(
            status="error",
            message=f"No parser registered for type: {parser_type!r}",
        )

    status_code, headers, body = _fetch_endpoint(endpoint_url, api_key)
    return parser_fn(status_code, headers, body)
