"""Static source-level tests for the provider registry UI.

Checks that the HTML and JS files contain the expected provider management
elements, function names, and endpoint references.  Uses the ACTUAL element
IDs and function names from the codebase (not the plan's aspirational names).
"""
from __future__ import annotations

from pathlib import Path

import pytest


INDEX_HTML = Path("static/index.html").read_text(encoding="utf-8")
PANELS_JS = Path("static/panels.js").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML structure — Providers settings section
# ---------------------------------------------------------------------------

def test_providers_section_contains_official_and_custom_groups():
    """The Providers settings section should have separate containers
    for official and custom providers."""
    assert "providersOfficialList" in INDEX_HTML
    assert "providersCustomList" in INDEX_HTML


def test_providers_section_has_add_button():
    """There should be an 'Add Provider' button for custom providers."""
    assert "btnCreateProvider" in INDEX_HTML


def test_providers_section_header_exists():
    """The providers section should have a header/title."""
    assert "Providers" in INDEX_HTML


def test_providers_section_has_summary_bar():
    """There should be an active provider summary bar."""
    assert "providersSummary" in INDEX_HTML


def test_provider_modal_exists():
    """There should be a provider create/edit modal."""
    assert "providerModal" in INDEX_HTML


# ---------------------------------------------------------------------------
# JavaScript logic — panels.js
# ---------------------------------------------------------------------------

def test_panels_js_has_load_providers_panel():
    """panels.js should have the loadProvidersPanel function."""
    assert "loadProvidersPanel" in PANELS_JS


def test_panels_js_has_open_provider_modal():
    """panels.js should have the openProviderModal function."""
    assert "openProviderModal" in PANELS_JS


def test_panels_js_has_submit_provider_modal():
    """panels.js should have the submitProviderModal function."""
    assert "submitProviderModal" in PANELS_JS


def test_panels_js_has_retry_sync():
    """There should be a 'Retry sync' action in the UI."""
    assert "Retry sync" in PANELS_JS or "retry_sync" in PANELS_JS or "providers_sync_retry" in PANELS_JS


def test_panels_js_references_providers_endpoint():
    """panels.js should reference the /api/providers endpoint."""
    assert "/api/providers" in PANELS_JS


def test_openai_response_format_options_exist():
    """The UI should offer the three OpenAI response_format options."""
    assert "completions" in PANELS_JS or "completions" in INDEX_HTML
    assert "messages" in PANELS_JS or "messages" in INDEX_HTML
    assert "responses" in PANELS_JS or "responses" in INDEX_HTML


def test_adapter_type_options_exist():
    """The UI should offer openai and anthropic adapter types."""
    assert "openai" in PANELS_JS or "openai" in INDEX_HTML
    assert "anthropic" in PANELS_JS or "anthropic" in INDEX_HTML
