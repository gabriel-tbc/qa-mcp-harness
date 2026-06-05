"""Tests for build_provider dispatch. SDK-free: providers create their client
lazily, so constructing them needs no vendor SDK."""

from __future__ import annotations

import pytest

from harness.providers.base import Provider
from harness.providers.openai_compat import DEFAULT_OLLAMA_BASE_URL
from harness.providers.registry import build_provider


def test_anthropic_requires_key():
    with pytest.raises(ValueError, match="anthropic"):
        build_provider("anthropic", "claude-x", api_key=None)


def test_anthropic_builds_with_key():
    p = build_provider("anthropic", "claude-x", api_key="sk-test")
    assert isinstance(p, Provider)
    assert p.name == "anthropic"
    assert p.model == "claude-x"


def test_openai_requires_key():
    with pytest.raises(ValueError, match="openai"):
        build_provider("openai", "gpt-x", api_key=None)


def test_gemini_requires_key():
    with pytest.raises(ValueError, match="gemini"):
        build_provider("gemini", "gemini-x", api_key=None)


def test_gemini_builds_with_key():
    p = build_provider("gemini", "gemini-x", api_key="g-test")
    assert isinstance(p, Provider)
    assert p.name == "gemini"
    assert p.model == "gemini-x"


def test_ollama_needs_no_key_and_defaults_base_url():
    p = build_provider("ollama", "qwen2.5")
    assert isinstance(p, Provider)
    assert p.name == "ollama"
    # The dummy key + default local URL are wired internally.
    assert p._base_url == DEFAULT_OLLAMA_BASE_URL  # type: ignore[attr-defined]


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        build_provider("mistral-cloud", "m", api_key="x")


def test_temperature_threads_through_to_provider():
    """The experiment-level temperature reaches the adapter (verifiable without
    a network call: it's stored on the constructed provider instance)."""
    p = build_provider("anthropic", "claude-x", api_key="k", temperature=0.7)
    assert p._temperature == 0.7  # type: ignore[attr-defined]

    p2 = build_provider("ollama", "qwen2.5", temperature=0.2)
    assert p2._temperature == 0.2  # type: ignore[attr-defined]
