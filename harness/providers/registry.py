"""build_provider — construct the right Provider by name.

SDK-free at import time: the adapter modules import their vendor SDK lazily, so
importing the registry (and constructing a provider) never requires the SDK —
only an actual `complete()` call does.
"""

from __future__ import annotations

from harness.providers.anthropic import AnthropicProvider
from harness.providers.base import Provider
from harness.providers.gemini import GeminiProvider
from harness.providers.openai_compat import DEFAULT_OLLAMA_BASE_URL, OpenAIProvider

SUPPORTED = ("anthropic", "openai", "gemini", "ollama")


def build_provider(
    provider: str,
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
) -> Provider:
    """Return a Provider for `provider`/`model`.

    - 'anthropic' → paid API (needs `api_key`).
    - 'openai'    → paid API (needs `api_key`); honors `base_url` if given.
    - 'gemini'    → paid API (needs `api_key`).
    - 'ollama'    → local, free (dummy key; defaults to the local Ollama URL).

    `temperature` lives at the experiment level — same value for all cases in
    the run, so accuracy is honestly comparable. None = each provider's default.
    """
    if provider == "anthropic":
        if not api_key:
            raise ValueError("Provider 'anthropic' needs an API key (set ANTHROPIC_API_KEY).")
        return AnthropicProvider(model, api_key, temperature=temperature)
    if provider == "openai":
        if not api_key:
            raise ValueError("Provider 'openai' needs an API key (set OPENAI_API_KEY).")
        return OpenAIProvider(
            model, api_key, base_url=base_url, name="openai", temperature=temperature
        )
    if provider == "gemini":
        if not api_key:
            raise ValueError("Provider 'gemini' needs an API key (set GEMINI_API_KEY).")
        return GeminiProvider(model, api_key, temperature=temperature)
    if provider == "ollama":
        return OpenAIProvider(
            model,
            api_key="ollama",
            base_url=base_url or DEFAULT_OLLAMA_BASE_URL,
            name="ollama",
            temperature=temperature,
        )
    raise ValueError(f"Unknown provider: {provider!r} (supported: {', '.join(SUPPORTED)}).")
