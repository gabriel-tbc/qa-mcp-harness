"""OpenAIProvider — adapter for OpenAI and any OpenAI-compatible endpoint.

The same wire format serves OpenAI's own models and a local Ollama server
(Ollama deliberately exposes an OpenAI-compatible API); choose between them with
`base_url`. One adapter, two services.

Two pure, SDK-free functions are unit-tested without `openai` installed:
    _tools_to_openai        neutral ToolSpec → OpenAI function shape
    _parse_openai_response  native response  → neutral ModelResponse

Only `complete()` touches the SDK (lazy). Note OpenAI returns tool-call
arguments as a JSON *string*, so the parser json-loads them (degrading to {} on
malformed JSON). Tokens live on `.usage.{prompt,completion}_tokens`.
"""

from __future__ import annotations

import json
import time
from typing import Any

from harness.providers.base import ModelResponse, ToolCall, ToolResult, ToolSpec, Usage

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


def _tools_to_openai(tools: list[ToolSpec]) -> list[dict]:
    """Neutral ToolSpec → OpenAI function-calling shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _parse_openai_response(response: Any) -> ModelResponse:
    """Map an OpenAI chat-completions response (duck-typed) → neutral ModelResponse."""
    choice = response.choices[0]
    msg = choice.message
    tool_calls: list[ToolCall] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        tool_calls.append(ToolCall(tc.function.name, args))
    usage_obj = getattr(response, "usage", None)
    usage = Usage(
        input_tokens=getattr(usage_obj, "prompt_tokens", None),
        output_tokens=getattr(usage_obj, "completion_tokens", None),
    )
    return ModelResponse(
        tool_calls=tool_calls,
        final_text=getattr(msg, "content", None) or "",
        stop_reason=getattr(choice, "finish_reason", None),
        usage=usage,
        raw=response,
    )


class OpenAIProvider:
    """Provider for OpenAI and OpenAI-compatible servers (incl. Ollama)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str | None = None,
        name: str = "openai",
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = None

    def _client_or_create(self):
        if self._client is None:
            from openai import OpenAI  # lazy

            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def complete(
        self,
        prompt: str,
        tools: list[ToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        client = self._client_or_create()
        # OpenAI-style chat: the system prompt is the first message with
        # role="system"; temperature is a top-level param.
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "tools": _tools_to_openai(tools),
            "tool_choice": "auto",
            "messages": messages,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — record, don't abort the dataset
            return ModelResponse(
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        out = _parse_openai_response(resp)
        out.latency_ms = (time.perf_counter() - t0) * 1000
        return out

    def continue_with_tool_results(
        self,
        history: list[dict],
        tool_results: list[ToolResult],
        tools: list[ToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        # Layer 4 multi-turn for OpenAI/Ollama: future slice.
        return ModelResponse(error="OpenAI-compat multi-turn not implemented (Layer 4 pending).")
