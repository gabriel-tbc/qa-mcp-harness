"""AnthropicProvider — the Claude adapter.

Two pure, SDK-free functions do the translation and are unit-tested without
`anthropic` installed:
    _tools_to_anthropic    neutral ToolSpec  → Anthropic tool format
    _parse_anthropic_response  native response → neutral ModelResponse

Only `complete()` touches the SDK: imported lazily, client created on first use,
so constructing the provider (and the whole registry) needs no SDK. API errors
are captured into `ModelResponse.error` rather than raised, so one bad run
degrades to a recorded failure instead of aborting a dataset.
"""

from __future__ import annotations

import time
from typing import Any

from harness.providers.base import ModelResponse, ToolCall, ToolResult, ToolSpec, Usage


def _tools_to_anthropic(tools: list[ToolSpec]) -> list[dict]:
    """Neutral ToolSpec → Anthropic `tools` shape (`input_schema` is raw JSON Schema)."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in tools
    ]


def _parse_anthropic_response(response: Any) -> ModelResponse:
    """Map an Anthropic Messages response (duck-typed) → neutral ModelResponse.

    Anthropic returns a list of content blocks; tool calls are `type="tool_use"`
    blocks with a `.input` dict, text is `type="text"` blocks. Tokens live on
    `.usage.{input,output}_tokens`.
    """
    tool_calls: list[ToolCall] = []
    text_parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "tool_use":
            tool_calls.append(ToolCall(block.name, dict(block.input)))
        elif btype == "text":
            text_parts.append(getattr(block, "text", ""))
    usage_obj = getattr(response, "usage", None)
    usage = Usage(
        input_tokens=getattr(usage_obj, "input_tokens", None),
        output_tokens=getattr(usage_obj, "output_tokens", None),
    )
    return ModelResponse(
        tool_calls=tool_calls,
        final_text="".join(text_parts),
        stop_reason=getattr(response, "stop_reason", None),
        usage=usage,
        raw=response,
    )


class AnthropicProvider:
    """Provider for Anthropic's Claude models. Satisfies the `Provider` protocol."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        name: str = "anthropic",
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = None

    def _client_or_create(self):
        if self._client is None:
            import anthropic  # lazy: only needed when actually calling the model

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        tools: list[ToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        client = self._client_or_create()
        # Anthropic takes `system` and `temperature` as top-level fields. We
        # only forward them when set, so behaviour without them is unchanged.
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "tools": _tools_to_anthropic(tools),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        t0 = time.perf_counter()
        try:
            resp = client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — record, don't abort the dataset
            return ModelResponse(
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        out = _parse_anthropic_response(resp)
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
        # Layer 4 multi-turn for Anthropic is a future slice (needs message
        # history + tool_result blocks). Layer 3 doesn't reach this method.
        return ModelResponse(error="Anthropic multi-turn not implemented (Layer 4 pending).")
