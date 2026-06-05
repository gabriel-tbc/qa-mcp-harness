"""GeminiProvider — Google Gemini adapter.

Gemini's wire shape is a THIRD distinct format (neither Anthropic's content
blocks nor OpenAI's tool_calls): tools are nested under a single
`function_declarations` list, and the response is `candidates[0].content.parts[]`
where each part is EITHER a `function_call` (`.name` + `.args` dict) OR `.text`.
Tokens live on `usage_metadata`. That this adapter is the *only* thing that
changes to support Gemini — runner, matching, report untouched — is the proof
the neutral core works.

Two pure, SDK-free functions are unit-tested without the SDK:
    _tools_to_gemini        neutral ToolSpec → Gemini function-declarations
    _parse_gemini_response  native response  → neutral ModelResponse

Only `complete()` touches the SDK (lazy). The exact `google-genai` call surface
is implementation detail (it has churned across versions); if it drifts it fails
loudly and is fixed against the SDK docs — the tested value is in the pure
conversion + parsing.
"""

from __future__ import annotations

import time
from typing import Any

from harness.providers.base import ModelResponse, ToolCall, ToolResult, ToolSpec, Usage


def _tools_to_gemini(tools: list[ToolSpec]) -> list[dict]:
    """Neutral ToolSpec → Gemini tools: a single Tool holding all function
    declarations (Gemini groups them under one `function_declarations` list)."""
    return [
        {
            "function_declarations": [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in tools
            ]
        }
    ]


def _parse_gemini_response(response: Any) -> ModelResponse:
    """Map a Gemini generate_content response (duck-typed) → neutral ModelResponse."""
    tool_calls: list[ToolCall] = []
    text_parts: list[str] = []
    stop_reason = None

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        cand = candidates[0]
        fr = getattr(cand, "finish_reason", None)
        stop_reason = str(fr) if fr is not None else None
        content = getattr(cand, "content", None)
        parts = (getattr(content, "parts", None) or []) if content is not None else []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc is not None:
                tool_calls.append(
                    ToolCall(getattr(fc, "name", ""), dict(getattr(fc, "args", {}) or {}))
                )
            else:
                txt = getattr(part, "text", None)
                if txt:
                    text_parts.append(txt)

    um = getattr(response, "usage_metadata", None)
    usage = Usage(
        input_tokens=getattr(um, "prompt_token_count", None),
        output_tokens=getattr(um, "candidates_token_count", None),
    )
    return ModelResponse(
        tool_calls=tool_calls,
        final_text="".join(text_parts),
        stop_reason=stop_reason,
        usage=usage,
        raw=response,
    )


class GeminiProvider:
    """Provider for Google's Gemini models. Satisfies the `Provider` protocol."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        name: str = "gemini",
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
            from google import genai  # lazy: only needed to actually call the model

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        tools: list[ToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        client = self._client_or_create()
        # Gemini takes `system_instruction` and `temperature` inside the config
        # dict. We only set them when provided.
        config: dict = {
            "tools": _tools_to_gemini(tools),
            "max_output_tokens": self._max_tokens,
        }
        if system_prompt:
            config["system_instruction"] = system_prompt
        if self._temperature is not None:
            config["temperature"] = self._temperature
        t0 = time.perf_counter()
        try:
            # SDK call surface (SURFACE / implementation detail): google-genai
            # coerces a dict config + dict tools. If a version rejects it, this
            # fails loudly and is adjusted against the SDK docs.
            resp = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 — record, don't abort the dataset
            return ModelResponse(
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        out = _parse_gemini_response(resp)
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
        # Layer 4 multi-turn for Gemini: future slice.
        return ModelResponse(error="Gemini multi-turn not implemented (Layer 4 pending).")
