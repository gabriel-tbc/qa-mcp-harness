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
        tool_calls.append(ToolCall(tc.function.name, args, id=getattr(tc, "id", None)))
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


def _assistant_message(tool_calls: list[ToolCall], content: str) -> dict:
    """Reconstruct the assistant turn the model produced, in OpenAI wire form.
    `content` may be empty when the turn was pure tool_calls."""
    msg: dict = {"role": "assistant", "content": content or ""}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id or f"call_{i}",
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for i, tc in enumerate(tool_calls)
        ]
    return msg


def _tool_messages(results: list[ToolResult]) -> list[dict]:
    """One `role=tool` message per executed call; `tool_call_id` correlates it
    with the assistant's tool_call that requested it."""
    return [
        {
            "role": "tool",
            "tool_call_id": r.call_id or f"call_{i}",
            "content": r.content_text,
        }
        for i, r in enumerate(results)
    ]


def _build_continue_messages(
    history: list[dict],
    tool_results: list[ToolResult],
    *,
    system_prompt: str | None = None,
) -> list[dict]:
    """Render the loop's NEUTRAL agent history + the freshest tool results into
    the OpenAI `messages` list. SDK-free, so it is unit-tested without `openai`.

    Neutral history items (built by `agent_loop.run_agent_turn`):
        {"role": "user", "content": str}
        {"role": "assistant", "tool_calls": list[ToolCall], "content": str}
        {"role": "tool", "results": list[ToolResult]}
    `tool_results` is the latest batch, not yet folded into `history`.
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for turn in history:
        role = turn.get("role")
        if role == "user":
            messages.append({"role": "user", "content": turn.get("content", "")})
        elif role == "assistant":
            messages.append(
                _assistant_message(turn.get("tool_calls") or [], turn.get("content") or "")
            )
        elif role == "tool":
            messages.extend(_tool_messages(turn.get("results") or []))
    messages.extend(_tool_messages(tool_results))
    return messages


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

    def _create_and_parse(self, messages: list[dict], tools: list[ToolSpec]) -> ModelResponse:
        """Issue one chat-completions call and map it to a neutral ModelResponse.
        Errors and latency are captured on the response, never raised — one bad
        turn degrades to a recorded failure instead of aborting the dataset."""
        client = self._client_or_create()
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

    def complete(
        self,
        prompt: str,
        tools: list[ToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        # OpenAI-style chat: the system prompt is the first message with
        # role="system"; the user prompt follows.
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._create_and_parse(messages, tools)

    def continue_with_tool_results(
        self,
        history: list[dict],
        tool_results: list[ToolResult],
        tools: list[ToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        # Layer 4 multi-turn: rebuild the whole conversation from the loop's
        # neutral history + the fresh tool results, then ask for the next move.
        messages = _build_continue_messages(history, tool_results, system_prompt=system_prompt)
        return self._create_and_parse(messages, tools)
