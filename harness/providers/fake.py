"""FakeProvider — a scripted Provider for tests.

Lets the eval engine, runner, and report be exercised end-to-end with NO model,
NO network, and NO API key. It satisfies the `Provider` protocol structurally.

Use the constructors for the common cases:
    FakeProvider.always_calls("qa_list_runs")                 # always picks a tool
    FakeProvider.always_calls("qa_get_run", {"params": {...}}) # tool + args
    FakeProvider.always_text("the answer is 1")               # no tool, final text
    FakeProvider(responder=lambda prompt, tools: ModelResponse(...))  # custom
    FakeProvider.scripted([resp1, resp2, ...])                # multi-turn (Layer 4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from harness.providers.base import ModelResponse, ToolCall, ToolResult, ToolSpec, Usage

Responder = Callable[[str, list[ToolSpec]], ModelResponse]


@dataclass
class FakeProvider:
    name: str = "fake"
    model: str = "fake-1"
    responder: Responder | None = None
    # Optional script of responses for multi-turn agent loops (Layer 4).
    # Each call to complete() or continue_with_tool_results() pops the next one.
    script: list[ModelResponse] = field(default_factory=list)
    _script_idx: int = 0

    def complete(
        self,
        prompt: str,
        tools: list[ToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        if self.script:
            return self._pop_script()
        if self.responder is not None:
            return self.responder(prompt, tools)
        return ModelResponse()

    def continue_with_tool_results(
        self,
        history: list[dict],
        tool_results: list[ToolResult],
        tools: list[ToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        if self.script:
            return self._pop_script()
        # No script: the loop is over — return empty final_text.
        return ModelResponse(stop_reason="end_turn")

    def _pop_script(self) -> ModelResponse:
        if self._script_idx >= len(self.script):
            return ModelResponse(stop_reason="end_turn")
        resp = self.script[self._script_idx]
        self._script_idx += 1
        return resp

    # ── constructors for common scripted behaviours ──────────────────────────

    @classmethod
    def always_calls(
        cls,
        tool: str,
        arguments: dict | None = None,
        *,
        usage: Usage | None = None,
        latency_ms: float | None = None,
        name: str = "fake",
        model: str = "fake-1",
    ) -> "FakeProvider":
        """A provider that always invokes `tool` with `arguments`."""
        resp = ModelResponse(
            tool_calls=[ToolCall(tool, arguments or {})],
            usage=usage or Usage(),
            latency_ms=latency_ms,
            stop_reason="tool_use",
        )
        return cls(name=name, model=model, responder=lambda p, t: resp)

    @classmethod
    def always_text(
        cls, text: str, *, name: str = "fake", model: str = "fake-1"
    ) -> "FakeProvider":
        """A provider that always answers with `text` and no tool call."""
        resp = ModelResponse(final_text=text, stop_reason="end_turn")
        return cls(name=name, model=model, responder=lambda p, t: resp)

    @classmethod
    def scripted(
        cls, responses: list[ModelResponse], *, name: str = "fake", model: str = "fake-1"
    ) -> "FakeProvider":
        """A provider that returns `responses` in order across complete() and
        continue_with_tool_results() — for multi-turn Layer 4 tests."""
        return cls(name=name, model=model, script=list(responses))
