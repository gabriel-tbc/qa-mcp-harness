"""Neutral, provider-independent types and the Provider contract.

This module is the heart of the harness's model-agnosticism. The eval engine,
the runner, the report — none of them know whether a turn ran on Claude, GPT,
Gemini, or a local Ollama model. They speak only the neutral types here:

    ToolSpec       a tool, provider-independently (≈ the MCP tool form)
    ToolCall       one tool invocation the model produced
    Usage          token accounting, normalized across providers
    ModelResponse  the full result of one model turn, provider-independently

A provider adapter (in a sibling module) does exactly two translations:
    1. ToolSpec  → its own request format
    2. its native response → ModelResponse

`Provider` is a *Protocol* (structural typing): an adapter satisfies it by
*having* the right attributes/methods, with no inheritance required.

Pure stdlib — no vendor SDKs imported here, so this module (and everything that
depends only on it) is testable without the `llm` extra or a network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolSpec:
    """A tool definition, provider-independently.

    `parameters` is a JSON Schema object — exactly what an MCP tool exposes as
    its `inputSchema`. Because MCP already speaks JSON Schema, this neutral form
    *is* the MCP form; adapters convert neutral → their own wire format (one hop
    from the source of truth, never provider → provider).
    """

    name: str
    description: str
    parameters: dict  # JSON Schema

    @classmethod
    def from_mcp(cls, mcp_tool: Any) -> "ToolSpec":
        """Build a ToolSpec from an MCP tool (duck-typed: .name/.description/.inputSchema)."""
        return cls(
            name=mcp_tool.name,
            description=(mcp_tool.description or "").strip(),
            parameters=mcp_tool.inputSchema,
        )

    @classmethod
    def from_mcp_list(cls, mcp_tools: Any) -> list["ToolSpec"]:
        return [cls.from_mcp(t) for t in mcp_tools]


@dataclass
class ToolCall:
    """One tool invocation the model produced, in neutral form.

    `arguments` is always a parsed dict here — adapters are responsible for
    turning provider-specific encodings (e.g. OpenAI's JSON *string*) into a dict.

    `id` carries the provider-specific tool_use/tool_call identifier when the
    SDK supplies one (Anthropic, OpenAI), so adapters can correlate the
    `ToolResult` we feed back with the original call. Gemini does not use ids;
    that's fine, it stays None.
    """

    name: str
    arguments: dict = field(default_factory=dict)
    id: str | None = None


@dataclass
class ToolResult:
    """The result of executing one ToolCall on the MCP, in neutral form.

    Adapters embed this into the next provider request in their native format.
    `is_error` reflects the MCP-level result (e.g. `result.isError`), not a
    transport failure of our own.
    """

    call_id: str | None
    name: str
    content_text: str
    is_error: bool = False


@dataclass
class Usage:
    """Token accounting, normalized across providers (different vendors name
    these differently; adapters map them here)."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ModelResponse:
    """The full result of one model turn, provider-independently.

    Every adapter maps its native response INTO this shape. Capturing it richly
    is what lets reports answer questions later (hallucinations need
    `final_text`; response size needs `usage`; latency needs `latency_ms`)
    without a structural limit. `raw` keeps the original response for deep
    debugging and is never relied on by the core.
    """

    tool_calls: list[ToolCall] = field(default_factory=list)
    final_text: str = ""
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    latency_ms: float | None = None
    error: str | None = None
    raw: Any = None

    @property
    def first_tool(self) -> ToolCall | None:
        """The first tool the model chose, or None if it called no tool.

        Layer 3 cares only about this first choice; Layer 4 walks `tool_calls`
        and `final_text` across a full agent loop.
        """
        return self.tool_calls[0] if self.tool_calls else None

    def to_dict(self) -> dict:
        """Report-friendly view. `raw` is omitted (can be huge / non-serializable);
        adapters that want to persist a tail of it can stash it elsewhere."""
        return {
            "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in self.tool_calls],
            "final_text": self.final_text,
            "stop_reason": self.stop_reason,
            "usage": self.usage.to_dict(),
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@runtime_checkable
class Provider(Protocol):
    """The contract every model adapter satisfies (structural — no inheritance).

    Attributes:
        name   provider identifier, e.g. "anthropic", "openai", "ollama".
        model  the specific model id, e.g. "claude-sonnet-4", "qwen2.5".

    Methods:
        complete(prompt, tools) -> ModelResponse
            Run one model turn: given a user prompt and the available tools
            (neutral ToolSpecs), return a neutral ModelResponse. Implementations
            should set `latency_ms` and capture `error` rather than raising, so a
            single bad run degrades to a recorded failure instead of aborting a
            whole dataset.

    (Layer 4 will add a multi-turn path + provider-specific tool-result feeding;
    intentionally not part of this contract yet — see the project plan.)
    """

    name: str
    model: str

    def complete(
        self,
        prompt: str,
        tools: list[ToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> ModelResponse: ...

    def continue_with_tool_results(
        self,
        history: list[dict],
        tool_results: list["ToolResult"],
        tools: list[ToolSpec],
        *,
        system_prompt: str | None = None,
    ) -> ModelResponse: ...
    """Advance an agent loop: feed back the results of the tools that the
    previous `complete()` (or this very method) asked to call, and let the
    model produce the next message — possibly more tool calls, possibly the
    final text. Required for Layer 4; Layer 3 providers can leave it as the
    default-from-Protocol no-op."""
