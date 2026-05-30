"""Layer 4 scaffold — the agent loop (NOT YET IMPLEMENTED).

Layer 3 needs only the model's FIRST tool choice, so its runner captures that
one `tool_use` and stops (see `harness/eval/runner.py`). Layer 4 is different:
to judge the model's final free-text answer, the tool has to actually run and
its result be fed back, so the model can compose a real answer. That is the
full agent loop:

    prompt + tools
      → model returns tool_use block(s)
      → execute each on the MCP via `session.call_tool(name, args)`
      → return the tool_result(s) to the model
      → repeat until the model returns a final text block
      → capture that text (the thing Layer 4 actually scores)

This module will own that loop. It is intentionally a stub: Layer 4 is
design-only for now (see `tests/layer4_output/README.md`). The shapes below fix
the contract so the report layer and oracles can be designed against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """One tool the model invoked during the loop, with the result it got back.
    Layer 4 reports the whole trail so a wrong answer can be traced to a bad
    tool result vs a bad summary."""

    name: str
    args: dict
    result_text: str


@dataclass
class AgentOutcome:
    """The product of one full agent turn: the final free-text answer plus the
    trail of tool calls that produced it."""

    final_text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


async def run_agent_turn(
    session: Any, model: Any, prompt: str, tools: list[dict]
) -> AgentOutcome:
    """Drive `model` + the MCP `session` to a final text answer.

    `session` — an initialized MCP ClientSession; it executes the tools.
    `model`   — a callable advancing the conversation one assistant message at a
                time (returning tool_use blocks or a final text block).
    `tools`   — the MCP tools in Anthropic shape (see `schema_convert`).

    NOT IMPLEMENTED — Layer 4 is design-only for now.
    """
    raise NotImplementedError("Layer 4 agent loop is scaffolded but not yet implemented.")
