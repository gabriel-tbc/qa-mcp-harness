"""Layer 4 — the agent loop.

The minimum-viable real loop:

    1. provider.complete(prompt, tools) returns either:
       - tool_calls  → execute each on the MCP, collect ToolResults, continue.
       - final_text  → done; return the captured trail.
    2. Hard cap on rounds, so a misbehaving model can't loop forever.
    3. Errors anywhere are recorded in `AgentOutcome.error`, not raised:
       a Layer 4 case must degrade to a recorded failure, not abort the dataset.

The loop knows nothing about which provider it is talking to — it only speaks
neutral types (ToolSpec, ToolCall, ToolResult, ModelResponse). That is the
whole point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.providers.base import ModelResponse, Provider, ToolCall, ToolResult, ToolSpec

DEFAULT_MAX_ROUNDS = 6


@dataclass
class ToolCallTrace:
    """One executed tool call: what the model asked, what came back."""

    name: str
    arguments: dict
    result_text: str
    is_error: bool


@dataclass
class AgentOutcome:
    """The product of one full agent turn: the final answer plus everything that
    happened to produce it. Layer 4 reports render this verbatim."""

    final_text: str = ""
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    rounds: int = 0
    stop_reason: str | None = None
    error: str | None = None
    # Token/latency totals for the whole loop (sums across model turns).
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None


def _result_text(call_result: Any) -> tuple[str, bool]:
    """Read a CallToolResult into (text, is_error). MCP returns content blocks
    and an isError flag; we concatenate the text and respect the flag."""
    parts: list[str] = []
    for block in getattr(call_result, "content", []) or []:
        t = getattr(block, "text", None)
        if t is not None:
            parts.append(t)
    return "\n".join(parts), bool(getattr(call_result, "isError", False))


def _ensure_call_ids(tool_calls: list[ToolCall], round_no: int) -> None:
    """Give every tool call a stable id so the assistant turn and its tool
    results can be correlated when we feed them back (some providers omit ids).
    Mutates in place; real provider ids are kept, only missing ones are filled."""
    for i, call in enumerate(tool_calls):
        if call.id is None:
            call.id = f"call_{round_no}_{i}"


def _assistant_turn(resp: ModelResponse) -> dict:
    """The neutral history item for an assistant message: the tool calls it made
    (with ids) plus any text. Providers render this back into their wire format."""
    return {"role": "assistant", "tool_calls": list(resp.tool_calls), "content": resp.final_text}


async def run_agent_turn(
    session: Any,
    provider: Provider,
    prompt: str,
    tools: list[ToolSpec],
    *,
    system_prompt: str | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> AgentOutcome:
    """Drive `provider` and the MCP `session` until the model produces final text.

    Stops on: final text, max rounds reached, or recorded error. This function
    OWNS the conversation as a neutral history (user → assistant(tool_calls) →
    tool(results) → …); each provider renders that history into its own wire
    format. The loop never holds a provider-specific message.
    """
    out = AgentOutcome()
    history: list[dict] = [{"role": "user", "content": prompt}]

    # ── Round 1: the initial completion ───────────────────────────────────────
    try:
        resp = provider.complete(prompt, tools, system_prompt=system_prompt)
    except Exception as exc:  # noqa: BLE001 — Layer 4 records, doesn't abort
        out.error = f"{type(exc).__name__}: {exc}"
        return out

    out.rounds = 1
    out.stop_reason = resp.stop_reason
    out.input_tokens = resp.usage.input_tokens
    out.output_tokens = resp.usage.output_tokens
    out.latency_ms = resp.latency_ms
    _ensure_call_ids(resp.tool_calls, out.rounds)
    history.append(_assistant_turn(resp))

    while True:
        if resp.error:
            out.error = resp.error
            return out

        if not resp.tool_calls:
            # Final text — we're done.
            out.final_text = resp.final_text
            return out

        if out.rounds >= max_rounds:
            out.error = f"max rounds reached ({max_rounds}) without final text"
            return out

        # Execute every tool call this round on the MCP.
        results: list[ToolResult] = []
        for call in resp.tool_calls:
            try:
                raw = await session.call_tool(call.name, call.arguments)
            except Exception as exc:  # noqa: BLE001
                results.append(
                    ToolResult(
                        call_id=call.id, name=call.name,
                        content_text=f"{type(exc).__name__}: {exc}", is_error=True,
                    )
                )
                out.tool_calls.append(
                    ToolCallTrace(call.name, call.arguments, f"{type(exc).__name__}: {exc}", True)
                )
                continue
            text, is_err = _result_text(raw)
            results.append(ToolResult(call.id, call.name, text, is_err))
            out.tool_calls.append(ToolCallTrace(call.name, call.arguments, text, is_err))

        # Ask the model for the next move.
        try:
            resp = provider.continue_with_tool_results(
                history, results, tools, system_prompt=system_prompt
            )
        except NotImplementedError as exc:
            out.error = str(exc)
            return out
        except Exception as exc:  # noqa: BLE001
            out.error = f"{type(exc).__name__}: {exc}"
            return out

        out.rounds += 1
        out.stop_reason = resp.stop_reason
        if resp.usage.input_tokens is not None:
            out.input_tokens = (out.input_tokens or 0) + resp.usage.input_tokens
        if resp.usage.output_tokens is not None:
            out.output_tokens = (out.output_tokens or 0) + resp.usage.output_tokens
        if resp.latency_ms is not None:
            out.latency_ms = (out.latency_ms or 0) + resp.latency_ms
        history.append({"role": "tool", "results": results})
        _ensure_call_ids(resp.tool_calls, out.rounds)
        history.append(_assistant_turn(resp))
