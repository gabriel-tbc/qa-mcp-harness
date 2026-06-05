"""Pure tests for the Anthropic adapter — tool conversion + response parsing.
No `anthropic` SDK, no network, no key: we fake the response shape with
dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.providers.anthropic import _parse_anthropic_response, _tools_to_anthropic
from harness.providers.base import ToolSpec


def test_tools_to_anthropic_shape():
    specs = [ToolSpec("qa_list_runs", "List runs", {"type": "object", "properties": {}})]
    out = _tools_to_anthropic(specs)
    assert out == [
        {
            "name": "qa_list_runs",
            "description": "List runs",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


# ── fake Anthropic Messages response ──────────────────────────────────────────


@dataclass
class _Block:
    type: str
    name: str = ""
    input: dict = field(default_factory=dict)
    text: str = ""


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _Resp:
    content: list
    stop_reason: str
    usage: _Usage


def test_parse_extracts_tool_use_and_usage():
    resp = _Resp(
        content=[
            _Block(type="text", text="Let me check. "),
            _Block(type="tool_use", name="qa_get_run", input={"params": {"run_id": "x"}}),
        ],
        stop_reason="tool_use",
        usage=_Usage(input_tokens=40, output_tokens=12),
    )
    out = _parse_anthropic_response(resp)
    assert out.first_tool.name == "qa_get_run"
    assert out.first_tool.arguments == {"params": {"run_id": "x"}}
    assert out.final_text == "Let me check. "
    assert out.stop_reason == "tool_use"
    assert out.usage.input_tokens == 40
    assert out.usage.total_tokens == 52


def test_parse_text_only_has_no_tool():
    resp = _Resp(
        content=[_Block(type="text", text="The answer is 1.")],
        stop_reason="end_turn",
        usage=_Usage(input_tokens=10, output_tokens=5),
    )
    out = _parse_anthropic_response(resp)
    assert out.first_tool is None
    assert out.final_text == "The answer is 1."
