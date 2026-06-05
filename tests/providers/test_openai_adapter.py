"""Pure tests for the OpenAI/Ollama adapter — tool conversion + response parsing.
No `openai` SDK, no network, no key: we fake the response shape with dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.providers.base import ToolSpec
from harness.providers.openai_compat import _parse_openai_response, _tools_to_openai


def test_tools_to_openai_shape():
    specs = [ToolSpec("qa_list_runs", "List runs", {"type": "object"})]
    out = _tools_to_openai(specs)
    assert out == [
        {
            "type": "function",
            "function": {
                "name": "qa_list_runs",
                "description": "List runs",
                "parameters": {"type": "object"},
            },
        }
    ]


# ── fake OpenAI chat-completions response ─────────────────────────────────────


@dataclass
class _Fn:
    name: str
    arguments: str


@dataclass
class _TC:
    function: _Fn


@dataclass
class _Msg:
    content: str | None
    tool_calls: list | None


@dataclass
class _Choice:
    message: _Msg
    finish_reason: str


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _Resp:
    choices: list
    usage: _Usage


def test_parse_extracts_tool_call_with_json_string_args():
    resp = _Resp(
        choices=[
            _Choice(
                message=_Msg(
                    content=None,
                    tool_calls=[_TC(_Fn("qa_get_run", '{"params": {"run_id": "x"}}'))],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=_Usage(prompt_tokens=30, completion_tokens=8),
    )
    out = _parse_openai_response(resp)
    assert out.first_tool.name == "qa_get_run"
    assert out.first_tool.arguments == {"params": {"run_id": "x"}}
    assert out.stop_reason == "tool_calls"
    assert out.usage.total_tokens == 38


def test_parse_malformed_args_degrade_to_empty():
    resp = _Resp(
        choices=[
            _Choice(
                message=_Msg(content=None, tool_calls=[_TC(_Fn("t", "not json"))]),
                finish_reason="tool_calls",
            )
        ],
        usage=_Usage(prompt_tokens=1, completion_tokens=1),
    )
    out = _parse_openai_response(resp)
    assert out.first_tool.name == "t"
    assert out.first_tool.arguments == {}


def test_parse_text_only_no_tool():
    resp = _Resp(
        choices=[_Choice(message=_Msg(content="the answer is 1", tool_calls=None),
                         finish_reason="stop")],
        usage=_Usage(prompt_tokens=5, completion_tokens=4),
    )
    out = _parse_openai_response(resp)
    assert out.first_tool is None
    assert out.final_text == "the answer is 1"
