"""Pure tests for the Gemini adapter — tool conversion + response parsing.
No `google-genai` SDK, no network, no key: we fake the response shape with
dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.providers.base import ToolSpec
from harness.providers.gemini import _parse_gemini_response, _tools_to_gemini


def test_tools_to_gemini_groups_under_function_declarations():
    specs = [
        ToolSpec("qa_list_runs", "List runs", {"type": "object"}),
        ToolSpec("qa_get_run", "Get a run", {"type": "object"}),
    ]
    out = _tools_to_gemini(specs)
    # Gemini nests ALL declarations under a single Tool object.
    assert len(out) == 1
    decls = out[0]["function_declarations"]
    assert [d["name"] for d in decls] == ["qa_list_runs", "qa_get_run"]
    assert decls[0]["parameters"] == {"type": "object"}


# ── fake Gemini generate_content response ─────────────────────────────────────


@dataclass
class _FC:
    name: str
    args: dict


@dataclass
class _Part:
    function_call: _FC | None = None
    text: str | None = None


@dataclass
class _Content:
    parts: list


@dataclass
class _Cand:
    content: _Content
    finish_reason: str


@dataclass
class _UM:
    prompt_token_count: int
    candidates_token_count: int


@dataclass
class _Resp:
    candidates: list
    usage_metadata: _UM


def test_parse_extracts_function_call_and_usage():
    resp = _Resp(
        candidates=[
            _Cand(
                content=_Content(
                    parts=[
                        _Part(text="Let me check. "),
                        _Part(function_call=_FC("qa_get_run", {"params": {"run_id": "x"}})),
                    ]
                ),
                finish_reason="STOP",
            )
        ],
        usage_metadata=_UM(prompt_token_count=50, candidates_token_count=9),
    )
    out = _parse_gemini_response(resp)
    assert out.first_tool.name == "qa_get_run"
    assert out.first_tool.arguments == {"params": {"run_id": "x"}}
    assert out.final_text == "Let me check. "
    assert out.stop_reason == "STOP"
    assert out.usage.input_tokens == 50
    assert out.usage.total_tokens == 59


def test_parse_text_only_has_no_tool():
    resp = _Resp(
        candidates=[
            _Cand(content=_Content(parts=[_Part(text="The answer is 1.")]), finish_reason="STOP")
        ],
        usage_metadata=_UM(prompt_token_count=10, candidates_token_count=5),
    )
    out = _parse_gemini_response(resp)
    assert out.first_tool is None
    assert out.final_text == "The answer is 1."


def test_parse_empty_candidates_is_safe():
    resp = _Resp(candidates=[], usage_metadata=_UM(prompt_token_count=0, candidates_token_count=0))
    out = _parse_gemini_response(resp)
    assert out.first_tool is None
    assert out.final_text == ""
