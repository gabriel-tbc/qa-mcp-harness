"""Pure tests for the neutral provider types. No model, no network, no key."""

from __future__ import annotations

from dataclasses import dataclass

from harness.providers.base import (
    ModelResponse,
    Provider,
    ToolCall,
    ToolSpec,
    Usage,
)


@dataclass
class _FakeMcpTool:
    name: str
    description: str
    inputSchema: dict


# ─── ToolSpec ────────────────────────────────────────────────────────────────


def test_toolspec_from_mcp_maps_and_strips():
    mcp = _FakeMcpTool("qa_list_runs", "  List runs  ", {"type": "object", "properties": {}})
    spec = ToolSpec.from_mcp(mcp)
    assert spec.name == "qa_list_runs"
    assert spec.description == "List runs"  # stripped
    assert spec.parameters == {"type": "object", "properties": {}}


def test_toolspec_from_mcp_list():
    mcp = [
        _FakeMcpTool("a", "A", {"type": "object"}),
        _FakeMcpTool("b", "B", {"type": "object"}),
    ]
    specs = ToolSpec.from_mcp_list(mcp)
    assert [s.name for s in specs] == ["a", "b"]


def test_toolspec_handles_missing_description():
    mcp = _FakeMcpTool("t", None, {"type": "object"})  # type: ignore[arg-type]
    assert ToolSpec.from_mcp(mcp).description == ""


# ─── Usage ───────────────────────────────────────────────────────────────────


def test_usage_total_sums():
    assert Usage(input_tokens=10, output_tokens=5).total_tokens == 15


def test_usage_total_none_when_both_missing():
    assert Usage().total_tokens is None


def test_usage_total_treats_missing_half_as_zero():
    assert Usage(input_tokens=10).total_tokens == 10


# ─── ModelResponse ───────────────────────────────────────────────────────────


def test_first_tool_returns_first():
    resp = ModelResponse(tool_calls=[ToolCall("a"), ToolCall("b")])
    assert resp.first_tool.name == "a"


def test_first_tool_none_when_no_calls():
    assert ModelResponse(final_text="hi").first_tool is None


def test_modelresponse_to_dict_omits_raw_and_includes_metrics():
    resp = ModelResponse(
        tool_calls=[ToolCall("qa_get_run", {"params": {"run_id": "x"}})],
        final_text="done",
        stop_reason="tool_use",
        usage=Usage(input_tokens=12, output_tokens=3),
        latency_ms=420.0,
        raw={"huge": "payload"},
    )
    d = resp.to_dict()
    assert "raw" not in d
    assert d["final_text"] == "done"
    assert d["usage"]["total_tokens"] == 15
    assert d["latency_ms"] == 420.0
    assert d["tool_calls"][0]["name"] == "qa_get_run"


# ─── Provider protocol (structural) ──────────────────────────────────────────


def test_fakeprovider_satisfies_protocol_structurally():
    from harness.providers.fake import FakeProvider

    p = FakeProvider.always_calls("qa_list_runs")
    # runtime_checkable Protocol: structural isinstance check.
    assert isinstance(p, Provider)


def test_plain_object_missing_complete_is_not_a_provider():
    class _NotAProvider:
        name = "x"
        model = "y"

    assert not isinstance(_NotAProvider(), Provider)
