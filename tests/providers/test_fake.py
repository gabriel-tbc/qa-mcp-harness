"""Tests for FakeProvider — the scripted provider used to test the engine."""

from __future__ import annotations

from harness.providers.base import ToolSpec
from harness.providers.fake import FakeProvider

_TOOLS = [ToolSpec("qa_list_runs", "List runs", {"type": "object"})]


def test_always_calls_returns_scripted_tool():
    p = FakeProvider.always_calls("qa_get_run", {"params": {"run_id": "x"}})
    resp = p.complete("show run x", _TOOLS)
    assert resp.first_tool.name == "qa_get_run"
    assert resp.first_tool.arguments == {"params": {"run_id": "x"}}
    assert resp.stop_reason == "tool_use"


def test_always_text_returns_final_text_and_no_tool():
    p = FakeProvider.always_text("the answer is 1")
    resp = p.complete("how many?", _TOOLS)
    assert resp.final_text == "the answer is 1"
    assert resp.first_tool is None


def test_custom_responder_sees_prompt_and_tools():
    seen = {}

    def responder(prompt, tools):
        seen["prompt"] = prompt
        seen["n_tools"] = len(tools)
        from harness.providers.base import ModelResponse

        return ModelResponse(final_text="ok")

    p = FakeProvider(responder=responder)
    p.complete("hello", _TOOLS)
    assert seen == {"prompt": "hello", "n_tools": 1}


def test_default_responder_returns_empty_response():
    p = FakeProvider()
    resp = p.complete("x", _TOOLS)
    assert resp.first_tool is None
    assert resp.final_text == ""


def test_identity_fields_default():
    p = FakeProvider.always_calls("t")
    assert p.name == "fake"
    assert p.model == "fake-1"
