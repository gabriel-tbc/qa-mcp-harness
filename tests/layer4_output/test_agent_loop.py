"""End-to-end Layer 4 tests with a scripted FakeProvider against the REAL MCP.

We never call a real LLM — but the MCP IS real, the tools ARE called, and the
loop IS the production loop. The FakeProvider only replaces the model's
decisions, which is what we want for deterministic testing of the loop itself.
"""

from __future__ import annotations

from harness.eval.agent_loop import run_agent_turn
from harness.eval.dataset_l4 import CheckSpec, EvalCaseL4
from harness.eval.runner_l4 import evaluate_case_l4
from harness.providers.base import ModelResponse, ToolCall, ToolSpec
from harness.providers.fake import FakeProvider


# ─── The agent loop itself (provider-level test) ─────────────────────────────


async def test_loop_executes_tool_then_returns_final_text(connect):
    """One round: model asks for qa_list_runs; loop executes it on the MCP;
    next response is final text. Verifies the round-trip and that tool_calls
    are captured in the trace."""
    provider = FakeProvider.scripted([
        ModelResponse(
            tool_calls=[ToolCall("qa_list_runs", {"params": {"response_format": "json"}}, id="t1")],
            stop_reason="tool_use",
        ),
        ModelResponse(final_text="there are 6 runs available", stop_reason="end_turn"),
    ])
    async with connect() as session:
        result = await session.list_tools()
        tools = ToolSpec.from_mcp_list(result.tools)
        out = await run_agent_turn(session, provider, "how many runs?", tools)

    assert out.error is None
    assert out.final_text == "there are 6 runs available"
    assert out.rounds == 2
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "qa_list_runs"


async def test_loop_caps_at_max_rounds(connect):
    """A misbehaving model that never returns text must be cut off, not loop
    forever. The error is recorded; the run is not aborted."""
    looping = ModelResponse(
        tool_calls=[ToolCall("qa_list_runs", {"params": {"response_format": "json"}}, id="x")],
        stop_reason="tool_use",
    )
    # Always asks for a tool — would loop forever without the cap.
    provider = FakeProvider.scripted([looping] * 10)
    async with connect() as session:
        tools = ToolSpec.from_mcp_list((await session.list_tools()).tools)
        out = await run_agent_turn(session, provider, "x", tools, max_rounds=3)
    assert out.error is not None
    assert "max rounds" in out.error


# ─── Layer 4 case scoring (oracle + extraction) ──────────────────────────────


def _regression_check_spec() -> CheckSpec:
    return CheckSpec(
        name="regression_count",
        kind="ground-truth-number",
        ground_truth_tool="qa_compare_runs",
        ground_truth_args={
            "params": {
                "run_a": "search-25_classification",
                "run_b": "search-26_classification",
                "response_format": "json",
            }
        },
        ground_truth_path="counts.regressions",
        extract_label="regression",
    )


async def test_case_pass_when_model_states_correct_count(connect):
    """The fixture story: 1 regression between search-25 and search-26. If the
    model's prose says '1 regression', the Check passes — verified against
    ground truth computed live from the MCP, not a hardcoded expected."""
    spec = _regression_check_spec()
    case = EvalCaseL4(id="L4-001", prompt="how many regressions?", checks=(spec,))

    # Scripted: round 1 calls qa_compare_runs; round 2 returns the right text.
    provider = FakeProvider.scripted([
        ModelResponse(
            tool_calls=[ToolCall("qa_compare_runs", spec.ground_truth_args, id="c1")],
            stop_reason="tool_use",
        ),
        ModelResponse(final_text="There is 1 regression.", stop_reason="end_turn"),
    ])
    result = await evaluate_case_l4(connect, case, provider, n_runs=1)
    run = result.runs[0]
    assert run.tools_called_ok is True
    assert run.oracle_ok is True
    assert run.passed is True
    assert run.checks[0].expected == 1
    assert run.checks[0].observed == 1


async def test_case_fails_oracle_when_model_hallucinates_count(connect):
    """Model called the right tool AND got the data, but its prose lies — the
    oracle catches it. tools_called_ok=True, oracle_ok=False — the signal
    separation tells us which half broke."""
    spec = _regression_check_spec()
    case = EvalCaseL4(id="L4-001", prompt="how many regressions?", checks=(spec,))
    provider = FakeProvider.scripted([
        ModelResponse(
            tool_calls=[ToolCall("qa_compare_runs", spec.ground_truth_args, id="c1")],
            stop_reason="tool_use",
        ),
        # Wrong number — hallucination. Truth from the fixture is 1.
        ModelResponse(final_text="There are 5 regressions.", stop_reason="end_turn"),
    ])
    result = await evaluate_case_l4(connect, case, provider, n_runs=1)
    run = result.runs[0]
    assert run.tools_called_ok is True
    assert run.oracle_ok is False
    assert run.checks[0].expected == 1
    assert run.checks[0].observed == 5


async def test_case_fails_loudly_when_text_has_no_extractable_number(connect):
    """Model refuses or vagues out — observed=None, oracle fails. The fact we
    didn't get an answer is itself a recorded failure mode, not a silent zero."""
    spec = _regression_check_spec()
    case = EvalCaseL4(id="L4-001", prompt="how many regressions?", checks=(spec,))
    provider = FakeProvider.scripted([
        ModelResponse(
            tool_calls=[ToolCall("qa_compare_runs", spec.ground_truth_args, id="c1")],
            stop_reason="tool_use",
        ),
        ModelResponse(final_text="I cannot answer that.", stop_reason="end_turn"),
    ])
    result = await evaluate_case_l4(connect, case, provider, n_runs=1)
    run = result.runs[0]
    assert run.oracle_ok is False
    assert run.checks[0].observed is None  # absence, not zero


async def test_pass_rate_over_n_runs(connect):
    """Layer 4 is also pass-rate over N: a case with 2 of 3 correct runs gives
    pass_rate=2/3 — the metric of choice for non-deterministic output."""
    spec = _regression_check_spec()
    case = EvalCaseL4(id="L4-001", prompt="how many regressions?", checks=(spec,))

    good = [
        ModelResponse(tool_calls=[ToolCall("qa_compare_runs", spec.ground_truth_args, id="g")],
                      stop_reason="tool_use"),
        ModelResponse(final_text="There is 1 regression.", stop_reason="end_turn"),
    ]
    bad = [
        ModelResponse(tool_calls=[ToolCall("qa_compare_runs", spec.ground_truth_args, id="b")],
                      stop_reason="tool_use"),
        ModelResponse(final_text="There are 7 regressions.", stop_reason="end_turn"),
    ]
    provider = FakeProvider.scripted(good + bad + good)  # pass, fail, pass
    result = await evaluate_case_l4(connect, case, provider, n_runs=3)
    assert abs(result.pass_rate - 2 / 3) < 1e-9
    assert result.oracle_rate == 2 / 3
    assert result.tool_use_rate == 1.0
