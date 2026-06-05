"""Runner tests with a FAKE provider — verifies the whole pass-rate/threshold
machinery without an API key. Uses the real MCP connection (for tools/list) but
injects a scripted FakeProvider in place of a real LLM.
"""

from __future__ import annotations

from harness.eval.dataset import EvalCase
from harness.eval.runner import evaluate_case, evaluate_dataset
from harness.providers.fake import FakeProvider


async def test_perfect_case_scores_full_pass_rate(connect):
    case = EvalCase(id="C1", prompt="list the runs", expected_tool="qa_list_runs")
    provider = FakeProvider.always_calls("qa_list_runs")
    result = await evaluate_case(connect, case, provider, n_runs=5)
    assert result.matches == 5
    assert result.pass_rate == 1.0
    assert result.passed(threshold=0.9) is True


async def test_wrong_tool_scores_zero(connect):
    case = EvalCase(id="C2", prompt="list the runs", expected_tool="qa_list_runs")
    provider = FakeProvider.always_calls("qa_get_run")
    result = await evaluate_case(connect, case, provider, n_runs=4)
    assert result.matches == 0
    assert result.passed(threshold=0.9) is False


async def test_args_expectation_enforced(connect):
    case = EvalCase(
        id="C3",
        prompt="show run x",
        expected_tool="qa_get_run",
        expected_args_contains={"run_id": "search-25_classification"},
    )
    # Right tool, right nested arg → pass.
    good = FakeProvider.always_calls(
        "qa_get_run", {"params": {"run_id": "search-25_classification"}}
    )
    assert (await evaluate_case(connect, case, good, n_runs=3)).pass_rate == 1.0
    # Right tool, wrong arg → fail.
    bad = FakeProvider.always_calls("qa_get_run", {"params": {"run_id": "nope"}})
    assert (await evaluate_case(connect, case, bad, n_runs=3)).pass_rate == 0.0


async def test_no_tool_call_scores_zero(connect):
    """A provider that answers with text and calls no tool fails a tool-expecting case."""
    case = EvalCase(id="C4", prompt="list the runs", expected_tool="qa_list_runs")
    provider = FakeProvider.always_text("I'm not going to call a tool")
    result = await evaluate_case(connect, case, provider, n_runs=3)
    assert result.matches == 0
    assert result.observed_tools == [None, None, None]


async def test_case_system_policy_resolves_and_lands_on_runrecord(connect):
    """A case naming `system_policy=default` must (a) load the policy file,
    (b) hand the EFFECTIVE text to Provider.complete, (c) record that exact
    text on each RunRecord. Reports must be reproducible from disk alone."""
    from harness.providers.base import ModelResponse, ToolCall

    class _CapturingProvider:
        name = "capture"
        model = "test"
        def __init__(self):
            self.seen_systems: list[str | None] = []
        def complete(self, prompt, tools, *, system_prompt=None):
            self.seen_systems.append(system_prompt)
            return ModelResponse(tool_calls=[ToolCall("qa_list_runs", {})])

    provider = _CapturingProvider()
    case = EvalCase(
        id="S1", prompt="list runs", expected_tool="qa_list_runs",
        system_policy="default",
    )
    result = await evaluate_case(connect, case, provider, n_runs=2)

    # Both runs saw the same non-empty resolved string, identical on every run.
    assert provider.seen_systems[0] is not None
    assert all(s == provider.seen_systems[0] for s in provider.seen_systems)
    # Every RunRecord carries the same effective text — no orphan numbers.
    assert all(r.system_prompt == provider.seen_systems[0] for r in result.runs)


async def test_override_substitutes_policy(connect):
    """When a case sets `system_prompt_override`, it WINS — the policy is not
    appended. Substitution keeps the diff explicit when an adversarial case
    needs to contradict the suite policy."""
    from harness.providers.base import ModelResponse, ToolCall

    class _CapturingProvider:
        name = "capture"
        model = "test"
        def __init__(self):
            self.seen: list[str | None] = []
        def complete(self, prompt, tools, *, system_prompt=None):
            self.seen.append(system_prompt)
            return ModelResponse(tool_calls=[ToolCall("qa_list_runs", {})])

    provider = _CapturingProvider()
    case = EvalCase(
        id="S1", prompt="list runs", expected_tool="qa_list_runs",
        system_policy="default",
        system_prompt_override="JUST DO IT",
    )
    await evaluate_case(connect, case, provider, n_runs=1)
    assert provider.seen == ["JUST DO IT"]  # override wins, no concatenation


async def test_rich_trace_flows_from_response_into_runrecord(connect):
    """latency + tokens + final_text from the provider's ModelResponse must land
    on the RunRecord (the 'capture everything' goal)."""
    from harness.providers.base import Usage

    case = EvalCase(id="C5", prompt="list the runs", expected_tool="qa_list_runs")
    provider = FakeProvider.always_calls(
        "qa_list_runs", usage=Usage(input_tokens=12, output_tokens=4), latency_ms=99.0
    )
    result = await evaluate_case(connect, case, provider, n_runs=2)
    r0 = result.runs[0]
    assert r0.input_tokens == 12
    assert r0.output_tokens == 4
    assert r0.latency_ms == 99.0


async def test_dataset_accuracy_aggregates(connect):
    cases = [
        EvalCase("C1", "list", "qa_list_runs"),
        EvalCase("C2", "list", "qa_compare_runs"),  # fake always says list → this fails
    ]
    provider = FakeProvider.always_calls("qa_list_runs")
    ds = await evaluate_dataset(connect, cases, provider, n_runs=2, threshold=0.9)
    assert ds.accuracy == 0.5  # one of two cases passes
