"""Runner tests with a FAKE model — verifies the whole pass-rate/threshold
machinery without an API key. Uses the real MCP connection (for tools/list) but
injects a deterministic ModelCall in place of a real LLM.
"""

from __future__ import annotations

from harness.eval.dataset import EvalCase
from harness.eval.runner import evaluate_case, evaluate_dataset


def _always(tool: str | None, args: dict | None = None):
    """A fake ModelCall that always returns the same (tool, args)."""
    def _call(prompt, tools):
        return tool, (args or {})
    return _call


async def test_perfect_case_scores_full_pass_rate(connect):
    case = EvalCase(id="C1", prompt="list the runs", expected_tool="qa_list_runs")
    result = await evaluate_case(connect, case, _always("qa_list_runs"), n_runs=5)
    assert result.matches == 5
    assert result.pass_rate == 1.0
    assert result.passed(threshold=0.9) is True


async def test_wrong_tool_scores_zero(connect):
    case = EvalCase(id="C2", prompt="list the runs", expected_tool="qa_list_runs")
    result = await evaluate_case(connect, case, _always("qa_get_run"), n_runs=4)
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
    good = _always("qa_get_run", {"params": {"run_id": "search-25_classification"}})
    assert (await evaluate_case(connect, case, good, n_runs=3)).pass_rate == 1.0
    # Right tool, wrong arg → fail.
    bad = _always("qa_get_run", {"params": {"run_id": "nope"}})
    assert (await evaluate_case(connect, case, bad, n_runs=3)).pass_rate == 0.0


async def test_dataset_accuracy_aggregates(connect):
    cases = [
        EvalCase("C1", "list", "qa_list_runs"),
        EvalCase("C2", "list", "qa_compare_runs"),  # fake always says list → this fails
    ]
    ds = await evaluate_dataset(connect, cases, _always("qa_list_runs"), n_runs=2, threshold=0.9)
    assert ds.accuracy == 0.5  # one of two cases passes
