"""Layer 3 report tests — verify the report SEPARATES tool-selection from
argument accuracy, and that it serializes to JSON / renders to Markdown. Uses
the fake model (real MCP connection for tools/list, injected ModelCall): no API
key, no tokens.
"""

from __future__ import annotations

import json

from harness.eval.dataset import EvalCase
from harness.eval.runner import build_layer3_report, evaluate_dataset
from harness.report import CaseReport, RunRecord, render_markdown


def _always(tool: str | None, args: dict | None = None):
    def _call(prompt, tools):
        return tool, (args or {})
    return _call


async def test_report_separates_tool_from_arg_failures(connect):
    """Right tool but wrong args must read as tool_ok=True, args_ok=False — the
    whole point of keeping the two metrics apart."""
    case = EvalCase(
        id="C1",
        prompt="show run x",
        expected_tool="qa_get_run",
        expected_args_contains={"run_id": "search-25_classification"},
    )
    wrong_args = _always("qa_get_run", {"params": {"run_id": "nope"}})
    ds = await evaluate_dataset(connect, [case], wrong_args, n_runs=3, threshold=0.9)
    report = build_layer3_report(ds, target="t", model="m", n_runs=3)

    c = report.cases[0]
    assert c.tool_selection_rate == 1.0   # tool always correct
    assert c.arg_accuracy == 0.0          # args always wrong (but measured!)
    assert c.pass_rate == 0.0             # so the case fails overall
    assert all(r.tool_ok and not r.args_ok for r in c.runs)


async def test_report_json_and_markdown_carry_the_signal(connect):
    case = EvalCase(id="C1", prompt="list the runs", expected_tool="qa_list_runs")
    ds = await evaluate_dataset(connect, [case], _always("qa_list_runs"), n_runs=2, threshold=0.9)
    report = build_layer3_report(
        ds, target="qa-toolkit-local", model="claude-sonnet-4-6", n_runs=2
    )

    # JSON: serializable and carries the separated summary metrics.
    payload = json.dumps(report.to_dict())
    d = json.loads(payload)
    assert d["layer"] == 3
    assert d["summary"]["tool_selection_rate"] == 1.0
    assert d["summary"]["arg_accuracy"] is None  # no args were expected → not 0.0

    # Markdown: shows the prompt and the per-run tool/args table.
    md = render_markdown(report)
    assert "list the runs" in md
    assert "tool_ok" in md and "args_ok" in md


def test_arg_accuracy_is_none_when_no_args_expected():
    """A case with no expected args has nothing to measure: arg_accuracy is None,
    not 0.0 (which would falsely read as 'always wrong')."""
    runs = [RunRecord(tool="qa_list_runs", args={}, tool_ok=True, args_ok=True)]
    c = CaseReport(
        id="C",
        prompt="p",
        expected_tool="qa_list_runs",
        expected_args_contains=None,
        runs=runs,
    )
    assert c.arg_accuracy is None
    assert c.tool_selection_rate == 1.0
    assert c.consistency == 1.0


def test_consistency_reflects_wavering():
    """consistency = share of runs choosing the most common tool."""
    runs = [
        RunRecord(tool="qa_list_runs", args={}, tool_ok=True, args_ok=True),
        RunRecord(tool="qa_get_run", args={}, tool_ok=False, args_ok=True),
        RunRecord(tool="qa_list_runs", args={}, tool_ok=True, args_ok=True),
        RunRecord(tool="qa_list_runs", args={}, tool_ok=True, args_ok=True),
    ]
    c = CaseReport(
        id="C", prompt="p", expected_tool="qa_list_runs",
        expected_args_contains=None, runs=runs,
    )
    assert c.consistency == 0.75            # 3 of 4 runs picked qa_list_runs
    assert c.tool_selection_rate == 0.75
