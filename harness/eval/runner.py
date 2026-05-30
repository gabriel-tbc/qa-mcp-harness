"""Layer 3 runner: measure tool-selection accuracy over N runs.

Design: the runner is decoupled from any specific model via a `ModelCall`
callable — `(prompt, anthropic_tools) -> (tool_name | None, tool_input)`. This
is what makes the deterministic machinery (pass-rate, matching, thresholds)
testable WITHOUT an API key: tests inject a fake `ModelCall`. The real,
Anthropic-backed `ModelCall` is built by `anthropic_model_call` and only used
when a key is present.

For tool-SELECTION accuracy we only need the model's *first* tool choice, so we
don't execute the tool or run the agent loop to completion. We capture the first
`tool_use` and stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from harness.eval.dataset import EvalCase
from harness.eval.matching import run_signals
from harness.eval.schema_convert import to_anthropic_tools
from harness.report import CaseReport, LayerReport, RunRecord, write_report

# (prompt, anthropic_tools) -> (chosen_tool_name_or_None, tool_input_dict)
ModelCall = Callable[[str, list[dict]], tuple[str | None, dict]]

# Zero-arg factory returning an async context manager yielding a ClientSession.
ConnectFactory = Callable[[], Any]


@dataclass
class CaseResult:
    case_id: str
    expected_tool: str
    n_runs: int
    matches: int
    observed_tools: list[str | None] = field(default_factory=list)
    # Richer per-run trail (tool + args + the two separate signals) that feeds
    # the report. Kept additive so the legacy surface above stays unchanged.
    runs: list[RunRecord] = field(default_factory=list)
    expected_args_contains: dict | None = None
    prompt: str = ""

    @property
    def pass_rate(self) -> float:
        return self.matches / self.n_runs if self.n_runs else 0.0

    def passed(self, threshold: float) -> bool:
        return self.pass_rate >= threshold


@dataclass
class DatasetResult:
    results: list[CaseResult]
    threshold: float

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed(self.threshold) for r in self.results) / len(self.results)


async def _list_anthropic_tools(connect: ConnectFactory) -> list[dict]:
    """Connect once, fetch tools/list, convert to Anthropic shape."""
    async with connect() as session:
        result = await session.list_tools()
    return to_anthropic_tools(result.tools)


async def evaluate_case(
    connect: ConnectFactory,
    case: EvalCase,
    model_call: ModelCall,
    *,
    n_runs: int = 5,
    anthropic_tools: list[dict] | None = None,
) -> CaseResult:
    """Run one case `n_runs` times; count how often the model's first tool call
    matched the expectation. `anthropic_tools` may be passed in to avoid
    reconnecting per case when evaluating a whole dataset."""
    if anthropic_tools is None:
        anthropic_tools = await _list_anthropic_tools(connect)

    runs: list[RunRecord] = []
    for _ in range(n_runs):
        tool_name, tool_input = model_call(case.prompt, anthropic_tools)
        tool_ok, args_ok = run_signals(
            tool_name, tool_input, case.expected_tool, case.expected_args_contains
        )
        runs.append(
            RunRecord(tool=tool_name, args=dict(tool_input or {}), tool_ok=tool_ok, args_ok=args_ok)
        )

    return CaseResult(
        case_id=case.id,
        expected_tool=case.expected_tool,
        n_runs=n_runs,
        matches=sum(r.passed for r in runs),
        observed_tools=[r.tool for r in runs],
        runs=runs,
        expected_args_contains=case.expected_args_contains,
        prompt=case.prompt,
    )


async def evaluate_dataset(
    connect: ConnectFactory,
    cases: list[EvalCase],
    model_call: ModelCall,
    *,
    n_runs: int = 5,
    threshold: float = 0.9,
) -> DatasetResult:
    """Evaluate every case. Connects once for the tool list, reuses it across cases."""
    anthropic_tools = await _list_anthropic_tools(connect)
    results = [
        await evaluate_case(
            connect, case, model_call, n_runs=n_runs, anthropic_tools=anthropic_tools
        )
        for case in cases
    ]
    return DatasetResult(results=results, threshold=threshold)


def build_layer3_report(
    ds: DatasetResult, *, target: str, model: str, n_runs: int
) -> LayerReport:
    """Map a `DatasetResult` into a persistable `LayerReport` (Layer 3)."""
    cases = [
        CaseReport(
            id=r.case_id,
            prompt=r.prompt,
            expected_tool=r.expected_tool,
            expected_args_contains=r.expected_args_contains,
            runs=r.runs,
        )
        for r in ds.results
    ]
    return LayerReport(
        layer=3,
        target=target,
        model=model,
        n_runs=n_runs,
        threshold=ds.threshold,
        cases=cases,
    )


# ─── Real model wiring (only used when an API key is present) ─────────────────


def anthropic_model_call(model: str, api_key: str) -> ModelCall:
    """Build a ModelCall backed by the Anthropic API. Imports the SDK lazily so
    the rest of the harness works without `anthropic` installed."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def _call(prompt: str, tools: list[dict]) -> tuple[str | None, dict]:
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            tools=tools,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return block.name, dict(block.input)
        return None, {}

    return _call


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _format_report(ds: DatasetResult) -> str:
    lines = [
        f"Tool-selection accuracy: {ds.accuracy:.0%} "
        f"(threshold {ds.threshold:.0%}, {len(ds.results)} cases)",
        "",
    ]
    for r in ds.results:
        mark = "PASS" if r.passed(ds.threshold) else "FAIL"
        lines.append(
            f"  [{mark}] {r.case_id}: {r.matches}/{r.n_runs} "
            f"(expected {r.expected_tool}; observed {r.observed_tools})"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse
    import asyncio
    import os

    from harness.clients.mcp_client import open_session
    from harness.config import active_target, anthropic_api_key
    from harness.eval.dataset import load_jsonl

    parser = argparse.ArgumentParser(description="Run a Layer 3 tool-calling eval.")
    parser.add_argument("dataset", help="Path to a .jsonl dataset")
    parser.add_argument("-n", "--runs", type=int, default=5, help="Runs per case")
    parser.add_argument("-t", "--threshold", type=float, default=0.9)
    parser.add_argument(
        "-m", "--model", default=os.environ.get("HARNESS_MODEL"),
        help="Model id (or set HARNESS_MODEL)",
    )
    args = parser.parse_args()

    key = anthropic_api_key()
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set.")
    if not args.model:
        raise SystemExit("No model: pass --model or set HARNESS_MODEL.")

    target = active_target()
    cases = load_jsonl(args.dataset)
    model_call = anthropic_model_call(args.model, key)

    def connect():
        return open_session(target)

    ds = asyncio.run(
        evaluate_dataset(connect, cases, model_call, n_runs=args.runs, threshold=args.threshold)
    )
    print(_format_report(ds))

    report = build_layer3_report(ds, target=target.name, model=args.model, n_runs=args.runs)
    json_path, md_path = write_report(report)
    print(f"\nReport written:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
