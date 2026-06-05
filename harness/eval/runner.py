"""Layer 3 runner: measure tool-selection accuracy over N runs.

The runner is decoupled from any specific model via the `Provider` protocol
(see `harness.providers`). It speaks only neutral types — `ToolSpec` in,
`ModelResponse` out — so it never knows whether a turn ran on Claude, GPT,
Gemini, or a local Ollama model. Tests inject a `FakeProvider`, so the
deterministic machinery (pass-rate, matching, thresholds, reporting) is verified
without an API key.

For tool-SELECTION accuracy we only need the model's *first* tool choice
(`ModelResponse.first_tool`); we don't execute the tool or run the full agent
loop (that's Layer 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from harness.eval.dataset import EvalCase
from harness.eval.matching import run_signals
from harness.eval.prompts import resolve_system_prompt
from harness.providers.base import ModelResponse, Provider, ToolSpec
from harness.report import CaseReport, LayerReport, RunRecord, write_report

# Zero-arg factory returning an async context manager yielding a ClientSession.
ConnectFactory = Callable[[], Any]


@dataclass
class CaseResult:
    case_id: str
    expected_tool: str
    n_runs: int
    matches: int
    observed_tools: list[str | None] = field(default_factory=list)
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


async def _list_tools(connect: ConnectFactory) -> list[ToolSpec]:
    """Connect once, fetch tools/list, convert to neutral ToolSpecs."""
    async with connect() as session:
        result = await session.list_tools()
    return ToolSpec.from_mcp_list(result.tools)


async def evaluate_case(
    connect: ConnectFactory,
    case: EvalCase,
    provider: Provider,
    *,
    n_runs: int = 5,
    tools: list[ToolSpec] | None = None,
) -> CaseResult:
    """Run one case `n_runs` times; count how often the model's first tool call
    matched the expectation. `tools` may be passed in to avoid reconnecting per
    case when evaluating a whole dataset."""
    if tools is None:
        tools = await _list_tools(connect)

    # Resolve once per case: the EFFECTIVE system prompt string the provider
    # sees, composed from the suite policy and any case override. Recorded
    # verbatim on every RunRecord so reports are reproducible.
    effective_system = resolve_system_prompt(
        case.system_policy, case.system_prompt_override
    )

    runs: list[RunRecord] = []
    for _ in range(n_runs):
        resp: ModelResponse = provider.complete(
            case.prompt, tools, system_prompt=effective_system
        )
        first = resp.first_tool
        tool_name = first.name if first else None
        tool_args = first.arguments if first else {}
        tool_ok, args_ok = run_signals(
            tool_name, tool_args, case.expected_tool, case.expected_args_contains
        )
        runs.append(
            RunRecord(
                tool=tool_name,
                args=dict(tool_args),
                tool_ok=tool_ok,
                args_ok=args_ok,
                final_text=resp.final_text,
                stop_reason=resp.stop_reason,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                latency_ms=resp.latency_ms,
                error=resp.error,
                # The exact text the model saw, not just a policy name.
                system_prompt=effective_system,
            )
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
    provider: Provider,
    *,
    n_runs: int = 5,
    threshold: float = 0.9,
) -> DatasetResult:
    """Evaluate every case. Connects once for the tool list, reuses it across cases."""
    tools = await _list_tools(connect)
    results = [
        await evaluate_case(connect, case, provider, n_runs=n_runs, tools=tools)
        for case in cases
    ]
    return DatasetResult(results=results, threshold=threshold)


def build_layer3_report(
    ds: DatasetResult,
    *,
    target: str,
    model: str,
    n_runs: int,
    temperature: float | None = None,
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
        temperature=temperature,
    )


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
    from harness.providers.registry import build_provider

    parser = argparse.ArgumentParser(description="Run a Layer 3 tool-calling eval.")
    parser.add_argument("dataset", help="Path to a .jsonl dataset")
    parser.add_argument("-n", "--runs", type=int, default=5, help="Runs per case")
    parser.add_argument("-t", "--threshold", type=float, default=0.9)
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature for the model (None = provider default). "
             "Same value applies to every case in the run; recorded in the report.",
    )
    parser.add_argument(
        "-m", "--model", default=os.environ.get("HARNESS_MODEL"),
        help="Model id (or set HARNESS_MODEL)",
    )
    parser.add_argument(
        "-p", "--provider",
        default=os.environ.get("HARNESS_PROVIDER", "anthropic"),
        choices=["anthropic", "openai", "gemini", "ollama"],
        help="Model provider. 'anthropic'/'openai'/'gemini' are paid APIs; "
             "'ollama' is local and free. Or set HARNESS_PROVIDER.",
    )
    args = parser.parse_args()

    if not args.model:
        raise SystemExit("No model: pass --model or set HARNESS_MODEL.")

    # Pick the right credential for the chosen provider.
    if args.provider == "anthropic":
        api_key = anthropic_api_key()
    elif args.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
    elif args.provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    else:  # ollama is local, no key
        api_key = None

    try:
        provider = build_provider(
            args.provider,
            args.model,
            api_key=api_key,
            base_url=os.environ.get("OLLAMA_BASE_URL"),
            temperature=args.temperature,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    target = active_target()
    cases = load_jsonl(args.dataset)

    def connect():
        return open_session(target)

    ds = asyncio.run(
        evaluate_dataset(connect, cases, provider, n_runs=args.runs, threshold=args.threshold)
    )
    print(_format_report(ds))

    report = build_layer3_report(
        ds,
        target=target.name,
        model=f"{provider.name}:{provider.model}",
        n_runs=args.runs,
        temperature=args.temperature,
    )
    json_path, md_path = write_report(report)
    print(f"\nReport written:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
