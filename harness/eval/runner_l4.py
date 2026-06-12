"""Layer 4 runner — pass-rate over N for free-text correctness.

For each case, N times:
    1. Compute ground truth: for every Check, call its tool on the MCP and
       extract the expected value at `ground_truth_path`.
    2. Run the agent loop with the case prompt (uses Layer 4's `run_agent_turn`).
    3. Score the model's `final_text` against each Check via the oracles.

The case passes a run iff:
    - the loop produced a final_text (no error / didn't max out), AND
    - every Check.passed is True.

`tools_called_ok` (the loop actually called at least one tool) is reported
beside `oracle_ok` (the prose matched the truth) — colappsing them hides which
half broke, exactly like the Layer 3 signal separation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from harness.eval.agent_loop import AgentOutcome, run_agent_turn
from harness.eval.dataset_l4 import CheckSpec, EvalCaseL4, get_path, load_l4_jsonl
from harness.eval.oracles import Check, ground_truth, number_check
from harness.eval.prompts import resolve_system_prompt
from harness.providers.base import Provider, ToolSpec

ConnectFactory = Callable[[], Any]


@dataclass
class L4RunRecord:
    """One model invocation within a case, Layer-4-shaped."""

    final_text: str
    rounds: int
    tools_called: list[dict] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    tools_called_ok: bool = False
    oracle_ok: bool = False
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    system_prompt: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.tools_called_ok and self.oracle_ok

    def to_dict(self) -> dict:
        return {
            "final_text": self.final_text,
            "rounds": self.rounds,
            "tools_called": self.tools_called,
            "checks": [c.to_dict() for c in self.checks],
            "tools_called_ok": self.tools_called_ok,
            "oracle_ok": self.oracle_ok,
            "passed": self.passed,
            "error": self.error,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "system_prompt": self.system_prompt,
        }


@dataclass
class L4CaseResult:
    case_id: str
    prompt: str
    n_runs: int
    runs: list[L4RunRecord]

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def oracle_rate(self) -> float:
        return sum(r.oracle_ok for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def tool_use_rate(self) -> float:
        return sum(r.tools_called_ok for r in self.runs) / len(self.runs) if self.runs else 0.0

    def passed(self, threshold: float) -> bool:
        return self.pass_rate >= threshold

    def to_dict(self) -> dict:
        return {
            "id": self.case_id,
            "prompt": self.prompt,
            "pass_rate": self.pass_rate,
            "oracle_rate": self.oracle_rate,
            "tool_use_rate": self.tool_use_rate,
            "runs": [r.to_dict() for r in self.runs],
        }


@dataclass
class L4DatasetResult:
    results: list[L4CaseResult]
    threshold: float

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed(self.threshold) for r in self.results) / len(self.results)


async def _ground_truth_values(session: Any, checks: tuple[CheckSpec, ...]) -> list[object]:
    """Compute expected values for every Check (parallel-friendly future,
    sequential for now)."""
    values: list[object] = []
    for spec in checks:
        payload = await ground_truth(session, spec.ground_truth_tool, spec.ground_truth_args)
        values.append(get_path(payload, spec.ground_truth_path))
    return values


def _score(specs: tuple[CheckSpec, ...], expected: list[object], text: str) -> list[Check]:
    out: list[Check] = []
    for spec, exp in zip(specs, expected):
        if spec.kind == "ground-truth-number":
            # Expected must be an int; if it isn't, the Check fails with that
            # mismatch visible in the report (not silently zero).
            try:
                exp_int = int(exp)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                out.append(
                    Check(spec.name, spec.kind, expected=exp, observed=None, passed=False)
                )
                continue
            out.append(number_check(spec.name, exp_int, text, spec.extract_label))
        else:
            out.append(
                Check(spec.name, spec.kind, expected=exp, observed=None, passed=False)
            )
    return out


async def evaluate_case_l4(
    connect: ConnectFactory,
    case: EvalCaseL4,
    provider: Provider,
    *,
    n_runs: int = 3,
    tools: list[ToolSpec] | None = None,
) -> L4CaseResult:
    effective_system = resolve_system_prompt(
        case.system_policy, case.system_prompt_override
    )

    runs: list[L4RunRecord] = []
    for _ in range(n_runs):
        async with connect() as session:
            if tools is None:
                listing = await session.list_tools()
                this_tools = ToolSpec.from_mcp_list(listing.tools)
            else:
                this_tools = tools

            expected = await _ground_truth_values(session, case.checks)
            outcome: AgentOutcome = await run_agent_turn(
                session, provider, case.prompt, this_tools, system_prompt=effective_system
            )

        checks = _score(case.checks, expected, outcome.final_text)
        tools_called_ok = bool(outcome.tool_calls)
        oracle_ok = all(c.passed for c in checks) if checks else False
        runs.append(
            L4RunRecord(
                final_text=outcome.final_text,
                rounds=outcome.rounds,
                tools_called=[
                    {"name": t.name, "arguments": t.arguments, "is_error": t.is_error}
                    for t in outcome.tool_calls
                ],
                checks=checks,
                tools_called_ok=tools_called_ok,
                oracle_ok=oracle_ok,
                error=outcome.error,
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
                latency_ms=outcome.latency_ms,
                system_prompt=effective_system,
            )
        )

    return L4CaseResult(case_id=case.id, prompt=case.prompt, n_runs=n_runs, runs=runs)


async def evaluate_dataset_l4(
    connect: ConnectFactory,
    cases: list[EvalCaseL4],
    provider: Provider,
    *,
    n_runs: int = 3,
    threshold: float = 0.9,
) -> L4DatasetResult:
    results = [
        await evaluate_case_l4(connect, c, provider, n_runs=n_runs) for c in cases
    ]
    return L4DatasetResult(results=results, threshold=threshold)


def build_layer4_report(
    result: L4DatasetResult,
    *,
    target: str,
    model: str,
    n_runs: int,
    temperature: float | None = None,
) -> dict:
    """Map an L4DatasetResult into a persistable report dict (mirrors Layer 3's
    build_layer3_report). Kept as a plain dict so the renderer stays pure."""
    all_runs = [r for c in result.results for r in c.runs]
    lat_vals = [r.latency_ms for r in all_runs if r.latency_ms is not None]
    return {
        "layer": 4,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target": target,
        "model": model,
        "n_runs": n_runs,
        "threshold": result.threshold,
        "temperature": temperature,
        "summary": {
            "cases": len(result.results),
            "cases_passed": sum(c.passed(result.threshold) for c in result.results),
            "accuracy": result.accuracy,
            "oracle_rate": (sum(r.oracle_ok for r in all_runs) / len(all_runs)) if all_runs else 0.0,
            "tool_use_rate": (
                sum(r.tools_called_ok for r in all_runs) / len(all_runs) if all_runs else 0.0
            ),
            "mean_latency_ms": (sum(lat_vals) / len(lat_vals)) if lat_vals else None,
        },
        "cases": [c.to_dict() for c in result.results],
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _format_report_l4(result: L4DatasetResult) -> str:
    lines = [
        f"Layer 4 accuracy: {result.accuracy:.0%} "
        f"(threshold {result.threshold:.0%}, {len(result.results)} cases)",
        "",
    ]
    for c in result.results:
        n = len(c.runs)
        mark = "PASS" if c.passed(result.threshold) else "FAIL"
        lines.append(
            f"  [{mark}] {c.case_id}: pass {sum(r.passed for r in c.runs)}/{n} · "
            f"oracle {sum(r.oracle_ok for r in c.runs)}/{n} · "
            f"tools {sum(r.tools_called_ok for r in c.runs)}/{n}"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse
    import asyncio
    import os

    from harness.clients.mcp_client import open_session
    from harness.config import active_target, anthropic_api_key
    from harness.providers.registry import build_provider
    from harness.report import write_report_l4

    parser = argparse.ArgumentParser(
        description="Run a Layer 4 output eval (full agent loop + ground-truth oracle)."
    )
    parser.add_argument("dataset", help="Path to a Layer 4 .jsonl dataset")
    parser.add_argument("-n", "--runs", type=int, default=3, help="Runs per case")
    parser.add_argument("-t", "--threshold", type=float, default=0.9)
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (None = provider default); recorded in the report.",
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
    cases = load_l4_jsonl(args.dataset)

    def connect():
        return open_session(target)

    result = asyncio.run(
        evaluate_dataset_l4(connect, cases, provider, n_runs=args.runs, threshold=args.threshold)
    )
    print(_format_report_l4(result))

    report = build_layer4_report(
        result,
        target=target.name,
        model=f"{provider.name}:{provider.model}",
        n_runs=args.runs,
        temperature=args.temperature,
    )
    json_path, md_path = write_report_l4(report)
    print(f"\nReport written:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
