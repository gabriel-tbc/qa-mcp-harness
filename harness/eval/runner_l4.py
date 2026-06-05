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
from typing import Any, Callable

from harness.eval.agent_loop import AgentOutcome, run_agent_turn
from harness.eval.dataset_l4 import CheckSpec, EvalCaseL4, get_path
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
