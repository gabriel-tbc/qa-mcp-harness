"""Shared report artifacts for the eval layers (3 now, 4 later).

Pure stdlib — no `anthropic`, no network — so reports are testable without the
`llm` extra. A layer builds a `LayerReport` and persists it as a pair: a
structured `.json` (for CI, diffs, history across runs) and a human-readable
`.md` (for demonstration). Reports are artifacts and live under `reports/`
(gitignored).

The defining choice here is that metrics stay SEPARATE: "did the model pick the
right tool" and "were the arguments right" are reported independently, because a
single collapsed pass/fail hides which half broke — and which half broke is
exactly the diagnostic signal about the MCP's design.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class RunRecord:
    """One model invocation within a case: what the model chose, and whether it
    matched the two expectations (tool name, and arguments) — kept apart."""

    tool: str | None
    args: dict
    tool_ok: bool
    args_ok: bool

    @property
    def passed(self) -> bool:
        return self.tool_ok and self.args_ok

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "args": self.args,
            "tool_ok": self.tool_ok,
            "args_ok": self.args_ok,
            "passed": self.passed,
        }


@dataclass
class CaseReport:
    """One dataset case: its expectation, the per-run records, and the metrics
    derived from them. All metrics are computed from `runs`."""

    id: str
    prompt: str
    expected_tool: str
    expected_args_contains: dict | None
    runs: list[RunRecord]

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def tool_selection_rate(self) -> float:
        """Fraction of runs that picked the expected tool."""
        if not self.runs:
            return 0.0
        return sum(r.tool_ok for r in self.runs) / len(self.runs)

    @property
    def arg_accuracy(self) -> float | None:
        """Fraction of runs with the right args, *among runs that picked the
        right tool*. `None` when no args were expected, or no run picked the
        tool — i.e. there is nothing to measure (distinct from 0.0)."""
        if self.expected_args_contains is None:
            return None
        tool_ok_runs = [r for r in self.runs if r.tool_ok]
        if not tool_ok_runs:
            return None
        return sum(r.args_ok for r in tool_ok_runs) / len(tool_ok_runs)

    @property
    def pass_rate(self) -> float:
        """Fraction of runs that got BOTH tool and args right."""
        if not self.runs:
            return 0.0
        return sum(r.passed for r in self.runs) / len(self.runs)

    @property
    def consistency(self) -> float:
        """Share of runs that chose the single most common tool. 1.0 = perfectly
        stable across runs; lower means the model wavered. This is the
        semi-indeterminism signal the layer exists to surface."""
        if not self.runs:
            return 0.0
        counts = Counter(r.tool for r in self.runs)
        return counts.most_common(1)[0][1] / len(self.runs)

    def passed(self, threshold: float) -> bool:
        return self.pass_rate >= threshold

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "expected_tool": self.expected_tool,
            "expected_args_contains": self.expected_args_contains,
            "tool_selection_rate": self.tool_selection_rate,
            "arg_accuracy": self.arg_accuracy,
            "pass_rate": self.pass_rate,
            "consistency": self.consistency,
            "runs": [r.to_dict() for r in self.runs],
        }


@dataclass
class LayerReport:
    """A whole eval run: header (what was tested, how) + per-case reports."""

    layer: int
    target: str
    model: str
    n_runs: int
    threshold: float
    cases: list[CaseReport]
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    @property
    def accuracy(self) -> float:
        """Share of cases passing the threshold."""
        if not self.cases:
            return 0.0
        return sum(c.passed(self.threshold) for c in self.cases) / len(self.cases)

    @property
    def tool_selection_rate(self) -> float:
        """Mean tool-selection rate over every run of every case."""
        runs = [r for c in self.cases for r in c.runs]
        if not runs:
            return 0.0
        return sum(r.tool_ok for r in runs) / len(runs)

    @property
    def arg_accuracy(self) -> float | None:
        """Mean arg-accuracy over runs that expected args AND picked the right
        tool. `None` if there is no such run."""
        runs = [
            r
            for c in self.cases
            if c.expected_args_contains is not None
            for r in c.runs
            if r.tool_ok
        ]
        if not runs:
            return None
        return sum(r.args_ok for r in runs) / len(runs)

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "generated_at": self.generated_at,
            "target": self.target,
            "model": self.model,
            "n_runs": self.n_runs,
            "threshold": self.threshold,
            "summary": {
                "cases": len(self.cases),
                "cases_passed": sum(c.passed(self.threshold) for c in self.cases),
                "accuracy": self.accuracy,
                "tool_selection_rate": self.tool_selection_rate,
                "arg_accuracy": self.arg_accuracy,
            },
            "cases": [c.to_dict() for c in self.cases],
        }


# ─── Rendering ───────────────────────────────────────────────────────────────


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _cell(args: dict) -> str:
    """Compact, markdown-table-safe rendering of a run's args."""
    s = json.dumps(args, ensure_ascii=False)
    if len(s) > 60:
        s = s[:57] + "..."
    return s.replace("|", "\\|")


def render_markdown(report: LayerReport) -> str:
    r = report
    out: list[str] = [
        f"# Layer {r.layer} — Tool-calling report",
        "",
        f"`target={r.target}` · `model={r.model}` · `n={r.n_runs}` · "
        f"`threshold={r.threshold:.0%}` · {r.generated_at}",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---|",
        f"| cases | {len(r.cases)} |",
        f"| cases passing threshold | "
        f"{sum(c.passed(r.threshold) for c in r.cases)}/{len(r.cases)} ({_pct(r.accuracy)}) |",
        f"| tool-selection rate | {_pct(r.tool_selection_rate)} |",
        f"| arg-accuracy (where tool ok) | {_pct(r.arg_accuracy)} |",
        "",
        "## Cases",
    ]
    for c in r.cases:
        verdict = "PASS" if c.passed(r.threshold) else "FAIL"
        args_part = (
            f" · args {_pct(c.arg_accuracy)}" if c.expected_args_contains is not None else ""
        )
        expected = f"`{c.expected_tool}`"
        if c.expected_args_contains is not None:
            expected += f" · args ⊇ `{json.dumps(c.expected_args_contains, ensure_ascii=False)}`"
        out += [
            "",
            f"### {c.id} — {verdict} "
            f"(tool {_pct(c.tool_selection_rate)}{args_part} · consistency {_pct(c.consistency)})",
            f"- **prompt:** {c.prompt}",
            f"- **expected:** {expected}",
            "",
            "| # | tool | args | tool_ok | args_ok |",
            "|---|---|---|---|---|",
        ]
        for i, run in enumerate(c.runs, start=1):
            out.append(
                f"| {i} | {run.tool or '∅'} | `{_cell(run.args)}` | "
                f"{'✓' if run.tool_ok else '✗'} | {'✓' if run.args_ok else '✗'} |"
            )
    out.append("")
    return "\n".join(out)


# ─── Persistence ─────────────────────────────────────────────────────────────


def _slug(s: str) -> str:
    """Filesystem-safe slug (Windows-safe: no ':' or other reserved chars)."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in s)


def write_report(report: LayerReport, base_dir: str | Path = "reports") -> tuple[Path, Path]:
    """Persist the report as a `.json` + `.md` pair under
    `base_dir/layer<N>/<timestamp>__<target>__<model>.*`. Returns the two paths."""
    out_dir = Path(base_dir) / f"layer{report.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # "2026-05-30T14:36:11Z" -> "20260530T143611Z" (no ':' / '-', Windows-safe).
    ts = report.generated_at.replace("-", "").replace(":", "")
    stem = f"{ts}__{_slug(report.target)}__{_slug(report.model)}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
