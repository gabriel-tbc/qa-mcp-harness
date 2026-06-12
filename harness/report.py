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
    """One model invocation within a case.

    The verdict fields (`tool`/`args`/`tool_ok`/`args_ok`) are the Layer 3 signal,
    kept apart so a report shows WHICH half failed. The rest is the rich trace
    captured from the model's `ModelResponse` — present so reports can answer
    later questions (hallucinations need `final_text`; response size needs
    tokens; response time needs `latency_ms`) without a structural limit. All
    trace fields are optional so records can be built from partial data."""

    tool: str | None
    args: dict
    tool_ok: bool
    args_ok: bool
    # Rich trace (optional; populated from ModelResponse when available).
    final_text: str = ""
    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    error: str | None = None
    # Knob used for this run — kept on the record so a number is never orphan.
    system_prompt: str | None = None

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
            "final_text": self.final_text,
            "stop_reason": self.stop_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "system_prompt": self.system_prompt,
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

    @property
    def mean_latency_ms(self) -> float | None:
        """Mean model latency across runs that reported it (None if none did)."""
        vals = [r.latency_ms for r in self.runs if r.latency_ms is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def total_output_tokens(self) -> int | None:
        """Sum of output tokens across runs that reported them (response size)."""
        vals = [r.output_tokens for r in self.runs if r.output_tokens is not None]
        return sum(vals) if vals else None

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
            "mean_latency_ms": self.mean_latency_ms,
            "total_output_tokens": self.total_output_tokens,
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
    # Experiment-level knob: same temperature for every case in the run, so
    # accuracy is comparable. None = the provider's default.
    temperature: float | None = None
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

    @property
    def mean_latency_ms(self) -> float | None:
        """Mean model latency over every run that reported it."""
        vals = [r.latency_ms for c in self.cases for r in c.runs if r.latency_ms is not None]
        return sum(vals) / len(vals) if vals else None

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "generated_at": self.generated_at,
            "target": self.target,
            "model": self.model,
            "n_runs": self.n_runs,
            "threshold": self.threshold,
            "temperature": self.temperature,
            "summary": {
                "cases": len(self.cases),
                "cases_passed": sum(c.passed(self.threshold) for c in self.cases),
                "accuracy": self.accuracy,
                "tool_selection_rate": self.tool_selection_rate,
                "arg_accuracy": self.arg_accuracy,
                "mean_latency_ms": self.mean_latency_ms,
            },
            "cases": [c.to_dict() for c in self.cases],
        }


# ─── Rendering ───────────────────────────────────────────────────────────────


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _ms(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0f}ms"


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
        f"`threshold={r.threshold:.0%}` · `temperature="
        f"{'default' if r.temperature is None else r.temperature}` · {r.generated_at}",
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
        f"| mean latency | {_ms(r.mean_latency_ms)} |",
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
        lat_part = f" · ~{_ms(c.mean_latency_ms)}" if c.mean_latency_ms is not None else ""
        out += [
            "",
            f"### {c.id} — {verdict} "
            f"(tool {_pct(c.tool_selection_rate)}{args_part} · consistency {_pct(c.consistency)}"
            f"{lat_part})",
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


# ─── Layer 4 rendering (output / free-text correctness) ──────────────────────
#
# Layer 4 records a different shape (final_text + checks, not tool/args), so it
# gets its own renderer. It reads a plain report dict (built in runner_l4), which
# keeps this function pure and testable from a literal dict — no model, no engine.


def _checks_cell(checks: list[dict]) -> str:
    if not checks:
        return "—"
    parts = [
        f"{c.get('name')} exp={c.get('expected')} obs={c.get('observed')} "
        f"{'✓' if c.get('passed') else '✗'}"
        for c in checks
    ]
    return "; ".join(parts).replace("|", "\\|")


def _answer_cell(final_text: str, error: str | None) -> str:
    if error:
        return ("⚠ " + error).replace("|", "\\|")[:80]
    s = " ".join((final_text or "").split())  # collapse newlines for the table
    if len(s) > 70:
        s = s[:67] + "..."
    return s.replace("|", "\\|") or "∅"


def render_markdown_l4(d: dict) -> str:
    """Render a Layer 4 report dict to Markdown. Shows, per run: rounds, the two
    separate signals (tools-called / oracle), every Check (expected vs observed),
    and the model's answer — the evidence the README asks Layer 4 to surface."""
    s = d.get("summary", {})
    hdr_temp = "default" if d.get("temperature") is None else d["temperature"]
    out: list[str] = [
        "# Layer 4 — Output report",
        "",
        f"`target={d['target']}` · `model={d['model']}` · `n={d['n_runs']}` · "
        f"`threshold={d['threshold']:.0%}` · `temperature={hdr_temp}` · {d['generated_at']}",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---|",
        f"| cases | {s.get('cases', 0)} |",
        f"| cases passing threshold | {s.get('cases_passed', 0)}/{s.get('cases', 0)} "
        f"({_pct(s.get('accuracy'))}) |",
        f"| oracle rate (all runs) | {_pct(s.get('oracle_rate'))} |",
        f"| tool-use rate (all runs) | {_pct(s.get('tool_use_rate'))} |",
        f"| mean latency | {_ms(s.get('mean_latency_ms'))} |",
        "",
        "## Cases",
    ]
    threshold = d["threshold"]
    for c in d.get("cases", []):
        runs = c.get("runs", [])
        n = len(runs)
        n_pass = sum(1 for r in runs if r.get("passed"))
        n_oracle = sum(1 for r in runs if r.get("oracle_ok"))
        n_tools = sum(1 for r in runs if r.get("tools_called_ok"))
        lat_vals = [r["latency_ms"] for r in runs if r.get("latency_ms") is not None]
        mean_lat = sum(lat_vals) / len(lat_vals) if lat_vals else None
        verdict = "PASS" if c.get("pass_rate", 0.0) >= threshold else "FAIL"
        lat_part = f" · ~{_ms(mean_lat)}" if mean_lat is not None else ""
        out += [
            "",
            f"### {c['id']} — {verdict} "
            f"(pass-rate {n_pass}/{n} · oracle {n_oracle}/{n} · tools {n_tools}/{n}{lat_part})",
            f"- **prompt:** {c['prompt']}",
            "",
            "| # | rounds | tools | oracle | checks | answer |",
            "|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(runs, start=1):
            out.append(
                f"| {i} | {r.get('rounds', '')} | "
                f"{'✓' if r.get('tools_called_ok') else '✗'} | "
                f"{'✓' if r.get('oracle_ok') else '✗'} | "
                f"{_checks_cell(r.get('checks', []))} | "
                f"{_answer_cell(r.get('final_text', ''), r.get('error'))} |"
            )
    out.append("")
    return "\n".join(out)


def write_report_l4(d: dict, base_dir: str | Path = "reports") -> tuple[Path, Path]:
    """Persist a Layer 4 report dict as `.json` + `.md` under
    `base_dir/layer4/<timestamp>__<target>__<model>.*`. Returns the two paths."""
    out_dir = Path(base_dir) / "layer4"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = d["generated_at"].replace("-", "").replace(":", "")
    stem = f"{ts}__{_slug(d['target'])}__{_slug(d['model'])}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown_l4(d), encoding="utf-8")
    return json_path, md_path


# ─── Layer 5 rendering (security / red-team — ASR) ───────────────────────────
#
# Different question, different bar: success is BAD and any ASR > 0 is a breach.
# The report keeps the trajectory (tools called + answer) so a breach is auditable.


def render_markdown_l5(d: dict) -> str:
    """Render a Layer 5 (security) report dict to Markdown. Per run: whether the
    attack succeeded, the detector's evidence, the tools the agent called, and
    its answer."""
    s = d.get("summary", {})
    hdr_temp = "default" if d.get("temperature") is None else d["temperature"]
    out: list[str] = [
        "# Layer 5 — Security (red-team) report",
        "",
        f"`target={d['target']}` · `model={d['model']}` · `n={d['n_runs']}` · "
        f"`temperature={hdr_temp}` · {d['generated_at']}",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---|",
        f"| attack cases | {s.get('cases', 0)} |",
        f"| cases breached (ASR>0) | {s.get('cases_breached', 0)}/{s.get('cases', 0)} |",
        f"| overall ASR | {_pct(s.get('overall_asr'))} (bar: 0%) |",
        f"| mean latency | {_ms(s.get('mean_latency_ms'))} |",
        "",
        "## Cases",
    ]
    for c in d.get("cases", []):
        runs = c.get("runs", [])
        n = len(runs)
        n_succ = sum(1 for r in runs if r.get("succeeded"))
        verdict = "BREACH" if c.get("asr", 0.0) > 0 else "resisted"
        out += [
            "",
            f"### {c['id']} — {verdict} "
            f"(ASR {n_succ}/{n} · {c.get('technique', '')} · {c.get('attack_kind', '')})",
            f"- **prompt:** {c['prompt']}",
            "",
            "| # | rounds | succeeded | evidence | tools called | answer |",
            "|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(runs, start=1):
            evidence = (r.get("evidence", "") or "").replace("|", "\\|")
            tools = ", ".join(t.get("name", "") for t in r.get("tools_called", [])) or "—"
            tools = tools.replace("|", "\\|")
            answer = _answer_cell(r.get("final_text", ""), r.get("error"))
            succeeded = "⚠ YES" if r.get("succeeded") else "no"
            out.append(
                f"| {i} | {r.get('rounds', '')} | {succeeded} | {evidence} | {tools} | {answer} |"
            )
    out.append("")
    return "\n".join(out)


def write_report_l5(d: dict, base_dir: str | Path = "reports") -> tuple[Path, Path]:
    """Persist a Layer 5 report dict as `.json` + `.md` under
    `base_dir/layer5/<timestamp>__<target>__<model>.*`. Returns the two paths."""
    out_dir = Path(base_dir) / "layer5"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = d["generated_at"].replace("-", "").replace(":", "")
    stem = f"{ts}__{_slug(d['target'])}__{_slug(d['model'])}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown_l5(d), encoding="utf-8")
    return json_path, md_path
