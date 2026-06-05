"""Layer 4 dataset — cases that score the model's final text answer.

A Layer 4 case is shaped differently from Layer 3:

    {
        "id", "prompt",
        "system_policy"?,
        "system_prompt_override"?,
        "checks": [                    # the deterministic oracles
            {
                "name": "regression_count",
                "kind": "ground-truth-number",
                "ground_truth_tool": "qa_compare_runs",
                "ground_truth_args": { "params": { "run_a": "...", "run_b": "..." } },
                "ground_truth_path": "counts.regressions",   # dotted path into result
                "extract_label": "regression"                # noun to search in prose
            }
        ]
    }

The case PASSES one run iff every Check in `checks` passes for that run. Layer 4
keeps signals separate the same way Layer 3 does: in the report, `oracle_ok`
(did the prose match the truth?) is reported beside `tools_called_ok` (did the
loop call any tools at all?).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CheckSpec:
    name: str
    kind: str  # "ground-truth-number"
    ground_truth_tool: str
    ground_truth_args: dict
    ground_truth_path: str
    extract_label: str


@dataclass(frozen=True)
class EvalCaseL4:
    id: str
    prompt: str
    checks: tuple[CheckSpec, ...]
    system_policy: str | None = None
    system_prompt_override: str | None = None


def _check_from_obj(obj: dict, lineno: int, fname: str) -> CheckSpec:
    for required in (
        "name", "kind",
        "ground_truth_tool", "ground_truth_args",
        "ground_truth_path", "extract_label",
    ):
        if required not in obj:
            raise ValueError(f"{fname}:{lineno}: check missing '{required}'")
    return CheckSpec(
        name=obj["name"], kind=obj["kind"],
        ground_truth_tool=obj["ground_truth_tool"],
        ground_truth_args=dict(obj["ground_truth_args"]),
        ground_truth_path=obj["ground_truth_path"],
        extract_label=obj["extract_label"],
    )


def load_l4_jsonl(path: str | Path) -> list[EvalCaseL4]:
    p = Path(path)
    cases: list[EvalCaseL4] = []
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p.name}:{lineno}: invalid JSON: {exc.msg}") from exc
        for required in ("id", "prompt", "checks"):
            if required not in obj:
                raise ValueError(f"{p.name}:{lineno}: missing '{required}'")
        checks = tuple(_check_from_obj(c, lineno, p.name) for c in obj["checks"])
        cases.append(
            EvalCaseL4(
                id=obj["id"], prompt=obj["prompt"], checks=checks,
                system_policy=obj.get("system_policy"),
                system_prompt_override=obj.get("system_prompt_override"),
            )
        )
    return cases


def get_path(payload: dict, dotted: str) -> object:
    """Follow a dotted path into a parsed JSON dict (e.g. 'counts.regressions')."""
    cur: object = payload
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur
