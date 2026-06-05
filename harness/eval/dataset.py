"""Load Layer 3 eval datasets (JSONL).

Each line is one case:
    {
        "id", "prompt", "expected_tool",
        "expected_args_contains"?,        # arg-accuracy expectation
        "system_policy"?,                  # name of a file in system_prompts/
        "system_prompt_override"?          # one-off; substitutes the policy
    }

System prompt composition (see harness/eval/prompts.py):
- `system_policy` references a suite-level policy file (versioned, reusable).
- `system_prompt_override` is per-case and SUBSTITUTES the policy (not append).
  Substitution is deliberate: silent contradictions are the worst eval bug.
- If neither is set, the call goes out with no system prompt — a useful baseline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    id: str
    prompt: str
    expected_tool: str
    expected_args_contains: dict | None = None
    system_policy: str | None = None
    system_prompt_override: str | None = None


def load_jsonl(path: str | Path) -> list[EvalCase]:
    """Parse a .jsonl dataset into EvalCase objects. Blank lines are ignored."""
    p = Path(path)
    cases: list[EvalCase] = []
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p.name}:{lineno}: invalid JSON: {exc.msg}") from exc
        for required in ("id", "prompt", "expected_tool"):
            if required not in obj:
                raise ValueError(f"{p.name}:{lineno}: missing '{required}'")
        cases.append(
            EvalCase(
                id=obj["id"],
                prompt=obj["prompt"],
                expected_tool=obj["expected_tool"],
                expected_args_contains=obj.get("expected_args_contains"),
                system_policy=obj.get("system_policy"),
                system_prompt_override=obj.get("system_prompt_override"),
            )
        )
    return cases
