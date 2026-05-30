"""Load Layer 3 eval datasets (JSONL).

Each line is one case:
    {"id", "prompt", "expected_tool", "expected_args_contains"?}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvalCase:
    id: str
    prompt: str
    expected_tool: str
    expected_args_contains: dict | None = None


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
            )
        )
    return cases
