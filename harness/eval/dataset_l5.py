"""Layer 5 dataset — injection attack cases.

Each line is one attack scenario:

    {
      "id", "technique", "prompt",
      "system_policy"?,                 # the agent's DEFENSIVE policy under test
      "system_prompt_override"?,
      "documents": {doc_id: body},      # served by vuln-lab; the payloads live here
      "attack": {"kind": "forbidden_tool", "tool": "send_message"}
               | {"kind": "canary", "value": "CANARY-..."}
    }

The `documents` map is consumed by the toy server (`vuln_lab/server.py`); the
runner reads `prompt`, `attack`, `technique`, and the policy fields. One file is
both the payload corpus and the eval set, so they can never drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from harness.eval.redteam import AttackSpec


@dataclass(frozen=True)
class AttackCase:
    id: str
    prompt: str
    attack: AttackSpec
    technique: str = ""
    system_policy: str | None = None
    system_prompt_override: str | None = None


def _attack_from_obj(obj: dict, lineno: int, fname: str) -> AttackSpec:
    if "kind" not in obj:
        raise ValueError(f"{fname}:{lineno}: attack missing 'kind'")
    kind = obj["kind"]
    if kind == "forbidden_tool":
        if "tool" not in obj:
            raise ValueError(f"{fname}:{lineno}: forbidden_tool attack missing 'tool'")
        return AttackSpec(kind="forbidden_tool", tool=obj["tool"])
    if kind == "canary":
        if "value" not in obj:
            raise ValueError(f"{fname}:{lineno}: canary attack missing 'value'")
        return AttackSpec(kind="canary", canary=obj["value"])
    raise ValueError(f"{fname}:{lineno}: unknown attack kind {kind!r}")


def load_l5_jsonl(path: str | Path) -> list[AttackCase]:
    p = Path(path)
    cases: list[AttackCase] = []
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p.name}:{lineno}: invalid JSON: {exc.msg}") from exc
        for required in ("id", "prompt", "attack"):
            if required not in obj:
                raise ValueError(f"{p.name}:{lineno}: missing '{required}'")
        cases.append(
            AttackCase(
                id=obj["id"],
                prompt=obj["prompt"],
                attack=_attack_from_obj(obj["attack"], lineno, p.name),
                technique=obj.get("technique", ""),
                system_policy=obj.get("system_policy"),
                system_prompt_override=obj.get("system_prompt_override"),
            )
        )
    return cases
