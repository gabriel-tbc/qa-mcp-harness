"""Layer 4 oracles — deterministic, IA-judge-free.

The chosen approach is GROUND TRUTH + EXTRACTION:

    1. Compute the truth by calling the MCP tool directly (`ground_truth`).
    2. Extract the matching claim from the model's prose (`extract_number`).
    3. Compare → a `Check`.

Why not `"1" in text`? It also matches "10", "11", "100". Why not a hardcoded
expected value? It goes stale the moment the data changes. Ground truth from
the source is robust AND auditable.

These primitives are kept narrow on purpose. A richer Check vocabulary (IDs,
booleans, sets) grows from real failures in Layer 4, not preemptively.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class Check:
    """One verifiable claim about the model's answer."""

    name: str
    kind: str       # "ground-truth-number" | (future: "ground-truth-id", ...)
    expected: Any
    observed: Any
    passed: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "expected": self.expected,
            "observed": self.observed,
            "passed": self.passed,
        }


# ─── Ground truth: call the MCP tool directly ────────────────────────────────


async def ground_truth(session: Any, tool: str, args: dict) -> dict:
    """Invoke a tool on the MCP and return its parsed JSON payload.

    qa-toolkit-mcp tools return JSON as a single text content block when called
    with `response_format='json'`. We parse it. If the tool returned an error
    or non-JSON text, we surface that as an `{"_error": ...}` dict so callers
    can decide what to do (Checks then fail explicitly, not silently).
    """
    # Tools in this MCP wrap arguments under `params` — caller passes that.
    raw = await session.call_tool(tool, args)
    if getattr(raw, "isError", False):
        return {"_error": _result_text(raw)}
    text = _result_text(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_error": "non-JSON payload", "_text": text}


def _result_text(raw: Any) -> str:
    parts: list[str] = []
    for block in getattr(raw, "content", []) or []:
        t = getattr(block, "text", None)
        if t is not None:
            parts.append(t)
    # Some MCP tools may wrap their string under {"result": "..."}; unwrap if so.
    text = "\n".join(parts)
    try:
        peeled = json.loads(text)
        if isinstance(peeled, dict) and "result" in peeled and isinstance(peeled["result"], str):
            return peeled["result"]
    except json.JSONDecodeError:
        pass
    return text


# ─── Extraction: pull a structured value out of free text ────────────────────

# Extract the integer associated with `label` from prose, structurally.
# Patterns are tried in order; the FIRST match wins. Explicit assignment
# (`label: N`, `label = N`, `label is N`) wins over juxtaposition, because
# juxtaposition can steal a number that belongs to a neighbouring label
# (e.g. "regressions: 5 fixes: 0" — looking for 'fix', "5 fixes" would
# wrongly fire if `N + label` ran first).
def _number_near(label: str, text: str) -> int | None:
    label_re = re.escape(label)
    # Plural suffix: 's' (regression→regressions) OR 'es' (fix→fixes).
    plural = r"(?:e?s)?"
    patterns = [
        # label: 3 / label = 3 / label is 3 / label count is 3
        rf"\b{label_re}{plural}\s*[:=]\s*(\d+)\b",
        rf"\b{label_re}{plural}\s+(?:count\s+)?(?:is|are|was|were)\s+(\d+)\b",
        # Number immediately precedes the label, optionally with one adjective
        # ("3 regressions", "1 real regression"). Bounded to one word between
        # so it can't span across an unrelated label.
        rf"\b(\d+)\s+(?:\w+\s+)?{label_re}{plural}\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def extract_number(text: str, label: str) -> int | None:
    """Pull the integer the model associated with `label` out of free text.

    Returns None when no unambiguous mention is found — callers treat that as
    a failed Check, not as zero (zero is a valid answer; absence is not).
    """
    return _number_near(label, text)


# ─── Compose into a Check ────────────────────────────────────────────────────


def number_check(name: str, expected: int, text: str, label: str) -> Check:
    """A ground-truth-number Check: expected from the MCP, observed extracted
    from the model's prose, passed if they match exactly."""
    observed = extract_number(text, label)
    return Check(
        name=name,
        kind="ground-truth-number",
        expected=expected,
        observed=observed,
        passed=observed is not None and observed == expected,
    )
