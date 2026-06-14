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
    # Which extraction strategy fired (for ground-truth-number). Lets reports
    # show whether `observed` came from "the model wrote '1' alone" vs "we
    # found it in the same sentence as the label" — diagnostic, not gating.
    strategy: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "expected": self.expected,
            "observed": self.observed,
            "passed": self.passed,
            "strategy": self.strategy,
        }


# ─── Ground truth: call the MCP tool directly ────────────────────────────────


async def ground_truth(session: Any, tool: str, args: dict) -> dict:
    """Invoke a tool on the MCP and return its parsed JSON payload.

    qa-toolkit-mcp tools return JSON as a single text content block when called
    with `response_format='json'`. We parse it. If the tool returned an error
    or non-JSON text, we surface that as an `{"_error": ...}` dict so callers
    can decide what to do (Checks then fail explicitly, not silently).
    """
    # The caller passes the tool's arguments as a flat dict (top-level keys);
    # we pass them straight through to the MCP.
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

# Extraction strategies. Tried in order; the FIRST that fires wins. The order
# is from MOST CONFIDENT (the whole text is a bare integer) to LEAST CONFIDENT
# (we relax to "label and a number are in the same sentence"). Each strategy is
# narrow on purpose so it either fires confidently or abstains — abstention
# beats a wrong answer, since the report shows None and you investigate.


def _strategy_bare_integer(text: str) -> int | None:
    """The whole trimmed text is a single integer. Handles prompts like
    'return only the integer' where the model obeys and skips the prose
    entirely. Most confident — no label disambiguation needed."""
    m = re.fullmatch(r"\s*(-?\d+)\s*", text)
    return int(m.group(1)) if m else None


def _strategy_adjacent_to_label(label: str, text: str) -> int | None:
    """The number sits right next to the label: 'regressions: 3', 'regression
    is 3', '3 regressions'. Explicit assignment is preferred over juxtaposition
    (`N label`) so a stray '5 fixes' nearby can't steal the integer that
    belongs to the next label over."""
    label_re = re.escape(label)
    # Plural suffix: 's' (regression→regressions) OR 'es' (fix→fixes).
    plural = r"(?:e?s)?"
    patterns = [
        rf"\b{label_re}{plural}\s*[:=]\s*(\d+)\b",
        rf"\b{label_re}{plural}\s+(?:count\s+)?(?:is|are|was|were)\s+(\d+)\b",
        rf"\b(\d+)\s+(?:\w+\s+)?{label_re}{plural}\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _strategy_same_sentence(label: str, text: str) -> int | None:
    """Loose fallback: words intervene between the label and the number, but
    they share a sentence. 'The number of regressions between runs A and B is 1.'
    Only fires if exactly ONE integer sits in that sentence — multiple ints
    are ambiguous, so we abstain rather than guess wrong."""
    label_re = re.escape(label)
    plural = r"(?:e?s)?"
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    found: list[int] = []
    for sent in sentences:
        if not re.search(rf"\b{label_re}{plural}\b", sent, flags=re.IGNORECASE):
            continue
        ints = [int(x) for x in re.findall(r"\b(\d+)\b", sent)]
        if len(ints) == 1:
            found.append(ints[0])
    # Multiple label-bearing sentences that agree → confident; that disagree → abstain.
    unique = set(found)
    return next(iter(unique)) if len(unique) == 1 else None


def extract_number(text: str, label: str) -> int | None:
    """Pull the integer the model associated with `label` out of free text.

    Returns None when no unambiguous mention is found — callers treat that as
    a failed Check, not as zero (zero is a valid answer; absence is not).

    Three strategies, first-match wins:
      1. bare_integer       — the whole text is just a number.
      2. adjacent_to_label  — the number sits right next to the label.
      3. same_sentence      — fallback for prose with words between them.
    """
    return _extract_number_with_strategy(text, label)[0]


def _extract_number_with_strategy(text: str, label: str) -> tuple[int | None, str | None]:
    """Same as `extract_number`, but also returns WHICH strategy fired (or
    None). Internal helper — lets Check report the matching strategy in the
    JSON, so a number that came from `bare_integer` is distinguishable from
    one that came from `same_sentence` when diagnosing a flaky case."""
    bare = _strategy_bare_integer(text)
    if bare is not None:
        return bare, "bare_integer"
    near = _strategy_adjacent_to_label(label, text)
    if near is not None:
        return near, "adjacent_to_label"
    same = _strategy_same_sentence(label, text)
    if same is not None:
        return same, "same_sentence"
    return None, None


# ─── Compose into a Check ────────────────────────────────────────────────────


def number_check(name: str, expected: int, text: str, label: str) -> Check:
    """A ground-truth-number Check: expected from the MCP, observed extracted
    from the model's prose, passed if they match exactly. Records which
    extraction strategy fired so a failed run is traceable."""
    observed, strategy = _extract_number_with_strategy(text, label)
    return Check(
        name=name,
        kind="ground-truth-number",
        expected=expected,
        observed=observed,
        passed=observed is not None and observed == expected,
        strategy=strategy,
    )
