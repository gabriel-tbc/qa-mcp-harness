"""Layer 4 scaffold — deterministic oracles for free-text answers (NOT YET
IMPLEMENTED).

The chosen approach is GROUND-TRUTH + EXTRACTION, not substring matching:

  1. Compute the truth by calling the MCP tool directly — e.g. the real
     regression count from `qa_compare_runs`. The oracle is *computed*, so it
     can't drift from the data and needs no AI judge.
  2. Extract the matching claim from the model's free text (a number, an ID) —
     structured extraction, NOT ``"1" in text`` (which also matches "10", "11").
  3. Compare extracted vs ground truth → a `Check`.

Why not plain `contains`? It's brittle on two axes: substrings collide
("1" ⊂ "10"), and a hardcoded expected string goes stale the moment the
underlying data changes. Ground truth read from the source is robust and
auditable. These are stubs; see `tests/layer4_output/README.md` for the design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Check:
    """One verifiable claim about the answer: what we expected (ground truth),
    what we observed (extracted from the text), and whether they match. Layer 4
    reports render one row per `Check`."""

    name: str
    kind: str  # "ground-truth" | "structural" | "metamorphic" | ...
    expected: Any
    observed: Any
    passed: bool


async def ground_truth(session: Any, tool: str, args: dict) -> dict:
    """Call the MCP tool directly to obtain the authoritative value to check the
    model's prose against. NOT IMPLEMENTED — design-only for now."""
    raise NotImplementedError("Layer 4 ground-truth oracle is scaffolded, not implemented.")


def extract_number(text: str, label: str) -> int | None:
    """Pull the integer associated with `label` out of free text (e.g. the count
    reported after 'regressions'). Structured extraction, not substring matching.

    NOT IMPLEMENTED — the parse strategy is part of the Layer 4 slice.
    """
    raise NotImplementedError("Layer 4 fact extraction is scaffolded, not implemented.")
