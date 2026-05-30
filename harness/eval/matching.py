"""Oracles for Layer 3: did the model call the right tool with the right args?

These are deterministic, IA-judge-free checks — the whole point of Layer 3. We
verify *which* tool was called and *whether* expected arguments are present, not
the quality of any answer.

Argument matching is recursive on purpose. Many MCP tools wrap their parameters
(e.g. this project's tools take a single `params` object, so the model emits
`{"params": {"run_id": "..."}}`). A flat comparison would miss that. `args_contain`
searches the whole nested structure, so a dataset can specify the *semantic*
expectation `{"run_id": "..."}` without caring how the tool wraps it.
"""

from __future__ import annotations

from typing import Any


def tool_matches(actual_tool: str | None, expected_tool: str) -> bool:
    """Exact tool-name match. `None` (model called no tool) never matches."""
    return actual_tool is not None and actual_tool == expected_tool


def args_contain(actual_input: Any, expected_contains: dict) -> bool:
    """True iff every (key, value) in `expected_contains` appears somewhere in
    `actual_input` (searched recursively through dicts and lists)."""
    return all(_found_kv(actual_input, k, v) for k, v in expected_contains.items())


def _found_kv(obj: Any, key: str, value: Any) -> bool:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and v == value:
                return True
            if _found_kv(v, key, value):
                return True
    elif isinstance(obj, list):
        return any(_found_kv(item, key, value) for item in obj)
    return False


def run_signals(
    actual_tool: str | None,
    actual_input: Any,
    expected_tool: str,
    expected_args_contains: dict | None,
) -> tuple[bool, bool]:
    """The two signals for one run, kept apart: ``(tool_ok, args_ok)``.

    Collapsing these into one verdict (as `case_passes` does) hides WHICH half
    failed — and that is precisely the diagnostic a report needs. `args_ok` is
    `True` when no args were expected (nothing to check) or when the expected
    args are present; it does NOT require the tool to match, so the two signals
    are independent.
    """
    tool_ok = tool_matches(actual_tool, expected_tool)
    args_ok = True if not expected_args_contains else args_contain(
        actual_input or {}, expected_args_contains
    )
    return tool_ok, args_ok


def case_passes(
    actual_tool: str | None,
    actual_input: Any,
    expected_tool: str,
    expected_args_contains: dict | None,
) -> bool:
    """Full per-run verdict: right tool, and (if specified) expected args present."""
    tool_ok, args_ok = run_signals(
        actual_tool, actual_input, expected_tool, expected_args_contains
    )
    return tool_ok and args_ok
