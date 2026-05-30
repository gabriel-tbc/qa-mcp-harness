"""Layer 2 — target-specific contract tests for qa-toolkit-mcp.

This is the *specialized* counterpart to the generic smoke suite: it knows
which tools qa-toolkit exposes and checks their concrete contract and response
shape. When you point the harness at a different MCP, you write a file like
this for that MCP; the generic suite keeps working unchanged.

These tests are skipped automatically if the active target isn't qa-toolkit.
"""

from __future__ import annotations

import pytest

from harness.clients.mcp_client import result_text

EXPECTED_TOOLS = {"qa_list_runs", "qa_get_run", "qa_compare_runs"}


@pytest.fixture(autouse=True)
def _only_qa_toolkit(target):
    if not target.name.startswith("qa-toolkit"):
        pytest.skip(f"target {target.name!r} is not qa-toolkit; skipping specific contract")


async def test_exposes_expected_tools(connect):
    async with connect() as session:
        result = await session.list_tools()
        names = {t.name for t in result.tools}
        assert EXPECTED_TOOLS.issubset(names), f"missing tools: {EXPECTED_TOOLS - names}"


async def test_qa_list_runs_returns_well_formed_payload(connect):
    async with connect() as session:
        result = await session.call_tool(
            "qa_list_runs", {"params": {"response_format": "json"}}
        )
    assert not result.isError, "qa_list_runs reported an error"
    text = result_text(result)
    assert "total" in text, f"unexpected payload: {text[:200]}"


async def test_qa_get_run_unknown_id_is_actionable(connect):
    """A non-existent run should yield an actionable error string, not a crash —
    the message should point the model at qa_list_runs."""
    async with connect() as session:
        result = await session.call_tool(
            "qa_get_run", {"params": {"run_id": "definitely-not-a-real-run"}}
        )
    text = result_text(result)
    assert "Error" in text
    assert "qa_list_runs" in text, "error message should guide the model to a next step"
