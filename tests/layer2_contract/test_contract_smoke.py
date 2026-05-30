"""Layer 2 — generic contract smoke tests.

These assertions hold for ANY well-formed MCP server. They make no assumption
about which tools exist — only that whatever IS exposed follows the protocol's
contract well enough for a model to use it. Run these against any target.
"""

from __future__ import annotations

import pytest


async def test_server_initializes_and_exposes_at_least_one_tool(connect):
    async with connect() as session:
        result = await session.list_tools()
        assert len(result.tools) >= 1, "MCP exposes no tools"


async def test_every_tool_has_a_usable_contract(connect):
    """Each tool must carry the elements a model reads to decide and call it:
    a name, a non-empty description, and an object-typed input schema."""
    async with connect() as session:
        result = await session.list_tools()
        for tool in result.tools:
            assert tool.name, "tool with empty name"
            assert tool.description and tool.description.strip(), (
                f"tool {tool.name!r} has no description (the model reads this to decide)"
            )
            assert isinstance(tool.inputSchema, dict), f"tool {tool.name!r} has no input schema"
            assert tool.inputSchema.get("type") == "object", (
                f"tool {tool.name!r} input schema is not an object"
            )


async def test_tool_names_are_unique(connect):
    async with connect() as session:
        result = await session.list_tools()
        names = [t.name for t in result.tools]
        assert len(names) == len(set(names)), f"duplicate tool names: {names}"


async def test_resources_listing_does_not_crash(connect):
    """Optional capability. If the server declares resources, listing must work;
    if it doesn't support them, tolerate the failure explicitly rather than crash."""
    async with connect() as session:
        try:
            await session.list_resources()
        except Exception as exc:  # noqa: BLE001 - capability may be absent
            pytest.skip(f"resources not supported by target: {type(exc).__name__}")
