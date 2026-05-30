"""Schema conversion tests.

Two flavors:
- pure: convert duck-typed fake tools (no MCP needed).
- integration-lite: convert the REAL tools from the live MCP (needs the MCP
  running, but NO model/API key).
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.eval.schema_convert import to_anthropic_tool, to_anthropic_tools


@dataclass
class _FakeTool:
    name: str
    description: str
    inputSchema: dict


def test_converts_single_tool():
    t = _FakeTool("foo", "  does foo  ", {"type": "object", "properties": {}})
    out = to_anthropic_tool(t)
    assert out == {
        "name": "foo",
        "description": "does foo",  # stripped
        "input_schema": {"type": "object", "properties": {}},
    }


def test_converts_list():
    tools = [
        _FakeTool("a", "A", {"type": "object"}),
        _FakeTool("b", "B", {"type": "object"}),
    ]
    out = to_anthropic_tools(tools)
    assert [t["name"] for t in out] == ["a", "b"]
    assert all("input_schema" in t for t in out)


async def test_converts_real_mcp_tools(connect):
    """Convert the live MCP's actual tools/list into Anthropic shape. Proves the
    end-to-end plumbing (connect → list_tools → convert) without a model."""
    async with connect() as session:
        result = await session.list_tools()
    anthropic_tools = to_anthropic_tools(result.tools)
    assert len(anthropic_tools) >= 1
    for t in anthropic_tools:
        assert t["name"]
        assert t["description"]
        assert t["input_schema"]["type"] == "object"
