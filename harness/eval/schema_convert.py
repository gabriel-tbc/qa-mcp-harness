"""Convert MCP tool definitions into the Anthropic tool format.

The model the harness drives (via the Anthropic SDK) expects tools shaped as
`{"name", "description", "input_schema"}`. An MCP tool from `tools/list` exposes
`.name`, `.description`, `.inputSchema`. This is a pure, mechanical mapping — and
because it's pure, it's cheap to test without any model or network.
"""

from __future__ import annotations

from typing import Any, Iterable


def to_anthropic_tool(mcp_tool: Any) -> dict:
    """Map one MCP tool (duck-typed: .name/.description/.inputSchema) to Anthropic shape."""
    return {
        "name": mcp_tool.name,
        "description": (mcp_tool.description or "").strip(),
        "input_schema": mcp_tool.inputSchema,
    }


def to_anthropic_tools(mcp_tools: Iterable[Any]) -> list[dict]:
    """Map a list of MCP tools to the Anthropic `tools` argument."""
    return [to_anthropic_tool(t) for t in mcp_tools]
