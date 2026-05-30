"""The keystone: a transport-agnostic way to open an MCP client session.

`open_session(target)` is an async context manager that yields a fully
initialized `ClientSession` regardless of whether the target speaks stdio or
Streamable HTTP. Everything above this layer (the test suites, the eval runner)
is written against `ClientSession` and never needs to know the transport.

This mirrors exactly how a real MCP host connects to a server — the harness IS
an MCP client. It never imports the server's source; it only connects to it.
"""

from __future__ import annotations

import contextlib
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from harness.config import Target


@contextlib.asynccontextmanager
async def open_session(target: Target) -> AsyncIterator[ClientSession]:
    """Open and initialize an MCP session against `target`.

    Usage:
        async with open_session(target) as session:
            tools = await session.list_tools()
    """
    if target.transport == "stdio":
        params = StdioServerParameters(
            command=target.command or "",
            args=list(target.args),
            env=dict(target.env) or None,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
        return

    if target.transport == "http":
        # Imported lazily so stdio-only environments don't depend on the
        # streamable-http client module being importable.
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(
            target.url or "", headers=dict(target.headers)
        ) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
        return

    raise ValueError(f"Unsupported transport: {target.transport!r}")


def result_text(call_result: Any) -> str:
    """Extract the concatenated text payload from a CallToolResult.

    Tools may return one or more content blocks; we join the text ones. This is
    a convenience for assertions; structured-content handling can be added later.
    """
    parts: list[str] = []
    for block in getattr(call_result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)
