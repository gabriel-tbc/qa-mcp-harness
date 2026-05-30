"""Shared fixtures: wire the active target into an MCP session.

Connection note: we expose a `connect()` *factory* rather than a yielded
`session` fixture. Tests open the session inside their own body
(`async with connect() as session: ...`). This keeps the connection's enter
and exit in the *same* asyncio task, which avoids anyio's
"cancel scope in a different task" error that occurs when pytest-asyncio runs
an async-generator fixture's teardown in a separate task.
"""

from __future__ import annotations

import pytest

from harness.config import Target, active_target
from harness.clients.mcp_client import open_session


@pytest.fixture(scope="session")
def target() -> Target:
    """The MCP under test, selected by HARNESS_TARGET."""
    return active_target()


@pytest.fixture
def connect(target: Target):
    """Return a factory that opens an MCP session as an async context manager.

    Usage:
        async def test_x(connect):
            async with connect() as session:
                ...
    """
    def _open():
        return open_session(target)

    return _open
