"""Harness configuration: which MCP to test, and how to connect to it.

The harness is MCP-agnostic. A *target* describes a single MCP under test:
its transport (stdio or http) and the parameters to reach it. Targets live as
TOML files under `targets/`. The active target is selected by the
`HARNESS_TARGET` env var (a target filename without extension); default is
`qa-toolkit-local`.

This indirection is the whole point: to test a different MCP — including a
remote HTTP one you don't own — you add a target file and flip an env var. No
code changes.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TARGETS_DIR = _PROJECT_ROOT / "targets"

# Load .env once (real env vars win — override=False).
load_dotenv(_PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Target:
    """A single MCP under test."""

    name: str
    transport: str  # "stdio" | "http"
    # stdio
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # http
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


def load_target(path: Path) -> Target:
    """Parse a target TOML file into a Target."""
    if not path.is_file():
        raise FileNotFoundError(f"Target file not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    transport = data.get("transport")

    if transport == "stdio":
        s = data.get("stdio", {})
        if "command" not in s:
            raise ValueError(f"stdio target {path.name} missing [stdio].command")
        return Target(
            name=data.get("name", path.stem),
            transport="stdio",
            command=s["command"],
            args=list(s.get("args", [])),
            env={str(k): str(v) for k, v in s.get("env", {}).items()},
        )

    if transport == "http":
        h = data.get("http", {})
        if "url" not in h:
            raise ValueError(f"http target {path.name} missing [http].url")
        return Target(
            name=data.get("name", path.stem),
            transport="http",
            url=h["url"],
            headers={str(k): str(v) for k, v in h.get("headers", {}).items()},
        )

    raise ValueError(f"Target {path.name}: unknown or missing transport {transport!r}")


def active_target() -> Target:
    """Resolve the target selected by HARNESS_TARGET (default: qa-toolkit-local)."""
    name = os.environ.get("HARNESS_TARGET", "qa-toolkit-local")
    return load_target(_TARGETS_DIR / f"{name}.toml")


def anthropic_api_key() -> str | None:
    """API key for Layer 3 evals. None if unset (Layer 3 tests should skip)."""
    return os.environ.get("ANTHROPIC_API_KEY") or None
