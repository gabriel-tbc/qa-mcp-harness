"""vuln-lab — a deliberately injectable toy MCP server used as a Layer 5 target.

It is a *fixture*, not part of the harness core: the harness connects to it
black-box over stdio, exactly like any other target. It lives in-repo only so the
red-team layer has something safe to attack. See `vuln_lab/server.py`.
"""
