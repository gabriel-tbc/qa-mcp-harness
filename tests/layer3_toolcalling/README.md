# Layer 3 — Tool-calling accuracy (the semi-indeterministic layer)

**Question this layer answers:** given a natural-language prompt and access to
the MCP's tools, does the model pick the *right tool* with the *right
arguments*? And does it do so *reliably* across repeated runs?

This is the layer where determinism breaks: the same prompt can yield different
tool choices on different runs. We don't fight that — we **measure** it.

## How it works (runner, to be built next)

For each case in a dataset:

1. Connect to the target MCP (via `harness.clients.mcp_client`), pull its real
   `tools/list`.
2. Hand those tools to a model along with the prompt.
3. Capture the **first tool call** the model makes (name + arguments).
4. Repeat N times. Compute **pass-rate** = (runs where the call matched the
   expectation) / N.
5. The case passes if pass-rate ≥ threshold (e.g. 0.9).

This needs a real model, so it requires `ANTHROPIC_API_KEY` and the `llm`
extra (`pip install -e ".[dev,llm]"`). Tests skip cleanly when the key is absent.

## Dataset format (`datasets/*.jsonl`)

One JSON object per line:

```json
{"id": "TC-001", "prompt": "What test runs do I have?", "expected_tool": "qa_list_runs"}
{"id": "TC-002", "prompt": "What failed in run search-25_classification?", "expected_tool": "qa_get_run", "expected_args_contains": {"run_id": "search-25_classification"}}
{"id": "TC-003", "prompt": "Compare search-25_classification against search-26_classification", "expected_tool": "qa_compare_runs"}
```

- `expected_tool` — the tool name the model should select.
- `expected_args_contains` (optional) — a subset of arguments that must be
  present (argument accuracy). Validated as "contains", not exact equality,
  because extra optional args are fine.

## Why this is the valuable layer

The oracle here is cheap and IA-judge-free: we check *which tool was called*,
not *whether the answer was good*. That's a structural check, not a judgment.
It directly measures whether the MCP's tool names / descriptions / schemas are
good enough for a model to use — i.e. it tests the **MCP's design**, not its code.
