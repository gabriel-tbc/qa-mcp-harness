# Layer 4 — Output indeterminacy (the free-text layer)

**Question this layer answers:** once the model has called the tools and writes
a final natural-language answer, is that answer *correct* — without using
another AI as the judge?

This is the hardest layer because the output is free text and there's no cheap
oracle. The strategy is to avoid LLM-as-judge as a gate and instead use
deterministic techniques, in rough order of preference:

1. **Structural / schema validation** — if the answer must contain a structured
   fact (an ID, a count, a status), parse it and assert.
2. **Fact extraction + verification** — extract a claim from the text
   ("regressions: 1", "run #search-26") and verify it against ground truth
   obtained by calling the MCP tools directly. Parsing + ground truth, not judgment.
3. **Metamorphic relations** — the answer to a transformed prompt must relate
   predictably to the original (e.g. swapping run A and B should swap
   regressions↔fixes in the narrative).
4. **Embedding similarity with calibrated thresholds** — deterministic math,
   not another model's opinion.
5. **Narrow-rubric LLM-as-judge** — only as a *complementary signal*, never as
   the pass/fail gate.

## Architecture — why Layer 4 needs a NEW runner (not Layer 3's)

Layer 3 captures the model's **first** `tool_use` and stops — it never executes
the tool (see `harness/eval/runner.py`). Layer 4 must run the **full agent loop**,
because the answer it scores only exists *after* the tool actually ran:

```
prompt + tools
  → model returns tool_use
  → execute it on the MCP via session.call_tool(name, args)
  → feed the tool_result back
  → repeat until the model returns final text
  → score that text
```

That loop is scaffolded (stub, not implemented) in
[`harness/eval/agent_loop.py`](../../harness/eval/agent_loop.py)
(`run_agent_turn → AgentOutcome(final_text, tool_calls)`).

## The oracle — ground-truth + extraction (chosen approach)

Not substring matching. The oracle is **computed from the MCP itself**:

1. Get the truth by calling the tool directly — e.g. the real regression count
   from `qa_compare_runs`.
2. **Extract** the matching claim from the model's prose (a number, an ID) —
   structured parsing, *not* `"1" in text` (which also matches "10", "11").
3. Compare extracted vs ground truth → a `Check`.

Stubbed in [`harness/eval/oracles.py`](../../harness/eval/oracles.py)
(`ground_truth(...)`, `extract_number(...)`, `Check`). Why not hardcoded
`contains`? It's brittle twice over: substrings collide, and a hardcoded
expected value goes stale the moment the data changes. Ground truth from the
source is robust and auditable.

## Report shape (reuses `harness/report.py`)

```
### L4-002 — pass-rate 5/5
- prompt: "Compare search-25 vs search-26 and tell me what regressed"   ← what we want
- checks:
  | check            | type                          | expected | observed | pass |
  |------------------|-------------------------------|----------|----------|------|
  | regression_count | ground-truth(qa_compare_runs) | 1        | 1        |  ✓   |   ← evidence it returned what we wanted
- ground truth: qa_compare_runs({...}) → regressions=1
- model's full answer: "<the entire free-text response>"                ← full response
```

The three things the report must show: the **prompt**, an **evidence snippet**
(the checks: expected vs extracted), and the model's **entire answer**.

## Thresholds

- Base, both layers: **pass-rate ≥ threshold over N runs** — the output is
  non-deterministic, so we measure reliability rather than asserting once.
- Per case: each `Check` contributes; the case passes if the required checks pass.
- Fuzzy checks (technique 4, embeddings) use a **calibrated cosine threshold**:
  collect labeled good/bad answers and pick the cut that best separates them
  (a precision/recall trade-off — the "threshold with different variables").
- LLM-as-judge: never the gate; only a complementary signal.

## Status

Scaffold only — design fixed, runner is a later slice. The first concrete
technique to implement is **fact-extraction + verification**: have the model
summarize a comparison, extract its regression count, and check it against
`qa_compare_runs` ground truth. Needs a model (`ANTHROPIC_API_KEY`), like Layer 3.
