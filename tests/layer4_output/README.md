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

## Status

Scaffold only. The first concrete technique to implement here is
**fact-extraction + verification**: ask the model to summarize a comparison,
extract its regression count, and check it against `qa_compare_runs` ground truth.

Needs a model (`ANTHROPIC_API_KEY`), like Layer 3.
