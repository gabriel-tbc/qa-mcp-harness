# Roadmap — qa-mcp-harness

What's working, what's designed, what's next. Each increment is self-contained:
it adds value and can be shipped independently. Priority is top-down within each
section.

---

## Done (v0.1)

### Layer 2 — Contract ✅
Generic smoke suite (any MCP) + target-specific suite (qa-toolkit). Tests the
things a model reads to decide: tool names, descriptions, input schemas, response
shape. No model needed.

### Layer 3 — Tool-calling ✅
Pass-rate runner over N runs, with **separate metrics** for tool-selection
accuracy and argument accuracy. Each run keeps the full trail (tool name + exact
arguments the model produced). Reports written to `reports/layer3/` as `.json`
(for CI / diff) and `.md` (for reading and sharing). Deterministic machinery
covered by fake-model tests (no API key); the real-LLM run is gated behind
`ANTHROPIC_API_KEY` + `HARNESS_MODEL`.

---

## Next increments

### Increment 1 — Layer 4 MVP: agent loop + single oracle (est. medium)

**What:** the first end-to-end Layer 4 run. Layer 3 stops at the model's *first*
tool choice; Layer 4 needs the full agent loop — execute the tool on the MCP, feed
the result back, get the final free-text answer — and then score that answer
deterministically.

**The oracle (chosen approach — ground-truth + extraction):**
1. Call the MCP tool directly to get the authoritative value (e.g. real regression
   count from `qa_compare_runs`). The truth is *computed*, not hardcoded.
2. Extract the matching claim from the model's prose (a number, an ID) using
   structured parsing — not `"1" in text` (which matches "10", "11"…).
3. Compare extracted vs ground truth → a `Check(expected, observed, passed)`.

**Files to implement** (stubs already exist in the repo):
- `harness/eval/agent_loop.py` — `run_agent_turn(session, model, prompt, tools)`
  → `AgentOutcome(final_text, tool_calls[])`. Drives the Anthropic `messages`
  API in a loop: tool_use → `session.call_tool` → tool_result → repeat until
  final text block.
- `harness/eval/oracles.py` — `ground_truth(session, tool, args)`,
  `extract_number(text, label)`, `Check`.
- `tests/layer4_output/test_layer4_mvp.py` — one dataset case (TC-003: compare
  two runs, extract regression count, check against `qa_compare_runs` ground
  truth). One fake-agent test (no API key) + one integration test (gated).

**Report:** reuses `harness/report.py`. Shape already designed in
`tests/layer4_output/README.md`: prompt · checks table (expected / observed / pass)
· ground-truth call shown · model's full answer.

---

### Increment 2 — Layer 3: consistency signal + flakiness detection (est. small)

**What:** surface which prompts are genuinely flaky (the model wavers across runs)
vs rock-solid. The `consistency` metric is already computed per case; this increment
makes it actionable.

- Add a `--flag-flaky` threshold to the CLI (e.g. `--flag-flaky 0.8`): cases below
  it are flagged in the terminal summary and the Markdown report.
- Add a "Flakiness" section to the Markdown report when flagged cases exist.
- A flaky case is a signal that the MCP tool's *description* is ambiguous —
  the fix is in the MCP, not the test.

---

### Increment 3 — CI integration (est. small)

**What:** make the harness runnable in a GitHub Actions workflow, without manual
setup.

- `ci.yml` workflow: install deps, run Layer 2 + the fake-model Layer 3 tests
  (`-m "not integration"`), upload the report artifact.
- `reports/.gitkeep` or a `reports/` section in the workflow summary so the
  Markdown report is visible in the Actions UI.
- Document the required secrets (`ANTHROPIC_API_KEY`, `HARNESS_MODEL`) for the
  optional integration job.

---

### Increment 4 — Multi-target support in a single run (est. medium)

**What:** run Layer 2 or 3 against several targets in one command and produce a
comparative report. Useful for "does this MCP work over HTTP the same way it does
over stdio?" or "which model handles these prompts best?".

- Extend the CLI to accept multiple `--target` values or a glob.
- `LayerReport` gets a `target` field (already there); `DatasetResult` stays
  per-target — comparison is a new `ComparisonReport` that diffs two
  `LayerReport`s (accuracy delta, cases that changed verdict).
- Markdown report gets a side-by-side summary table.

---

### Increment 5 — Layer 4: metamorphic oracle (est. medium)

**What:** a second, stronger oracle for comparisons. Swapping run A and B in a
"compare A vs B" prompt should flip regressions ↔ fixes in the answer. This is a
*metamorphic relation* — no ground truth needed, just a structural invariant of the
MCP's semantics.

- `oracles.py`: `metamorphic_check(outcome_a, outcome_b, relation)`.
- Dataset gets a `metamorphic_pairs` list: two prompts with inverted arguments
  and the expected relation (e.g. `regressions_in_a == fixes_in_b`).
- Cheap to run (reuses the agent loop from Increment 1); zero extra API calls for
  ground truth.

---

### Increment 6 — Layer 4: embedding similarity with calibrated threshold (est. large)

**What:** for facts that resist structured extraction (narrative descriptions,
explanations), use cosine similarity between the model's answer and a reference
answer — with a threshold *calibrated* on labeled examples, not guessed.

- Collect a small set of labeled answers (good / bad) per case.
- Fit the cosine threshold that best separates them (precision / recall trade-off).
- Store the calibrated threshold in the dataset alongside the case.
- Report shows: similarity score, threshold, pass/fail — never just "similar".
- LLM-as-judge remains a *complementary signal* (logged, never the gate).

---

### Increment 7 — Pluggable target adapters (est. medium)

**What:** today every target TOML maps to `open_session()` in `mcp_client.py`.
Some MCPs need pre-flight (OAuth token refresh, health check, warm-up call). A
thin adapter layer between the target and the session would let those live in code
without polluting the generic client.

- `TargetAdapter` protocol: `prepare(target) -> Target` (can mutate headers,
  env, etc.) called once before any session opens.
- Adapters live in `harness/adapters/`, one file per MCP family.
- Default adapter is a no-op (current behaviour preserved).

---

## Design decisions already made (don't revisit without a reason)

| Decision | Why |
|---|---|
| Layer 3 stops at first tool_use, never executes the tool | Measures *selection* cleanly; execution belongs to Layer 4 |
| Arg matching is `contains`, not exact equality | Extra optional args from the model are fine; we check the semantic expectation |
| Ground-truth oracle over `contains` substring | Substrings collide ("1" ⊂ "10"); truth from the source doesn't go stale |
| LLM-as-judge is never the gate | Non-deterministic judge on a non-deterministic output is untestable |
| Real target files are gitignored | They hold local paths and possibly auth; only templates are versioned |
| Reports are gitignored | Artifacts; regenerated by the runner; not source |
