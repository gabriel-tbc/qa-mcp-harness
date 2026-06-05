# qa-mcp-harness — Project Plan

> Living design document. The harness is a **generic, black-box, model-agnostic
> test harness for MCP servers**. This file is the north star: vision,
> architecture, and the build roadmap. It is meant to outlast any single slice.

---

## 1. Why this exists (vision & goals)

A test harness that connects to **any** MCP server as a client — local (stdio)
or remote (HTTP) — and evaluates it across layers, **without ever importing the
server's source**, and against **any model provider** (Anthropic, OpenAI,
Google, local Ollama) behind one uniform interface.

It serves three purposes, in order:

1. **Portfolio** — a public artifact that demonstrates depth in AI/agent testing.
2. **Experience** — a place to practice the exact constraints of the job
   (testing an MCP you can reach but cannot edit; comparing models; measuring
   non-determinism with real oracles).
3. **Reusable base** — a clean template from which similar testing services can
   be built, including as part of a product/B2B offering.

Because of (3), this is **not an MVP to ship fast**. It is a durable codebase:
calm, well-factored, well-tested, designed to grow for years.

---

## 2. Principles (the non-negotiables)

- **Black-box.** The harness is an MCP *client*. It never imports a server's
  code. Everything is reached through configuration (a *target*).
- **Model-agnostic.** Provider-specific code lives behind one boundary (the
  provider adapters). The core never knows which model ran.
- **Deterministic oracles, no AI-judge-as-gate.** Verdicts come from structural
  checks, ground truth computed from the MCP itself, metamorphic relations, or
  calibrated thresholds — not from another model's opinion. (LLM-as-judge is at
  most a complementary signal.)
- **Measure non-determinism, don't fight it.** Run N times, report pass-rate and
  consistency. Reliability is a metric, not an assertion.
- **Capture everything.** Reports persist the full trace per run (inputs,
  outputs, tool calls, tokens, latency, raw responses). We can always filter
  later; we must never be structurally unable to answer a question.
- **Separate signals.** "Picked the right tool" and "passed the right arguments"
  are reported independently. A collapsed pass/fail hides which half broke.
- **Pure where possible.** Deterministic logic (conversion, matching, metrics,
  reporting) is tested without any model or network. Real-model runs are gated.

---

## 3. The testing model

### Layers (what is under test)

| Layer | Under test | Needs a model? |
|---|---|---|
| **1 — API** | the API behind the MCP | (tested at the API, outside this harness) |
| **2 — Contract** | the MCP's tools return well-formed, consistent data | No |
| **3 — Tool-calling** | the model picks the right tool + args, reliably (pass-rate over N) | Yes |
| **4 — Output** | the model's final free-text answer is correct, checked deterministically | Yes |

### Orthogonal axes (how/where it runs)

These are NOT extra layers; they are dimensions any layer can vary along:

- **Transport:** stdio (local) | Streamable HTTP (remote).
- **Provider:** anthropic | openai | ollama | gemini.
- **Mode:** automated harness (reproducible, scalable) | exploratory (real
  client, human-driven, ToS-safe — for what automation can't cover).

---

## 4. Architecture

```
                          ┌─────────────────────────┐
                          │        Eval engine        │   ← provider-agnostic core
                          │  runner · matching ·      │
                          │  oracles · agent loop     │
                          └───────────┬───────────────┘
                                      │ speaks NEUTRAL types only
                ┌─────────────────────┼──────────────────────┐
                ▼                     ▼                      ▼
        ┌───────────────┐    ┌────────────────┐    ┌────────────────┐
        │  MCP client    │    │  Provider layer │    │  Report layer   │
        │ (stdio / http) │    │  (adapters)     │    │ (JSON + MD)     │
        └───────┬────────┘    └───────┬─────────┘    └────────────────┘
                │                     │
                ▼            ┌─────────┼──────────┬─────────────┐
          the MCP server     ▼         ▼          ▼             ▼
          under test     Anthropic  OpenAI/    Gemini      (future)
                                    Ollama
```

### The neutral core (the heart of model-agnosticism)

Two neutral representations the whole core speaks:

- **`ToolSpec`** — a tool, provider-independently: `name`, `description`,
  `parameters` (JSON Schema). This is essentially the MCP tool form — and since
  MCP `inputSchema` *is* JSON Schema, it is already neutral. Built once from the
  MCP via `ToolSpec.from_mcp(...)`.
- **`ModelResponse`** — the result of one model turn, provider-independently:
  `tool_calls: list[ToolCall]`, `final_text`, `stop_reason`, `usage` (tokens),
  `latency_ms`, `error`, `raw` (the original provider response, for debugging).

### The Provider boundary

A **`Provider`** is a Protocol (structural interface) every adapter satisfies:

- `name`, `model` — identity.
- `complete(prompt, tools: list[ToolSpec]) -> ModelResponse` — one turn.
- *(Layer 4, later)* a way to feed a tool result back into the conversation;
  the format is provider-specific, so it is hidden here.

Each adapter does exactly two translations and nothing else:

1. **`ToolSpec` → provider tool format** (request side).
2. **provider native response → `ModelResponse`** (response side).

Because the *neutral* form is the MCP form, adapters convert **neutral →
provider** (one hop from the source of truth) — never provider → provider (the
current Anthropic→OpenAI smell this refactor removes).

### Why this makes debugging clean (the explicit goal)

A parse error or a slowdown is isolated to **one adapter**, measured at **one
boundary**. The core (matching, metrics, MCP client) is provably not the cause.
"Is it the model, the MCP, or the harness?" is answered by looking at one place.
This is also the *enabler* of differential testing across models — the same
dataset run on Claude/GPT/Gemini, divergences pointing at prompt/MCP bugs.

---

## 5. Target directory structure (end state)

```
harness/
  config.py                 Target loading (stdio/http) + settings
  clients/
    mcp_client.py           open_session(): transport-agnostic MCP client
  providers/                ← THE model-agnostic boundary
    base.py                 ToolSpec, ToolCall, Usage, ModelResponse, Provider
    fake.py                 FakeProvider (scripted, for tests — no network)
    anthropic.py            AnthropicProvider
    openai_compat.py        OpenAIProvider  (OpenAI + Ollama via base_url)
    gemini.py               GeminiProvider  (future)
    registry.py             build_provider(name, model, ...)
  eval/
    dataset.py              EvalCase loading (JSONL)
    matching.py             tool / argument oracles (Layer 3)
    runner.py               pass-rate engine (speaks Provider + neutral types)
    oracles.py              ground-truth + extraction oracles (Layer 4)
    agent_loop.py           full agent loop (Layer 4)
  report.py                 RunRecord/CaseReport/LayerReport + JSON/MD render
targets/                    one TOML per MCP under test (real ones gitignored)
tests/
  layer2_contract/          generic + target-specific contract tests
  layer3_toolcalling/       tool-selection / argument accuracy
  layer4_output/            free-text correctness (design → impl)
  providers/                per-adapter pure + contract tests
reports/                    generated run artifacts (.json + .md), gitignored
```

---

## 6. Observability / reporting design

The report is the product of a run; capture richly now, filter later.

**`RunRecord` (per model invocation) — target fields:**

| Field | Purpose |
|---|---|
| `tool`, `args` | the model's choice (Layer 3) |
| `raw_args` | arguments before parsing (diagnose malformed output) |
| `tool_ok`, `args_ok` | the two separate signals |
| `final_text` | the model's full answer, untruncated (Layer 4, hallucinations) |
| `stop_reason` | why generation stopped |
| `input_tokens`, `output_tokens` | response size, cost |
| `latency_ms` | response time, measured at the provider boundary |
| `error` | exception / refusal / malformed |
| `raw` (optional, JSON tail) | original provider response for deep debugging |

Most of these come straight from `ModelResponse` — designing the neutral
response richly is what makes the rich report possible. A future dashboard reads
whatever subset of these it needs.

---

## 7. Build roadmap

### Phase 0 — done

- Layer 2 contract (generic + qa-toolkit specific).
- Layer 3 runner (anthropic + ollama via `ModelCall`), JSON/MD reports.
- Layer 4 design fixed (agent loop + ground-truth oracle, stubbed).

### Phase 1 — the provider refactor (CURRENT)

- **(a) Neutral core + Provider protocol.** `providers/base.py` (`ToolSpec`,
  `ToolCall`, `Usage`, `ModelResponse`, `Provider`) + `providers/fake.py`
  (`FakeProvider`) + pure tests. No wiring yet; existing suite stays green.
- **(b) Migrate adapters + rewire runner.** Port the current Anthropic and
  OpenAI/Ollama logic into `providers/anthropic.py` and
  `providers/openai_compat.py`. `runner.evaluate_case` takes a `Provider` and
  reads `ModelResponse`. `registry.build_provider(...)` replaces the ad-hoc
  factories. Fake-model tests become fake-provider tests.
- **(c) Enrich `RunRecord` from `ModelResponse`.** Flow `final_text`,
  `latency_ms`, tokens, `stop_reason`, `error`, `raw_args` into the report;
  render the new fields in JSON (and a sensible subset in MD).

Each of (a)/(b)/(c) lands green before the next starts.

### Phase 2 — breadth & Layer 4

- `providers/gemini.py` (own format) → unlocks 3-way differential testing.
- Layer 4: implement `agent_loop.run_agent_turn` against the Provider boundary
  (incl. provider-specific tool-result feedback) + `oracles` (ground truth +
  extraction).

### Phase 3 — productization (the B2B base)

- A dashboard over the `reports/` artifacts (read-only, shares the report schema).
- Packaging: configurable target + provider + dataset as a reusable service.
- CI integration patterns (run on schedule, diff against history, alert on
  regression in tool-selection accuracy).

---

## 8. Testing strategy

- **Pure tests** for every deterministic piece (tool conversion per adapter,
  response parsing per adapter, matching, metrics, report rendering). No key.
- **Adapter contract tests:** feed a canned provider-native response, assert the
  `ModelResponse` mapping. Catches provider-shape drift in isolation.
- **Engine tests with `FakeProvider`:** exercise pass-rate / thresholds / report
  with scripted responses. No key.
- **Gated integration tests:** one per real provider, skipped unless its
  key/endpoint is configured. The only billed/online tests.

---

## 9. Decisions log

- **Own adapter layer, not LiteLLM.** More code, but full control and a stronger
  portfolio artifact; LiteLLM remains a fallback if speed ever outweighs control.
- **Neutral form = MCP form (JSON Schema).** Avoids provider→provider conversion.
- **Anthropic was the accidental internal format** (its tool schema ≈ raw JSON
  Schema). Phase 1 removes that coupling.
- **stdio for the local toy MCP; HTTP for remote/work MCPs.** Both behind
  `open_session`.
