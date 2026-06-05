# qa-mcp-harness

A **generic, black-box test harness for MCP servers.** It connects to any MCP as a
client — local (stdio) or remote (Streamable HTTP) — and tests it across three
layers, without ever importing the server's source code.

It mirrors a real-world constraint: often you can *test* an MCP but not *edit* it
(you have the endpoint, not the repo). The harness treats every MCP as a black box
reached purely through configuration.

## The three layers

| Layer | Question it answers | Needs a model? |
|---|---|---|
| **2 — Contract** | Do the MCP's tools return well-formed, consistent data (schema, response shape)? | No |
| **3 — Tool-calling** | Given a prompt and the MCP's tools, does the model pick the right tool with the right arguments — reliably across repeated runs (pass-rate over N)? | Yes |
| **4 — Output** | Is the model's final free-text answer correct, checked deterministically (no AI judge)? | Yes |

(Layer 1 — the API behind the MCP — is tested at the API itself, outside this harness.)

## How targets work

The harness is MCP-agnostic. A **target** describes one MCP under test. Targets are
TOML files under `targets/`:

```toml
# targets/qa-toolkit-local.toml
name = "qa-toolkit-local"
transport = "stdio"
[stdio]
command = 'C:\path\to\server\.venv\Scripts\python.exe'
args = ["-m", "qa_toolkit_mcp.server"]
[stdio.env]
QA_TOOLKIT_RUNS_DIR = 'C:\path\to\runs'
```

Pick the active target with the `HARNESS_TARGET` env var (filename without `.toml`).
To test a **remote** MCP you don't own, add an `http` target:

```toml
name = "work-mcp"
transport = "http"
[http]
url = "https://your-mcp.example.com/mcp"
[http.headers]
Authorization = "Bearer ..."
```

No code changes — just a new target file and an env var. That's the whole design.

Real target files are gitignored (they hold machine-specific paths, and possibly
auth); the repo tracks only `*.example.toml` templates and `example-http.toml`.
Copy a template to its real name to use it.

## Reports

Each Layer 3 run writes a report artifact under `reports/` (gitignored):

- a **`.json`** for CI, diffing, and history across runs, and
- a **`.md`** for reading and sharing.

Tool selection and argument accuracy are reported as **separate metrics**, so a
failure tells you *which* half broke — the tool choice or the arguments. Every case
keeps a per-run trail (the exact tool and arguments the model produced) plus a
consistency score (how stable the choice was across the N runs). Layer 4 will reuse
the same report format.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"          # add ,llm for Layers 3/4: pip install -e ".[dev,llm]"
copy .env.example .env           # then set ANTHROPIC_API_KEY + HARNESS_MODEL for Layer 3
copy targets\qa-toolkit-local.example.toml targets\qa-toolkit-local.toml  # then edit the paths
```

## Run

Layer 2 needs no model:

```powershell
pytest tests/layer2_contract -v                                  # default target
$env:HARNESS_TARGET = "work-mcp"; pytest tests/layer2_contract   # a different target
```

Layer 3 drives a real model. Set `ANTHROPIC_API_KEY` and `HARNESS_MODEL` in `.env`,
then run the eval and write a report:

```powershell
python -m harness.eval.runner tests/layer3_toolcalling/datasets/qa_toolkit.example.jsonl -n 5
```

Its deterministic machinery (matching, pass-rate, reporting) is covered by
fake-model tests that need no API key. Exclude the single billed test with:

```powershell
pytest tests/layer3_toolcalling -m "not integration"
```

## Layout

```
harness/
  config.py              Target loading (stdio/http) + settings
  clients/mcp_client.py  open_session(): transport-agnostic MCP client (keystone)
  providers/             Model-agnostic boundary: ToolSpec/ModelResponse/Provider + adapters
  eval/                  Layer 3 engine: dataset, matching, runner, oracles, agent loop
  report.py              Report model + JSON/Markdown rendering (shared by Layers 3-4)
targets/                 One TOML per MCP under test
tests/
  layer2_contract/       Generic + target-specific contract tests
  layer3_toolcalling/    Tool-selection / argument accuracy (datasets + runner)
  layer4_output/         Free-text correctness without an AI judge (design)
reports/                 Generated run artifacts (.json + .md), gitignored
```

## Status

v0.1.

- **Layer 2 (contract)** — working against a local stdio MCP.
- **Layer 3 (tool-calling)** — working: pass-rate runner with separate tool- and
  argument-accuracy metrics and JSON + Markdown reports. Model-agnostic behind a
  **provider layer** (`anthropic` · `openai` · `gemini` · `ollama`, selected by
  `--provider`/`HARNESS_PROVIDER`); the core speaks only neutral `ToolSpec` /
  `ModelResponse`. Adding Gemini (a third wire format) touched only its adapter
  + the registry — runner/matching/report untouched, the proof the abstraction
  holds. Reports capture the full per-run trace (final text, tokens,
  latency, stop reason, error). Real-LLM runs are gated behind a key; the
  deterministic machinery is covered by fake-provider tests.
- **Layer 4 (output)** — **minimum vertical working**: real agent loop runs
  against the real MCP; ground-truth oracle computes the expected value from
  the MCP itself (no hardcoded expectations, no AI judge); `extract_number`
  pulls the model's claim from prose with word boundaries (no substring
  collisions). Signals separated: `tools_called_ok` vs `oracle_ok`. Tested
  end-to-end with a scripted FakeProvider against the real MCP; live-model
  multi-turn for Anthropic/OpenAI/Gemini is a future slice.

## License

MIT — see [LICENSE](LICENSE).
