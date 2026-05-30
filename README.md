# qa-mcp-harness

A **generic, black-box test harness for MCP servers.** It connects to any MCP as
a client — local (stdio) or remote (Streamable HTTP) — and tests it across three
layers, without ever importing the server's source code.

It's built to mirror the real-world constraint: often you can *test* an MCP but
not *edit* it (you have the endpoint, not the repo). This harness treats every
MCP as a black box reached through configuration.

## The three layers

| Layer | Tests | Needs a model? | Status |
|---|---|---|---|
| **2 — Contract** | The MCP's tools return well-formed, consistent data (schema, response shape). | No | ✅ working |
| **3 — Tool-calling** | Given a prompt + the MCP's tools, the model picks the right tool/args, reliably (pass-rate over N). | Yes | 🚧 scaffold |
| **4 — Output** | The model's final free-text answer is correct, without an AI judge. | Yes | 🚧 scaffold |

(Layer 1 — the API behind the MCP — is tested at the API itself, outside this harness.)

## How targets work

The harness is MCP-agnostic. A **target** describes one MCP under test. Targets
are TOML files under `targets/`:

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

Pick the active target with the `HARNESS_TARGET` env var (filename without
`.toml`). To test a **remote** MCP you don't own, add an `http` target:

```toml
name = "work-mcp"
transport = "http"
[http]
url = "https://your-mcp.example.com/mcp"
[http.headers]
Authorization = "Bearer ..."
```

No code changes — just a new target file and an env var. That's the whole design.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"          # add ,llm for Layer 3/4: pip install -e ".[dev,llm]"
copy .env.example .env
```

## Run

```powershell
# Layer 2 (no model needed) against the default target:
pytest tests/layer2_contract -v

# Against a different target:
$env:HARNESS_TARGET = "work-mcp"; pytest tests/layer2_contract
```

## Layout

```
harness/
  config.py              Target loading (stdio/http) + settings
  clients/mcp_client.py  open_session(): transport-agnostic MCP client (keystone)
targets/                 One TOML per MCP under test
tests/
  layer2_contract/       Generic + target-specific contract tests
  layer3_toolcalling/    Tool-selection/argument accuracy (datasets + runner)
  layer4_output/         Free-text correctness without an AI judge
```

## Status

v0.1 — Layer 2 working against a local stdio MCP. Layers 3 and 4 scaffolded
(datasets + contracts defined; model-driven runner is the next slice).

## License

MIT.
