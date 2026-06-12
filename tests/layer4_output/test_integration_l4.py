"""Layer 4 integration test — the full agent loop against a REAL local model.

Opt-in: set HARNESS_OLLAMA_IT=1 with Ollama running and the model pulled. It
exercises real multi-turn tool feedback end to end via openai_compat. The
deterministic loop + oracle are already covered by the FakeProvider tests
(test_agent_loop.py); this one is the only online path and stays out of the
default run because it needs a live Ollama plus the qa-toolkit MCP.

(anthropic/gemini multi-turn is not implemented yet — Ollama is the iterate
path, so this gate is on Ollama, not on a paid API key.)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.eval.dataset_l4 import load_l4_jsonl
from harness.eval.runner_l4 import evaluate_dataset_l4

DATASET = Path(__file__).parent / "datasets" / "qa_toolkit.example.jsonl"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.environ.get("HARNESS_OLLAMA_IT"),
    reason="set HARNESS_OLLAMA_IT=1 (needs a live Ollama + the qa-toolkit MCP)",
)
async def test_real_ollama_layer4(connect):
    pytest.importorskip("openai")
    from harness.providers.registry import build_provider

    model = os.environ.get("HARNESS_MODEL", "qwen2.5")
    provider = build_provider(
        "ollama", model, api_key=None, base_url=os.environ.get("OLLAMA_BASE_URL")
    )
    cases = load_l4_jsonl(DATASET)
    result = await evaluate_dataset_l4(connect, cases, provider, n_runs=2, threshold=0.9)

    for c in result.results:
        n = len(c.runs)
        print(
            f"  {c.case_id}: pass {sum(r.passed for r in c.runs)}/{n} · "
            f"oracle {sum(r.oracle_ok for r in c.runs)}/{n} · "
            f"tools {sum(r.tools_called_ok for r in c.runs)}/{n}"
        )

    # We don't gate on the oracle (that's the metric we MEASURE), but the loop
    # must at least drive a tool — the prompt requires qa_compare_runs.
    assert any(r.tools_called_ok for c in result.results for r in c.runs), "loop never called a tool"
