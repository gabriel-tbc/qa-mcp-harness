"""Layer 3 integration test — drives a REAL model via the provider registry.

Skipped automatically unless BOTH ANTHROPIC_API_KEY and HARNESS_MODEL are set.
This is the only test that costs money and exercises non-determinism, so it
lives behind a gate; the deterministic machinery is covered by the fake-provider
tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.eval.dataset import load_jsonl
from harness.eval.runner import evaluate_dataset

DATASET = Path(__file__).parent / "datasets" / "qa_toolkit.example.jsonl"

pytestmark = pytest.mark.integration

_HAVE_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
_HAVE_MODEL = bool(os.environ.get("HARNESS_MODEL"))


@pytest.mark.skipif(
    not (_HAVE_KEY and _HAVE_MODEL),
    reason="needs ANTHROPIC_API_KEY and HARNESS_MODEL",
)
async def test_real_model_tool_selection(connect):
    pytest.importorskip("anthropic")
    from harness.providers.registry import build_provider

    cases = load_jsonl(DATASET)
    provider = build_provider(
        "anthropic", os.environ["HARNESS_MODEL"], api_key=os.environ["ANTHROPIC_API_KEY"]
    )
    # Modest N to keep cost low in CI; raise locally for tighter statistics.
    ds = await evaluate_dataset(connect, cases, provider, n_runs=3, threshold=0.9)

    # We don't hard-assert 100% (that's the metric we're MEASURING, not a gate),
    # but a well-designed MCP should comfortably select the obvious tool. We
    # assert a sane floor and print the report for inspection.
    print("\n" + _report(ds))
    assert ds.accuracy >= 0.6, f"tool-selection accuracy too low: {ds.accuracy:.0%}"


def _report(ds) -> str:
    lines = [f"accuracy={ds.accuracy:.0%} threshold={ds.threshold:.0%}"]
    for r in ds.results:
        lines.append(f"  {r.case_id}: {r.matches}/{r.n_runs} observed={r.observed_tools}")
    return "\n".join(lines)
