"""Pure test for the Layer 4 Markdown renderer.

The renderer reads a plain report dict (no model, no engine), so we feed it a
literal one and assert it surfaces the three things the layer's README demands:
the prompt, the per-check evidence (expected vs observed), and the model's answer.
"""

from __future__ import annotations

from harness.report import render_markdown_l4


def _sample_report() -> dict:
    return {
        "layer": 4,
        "generated_at": "2026-06-12T10:00:00Z",
        "target": "qa-toolkit-local",
        "model": "ollama:qwen2.5",
        "n_runs": 2,
        "threshold": 0.9,
        "temperature": None,
        "summary": {
            "cases": 1,
            "cases_passed": 0,
            "accuracy": 0.0,
            "oracle_rate": 0.5,
            "tool_use_rate": 1.0,
            "mean_latency_ms": 1200.0,
        },
        "cases": [
            {
                "id": "L4-001",
                "prompt": "how many regressions?",
                "pass_rate": 0.5,
                "oracle_rate": 0.5,
                "tool_use_rate": 1.0,
                "runs": [
                    {
                        "rounds": 2,
                        "tools_called_ok": True,
                        "oracle_ok": True,
                        "passed": True,
                        "final_text": "There is 1 regression.",
                        "error": None,
                        "latency_ms": 1000,
                        "checks": [
                            {"name": "regression_count", "expected": 1, "observed": 1, "passed": True}
                        ],
                    },
                    {
                        "rounds": 2,
                        "tools_called_ok": True,
                        "oracle_ok": False,
                        "passed": False,
                        "final_text": "There are 5 regressions.",
                        "error": None,
                        "latency_ms": 1400,
                        "checks": [
                            {"name": "regression_count", "expected": 1, "observed": 5, "passed": False}
                        ],
                    },
                ],
            }
        ],
    }


def test_render_l4_surfaces_prompt_checks_and_answer():
    md = render_markdown_l4(_sample_report())

    assert "# Layer 4 — Output report" in md
    # pass_rate 0.5 < threshold 0.9 → the case fails, shown as a per-run tally.
    assert "L4-001 — FAIL" in md
    assert "pass-rate 1/2" in md
    # The evidence: expected vs observed, both the pass and the caught hallucination.
    assert "exp=1 obs=1 ✓" in md
    assert "exp=1 obs=5 ✗" in md
    # The model's full answers are present (untruncated here, both short).
    assert "There is 1 regression." in md
    assert "There are 5 regressions." in md


def test_render_l4_marks_pass_when_rate_meets_threshold():
    d = _sample_report()
    d["cases"][0]["pass_rate"] = 1.0
    assert "L4-001 — PASS" in render_markdown_l4(d)
