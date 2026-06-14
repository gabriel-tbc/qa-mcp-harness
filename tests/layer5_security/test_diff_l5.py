"""Pure tests for the Layer 5 A/B diff — feed two literal report dicts and
assert the diff captures the deltas a reader would care about."""

from __future__ import annotations

from harness.eval.diff_l5 import build_diff, render_markdown


def _report(label: str | None, cases: list[dict], overall_asr: float, breached: int) -> dict:
    return {
        "layer": 5,
        "generated_at": "2026-06-14T10:00:00Z",
        "target": "vuln-lab",
        "model": "ollama:qwen2.5",
        "n_runs": 3,
        "temperature": None,
        "system_policy": label,
        "summary": {
            "cases": len(cases),
            "cases_breached": breached,
            "overall_asr": overall_asr,
            "mean_latency_ms": None,
        },
        "cases": cases,
    }


def _case(cid: str, technique: str, kind: str, asr: float) -> dict:
    return {"id": cid, "technique": technique, "attack_kind": kind, "asr": asr, "runs": []}


def test_diff_captures_per_case_deltas_and_overall():
    a = _report(
        "default",
        [
            _case("INJ-001", "override", "forbidden_tool", 1.0),
            _case("INJ-004", "exfil", "canary", 1.0),
            _case("CTRL-006", "control", "forbidden_tool", 0.0),
        ],
        overall_asr=0.56,
        breached=2,
    )
    b = _report(
        "hardened",
        [
            _case("INJ-001", "override", "forbidden_tool", 0.0),
            _case("INJ-004", "exfil", "canary", 0.66),
            _case("CTRL-006", "control", "forbidden_tool", 0.0),
        ],
        overall_asr=0.33,
        breached=1,
    )

    d = build_diff(a, b)

    assert d["a"]["label"] == "default"
    assert d["b"]["label"] == "hardened"
    assert d["overall_delta"] == pytest_approx(b["summary"]["overall_asr"] - a["summary"]["overall_asr"])
    assert d["breaches_delta"] == -1

    by_id = {r["id"]: r for r in d["rows"]}
    assert by_id["INJ-001"]["delta"] == pytest_approx(-1.0)  # full fix
    assert by_id["INJ-004"]["delta"] == pytest_approx(-0.34, abs=0.01)
    assert by_id["CTRL-006"]["delta"] == 0.0  # benign control unaffected


def test_diff_surfaces_only_in_one_side():
    a = _report("x", [_case("INJ-001", "t", "k", 1.0), _case("INJ-002", "t", "k", 0.5)], 0.75, 2)
    b = _report("y", [_case("INJ-001", "t", "k", 0.0), _case("INJ-003", "t", "k", 0.3)], 0.15, 1)
    d = build_diff(a, b)
    assert d["only_in_a"] == ["INJ-002"]
    assert d["only_in_b"] == ["INJ-003"]
    assert [r["id"] for r in d["rows"]] == ["INJ-001"]


def test_render_markdown_shows_signed_pp_and_verdicts():
    a = _report(
        "default",
        [_case("INJ-001", "override", "forbidden_tool", 1.0),
         _case("CTRL-006", "control", "forbidden_tool", 0.0)],
        overall_asr=0.5,
        breached=1,
    )
    b = _report(
        "hardened",
        [_case("INJ-001", "override", "forbidden_tool", 0.0),
         _case("CTRL-006", "control", "forbidden_tool", 0.0)],
        overall_asr=0.0,
        breached=0,
    )
    md = render_markdown(build_diff(a, b))
    assert "default → hardened" in md
    assert "INJ-001" in md and "improved" in md
    assert "-100pp" in md      # the full fix
    assert "-50pp" in md       # overall delta
    assert " 0pp" in md        # unchanged control row


def test_diff_rejects_non_layer5_reports():
    import pytest as _pytest

    bad = _report("x", [], 0.0, 0)
    bad["layer"] = 3
    good = _report("y", [], 0.0, 0)
    with _pytest.raises(ValueError):
        build_diff(bad, good)


# tiny shim so we don't need to import pytest at module scope for the asserts above
import pytest as _pytest  # noqa: E402

pytest_approx = _pytest.approx
