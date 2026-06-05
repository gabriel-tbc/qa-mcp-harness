"""Pure tests for dataset loading. No model, no network, no key."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.eval.dataset import load_jsonl

DATASET = Path(__file__).parent / "datasets" / "qa_toolkit.example.jsonl"


def test_loads_example_dataset():
    cases = load_jsonl(DATASET)
    assert len(cases) == 5
    ids = {c.id for c in cases}
    assert "TC-001" in ids


def test_expected_args_contains_is_parsed():
    cases = {c.id: c for c in load_jsonl(DATASET)}
    assert cases["TC-002"].expected_args_contains == {"run_id": "search-25_classification"}
    assert cases["TC-001"].expected_args_contains is None


def test_system_policy_and_override_optional_and_loaded(tmp_path):
    p = tmp_path / "ds.jsonl"
    p.write_text(
        '{"id":"S1","prompt":"x","expected_tool":"t","system_policy":"default"}\n'
        '{"id":"S2","prompt":"x","expected_tool":"t","system_prompt_override":"raw"}\n'
        '{"id":"S3","prompt":"x","expected_tool":"t"}\n',
        encoding="utf-8",
    )
    cases = {c.id: c for c in load_jsonl(p)}
    assert cases["S1"].system_policy == "default"
    assert cases["S1"].system_prompt_override is None
    assert cases["S2"].system_prompt_override == "raw"
    assert cases["S3"].system_policy is None and cases["S3"].system_prompt_override is None


def test_missing_required_field_raises(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "X", "prompt": "no expected tool"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="expected_tool"):
        load_jsonl(bad)
