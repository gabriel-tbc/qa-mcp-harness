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


def test_missing_required_field_raises(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "X", "prompt": "no expected tool"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="expected_tool"):
        load_jsonl(bad)
