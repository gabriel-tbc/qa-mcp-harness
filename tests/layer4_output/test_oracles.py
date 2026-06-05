"""Pure tests for the Layer 4 oracles — extraction + Check composition.
No model, no network, no MCP.
"""

from __future__ import annotations

from harness.eval.oracles import extract_number, number_check


def test_extract_number_basic_singular_and_plural():
    assert extract_number("there is 1 regression", "regression") == 1
    assert extract_number("there are 3 regressions", "regression") == 3


def test_extract_number_label_colon():
    assert extract_number("Summary — regressions: 5 fixes: 0", "regression") == 5
    assert extract_number("Summary — regressions: 5 fixes: 0", "fix") == 0


def test_extract_number_does_not_collide_with_substrings():
    """The reason we don't use `'1' in text`: it would also match 10, 11, 100."""
    text = "we had 10 fixes today"
    assert extract_number(text, "fix") == 10
    # Asking for a number that isn't there returns None, not a substring hit.
    assert extract_number("we had 10 fixes today", "regression") is None


def test_extract_number_handles_filler_words():
    assert extract_number("1 real regression and 2 known regressions", "regression") in (1, 2)
    # Either match is acceptable as long as the extraction is structural.


def test_number_check_passes_when_match():
    c = number_check("regressions", expected=1, text="1 regression detected", label="regression")
    assert c.passed is True
    assert c.observed == 1


def test_number_check_fails_loudly_when_unparseable():
    """No number found in text → check fails with observed=None (not silently 0)."""
    c = number_check("regressions", expected=1, text="model said nothing useful", label="regression")
    assert c.passed is False
    assert c.observed is None
