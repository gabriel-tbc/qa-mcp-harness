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
    assert c.strategy is None


# ── Strategy 1: bare integer (the model obeyed "return only the integer") ────


def test_bare_integer_returns_it():
    """Caught in the wild: the model answered '1' alone (the prompt asked for
    integer-only). Previous oracle missed it because the label wasn't there."""
    c = number_check("regressions", expected=1, text="1", label="regression")
    assert c.observed == 1
    assert c.passed is True
    assert c.strategy == "bare_integer"


def test_bare_integer_handles_whitespace_and_newlines():
    assert extract_number("  1  \n", "regression") == 1
    assert extract_number("\n7\n", "fix") == 7


def test_bare_integer_does_not_match_when_prose_present():
    """Anything beyond the integer disables this strategy — falls through to
    the label-based ones, where 'regression' must appear to claim the number."""
    assert extract_number("the answer", "regression") is None
    # Number plus extra word → bare strategy off → no label here → None.
    assert extract_number("1 hour", "regression") is None


# ── Strategy 3: same-sentence fallback (label and number with prose between) ─


def test_same_sentence_picks_lone_int_in_label_sentence():
    """The other case from the failed run: 'The number of regressions between
    runs `A` and `B` is 1.' — many words between the label and the integer,
    but they share a sentence."""
    text = "The number of regressions between runs `A` and `B` is 1."
    c = number_check("regressions", expected=1, text=text, label="regression")
    assert c.observed == 1
    assert c.passed is True
    assert c.strategy == "same_sentence"


def test_same_sentence_abstains_when_multiple_ints_in_sentence():
    """If the label-bearing sentence has more than one integer we abstain
    rather than guess — abstention surfaces as observed=None, which the
    report makes visible, instead of silently picking a wrong one."""
    text = "Between run 25 and run 26 there are some regressions."
    assert extract_number(text, "regression") is None


def test_same_sentence_does_not_steal_across_sentences():
    """Label and integer in different sentences must not get glued together.
    Here 'regression' is only in sentence 2 and there's no integer there."""
    text = "We saw 5 fixes. Regressions are mentioned in the appendix."
    assert extract_number(text, "regression") is None


def test_adjacent_still_wins_over_same_sentence():
    """When both strategies could apply, the more specific (adjacent) fires
    first — same-sentence stays the fallback, not the default."""
    c = number_check("regressions", expected=3, text="regressions: 3", label="regression")
    assert c.strategy == "adjacent_to_label"
    assert c.observed == 3
