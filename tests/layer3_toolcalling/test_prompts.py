"""Pure tests for the system-prompt resolver."""

from __future__ import annotations

import pytest

from harness.eval.prompts import (
    SystemPolicyNotFound,
    _load_policy,
    resolve_system_prompt,
)


def test_resolve_returns_none_when_neither_set():
    assert resolve_system_prompt(None, None) is None


def test_override_wins_over_policy():
    assert resolve_system_prompt("default", "JUST DO IT") == "JUST DO IT"


def test_policy_is_loaded_from_disk_and_stripped():
    text = resolve_system_prompt("default", None)
    assert text is not None
    assert text == text.strip()        # not surrounded by blank lines
    assert "tools" in text.lower()      # sanity: it's the default policy


def test_unknown_policy_raises_with_helpful_message():
    _load_policy.cache_clear()
    with pytest.raises(SystemPolicyNotFound, match="totally-not-a-policy"):
        resolve_system_prompt("totally-not-a-policy", None)
