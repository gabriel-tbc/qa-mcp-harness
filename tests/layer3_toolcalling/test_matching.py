"""Pure tests for Layer 3 matching oracles. No model, no network, no key."""

from __future__ import annotations

from harness.eval.matching import args_contain, case_passes, tool_matches


def test_tool_matches_exact():
    assert tool_matches("qa_list_runs", "qa_list_runs") is True
    assert tool_matches("qa_get_run", "qa_list_runs") is False


def test_tool_matches_none_never_matches():
    assert tool_matches(None, "qa_list_runs") is False


def test_args_contain_flat():
    assert args_contain({"run_id": "x"}, {"run_id": "x"}) is True
    assert args_contain({"run_id": "y"}, {"run_id": "x"}) is False


def test_args_contain_handles_params_wrapping():
    """The real gotcha: our tools wrap arguments under `params`, so the model
    emits {"params": {"run_id": "x"}}. The dataset specifies the semantic
    expectation {"run_id": "x"} and the recursive search must find it."""
    actual = {"params": {"run_id": "search-25_classification", "include_passed": True}}
    assert args_contain(actual, {"run_id": "search-25_classification"}) is True
    assert args_contain(actual, {"include_passed": True}) is True
    assert args_contain(actual, {"run_id": "other"}) is False


def test_args_contain_multiple_keys_all_required():
    actual = {"params": {"run_id": "x", "include_passed": True}}
    assert args_contain(actual, {"run_id": "x", "include_passed": True}) is True
    assert args_contain(actual, {"run_id": "x", "include_passed": False}) is False


def test_case_passes_full_verdict():
    # right tool, no arg expectation → pass
    assert case_passes("qa_list_runs", {}, "qa_list_runs", None) is True
    # right tool, arg present → pass
    assert case_passes(
        "qa_get_run", {"params": {"run_id": "x"}}, "qa_get_run", {"run_id": "x"}
    ) is True
    # right tool, arg missing → fail
    assert case_passes(
        "qa_get_run", {"params": {"run_id": "y"}}, "qa_get_run", {"run_id": "x"}
    ) is False
    # wrong tool → fail regardless of args
    assert case_passes("qa_list_runs", {}, "qa_get_run", None) is False
