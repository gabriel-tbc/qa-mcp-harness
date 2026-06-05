"""Resolve the effective system prompt for a case.

Composition model (deliberately simple): a case declares which suite-level
policy it uses (by name, mapping to `system_prompts/<name>.md`) and may also
declare a one-off `system_prompt_override` that SUBSTITUTES the policy for that
case. Substitution, not concatenation: silent contradictions are the most
expensive class of eval bugs, and substitution makes the diff explicit.

The function returns the *effective* system prompt string the provider will
see. The runner records that exact string on every RunRecord so a number is
never orphan and a report is fully reproducible.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "system_prompts"


class SystemPolicyNotFound(ValueError):
    """The case named a policy that does not exist under system_prompts/."""


@lru_cache(maxsize=64)
def _load_policy(name: str) -> str:
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.is_file():
        raise SystemPolicyNotFound(
            f"system policy {name!r} not found at {path}. "
            f"Create system_prompts/{name}.md or pick an existing policy."
        )
    return path.read_text(encoding="utf-8").strip()


def resolve_system_prompt(
    system_policy: str | None, system_prompt_override: str | None
) -> str | None:
    """Return the effective system prompt string, or None if neither is set.

    Precedence: an explicit `system_prompt_override` wins outright (substitutes
    the policy). Otherwise the named policy is loaded from disk. If neither is
    provided, returns None — the call goes out with no system prompt at all,
    which is itself a useful baseline.
    """
    if system_prompt_override is not None:
        return system_prompt_override
    if system_policy is not None:
        return _load_policy(system_policy)
    return None
