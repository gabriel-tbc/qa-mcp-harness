"""Layer 5 diff — compare two ASR reports (baseline vs hardened).

Given two Layer 5 report JSONs, print a case-by-case table of ASR deltas plus
a single-line summary of overall change. The point: when you A/B a defensive
system policy against the same corpus, "endurecí y bajó el ASR" is a claim;
a delta table is evidence.

Usage:
    python -m harness.eval.diff_l5 baseline.json hardened.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _signed_pp(delta: float) -> str:
    """A pp (percentage-point) delta with explicit sign. Negative = good in this
    layer (ASR fell), so we render the sign explicitly to keep the direction
    unambiguous when readers skim the table."""
    if delta == 0:
        return " 0pp"
    pp = round(delta * 100)
    return f"{pp:+d}pp"


def _verdict(delta: float) -> str:
    if delta < 0:
        return "↓ improved"
    if delta > 0:
        return "↑ worse"
    return "= same"


def build_diff(a: dict, b: dict) -> dict:
    """Diff two Layer 5 report dicts. `a` is treated as the baseline, `b` as
    the variant being evaluated. Cases are matched by `id`; cases present in
    only one side are surfaced in `only_in_a` / `only_in_b`."""
    if a.get("layer") != 5 or b.get("layer") != 5:
        raise ValueError("diff_l5 expects two Layer-5 reports")

    by_id_a = {c["id"]: c for c in a.get("cases", [])}
    by_id_b = {c["id"]: c for c in b.get("cases", [])}
    common = [cid for cid in by_id_a if cid in by_id_b]
    only_in_a = [cid for cid in by_id_a if cid not in by_id_b]
    only_in_b = [cid for cid in by_id_b if cid not in by_id_a]

    rows = []
    for cid in common:
        ca, cb = by_id_a[cid], by_id_b[cid]
        asr_a = ca.get("asr", 0.0)
        asr_b = cb.get("asr", 0.0)
        rows.append(
            {
                "id": cid,
                "technique": ca.get("technique", "") or cb.get("technique", ""),
                "attack_kind": ca.get("attack_kind", "") or cb.get("attack_kind", ""),
                "asr_a": asr_a,
                "asr_b": asr_b,
                "delta": asr_b - asr_a,
            }
        )

    overall_a = a.get("summary", {}).get("overall_asr", 0.0)
    overall_b = b.get("summary", {}).get("overall_asr", 0.0)
    breaches_a = a.get("summary", {}).get("cases_breached", 0)
    breaches_b = b.get("summary", {}).get("cases_breached", 0)

    return {
        "a": {
            "label": a.get("system_policy") or "(per-case)",
            "model": a.get("model"),
            "n_runs": a.get("n_runs"),
            "target": a.get("target"),
            "overall_asr": overall_a,
            "cases_breached": breaches_a,
        },
        "b": {
            "label": b.get("system_policy") or "(per-case)",
            "model": b.get("model"),
            "n_runs": b.get("n_runs"),
            "target": b.get("target"),
            "overall_asr": overall_b,
            "cases_breached": breaches_b,
        },
        "rows": rows,
        "overall_delta": overall_b - overall_a,
        "breaches_delta": breaches_b - breaches_a,
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
    }


def render_markdown(diff: dict) -> str:
    """Render a diff as a Markdown table. ASR↓ is good in this layer, so the
    'verdict' column reads the delta direction in human terms."""
    a, b = diff["a"], diff["b"]
    out: list[str] = [
        f"# Layer 5 A/B — {a['label']} → {b['label']}",
        "",
        f"`target={a['target']}` · `model={a['model']}` · `n={a['n_runs']}` · "
        f"comparing **{a['label']}** vs **{b['label']}**",
        "",
        "## Summary",
        "",
        "| metric | A | B | Δ |",
        "|---|---|---|---|",
        f"| overall ASR | {_pct(a['overall_asr'])} | {_pct(b['overall_asr'])} | "
        f"**{_signed_pp(diff['overall_delta'])}** |",
        f"| cases breached | {a['cases_breached']} | {b['cases_breached']} | "
        f"{diff['breaches_delta']:+d} |",
        "",
        "## Per case",
        "",
        "| id | technique | attack | ASR A | ASR B | Δ | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in diff["rows"]:
        out.append(
            f"| {r['id']} | {r['technique']} | {r['attack_kind']} | "
            f"{_pct(r['asr_a'])} | {_pct(r['asr_b'])} | "
            f"**{_signed_pp(r['delta'])}** | {_verdict(r['delta'])} |"
        )
    if diff["only_in_a"]:
        out += ["", f"_Only in A:_ {', '.join(diff['only_in_a'])}"]
    if diff["only_in_b"]:
        out += ["", f"_Only in B:_ {', '.join(diff['only_in_b'])}"]
    out.append("")
    return "\n".join(out)


def main() -> None:
    # Windows consoles default to cp1252, which chokes on the Unicode arrows
    # in the table (↓/↑/→). Force stdout to utf-8 so the Markdown output
    # renders on every platform without escaping.
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(
        description="Diff two Layer 5 ASR reports (e.g. baseline vs hardened policy)."
    )
    parser.add_argument("baseline", help="Path to the A (baseline) report JSON")
    parser.add_argument("variant", help="Path to the B (variant) report JSON")
    parser.add_argument(
        "--json", action="store_true", help="Emit the diff as JSON instead of Markdown."
    )
    args = parser.parse_args()

    diff = build_diff(_load(args.baseline), _load(args.variant))
    if args.json:
        print(json.dumps(diff, indent=2, ensure_ascii=False))
    else:
        print(render_markdown(diff))


if __name__ == "__main__":
    main()
