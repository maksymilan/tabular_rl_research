"""How much training signal does the post-reduction magnitude cap actually remove?

Reads the frozen per-turn coefficients of the 2026-09-15 cap=1.0 run
(`cap_audit_20260916/cap1/trajectories.jsonl`), which stores the *uncapped*
`trajectory_token_mean` reduction coefficient for every transition, and reports
the clipping curve for a range of cap values, split by sign, by group correct
count, and by the local Harness-error override channel.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SOURCE = Path(
    "docs/reports/rl/cap_audit_20260916/cap1/trajectories.jsonl"
)
CAPS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)


def quantiles(values: list[float], points=(0.5, 0.75, 0.9, 0.95, 0.99)) -> dict[str, float]:
    ordered = sorted(values)
    out = {}
    for point in points:
        index = min(len(ordered) - 1, int(point * len(ordered)))
        out[f"p{int(point * 100)}"] = ordered[index]
    return out


def main() -> int:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    eligible = [row for row in rows if row.get("process_update")]
    transitions = []
    for row in eligible:
        for turn in row.get("turns") or []:
            coefficient = float(turn.get("coefficient") or 0.0)
            transitions.append(
                {
                    "coefficient": coefficient,
                    "abs": abs(coefficient),
                    "sign": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero",
                    "error_kind": turn.get("error_kind"),
                    "tool": (turn.get("action") or {}).get("tool"),
                    "group_correct": row.get("group_correct"),
                    "trajectory_key": (row.get("update"), row.get("trajectory_id")),
                    "correct": row.get("correct"),
                }
            )
    nonzero = [t for t in transitions if t["abs"] > 0]
    total_mass = sum(t["abs"] for t in transitions)
    print(f"trajectories: {len(eligible)} | transitions: {len(transitions)} | nonzero: {len(nonzero)}")
    print(f"total |coefficient| mass: {total_mass:.2f}")
    print("|coefficient| distribution (nonzero):",
          {k: round(v, 3) for k, v in quantiles([t["abs"] for t in nonzero]).items()},
          "| max", round(max(t["abs"] for t in nonzero), 2))
    print("mean |coefficient|: positive",
          round(statistics.mean([t["abs"] for t in nonzero if t["sign"] == "positive"]), 3),
          "| negative",
          round(statistics.mean([t["abs"] for t in nonzero if t["sign"] == "negative"]), 3))

    error_turns = [t for t in transitions if t["error_kind"]]
    error_mass = sum(t["abs"] for t in error_turns)
    print(f"\nlocal Harness-error override turns: {len(error_turns)} | mass {error_mass:.2f}"
          f" ({error_mass / total_mass:.1%} of total)")
    if error_turns:
        print("  error-turn |coefficient| quantiles:",
              {k: round(v, 3) for k, v in quantiles([t["abs"] for t in error_turns]).items()},
              "| max", round(max(t["abs"] for t in error_turns), 2))

    print("\ncap   clipped_tr   clipped_traj   removed_mass   removed_pos   removed_neg   err_mass_kept")
    trajectory_total = len(eligible)
    for cap in CAPS:
        clipped = [t for t in transitions if t["abs"] > cap]
        removed = sum(t["abs"] - cap for t in clipped)
        removed_pos = sum(t["abs"] - cap for t in clipped if t["sign"] == "positive")
        removed_neg = sum(t["abs"] - cap for t in clipped if t["sign"] == "negative")
        hit_trajectories = len({t["trajectory_key"] for t in clipped})
        kept_error = sum(min(cap, t["abs"]) for t in error_turns)
        print(f"{cap:>4} {len(clipped):>12} {hit_trajectories:>14} "
              f"{removed / total_mass:>13.1%} {removed_pos / total_mass:>13.1%} "
              f"{removed_neg / total_mass:>13.1%} {kept_error / error_mass if error_mass else 0:>14.1%}")

    print("\nremoved mass by group correct count (cap=1.0 vs 1.5):")
    by_group = defaultdict(float)
    group_total = defaultdict(float)
    for t in transitions:
        key = str(t["group_correct"])
        group_total[key] += t["abs"]
        if t["abs"] > 1.0:
            by_group[key] += t["abs"] - 1.0
    for key in sorted(group_total, key=lambda value: int(value)):
        share = by_group[key] / group_total[key] if group_total[key] else 0.0
        print(f"  group_correct={key:>2}: mass {group_total[key]:7.1f} | removed@1.0 {share:5.1%}")

    print("\nremoved mass by tool (cap=1.0, top 8):")
    tool_total, tool_removed = Counter(), Counter()
    for t in transitions:
        tool = str(t["tool"])
        tool_total[tool] += t["abs"]
        if t["abs"] > 1.0:
            tool_removed[tool] += t["abs"] - 1.0
    for tool, _ in tool_total.most_common(8):
        print(f"  {tool:<24} mass {tool_total[tool]:7.1f} | removed@1.0 "
              f"{tool_removed[tool] / tool_total[tool] if tool_total[tool] else 0:5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
