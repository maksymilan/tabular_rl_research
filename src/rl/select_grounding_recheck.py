#!/usr/bin/env python3
"""Select deterministic external-review cases for a stronger second opinion."""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

FORCED_RISK_FLAGS = frozenset({"common_literal"})


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--ids-output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--pass-control", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    packages = read_jsonl(args.packages)
    reviews = {row["trajectory_id"]: row for row in read_jsonl(args.reviews)}
    selected: dict[str, set[str]] = collections.defaultdict(set)
    pass_candidates = []
    for package in packages:
        trajectory_id = package["trajectory_id"]
        result = reviews.get(trajectory_id) or {}
        review = result.get("review") or {}
        overall = review.get("overall")
        risk_flags = {
            flag
            for edge in package.get("grounding_edges") or []
            for flag in edge.get("risk_flags") or []
        } | set((package.get("final_dependency") or {}).get("risk_flags") or [])
        if not review:
            selected[trajectory_id].add("review_error")
        elif overall != "pass":
            selected[trajectory_id].add(f"flash_{overall}")
        forced_risks = risk_flags & FORCED_RISK_FLAGS
        if forced_risks:
            selected[trajectory_id].update(f"risk:{flag}" for flag in sorted(forced_risks))
        if overall == "pass" and not risk_flags:
            pass_candidates.append(trajectory_id)

    rng = random.Random(args.seed)
    control_ids = sorted(rng.sample(pass_candidates, min(args.pass_control, len(pass_candidates))))
    for trajectory_id in control_ids:
        selected[trajectory_id].add("flash_pass_control")

    package_order = {package["trajectory_id"]: index for index, package in enumerate(packages)}
    ordered_ids = sorted(selected, key=package_order.__getitem__)
    selection = [
        {"trajectory_id": trajectory_id, "reasons": sorted(selected[trajectory_id])}
        for trajectory_id in ordered_ids
    ]
    reason_counts = collections.Counter(reason for row in selection for reason in row["reasons"])
    for path in (args.ids_output, args.selection_output, args.summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.ids_output.write_text("".join(f"{trajectory_id}\n" for trajectory_id in ordered_ids), encoding="utf-8")
    args.selection_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selection), encoding="utf-8"
    )
    summary = {
        "packages": len(packages),
        "selected": len(selection),
        "pass_control_requested": args.pass_control,
        "pass_control_selected": len(control_ids),
        "seed": args.seed,
        "forced_risk_flags": sorted(FORCED_RISK_FLAGS),
        "reason_counts": dict(sorted(reason_counts.items())),
    }
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
