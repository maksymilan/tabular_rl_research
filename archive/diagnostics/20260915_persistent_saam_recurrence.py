#!/usr/bin/env python3
"""Audit exact state/action recurrence across adjacent optimizer updates.

This diagnostic is gold-free: it uses the immutable LineageReplay identity and
only reads rollout rows needed for process_update, example identity, outcome,
state digest, and action digest.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--snapshot-src", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.snapshot_src))
    from rl.scenarios.diagnostics.audit_saam_lineage_replay import (  # noqa: PLC0415
        LineageReplay,
        _process_update,
    )

    strict = defaultdict(lambda: {"all": set(), "correct": set(), "wrong": set()})
    action_only = defaultdict(lambda: {"all": set(), "correct": set(), "wrong": set()})
    examples = defaultdict(set)
    rows = eligible_rows = events = matched = action_matched = 0
    with args.rollouts.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            if not _process_update(row):
                continue
            eligible_rows += 1
            step = int(row.get("policy_global_step", -1))
            example = int(row["example_index"])
            examples[step].add(example)
            side = "correct" if bool(row.get("correct")) else "wrong"
            events_for_row, _ = LineageReplay(row).replay()
            for event in events_for_row:
                events += 1
                digest = event.get("action_digest")
                if event.get("matched") and digest:
                    matched += 1
                    key = (example, event["state_digest"], digest)
                    strict[(step, example)]["all"].add(key)
                    strict[(step, example)][side].add(key)
                if event.get("action_matchable") and digest:
                    action_matched += 1
                    key = (example, digest)
                    action_only[(step, example)]["all"].add(key)
                    action_only[(step, example)][side].add(key)

    result = {
        "schema": "persistent-saam-recurrence-v1",
        "gold_sql_read": False,
        "rows": rows,
        "eligible_rows": eligible_rows,
        "events": events,
        "matched_events": matched,
        "action_matchable_events": action_matched,
        "adjacent": {},
    }
    for name, data in (("strict_state_action", strict), ("action_only", action_only)):
        by_step = defaultdict(set)
        by_side = defaultdict(set)
        for (step, _example), values in data.items():
            by_step[step].update(values["all"])
            for side in ("correct", "wrong"):
                by_side[(step, side)].update(values[side])
        result["adjacent"][name] = {}
        for step in range(3):
            previous = by_step[step]
            following = by_step[step + 1]
            overlap = previous & following
            previous_success = by_side[(step, "correct")]
            following_wrong = by_side[(step + 1, "wrong")]
            result["adjacent"][name][f"{step}->{step + 1}"] = {
                "previous_unique_keys": len(previous),
                "following_unique_keys": len(following),
                "overlap_unique_keys": len(overlap),
                "p_following_given_previous": len(overlap) / len(previous) if previous else 0.0,
                "p_previous_given_following": len(overlap) / len(following) if following else 0.0,
                "previous_success_keys": len(previous_success),
                "previous_success_to_following_wrong": len(previous_success & following_wrong),
                "p_success_key_following_wrong": (
                    len(previous_success & following_wrong) / len(previous_success)
                    if previous_success else 0.0
                ),
                "shared_examples": len(examples[step] & examples[step + 1]),
            }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
