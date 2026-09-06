#!/usr/bin/env python3
"""Offline comparison of Vanilla, lineage A+B, and first-branch credit.

The branch variant is deliberately small and search-free.  Within one K8 group it
finds an exact executed-prefix state with at least two successful lineage actions
and mixed terminal outcomes.  For the first such state visited by a trajectory it
replaces that trajectory's coefficient on the first branch action with the
empirical relative branch value

    (mean_outcome(state, action) - mean_outcome(state)) /
        sqrt(mean_outcome(state) * (1 - mean_outcome(state))).

All other events use lineage-aware asymmetric credit with deterministic error
override.  The replay never reads gold_sql and never performs search or rollout.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl" / "diagnostics"))

import rl.scenarios.diagnostics.audit_saam_lineage_replay as lineage  # noqa: E402
from rl.scenarios.diagnostics.replay_lineage_asymmetric_credit import _group_rows, _local_statuses, _read_rows  # noqa: E402
from rl.frameworks.trl.transition_batch import standardized_group_advantages  # noqa: E402


def _mass(records: list[dict[str, Any]], field: str) -> dict[str, float | int]:
    values = [float(row[field]) for row in records]
    return {
        "events": len(values),
        "positive_events": sum(value > 1e-12 for value in values),
        "negative_events": sum(value < -1e-12 for value in values),
        "zero_events": sum(abs(value) <= 1e-12 for value in values),
        "positive_mass": sum(max(value, 0.0) for value in values),
        "negative_mass": sum(max(-value, 0.0) for value in values),
    }


def _direct_conflict(records: list[dict[str, Any]], field: str) -> float:
    grouped: dict[tuple[int, int, str], list[float]] = collections.defaultdict(list)
    for row in records:
        key = row.get("action_key")
        if key is not None:
            grouped[(row["policy_global_step"], row["example_index"], str(key))].append(float(row[field]))
    total = 0.0
    for values in grouped.values():
        positive = sum(max(value, 0.0) for value in values)
        negative = sum(max(-value, 0.0) for value in values)
        total += 2.0 * min(positive, negative)
    return total


def replay(
    rows: list[dict[str, Any]],
    *,
    expected_group_size: int,
    error_penalty: float,
    homogeneous_action_only: bool = True,
) -> dict[str, Any]:
    if not math.isfinite(error_penalty) or error_penalty <= 0:
        raise ValueError("error_penalty must be positive and finite")
    grouped = _group_rows(rows)
    all_records: list[dict[str, Any]] = []
    branch_state_count = 0
    branch_event_count = 0
    branch_onset_count = 0
    branch_sign_aligned = 0
    branch_sign_comparable = 0

    for group_key, group in sorted(grouped.items()):
        if len(group) != expected_group_size:
            raise ValueError(f"group {group_key} has {len(group)} rows, expected {expected_group_size}")
        eligible = [lineage._process_update(row) for row in group]
        advantages = standardized_group_advantages(
            [1.0 if row.get("correct") is True else 0.0 for row in group], eligible
        )
        prepared: list[dict[str, Any]] = []
        successful_by_key: dict[str, set[bool]] = collections.defaultdict(set)
        state_events: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row, is_eligible, advantage in zip(group, eligible, advantages, strict=True):
            events, _ = lineage.LineageReplay(row).replay()
            statuses = _local_statuses(row)
            if len(events) != len(statuses):
                raise ValueError(f"event/status mismatch for {row.get('trajectory_id')}")
            event_rows: list[dict[str, Any]] = []
            for event, status in zip(events, statuses, strict=True):
                action_key = event.get("action_digest") if event.get("action_matchable") else None
                item = {
                    "event": event,
                    "status": status,
                    "action_key": action_key,
                    "state_key": event.get("state_digest") if event.get("matched") else None,
                    "correct": bool(row.get("correct")),
                    "trajectory_id": str(row.get("trajectory_id") or ""),
                    "policy_global_step": int(row["policy_global_step"]),
                    "example_index": int(row["example_index"]),
                    "eligible": bool(is_eligible),
                    "old_advantage": float(advantage) if is_eligible else 0.0,
                }
                event_rows.append(item)
                if is_eligible and action_key is not None and status["kind"] == "success":
                    successful_by_key[action_key].add(bool(row.get("correct")))
                    if item["state_key"] is not None:
                        state_events[item["state_key"]].append(item)
            prepared.append({"row": row, "events": event_rows})

        mixed_success_keys = {key for key, outcomes in successful_by_key.items() if outcomes == {True, False}}
        branch_states: dict[str, dict[str, Any]] = {}
        for state_key, values in state_events.items():
            action_keys = {str(item["action_key"]) for item in values}
            outcomes = {bool(item["correct"]) for item in values}
            if len(action_keys) < 2 or outcomes != {True, False}:
                continue
            branch_state_count += 1
            by_action: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
            for item in values:
                by_action[str(item["action_key"])].append(item)
            state_mean = sum(float(item["correct"]) for item in values) / len(values)
            variance = state_mean * (1.0 - state_mean)
            if variance <= 0.0:
                continue
            action_values: dict[str, float] = {}
            for action_key, action_values_rows in by_action.items():
                # A branch action that itself appears on both terminal sides is
                # not a clean branch choice.  A+B handles it as a shared action;
                # only outcome-homogeneous branch actions receive the extra
                # first-divergence value in the conservative arm.
                if homogeneous_action_only and len({bool(item["correct"]) for item in action_values_rows}) != 1:
                    continue
                action_mean = sum(float(item["correct"]) for item in action_values_rows) / len(action_values_rows)
                action_values[action_key] = (action_mean - state_mean) / math.sqrt(variance)
            branch_states[state_key] = {"values": values, "action_values": action_values}
            branch_event_count += len(values)

        # Only the first branch state in a trajectory is changed by the branch arm.
        first_branch_depth: dict[str, int] = {}
        for state in branch_states.values():
            for item in state["values"]:
                tid = item["trajectory_id"]
                depth = int(item["event"]["depth"])
                first_branch_depth[tid] = min(depth, first_branch_depth.get(tid, depth))

        for item_group in prepared:
            row = item_group["row"]
            tid = str(row.get("trajectory_id") or "")
            changed = False
            for item in item_group["events"]:
                event = item["event"]
                status = item["status"]
                action_key = item["action_key"]
                old = float(item["old_advantage"])
                if not item["eligible"]:
                    effective = 0.0
                    decision = "trajectory_ineligible_zero"
                elif status["kind"] == "infrastructure_timeout":
                    effective = 0.0
                    decision = "infrastructure_timeout_zero"
                elif status["local_error"]:
                    effective = -max(abs(old), error_penalty)
                    decision = f"deterministic_error_negative:{status['kind']}"
                else:
                    branch_value = None
                    state_key = item["state_key"]
                    if (
                        state_key in branch_states
                        and action_key is not None
                        and first_branch_depth.get(tid) == int(event["depth"])
                    ):
                        branch_value = branch_states[state_key]["action_values"].get(str(action_key))
                    if branch_value is not None and abs(branch_value) > 1e-12:
                        effective = float(branch_value)
                        decision = "first_branch_relative_value"
                        changed = True
                        branch_onset_count += 1
                        branch_sign_comparable += 1
                        branch_sign_aligned += int((effective > 0) == bool(row.get("correct")))
                    elif action_key in mixed_success_keys:
                        if bool(row.get("correct")):
                            effective = old
                            decision = "shared_success_correct_keep_positive"
                        else:
                            effective = 0.0
                            decision = "shared_success_wrong_suppress_negative"
                    else:
                        effective = old
                        decision = "unmatched_success_keep_trajectory"
                all_records.append(
                    {
                        "policy_global_step": int(row["policy_global_step"]),
                        "example_index": int(row["example_index"]),
                        "trajectory_id": tid,
                        "correct": bool(row.get("correct")),
                        "eligible": bool(item["eligible"]),
                        "depth": int(event["depth"]),
                        "action_key": action_key,
                        "old_advantage": old,
                        "effective_advantage": effective,
                        "decision": decision,
                    }
                )

    eligible_records = [row for row in all_records if row["eligible"]]
    old = _mass(eligible_records, "old_advantage")
    effective = _mass(eligible_records, "effective_advantage")
    # Compare to ordinary trajectory credit and to A+B using the same event pool.
    base_records = [dict(row, effective_advantage=row["old_advantage"]) for row in eligible_records]
    return {
        "schema_version": "branch-onset-relative-credit-replay-v1",
        "gold_sql_read": False,
        "error_penalty": error_penalty,
        "homogeneous_action_only": homogeneous_action_only,
        "groups": len(grouped),
        "trajectories": len(rows),
        "eligible_trajectories": sum(lineage._process_update(row) for row in rows),
        "eligible_events": len(eligible_records),
        "branch_states": branch_state_count,
        "branch_candidate_events": branch_event_count,
        "branch_onset_events": branch_onset_count,
        "branch_sign_alignment": branch_sign_aligned / branch_sign_comparable if branch_sign_comparable else 0.0,
        "old_credit": old,
        "effective_credit": effective,
        "old_action_direct_conflict": _direct_conflict(base_records, "effective_advantage"),
        "effective_action_direct_conflict": _direct_conflict(eligible_records, "effective_advantage"),
        "decision_counts": dict(collections.Counter(row["decision"] for row in eligible_records)),
        "correct_branch_onset_events": sum(
            row["decision"] == "first_branch_relative_value" and row["correct"] for row in eligible_records
        ),
        "wrong_branch_onset_events": sum(
            row["decision"] == "first_branch_relative_value" and not row["correct"] for row in eligible_records
        ),
        "records": all_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--expected-group-size", type=int, default=8)
    parser.add_argument("--error-penalty", type=float, default=1.0)
    parser.add_argument(
        "--allow-mixed-action-branch",
        action="store_true",
        help="diagnostic only: assign branch value to action keys with mixed terminal outcomes",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(
        _read_rows(args.rollouts),
        expected_group_size=args.expected_group_size,
        error_penalty=args.error_penalty,
        homogeneous_action_only=not args.allow_mixed_action_branch,
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {key: value for key, value in result.items() if key != "records"}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
