#!/usr/bin/env python3
"""Offline replay of a temporal SAAM guard over scored rollout records.

The production SAAM mask is batch-local.  This diagnostic adds a deliberately
small historical guard: for an exact Harness state-action key, remember the
sign of the last non-zero *accepted* update.  If a later batch proposes the
opposite sign, zero that key's current coefficients and keep the previous sign
as the anchor.  It is a replay diagnostic, not a claim about an exact gradient
vector and not a production training implementation.

Only terminal correctness and Harness-owned action/outcome state are used.  A
gold_sql field may be present in the raw audit record, but is never read.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from frameworks.trl.state_action_ambiguity import (  # noqa: E402
    _canonical_action,
    _harness_outcome,
    _sha256_json,
    _stable_json,
)
from frameworks.trl.transition_batch import standardized_group_advantages  # noqa: E402


SCHEMA_VERSION = "saam-temporal-replay-v1"


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            yield value


def _process_update(row: dict[str, Any]) -> bool:
    explicit = row.get("process_update")
    if isinstance(explicit, bool):
        return explicit
    if row.get("optimization_exclusion") is not None:
        return False
    result_reward = row.get("result_reward")
    return isinstance(result_reward, dict) and "value" in result_reward


def _events(row: dict[str, Any], advantage: float) -> list[dict[str, Any]]:
    if not _process_update(row):
        return []
    example_index = int(row["example_index"])
    correct = bool(row["correct"])
    prior_executions: list[Any] = []
    prefix_matchable = True
    result: list[dict[str, Any]] = []
    for depth, turn in enumerate(row.get("turns", [])):
        authored_action = _canonical_action(
            turn.get("parsed"), normalize_describe_tables=False
        )
        matched_action = _canonical_action(
            turn.get("parsed"), normalize_describe_tables=True
        )
        if prefix_matchable and matched_action is not None:
            state_signature = _sha256_json(prior_executions)
            action_signature = _stable_json(matched_action)
            key = _sha256_json(
                {
                    "example_index": example_index,
                    "state": state_signature,
                    "action": action_signature,
                }
            )
            result.append(
                {
                    "key": key,
                    "action": action_signature,
                    "tool": matched_action["tool"],
                    "depth": depth,
                    "correct": correct,
                    "advantage": float(advantage),
                }
            )
        if authored_action is None:
            prefix_matchable = False
            prior_executions.append(
                {
                    "unmatchable_turn": depth,
                    "trajectory_id": str(row.get("trajectory_id") or ""),
                    "outcome": _harness_outcome(turn),
                }
            )
        else:
            prior_executions.append(
                {"action": authored_action, "outcome": _harness_outcome(turn)}
            )
    return result


def replay(rows: Iterable[dict[str, Any]], *, expected_group_size: int) -> dict[str, Any]:
    grouped_rows: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(int(row["policy_global_step"]), int(row["example_index"]))].append(row)
    if not grouped_rows:
        raise ValueError("rollout log is empty")

    history_sign: dict[str, int] = {}
    totals = Counter()
    masses = Counter()
    by_tool: Counter[str] = Counter()
    history_conflict_keys: list[dict[str, Any]] = []
    # These are retained only for the final coarse action collision audit.
    applied_events: list[dict[str, Any]] = []

    for (policy_step, example_index), group in sorted(grouped_rows.items()):
        if len(group) != expected_group_size:
            raise ValueError(
                f"step={policy_step} example={example_index}: "
                f"group size {len(group)} != {expected_group_size}"
            )
        eligible = [_process_update(row) for row in group]
        rewards = [1.0 if row.get("correct") is True else 0.0 for row in group]
        advantages = standardized_group_advantages(rewards, eligible)
        events: list[dict[str, Any]] = []
        totals["eligible_episodes"] += sum(eligible)
        for row, advantage in zip(group, advantages, strict=True):
            events.extend(_events(row, advantage))
        totals["matchable_events"] += len(events)

        by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            by_key[event["key"]].append(event)

        for key, key_events in by_key.items():
            # This is exactly the current batch-local SAAM decision.
            mixed = {event["correct"] for event in key_events} == {False, True}
            for event in key_events:
                event["local_advantage"] = (
                    0.0 if mixed else event["advantage"]
                )
            local_sum = sum(event["local_advantage"] for event in key_events)
            previous_sign = history_sign.get(key)
            current_sign = 0 if local_sum == 0.0 else (1 if local_sum > 0 else -1)
            historical_conflict = (
                current_sign != 0
                and previous_sign is not None
                and current_sign != previous_sign
            )
            if historical_conflict:
                totals["historical_conflict_key_occurrences"] += 1
                totals["historical_conflict_events"] += sum(
                    event["local_advantage"] != 0.0 for event in key_events
                )
                totals["historical_conflict_all_events"] += len(key_events)
                totals["historical_conflict_positive_events"] += sum(
                    event["local_advantage"] > 0.0 for event in key_events
                )
                totals["historical_conflict_negative_events"] += sum(
                    event["local_advantage"] < 0.0 for event in key_events
                )
                for event in key_events:
                    value = float(event["local_advantage"])
                    masses["historical_removed_absolute_mass"] += abs(value)
                    masses[
                        "historical_removed_correct_absolute_mass"
                        if event["correct"]
                        else "historical_removed_incorrect_absolute_mass"
                    ] += abs(value)
                    if value > 0.0:
                        masses["historical_removed_positive_mass"] += value
                    elif value < 0.0:
                        masses["historical_removed_negative_mass"] += -value
                history_conflict_keys.append(
                    {
                        "policy_global_step": policy_step,
                        "example_index": example_index,
                        "key": key,
                        "tool": key_events[0]["tool"],
                        "previous_sign": previous_sign,
                        "current_sign": current_sign,
                        "local_sum": local_sum,
                        "depths": sorted({event["depth"] for event in key_events}),
                    }
                )
                for event in key_events:
                    event["temporal_advantage"] = 0.0
                # Hysteresis: an unaccepted opposite sign does not replace the
                # historical anchor.
            else:
                for event in key_events:
                    event["temporal_advantage"] = event["local_advantage"]
                if current_sign != 0:
                    history_sign[key] = current_sign
            applied_events.extend(
                {
                    **event,
                    "policy_step": policy_step,
                    "example_index": example_index,
                }
                for event in key_events
            )

    # Verify the intended exact-key property after temporal masking.
    by_key_update: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    by_action_update: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for event in applied_events:
        by_key_update[(event["policy_step"], event["key"])].append(event)
        by_action_update[
            (event["policy_step"], event["example_index"], event["action"])
        ].append(event)

    exact_conflict_mass = 0.0
    coarse_conflict_mass = 0.0
    coarse_conflict_groups = 0
    for events_for_key in by_key_update.values():
        positive = sum(
            max(0.0, float(event["temporal_advantage"]))
            for event in events_for_key
        )
        negative = sum(
            max(0.0, -float(event["temporal_advantage"]))
            for event in events_for_key
        )
        exact_conflict_mass += 2.0 * min(positive, negative)
    for events_for_action in by_action_update.values():
        positive = sum(
            max(0.0, float(event["temporal_advantage"]))
            for event in events_for_action
        )
        negative = sum(
            max(0.0, -float(event["temporal_advantage"]))
            for event in events_for_action
        )
        if positive > 0.0 and negative > 0.0:
            coarse_conflict_groups += 1
            coarse_conflict_mass += 2.0 * min(positive, negative)

    # Compare exact-key signs before and after the historical guard.
    sign_sequences: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (policy_step, key), events_for_key in by_key_update.items():
        sign_sequences[key].append(
            (policy_step, sum(event["temporal_advantage"] for event in events_for_key))
        )
    temporal_nonzero_transitions = 0
    temporal_sign_flips = 0
    temporal_zero_touch = 0
    for sequence in sign_sequences.values():
        sequence.sort()
        for (_, previous), (_, current) in zip(sequence, sequence[1:]):
            if previous == 0.0 or current == 0.0:
                temporal_zero_touch += 1
            else:
                temporal_nonzero_transitions += 1
                temporal_sign_flips += int((previous > 0.0) != (current > 0.0))

    tool_counts = Counter(item["tool"] for item in history_conflict_keys)
    return {
        "schema_version": SCHEMA_VERSION,
        "history_rule": (
            "batch-local SAAM first; exact state-action key remembers last "
            "nonzero accepted sign; opposite current sign is zeroed and does "
            "not overwrite the historical sign"
        ),
        "expected_group_size": expected_group_size,
        "gold_sql_read": False,
        "groups": len(grouped_rows),
        "distinct_history_keys": len(history_sign),
        **dict(totals),
        "historical_conflict_keys_by_tool": dict(sorted(tool_counts.items())),
        "historical_removed_absolute_mass": masses["historical_removed_absolute_mass"],
        "historical_removed_positive_mass": masses["historical_removed_positive_mass"],
        "historical_removed_negative_mass": masses["historical_removed_negative_mass"],
        "historical_removed_correct_absolute_mass": masses[
            "historical_removed_correct_absolute_mass"
        ],
        "historical_removed_incorrect_absolute_mass": masses[
            "historical_removed_incorrect_absolute_mass"
        ],
        "exact_conflict_mass_after_temporal_guard": exact_conflict_mass,
        "coarse_action_conflict_groups_after_temporal_guard": coarse_conflict_groups,
        "coarse_action_conflict_mass_after_temporal_guard": coarse_conflict_mass,
        "temporal_nonzero_transitions": temporal_nonzero_transitions,
        "temporal_sign_flips": temporal_sign_flips,
        "temporal_zero_touch": temporal_zero_touch,
        "temporal_sign_flip_rate_nonzero": (
            temporal_sign_flips / temporal_nonzero_transitions
            if temporal_nonzero_transitions
            else 0.0
        ),
        "historical_conflict_examples": history_conflict_keys[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(_rows(args.rollouts), expected_group_size=args.group_size)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
