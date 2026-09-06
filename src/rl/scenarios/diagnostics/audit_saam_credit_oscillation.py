#!/usr/bin/env python3
"""Audit semantic state-action credit conflict and cross-update sign reversals.

This consumes append-only online ``rollouts.jsonl`` records.  It uses binary
terminal correctness and Harness-owned action/outcome evidence only; gold fields
and model-authored reasoning are never read.  Because rollout logs do not retain
the trainer's response-token weights, cross-update results are explicitly a raw
GRPO-advantage sign proxy, not an exact gradient-vector measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from rl.frameworks.trl.state_action_ambiguity import (  # noqa: E402
    _canonical_action,
    _harness_outcome,
    _sha256_json,
    _stable_json,
)
from rl.frameworks.trl.transition_batch import standardized_group_advantages  # noqa: E402


SCHEMA_VERSION = "saam-credit-oscillation-audit-v1"


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
    """Read either the newer explicit flag or the legacy raw rollout record.

    Historical online ``rollouts.jsonl`` stored the Harness audit record rather
    than the enclosing ``RolloutSample``.  In that schema the trainer's
    fail-closed exclusion is preserved as ``optimization_exclusion`` and a
    scored row is identified by its structured ``result_reward`` object.
    """
    explicit = row.get("process_update")
    if isinstance(explicit, bool):
        return explicit
    if row.get("optimization_exclusion") is not None:
        return False
    result_reward = row.get("result_reward")
    return isinstance(result_reward, dict) and "value" in result_reward


def _events(
    row: dict[str, Any],
    *,
    advantage: float,
) -> list[dict[str, Any]]:
    if not _process_update(row):
        return []
    example_index = int(row["example_index"])
    correct = bool(row["correct"])
    turns = row.get("turns")
    if not isinstance(turns, list):
        raise ValueError("eligible rollout has no audited turns")
    prior_executions: list[Any] = []
    prefix_matchable = True
    result = []
    for depth, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError("audited turn is not an object")
        authored_action = _canonical_action(
            turn.get("parsed"), normalize_describe_tables=False
        )
        matched_action = _canonical_action(
            turn.get("parsed"), normalize_describe_tables=True
        )
        if prefix_matchable and matched_action is not None:
            state_signature = _sha256_json(prior_executions)
            action_signature = _stable_json(matched_action)
            semantic_key = _sha256_json(
                {
                    "example_index": example_index,
                    "state": state_signature,
                    "action": action_signature,
                }
            )
            result.append(
                {
                    "semantic_key": semantic_key,
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


def audit_rollouts(
    rows: Iterable[dict[str, Any]],
    *,
    credit_assignment: str,
    expected_group_size: int,
) -> dict[str, Any]:
    if credit_assignment not in {"trajectory", "saam-strict"}:
        raise ValueError(f"unsupported credit assignment: {credit_assignment}")
    if expected_group_size < 2:
        raise ValueError("expected group size must be at least two")
    grouped_rows = defaultdict(list)
    for row in rows:
        if "gold_sql" in row:
            # Its presence in a Harness audit is allowed, but this function never
            # reads the value.  Record the contract explicitly in the receipt.
            pass
        key = (int(row["policy_global_step"]), int(row["example_index"]))
        grouped_rows[key].append(row)
    if not grouped_rows:
        raise ValueError("rollout log is empty")

    batch_key_values = defaultdict(list)
    key_metadata: dict[str, tuple[str, int]] = {}
    groups_with_mixed_reward = 0
    eligible_episodes = 0
    for (policy_step, example_index), group in sorted(grouped_rows.items()):
        if len(group) != expected_group_size:
            raise ValueError(
                f"step={policy_step} example={example_index}: "
                f"group size {len(group)} != {expected_group_size}"
            )
        eligible = [_process_update(row) for row in group]
        rewards = [1.0 if row.get("correct") is True else 0.0 for row in group]
        advantages = standardized_group_advantages(rewards, eligible)
        eligible_episodes += sum(eligible)
        eligible_outcomes = {
            bool(row["correct"])
            for row, keep in zip(group, eligible, strict=True)
            if keep
        }
        groups_with_mixed_reward += int(eligible_outcomes == {False, True})
        for row, advantage in zip(group, advantages, strict=True):
            for event in _events(row, advantage=advantage):
                key = event["semantic_key"]
                batch_key_values[(policy_step, key)].append(event)
                key_metadata[key] = (event["tool"], event["depth"])

    mixed_keys = 0
    mixed_events = 0
    vanilla_positive_mass = 0.0
    vanilla_negative_mass = 0.0
    vanilla_two_sided_conflict_mass = 0.0
    applied_direct_conflict_mass = 0.0
    applied_by_step_and_key = defaultdict(float)
    mixed_key_occurrences_by_tool = defaultdict(int)
    for (policy_step, semantic_key), events in batch_key_values.items():
        outcomes = {event["correct"] for event in events}
        mixed = outcomes == {False, True}
        values = [float(event["advantage"]) for event in events]
        positive = sum(value for value in values if value > 0.0)
        negative = sum(-value for value in values if value < 0.0)
        if mixed:
            mixed_keys += 1
            mixed_events += len(events)
            vanilla_positive_mass += positive
            vanilla_negative_mass += negative
            vanilla_two_sided_conflict_mass += 2.0 * min(positive, negative)
            tool, _ = key_metadata[semantic_key]
            mixed_key_occurrences_by_tool[tool] += 1
        applied_values = (
            [0.0] * len(values)
            if credit_assignment == "saam-strict" and mixed
            else values
        )
        applied_positive = sum(value for value in applied_values if value > 0.0)
        applied_negative = sum(-value for value in applied_values if value < 0.0)
        applied_direct_conflict_mass += 2.0 * min(
            applied_positive, applied_negative
        )
        applied_by_step_and_key[(policy_step, semantic_key)] = sum(applied_values)

    appearances_by_key = defaultdict(list)
    for (policy_step, semantic_key), value in applied_by_step_and_key.items():
        appearances_by_key[semantic_key].append((policy_step, value))
    recurrent_keys = 0
    comparable_sign_transitions = 0
    sign_flips = 0
    transitions_touching_zero = 0
    for appearances in appearances_by_key.values():
        appearances.sort()
        if len(appearances) < 2:
            continue
        recurrent_keys += 1
        for (_, previous), (_, current) in zip(appearances, appearances[1:]):
            if previous == 0.0 or current == 0.0:
                transitions_touching_zero += 1
                continue
            comparable_sign_transitions += 1
            sign_flips += int((previous > 0.0) != (current > 0.0))

    return {
        "schema_version": SCHEMA_VERSION,
        "credit_assignment": credit_assignment,
        "expected_group_size": expected_group_size,
        "reward_authority": "binary terminal correctness only",
        "gold_sql_read": False,
        "coefficient_scope": "raw group-standardized trajectory advantage",
        "gradient_vector_claimed": False,
        "groups": len(grouped_rows),
        "groups_with_mixed_reward": groups_with_mixed_reward,
        "eligible_episodes": eligible_episodes,
        "semantic_state_action_occurrences": len(batch_key_values),
        "mixed_state_action_occurrences": mixed_keys,
        "mixed_events": mixed_events,
        "mixed_key_occurrences_by_tool": dict(
            sorted(mixed_key_occurrences_by_tool.items())
        ),
        "vanilla_mixed_positive_mass": vanilla_positive_mass,
        "vanilla_mixed_negative_mass": vanilla_negative_mass,
        "vanilla_two_sided_direct_conflict_mass": (
            vanilla_two_sided_conflict_mass
        ),
        "applied_two_sided_direct_conflict_mass": applied_direct_conflict_mass,
        "recurrent_semantic_keys": recurrent_keys,
        "comparable_nonzero_sign_transitions": comparable_sign_transitions,
        "credit_sign_flips": sign_flips,
        "credit_sign_flip_rate": (
            sign_flips / comparable_sign_transitions
            if comparable_sign_transitions
            else 0.0
        ),
        "credit_sign_flips_per_recurrent_transition": (
            sign_flips
            / (comparable_sign_transitions + transitions_touching_zero)
            if comparable_sign_transitions + transitions_touching_zero
            else 0.0
        ),
        "cross_update_transitions_touching_zero": transitions_touching_zero,
        "batch_local_guarantee_passed": (
            credit_assignment != "saam-strict"
            or applied_direct_conflict_mass == 0.0
        ),
        "cross_update_no_flip_guaranteed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument(
        "--credit-assignment",
        choices=("trajectory", "saam-strict"),
        required=True,
    )
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_rollouts(
        _rows(args.rollouts),
        credit_assignment=args.credit_assignment,
        expected_group_size=args.group_size,
    )
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
