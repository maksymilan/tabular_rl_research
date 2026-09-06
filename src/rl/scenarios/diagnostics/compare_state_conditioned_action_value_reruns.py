#!/usr/bin/env python3
"""Compare state-conditioned full-action preferences across two frozen rollout reruns."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from rl.scenarios.diagnostics.audit_state_conditioned_prefix_structure import (
    DecisionEvent,
    _parse_episode,
    load_jsonl,
)


SCHEMA_VERSION = "state-conditioned-action-value-rerun-comparison-v1"


def _state_groups(
    rows: Sequence[dict[str, Any]],
    *,
    representation: str,
) -> tuple[
    dict[tuple[str, str, int], dict[str, list[DecisionEvent]]],
    dict[str, set[bool]],
    list[str],
]:
    groups: dict[
        tuple[str, str, int], dict[str, list[DecisionEvent]]
    ] = defaultdict(lambda: defaultdict(list))
    task_outcomes: dict[str, set[bool]] = defaultdict(set)
    issues: list[str] = []
    for row_index, row in enumerate(rows):
        episode, episode_issues = _parse_episode(row, row_index=row_index)
        issues.extend(episode_issues)
        if episode is None or not episode.eligible:
            continue
        task_outcomes[episode.task_id].add(episode.correct)
        for event in episode.events:
            key = (event.task_id, event.state_key(representation), event.depth)
            groups[key][event.action_signature].append(event)
    return groups, task_outcomes, issues


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def compare_rows(
    rows_a: Sequence[dict[str, Any]],
    rows_b: Sequence[dict[str, Any]],
    *,
    minimum_action_count: int = 2,
    representation: str = "strict_policy_prompt",
) -> dict[str, Any]:
    groups_a, outcomes_a, issues_a = _state_groups(
        rows_a,
        representation=representation,
    )
    groups_b, outcomes_b, issues_b = _state_groups(
        rows_b,
        representation=representation,
    )
    shared_states = sorted(set(groups_a) & set(groups_b))
    comparisons: list[dict[str, Any]] = []
    for task_id, state_hash, depth in shared_states:
        actions_a = groups_a[(task_id, state_hash, depth)]
        actions_b = groups_b[(task_id, state_hash, depth)]
        supported_actions = sorted(
            action
            for action in set(actions_a) & set(actions_b)
            if len(actions_a[action]) >= minimum_action_count
            and len(actions_b[action]) >= minimum_action_count
        )
        for left_action, right_action in itertools.combinations(supported_actions, 2):
            left_a = actions_a[left_action]
            right_a = actions_a[right_action]
            left_b = actions_b[left_action]
            right_b = actions_b[right_action]
            rate_left_a = sum(event.correct for event in left_a) / len(left_a)
            rate_right_a = sum(event.correct for event in right_a) / len(right_a)
            rate_left_b = sum(event.correct for event in left_b) / len(left_b)
            rate_right_b = sum(event.correct for event in right_b) / len(right_b)
            difference_a = rate_left_a - rate_right_a
            difference_b = rate_left_b - rate_right_b
            sign_a = _sign(difference_a)
            sign_b = _sign(difference_b)
            if sign_a == 0 or sign_b == 0:
                stability = "zero_in_one_or_both"
            elif sign_a == sign_b:
                stability = "same_nonzero_sign"
            else:
                stability = "opposite_nonzero_sign"
            comparisons.append(
                {
                    "task_id": task_id,
                    "state_hash": state_hash,
                    "depth": depth,
                    "left_action": left_action,
                    "right_action": right_action,
                    "run_a": {
                        "left_count": len(left_a),
                        "left_successes": sum(event.correct for event in left_a),
                        "right_count": len(right_a),
                        "right_successes": sum(event.correct for event in right_a),
                        "rate_difference": difference_a,
                    },
                    "run_b": {
                        "left_count": len(left_b),
                        "left_successes": sum(event.correct for event in left_b),
                        "right_count": len(right_b),
                        "right_successes": sum(event.correct for event in right_b),
                        "rate_difference": difference_b,
                    },
                    "stability": stability,
                }
            )

    def comparison_summary(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(item["stability"] for item in selected)
        nonzero = counts["same_nonzero_sign"] + counts["opposite_nonzero_sign"]
        return {
            "action_pair_comparisons": len(selected),
            "stability_counts": dict(sorted(counts.items())),
            "same_sign_fraction_among_nonzero": (
                counts["same_nonzero_sign"] / nonzero if nonzero else 0.0
            ),
        }

    tasks_a = set(outcomes_a)
    tasks_b = set(outcomes_b)
    mixed_a = {task_id for task_id, values in outcomes_a.items() if values == {False, True}}
    mixed_b = {task_id for task_id, values in outcomes_b.items() if values == {False, True}}
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "gold_sql_read": False,
            "gold_answer_read": False,
            "model_calls": 0,
            "optimizer_updates": 0,
            "state_identity": representation,
            "action_identity": "parsed tool plus full arguments",
            "minimum_action_count_per_run": minimum_action_count,
        },
        "observed": {
            "run_a_rows": len(rows_a),
            "run_b_rows": len(rows_b),
            "run_a_issues": len(issues_a),
            "run_b_issues": len(issues_b),
            "run_a_eligible_tasks": len(tasks_a),
            "run_b_eligible_tasks": len(tasks_b),
            "shared_eligible_tasks": len(tasks_a & tasks_b),
            "shared_exact_states": len(shared_states),
        },
        "mixed_task_stability": {
            "run_a_mixed_tasks": len(mixed_a),
            "run_b_mixed_tasks": len(mixed_b),
            "mixed_in_both": len(mixed_a & mixed_b),
            "mixed_union": len(mixed_a | mixed_b),
            "jaccard": (
                len(mixed_a & mixed_b) / len(mixed_a | mixed_b)
                if mixed_a | mixed_b
                else 0.0
            ),
            "run_a_only": sorted(mixed_a - mixed_b),
            "run_b_only": sorted(mixed_b - mixed_a),
            "both": sorted(mixed_a & mixed_b),
        },
        "all_depths": comparison_summary(comparisons),
        "initial": comparison_summary(
            [item for item in comparisons if item["depth"] == 0]
        ),
        "noninitial": comparison_summary(
            [item for item in comparisons if item["depth"] > 0]
        ),
        "comparisons": comparisons,
        "issues": {
            "run_a": issues_a,
            "run_b": issues_b,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--minimum-action-count", type=int, default=2)
    parser.add_argument(
        "--representation",
        choices=(
            "tool_prefix",
            "exact_environment_prefix",
            "strict_policy_prompt",
        ),
        default="strict_policy_prompt",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_rows(
        load_jsonl(args.run_a),
        load_jsonl(args.run_b),
        minimum_action_count=args.minimum_action_count,
        representation=args.representation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["observed"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
