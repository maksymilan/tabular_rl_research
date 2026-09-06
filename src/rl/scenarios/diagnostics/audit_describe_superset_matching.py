#!/usr/bin/env python3
"""Audit directional describe-table superset matching for ambiguity-masked GRPO."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from rl.scenarios.diagnostics.audit_state_conditioned_prefix_structure import (
    DecisionEvent,
    Episode,
    _parse_episode,
    load_jsonl,
)


SCHEMA_VERSION = "describe-superset-ambiguity-audit-v1"


def _describe_tables(action_signature: str) -> frozenset[str] | None:
    value = json.loads(action_signature)
    if value.get("tool") != "describe_table":
        return None
    arguments = value.get("arguments")
    tables = arguments.get("tables") if isinstance(arguments, dict) else None
    if not (
        isinstance(tables, list)
        and tables
        and all(isinstance(table, str) and table for table in tables)
    ):
        return None
    return frozenset(tables)


def _parse_eligible_episodes(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, list[Episode]], list[str]]:
    episodes_by_task: dict[str, list[Episode]] = defaultdict(list)
    issues: list[str] = []
    for row_index, row in enumerate(rows):
        episode, episode_issues = _parse_episode(row, row_index=row_index)
        issues.extend(episode_issues)
        if episode is not None and episode.eligible:
            episodes_by_task[episode.task_id].append(episode)
    return episodes_by_task, issues


def _mask_summary(
    episodes_by_task: dict[str, list[Episode]],
    *,
    masked_event_keys: set[tuple[str, int]],
) -> dict[str, Any]:
    all_events = [
        event
        for episodes in episodes_by_task.values()
        for episode in episodes
        for event in episode.events
    ]
    mixed_task_ids = {
        task_id
        for task_id, episodes in episodes_by_task.items()
        if {episode.correct for episode in episodes} == {False, True}
    }
    mixed_events = [event for event in all_events if event.task_id in mixed_task_ids]
    masked_events = [
        event
        for event in mixed_events
        if (event.episode_id, event.depth) in masked_event_keys
    ]
    trajectory_counts: Counter[str] = Counter()
    total_mass = 0.0
    masked_mass = 0.0
    total_response_tokens = 0
    masked_response_tokens = 0
    known_token_episodes = 0
    skipped_token_episodes = 0
    for task_id in mixed_task_ids:
        episodes = episodes_by_task[task_id]
        success_fraction = sum(episode.correct for episode in episodes) / len(episodes)
        standard_deviation = math.sqrt(
            success_fraction * (1.0 - success_fraction)
        )
        for episode in episodes:
            label = "correct" if episode.correct else "incorrect"
            flags = [
                (event.episode_id, event.depth) in masked_event_keys
                for event in episode.events
            ]
            trajectory_counts[f"{label}_trajectories"] += 1
            if any(flags):
                trajectory_counts[f"{label}_with_masked_turn"] += 1
            if flags and all(flags):
                trajectory_counts[f"{label}_all_turns_masked"] += 1
            if any(not flag for flag in flags):
                trajectory_counts[f"{label}_retaining_unmasked_turn"] += 1

            token_counts = [event.response_token_count for event in episode.events]
            if not token_counts or any(count is None for count in token_counts):
                skipped_token_episodes += 1
                continue
            trajectory_tokens = sum(int(count) for count in token_counts)
            if trajectory_tokens <= 0:
                skipped_token_episodes += 1
                continue
            known_token_episodes += 1
            advantage = (
                (1.0 - success_fraction) / standard_deviation
                if episode.correct
                else -success_fraction / standard_deviation
            )
            for event, masked in zip(episode.events, flags):
                response_tokens = int(event.response_token_count or 0)
                mass = abs(advantage) * response_tokens / trajectory_tokens
                total_mass += mass
                total_response_tokens += response_tokens
                if masked:
                    masked_mass += mass
                    masked_response_tokens += response_tokens

    return {
        "masked_events": len(masked_events),
        "masked_initial_events": sum(event.depth == 0 for event in masked_events),
        "masked_noninitial_events": sum(event.depth > 0 for event in masked_events),
        "masked_tasks": len({event.task_id for event in masked_events}),
        "all_eligible_events": len(all_events),
        "mixed_task_events": len(mixed_events),
        "masked_all_eligible_event_fraction": (
            len(masked_events) / len(all_events) if all_events else 0.0
        ),
        "masked_mixed_task_event_fraction": (
            len(masked_events) / len(mixed_events) if mixed_events else 0.0
        ),
        "known_token_count_episodes": known_token_episodes,
        "skipped_missing_token_count_episodes": skipped_token_episodes,
        "absolute_policy_coefficient_mass": total_mass,
        "masked_absolute_policy_coefficient_mass": masked_mass,
        "masked_absolute_policy_coefficient_mass_fraction": (
            masked_mass / total_mass if total_mass else 0.0
        ),
        "response_tokens": total_response_tokens,
        "masked_response_tokens": masked_response_tokens,
        "masked_response_token_fraction": (
            masked_response_tokens / total_response_tokens
            if total_response_tokens
            else 0.0
        ),
        "trajectory_counts": dict(sorted(trajectory_counts.items())),
    }


def _representation_audit(
    episodes_by_task: dict[str, list[Episode]],
    *,
    representation: str,
) -> dict[str, Any]:
    state_groups: dict[tuple[str, str], list[DecisionEvent]] = defaultdict(list)
    for task_id, episodes in episodes_by_task.items():
        for episode in episodes:
            for event in episode.events:
                state_groups[(task_id, event.state_key(representation))].append(event)

    exact_masked: set[tuple[str, int]] = set()
    relaxed_masked: set[tuple[str, int]] = set()
    relation_counts: Counter[str] = Counter()
    relation_tasks: dict[str, set[str]] = defaultdict(set)
    extra_table_histogram: Counter[int] = Counter()
    added_event_keys: set[tuple[str, int]] = set()
    added_noninitial_event_keys: set[tuple[str, int]] = set()
    examples: list[dict[str, Any]] = []

    for (task_id, state_hash), events in state_groups.items():
        action_groups: dict[str, list[DecisionEvent]] = defaultdict(list)
        for event in events:
            action_groups[event.action_signature].append(event)
        for action_events in action_groups.values():
            if len(action_events) >= 2 and {
                event.correct for event in action_events
            } == {False, True}:
                exact_masked.update(
                    (event.episode_id, event.depth) for event in action_events
                )

        correct_describe: dict[str, list[DecisionEvent]] = {}
        incorrect_describe: dict[str, list[DecisionEvent]] = {}
        for action, action_events in action_groups.items():
            if _describe_tables(action) is None:
                continue
            correct_events = [event for event in action_events if event.correct]
            incorrect_events = [event for event in action_events if not event.correct]
            if correct_events:
                correct_describe[action] = correct_events
            if incorrect_events:
                incorrect_describe[action] = incorrect_events

        for correct_action, correct_events in correct_describe.items():
            correct_tables = _describe_tables(correct_action)
            assert correct_tables is not None
            for incorrect_action, incorrect_events in incorrect_describe.items():
                incorrect_tables = _describe_tables(incorrect_action)
                assert incorrect_tables is not None
                if incorrect_tables == correct_tables:
                    relation = "equal"
                elif incorrect_tables > correct_tables:
                    relation = "incorrect_strict_superset"
                elif correct_tables > incorrect_tables:
                    relation = "incorrect_strict_subset_rejected"
                else:
                    relation = "incomparable_rejected"
                relation_counts[f"unique_action_pairs_{relation}"] += 1
                relation_tasks[f"unique_action_pairs_{relation}"].add(task_id)
                relation_counts[f"event_pairs_{relation}"] += (
                    len(correct_events) * len(incorrect_events)
                )
                if relation not in {"equal", "incorrect_strict_superset"}:
                    continue
                if relation == "incorrect_strict_superset":
                    extra_table_histogram[
                        len(incorrect_tables - correct_tables)
                    ] += 1
                endpoints = correct_events + incorrect_events
                endpoint_keys = {
                    (event.episode_id, event.depth) for event in endpoints
                }
                relaxed_masked.update(endpoint_keys)
                new_keys = endpoint_keys - exact_masked
                added_event_keys.update(new_keys)
                added_noninitial_event_keys.update(
                    (event.episode_id, event.depth)
                    for event in endpoints
                    if event.depth > 0
                    and (event.episode_id, event.depth) in new_keys
                )
                if relation == "incorrect_strict_superset" and len(examples) < 20:
                    examples.append(
                        {
                            "task_id": task_id,
                            "state_hash": state_hash,
                            "depth": min(event.depth for event in endpoints),
                            "correct_tables": sorted(correct_tables),
                            "incorrect_tables": sorted(incorrect_tables),
                            "correct_event_count": len(correct_events),
                            "incorrect_event_count": len(incorrect_events),
                        }
                    )

    relaxed_masked.update(exact_masked)
    exact_summary = _mask_summary(
        episodes_by_task,
        masked_event_keys=exact_masked,
    )
    relaxed_summary = _mask_summary(
        episodes_by_task,
        masked_event_keys=relaxed_masked,
    )
    return {
        "representation": representation,
        "describe_relation_counts": dict(sorted(relation_counts.items())),
        "describe_relation_task_counts": {
            key: len(value) for key, value in sorted(relation_tasks.items())
        },
        "strict_superset_extra_table_histogram": {
            str(key): value for key, value in sorted(extra_table_histogram.items())
        },
        "exact_full_action_mask": exact_summary,
        "relaxed_describe_superset_mask": relaxed_summary,
        "delta": {
            "added_events": len(added_event_keys),
            "added_noninitial_events": len(added_noninitial_event_keys),
            "added_tasks": len(
                {
                    event.task_id
                    for events in state_groups.values()
                    for event in events
                    if (event.episode_id, event.depth) in added_event_keys
                }
            ),
            "added_absolute_policy_coefficient_mass": (
                relaxed_summary["masked_absolute_policy_coefficient_mass"]
                - exact_summary["masked_absolute_policy_coefficient_mass"]
            ),
            "added_absolute_policy_coefficient_mass_fraction_points": (
                relaxed_summary[
                    "masked_absolute_policy_coefficient_mass_fraction"
                ]
                - exact_summary[
                    "masked_absolute_policy_coefficient_mass_fraction"
                ]
            ),
            "added_response_tokens": (
                relaxed_summary["masked_response_tokens"]
                - exact_summary["masked_response_tokens"]
            ),
        },
        "strict_superset_examples": examples,
    }


def audit_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    episodes_by_task, issues = _parse_eligible_episodes(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "gold_sql_read": False,
            "gold_answer_read": False,
            "model_calls": 0,
            "optimizer_updates": 0,
            "non_describe_action_identity": "exact parsed tool plus full arguments",
            "describe_match_rule": (
                "within the same state, incorrect tables must be a superset of "
                "a correct trajectory's describe tables"
            ),
            "directional_not_equivalence_relation": True,
            "empty_describe_table_list_allowed": False,
        },
        "observed": {
            "rows": len(rows),
            "eligible_episodes": sum(len(value) for value in episodes_by_task.values()),
            "eligible_tasks": len(episodes_by_task),
            "issues": len(issues),
        },
        "representations": {
            representation: _representation_audit(
                episodes_by_task,
                representation=representation,
            )
            for representation in (
                "exact_environment_prefix",
                "strict_policy_prompt",
            )
        },
        "issues": issues,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_rows(load_jsonl(args.trajectories))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["observed"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
