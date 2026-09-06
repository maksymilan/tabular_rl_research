#!/usr/bin/env python3
"""Audit repeated decision states in a fixed multi-rollout agent pool.

The audit is deliberately reward-minimal.  It never reads gold SQL or answer rows.  Terminal
correctness is used only to ask whether an already repeated decision state contains mixed
continuation outcomes.

Three state identities are reported:

* ``tool_prefix``: prior tool names only (an intentionally loose upper bound);
* ``exact_environment_prefix``: prior parsed ``tool + full arguments`` and Harness feedback;
* ``strict_policy_prompt``: the exact token ids supplied to the policy before the next action.

The next-action identity always contains the parsed tool and complete argument object.  JSON
object key order is ignored.  Only ``describe_table.tables`` is sorted because that public field
is a proven unordered set of catalog tables; all other list order is conservatively preserved.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from sft.action_carrier import ActionCarrierError, parse_action_carrier


SCHEMA_VERSION = "state-conditioned-prefix-structure-audit-v1"
STATE_REPRESENTATIONS = (
    "tool_prefix",
    "exact_environment_prefix",
    "strict_policy_prompt",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_observation(content: Any) -> str:
    if not isinstance(content, str):
        return _stable_json(content)
    try:
        return _stable_json(json.loads(content))
    except (json.JSONDecodeError, TypeError):
        return content


def _normalize_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(_stable_json(arguments))
    if tool == "describe_table":
        tables = normalized.get("tables")
        if isinstance(tables, list) and all(isinstance(table, str) for table in tables):
            normalized["tables"] = sorted(tables)
    return normalized


def _action_signature(call: dict[str, Any], *, schema_normalized: bool) -> str:
    tool = call["tool"]
    arguments = call["arguments"]
    return _stable_json(
        {
            "tool": tool,
            "arguments": (
                _normalize_arguments(tool, arguments)
                if schema_normalized
                else arguments
            ),
        }
    )


def _return_columns_ablated_signature(action_signature: str) -> str:
    """Diagnostic-only coarsening; it is not asserted to preserve semantics."""
    value = json.loads(action_signature)
    arguments = value.get("arguments")
    if isinstance(arguments, dict):
        arguments.pop("return_columns", None)
    return _stable_json(value)


@dataclass(frozen=True)
class DecisionEvent:
    episode_id: str
    task_id: str
    sample_index: int
    depth: int
    correct: bool
    tool: str
    action_signature: str
    authored_action_signature: str
    observation_signature: str
    tool_prefix_state: str
    exact_environment_prefix_state: str
    strict_policy_prompt_state: str
    response_token_count: int | None

    def state_key(self, representation: str) -> str:
        if representation == "tool_prefix":
            return self.tool_prefix_state
        if representation == "exact_environment_prefix":
            return self.exact_environment_prefix_state
        if representation == "strict_policy_prompt":
            return self.strict_policy_prompt_state
        raise KeyError(representation)


@dataclass(frozen=True)
class Episode:
    episode_id: str
    task_id: str
    sample_index: int
    correct: bool
    eligible: bool
    events: tuple[DecisionEvent, ...]


def _adapt_flat_training_rollout(row: dict[str, Any]) -> dict[str, Any]:
    """Adapt frozen TRL rollout logs without reading their gold fields."""
    if isinstance(row.get("environment"), dict) and isinstance(row.get("sample"), dict):
        return row
    turns = row.get("turns")
    final_messages = row.get("final_messages")
    trajectory_id = row.get("trajectory_id")
    example_index = row.get("example_index")
    if not (
        isinstance(turns, list)
        and isinstance(final_messages, list)
        and isinstance(trajectory_id, str)
        and type(example_index) is int
        and isinstance(row.get("correct"), bool)
    ):
        return row
    sample_suffix = trajectory_id.rsplit("_sample_", 1)
    if len(sample_suffix) != 2 or not sample_suffix[1].isdigit():
        return row
    policy_turns = []
    for turn in turns:
        model_input = turn.get("model_input") if isinstance(turn, dict) else None
        policy_turns.append({"prompt_messages": model_input})
    result_reward = row.get("result_reward")
    process_update = (
        isinstance(result_reward, dict)
        and result_reward.get("executable_terminal") is True
    )
    return {
        "environment": {
            "dataset_split": "train",
            "task_id": f"bird_train_{example_index:05d}",
        },
        "sample": {
            "correct": row["correct"],
            "process_update": process_update,
            "audit_record": {
                "sample_index": int(sample_suffix[1]),
                "protocol_version": row.get("protocol_version"),
                "protocol_hash": row.get("protocol_hash"),
                "final_messages": final_messages,
            },
        },
        "policy_turns": policy_turns,
    }


def _parse_episode(row: dict[str, Any], *, row_index: int) -> tuple[Episode | None, list[str]]:
    row = _adapt_flat_training_rollout(row)
    issues: list[str] = []
    environment = row.get("environment")
    sample = row.get("sample")
    if not isinstance(environment, dict) or not isinstance(sample, dict):
        return None, [f"row[{row_index}]: missing environment/sample objects"]
    audit_record = sample.get("audit_record")
    if not isinstance(audit_record, dict):
        return None, [f"row[{row_index}]: missing sample.audit_record"]
    task_id = environment.get("task_id")
    sample_index = audit_record.get("sample_index")
    correct = sample.get("correct")
    messages = audit_record.get("final_messages")
    policy_turns = row.get("policy_turns")
    if not isinstance(task_id, str) or not task_id:
        return None, [f"row[{row_index}]: missing task_id"]
    if type(sample_index) is not int:
        return None, [f"row[{row_index}]: missing integer sample_index"]
    if not isinstance(correct, bool):
        return None, [f"row[{row_index}]: missing binary correctness"]
    if not isinstance(messages, list) or not isinstance(policy_turns, list):
        return None, [f"row[{row_index}]: missing final_messages/policy_turns"]

    assistant_positions = [
        position
        for position, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if len(assistant_positions) != len(policy_turns):
        return None, [
            f"row[{row_index}]: {len(assistant_positions)} assistant messages but "
            f"{len(policy_turns)} policy turns"
        ]

    episode_id = f"{task_id}:sample{sample_index}"
    prior_tools: list[str] = []
    prior_executions: list[list[str]] = []
    events: list[DecisionEvent] = []
    for depth, (message_position, policy_turn) in enumerate(
        zip(assistant_positions, policy_turns)
    ):
        assistant_message = messages[message_position]
        content = assistant_message.get("content")
        if not isinstance(content, str):
            return None, [f"{episode_id}:turn{depth}: assistant content is not text"]
        try:
            _, call = parse_action_carrier(content)
        except ActionCarrierError as exc:
            error_code = getattr(exc, "code", type(exc).__name__)
            return None, [
                f"{episode_id}:turn{depth}: unparseable carrier {error_code}"
            ]
        if (
            not isinstance(call, dict)
            or not isinstance(call.get("tool"), str)
            or not isinstance(call.get("arguments"), dict)
        ):
            return None, [f"{episode_id}:turn{depth}: invalid parsed action shape"]

        observation = "__NO_OBSERVATION__"
        if message_position + 1 < len(messages):
            next_message = messages[message_position + 1]
            if isinstance(next_message, dict) and next_message.get("role") == "user":
                observation = _canonical_observation(next_message.get("content"))

        prompt_ids = policy_turn.get("prompt_ids") if isinstance(policy_turn, dict) else None
        prompt_messages = (
            policy_turn.get("prompt_messages")
            if isinstance(policy_turn, dict)
            else None
        )
        if (
            isinstance(prompt_ids, list)
            and prompt_ids
            and all(type(token) is int for token in prompt_ids)
        ):
            strict_prompt_state = _sha256_text(_stable_json(prompt_ids))
        elif isinstance(prompt_messages, list) and prompt_messages:
            strict_prompt_state = _sha256_text(_stable_json(prompt_messages))
        else:
            return None, [f"{episode_id}:turn{depth}: invalid policy prompt ids"]
        response_ids = (
            policy_turn.get("response_ids")
            if isinstance(policy_turn, dict)
            else None
        )
        response_token_count = (
            len(response_ids)
            if isinstance(response_ids, list)
            and response_ids
            and all(type(token) is int for token in response_ids)
            else None
        )

        tool = call["tool"]
        action_signature = _action_signature(call, schema_normalized=True)
        authored_action_signature = _action_signature(call, schema_normalized=False)
        events.append(
            DecisionEvent(
                episode_id=episode_id,
                task_id=task_id,
                sample_index=sample_index,
                depth=depth,
                correct=correct,
                tool=tool,
                action_signature=action_signature,
                authored_action_signature=authored_action_signature,
                observation_signature=observation,
                tool_prefix_state=_sha256_text(_stable_json(prior_tools)),
                exact_environment_prefix_state=_sha256_text(
                    _stable_json(prior_executions)
                ),
                strict_policy_prompt_state=strict_prompt_state,
                response_token_count=response_token_count,
            )
        )
        prior_tools.append(tool)
        prior_executions.append([authored_action_signature, observation])

    return (
        Episode(
            episode_id=episode_id,
            task_id=task_id,
            sample_index=sample_index,
            correct=correct,
            eligible=sample.get("process_update") is True,
            events=tuple(events),
        ),
        issues,
    )


def _longest_common_prefix(sequences: Sequence[Sequence[str]]) -> int:
    if not sequences:
        return 0
    limit = min(len(sequence) for sequence in sequences)
    for index in range(limit):
        if len({sequence[index] for sequence in sequences}) != 1:
            return index
    return limit


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return {
        str(key): count
        for key, count in sorted(Counter(values).items())
    }


def _first_action_summary(episodes_by_task: dict[str, list[Episode]]) -> dict[str, Any]:
    all_tools: Counter[str] = Counter()
    unanimous_tool_tasks = 0
    unanimous_schema_normalized_action_tasks = 0
    unanimous_authored_action_tasks = 0
    all_describe_tasks = 0
    same_tool_different_arguments_tasks = 0
    unique_tool_histogram: Counter[int] = Counter()
    unique_action_histogram: Counter[int] = Counter()
    unique_authored_action_histogram: Counter[int] = Counter()
    modal_exact_shares: list[float] = []
    variation_examples: list[dict[str, Any]] = []

    for task_id, episodes in sorted(episodes_by_task.items()):
        first_events = [episode.events[0] for episode in episodes if episode.events]
        if not first_events:
            continue
        tool_counts = Counter(event.tool for event in first_events)
        action_counts = Counter(event.action_signature for event in first_events)
        authored_action_counts = Counter(
            event.authored_action_signature for event in first_events
        )
        all_tools.update(tool_counts)
        unique_tool_histogram[len(tool_counts)] += 1
        unique_action_histogram[len(action_counts)] += 1
        unique_authored_action_histogram[len(authored_action_counts)] += 1
        if len(tool_counts) == 1:
            unanimous_tool_tasks += 1
        if len(action_counts) == 1:
            unanimous_schema_normalized_action_tasks += 1
        if len(authored_action_counts) == 1:
            unanimous_authored_action_tasks += 1
        if set(tool_counts) == {"describe_table"}:
            all_describe_tasks += 1
        if len(tool_counts) == 1 and len(action_counts) > 1:
            same_tool_different_arguments_tasks += 1
        modal_exact_shares.append(max(action_counts.values()) / len(first_events))
        if len(action_counts) > 1:
            variation_examples.append(
                {
                    "task_id": task_id,
                    "trajectories": len(first_events),
                    "actions": [
                        {
                            "action": signature,
                            "count": count,
                        }
                        for signature, count in action_counts.most_common()
                    ],
                }
            )

    return {
        "first_action_tool_counts": dict(sorted(all_tools.items())),
        "tasks_with_first_action": len(modal_exact_shares),
        "unanimous_first_tool_tasks": unanimous_tool_tasks,
        "unanimous_schema_normalized_first_action_tasks": (
            unanimous_schema_normalized_action_tasks
        ),
        "unanimous_authored_first_action_tasks": unanimous_authored_action_tasks,
        "all_first_tools_describe_table_tasks": all_describe_tasks,
        "same_tool_but_different_arguments_tasks": same_tool_different_arguments_tasks,
        "unique_first_tools_per_task_histogram": {
            str(key): value for key, value in sorted(unique_tool_histogram.items())
        },
        "unique_schema_normalized_first_actions_per_task_histogram": {
            str(key): value for key, value in sorted(unique_action_histogram.items())
        },
        "unique_authored_first_actions_per_task_histogram": {
            str(key): value
            for key, value in sorted(unique_authored_action_histogram.items())
        },
        "mean_modal_schema_normalized_first_action_share": (
            statistics.fmean(modal_exact_shares) if modal_exact_shares else 0.0
        ),
        "variation_examples": variation_examples,
    }


def _prefix_summary(episodes_by_task: dict[str, list[Episode]]) -> dict[str, Any]:
    tool_lcp: list[int] = []
    exact_lcp: list[int] = []
    authored_lcp: list[int] = []
    execution_lcp: list[int] = []
    for episodes in episodes_by_task.values():
        tool_sequences = [[event.tool for event in episode.events] for episode in episodes]
        exact_sequences = [
            [event.action_signature for event in episode.events] for episode in episodes
        ]
        authored_sequences = [
            [event.authored_action_signature for event in episode.events]
            for episode in episodes
        ]
        execution_sequences = [
            [
                _stable_json(
                    [event.authored_action_signature, event.observation_signature]
                )
                for event in episode.events
            ]
            for episode in episodes
        ]
        tool_lcp.append(_longest_common_prefix(tool_sequences))
        exact_lcp.append(_longest_common_prefix(exact_sequences))
        authored_lcp.append(_longest_common_prefix(authored_sequences))
        execution_lcp.append(_longest_common_prefix(execution_sequences))
    return {
        "all_trajectories_tool_name_lcp_histogram": _histogram(tool_lcp),
        "all_trajectories_schema_normalized_action_lcp_histogram": _histogram(exact_lcp),
        "all_trajectories_authored_action_lcp_histogram": _histogram(authored_lcp),
        "all_trajectories_exact_execution_lcp_histogram": _histogram(execution_lcp),
        "tasks_with_schema_normalized_action_lcp_at_least_1": sum(
            value >= 1 for value in exact_lcp
        ),
        "tasks_with_schema_normalized_action_lcp_at_least_2": sum(
            value >= 2 for value in exact_lcp
        ),
    }


def _anchor_summary(
    episodes_by_task: dict[str, list[Episode]],
    *,
    representation: str,
    mixed_task_ids: set[str],
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[DecisionEvent]] = defaultdict(list)
    all_events: list[DecisionEvent] = []
    for task_id, episodes in episodes_by_task.items():
        for episode in episodes:
            for event in episode.events:
                all_events.append(event)
                groups[(task_id, event.state_key(representation))].append(event)

    counts = Counter()
    covered_events = Counter()
    tasks_by_kind: dict[str, set[str]] = defaultdict(set)
    depth_histograms: dict[str, Counter[int]] = defaultdict(Counter)
    branch_structure_counts: Counter[str] = Counter()
    branch_structure_tasks: dict[str, set[str]] = defaultdict(set)
    branch_structure_events: Counter[str] = Counter()
    ambiguity_counts: Counter[str] = Counter()
    ambiguity_tasks: dict[str, set[str]] = defaultdict(set)
    ambiguity_events: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []

    for (task_id, state_key), events in groups.items():
        if len(events) < 2:
            continue
        depth_values = {event.depth for event in events}
        depth = min(depth_values)
        noninitial = depth > 0
        action_groups: dict[str, list[DecisionEvent]] = defaultdict(list)
        for event in events:
            action_groups[event.action_signature].append(event)
        outcomes = {event.correct for event in events}
        repeated = True
        branching = len(action_groups) >= 2
        mixed = outcomes == {False, True}
        informative = branching and mixed
        supported_actions = {
            action: action_events
            for action, action_events in action_groups.items()
            if len(action_events) >= 2
        }
        supported_rates = {
            action: sum(event.correct for event in action_events) / len(action_events)
            for action, action_events in supported_actions.items()
        }
        supported_contrast = (
            len(supported_rates) >= 2
            and max(supported_rates.values()) > min(supported_rates.values())
        )
        mixed_action_groups = [
            action_events
            for action_events in action_groups.values()
            if len(action_events) >= 2
            and {event.correct for event in action_events} == {False, True}
        ]
        if mixed_action_groups:
            ambiguity_counts["states_with_mixed_state_action"] += 1
            ambiguity_tasks["states_with_mixed_state_action"].add(task_id)
            ambiguity_events["states_with_mixed_state_action"] += sum(
                len(action_events) for action_events in mixed_action_groups
            )
            ambiguity_counts["mixed_state_action_groups"] += len(
                mixed_action_groups
            )
            ambiguity_tasks["mixed_state_action_groups"].add(task_id)
            ambiguity_events["mixed_state_action_groups"] += sum(
                len(action_events) for action_events in mixed_action_groups
            )
            if noninitial:
                ambiguity_counts["noninitial_states_with_mixed_state_action"] += 1
                ambiguity_tasks["noninitial_states_with_mixed_state_action"].add(
                    task_id
                )
                ambiguity_events["noninitial_states_with_mixed_state_action"] += sum(
                    len(action_events) for action_events in mixed_action_groups
                )
                ambiguity_counts["noninitial_mixed_state_action_groups"] += len(
                    mixed_action_groups
                )
                ambiguity_tasks["noninitial_mixed_state_action_groups"].add(task_id)
                ambiguity_events["noninitial_mixed_state_action_groups"] += sum(
                    len(action_events) for action_events in mixed_action_groups
                )
        if mixed and not branching:
            ambiguity_counts["single_action_mixed_outcome_states"] += 1
            ambiguity_tasks["single_action_mixed_outcome_states"].add(task_id)
            ambiguity_events["single_action_mixed_outcome_states"] += len(events)
            if noninitial:
                ambiguity_counts["noninitial_single_action_mixed_outcome_states"] += 1
                ambiguity_tasks[
                    "noninitial_single_action_mixed_outcome_states"
                ].add(task_id)
                ambiguity_events[
                    "noninitial_single_action_mixed_outcome_states"
                ] += len(events)
        action_tools = {
            json.loads(action)["tool"] for action in action_groups
        }
        return_columns_ablated_actions = {
            _return_columns_ablated_signature(action) for action in action_groups
        }
        branch_structure_flags = {
            "same_tool_branching": branching and len(action_tools) == 1,
            "cross_tool_branching": branching and len(action_tools) >= 2,
            "return_columns_only_branching": (
                branching and len(return_columns_ablated_actions) == 1
            ),
            "return_columns_sensitive_branching": (
                branching
                and len(return_columns_ablated_actions) < len(action_groups)
            ),
        }
        for kind, active in branch_structure_flags.items():
            if not active:
                continue
            branch_structure_counts[kind] += 1
            branch_structure_tasks[kind].add(task_id)
            branch_structure_events[kind] += len(events)
            if noninitial:
                branch_structure_counts[f"noninitial_{kind}"] += 1
                branch_structure_tasks[f"noninitial_{kind}"].add(task_id)
                branch_structure_events[f"noninitial_{kind}"] += len(events)

        flags = {
            "repeated": repeated,
            "branching": branching,
            "mixed_outcome": mixed,
            "informative": informative,
            "supported_contrast": supported_contrast,
        }
        for kind, active in flags.items():
            if not active:
                continue
            counts[kind] += 1
            covered_events[kind] += len(events)
            tasks_by_kind[kind].add(task_id)
            depth_histograms[kind][depth] += 1
            if noninitial:
                counts[f"noninitial_{kind}"] += 1
                covered_events[f"noninitial_{kind}"] += len(events)
                tasks_by_kind[f"noninitial_{kind}"].add(task_id)
                depth_histograms[f"noninitial_{kind}"][depth] += 1

        if informative:
            action_stats = []
            for action, action_events in sorted(
                action_groups.items(),
                key=lambda item: (-len(item[1]), item[0]),
            ):
                successes = sum(event.correct for event in action_events)
                action_stats.append(
                    {
                        "action": action,
                        "count": len(action_events),
                        "successes": successes,
                        "success_rate": successes / len(action_events),
                    }
                )
            candidates.append(
                {
                    "task_id": task_id,
                    "state_hash": state_key,
                    "depth": depth,
                    "noninitial": noninitial,
                    "trajectories": len(events),
                    "actions": action_stats,
                    "supported_contrast": supported_contrast,
                    "unique_tools": len(action_tools),
                    "return_columns_ablated_actions": len(
                        return_columns_ablated_actions
                    ),
                    "return_columns_only_branching": (
                        branch_structure_flags["return_columns_only_branching"]
                    ),
                }
            )

    event_total = len(all_events)
    mixed_task_event_total = sum(
        event.task_id in mixed_task_ids for event in all_events
    )
    return {
        "representation": representation,
        "decision_events": event_total,
        "state_groups": len(groups),
        "counts": dict(sorted(counts.items())),
        "task_counts": {
            key: len(value) for key, value in sorted(tasks_by_kind.items())
        },
        "covered_event_counts": dict(sorted(covered_events.items())),
        "covered_event_fractions": {
            key: (value / event_total if event_total else 0.0)
            for key, value in sorted(covered_events.items())
        },
        "mixed_task_decision_events": mixed_task_event_total,
        "covered_mixed_task_event_fractions": {
            key: (value / mixed_task_event_total if mixed_task_event_total else 0.0)
            for key, value in sorted(covered_events.items())
            if key in {"informative", "noninitial_informative", "supported_contrast", "noninitial_supported_contrast"}
        },
        "branch_structure": {
            "contract": (
                "return_columns ablation is a sensitivity label only; it is not "
                "used to merge actions or assert semantic equivalence"
            ),
            "counts": dict(sorted(branch_structure_counts.items())),
            "task_counts": {
                key: len(value)
                for key, value in sorted(branch_structure_tasks.items())
            },
            "covered_event_counts": dict(sorted(branch_structure_events.items())),
        },
        "credit_ambiguity": {
            "definition": (
                "same state and same parsed full action have both correct and "
                "incorrect terminal descendants"
            ),
            "counts": dict(sorted(ambiguity_counts.items())),
            "task_counts": {
                key: len(value)
                for key, value in sorted(ambiguity_tasks.items())
            },
            "covered_event_counts": dict(sorted(ambiguity_events.items())),
            "covered_event_fractions": {
                key: (value / event_total if event_total else 0.0)
                for key, value in sorted(ambiguity_events.items())
            },
        },
        "depth_histograms": {
            key: _histogram(histogram.elements())
            for key, histogram in sorted(depth_histograms.items())
        },
        "informative_candidates": sorted(
            candidates,
            key=lambda item: (
                not item["noninitial"],
                not item["supported_contrast"],
                -item["trajectories"],
                item["task_id"],
                item["depth"],
            ),
        ),
    }


def _mean_losses(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "state_brier": 0.0,
            "action_brier": 0.0,
            "action_minus_state_brier": 0.0,
            "state_log_loss": 0.0,
            "action_log_loss": 0.0,
            "action_minus_state_log_loss": 0.0,
        }
    result = {}
    for metric in ("brier", "log_loss"):
        state_mean = statistics.fmean(record[f"state_{metric}"] for record in records)
        action_mean = statistics.fmean(record[f"action_{metric}"] for record in records)
        result[f"state_{metric}"] = state_mean
        result[f"action_{metric}"] = action_mean
        result[f"action_minus_state_{metric}"] = action_mean - state_mean
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _task_cluster_bootstrap(
    records: Sequence[dict[str, Any]],
    *,
    replicates: int = 5000,
    seed: int = 20260828,
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task_id"]].append(record)
    task_ids = sorted(by_task)
    if not task_ids:
        return {"replicates": 0, "seed": seed, "task_clusters": 0}
    rng = random.Random(seed)
    deltas: dict[str, list[float]] = {
        "brier": [],
        "log_loss": [],
    }
    for _ in range(replicates):
        sampled_records: list[dict[str, Any]] = []
        for _task_index in task_ids:
            sampled_task = task_ids[rng.randrange(len(task_ids))]
            sampled_records.extend(by_task[sampled_task])
        losses = _mean_losses(sampled_records)
        for metric in deltas:
            deltas[metric].append(losses[f"action_minus_state_{metric}"])
    return {
        "replicates": replicates,
        "seed": seed,
        "task_clusters": len(task_ids),
        "action_minus_state_95pct_ci": {
            metric: [
                _percentile(values, 0.025),
                _percentile(values, 0.975),
            ]
            for metric, values in deltas.items()
        },
    }


def _leave_one_out_action_value(
    episodes_by_task: dict[str, list[Episode]],
    *,
    representation: str,
    action_identity: str = "full_action",
) -> dict[str, Any]:
    """Test whether repeated full actions predict continuation outcome beyond state mean.

    Each outcome is removed from both estimates.  A Beta(1, 1) prior prevents singleton
    remainder estimates from becoming 0/1.  The test is deliberately restricted to noninitial,
    branching, mixed-outcome states and to events whose complete action occurs at least twice.
    """
    groups: dict[tuple[str, str], list[DecisionEvent]] = defaultdict(list)
    for task_id, episodes in episodes_by_task.items():
        for episode in episodes:
            for event in episode.events:
                groups[(task_id, event.state_key(representation))].append(event)

    records: list[dict[str, Any]] = []
    state_count = 0
    for (task_id, state_key), events in groups.items():
        if min(event.depth for event in events) == 0:
            continue
        full_action_groups: dict[str, list[DecisionEvent]] = defaultdict(list)
        for event in events:
            full_action_groups[event.action_signature].append(event)
        if (
            len(full_action_groups) < 2
            or {event.correct for event in events} != {False, True}
        ):
            continue
        supported_events = [
            event
            for event in events
            if len(full_action_groups[event.action_signature]) >= 2
        ]
        if not supported_events:
            continue
        action_groups: dict[str, list[DecisionEvent]] = defaultdict(list)
        for event in events:
            if action_identity == "full_action":
                value_key = event.action_signature
            elif action_identity == "return_columns_ablated":
                value_key = _return_columns_ablated_signature(event.action_signature)
            elif action_identity == "tool_only":
                value_key = json.loads(event.action_signature)["tool"]
            else:
                raise ValueError(f"unknown action identity: {action_identity}")
            action_groups[value_key].append(event)
        state_count += 1
        state_successes = sum(event.correct for event in events)
        for event in supported_events:
            if action_identity == "full_action":
                value_key = event.action_signature
            elif action_identity == "return_columns_ablated":
                value_key = _return_columns_ablated_signature(event.action_signature)
            else:
                value_key = json.loads(event.action_signature)["tool"]
            action_events = action_groups[value_key]
            action_successes = sum(candidate.correct for candidate in action_events)
            outcome = float(event.correct)
            state_other_count = len(events) - 1
            action_other_count = len(action_events) - 1
            state_probability = (
                state_successes - outcome + 1.0
            ) / (state_other_count + 2.0)
            action_probability = (
                action_successes - outcome + 1.0
            ) / (action_other_count + 2.0)
            records.append(
                {
                    "task_id": task_id,
                    "state_hash": state_key,
                    "outcome": outcome,
                    "state_brier": (state_probability - outcome) ** 2,
                    "action_brier": (action_probability - outcome) ** 2,
                    "state_log_loss": -(
                        outcome * math.log(state_probability)
                        + (1.0 - outcome) * math.log(1.0 - state_probability)
                    ),
                    "action_log_loss": -(
                        outcome * math.log(action_probability)
                        + (1.0 - outcome) * math.log(1.0 - action_probability)
                    ),
                }
            )

    losses = _mean_losses(records)
    return {
        "representation": representation,
        "contract": {
            "target": "terminal binary correctness",
            "gold_sql_read": False,
            "selection_warning": (
                "exploratory only: the source groups were selected to have mixed terminal outcomes"
            ),
            "state_filter": "noninitial, branching, mixed-outcome",
            "prediction_case_filter": "parsed full action count >= 2 within state",
            "action_value_identity": action_identity,
            "held_out_outcome_removed": True,
            "smoothing": "Beta(1,1)",
            "negative_delta_means_action_conditioning_is_better": True,
        },
        "eligible_states": state_count,
        "prediction_cases": len(records),
        "task_clusters": len({record["task_id"] for record in records}),
        "losses": losses,
        "task_cluster_bootstrap": _task_cluster_bootstrap(records),
    }


def _trajectory_token_ambiguity_mass(
    episodes_by_task: dict[str, list[Episode]],
    *,
    representation: str,
) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[DecisionEvent]] = defaultdict(list)
    for task_id, episodes in episodes_by_task.items():
        for episode in episodes:
            for event in episode.events:
                groups[
                    (task_id, event.state_key(representation), event.action_signature)
                ].append(event)
    ambiguous_event_keys = {
        (event.episode_id, event.depth)
        for events in groups.values()
        if len(events) >= 2 and {event.correct for event in events} == {False, True}
        for event in events
    }

    total_mass = 0.0
    ambiguous_mass = 0.0
    total_response_tokens = 0
    ambiguous_response_tokens = 0
    known_episodes = 0
    skipped_episodes = 0
    contributing_events = 0
    ambiguous_contributing_events = 0
    trajectory_counts: Counter[str] = Counter()
    for episodes in episodes_by_task.values():
        outcomes = [episode.correct for episode in episodes]
        if set(outcomes) != {False, True}:
            continue
        success_fraction = sum(outcomes) / len(outcomes)
        standard_deviation = math.sqrt(
            success_fraction * (1.0 - success_fraction)
        )
        for episode in episodes:
            outcome_label = "correct" if episode.correct else "incorrect"
            trajectory_counts[f"{outcome_label}_trajectories"] += 1
            ambiguous_flags = [
                (event.episode_id, event.depth) in ambiguous_event_keys
                for event in episode.events
            ]
            if any(ambiguous_flags):
                trajectory_counts[f"{outcome_label}_with_ambiguous_turn"] += 1
            if ambiguous_flags and all(ambiguous_flags):
                trajectory_counts[f"{outcome_label}_all_turns_ambiguous"] += 1
            if any(not flag for flag in ambiguous_flags):
                trajectory_counts[f"{outcome_label}_retaining_unmasked_turn"] += 1
            token_counts = [event.response_token_count for event in episode.events]
            if not token_counts or any(count is None for count in token_counts):
                skipped_episodes += 1
                continue
            known_episodes += 1
            trajectory_tokens = sum(int(count) for count in token_counts)
            if trajectory_tokens <= 0:
                skipped_episodes += 1
                continue
            advantage = (
                (1.0 - success_fraction) / standard_deviation
                if episode.correct
                else -success_fraction / standard_deviation
            )
            for event in episode.events:
                response_tokens = int(event.response_token_count or 0)
                coefficient_mass = abs(advantage) * response_tokens / trajectory_tokens
                total_mass += coefficient_mass
                total_response_tokens += response_tokens
                contributing_events += 1
                if (event.episode_id, event.depth) in ambiguous_event_keys:
                    ambiguous_mass += coefficient_mass
                    ambiguous_response_tokens += response_tokens
                    ambiguous_contributing_events += 1
    return {
        "representation": representation,
        "contract": {
            "policy_reduction": "trajectory_token_mean",
            "quantity": (
                "absolute standardized-advantage times stored response-token share; "
                "the common batch mean-turn scale cancels in the reported fraction"
            ),
            "not_a_gradient_norm": True,
        },
        "known_token_count_episodes": known_episodes,
        "skipped_missing_token_count_episodes": skipped_episodes,
        "contributing_events": contributing_events,
        "ambiguous_contributing_events": ambiguous_contributing_events,
        "absolute_coefficient_mass": total_mass,
        "ambiguous_absolute_coefficient_mass": ambiguous_mass,
        "ambiguous_absolute_coefficient_mass_fraction": (
            ambiguous_mass / total_mass if total_mass else 0.0
        ),
        "response_tokens": total_response_tokens,
        "ambiguous_response_tokens": ambiguous_response_tokens,
        "ambiguous_response_token_fraction": (
            ambiguous_response_tokens / total_response_tokens
            if total_response_tokens
            else 0.0
        ),
        "trajectory_counts": dict(sorted(trajectory_counts.items())),
    }


def audit_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    episodes: list[Episode] = []
    issues: list[str] = []
    splits: Counter[str] = Counter()
    protocol_versions: Counter[str] = Counter()
    protocol_hashes: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        adapted_row = _adapt_flat_training_rollout(row)
        environment = adapted_row.get("environment")
        if isinstance(environment, dict):
            split = environment.get("dataset_split")
            if isinstance(split, str):
                splits[split] += 1
        sample = adapted_row.get("sample")
        audit_record = sample.get("audit_record") if isinstance(sample, dict) else None
        if isinstance(audit_record, dict):
            protocol_versions[str(audit_record.get("protocol_version"))] += 1
            protocol_hashes[str(audit_record.get("protocol_hash"))] += 1
        episode, row_issues = _parse_episode(adapted_row, row_index=row_index)
        issues.extend(row_issues)
        if episode is not None:
            episodes.append(episode)

    eligible_episodes = [episode for episode in episodes if episode.eligible]
    episodes_by_task: dict[str, list[Episode]] = defaultdict(list)
    for episode in eligible_episodes:
        episodes_by_task[episode.task_id].append(episode)
    for task_episodes in episodes_by_task.values():
        task_episodes.sort(key=lambda episode: episode.sample_index)

    decision_events = sum(len(episode.events) for episode in eligible_episodes)
    mixed_task_ids = {
        task_id
        for task_id, task_episodes in episodes_by_task.items()
        if {episode.correct for episode in task_episodes} == {False, True}
    }
    mixed_task_decision_events = sum(
        len(episode.events)
        for task_id, task_episodes in episodes_by_task.items()
        if task_id in mixed_task_ids
        for episode in task_episodes
    )
    first_action = _first_action_summary(episodes_by_task)
    prefix = _prefix_summary(episodes_by_task)
    anchors = {
        representation: _anchor_summary(
            episodes_by_task,
            representation=representation,
            mixed_task_ids=mixed_task_ids,
        )
        for representation in STATE_REPRESENTATIONS
    }
    predictive_validation = _leave_one_out_action_value(
        episodes_by_task,
        representation="strict_policy_prompt",
    )
    predictive_validation["coarsening_sensitivity"] = {
        action_identity: _leave_one_out_action_value(
            episodes_by_task,
            representation="strict_policy_prompt",
            action_identity=action_identity,
        )
        for action_identity in ("return_columns_ablated", "tool_only")
    }
    predictive_validation["environment_state_sensitivity"] = (
        _leave_one_out_action_value(
            episodes_by_task,
            representation="exact_environment_prefix",
        )
    )
    ambiguity_weighting = {
        representation: _trajectory_token_ambiguity_mass(
            episodes_by_task,
            representation=representation,
        )
        for representation in ("exact_environment_prefix", "strict_policy_prompt")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "gold_sql_read": False,
            "gold_answer_read": False,
            "model_calls": 0,
            "optimizer_updates": 0,
            "next_action_identity": "parsed-tool-plus-full-arguments-v1",
            "json_object_key_order": "ignored",
            "schema_normalization": {
                "describe_table.tables": "sorted-as-proven-unordered",
                "all_other_list_fields": "order-preserved",
            },
            "primary_training_eligibility": "sample.process_update-is-true",
            "strict_policy_state_source": (
                "stored prompt_ids when available; otherwise exact stored model_input "
                "messages as a conservative equality key"
            ),
        },
        "identity": {
            "dataset_split_counts": dict(sorted(splits.items())),
            "protocol_version_counts": dict(sorted(protocol_versions.items())),
            "protocol_hash_counts": dict(sorted(protocol_hashes.items())),
        },
        "observed": {
            "rows": len(rows),
            "parsed_episodes": len(episodes),
            "eligible_episodes": len(eligible_episodes),
            "eligible_tasks": len(episodes_by_task),
            "eligible_correct": sum(episode.correct for episode in eligible_episodes),
            "eligible_incorrect": sum(not episode.correct for episode in eligible_episodes),
            "eligible_decision_events": decision_events,
            "eligible_mixed_tasks": len(mixed_task_ids),
            "eligible_mixed_task_decision_events": mixed_task_decision_events,
            "mean_decision_turns": (
                decision_events / len(eligible_episodes) if eligible_episodes else 0.0
            ),
            "issues": len(issues),
        },
        "first_action": first_action,
        "common_prefix": prefix,
        "state_anchors": anchors,
        "predictive_validation": predictive_validation,
        "ambiguity_weighting": ambiguity_weighting,
        "issues": issues,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.suffix == ".gz":
        source = gzip.open(path, "rt", encoding="utf-8")
    else:
        source = path.open("r", encoding="utf-8")
    with source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def load_group_dirs(
    paths: Sequence[Path],
    *,
    only_clean_mixed_groups: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    source_files = 0
    duplicate_groups = 0
    rejected_non_k8 = 0
    rejected_unclean = 0
    rejected_homogeneous = 0
    for path in paths:
        if not path.is_dir():
            raise ValueError(f"group directory does not exist: {path}")
        for group_path in sorted(path.glob("*.json")):
            source_files += 1
            value = json.loads(group_path.read_text(encoding="utf-8"))
            if not isinstance(value, list) or not all(
                isinstance(row, dict) for row in value
            ):
                raise ValueError(f"group file must contain an object array: {group_path}")
            if len(value) != 8:
                rejected_non_k8 += 1
                if only_clean_mixed_groups:
                    continue
            task_ids = {
                (row.get("environment") or {}).get("task_id") for row in value
            }
            if len(task_ids) != 1 or not isinstance(next(iter(task_ids)), str):
                raise ValueError(f"group has inconsistent task identity: {group_path}")
            task_id = next(iter(task_ids))
            clean = all(
                (row.get("sample") or {}).get("process_update") is True
                for row in value
            )
            outcomes = {
                (row.get("sample") or {}).get("correct") for row in value
            }
            if only_clean_mixed_groups and not clean:
                rejected_unclean += 1
                continue
            if only_clean_mixed_groups and outcomes != {False, True}:
                rejected_homogeneous += 1
                continue
            if task_id in groups:
                if groups[task_id] != value:
                    raise ValueError(f"non-identical duplicate group: {task_id}")
                duplicate_groups += 1
                continue
            groups[task_id] = value
    rows = [row for task_id in sorted(groups) for row in groups[task_id]]
    return rows, {
        "source_group_directories": [str(path) for path in paths],
        "source_group_files": source_files,
        "selected_groups": len(groups),
        "selected_rows": len(rows),
        "duplicate_groups": duplicate_groups,
        "rejected_non_k8": rejected_non_k8,
        "rejected_unclean": rejected_unclean,
        "rejected_homogeneous": rejected_homogeneous,
        "only_clean_mixed_groups": only_clean_mixed_groups,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trajectories", type=Path)
    source.add_argument("--group-dir", type=Path, action="append")
    parser.add_argument("--only-clean-mixed-groups", action="store_true")
    parser.add_argument("--selected-jsonl-gz-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.trajectories is not None:
        if args.only_clean_mixed_groups:
            raise ValueError("--only-clean-mixed-groups requires --group-dir")
        rows = load_jsonl(args.trajectories)
        source_selection = {
            "trajectories": str(args.trajectories),
            "selected_rows": len(rows),
            "only_clean_mixed_groups": False,
        }
    else:
        rows, source_selection = load_group_dirs(
            args.group_dir,
            only_clean_mixed_groups=args.only_clean_mixed_groups,
        )
    if args.selected_jsonl_gz_output is not None:
        args.selected_jsonl_gz_output.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.selected_jsonl_gz_output, "wt", encoding="utf-8") as target:
            for row in rows:
                target.write(_stable_json(row) + "\n")
    result = audit_rows(rows)
    result["source_selection"] = source_selection
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["observed"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
