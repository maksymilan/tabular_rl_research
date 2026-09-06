#!/usr/bin/env python3
"""Canonical audit for online result-only GRPO training artifacts.

The analyzer is intentionally read-only.  It reconstructs the exact population-
standardized group advantages from persisted rollout rewards, binds completed
optimizer updates to their Trainer checkpoints, and reports any partially collected
next update separately.  It does not load the model or infer learning from ``loss``
alone; LoRA movement remains owned by ``compare_lora_updates.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from rl.diagnostics.io import read_json as _diagnostic_read_json
from rl.diagnostics.io import read_jsonl as _diagnostic_read_jsonl
from rl.diagnostics.io import sha256_file as _diagnostic_sha256_file

try:
    import torch
except ImportError:  # The production TRL runtime has Torch; local audits may not.
    torch = None

try:
    from safetensors import safe_open
except ImportError:  # Kept optional for dependency-light local artifact inspection.
    safe_open = None


SCHEMA_VERSION = "grpo-training-audit-v1"
# Exhaustive Torch-FP32 enumeration of the 42 heterogeneous K=8 reward-count
# compositions on {0, 0.2, 1} has a maximum normalized-group sum residual of
# 1.0728836059570312e-6.  This check is about credit direction, not float64
# re-centering, so retain a narrow margin above the production arithmetic while
# still failing closed on a materially non-zero group mean.
GROUP_ADVANTAGE_SUM_ABS_TOLERANCE = 2e-6


def load_json(path: Path) -> dict[str, Any]:
    return dict(_diagnostic_read_json(path, require_object=True))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in _diagnostic_read_jsonl(path)]


def sha256_file(path: Path) -> str:
    return _diagnostic_sha256_file(path)


def checkpoint_precision_audit(checkpoint_dir: Path) -> dict[str, Any]:
    """Audit persisted LoRA tensors and Adam moments without loading the base model."""
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    optimizer_path = checkpoint_dir / "optimizer.pt"
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_dir),
        "available": False,
        "adapter_tensor_count": 0,
        "adapter_dtype_counts": {},
        "non_fp32_adapter_tensors": [],
        "optimizer_state_entries": 0,
        "optimizer_tensor_count": 0,
        "optimizer_tensor_dtype_counts": {},
        "adam_moment_tensors": 0,
        "adam_moment_dtype_counts": {},
        "non_fp32_adam_moments": [],
        "passes": False,
    }
    if torch is None or safe_open is None:
        result["unavailable_reason"] = "torch_or_safetensors_unavailable"
        return result
    missing = [
        str(path.name)
        for path in (adapter_path, optimizer_path)
        if not path.is_file()
    ]
    if missing:
        result["unavailable_reason"] = "missing_checkpoint_artifacts"
        result["missing"] = missing
        return result

    adapter_dtypes: Counter[str] = Counter()
    non_fp32_adapter_tensors = []
    with safe_open(
        adapter_path, framework="pt", device="cpu"
    ) as adapter_source:
        adapter_keys = list(adapter_source.keys())
        for key in adapter_keys:
            dtype = str(adapter_source.get_tensor(key).dtype)
            adapter_dtypes[dtype] += 1
            if dtype != "torch.float32":
                non_fp32_adapter_tensors.append({"name": key, "dtype": dtype})

    optimizer = torch.load(
        optimizer_path, map_location="cpu", weights_only=True
    )
    states = optimizer.get("state") or {}
    optimizer_dtypes: Counter[str] = Counter()
    moment_dtypes: Counter[str] = Counter()
    non_fp32_moments = []
    optimizer_tensor_count = 0
    moment_tensors = 0
    for parameter_id, state in states.items():
        for state_name, value in state.items():
            if not torch.is_tensor(value):
                continue
            optimizer_tensor_count += 1
            dtype = str(value.dtype)
            optimizer_dtypes[dtype] += 1
            if state_name not in {"exp_avg", "exp_avg_sq"}:
                continue
            moment_tensors += 1
            moment_dtypes[dtype] += 1
            if dtype != "torch.float32":
                non_fp32_moments.append(
                    {
                        "parameter_id": str(parameter_id),
                        "state": state_name,
                        "dtype": dtype,
                    }
                )

    result.update(
        {
            "available": True,
            "adapter_tensor_count": len(adapter_keys),
            "adapter_dtype_counts": dict(sorted(adapter_dtypes.items())),
            "non_fp32_adapter_tensors": non_fp32_adapter_tensors,
            "optimizer_state_entries": len(states),
            "optimizer_tensor_count": optimizer_tensor_count,
            "optimizer_tensor_dtype_counts": dict(
                sorted(optimizer_dtypes.items())
            ),
            "adam_moment_tensors": moment_tensors,
            "adam_moment_dtype_counts": dict(sorted(moment_dtypes.items())),
            "non_fp32_adam_moments": non_fp32_moments,
            "passes": (
                bool(adapter_keys)
                and not non_fp32_adapter_tensors
                and moment_tensors > 0
                and not non_fp32_moments
            ),
        }
    )
    return result


def implementation_provenance(
    run_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Verify the immutable source snapshot used by a long-running trainer."""
    lock_path = run_dir / "implementation_lock.json"
    lock = load_json(lock_path) if lock_path.is_file() else None
    manifest_hashes = manifest.get("implementation_source_sha256") or {}
    lock_hashes = (lock or {}).get("files") or {}
    expected_hashes = lock_hashes or manifest_hashes
    snapshot_root = run_dir / "implementation_source_snapshot"
    missing_files = []
    hash_mismatches = {}
    for relative_path, expected_hash in sorted(expected_hashes.items()):
        path = snapshot_root / relative_path
        if not path.is_file():
            missing_files.append(relative_path)
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            hash_mismatches[relative_path] = {
                "expected": expected_hash,
                "actual": actual_hash,
            }
    lock_manifest_overlap = sorted(set(lock_hashes) & set(manifest_hashes))
    lock_manifest_mismatches = {
        path: {"lock": lock_hashes[path], "manifest": manifest_hashes[path]}
        for path in lock_manifest_overlap
        if lock_hashes[path] != manifest_hashes[path]
    }
    source_snapshot_matches = bool(expected_hashes) and not (
        missing_files or hash_mismatches or lock_manifest_mismatches
    )
    return {
        "implementation_lock_present": lock is not None,
        "expected_hash_source": (
            "implementation_lock"
            if lock_hashes
            else "run_manifest"
            if manifest_hashes
            else None
        ),
        "expected_file_count": len(expected_hashes),
        "snapshot_present": snapshot_root.is_dir(),
        "missing_files": missing_files,
        "hash_mismatches": hash_mismatches,
        "lock_manifest_mismatches": lock_manifest_mismatches,
        "source_snapshot_matches": source_snapshot_matches,
        "initial_adapter_sha256": (
            (lock or {}).get("initial_adapter_sha256")
            or manifest.get("initial_adapter_sha256")
        ),
    }


def standardized_group_advantages(
    rewards: Sequence[float],
    eligible: Sequence[bool],
    *,
    epsilon: float = 1e-6,
) -> list[float]:
    """Match ``transition_batch.standardized_group_advantages`` exactly."""
    if len(rewards) != len(eligible):
        raise ValueError("reward and eligibility vectors must have equal length")
    kept = [index for index, value in enumerate(eligible) if value]
    result = [0.0] * len(rewards)
    if not kept:
        return result
    values = [float(rewards[index]) for index in kept]
    if all(value == values[0] for value in values[1:]):
        return result
    if torch is not None:
        reward_tensor = torch.tensor(values, dtype=torch.float32)
        std_tensor = reward_tensor.std(unbiased=False)
        if float(std_tensor) == 0.0:
            return result
        normalized = (
            reward_tensor - reward_tensor.mean()
        ) / (std_tensor + epsilon)
        for source_index, value in zip(kept, normalized.tolist(), strict=True):
            result[source_index] = float(value)
        return result
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0.0:
        return result
    for index in kept:
        result[index] = (float(rewards[index]) - mean) / (std + epsilon)
    return result


def reward_value(row: dict[str, Any]) -> float:
    reward = row.get("result_reward") or {}
    if "value" not in reward:
        raise ValueError(
            f"rollout {row.get('trajectory_id')} has no result_reward.value"
        )
    return float(reward["value"])


def optimization_eligible(row: dict[str, Any]) -> bool:
    return row.get("optimization_exclusion") is None


def is_timeout_event(event: dict[str, Any]) -> bool:
    return (
        event.get("failure_type") == "timeout_error"
        or event.get("error_type") == "timeout_error"
        or event.get("error_code") == "tool_execution_timeout"
    )


def timeout_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in (row.get("error_events") or [])
        if is_timeout_event(event)
    ]


def timeout_observed(row: dict[str, Any]) -> bool:
    if row.get("failure_type") == "timeout_error":
        return True
    return bool(timeout_events(row))


def _counter(values: Sequence[Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(
            Counter(values).items(), key=lambda item: str(item[0])
        )
    }


def _groups(
    rows: Sequence[dict[str, Any]], group_size: int
) -> tuple[list[list[dict[str, Any]]], int]:
    complete = len(rows) // group_size
    groups = [
        list(rows[offset : offset + group_size])
        for offset in range(0, complete * group_size, group_size)
    ]
    for position, group in enumerate(groups):
        indices = {int(row["example_index"]) for row in group}
        if len(indices) != 1:
            raise ValueError(
                f"GRPO group {position} mixes example indices: {sorted(indices)}"
            )
    return groups, len(rows) - complete * group_size


def trajectory_action_signature(row: dict[str, Any]) -> str:
    actions = []
    for turn in row.get("turns") or []:
        parsed = turn.get("parsed") or {}
        tool = parsed.get("tool")
        if tool is None:
            continue
        actions.append({"tool": str(tool), "arguments": parsed.get("arguments")})
    payload = json.dumps(
        actions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def summarize_policy_behavior(
    rows: Sequence[dict[str, Any]], rows_per_update: int
) -> dict[str, Any]:
    by_step: dict[int, list[dict[str, Any]]] = {}
    explicit = 0
    for row_index, row in enumerate(rows):
        raw_step = row.get("policy_global_step")
        if raw_step is not None:
            explicit += 1
        step = int(raw_step) if raw_step is not None else row_index // rows_per_update
        by_step.setdefault(step, []).append(row)

    step_summaries = []
    task_step_rows: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for step, step_rows in sorted(by_step.items()):
        signatures = [trajectory_action_signature(row) for row in step_rows]
        tools = [
            str((turn.get("parsed") or {}).get("tool"))
            for row in step_rows
            for turn in (row.get("turns") or [])
            if (turn.get("parsed") or {}).get("tool") is not None
        ]
        rewards = [reward_value(row) for row in step_rows]
        step_summaries.append(
            {
                "policy_global_step": step,
                "rows": len(step_rows),
                "tasks": len({int(row["example_index"]) for row in step_rows}),
                "correct_rate": sum(bool(row.get("correct")) for row in step_rows)
                / len(step_rows),
                "legal_rate": sum(bool(row.get("legal")) for row in step_rows)
                / len(step_rows),
                "reward_mean": statistics.fmean(rewards),
                "unique_action_signatures": len(set(signatures)),
                "action_signature_diversity": len(set(signatures)) / len(signatures),
                "tool_counts": _counter(tools),
            }
        )
        for row in step_rows:
            task_step_rows.setdefault(int(row["example_index"]), {}).setdefault(
                step, []
            ).append(row)

    repeated = []
    for example_index, occurrences in sorted(task_step_rows.items()):
        if len(occurrences) < 2:
            continue
        first_step = min(occurrences)
        last_step = max(occurrences)
        first_rows = occurrences[first_step]
        last_rows = occurrences[last_step]
        first_signatures = {
            trajectory_action_signature(row) for row in first_rows
        }
        last_signatures = {trajectory_action_signature(row) for row in last_rows}
        union = first_signatures | last_signatures
        first_rewards = [reward_value(row) for row in first_rows]
        last_rewards = [reward_value(row) for row in last_rows]
        repeated.append(
            {
                "example_index": example_index,
                "first_policy_step": first_step,
                "last_policy_step": last_step,
                "signature_jaccard": len(first_signatures & last_signatures)
                / len(union)
                if union
                else 1.0,
                "signature_sets_equal": first_signatures == last_signatures,
                "reward_mean_delta": statistics.fmean(last_rewards)
                - statistics.fmean(first_rewards),
                "correct_rate_delta": sum(
                    bool(row.get("correct")) for row in last_rows
                )
                / len(last_rows)
                - sum(bool(row.get("correct")) for row in first_rows)
                / len(first_rows),
                "legal_rate_delta": sum(bool(row.get("legal")) for row in last_rows)
                / len(last_rows)
                - sum(bool(row.get("legal")) for row in first_rows)
                / len(first_rows),
            }
        )

    return {
        "explicit_policy_step_coverage": explicit / len(rows) if rows else 0.0,
        "policy_steps": step_summaries,
        "repeated_task_comparison": {
            "tasks": len(repeated),
            "signature_sets_changed": sum(
                not row["signature_sets_equal"] for row in repeated
            ),
            "mean_signature_jaccard": statistics.fmean(
                row["signature_jaccard"] for row in repeated
            )
            if repeated
            else None,
            "mean_reward_delta": statistics.fmean(
                row["reward_mean_delta"] for row in repeated
            )
            if repeated
            else None,
            "mean_correct_rate_delta": statistics.fmean(
                row["correct_rate_delta"] for row in repeated
            )
            if repeated
            else None,
            "mean_legal_rate_delta": statistics.fmean(
                row["legal_rate_delta"] for row in repeated
            )
            if repeated
            else None,
            "per_task": repeated,
        },
    }


def summarize_rows(
    rows: Sequence[dict[str, Any]], group_size: int
) -> dict[str, Any]:
    groups, incomplete_group_rows = _groups(rows, group_size)
    advantages: list[float] = []
    heterogeneous_groups = 0
    homogeneous_groups = 0
    group_advantage_sum_max_abs = 0.0
    pairwise_reward_order_violations = 0
    heterogeneous_extreme_sign_violations = 0
    homogeneous_nonzero_advantage_groups = 0
    group_reward_shapes: Counter[tuple[tuple[float, int], ...]] = Counter()
    group_summaries: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        rewards = [reward_value(row) for row in group]
        eligible = [optimization_eligible(row) for row in group]
        group_advantages = standardized_group_advantages(rewards, eligible)
        advantages.extend(group_advantages)
        eligible_indices = [
            index for index, keep in enumerate(eligible) if keep
        ]
        eligible_rewards = [rewards[index] for index in eligible_indices]
        eligible_advantages = [
            group_advantages[index] for index in eligible_indices
        ]
        group_advantage_sum_max_abs = max(
            group_advantage_sum_max_abs,
            abs(sum(eligible_advantages)),
        )
        for left in eligible_indices:
            for right in eligible_indices:
                if (
                    rewards[left] < rewards[right]
                    and group_advantages[left] >= group_advantages[right]
                ):
                    pairwise_reward_order_violations += 1
        if len(set(eligible_rewards)) > 1:
            heterogeneous_groups += 1
            minimum_reward = min(eligible_rewards)
            maximum_reward = max(eligible_rewards)
            if any(
                group_advantages[index] >= 0.0
                for index in eligible_indices
                if rewards[index] == minimum_reward
            ) or any(
                group_advantages[index] <= 0.0
                for index in eligible_indices
                if rewards[index] == maximum_reward
            ):
                heterogeneous_extreme_sign_violations += 1
        else:
            homogeneous_groups += 1
            if any(value != 0.0 for value in eligible_advantages):
                homogeneous_nonzero_advantage_groups += 1
        group_reward_shapes[
            tuple(sorted(Counter(rewards).items()))
        ] += 1
        timeout_rows = [row for row in group if timeout_observed(row)]
        group_summaries.append(
            {
                "group_index": group_index,
                "example_index": int(group[0]["example_index"]),
                "trajectory_ids": [row.get("trajectory_id") for row in group],
                "rewards": rewards,
                "optimization_eligible": eligible,
                "advantages": group_advantages,
                "heterogeneous": len(set(eligible_rewards)) > 1,
                "turns": [len(row.get("turns") or []) for row in group],
                "policy_global_steps": sorted(
                    {
                        int(row["policy_global_step"])
                        for row in group
                        if row.get("policy_global_step") is not None
                    }
                ),
                "policy_synced_global_steps": sorted(
                    {
                        int(row["policy_synced_global_step"])
                        for row in group
                        if row.get("policy_synced_global_step") is not None
                    }
                ),
                "policy_micro_steps": sorted(
                    {
                        int(row["policy_micro_step"])
                        for row in group
                        if row.get("policy_micro_step") is not None
                    }
                ),
                "timeout_trajectories": len(timeout_rows),
                "timeout_events": sum(
                    len(timeout_events(row)) for row in timeout_rows
                ),
                "timeout_recovered_correct": sum(
                    bool(row.get("correct")) for row in timeout_rows
                ),
            }
        )

    complete_rows = [row for group in groups for row in group]
    rewards = [reward_value(row) for row in complete_rows]
    eligible = [optimization_eligible(row) for row in complete_rows]
    turn_counts = [len(row.get("turns") or []) for row in complete_rows]
    elapsed = [float(row.get("elapsed_seconds") or 0.0) for row in complete_rows]
    nonzero_advantages = sum(value != 0.0 for value in advantages)
    # The execution-ladder K=8 baseline has a finite categorical reward support.
    # A nonzero coefficient below this threshold cannot be a meaningful relative
    # signal for any of its reward-count compositions; it is floating-point noise
    # (the historical mixed-group defect was about 8.8e-17).  Keep this separate
    # from the ordinary nonzero count so the readiness audit fails closed.
    tiny_nonzero_advantages = sum(
        value != 0.0 and abs(value) < 1e-12 for value in advantages
    )
    reward_tier_advantages: dict[str, dict[str, Any]] = {}
    for reward in sorted(set(rewards)):
        tier = [
            (advantage, turn_count)
            for row_reward, advantage, turn_count, keep in zip(
                rewards,
                advantages,
                turn_counts,
                eligible,
                strict=True,
            )
            if keep and row_reward == reward
        ]
        if not tier:
            continue
        tier_advantages = [advantage for advantage, _ in tier]
        tier_turns = sum(turn_count for _, turn_count in tier)
        reward_tier_advantages[str(reward)] = {
            "trajectories": len(tier),
            "positive_advantages": sum(value > 0.0 for value in tier_advantages),
            "negative_advantages": sum(value < 0.0 for value in tier_advantages),
            "zero_advantages": sum(value == 0.0 for value in tier_advantages),
            "advantage_mean": statistics.fmean(tier_advantages),
            "advantage_min": min(tier_advantages),
            "advantage_max": max(tier_advantages),
            "turns": tier_turns,
            "turn_weighted_advantage_mean": (
                sum(
                    advantage * turn_count
                    for advantage, turn_count in tier
                )
                / tier_turns
                if tier_turns
                else 0.0
            ),
        }
    null_policy_transitions = sum(
        turn_count
        for advantage, turn_count in zip(advantages, turn_counts, strict=True)
        if advantage == 0.0
    )
    contributing_policy_transitions = sum(turn_counts) - null_policy_transitions
    eligible_count = sum(eligible)
    timeout_rows = [row for row in complete_rows if timeout_observed(row)]
    structured_timeout_events = [
        event for row in timeout_rows for event in timeout_events(row)
    ]
    timeout_state_preservation_violations = sum(
        event.get("state_before_hash") != event.get("state_after_hash")
        or not bool((event.get("details") or {}).get("state_preserved"))
        for event in structured_timeout_events
    )
    timeout_missing_structured_events = sum(
        not timeout_events(row) for row in timeout_rows
    )
    group_example_indices = [int(group[0]["example_index"]) for group in groups]
    example_group_counts = Counter(group_example_indices)
    duplicate_example_groups = {
        str(example_index): count
        for example_index, count in sorted(example_group_counts.items())
        if count > 1
    }
    return {
        "rows": len(rows),
        "complete_rows": len(complete_rows),
        "incomplete_group_rows": incomplete_group_rows,
        "groups": len(groups),
        "unique_prompt_groups": len(example_group_counts),
        "duplicate_example_groups": duplicate_example_groups,
        "distinct_prompt_groups_passes": not duplicate_example_groups,
        "heterogeneous_groups": heterogeneous_groups,
        "homogeneous_groups": homogeneous_groups,
        "heterogeneous_group_rate": (
            heterogeneous_groups / len(groups) if groups else 0.0
        ),
        "reward_counts": _counter(rewards),
        "reward_mean": statistics.fmean(rewards) if rewards else 0.0,
        "correct": sum(bool(row.get("correct")) for row in complete_rows),
        "legal": sum(bool(row.get("legal")) for row in complete_rows),
        "optimization_eligible": eligible_count,
        "failure_types": _counter(
            row.get("failure_type") for row in complete_rows
        ),
        "timeout_trajectories": len(timeout_rows),
        "timeout_events": len(structured_timeout_events),
        "timeout_recovered_correct": sum(
            bool(row.get("correct")) for row in timeout_rows
        ),
        "timeout_recovered_legal": sum(
            bool(row.get("legal")) for row in timeout_rows
        ),
        "timeout_unrecovered": sum(
            not bool(row.get("correct")) for row in timeout_rows
        ),
        "timeout_missing_structured_events": timeout_missing_structured_events,
        "timeout_state_preservation_violations": (
            timeout_state_preservation_violations
        ),
        "timeout_contract_passes": (
            timeout_missing_structured_events == 0
            and timeout_state_preservation_violations == 0
        ),
        "rollout_tokenization_warnings": sum(
            bool(row.get("rollout_tokenization_warning")) for row in complete_rows
        ),
        "turns": sum(turn_counts),
        "mean_turns": statistics.fmean(turn_counts) if turn_counts else 0.0,
        "elapsed_seconds_sum": sum(elapsed),
        "elapsed_seconds_mean": statistics.fmean(elapsed) if elapsed else 0.0,
        "nonzero_advantage_trajectories": nonzero_advantages,
        "tiny_nonzero_advantage_trajectories": tiny_nonzero_advantages,
        "exact_zero_contract_passes": tiny_nonzero_advantages == 0,
        "nonzero_advantage_fraction": (
            nonzero_advantages / eligible_count if eligible_count else 0.0
        ),
        "contributing_policy_transitions": contributing_policy_transitions,
        "null_policy_transitions": null_policy_transitions,
        "null_policy_transition_fraction": (
            null_policy_transitions / sum(turn_counts) if turn_counts else 0.0
        ),
        "advantage_min": min(advantages, default=0.0),
        "advantage_max": max(advantages, default=0.0),
        "advantage_mean": statistics.fmean(advantages) if advantages else 0.0,
        "reward_tier_advantages": reward_tier_advantages,
        "credit_direction_checks": {
            "pairwise_reward_order_violations": pairwise_reward_order_violations,
            "heterogeneous_extreme_sign_violations": (
                heterogeneous_extreme_sign_violations
            ),
            "homogeneous_nonzero_advantage_groups": (
                homogeneous_nonzero_advantage_groups
            ),
            "group_advantage_sum_max_abs": group_advantage_sum_max_abs,
            "group_advantage_sum_abs_tolerance": (
                GROUP_ADVANTAGE_SUM_ABS_TOLERANCE
            ),
            "passes": (
                pairwise_reward_order_violations == 0
                and heterogeneous_extreme_sign_violations == 0
                and homogeneous_nonzero_advantage_groups == 0
                and group_advantage_sum_max_abs
                <= GROUP_ADVANTAGE_SUM_ABS_TOLERANCE
            ),
        },
        "group_reward_shapes": {
            json.dumps(shape, separators=(",", ":")): count
            for shape, count in sorted(
                group_reward_shapes.items(), key=lambda item: str(item[0])
            )
        },
        "group_summaries": group_summaries,
    }


def checkpoint_states(run_dir: Path) -> dict[int, dict[str, Any]]:
    states: dict[int, dict[str, Any]] = {}
    for path in run_dir.glob("checkpoint-*/trainer_state.json"):
        try:
            checkpoint_step = int(path.parent.name.split("-", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"invalid checkpoint directory: {path.parent}") from exc
        state = load_json(path)
        if int(state.get("global_step") or -1) != checkpoint_step:
            raise ValueError(f"checkpoint/global_step mismatch at {path}")
        states[checkpoint_step] = state
    return dict(sorted(states.items()))


def trainer_metrics(state: dict[str, Any], step: int) -> dict[str, Any]:
    entries = [
        row
        for row in (state.get("log_history") or [])
        if int(row.get("step") or -1) == step
    ]
    if len(entries) != 1:
        raise ValueError(
            f"checkpoint {step} has {len(entries)} matching log-history rows"
        )
    return entries[0]


def analyze(run_dir: Path) -> dict[str, Any]:
    manifest = load_json(run_dir / "run_manifest.json")
    rows = load_jsonl(run_dir / "rollouts.jsonl")
    group_size = int(manifest["group_size"])
    prompts_per_update = int(manifest["prompts_per_update"])
    accumulation = int(manifest["gradient_accumulation_steps"])
    groups_per_update = prompts_per_update * accumulation
    rows_per_update = group_size * groups_per_update
    states = checkpoint_states(run_dir)
    completed_steps = sorted(states)
    if completed_steps and completed_steps != list(range(1, max(completed_steps) + 1)):
        raise ValueError(f"checkpoint sequence is not contiguous: {completed_steps}")
    required_rows = len(completed_steps) * rows_per_update
    if len(rows) < required_rows:
        raise ValueError(
            f"rollout log has {len(rows)} rows but checkpoints require {required_rows}"
        )

    protocol_versions = sorted({str(row.get("protocol_version")) for row in rows})
    protocol_hashes = sorted({str(row.get("protocol_hash")) for row in rows})
    expected_version = str(manifest.get("protocol_version"))
    expected_hash = str(manifest.get("protocol_hash"))
    contract_matches = (
        protocol_versions in ([], [expected_version])
        and protocol_hashes in ([], [expected_hash])
    )
    policy_step_rows = [
        row for row in rows if row.get("policy_global_step") is not None
    ]
    policy_sync_rows = [
        row for row in rows if row.get("policy_synced_global_step") is not None
    ]
    policy_micro_rows = [
        row for row in rows if row.get("policy_micro_step") is not None
    ]
    policy_schedule_matches = all(
        int(row["policy_global_step"]) == row_index // rows_per_update
        for row_index, row in enumerate(rows)
        if row.get("policy_global_step") is not None
    )
    sync_schedule_matches = all(
        int(row["policy_synced_global_step"]) == row_index // rows_per_update
        for row_index, row in enumerate(rows)
        if row.get("policy_synced_global_step") is not None
    )
    micro_schedule_matches = all(
        int(row["policy_micro_step"]) == row_index // group_size
        for row_index, row in enumerate(rows)
        if row.get("policy_micro_step") is not None
    )

    updates = []
    for step in completed_steps:
        start = (step - 1) * rows_per_update
        end = step * rows_per_update
        summary = summarize_rows(rows[start:end], group_size)
        summary.update(
            {
                "step": step,
                "row_start": start,
                "row_end": end,
                "trainer_metrics": trainer_metrics(states[step], step),
                "checkpoint_precision": checkpoint_precision_audit(
                    run_dir / f"checkpoint-{step}"
                ),
            }
        )
        updates.append(summary)

    partial_rows = rows[required_rows:]
    final_precision_path = run_dir / "training_precision.json"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "analysis_provenance": {
            "analyzer_sha256": sha256_file(Path(__file__).resolve()),
            "torch_available": torch is not None,
            "torch_version": getattr(torch, "__version__", None),
            "group_advantage_sum_abs_tolerance": (
                GROUP_ADVANTAGE_SUM_ABS_TOLERANCE
            ),
        },
        "contract": {
            "manifest_protocol_version": manifest.get("protocol_version"),
            "manifest_protocol_hash": manifest.get("protocol_hash"),
            "rollout_protocol_versions": protocol_versions,
            "rollout_protocol_hashes": protocol_hashes,
            "matches": contract_matches,
        },
        "configuration": {
            "reward_mode": manifest.get("reward_mode"),
            "result_reward_profile": manifest.get("result_reward_profile"),
            "policy_reduction": manifest.get("policy_reduction"),
            "records": int(manifest["records"]),
            "group_size": group_size,
            "prompts_per_update": prompts_per_update,
            "gradient_accumulation_steps": accumulation,
            "groups_per_update": groups_per_update,
            "rows_per_update": rows_per_update,
            "optimizer_steps": int(manifest["optimizer_steps"]),
            "learning_rate": float(manifest["learning_rate"]),
        },
        "implementation_provenance": implementation_provenance(
            run_dir, manifest
        ),
        "progress": {
            "rollout_rows": len(rows),
            "completed_optimizer_steps": len(completed_steps),
            "planned_optimizer_steps": int(manifest["optimizer_steps"]),
            "partial_next_update_rows": len(partial_rows),
            "partial_complete_groups": len(partial_rows) // group_size,
        },
        "online_policy_binding": {
            "policy_global_step_coverage": len(policy_step_rows) / len(rows)
            if rows
            else 0.0,
            "policy_synced_global_step_coverage": len(policy_sync_rows) / len(rows)
            if rows
            else 0.0,
            "policy_micro_step_coverage": len(policy_micro_rows) / len(rows)
            if rows
            else 0.0,
            "policy_global_step_counts": _counter(
                [row["policy_global_step"] for row in policy_step_rows]
            ),
            "policy_synced_global_step_counts": _counter(
                [row["policy_synced_global_step"] for row in policy_sync_rows]
            ),
            "policy_global_step_schedule_matches": policy_schedule_matches,
            "policy_sync_schedule_matches": sync_schedule_matches,
            "policy_micro_step_schedule_matches": micro_schedule_matches,
        },
        "sampled_policy_behavior": summarize_policy_behavior(
            rows, rows_per_update
        ),
        "updates": updates,
        "partial_next_update": summarize_rows(partial_rows, group_size),
        "final_precision": (
            load_json(final_precision_path) if final_precision_path.is_file() else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = analyze(args.run_dir)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists() and not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
