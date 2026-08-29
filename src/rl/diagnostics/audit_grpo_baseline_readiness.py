#!/usr/bin/env python3
"""Combine the independent GRPO training, movement, and evaluation audits.

This module deliberately separates an engineering-valid evaluated baseline from an
observed accuracy benefit. A correct pipeline is not called beneficial merely because
its loss is finite, and an accuracy delta is not trusted unless the training contract,
credit direction, optimizer precision, parameter movement, and online-policy binding
all pass.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "grpo-baseline-readiness-v1"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _positive_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _probability_below(value: Any, threshold: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and 0.0 <= number < threshold


def _precision_passes(precision: dict[str, Any] | None) -> bool:
    if not precision:
        return False
    trainable = precision.get("trainable_parameters") or {}
    optimizer = precision.get("optimizer_state") or {}
    return (
        int(trainable.get("trainable_tensors") or 0) > 0
        and not (trainable.get("non_fp32_tensors") or [])
        and int(optimizer.get("moment_tensors") or 0) > 0
        and not (optimizer.get("non_fp32_moments") or [])
    )


def _movement(lora: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    checkpoints = list(lora.get("checkpoints") or [])
    if len(checkpoints) != 1:
        return False, {"checkpoint": None, "raw": None, "effective": None}
    checkpoint = checkpoints[0]
    raw = (lora.get("raw_adapter") or {}).get(checkpoint) or {}
    effective = (lora.get("effective_lora") or {}).get(checkpoint) or {}
    artifact = (lora.get("checkpoint_artifacts") or {}).get(checkpoint) or {}
    passes = _positive_finite(raw.get("update_norm")) and _positive_finite(
        effective.get("update_norm")
    )
    return passes, {
        "checkpoint": checkpoint,
        "checkpoint_adapter_sha256": artifact.get("adapter_sha256"),
        "raw_update_norm": raw.get("update_norm"),
        "raw_update_over_reference": raw.get("update_over_reference"),
        "effective_update_norm": effective.get("update_norm"),
        "effective_update_over_reference": effective.get(
            "update_over_reference"
        ),
    }


def audit(
    training: dict[str, Any],
    lora: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    candidate: str,
    baseline: str,
    expected_eval_count: int = 1534,
    expected_protocol_version: str = "version36",
    expected_protocol_hash: str = "20a8d3b4356d883c",
    expected_group_size: int = 8,
    expected_reward_profile: str = "execution-ladder",
    expected_policy_reduction: str = "trajectory_token_mean",
) -> dict[str, Any]:
    progress = training.get("progress") or {}
    planned_steps = int(progress.get("planned_optimizer_steps") or 0)
    completed_steps = int(progress.get("completed_optimizer_steps") or 0)
    updates = list(training.get("updates") or [])
    binding = training.get("online_policy_binding") or {}
    provenance = training.get("implementation_provenance") or {}
    behavior = training.get("sampled_policy_behavior") or {}
    repeated = behavior.get("repeated_task_comparison") or {}
    movement_passes, movement = _movement(lora)
    training_contract = training.get("contract") or {}
    training_configuration = training.get("configuration") or {}

    arms = evaluation.get("arms") or {}
    comparison_key = f"{candidate}_vs_{baseline}"
    comparison = (evaluation.get("comparisons") or {}).get(comparison_key) or {}
    accuracy = comparison.get("accuracy") or {}
    legal = comparison.get("legal") or {}
    sequence = comparison.get("sequence_change") or {}
    candidate_arm = arms.get(candidate) or {}
    baseline_arm = arms.get(baseline) or {}
    candidate_identity = candidate_arm.get("evaluation_identity") or {}
    eval_total = int((evaluation.get("cohort") or {}).get("total") or 0)
    evaluation_contract = evaluation.get("evaluation_contract") or {}

    checks = {
        "training_complete": (
            planned_steps > 0
            and completed_steps == planned_steps
            and len(updates) == planned_steps
            and int(progress.get("partial_next_update_rows") or 0) == 0
        ),
        "protocol_contract_matches": (
            bool(training_contract.get("matches"))
            and training_contract.get("manifest_protocol_version")
            == expected_protocol_version
            and training_contract.get("manifest_protocol_hash")
            == expected_protocol_hash
        ),
        "training_objective_matches": (
            training_configuration.get("reward_mode") == "result-only"
            and training_configuration.get("result_reward_profile")
            == expected_reward_profile
            and training_configuration.get("policy_reduction")
            == expected_policy_reduction
            and int(training_configuration.get("group_size") or 0)
            == expected_group_size
        ),
        "source_snapshot_matches": bool(provenance.get("source_snapshot_matches")),
        "credit_direction_passes_all_updates": bool(updates)
        and all(
            bool((update.get("credit_direction_checks") or {}).get("passes"))
            for update in updates
        ),
        "exact_zero_contract_passes_all_updates": bool(updates)
        and all(bool(update.get("exact_zero_contract_passes")) for update in updates),
        "timeout_contract_passes_all_updates": bool(updates)
        and all(bool(update.get("timeout_contract_passes")) for update in updates),
        "nonzero_relative_signal_all_updates": bool(updates)
        and all(
            int(update.get("heterogeneous_groups") or 0) > 0
            and int(update.get("nonzero_advantage_trajectories") or 0) > 0
            for update in updates
        ),
        "distinct_prompt_groups_all_updates": bool(updates)
        and all(
            bool(update.get("distinct_prompt_groups_passes"))
            and int(update.get("unique_prompt_groups") or 0)
            == int(update.get("groups") or 0)
            for update in updates
        ),
        "positive_finite_gradient_all_updates": bool(updates)
        and all(
            _positive_finite(
                (update.get("trainer_metrics") or {}).get("grad_norm")
            )
            for update in updates
        ),
        "fp32_trainable_and_adam_moments": _precision_passes(
            training.get("final_precision")
        ),
        "fp32_persisted_checkpoints_all_updates": bool(updates)
        and all(
            bool((update.get("checkpoint_precision") or {}).get("passes"))
            for update in updates
        ),
        "online_policy_binding_complete": (
            float(binding.get("policy_global_step_coverage") or 0.0) == 1.0
            and float(binding.get("policy_synced_global_step_coverage") or 0.0)
            == 1.0
            and float(binding.get("policy_micro_step_coverage") or 0.0) == 1.0
            and bool(binding.get("policy_global_step_schedule_matches"))
            and bool(binding.get("policy_sync_schedule_matches"))
            and bool(binding.get("policy_micro_step_schedule_matches"))
        ),
        "adapter_and_effective_lora_moved": movement_passes,
        "evaluation_complete": (
            eval_total == expected_eval_count
            and int(candidate_arm.get("total") or 0) == expected_eval_count
            and int(baseline_arm.get("total") or 0) == expected_eval_count
            and bool(comparison)
        ),
        "evaluation_contract_matches": (
            evaluation_contract.get("protocol_version")
            == [expected_protocol_version]
            and evaluation_contract.get("protocol_hash")
            == [expected_protocol_hash]
            and evaluation_contract.get("temperature") == [0.0]
            and evaluation_contract.get("top_p") == [1.0]
            and evaluation_contract.get("denotation_comparison") == ["bird-set"]
        ),
        "evaluation_adapter_identity_matches": (
            candidate_identity.get("schema_version")
            == "evaluation-model-identity-v1"
            and Path(str(candidate_identity.get("adapter_path"))).resolve()
            == Path(str(movement.get("checkpoint"))).resolve()
            and bool(movement.get("checkpoint_adapter_sha256"))
            and candidate_identity.get("adapter_sha256")
            == movement.get("checkpoint_adapter_sha256")
            and candidate_identity.get("protocol_version")
            == expected_protocol_version
            and candidate_identity.get("protocol_hash") == expected_protocol_hash
        ),
        "deterministic_policy_sequence_changed": int(
            sequence.get("exact_action_sequence_changed") or 0
        )
        > 0,
        "accuracy_gain_observed": int(accuracy.get("net") or 0) > 0,
        "accuracy_gain_statistically_supported": (
            int(accuracy.get("net") or 0) > 0
            and _probability_below(accuracy.get("exact_mcnemar_p"), 0.05)
        ),
        "legal_nonregression_observed": int(legal.get("net") or 0) >= 0,
    }
    pipeline_check_names = (
        "training_complete",
        "protocol_contract_matches",
        "training_objective_matches",
        "source_snapshot_matches",
        "credit_direction_passes_all_updates",
        "exact_zero_contract_passes_all_updates",
        "timeout_contract_passes_all_updates",
        "nonzero_relative_signal_all_updates",
        "distinct_prompt_groups_all_updates",
        "positive_finite_gradient_all_updates",
        "fp32_trainable_and_adam_moments",
        "fp32_persisted_checkpoints_all_updates",
        "online_policy_binding_complete",
        "adapter_and_effective_lora_moved",
        "evaluation_complete",
        "evaluation_contract_matches",
        "evaluation_adapter_identity_matches",
    )
    pipeline_valid = all(checks[name] for name in pipeline_check_names)
    policy_change_proven = (
        pipeline_valid
        and checks["adapter_and_effective_lora_moved"]
        and checks["deterministic_policy_sequence_changed"]
    )
    positive_accuracy_baseline_observed = (
        pipeline_valid
        and policy_change_proven
        and checks["accuracy_gain_observed"]
    )
    strict_positive_baseline_observed = (
        positive_accuracy_baseline_observed
        and checks["legal_nonregression_observed"]
    )
    statistically_supported_positive_baseline = (
        positive_accuracy_baseline_observed
        and checks["accuracy_gain_statistically_supported"]
    )
    statistically_supported_strict_positive_baseline = (
        statistically_supported_positive_baseline
        and checks["legal_nonregression_observed"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate": candidate,
        "baseline": baseline,
        "checks": checks,
        "status": {
            "engineering_valid_evaluated_baseline": pipeline_valid,
            "policy_change_proven": policy_change_proven,
            "positive_accuracy_baseline_observed": (
                positive_accuracy_baseline_observed
            ),
            "positive_accuracy_with_legal_nonregression_observed": (
                strict_positive_baseline_observed
            ),
            "statistically_supported_positive_accuracy_baseline": (
                statistically_supported_positive_baseline
            ),
            "statistically_supported_positive_accuracy_with_legal_nonregression": (
                statistically_supported_strict_positive_baseline
            ),
        },
        "training": {
            "completed_optimizer_steps": completed_steps,
            "planned_optimizer_steps": planned_steps,
            "repeated_rollout_tasks": repeated.get("tasks"),
            "repeated_signature_sets_changed": repeated.get(
                "signature_sets_changed"
            ),
            "mean_signature_jaccard": repeated.get("mean_signature_jaccard"),
        },
        "parameter_movement": movement,
        "evaluation": {
            "total": eval_total,
            "candidate_correct": candidate_arm.get("correct"),
            "baseline_correct": baseline_arm.get("correct"),
            "candidate_legal": candidate_arm.get("legal"),
            "baseline_legal": baseline_arm.get("legal"),
            "accuracy_gains": accuracy.get("gains"),
            "accuracy_regressions": accuracy.get("regressions"),
            "accuracy_net": accuracy.get("net"),
            "accuracy_net_rate": accuracy.get("net_rate"),
            "accuracy_net_percentage_points": accuracy.get(
                "net_percentage_points"
            ),
            "accuracy_exact_mcnemar_p": accuracy.get("exact_mcnemar_p"),
            "legal_net": legal.get("net"),
            "legal_net_rate": legal.get("net_rate"),
            "legal_net_percentage_points": legal.get("net_percentage_points"),
            "sequence_changed": sequence.get("exact_action_sequence_changed"),
            "first_action_changed": sequence.get("first_exact_action_changed"),
            "normalized_action_edit_distance": sequence.get(
                "normalized_exact_edit_distance"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-audit", type=Path, required=True)
    parser.add_argument("--lora-comparison", type=Path, required=True)
    parser.add_argument("--evaluation-analysis", type=Path, required=True)
    parser.add_argument("--candidate", default="candidate")
    parser.add_argument("--baseline", default="sft2")
    parser.add_argument("--expected-eval-count", type=int, default=1534)
    parser.add_argument("--expected-protocol-version", default="version36")
    parser.add_argument(
        "--expected-protocol-hash", default="20a8d3b4356d883c"
    )
    parser.add_argument("--expected-group-size", type=int, default=8)
    parser.add_argument(
        "--expected-reward-profile", default="execution-ladder"
    )
    parser.add_argument(
        "--expected-policy-reduction", default="trajectory_token_mean"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {args.output}")
    result = audit(
        load_json(args.training_audit),
        load_json(args.lora_comparison),
        load_json(args.evaluation_analysis),
        candidate=args.candidate,
        baseline=args.baseline,
        expected_eval_count=args.expected_eval_count,
        expected_protocol_version=args.expected_protocol_version,
        expected_protocol_hash=args.expected_protocol_hash,
        expected_group_size=args.expected_group_size,
        expected_reward_profile=args.expected_reward_profile,
        expected_policy_reduction=args.expected_policy_reduction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), **result["status"]}))


if __name__ == "__main__":
    main()
