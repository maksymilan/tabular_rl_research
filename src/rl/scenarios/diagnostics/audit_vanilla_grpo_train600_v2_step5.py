#!/usr/bin/env python3
"""Fail-closed step-5 audit for representative600 vanilla-GRPO v2.

This audit deliberately layers on the frozen v1 scientific/optimizer audit
instead of changing it.  V2 changes only the runtime-exclusion contract:

* generation-length truncation, a structured state-preserving tool timeout,
  and context overflow are reported and excluded from optimization when their
  structured evidence agrees with ``process_update=false``;
* known exclusions use preregistered broad safety ceilings (95% overall and
  90% per-step eligibility; 5% length, 2% timeout, and 0.25% context caps)
  instead of v1's brittle 97%/3% pair;
* unknown, unstructured, or contradictory exclusions still fail closed.

All task/cohort identity, K=8 grouping, binary terminal reward, mixed-group,
optimizer, importance-sampling, QLoRA synchronization, precision, movement,
and no-generation-OOM gates remain mandatory.

Exit codes retain the frozen receipt contract: 0 passes, 1 is a scientific or
runtime gate failure, and 2 is malformed/mutable/inconsistent input.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

try:
    from rl.scenarios.diagnostics import audit_vanilla_grpo_train600_step5 as v1
except ImportError:
    import rl.scenarios.diagnostics.audit_vanilla_grpo_train600_step5 as v1


SCHEMA_VERSION = "vanilla-grpo-train600-step5-audit-v2"
EXPECTED_EXPERIMENT_NAME = (
    "qwen3_8b_atomic_v26_vanilla_grpo_representative600_onepass_v2"
)
# Frozen together with the v2 experiment YAML before deployment.  Tests replace
# this identity with their synthetic fixture's digest.
EXPECTED_EXPERIMENT_CONFIG_SHA256 = (
    "c932d7f61ea55d20739fe433808aca1fe95117e0d68a3d7ee93717e5cd175a1e"
)
GATE_SCRIPT_RELATIVE_PATH = (
    "src/rl/scenarios/diagnostics/audit_vanilla_grpo_train600_v2_step5.py"
)
GATE_RECEIPT_NAME = "step5_gate.json"
KNOWN_RUNTIME_EXCLUSIONS = (
    "generation_length",
    "structured_timeout",
    "context_overflow",
)
MIN_OVERALL_ELIGIBLE_FRACTION = 0.95
MIN_STEP_ELIGIBLE_FRACTION = 0.90
MAX_GENERATION_LENGTH_FRACTION = 0.05
MAX_TIMEOUT_TRAJECTORY_FRACTION = 0.02
MAX_CONTEXT_OVERFLOW_FRACTION = 0.0025
_LENGTH_REASONS = frozenset(
    {"length", "max_length", "max_tokens", "max_new_tokens"}
)


InputStructureError = v1.InputStructureError
sha256_file = v1.sha256_file


def _auditor_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _contract() -> dict[str, Any]:
    contract = dict(v1._contract())
    contract.pop("minimum_eligible_fraction", None)
    contract.pop("maximum_generation_length_exclusion_fraction", None)
    contract["runtime_exclusion_policy"] = {
        "known_categories": list(KNOWN_RUNTIME_EXCLUSIONS),
        "requires_explicit_process_update_false": True,
        "requires_optimization_exclusion": "nonsemantic_runtime_failure",
        "minimum_overall_eligible_fraction": MIN_OVERALL_ELIGIBLE_FRACTION,
        "minimum_each_step_eligible_fraction": MIN_STEP_ELIGIBLE_FRACTION,
        "maximum_generation_length_fraction": MAX_GENERATION_LENGTH_FRACTION,
        "maximum_timeout_trajectory_fraction": MAX_TIMEOUT_TRAJECTORY_FRACTION,
        "maximum_context_overflow_fraction": MAX_CONTEXT_OVERFLOW_FRACTION,
        "groups_with_fewer_than_two_eligible": "report_only",
        "unknown_or_contradictory_exclusion": "fail_closed",
    }
    contract["forbidden_runtime_failures"] = ["generation_oom"]
    return contract


def _base_report(paths: dict[str, Path]) -> dict[str, Any]:
    report = v1._base_report(paths)
    report["schema_version"] = SCHEMA_VERSION
    report["auditor_sha256"] = _auditor_sha256()
    report["contract"] = _contract()
    return report


def _generation_truncation_valid(value: Any, expected_max_tokens: Any) -> bool:
    if not isinstance(value, dict):
        return False
    expected_keys = {
        "schema_version",
        "kind",
        "detection",
        "turn_index",
        "completion_tokens",
        "max_new_tokens",
        "finish_reason",
        "finish_reason_field",
        "stop_reason",
        "stop_reason_field",
        "prompt_tokens",
        "response_token_ids_sha256",
    }
    if set(value) != expected_keys:
        return False
    turn_index = value.get("turn_index")
    completion_tokens = value.get("completion_tokens")
    max_new_tokens = value.get("max_new_tokens")
    prompt_tokens = value.get("prompt_tokens")
    response_token_ids_sha256 = value.get("response_token_ids_sha256")
    if not (
        value.get("schema_version") == "vllm-generation-truncation-v2"
        and value.get("kind") == "length"
        and type(turn_index) is int
        and turn_index >= 0
        and type(completion_tokens) is int
        and completion_tokens > 0
        and type(max_new_tokens) is int
        and max_new_tokens > 0
        and type(expected_max_tokens) is int
        and max_new_tokens == expected_max_tokens
        and completion_tokens == max_new_tokens
        and type(prompt_tokens) is int
        and prompt_tokens >= 0
        and isinstance(response_token_ids_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", response_token_ids_sha256) is not None
    ):
        return False

    finish_reason = value.get("finish_reason")
    stop_reason = value.get("stop_reason")
    normalized_finish = (
        finish_reason.strip().lower() if isinstance(finish_reason, str) else None
    )
    normalized_stop = (
        stop_reason.strip().lower() if isinstance(stop_reason, str) else None
    )
    detection = value.get("detection")
    if detection == "max_new_tokens_reached":
        return (
            completion_tokens >= max_new_tokens
            and finish_reason is None
            and value.get("finish_reason_field") is None
            and stop_reason is None
            and value.get("stop_reason_field") is None
        )
    if detection == "explicit_finish_reason":
        return (
            normalized_finish in _LENGTH_REASONS
            and isinstance(value.get("finish_reason_field"), str)
            and bool(value["finish_reason_field"])
        )
    if detection == "explicit_stop_reason":
        return (
            normalized_stop in _LENGTH_REASONS
            and isinstance(value.get("stop_reason_field"), str)
            and bool(value["stop_reason_field"])
        )
    return False


def _context_overflow_valid(
    value: Any,
    *,
    expected_max_tokens: Any,
    expected_max_context_tokens: Any,
) -> bool:
    if not isinstance(value, dict):
        return False
    expected_keys = {
        "schema_version",
        "kind",
        "detection",
        "turn_index",
        "prompt_tokens",
        "max_new_tokens",
        "max_context_tokens",
        "required_tokens",
    }
    if set(value) != expected_keys:
        return False
    turn_index = value.get("turn_index")
    prompt_tokens = value.get("prompt_tokens")
    max_new_tokens = value.get("max_new_tokens")
    max_context_tokens = value.get("max_context_tokens")
    required_tokens = value.get("required_tokens")
    return bool(
        value.get("schema_version") == "vllm-context-overflow-v1"
        and value.get("kind") == "context_overflow"
        and value.get("detection")
        == "prompt_plus_max_new_tokens_exceeds_context"
        and type(turn_index) is int
        and turn_index >= 0
        and type(prompt_tokens) is int
        and prompt_tokens >= 0
        and type(max_new_tokens) is int
        and max_new_tokens == expected_max_tokens
        and type(max_context_tokens) is int
        and max_context_tokens == expected_max_context_tokens
        and type(required_tokens) is int
        and required_tokens == prompt_tokens + max_new_tokens
        and required_tokens > max_context_tokens
    )


def _v2_no_credit_contract(manifest: dict[str, Any]) -> bool:
    reference_policy = manifest.get("reference_policy") or {}
    experiment_config = manifest.get("experiment_config") or {}
    config_rank = experiment_config.get("rank_loss") or {}
    config_optimizer = experiment_config.get("optimizer") or {}
    config_rollout = experiment_config.get("rollout") or {}
    forbidden_custom_manifest_fields = [
        key
        for container in (manifest, experiment_config)
        for key in container
        if str(key).startswith(
            ("custom_credit", "dense_credit", "process_credit", "reward_shaping")
        )
    ]
    return bool(
        reference_policy.get("enabled") is False
        and reference_policy.get("kl_beta") == 0.0
        and experiment_config.get("reward_type") == "result"
        and experiment_config.get("result_reward_profile") == "binary"
        and experiment_config.get("experiment_name") == EXPECTED_EXPERIMENT_NAME
        and experiment_config.get("process_reward_config") is None
        and config_rank.get("enabled") is False
        and config_rank.get("coefficient") == 0.0
        and config_optimizer.get("kl_beta") == 0.0
        and config_optimizer.get("ppo_iterations") == 1
        and config_optimizer.get("steps") == v1.PLANNED_STEPS
        and config_rollout.get("group_size") == v1.GROUP_SIZE
        and config_rollout.get("prompts_per_update") == v1.PROMPTS_PER_STEP
        and not forbidden_custom_manifest_fields
    )


def _runtime_exclusion_kind(
    row: dict[str, Any],
    *,
    expected_max_tokens: Any,
    expected_max_context_tokens: Any,
) -> str | None:
    """Return the one valid exclusion kind, ``None`` for an eligible row.

    This is the stable v2 runtime-exclusion projection shared by the step-5
    gate and the completion audit.  Any unknown, overlapping, malformed, or
    optimizer-inconsistent evidence raises ``InputStructureError``.
    """
    trajectory = str(row.get("trajectory_id") or "<missing-trajectory-id>")
    process_update = row.get("process_update")
    if type(process_update) is not bool:
        raise InputStructureError(
            f"{trajectory}: process_update must be an explicit boolean"
        )
    if row.get("rollout_tokenization_warning") is True:
        raise InputStructureError(f"{trajectory}: tokenization warning is forbidden")
    failure = row.get("failure_type")
    if failure == "generation_oom":
        raise InputStructureError(f"{trajectory}: generation OOM is forbidden")

    truncation = row.get("generation_truncation")
    generation_marker = failure == "generation_length" or truncation is not None
    timeout_events = _deduplicated_structured_timeout_events(row)
    timeout_marker = v1._timeout_observed(row)
    context_evidence = row.get("context_overflow")
    context_marker = failure == "context_overflow" or context_evidence is not None
    if generation_marker and context_marker:
        raise InputStructureError(
            f"{trajectory}: generation-length and context-overflow evidence conflict"
        )

    exclusion = row.get("optimization_exclusion")
    if not (generation_marker or timeout_marker or context_marker):
        if exclusion is not None or process_update is not True:
            raise InputStructureError(
                f"{trajectory}: optimizer exclusion lacks a known runtime cause"
            )
        return None

    if exclusion != "nonsemantic_runtime_failure" or process_update is not False:
        raise InputStructureError(
            f"{trajectory}: known runtime failure must have "
            "optimization_exclusion=nonsemantic_runtime_failure and "
            "process_update=false"
        )
    if timeout_marker and (
        not timeout_events
        or not all(
            v1._timeout_event_preserves_state(event) for event in timeout_events
        )
    ):
        raise InputStructureError(
            f"{trajectory}: timeout evidence is absent or not state-preserving"
        )
    if generation_marker:
        if failure != "generation_length" or not _generation_truncation_valid(
            truncation, expected_max_tokens
        ):
            raise InputStructureError(
                f"{trajectory}: generation-length evidence is malformed"
            )
        return "generation_length"
    if context_marker:
        # Terminal failure takes precedence over an earlier, independently
        # verified state-preserving timeout in the same trajectory.
        if failure != "context_overflow" or not _context_overflow_valid(
            context_evidence,
            expected_max_tokens=expected_max_tokens,
            expected_max_context_tokens=expected_max_context_tokens,
        ):
            raise InputStructureError(
                f"{trajectory}: context-overflow evidence is malformed"
            )
        return "context_overflow"
    if timeout_marker:
        return "structured_timeout"
    raise InputStructureError(f"{trajectory}: unreachable runtime exclusion state")


def _deduplicated_structured_timeout_events(
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    """Project exact timeout events once across row- and turn-level mirrors."""
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in v1._structured_timeout_events(row):
        identity = json.dumps(
            event,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if identity in seen:
            continue
        seen.add(identity)
        events.append(event)
    return events


def _expected_checkpoint_gate(
    run_dir: Path, tasks_manifest_path: Path
) -> dict[str, Any]:
    return {
        "schema_version": "trl-synchronous-checkpoint-gate-v1",
        "step": v1.AUDIT_STEPS,
        "script_relative_path": GATE_SCRIPT_RELATIVE_PATH,
        "script_sha256": _auditor_sha256(),
        "receipt": str((run_dir / GATE_RECEIPT_NAME).resolve()),
        "tasks_manifest": str(tasks_manifest_path.resolve()),
        "tasks_manifest_sha256": v1.EXPECTED_TASKS_MANIFEST_SHA256,
        "execution": "synchronous_on_save_before_next_optimizer_step",
        "resume_policy": "checkpoint_at_or_after_gate_requires_verified_receipt",
    }


def _runtime_exclusion_projection(
    rows: list[dict[str, Any]],
    expected_max_tokens: Any,
    expected_max_context_tokens: Any,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, str]]:
    explicit_process_update = True
    generation_length_contract = True
    timeout_contract = True
    context_overflow_contract = True
    exclusion_contract = True
    known_unique = 0
    generation_length_count = 0
    timeout_trajectory_count = 0
    timeout_event_count = 0
    context_overflow_count = 0
    process_update_true = 0
    process_update_false = 0
    unknown_or_contradictory: list[str] = []
    invalid_generation_length: list[str] = []
    invalid_timeout: list[str] = []
    invalid_context_overflow: list[str] = []
    eligible_by_step: list[int] = []
    groups_with_fewer_than_two_eligible = 0

    for position, row in enumerate(rows):
        label = str(row.get("trajectory_id") or f"row:{position}")
        process_update = row.get("process_update")
        explicit_process_update &= type(process_update) is bool
        process_update_true += int(process_update is True)
        process_update_false += int(process_update is False)

        failure = row.get("failure_type")
        truncation = row.get("generation_truncation")
        generation_marker = failure == "generation_length" or truncation is not None
        context_marker = (
            failure == "context_overflow" or row.get("context_overflow") is not None
        )
        timeout_events = _deduplicated_structured_timeout_events(row)
        timeout_marker = v1._timeout_observed(row)
        if generation_marker:
            generation_length_count += 1

        if timeout_marker:
            timeout_trajectory_count += 1
            timeout_event_count += len(timeout_events)

        if context_marker:
            context_overflow_count += 1

        try:
            kind = _runtime_exclusion_kind(
                row,
                expected_max_tokens=expected_max_tokens,
                expected_max_context_tokens=expected_max_context_tokens,
            )
        except InputStructureError:
            exclusion_contract = False
            unknown_or_contradictory.append(label)
            if generation_marker:
                generation_length_contract = False
                invalid_generation_length.append(label)
            if timeout_marker:
                timeout_contract = False
                invalid_timeout.append(label)
            if context_marker:
                context_overflow_contract = False
                invalid_context_overflow.append(label)
        else:
            known_unique += int(kind is not None)

    for step in range(v1.AUDIT_STEPS):
        block = rows[
            step * v1.ROWS_PER_STEP : (step + 1) * v1.ROWS_PER_STEP
        ]
        eligible_by_step.append(
            sum(row.get("process_update") is True for row in block)
        )
        by_task: dict[Any, int] = {}
        for row in block:
            task = row.get("example_index")
            by_task.setdefault(task, 0)
            by_task[task] += int(row.get("process_update") is True)
        groups_with_fewer_than_two_eligible += sum(
            eligible < 2 for eligible in by_task.values()
        )

    denominator = v1.EXPECTED_ROLLOUTS
    overall_eligible_fraction = process_update_true / denominator
    step_eligible_fractions = [
        eligible / v1.ROWS_PER_STEP for eligible in eligible_by_step
    ]
    generation_length_fraction = generation_length_count / denominator
    timeout_trajectory_fraction = timeout_trajectory_count / denominator
    context_overflow_fraction = context_overflow_count / denominator

    checks = {
        "explicit_process_update_boolean": explicit_process_update,
        "generation_length_exclusions_have_exact_evidence": (
            generation_length_contract
        ),
        "structured_timeouts_are_excluded_and_state_preserving": timeout_contract,
        "context_overflow_exclusions_are_explicit": context_overflow_contract,
        "no_unknown_or_contradictory_optimization_exclusion": exclusion_contract,
        "overall_eligible_at_least_95_percent": (
            overall_eligible_fraction >= MIN_OVERALL_ELIGIBLE_FRACTION
        ),
        "each_step_eligible_at_least_90_percent": (
            len(step_eligible_fractions) == v1.AUDIT_STEPS
            and all(
                fraction >= MIN_STEP_ELIGIBLE_FRACTION
                for fraction in step_eligible_fractions
            )
        ),
        "generation_length_exclusions_at_most_5_percent": (
            generation_length_fraction <= MAX_GENERATION_LENGTH_FRACTION
        ),
        "timeout_trajectories_at_most_2_percent": (
            timeout_trajectory_fraction <= MAX_TIMEOUT_TRAJECTORY_FRACTION
        ),
        "context_overflow_at_most_0_25_percent": (
            context_overflow_fraction <= MAX_CONTEXT_OVERFLOW_FRACTION
        ),
    }
    observed = {
        "process_update_true": process_update_true,
        "process_update_false": process_update_false,
        "known_runtime_exclusions": known_unique,
        "known_runtime_exclusion_fraction": (
            known_unique / len(rows) if rows else 0.0
        ),
        "generation_length_exclusions": generation_length_count,
        "structured_timeout_trajectories": timeout_trajectory_count,
        "structured_timeout_events": timeout_event_count,
        "context_overflow_exclusions": context_overflow_count,
        "unknown_or_contradictory_exclusions": len(unknown_or_contradictory),
        "invalid_generation_length_trajectory_ids": invalid_generation_length,
        "invalid_timeout_trajectory_ids": invalid_timeout,
        "invalid_context_overflow_trajectory_ids": invalid_context_overflow,
        "unknown_or_contradictory_trajectory_ids": unknown_or_contradictory,
        "eligible_by_step": eligible_by_step,
        "eligible_fraction_by_step": step_eligible_fractions,
        "generation_length_exclusion_fraction": generation_length_fraction,
        "structured_timeout_trajectory_fraction": timeout_trajectory_fraction,
        "context_overflow_exclusion_fraction": context_overflow_fraction,
        "groups_with_fewer_than_two_eligible": groups_with_fewer_than_two_eligible,
    }
    details = {
        "explicit_process_update_boolean": (
            f"true={process_update_true} false={process_update_false} rows={len(rows)}"
        ),
        "generation_length_exclusions_have_exact_evidence": (
            f"observed={generation_length_count} invalid={invalid_generation_length}"
        ),
        "structured_timeouts_are_excluded_and_state_preserving": (
            f"trajectories={timeout_trajectory_count} events={timeout_event_count} "
            f"invalid={invalid_timeout}"
        ),
        "context_overflow_exclusions_are_explicit": (
            f"observed={context_overflow_count} invalid={invalid_context_overflow}"
        ),
        "no_unknown_or_contradictory_optimization_exclusion": (
            f"observed={len(unknown_or_contradictory)} "
            f"trajectory_ids={unknown_or_contradictory}"
        ),
        "overall_eligible_at_least_95_percent": (
            f"observed={process_update_true}/{denominator} "
            f"({overall_eligible_fraction:.6f})"
        ),
        "each_step_eligible_at_least_90_percent": (
            f"eligible_by_step={eligible_by_step} "
            f"fractions={step_eligible_fractions}"
        ),
        "generation_length_exclusions_at_most_5_percent": (
            f"observed={generation_length_count}/{denominator} "
            f"({generation_length_fraction:.6f})"
        ),
        "timeout_trajectories_at_most_2_percent": (
            f"observed={timeout_trajectory_count}/{denominator} "
            f"({timeout_trajectory_fraction:.6f})"
        ),
        "context_overflow_at_most_0_25_percent": (
            f"observed={context_overflow_count}/{denominator} "
            f"({context_overflow_fraction:.6f})"
        ),
    }
    return checks, observed, details


def audit(
    run_dir: Path,
    tasks_path: Path,
    tasks_manifest_path: Path,
    initial_adapter: Path,
    *,
    allow_continued_run: bool = False,
) -> dict[str, Any]:
    """Recompute the complete deterministic v2 step-5 receipt."""
    original_config_sha = v1.EXPECTED_EXPERIMENT_CONFIG_SHA256
    try:
        v1.EXPECTED_EXPERIMENT_CONFIG_SHA256 = EXPECTED_EXPERIMENT_CONFIG_SHA256
        report = v1.audit(
            run_dir,
            tasks_path,
            tasks_manifest_path,
            initial_adapter,
            allow_continued_run=allow_continued_run,
        )
    finally:
        v1.EXPECTED_EXPERIMENT_CONFIG_SHA256 = original_config_sha

    manifest = v1._load_object(run_dir / "run_manifest.json", "run manifest")
    all_rows = v1._load_jsonl(run_dir / "rollouts.jsonl", "rollouts")
    rows = all_rows[: v1.EXPECTED_ROLLOUTS] if allow_continued_run else all_rows
    experiment_config = manifest.get("experiment_config") or {}
    rollout_config = experiment_config.get("rollout") or {}
    expected_max_tokens = rollout_config.get("max_new_tokens")
    expected_max_context_tokens = rollout_config.get("max_context_tokens")

    report["schema_version"] = SCHEMA_VERSION
    report["auditor_sha256"] = _auditor_sha256()
    report["contract"] = _contract()

    replaced_checks = {
        "synchronous_checkpoint_gate_contract",
        "no_process_rank_or_custom_credit",
        "eligible_at_least_97_percent",
        "generation_length_exclusions_at_most_3_percent",
        "generation_length_exclusions_are_structured",
        "no_context_overflow",
        "structured_timeouts_are_excluded_and_state_preserving",
        "no_unknown_optimization_exclusion",
    }
    checks = {
        name: value
        for name, value in report["checks"].items()
        if name not in replaced_checks
    }
    issues = [
        issue
        for issue in report["issues"]
        if issue.get("gate") not in replaced_checks
    ]

    expected_gate = _expected_checkpoint_gate(run_dir, tasks_manifest_path)
    checkpoint_gate_ok = manifest.get("checkpoint_gate") == expected_gate
    checks["synchronous_checkpoint_gate_contract"] = checkpoint_gate_ok
    if not checkpoint_gate_ok:
        issues.append(
            {
                "gate": "synchronous_checkpoint_gate_contract",
                "detail": (
                    f"observed={manifest.get('checkpoint_gate')} "
                    f"expected={expected_gate}"
                ),
            }
        )

    no_credit = _v2_no_credit_contract(manifest)
    checks["no_process_rank_or_custom_credit"] = no_credit
    if not no_credit:
        issues.append(
            {
                "gate": "no_process_rank_or_custom_credit",
                "detail": "v2 manifest contains process, rank, custom, or non-vanilla credit",
            }
        )

    runtime_checks, runtime_observed, runtime_details = (
        _runtime_exclusion_projection(
            rows, expected_max_tokens, expected_max_context_tokens
        )
    )
    checks.update(runtime_checks)
    for name, passed in runtime_checks.items():
        if not passed:
            issues.append({"gate": name, "detail": runtime_details[name]})

    report["checks"] = checks
    report["issues"] = issues
    report["observed"].update(runtime_observed)
    report["observed"]["eligible"] = runtime_observed["process_update_true"]
    report["observed"]["eligible_fraction"] = (
        runtime_observed["process_update_true"] / len(rows) if rows else 0.0
    )
    report["observed"]["unknown_optimization_exclusions"] = runtime_observed[
        "unknown_or_contradictory_exclusions"
    ]
    passes = all(checks.values())
    report["status"] = {
        "outcome": "pass" if passes else "gate_failed",
        "passes": passes,
        "continuation_admitted": passes,
        "exit_code": 0 if passes else 1,
    }
    return report


def _input_error_report(paths: dict[str, Path], exc: Exception) -> dict[str, Any]:
    report = _base_report(paths)
    report["issues"] = [
        {
            "gate": "input_structure",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    ]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--tasks-manifest", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)
    paths = {
        "run_dir": args.run_dir,
        "tasks": args.tasks,
        "tasks_manifest": args.tasks_manifest,
        "initial_adapter": args.initial_adapter,
    }
    if not args.output.is_absolute():
        print(
            json.dumps(
                {
                    "outcome": "input_error",
                    "passes": False,
                    "continuation_admitted": False,
                    "exit_code": 2,
                    "error": f"output must be an absolute path: {args.output}",
                },
                ensure_ascii=False,
            )
        )
        return 2

    if args.verify_existing:
        try:
            existing = v1._load_object(args.output, "existing receipt")
            if existing.get("schema_version") != SCHEMA_VERSION:
                raise InputStructureError(
                    "existing receipt schema mismatch: "
                    f"{existing.get('schema_version')!r}"
                )
            recomputed = audit(
                args.run_dir,
                args.tasks,
                args.tasks_manifest,
                args.initial_adapter,
                allow_continued_run=True,
            )
            if existing != recomputed:
                raise InputStructureError(
                    "existing receipt differs from the complete recomputed audit"
                )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "outcome": "input_error",
                        "passes": False,
                        "continuation_admitted": False,
                        "exit_code": 2,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                )
            )
            return 2
        print(json.dumps(existing["status"], ensure_ascii=False, sort_keys=True))
        return int(existing["status"]["exit_code"])

    if args.output.exists() or args.output.is_symlink():
        print(
            json.dumps(
                {
                    "outcome": "input_error",
                    "passes": False,
                    "continuation_admitted": False,
                    "exit_code": 2,
                    "error": "refusing to overwrite existing receipt",
                },
                ensure_ascii=False,
            )
        )
        return 2
    try:
        result = audit(
            args.run_dir,
            args.tasks,
            args.tasks_manifest,
            args.initial_adapter,
        )
        exit_code = int(result["status"]["exit_code"])
    except Exception as exc:
        result = _input_error_report(paths, exc)
        exit_code = 2
    try:
        v1._atomic_json(args.output, result)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "outcome": "input_error",
                    "passes": False,
                    "continuation_admitted": False,
                    "exit_code": 2,
                    "error": f"cannot atomically write receipt: {exc}",
                },
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(result["status"], ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
