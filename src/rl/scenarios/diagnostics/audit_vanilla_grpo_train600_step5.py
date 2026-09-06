#!/usr/bin/env python3
"""Fail-closed step-5 audit for the frozen train600 vanilla-GRPO baseline.

Exit codes are part of the launcher contract:

* 0: every gate passes and continuation to step 20 is admitted;
* 1: inputs are readable, but at least one scientific/runtime gate fails;
* 2: an input or receipt is missing, malformed, mutable, or inconsistent.

The receipt is deterministic and immutable.  ``--verify-existing`` recomputes
the complete audit and requires byte-equivalent JSON content instead of merely
trusting a prior ``passes`` flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "vanilla-grpo-train600-step5-audit-v1"
EXPECTED_TASKS_SHA256 = (
    "b5a83c373e9be094ea7355c0bc23c9212249491491f44dfdcb3b09ea49b2457e"
)
EXPECTED_TASKS_MANIFEST_SHA256 = (
    "9212d1f7ca4fb63576eb7e846a9b05f7d159e131fe992495b99e350b6aae8dd6"
)
EXPECTED_EXPERIMENT_CONFIG_SHA256 = (
    "445a1a326abef1bc2c7d16cc8489eb55c264ba1ee80a6151ea4de9e4c2d3398a"
)
EXPECTED_INITIAL_ADAPTER_SHA256 = (
    "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5"
)
EXPECTED_BASE_MODEL_IDENTITY_SHA256 = (
    "85bd3b7d908acb3a9b9c7ec57b98d6b9e3b2fb427685ae808d1c43173279cecc"
)
EXPECTED_PROTOCOL_VERSION = "version26"
EXPECTED_PROTOCOL_HASH = "4da19387399bd3a5"
EXPECTED_STUDENT_PROMPT_SHA256 = (
    "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
)
EXPECTED_ENVIRONMENT_IMPLEMENTATION = "atomic-v26-isolated-v1"

EXPECTED_TASKS = 600
AUDIT_STEPS = 5
PLANNED_STEPS = 20
PROMPTS_PER_STEP = 30
GROUP_SIZE = 8
ROWS_PER_STEP = PROMPTS_PER_STEP * GROUP_SIZE
EXPECTED_ROLLOUTS = AUDIT_STEPS * ROWS_PER_STEP
EXPECTED_UNIQUE_PREFIX_TASKS = AUDIT_STEPS * PROMPTS_PER_STEP
EXPECTED_SYNCED_QLORA_LAYERS = 252
EXPECTED_LEARNING_RATE = 8e-7
MIN_ELIGIBLE_FRACTION = 0.97
MAX_GENERATION_LENGTH_FRACTION = 0.03
MIN_LEGAL_FRACTION = 0.90
MIN_MIXED_GROUPS = 30
MIN_MIXED_GROUPS_PER_STEP = 1
MAX_GRAD_NORM = 1.0
MAX_IMPORTANCE_LOG_RATIO_ABS_MEAN = 0.05
MIN_IMPORTANCE_APPLIED_MEAN = 0.95
MAX_IMPORTANCE_APPLIED_MEAN = 1.05
MAX_IMPORTANCE_CAP_FRACTION = 1e-3


class InputStructureError(ValueError):
    """The requested audit cannot be interpreted safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_manifest_sha256(manifest: dict[str, Any]) -> str:
    """Hash identity fields while ignoring the expected resume-path mutation."""

    normalized = dict(manifest)
    normalized["resume_from_checkpoint"] = None
    return _canonical_sha256(normalized)


def _rollout_prefix_sha256(path: Path) -> str:
    raw_lines = path.read_bytes().splitlines(keepends=True)
    if len(raw_lines) < EXPECTED_ROLLOUTS:
        raise InputStructureError(
            f"rollouts are behind step 5: {len(raw_lines)} < {EXPECTED_ROLLOUTS}"
        )
    prefix = raw_lines[:EXPECTED_ROLLOUTS]
    if prefix and not prefix[-1].endswith(b"\n"):
        raise InputStructureError("step-5 rollout prefix has an unterminated final row")
    return hashlib.sha256(b"".join(prefix)).hexdigest()


def _auditor_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _finite_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise InputStructureError(f"invalid {label}: {path}")


def _directory(path: Path, label: str) -> None:
    if not path.is_dir() or path.is_symlink():
        raise InputStructureError(f"invalid {label}: {path}")


def _load_object(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputStructureError(f"malformed {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InputStructureError(f"{label} must be one JSON object: {path}")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _regular_file(path, label)
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise InputStructureError(
                        f"empty {label} row {line_number}: {path}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise InputStructureError(
                        f"non-object {label} row {line_number}: {path}"
                    )
                rows.append(value)
    except InputStructureError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputStructureError(f"malformed {label}: {path}: {exc}") from exc
    return rows


def _contract() -> dict[str, Any]:
    return {
        "cohort": "frozen representative600 in frozen file order",
        "tasks": EXPECTED_TASKS,
        "audit_optimizer_steps": AUDIT_STEPS,
        "planned_optimizer_steps": PLANNED_STEPS,
        "prompts_per_step": PROMPTS_PER_STEP,
        "group_size": GROUP_SIZE,
        "rollouts": EXPECTED_ROLLOUTS,
        "unique_tasks_in_first_five_steps": EXPECTED_UNIQUE_PREFIX_TASKS,
        "reward": "binary-result-only",
        "minimum_mixed_groups": MIN_MIXED_GROUPS,
        "minimum_mixed_groups_per_step": MIN_MIXED_GROUPS_PER_STEP,
        "minimum_eligible_fraction": MIN_ELIGIBLE_FRACTION,
        "maximum_generation_length_exclusion_fraction": (
            MAX_GENERATION_LENGTH_FRACTION
        ),
        "minimum_legal_fraction": MIN_LEGAL_FRACTION,
        "forbidden_runtime_failures": ["context_overflow", "generation_oom"],
        "maximum_tokenization_warnings": 0,
        "learning_rate": EXPECTED_LEARNING_RATE,
        "maximum_grad_norm": MAX_GRAD_NORM,
        "synced_qlora_layers_per_step": EXPECTED_SYNCED_QLORA_LAYERS,
        "maximum_importance_log_ratio_abs_mean": (
            MAX_IMPORTANCE_LOG_RATIO_ABS_MEAN
        ),
        "importance_applied_ratio_mean_interval": [
            MIN_IMPORTANCE_APPLIED_MEAN,
            MAX_IMPORTANCE_APPLIED_MEAN,
        ],
        "maximum_importance_cap_exceeded_fraction": (
            MAX_IMPORTANCE_CAP_FRACTION
        ),
        "checkpoint_movement": "raw-and-effective-finite-positive-vs-sft1",
    }


def _base_report(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "auditor_sha256": _auditor_sha256(),
        "contract": _contract(),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": None}
            for name, path in paths.items()
        },
        "artifact_sha256": {},
        "checks": {},
        "observed": {},
        "precision": None,
        "movement": None,
        "issues": [],
        "status": {
            "outcome": "input_error",
            "passes": False,
            "continuation_admitted": False,
            "exit_code": 2,
        },
    }


def _task_id(row: dict[str, Any]) -> str | None:
    value = row.get("example_id") or row.get("instance_id")
    return str(value) if value is not None and str(value) else None


def _task_gold_sql(row: dict[str, Any]) -> Any:
    return row.get("gold_sql") or row.get("query")


def _timeout_event(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("failure_type") == "timeout_error"
        or value.get("error_type") == "timeout_error"
        or value.get("execution_error_type") == "timeout_error"
        or value.get("recovered_from_error_type") == "timeout_error"
        or value.get("error_code") == "tool_execution_timeout"
        or value.get("code") == "tool_execution_timeout"
    )


def _structured_timeout_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = [
        event
        for event in (row.get("error_events") or [])
        if _timeout_event(event)
    ]
    for turn in row.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        event = turn.get("error_event")
        if _timeout_event(event):
            result.append(event)
    return result


def _timeout_observed(row: dict[str, Any]) -> bool:
    return row.get("failure_type") == "timeout_error" or bool(
        _structured_timeout_events(row)
    )


def _timeout_event_preserves_state(event: dict[str, Any]) -> bool:
    before = event.get("state_before_hash")
    after = event.get("state_after_hash")
    return (
        isinstance(before, str)
        and bool(before)
        and before == after
        and (event.get("details") or {}).get("state_preserved") is True
    )


def _checkpoint_precision(checkpoint: Path) -> dict[str, Any]:
    try:
        # Direct deployment executes this file by absolute path, making the
        # diagnostics directory (rather than the repository root) sys.path[0].
        # Package import is retained for tests and ``python -m`` callers.
        try:
            from rl.scenarios.diagnostics.analyze_grpo_training import checkpoint_precision_audit
        except ImportError:
            from rl.scenarios.diagnostics.analyze_grpo_training import (
                checkpoint_precision_audit,
            )
    except Exception as exc:  # pragma: no cover - production dependency failure
        raise InputStructureError(
            f"checkpoint precision dependencies are unavailable: {exc}"
        ) from exc
    try:
        result = checkpoint_precision_audit(checkpoint)
    except Exception as exc:
        raise InputStructureError(f"cannot inspect checkpoint precision: {exc}") from exc
    if not result.get("available"):
        raise InputStructureError(
            "checkpoint precision audit unavailable: "
            f"{result.get('unavailable_reason')!r}"
        )
    return result


def _checkpoint_movement(
    initial_adapter: Path, checkpoint: Path
) -> dict[str, Any]:
    try:
        try:
            from rl.scenarios.diagnostics.compare_lora_updates import compare
        except ImportError:
            from rl.scenarios.diagnostics.compare_lora_updates import compare
    except Exception as exc:  # pragma: no cover - production dependency failure
        raise InputStructureError(
            f"LoRA movement dependencies are unavailable: {exc}"
        ) from exc
    try:
        return compare(initial_adapter, [checkpoint])
    except Exception as exc:
        raise InputStructureError(f"cannot compare LoRA movement: {exc}") from exc


def _movement_values(
    movement: dict[str, Any], checkpoint: Path
) -> tuple[Any, Any, Any, Any]:
    name = str(checkpoint)
    raw = (movement.get("raw_adapter") or {}).get(name) or {}
    effective = (movement.get("effective_lora") or {}).get(name) or {}
    return (
        raw.get("update_norm"),
        raw.get("update_over_reference"),
        effective.get("update_norm"),
        effective.get("update_over_reference"),
    )


def _metric(row: dict[str, Any], name: str) -> Any:
    """Read the exact TRL metric key persisted by the current trainer."""
    return row.get(name)


def audit(
    run_dir: Path,
    tasks_path: Path,
    tasks_manifest_path: Path,
    initial_adapter: Path,
    *,
    allow_continued_run: bool = False,
) -> dict[str, Any]:
    """Recompute the complete deterministic step-5 continuation receipt."""
    paths = {
        "run_dir": run_dir,
        "tasks": tasks_path,
        "tasks_manifest": tasks_manifest_path,
        "initial_adapter": initial_adapter,
    }
    report = _base_report(paths)
    for name, path in paths.items():
        if not path.is_absolute():
            raise InputStructureError(f"{name} must be an absolute path: {path}")
    _directory(run_dir, "run directory")
    _directory(initial_adapter, "initial adapter directory")

    checkpoint = run_dir / "checkpoint-5"
    required_files = {
        "run_manifest": run_dir / "run_manifest.json",
        "implementation_lock": run_dir / "implementation_lock.json",
        "rollouts": run_dir / "rollouts.jsonl",
        "checkpoint_state": checkpoint / "trainer_state.json",
        "checkpoint_weights": checkpoint / "adapter_model.safetensors",
        "checkpoint_config": checkpoint / "adapter_config.json",
        "checkpoint_optimizer": checkpoint / "optimizer.pt",
        "checkpoint_scheduler": checkpoint / "scheduler.pt",
        "checkpoint_rng": checkpoint / "rng_state.pth",
        "initial_weights": initial_adapter / "adapter_model.safetensors",
        "initial_config": initial_adapter / "adapter_config.json",
        "tasks": tasks_path,
        "tasks_manifest": tasks_manifest_path,
    }
    _directory(checkpoint, "checkpoint-5 directory")
    for label, path in required_files.items():
        _regular_file(path, label)
    tasks = _load_jsonl(tasks_path, "tasks")
    tasks_manifest = _load_object(tasks_manifest_path, "tasks manifest")
    manifest = _load_object(required_files["run_manifest"], "run manifest")
    implementation_lock = _load_object(
        required_files["implementation_lock"], "implementation lock"
    )
    all_rows = _load_jsonl(required_files["rollouts"], "rollouts")
    if allow_continued_run:
        if len(all_rows) < EXPECTED_ROLLOUTS:
            raise InputStructureError(
                f"continued run has only {len(all_rows)} rollout rows"
            )
        rows = all_rows[:EXPECTED_ROLLOUTS]
    else:
        rows = all_rows
    state = _load_object(required_files["checkpoint_state"], "trainer state")

    artifact_sha256 = {
        label: sha256_file(path)
        for label, path in required_files.items()
        if label not in {"run_manifest", "rollouts"}
    }
    artifact_sha256["run_manifest"] = _stable_manifest_sha256(manifest)
    artifact_sha256["rollouts"] = _rollout_prefix_sha256(
        required_files["rollouts"]
    )
    report["artifact_sha256"] = artifact_sha256
    for name in ("tasks", "tasks_manifest"):
        report["inputs"][name]["sha256"] = artifact_sha256[name]
    report["inputs"]["initial_adapter"]["sha256"] = artifact_sha256[
        "initial_weights"
    ]

    task_indices: list[int] = []
    task_ids: list[str] = []
    tasks_by_index: dict[int, dict[str, Any]] = {}
    for position, task in enumerate(tasks):
        index = task.get("example_index")
        identity = _task_id(task)
        if type(index) is not int or identity is None:
            raise InputStructureError(
                f"tasks[{position}] lacks integer example_index or task identity"
            )
        if not isinstance(task.get("db_id"), str) or not isinstance(
            task.get("question"), str
        ):
            raise InputStructureError(
                f"tasks[{position}] lacks db_id/question strings"
            )
        if _task_gold_sql(task) is None:
            raise InputStructureError(f"tasks[{position}] lacks gold SQL")
        task_indices.append(index)
        task_ids.append(identity)
        tasks_by_index[index] = task

    issues: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}

    def check(name: str, condition: bool, detail: str) -> None:
        condition = bool(condition)
        checks[name] = condition
        if not condition:
            issues.append({"gate": name, "detail": detail})

    check(
        "frozen_tasks_digest",
        artifact_sha256["tasks"] == EXPECTED_TASKS_SHA256,
        f"observed={artifact_sha256['tasks']} expected={EXPECTED_TASKS_SHA256}",
    )
    check(
        "frozen_tasks_manifest_digest",
        artifact_sha256["tasks_manifest"] == EXPECTED_TASKS_MANIFEST_SHA256,
        (
            f"observed={artifact_sha256['tasks_manifest']} "
            f"expected={EXPECTED_TASKS_MANIFEST_SHA256}"
        ),
    )
    check(
        "tasks_are_exactly_600_unique",
        len(tasks) == EXPECTED_TASKS
        and len(set(task_indices)) == EXPECTED_TASKS
        and len(set(task_ids)) == EXPECTED_TASKS
        and len(tasks_by_index) == EXPECTED_TASKS,
        (
            f"rows={len(tasks)} unique_indices={len(set(task_indices))} "
            f"unique_ids={len(set(task_ids))}"
        ),
    )
    cohort_output = tasks_manifest.get("output") or {}
    check(
        "representative600_manifest_contract",
        tasks_manifest.get("schema_version")
        == "bird-train-vanilla-grpo-cohort-v1"
        and tasks_manifest.get("status") == "frozen_training_cohort"
        and tasks_manifest.get("all_acceptance_gates_passed") is True
        and cohort_output.get("records") == EXPECTED_TASKS
        and cohort_output.get("sha256") == artifact_sha256["tasks"]
        and cohort_output.get("task_ids_in_frozen_order") == task_ids,
        "tasks manifest does not bind the accepted representative600 cohort",
    )
    check(
        "initial_adapter_digest",
        artifact_sha256["initial_weights"] == EXPECTED_INITIAL_ADAPTER_SHA256,
        (
            f"observed={artifact_sha256['initial_weights']} "
            f"expected={EXPECTED_INITIAL_ADAPTER_SHA256}"
        ),
    )

    expected_manifest = {
        "schema_version": "table-agent-trl-transition-grpo-v2",
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
        "initial_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
        "examples_json_sha256": EXPECTED_TASKS_SHA256,
        "experiment_config_sha256": EXPECTED_EXPERIMENT_CONFIG_SHA256,
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
        "policy_reduction": "trajectory_token_mean",
        "records": EXPECTED_TASKS,
        "expected_records": EXPECTED_TASKS,
        "group_size": GROUP_SIZE,
        "prompts_per_update": PROMPTS_PER_STEP,
        "optimizer_steps": PLANNED_STEPS,
        "save_steps": 1,
        "save_total_limit": 20,
        "ppo_iterations": 1,
        "gradient_accumulation_steps": 1,
        "seed": 20260812,
        "trainable_part": "all",
        "policy_loss_coefficient": 1.0,
        "rank_loss_coefficient": 0.0,
        "optimizer_name": "adamw_torch",
        "learning_rate": EXPECTED_LEARNING_RATE,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "kl_beta": 0.0,
        "clip_epsilon": 0.2,
        "clip_epsilon_high": 0.2,
        "adam_beta1": 0.9,
        "adam_beta2": 0.98,
        "vllm_importance_sampling_mode": "token_truncate",
        "vllm_importance_sampling_cap": 3.0,
        "temperature": 0.8,
        "top_p": 1.0,
        "fixed_rollout_pool": None,
        "fixed_pool_manifest": None,
    }
    manifest_mismatches = {
        key: {"observed": manifest.get(key), "expected": expected}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    check(
        "vanilla_grpo_manifest_contract",
        not manifest_mismatches,
        f"mismatches={manifest_mismatches}",
    )
    checkpoint_gate = manifest.get("checkpoint_gate") or {}
    expected_gate = {
        "schema_version": "trl-synchronous-checkpoint-gate-v1",
        "step": AUDIT_STEPS,
        "script_relative_path": (
            "src/rl/scenarios/diagnostics/audit_vanilla_grpo_train600_step5.py"
        ),
        "script_sha256": _auditor_sha256(),
        "receipt": str((run_dir / "step5_gate.json").resolve()),
        "tasks_manifest": str(tasks_manifest_path.resolve()),
        "tasks_manifest_sha256": EXPECTED_TASKS_MANIFEST_SHA256,
        "execution": "synchronous_on_save_before_next_optimizer_step",
        "resume_policy": (
            "checkpoint_at_or_after_gate_requires_verified_receipt"
        ),
    }
    check(
        "synchronous_checkpoint_gate_contract",
        checkpoint_gate == expected_gate,
        f"observed={checkpoint_gate} expected={expected_gate}",
    )
    reference_policy = manifest.get("reference_policy") or {}
    experiment_config = manifest.get("experiment_config") or {}
    config_rank = experiment_config.get("rank_loss") or {}
    config_optimizer = experiment_config.get("optimizer") or {}
    config_rollout = experiment_config.get("rollout") or {}
    forbidden_custom_manifest_fields = sorted(
        key
        for container in (manifest, experiment_config)
        for key in container
        if str(key).startswith(
            ("custom_credit", "dense_credit", "process_credit", "reward_shaping")
        )
    )
    check(
        "no_process_rank_or_custom_credit",
        reference_policy.get("enabled") is False
        and reference_policy.get("kl_beta") == 0.0
        and experiment_config.get("reward_type") == "result"
        and experiment_config.get("result_reward_profile") == "binary"
        and experiment_config.get("experiment_name")
        == "qwen3_8b_atomic_v26_vanilla_grpo_representative600_onepass"
        and experiment_config.get("process_reward_config") is None
        and config_rank.get("enabled") is False
        and config_rank.get("coefficient") == 0.0
        and config_optimizer.get("kl_beta") == 0.0
        and config_optimizer.get("ppo_iterations") == 1
        and config_optimizer.get("steps") == PLANNED_STEPS
        and config_rollout.get("group_size") == GROUP_SIZE
        and config_rollout.get("prompts_per_update") == PROMPTS_PER_STEP
        and not forbidden_custom_manifest_fields,
        (
            "manifest contains process, rank, custom, reference, or "
            f"non-vanilla credit; custom_fields={forbidden_custom_manifest_fields}"
        ),
    )
    base_identity = manifest.get("base_model_identity") or {}
    check(
        "base_model_identity",
        base_identity.get("aggregate_sha256")
        == EXPECTED_BASE_MODEL_IDENTITY_SHA256,
        f"observed={base_identity.get('aggregate_sha256')!r}",
    )

    lock_files = implementation_lock.get("files") or {}
    manifest_files = manifest.get("implementation_source_sha256") or {}
    lock_identity = (
        implementation_lock.get("schema_version") == "trl-implementation-lock-v1"
        and implementation_lock.get("initial_adapter_sha256")
        == EXPECTED_INITIAL_ADAPTER_SHA256
        and implementation_lock.get("protocol_version")
        == EXPECTED_PROTOCOL_VERSION
        and implementation_lock.get("protocol_hash") == EXPECTED_PROTOCOL_HASH
        and implementation_lock.get("student_prompt_sha256")
        == EXPECTED_STUDENT_PROMPT_SHA256
        and implementation_lock.get("base_model_identity") == base_identity
        and implementation_lock.get("reference_policy") == reference_policy
        and implementation_lock.get("experiment_config_sha256")
        == EXPECTED_EXPERIMENT_CONFIG_SHA256
        and implementation_lock.get("checkpoint_gate") == checkpoint_gate
        and bool(lock_files)
        and lock_files == manifest_files
    )
    check(
        "implementation_lock_identity",
        lock_identity,
        "implementation lock and run manifest identities differ",
    )
    snapshot = run_dir / "implementation_source_snapshot"
    snapshot_mismatches = []
    for relative, expected_sha in sorted(lock_files.items()):
        path = snapshot / relative
        if (
            not isinstance(expected_sha, str)
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != expected_sha
        ):
            snapshot_mismatches.append(relative)
    check(
        "immutable_implementation_snapshot",
        bool(lock_files) and not snapshot_mismatches,
        f"mismatches={snapshot_mismatches[:10]}",
    )

    check(
        "checkpoint5_state",
        state.get("global_step") == AUDIT_STEPS
        and state.get("max_steps") == PLANNED_STEPS,
        (
            f"global_step={state.get('global_step')!r} "
            f"max_steps={state.get('max_steps')!r}"
        ),
    )
    later_checkpoints = []
    invalid_checkpoint_entries = []
    for path in run_dir.glob("checkpoint-*"):
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            step = int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            invalid_checkpoint_entries.append(path.name)
            continue
        if step > AUDIT_STEPS:
            later_checkpoints.append(path.name)
    check(
        "no_post_step5_checkpoint",
        not invalid_checkpoint_entries
        and (allow_continued_run or not later_checkpoints),
        (
            f"later={sorted(later_checkpoints)} "
            f"invalid={sorted(invalid_checkpoint_entries)}"
        ),
    )

    log_rows: dict[int, dict[str, Any]] = {}
    duplicate_log_steps: list[int] = []
    for value in state.get("log_history") or []:
        if not isinstance(value, dict) or type(value.get("step")) is not int:
            continue
        step = int(value["step"])
        if not 1 <= step <= AUDIT_STEPS:
            continue
        if step in log_rows:
            duplicate_log_steps.append(step)
        log_rows[step] = value
    exact_log_steps = set(log_rows) == set(range(1, AUDIT_STEPS + 1)) and not (
        duplicate_log_steps
    )
    check(
        "five_exact_trainer_metric_rows",
        exact_log_steps,
        f"steps={sorted(log_rows)} duplicates={duplicate_log_steps}",
    )
    metric_summaries = []
    metrics_finite = exact_log_steps
    gradients_bounded = exact_log_steps
    learning_rates_exact = exact_log_steps
    layers_synced = exact_log_steps
    importance_means_valid = exact_log_steps
    importance_log_ratios_valid = exact_log_steps
    importance_caps_valid = exact_log_steps
    for step in range(1, AUDIT_STEPS + 1):
        row = log_rows.get(step) or {}
        loss = row.get("loss")
        grad_norm = row.get("grad_norm")
        learning_rate = row.get("learning_rate")
        qlora_layers = _metric(row, "rollout/qlora_layers_synced")
        log_ratio_abs_mean = _metric(
            row, "sampling/vllm_importance/log_ratio_abs_mean"
        )
        applied_mean = _metric(
            row, "sampling/vllm_importance/applied_ratio_mean"
        )
        cap_fraction = _metric(
            row, "sampling/vllm_importance/cap_exceeded_fraction"
        )
        metrics_finite &= _finite_number(loss) and _finite_number(grad_norm)
        gradients_bounded &= _finite_number(grad_norm) and (
            0.0 < float(grad_norm) <= MAX_GRAD_NORM
        )
        learning_rates_exact &= _finite_number(learning_rate) and math.isclose(
            float(learning_rate), EXPECTED_LEARNING_RATE, rel_tol=0.0, abs_tol=1e-15
        )
        layers_synced &= _finite_number(qlora_layers) and float(
            qlora_layers
        ) == float(EXPECTED_SYNCED_QLORA_LAYERS)
        importance_log_ratios_valid &= _finite_number(log_ratio_abs_mean) and (
            0.0
            <= float(log_ratio_abs_mean)
            <= MAX_IMPORTANCE_LOG_RATIO_ABS_MEAN
        )
        importance_means_valid &= _finite_number(applied_mean) and (
            MIN_IMPORTANCE_APPLIED_MEAN
            <= float(applied_mean)
            <= MAX_IMPORTANCE_APPLIED_MEAN
        )
        importance_caps_valid &= _finite_number(cap_fraction) and (
            0.0 <= float(cap_fraction) <= MAX_IMPORTANCE_CAP_FRACTION
        )
        metric_summaries.append(
            {
                "step": step,
                "loss": loss,
                "grad_norm": grad_norm,
                "learning_rate": learning_rate,
                "qlora_layers_synced": qlora_layers,
                "importance_log_ratio_abs_mean": log_ratio_abs_mean,
                "importance_applied_ratio_mean": applied_mean,
                "importance_cap_exceeded_fraction": cap_fraction,
            }
        )
    check("finite_loss_and_gradients", metrics_finite, "non-finite metric observed")
    check(
        "positive_bounded_gradients",
        gradients_bounded,
        "grad_norm must be finite and in (0, 1]",
    )
    check(
        "constant_learning_rate_8e_7",
        learning_rates_exact,
        "every logged learning rate must equal 8e-7",
    )
    check(
        "all_252_qlora_layers_synced",
        layers_synced,
        "each step must report exactly 252 synchronized QLoRA layers",
    )
    check(
        "importance_log_ratio_abs_mean_bounded",
        importance_log_ratios_valid,
        "each log_ratio_abs_mean must be in [0, 0.05]",
    )
    check(
        "importance_applied_mean_in_range",
        importance_means_valid,
        "each applied_ratio_mean must be in [0.95, 1.05]",
    )
    check(
        "importance_cap_fraction_bounded",
        importance_caps_valid,
        "each cap_exceeded_fraction must be in [0, 1e-3]",
    )

    check(
        "exact_1200_rollouts",
        len(rows) == EXPECTED_ROLLOUTS,
        f"observed={len(rows)} expected={EXPECTED_ROLLOUTS}",
    )
    known_task_indices = set(task_indices)
    row_shape_valid = True
    protocol_rows_valid = True
    task_binding_valid = True
    binary_reward_valid = True
    no_process_or_custom_rows = True
    schedule_valid = len(rows) == EXPECTED_ROLLOUTS
    group_structure_valid = len(rows) == EXPECTED_ROLLOUTS
    prefix_task_indices: list[int] = []
    eligible_count = 0
    legal_count = 0
    correct_count = 0
    generation_length_count = 0
    generation_oom_count = 0
    context_overflow_count = 0
    tokenization_warning_count = 0
    timeout_trajectory_count = 0
    timeout_event_count = 0
    timeout_contract_valid = True
    generation_length_contract_valid = True
    invalid_exclusion_count = 0
    mixed_groups_total = 0
    mixed_groups_by_step: list[int] = []
    step_summaries: list[dict[str, Any]] = []

    for position, row in enumerate(rows):
        required_types = (
            type(row.get("example_index")) is int
            and isinstance(row.get("trajectory_id"), str)
            and isinstance(row.get("db_id"), str)
            and isinstance(row.get("question"), str)
            and isinstance(row.get("correct"), bool)
            and isinstance(row.get("legal"), bool)
            and isinstance(row.get("turns"), list)
            and isinstance(row.get("error_events"), list)
            and (
                row.get("failure_type") is None
                or isinstance(row.get("failure_type"), str)
            )
        )
        row_shape_valid &= required_types
        protocol_rows_valid &= (
            row.get("protocol_version") == EXPECTED_PROTOCOL_VERSION
            and row.get("protocol_hash") == EXPECTED_PROTOCOL_HASH
            and row.get("environment_implementation")
            == EXPECTED_ENVIRONMENT_IMPLEMENTATION
        )
        index = row.get("example_index")
        task = tasks_by_index.get(index) if type(index) is int else None
        task_binding_valid &= (
            task is not None
            and row.get("db_id") == task.get("db_id")
            and row.get("question") == task.get("question")
            and row.get("gold_sql") == _task_gold_sql(task)
        )
        correct = row.get("correct")
        legal = row.get("legal")
        exclusion = row.get("optimization_exclusion")
        eligible = exclusion is None
        result_reward = row.get("result_reward")
        expected_reward = float(bool(correct)) if eligible else 0.0
        binary_reward_valid &= (
            isinstance(correct, bool)
            and isinstance(legal, bool)
            and isinstance(result_reward, dict)
            and result_reward.get("profile") == "binary"
            and result_reward.get("correct") is correct
            and result_reward.get("executable_terminal") is legal
            and _finite_number(result_reward.get("value"))
            and float(result_reward.get("value")) == expected_reward
        )
        forbidden_row_fields = (
            row.get("process_reward") is not None
            or row.get("process_reward_exclusion") is not None
            or row.get("process_admission_policy") is not None
            or row.get("counterfactual_completeness") is not None
            or row.get("step_rewards") is not None
            or any(
                str(key).startswith(("rank_reward", "custom_credit", "dense_credit"))
                for key in row
            )
        )
        no_process_or_custom_rows &= not forbidden_row_fields
        eligible_count += int(eligible)
        legal_count += int(legal is True)
        correct_count += int(correct is True)
        failure = row.get("failure_type")
        generation_length = failure == "generation_length" or row.get(
            "generation_truncation"
        ) is not None
        generation_length_count += int(generation_length)
        generation_oom_count += int(failure == "generation_oom")
        context_overflow_count += int(failure == "context_overflow")
        tokenization_warning_count += int(
            row.get("rollout_tokenization_warning") is True
        )
        timeout = _timeout_observed(row)
        structured_events = _structured_timeout_events(row)
        timeout_trajectory_count += int(timeout)
        timeout_event_count += len(structured_events)
        if timeout:
            timeout_contract_valid &= (
                exclusion == "nonsemantic_runtime_failure"
                and bool(structured_events)
                and all(_timeout_event_preserves_state(event) for event in structured_events)
            )
        if generation_length:
            generation_length_contract_valid &= (
                failure == "generation_length"
                and isinstance(row.get("generation_truncation"), dict)
                and exclusion == "nonsemantic_runtime_failure"
            )
        sanctioned_exclusion = eligible or generation_length or timeout
        if not sanctioned_exclusion or (
            not eligible and exclusion != "nonsemantic_runtime_failure"
        ):
            invalid_exclusion_count += 1

    if len(rows) == EXPECTED_ROLLOUTS:
        for step in range(AUDIT_STEPS):
            block = rows[step * ROWS_PER_STEP : (step + 1) * ROWS_PER_STEP]
            step_schedule = (
                {row.get("policy_global_step") for row in block} == {step}
                and {row.get("policy_synced_global_step") for row in block} == {step}
                and all(type(row.get("policy_micro_step")) is int for row in block)
            )
            schedule_valid &= step_schedule
            counts = Counter(row.get("example_index") for row in block)
            block_structure = (
                len(counts) == PROMPTS_PER_STEP
                and set(counts.values()) == {GROUP_SIZE}
                and set(counts) <= known_task_indices
            )
            group_structure_valid &= block_structure
            prefix_task_indices.extend(
                index for index in counts if type(index) is int
            )
            mixed = 0
            for index in counts:
                group = [row for row in block if row.get("example_index") == index]
                trajectory_ids = {
                    row.get("trajectory_id") for row in group
                }
                expected_ids = {
                    f"rl_{index}_sample_{sample}" for sample in range(GROUP_SIZE)
                }
                group_structure_valid &= trajectory_ids == expected_ids
                eligible_outcomes = {
                    bool(row.get("correct"))
                    for row in group
                    if row.get("optimization_exclusion") is None
                    and isinstance(row.get("correct"), bool)
                }
                mixed += int(eligible_outcomes == {False, True})
            mixed_groups_total += mixed
            mixed_groups_by_step.append(mixed)
            step_summaries.append(
                {
                    "policy_global_step": step,
                    "rows": len(block),
                    "prompt_groups": len(counts),
                    "mixed_eligible_groups": mixed,
                    "task_indices": sorted(
                        index for index in counts if type(index) is int
                    ),
                }
            )
    unique_prefix_tasks = len(set(prefix_task_indices))
    check("rollout_rows_well_formed", row_shape_valid, "one or more rows have invalid types")
    check(
        "rollout_protocol_identity",
        protocol_rows_valid,
        "one or more rows are not isolated atomic version26",
    )
    check(
        "rollout_task_binding",
        task_binding_valid,
        "one or more rows differ from the frozen task record",
    )
    check(
        "policy_steps_0_through_4",
        schedule_valid,
        "policy_global_step/policy_synced_global_step schedule is not exact 0..4",
    )
    check(
        "five_blocks_are_30_by_k8",
        group_structure_valid,
        "one or more steps are not 30 exact K=8 groups",
    )
    check(
        "first_five_blocks_have_150_unique_tasks",
        len(prefix_task_indices) == EXPECTED_UNIQUE_PREFIX_TASKS
        and unique_prefix_tasks == EXPECTED_UNIQUE_PREFIX_TASKS,
        (
            f"groups={len(prefix_task_indices)} "
            f"unique_tasks={unique_prefix_tasks}"
        ),
    )
    check(
        "binary_terminal_reward_only",
        binary_reward_valid and no_process_or_custom_rows,
        "one or more rollout rows contain non-binary or non-terminal credit",
    )
    check(
        "at_least_one_mixed_group_per_step",
        len(mixed_groups_by_step) == AUDIT_STEPS
        and all(value >= MIN_MIXED_GROUPS_PER_STEP for value in mixed_groups_by_step),
        f"mixed_by_step={mixed_groups_by_step}",
    )
    check(
        "at_least_30_mixed_groups",
        mixed_groups_total >= MIN_MIXED_GROUPS,
        f"observed={mixed_groups_total} expected_at_least={MIN_MIXED_GROUPS}",
    )
    eligible_fraction = eligible_count / EXPECTED_ROLLOUTS
    legal_fraction = legal_count / EXPECTED_ROLLOUTS
    generation_length_fraction = generation_length_count / EXPECTED_ROLLOUTS
    check(
        "eligible_at_least_97_percent",
        eligible_fraction >= MIN_ELIGIBLE_FRACTION,
        f"observed={eligible_count}/{EXPECTED_ROLLOUTS} ({eligible_fraction:.6f})",
    )
    check(
        "generation_length_exclusions_at_most_3_percent",
        generation_length_fraction <= MAX_GENERATION_LENGTH_FRACTION,
        (
            f"observed={generation_length_count}/{EXPECTED_ROLLOUTS} "
            f"({generation_length_fraction:.6f})"
        ),
    )
    check(
        "generation_length_exclusions_are_structured",
        generation_length_contract_valid,
        "every generation-length exclusion must retain truncation evidence",
    )
    check(
        "legal_at_least_90_percent",
        legal_fraction >= MIN_LEGAL_FRACTION,
        f"observed={legal_count}/{EXPECTED_ROLLOUTS} ({legal_fraction:.6f})",
    )
    check(
        "no_generation_oom",
        generation_oom_count == 0,
        f"observed={generation_oom_count}",
    )
    check(
        "no_context_overflow",
        context_overflow_count == 0,
        f"observed={context_overflow_count}",
    )
    check(
        "no_tokenization_warning",
        tokenization_warning_count == 0,
        f"observed={tokenization_warning_count}",
    )
    check(
        "structured_timeouts_are_excluded_and_state_preserving",
        timeout_contract_valid,
        (
            f"timeout_trajectories={timeout_trajectory_count} "
            f"structured_events={timeout_event_count}"
        ),
    )
    check(
        "no_unknown_optimization_exclusion",
        invalid_exclusion_count == 0,
        f"observed={invalid_exclusion_count}",
    )

    precision = _checkpoint_precision(checkpoint)
    report["precision"] = precision
    check(
        "checkpoint_fp32_trainables_and_adam",
        precision.get("passes") is True
        and int(precision.get("adapter_tensor_count") or 0) > 0
        and int(precision.get("adam_moment_tensors") or 0) > 0
        and not precision.get("non_fp32_adapter_tensors")
        and not precision.get("non_fp32_adam_moments"),
        "checkpoint adapters or Adam moments are absent/non-FP32",
    )

    movement = _checkpoint_movement(initial_adapter, checkpoint)
    report["movement"] = movement
    raw_norm, raw_relative, effective_norm, effective_relative = _movement_values(
        movement, checkpoint
    )
    movement_values = [raw_norm, raw_relative, effective_norm, effective_relative]
    check(
        "checkpoint5_raw_and_effective_lora_moved",
        movement.get("lora_module_count") == EXPECTED_SYNCED_QLORA_LAYERS
        and all(_finite_number(value) and float(value) > 0.0 for value in movement_values),
        (
            f"lora_modules={movement.get('lora_module_count')!r} "
            f"raw_norm={raw_norm!r} raw_relative={raw_relative!r} "
            f"effective_norm={effective_norm!r} "
            f"effective_relative={effective_relative!r}"
        ),
    )

    report["checks"] = checks
    report["issues"] = issues
    report["observed"] = {
        "rollouts": len(rows),
        "unique_tasks_in_first_five_steps": unique_prefix_tasks,
        "eligible": eligible_count,
        "eligible_fraction": eligible_fraction,
        "legal": legal_count,
        "legal_fraction": legal_fraction,
        "correct": correct_count,
        "generation_length_exclusions": generation_length_count,
        "generation_length_exclusion_fraction": generation_length_fraction,
        "generation_oom": generation_oom_count,
        "context_overflow": context_overflow_count,
        "tokenization_warnings": tokenization_warning_count,
        "timeout_trajectories": timeout_trajectory_count,
        "structured_timeout_events": timeout_event_count,
        "unknown_optimization_exclusions": invalid_exclusion_count,
        "mixed_groups": mixed_groups_total,
        "mixed_groups_by_step": mixed_groups_by_step,
        "steps": step_summaries,
        "trainer_metrics": metric_summaries,
    }
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if not path.is_absolute():
        raise InputStructureError(f"output must be an absolute path: {path}")
    if path.is_symlink():
        raise InputStructureError(f"output may not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".next", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(payload, target, ensure_ascii=False, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise InputStructureError(
                f"refusing to overwrite concurrently published receipt: {path}"
            ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


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
            existing = _load_object(args.output, "existing receipt")
            if existing.get("schema_version") != SCHEMA_VERSION:
                raise InputStructureError(
                    f"existing receipt schema mismatch: {existing.get('schema_version')!r}"
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
        _atomic_json(args.output, result)
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
