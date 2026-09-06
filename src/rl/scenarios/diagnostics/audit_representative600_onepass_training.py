#!/usr/bin/env python3
"""Fail-closed completion audit for representative600 one-pass vanilla GRPO.

The only evaluation policy admitted by this audit is ``final``, byte-identical
to ``checkpoint-20``.  Earlier checkpoints may remain for recovery or
diagnostics, but are explicitly excluded from checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Package import in tests and repository entry points.
    from . import audit_earlystop_mixed180_training as common
    from . import audit_vanilla_grpo_train600_step5 as step5_audit
except ImportError:  # Direct execution from a deployed diagnostics directory.
    import rl.scenarios.diagnostics.audit_earlystop_mixed180_training as common  # type: ignore[no-redef]
    import rl.scenarios.diagnostics.audit_vanilla_grpo_train600_step5 as step5_audit  # type: ignore[no-redef]


SCHEMA_VERSION = "representative600-onepass-training-completion-audit-v1"
EXPECTED_TASKS_SHA256 = (
    "b5a83c373e9be094ea7355c0bc23c9212249491491f44dfdcb3b09ea49b2457e"
)
EXPECTED_COHORT_MANIFEST_SHA256 = (
    "9212d1f7ca4fb63576eb7e846a9b05f7d159e131fe992495b99e350b6aae8dd6"
)
EXPECTED_CONFIG_SHA256 = (
    "445a1a326abef1bc2c7d16cc8489eb55c264ba1ee80a6151ea4de9e4c2d3398a"
)
EXPECTED_INITIAL_ADAPTER_SHA256 = (
    "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5"
)
EXPECTED_INITIAL_ADAPTER_PATH = (
    "/home/dengyan/tabular_rl_outputs/checkpoints/"
    "qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560"
)
EXPECTED_PROTOCOL_VERSION = "version26"
EXPECTED_PROTOCOL_HASH = "4da19387399bd3a5"
EXPECTED_STUDENT_PROMPT_SHA256 = (
    "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
)
EXPECTED_TOOL_SCHEMA_SHA256 = (
    "e1533d05dcacc028c57d55d358dae109bd980e88a6c1d58c2fb456d68379aad6"
)
EXPECTED_RUNTIME_TREE_SHA256 = (
    "5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab"
)
EXPECTED_RUNTIME_ROOT = (
    "/home/dengyan/tabular_rl_outputs/"
    "runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de"
)
EXPECTED_BASE_MODEL_IDENTITY = {
    "schema_version": "trl-base-model-identity-v1",
    "aggregate_sha256": (
        "85bd3b7d908acb3a9b9c7ec57b98d6b9e3b2fb427685ae808d1c43173279cecc"
    ),
    "files_sha256": {
        "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
        "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
        "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "model-00001-of-00005.safetensors": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
        "model-00002-of-00005.safetensors": "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
        "model-00003-of-00005.safetensors": "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
        "model-00004-of-00005.safetensors": "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
        "model-00005-of-00005.safetensors": "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
        "model.safetensors.index.json": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
        "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    },
}
EXPECTED_RECORDED_PACKAGE_VERSIONS = {
    "framework": "trl",
    "framework_version": "0.29.0",
    "torch_version": "2.9.0",
    "transformers_version": "4.57.6",
    "vllm_version": "0.12.0",
    "accelerate_version": "1.14.0",
    "peft_version": "0.19.1",
    "bitsandbytes_version": "0.50.0",
}

EXPECTED_RECORDS = 600
EXPECTED_STEPS = 20
PROMPTS_PER_STEP = 30
GROUP_SIZE = 8
ROWS_PER_STEP = PROMPTS_PER_STEP * GROUP_SIZE
EXPECTED_ROLLOUTS = EXPECTED_STEPS * ROWS_PER_STEP
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CHECKPOINT_PATTERN = re.compile(r"checkpoint-([0-9]+)")
DIAGNOSTIC_CHECKPOINT_PATTERN = re.compile(r"diagnostic_checkpoint-([0-9]+)")
RESUME_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state.pth",
    "training_args.bin",
)
REQUIRED_IMPLEMENTATION_FILES = {
    "src/rl/experiment_config.py",
    "src/rl/task_loader.py",
    "src/rl/rollout_scoring.py",
    "src/rl/terminal_reward.py",
    "src/rl/tool_environment_v26.py",
    "src/rl/frameworks/trl/rollout.py",
    "src/rl/frameworks/trl/checkpoint_gate.py",
    "src/rl/frameworks/trl/run_transition_grpo.py",
    "src/rl/frameworks/trl/transition_batch.py",
    "src/rl/frameworks/trl/transition_grpo.py",
    "src/rl/frameworks/trl/training_precision.py",
    "src/rl/frameworks/trl/trajectory_ranking.py",
    "src/rl/scenarios/diagnostics/audit_vanilla_grpo_train600_step5.py",
    "src/rl/scenarios/diagnostics/analyze_grpo_training.py",
    "src/rl/scenarios/diagnostics/compare_lora_updates.py",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_fields(
    record: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    for field, expected_value in expected.items():
        _require(
            record.get(field) == expected_value,
            f"{label} {field} mismatch",
        )


def _task_id(row: Mapping[str, Any]) -> str:
    value = row.get("example_id") or row.get("instance_id")
    _require(isinstance(value, str) and bool(value), "training task lacks task id")
    return str(value)


def _task_index(row: Mapping[str, Any]) -> int:
    value = row.get("example_index")
    _require(type(value) is int, "training task lacks integer example_index")
    return int(value)


def _validate_cohort(
    tasks_path: Path, cohort_path: Path
) -> tuple[list[int], list[str], list[dict[str, Any]]]:
    _require(
        common.sha256_file(tasks_path) == EXPECTED_TASKS_SHA256,
        "representative600 task SHA mismatch",
    )
    _require(
        common.sha256_file(cohort_path) == EXPECTED_COHORT_MANIFEST_SHA256,
        "representative600 cohort manifest SHA mismatch",
    )
    tasks = common.load_jsonl(tasks_path)
    _require(len(tasks) == EXPECTED_RECORDS, "representative600 row count mismatch")
    task_indices = [_task_index(row) for row in tasks]
    task_ids = [_task_id(row) for row in tasks]
    _require(len(set(task_indices)) == EXPECTED_RECORDS, "task indices are not unique")
    _require(len(set(task_ids)) == EXPECTED_RECORDS, "task ids are not unique")
    _require(
        len({row.get("db_id") for row in tasks}) == 69,
        "representative600 database coverage mismatch",
    )

    cohort = common.load_object(cohort_path)
    _require_fields(
        cohort,
        {
            "schema_version": "bird-train-vanilla-grpo-cohort-v1",
            "status": "frozen_training_cohort",
            "all_acceptance_gates_passed": True,
            "gold_visibility": (
                "gold SQL remains Harness-only and is never rendered to the actor"
            ),
        },
        "cohort",
    )
    selection = cohort.get("selection") or {}
    _require_fields(
        selection,
        {
            "count": EXPECTED_RECORDS,
            "public_fields_used": [
                "example_id",
                "db_id",
                "question",
                "external_knowledge",
            ],
            "forbidden_fields_not_used": [
                "gold_sql",
                "query",
                "gold_exec_results",
                "gold_sql_path",
            ],
            "gold_used_for_selection_or_order": False,
        },
        "cohort selection",
    )
    output = cohort.get("output") or {}
    _require_fields(
        output,
        {
            "sha256": EXPECTED_TASKS_SHA256,
            "records": EXPECTED_RECORDS,
            "unique_databases": 69,
            "task_ids_in_frozen_order": task_ids,
        },
        "cohort output",
    )
    gates = cohort.get("acceptance_gates")
    _require(
        isinstance(gates, dict) and gates and all(value is True for value in gates.values()),
        "cohort acceptance gates are not all true",
    )
    return task_indices, task_ids, tasks


def _expected_runtime_identity() -> dict[str, Any]:
    return {
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
        "initial_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
        "runtime_content_tree_sha256": EXPECTED_RUNTIME_TREE_SHA256,
    }


def _validate_experiment_config(config: Mapping[str, Any]) -> None:
    custom_fields = sorted(
        key
        for key in config
        if "custom" in str(key).lower() or "dense_reward" in str(key).lower()
    )
    _require(not custom_fields, f"embedded config contains custom credit: {custom_fields}")
    _require_fields(
        config,
        {
            "schema_version": "table-agent-rl-experiment-v1",
            "experiment_name": (
                "qwen3_8b_atomic_v26_vanilla_grpo_"
                "representative600_onepass"
            ),
            "admission_status": "allowed_result_only_control",
            "reward_type": "result",
            "result_reward_profile": "binary",
            "policy_reduction": "trajectory_token_mean",
            "expected_records": EXPECTED_RECORDS,
            "process_reward_config": None,
            "process_loss": True,
            "trainable_part": "all",
        },
        "embedded experiment config",
    )
    _require(
        config.get("rank_loss")
        == {"enabled": False, "coefficient": 0.0, "beta": 0.1},
        "embedded experiment config enables rank loss",
    )
    _require_fields(
        config.get("optimizer") or {},
        {
            "name": "adamw_torch",
            "learning_rate": 8e-7,
            "weight_decay": 0.1,
            "steps": EXPECTED_STEPS,
            "ppo_iterations": 1,
            "lr_scheduler_type": "constant",
            "warmup_ratio": 0.0,
            "kl_beta": 0.0,
            "clip_epsilon": 0.2,
            "clip_epsilon_high": 0.2,
            "gradient_accumulation_steps": 1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.98,
        },
        "embedded optimizer config",
    )
    _require_fields(
        config.get("rollout") or {},
        {
            "prompts_per_update": PROMPTS_PER_STEP,
            "group_size": GROUP_SIZE,
            "max_agent_steps": 30,
            "max_new_tokens": 2048,
            "max_context_tokens": 16384,
            "history_turns": 4,
            "temperature": 0.8,
            "top_p": 1.0,
            "top_k": 0,
            "enable_thinking": True,
        },
        "embedded rollout config",
    )
    runtime = config.get("runtime_contract") or {}
    _require_fields(
        runtime,
        {
            "runtime_root": EXPECTED_RUNTIME_ROOT,
            "runtime_content_tree_sha256": EXPECTED_RUNTIME_TREE_SHA256,
            "protocol_version": EXPECTED_PROTOCOL_VERSION,
            "protocol_hash": EXPECTED_PROTOCOL_HASH,
            "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
            "initial_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
            "reference_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
            "base_model_identity": EXPECTED_BASE_MODEL_IDENTITY,
        },
        "embedded runtime contract",
    )


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    tasks_manifest_path: Path,
) -> None:
    _require_fields(
        manifest,
        {
            **EXPECTED_RECORDED_PACKAGE_VERSIONS,
            "protocol_version": EXPECTED_PROTOCOL_VERSION,
            "protocol_hash": EXPECTED_PROTOCOL_HASH,
            "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
            "tool_schema_sha256": EXPECTED_TOOL_SCHEMA_SHA256,
            "initial_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
            "examples_json_sha256": EXPECTED_TASKS_SHA256,
            "reward_mode": "result-only",
            "result_reward_profile": "binary",
            "policy_reduction": "trajectory_token_mean",
            "records": EXPECTED_RECORDS,
            "expected_records": EXPECTED_RECORDS,
            "group_size": GROUP_SIZE,
            "prompts_per_update": PROMPTS_PER_STEP,
            "optimizer_steps": EXPECTED_STEPS,
            "save_steps": 1,
            "save_total_limit": 20,
            "ppo_iterations": 1,
            "gradient_accumulation_steps": 1,
            "seed": 20260812,
            "trainable_part": "all",
            "policy_loss_coefficient": 1.0,
            "rank_loss_coefficient": 0.0,
            "rank_beta": 0.1,
            "rank_score_tokens": "all",
            "rank_score_scope": "full_trajectory",
            "rank_score_reduction": "sum_tokens",
            "rank_update_scope": "full_trajectory",
            "optimizer_name": "adamw_torch",
            "learning_rate": 8e-7,
            "lr_scheduler_type": "constant",
            "warmup_ratio": 0.0,
            "kl_beta": 0.0,
            "clip_epsilon": 0.2,
            "clip_epsilon_high": 0.2,
            "adam_beta1": 0.9,
            "adam_beta2": 0.98,
            "fixed_rollout_pool": None,
            "fixed_rollout_pool_sha256": None,
            "fixed_pool_manifest": None,
            "fixed_pool_manifest_sha256": None,
            "process_reward_config_sha256": None,
            "counterfactual_manifest_sha256": None,
            "process_admission_policy": "counterfactual-completeness",
            "selection_sha256": None,
            "experiment_config_sha256": EXPECTED_CONFIG_SHA256,
            "base_model_identity": EXPECTED_BASE_MODEL_IDENTITY,
        },
        "run manifest",
    )
    custom_fields = sorted(
        key
        for key in manifest
        if "custom" in str(key).lower() or "dense_reward" in str(key).lower()
    )
    _require(not custom_fields, f"run manifest contains custom credit: {custom_fields}")
    gate = manifest.get("checkpoint_gate") or {}
    _require_fields(
        gate,
        {
            "schema_version": "trl-synchronous-checkpoint-gate-v1",
            "step": 5,
            "script_relative_path": (
                "src/rl/scenarios/diagnostics/audit_vanilla_grpo_train600_step5.py"
            ),
            "receipt": str((run_dir / "step5_gate.json").resolve()),
            "tasks_manifest": str(tasks_manifest_path.resolve()),
            "tasks_manifest_sha256": EXPECTED_COHORT_MANIFEST_SHA256,
            "execution": "synchronous_on_save_before_next_optimizer_step",
            "resume_policy": (
                "checkpoint_at_or_after_gate_requires_verified_receipt"
            ),
        },
        "checkpoint gate",
    )
    gate_sha = gate.get("script_sha256")
    _require(
        isinstance(gate_sha, str) and SHA256_PATTERN.fullmatch(gate_sha) is not None,
        "checkpoint gate script SHA is invalid",
    )
    _require(
        manifest.get("reference_policy")
        == {
            "schema_version": "trl-frozen-reference-policy-v1",
            "enabled": False,
            "kl_beta": 0.0,
            "adapter_name": None,
            "adapter_path": None,
            "adapter_sha256": None,
            "expected_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
            "equals_initial_adapter": True,
            "load_audit": None,
            "after_trainer_init_audit": None,
        },
        "reference policy is not the disabled vanilla control",
    )
    runtime_identity = manifest.get("runtime_identity_audit") or {}
    expected_runtime = _expected_runtime_identity()
    _require_fields(
        runtime_identity,
        {
            "schema_version": "trl-runtime-identity-audit-v1",
            "pinned": True,
            "expected": expected_runtime,
            "actual": expected_runtime,
        },
        "runtime identity audit",
    )
    _require_fields(
        runtime_identity.get("reference") or {},
        {
            "enabled": False,
            "expected_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
            "actual_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
            "equals_initial_adapter": True,
        },
        "runtime reference identity",
    )
    modules = manifest.get("runtime_module_audit") or {}
    _require_fields(
        modules,
        {
            "runtime_root": EXPECTED_RUNTIME_ROOT,
            "tool_environment_factory_module": "tool_environment_v26",
            "module_paths": {
                "protocol": f"{EXPECTED_RUNTIME_ROOT}/src/sft/protocol.py",
                "rollout": f"{EXPECTED_RUNTIME_ROOT}/src/eval/rollout.py",
                "executor": f"{EXPECTED_RUNTIME_ROOT}/src/harness/executor.py",
                "tool_schemes": f"{EXPECTED_RUNTIME_ROOT}/src/sft/tool_schemes.py",
            },
        },
        "runtime module audit",
    )
    _require_fields(
        manifest.get("rollout_settings") or {},
        {
            "context_mode": "rolling-legal-history",
            "denotation_comparison": "bird-set",
            "enable_thinking": True,
            "history_turns": 4,
            "max_batch_calls": 5,
            "max_context_tokens": 16384,
            "max_new_tokens": 2048,
            "max_steps": 30,
            "min_p": 0.0,
            "process_admission_policy": "counterfactual-completeness",
            "repetition_penalty": 1.0,
            "result_reward_profile": "binary",
            "reward_mode": "result-only",
            "temperature": 0.8,
            "tool_execution_timeout_seconds": 10.0,
            "tool_scheme": "atomic",
            "top_k": 0,
            "top_p": 1.0,
        },
        "rollout settings",
    )
    _validate_experiment_config(manifest.get("experiment_config") or {})


def _validate_implementation_lock(
    run_dir: Path,
    manifest: Mapping[str, Any],
    lock_path: Path,
    expected_lock_sha256: str,
) -> dict[str, str]:
    _require(
        SHA256_PATTERN.fullmatch(expected_lock_sha256) is not None,
        "expected implementation lock SHA-256 must be lowercase hexadecimal",
    )
    _require(
        common.sha256_file(lock_path) == expected_lock_sha256,
        "implementation lock SHA mismatch",
    )
    lock = common.load_object(lock_path)
    _require_fields(
        lock,
        {
            "schema_version": "trl-implementation-lock-v1",
            "initial_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
            "protocol_version": EXPECTED_PROTOCOL_VERSION,
            "protocol_hash": EXPECTED_PROTOCOL_HASH,
            "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
            "base_model_identity": EXPECTED_BASE_MODEL_IDENTITY,
            "reference_policy": manifest.get("reference_policy"),
            "experiment_config_sha256": EXPECTED_CONFIG_SHA256,
            "checkpoint_gate": manifest.get("checkpoint_gate"),
            "files": manifest.get("implementation_source_sha256"),
        },
        "implementation lock",
    )
    files = lock.get("files")
    _require(isinstance(files, dict) and bool(files), "implementation lock is empty")
    _require(
        REQUIRED_IMPLEMENTATION_FILES <= set(files),
        "implementation lock omits critical training sources",
    )
    gate = manifest.get("checkpoint_gate") or {}
    gate_relative = gate.get("script_relative_path")
    _require(
        isinstance(gate_relative, str)
        and files.get(gate_relative) == gate.get("script_sha256"),
        "checkpoint gate source is not exactly implementation-locked",
    )
    snapshot = run_dir / "implementation_source_snapshot"
    _require(snapshot.is_dir() and not snapshot.is_symlink(), "invalid source snapshot")
    mismatches: list[str] = []
    normalized: dict[str, str] = {}
    for relative, expected in sorted(files.items()):
        _require(
            isinstance(relative, str)
            and bool(relative)
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            f"unsafe implementation source path: {relative!r}",
        )
        _require(
            isinstance(expected, str) and SHA256_PATTERN.fullmatch(expected) is not None,
            f"invalid implementation source SHA: {relative}",
        )
        source = snapshot / relative
        if (
            not source.is_file()
            or source.is_symlink()
            or common.sha256_file(source) != expected
        ):
            mismatches.append(relative)
        normalized[relative] = expected
    _require(not mismatches, f"implementation snapshot mismatch: {mismatches[:3]}")
    return normalized


def _checkpoint_info(path: Path, step: int, label: str) -> dict[str, Any]:
    _require(path.is_dir() and not path.is_symlink(), f"invalid {label}: {path}")
    for name in RESUME_FILES:
        common._regular(path / name, f"{label}_{name}")
    state = common.load_object(path / "trainer_state.json")
    _require(state.get("global_step") == step, f"{label} global_step mismatch")
    _require(state.get("max_steps") == EXPECTED_STEPS, f"{label} max_steps mismatch")
    return {
        "path": str(path),
        "step": step,
        "purpose": "primary-final-alias" if step == EXPECTED_STEPS else "resume-or-audit-only",
        "evaluation_admitted": step == EXPECTED_STEPS,
        "trainer_state_sha256": common.sha256_file(path / "trainer_state.json"),
        "adapter_sha256": common.sha256_file(path / "adapter_model.safetensors"),
        "adapter_config_sha256": common.sha256_file(path / "adapter_config.json"),
    }


def _validate_checkpoint_inventory(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    primary: dict[str, Any] | None = None
    intermediate: list[dict[str, Any]] = []
    for child in sorted(run_dir.iterdir(), key=lambda value: value.name):
        if child.name.startswith("checkpoint-"):
            match = CHECKPOINT_PATTERN.fullmatch(child.name)
            _require(match is not None, f"invalid checkpoint directory name: {child.name}")
            step = int(match.group(1))
            _require(1 <= step <= EXPECTED_STEPS, f"checkpoint step out of range: {step}")
            info = _checkpoint_info(child, step, child.name)
            if step == EXPECTED_STEPS:
                _require(primary is None, "duplicate checkpoint-20")
                primary = info
            else:
                intermediate.append(info)
        elif child.name.startswith("diagnostic_checkpoint-"):
            match = DIAGNOSTIC_CHECKPOINT_PATTERN.fullmatch(child.name)
            _require(
                match is not None,
                f"invalid diagnostic checkpoint directory name: {child.name}",
            )
            step = int(match.group(1))
            _require(
                1 <= step < EXPECTED_STEPS,
                f"diagnostic checkpoint step out of range: {step}",
            )
            intermediate.append(_checkpoint_info(child, step, child.name))
    _require(primary is not None, "missing checkpoint-20")
    return primary, intermediate


def _trainer_metrics(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = [
        row
        for row in (state.get("log_history") or [])
        if isinstance(row, dict) and "step" in row and "grad_norm" in row
    ]
    by_step: dict[int, dict[str, Any]] = {}
    for row in entries:
        step = int(row["step"])
        _require(step not in by_step, f"duplicate trainer metric step {step}")
        by_step[step] = row
    _require(
        set(by_step) == set(range(1, EXPECTED_STEPS + 1)),
        "trainer metrics are not exact steps 1..20",
    )
    for step, row in by_step.items():
        gradient = float(row["grad_norm"])
        _require(math.isfinite(gradient) and gradient > 0.0, f"step {step} grad_norm invalid")
        loss = row.get("loss")
        _require(
            type(loss) in {int, float} and math.isfinite(float(loss)),
            f"step {step} loss invalid",
        )
        _require(row.get("rollout/episodes") == ROWS_PER_STEP, f"step {step} episode count mismatch")
        _require(row.get("rollout/qlora_layers_synced") == 252.0, f"step {step} policy sync mismatch")
        _require(row.get("learning_rate") == 8e-7, f"step {step} learning rate mismatch")
        for key in (
            "sampling/vllm_importance/log_ratio_abs_mean",
            "sampling/vllm_importance/applied_ratio_mean",
            "sampling/vllm_importance/cap_exceeded_fraction",
        ):
            value = row.get(key)
            _require(
                type(value) in {int, float} and math.isfinite(float(value)),
                f"step {step} {key} invalid",
            )
        _require(
            0.0 <= float(row["sampling/vllm_importance/cap_exceeded_fraction"]) <= 1.0,
            f"step {step} importance cap fraction invalid",
        )
        _require(
            float(row["sampling/vllm_importance/log_ratio_abs_mean"])
            <= step5_audit.MAX_IMPORTANCE_LOG_RATIO_ABS_MEAN,
            f"step {step} importance log-ratio mismatch exceeded the frozen bound",
        )
        _require(
            step5_audit.MIN_IMPORTANCE_APPLIED_MEAN
            <= float(row["sampling/vllm_importance/applied_ratio_mean"])
            <= step5_audit.MAX_IMPORTANCE_APPLIED_MEAN,
            f"step {step} importance applied mean exceeded the frozen interval",
        )
        _require(
            float(row["sampling/vllm_importance/cap_exceeded_fraction"])
            <= step5_audit.MAX_IMPORTANCE_CAP_FRACTION,
            f"step {step} importance cap fraction exceeded the frozen bound",
        )
    return [by_step[step] for step in range(1, EXPECTED_STEPS + 1)]


def _validate_binary_reward(row: Mapping[str, Any], position: int) -> None:
    forbidden = {
        "process_reward",
        "process_reward_exclusion",
        "process_credit",
        "dense_reward",
        "custom_credit",
        "custom_reward",
        "rank_reward",
        "rank_score",
    }
    present = sorted(forbidden & set(row))
    _require(not present, f"rollout {position} contains non-vanilla credit: {present}")
    reward = row.get("result_reward")
    _require(isinstance(reward, dict), f"rollout {position} lacks result_reward")
    _require(
        set(reward) == {"profile", "correct", "executable_terminal", "value"},
        f"rollout {position} result_reward schema mismatch",
    )
    _require(reward.get("profile") == "binary", f"rollout {position} reward profile mismatch")
    _require(type(reward.get("correct")) is bool, f"rollout {position} reward correctness invalid")
    _require(
        type(reward.get("executable_terminal")) is bool,
        f"rollout {position} terminal flag invalid",
    )
    value = reward.get("value")
    _require(type(value) in {int, float} and float(value) in {0.0, 1.0}, f"rollout {position} reward is not binary")
    _require(type(row.get("correct")) is bool, f"rollout {position} correctness invalid")
    _require(type(row.get("legal")) is bool, f"rollout {position} legality invalid")
    _require(reward["correct"] is row["correct"], f"rollout {position} reward correctness mismatch")
    _require(
        reward["executable_terminal"] is row["legal"],
        f"rollout {position} reward legality mismatch",
    )
    exclusion = row.get("optimization_exclusion")
    expected_value = float(exclusion is None and row["correct"] is True)
    _require(
        float(value) == expected_value,
        f"rollout {position} binary reward/exclusion mismatch",
    )


def _validate_runtime_outcome(row: Mapping[str, Any], position: int) -> None:
    failure = row.get("failure_type")
    _require(
        failure not in {"generation_oom", "context_overflow"},
        f"rollout {position} has forbidden runtime failure: {failure}",
    )
    _require(
        row.get("rollout_tokenization_warning") is not True,
        f"rollout {position} has a tokenization warning",
    )
    exclusion = row.get("optimization_exclusion")
    generation_length_evidence = failure == "generation_length" or row.get(
        "generation_truncation"
    ) is not None
    timeout_events = step5_audit._structured_timeout_events(dict(row))
    timeout = failure == "timeout_error" or bool(timeout_events)
    if exclusion is None:
        _require(
            not generation_length_evidence and not timeout,
            f"rollout {position} nonsemantic runtime failure was not excluded",
        )
        return
    _require(
        exclusion == "nonsemantic_runtime_failure",
        f"rollout {position} has unknown optimization exclusion",
    )
    generation_length = failure == "generation_length" and isinstance(
        row.get("generation_truncation"), dict
    )
    _require(
        generation_length or timeout,
        f"rollout {position} exclusion lacks sanctioned runtime evidence",
    )
    if timeout:
        _require(
            bool(timeout_events)
            and all(
                step5_audit._timeout_event_preserves_state(event)
                for event in timeout_events
            ),
            f"rollout {position} timeout did not preserve state",
        )


def _validate_rollouts(
    rows: list[dict[str, Any]], tasks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(len(rows) == EXPECTED_ROLLOUTS, f"rollout count mismatch: {len(rows)}")
    tasks_by_index = {_task_index(task): task for task in tasks}
    task_indices = list(tasks_by_index)
    expected_tasks = set(task_indices)
    trajectory_ids: list[str] = []
    step_summaries: list[dict[str, Any]] = []
    for step in range(EXPECTED_STEPS):
        block = rows[step * ROWS_PER_STEP : (step + 1) * ROWS_PER_STEP]
        _require(
            {row.get("policy_global_step") for row in block} == {step},
            f"policy_global_step mismatch at step {step}",
        )
        _require(
            {row.get("policy_synced_global_step") for row in block} == {step},
            f"policy_synced_global_step mismatch at step {step}",
        )
        counts = Counter(_task_index(row) for row in block)
        _require(
            len(counts) == PROMPTS_PER_STEP and set(counts.values()) == {GROUP_SIZE},
            f"step {step} is not 30 tasks x K8",
        )
        _require(set(counts) <= expected_tasks, f"step {step} contains unknown task")
        for offset, row in enumerate(block):
            position = step * ROWS_PER_STEP + offset
            _require(
                type(row.get("policy_global_step")) is int,
                f"rollout {position} policy_global_step is not an integer",
            )
            _require(
                type(row.get("policy_synced_global_step")) is int,
                f"rollout {position} policy_synced_global_step is not an integer",
            )
            _require_fields(
                row,
                {
                    "policy_global_step": step,
                    "policy_synced_global_step": step,
                    "protocol_version": EXPECTED_PROTOCOL_VERSION,
                    "protocol_hash": EXPECTED_PROTOCOL_HASH,
                    "tool_scheme": "atomic",
                    "environment_implementation": "atomic-v26-isolated-v1",
                    "context_mode": "rolling-legal-history",
                    "denotation_comparison": "bird-set",
                },
                f"rollout {position}",
            )
            task = tasks_by_index[_task_index(row)]
            expected_gold = task.get("gold_sql") or task.get("query")
            _require_fields(
                row,
                {
                    "db_id": task.get("db_id"),
                    "question": task.get("question"),
                    "gold_sql": expected_gold,
                },
                f"rollout {position} frozen-task binding",
            )
            trajectory_id = row.get("trajectory_id")
            _require(
                isinstance(trajectory_id, str) and bool(trajectory_id),
                f"rollout {position} lacks trajectory_id",
            )
            trajectory_ids.append(trajectory_id)
            _validate_runtime_outcome(row, position)
            _validate_binary_reward(row, position)
        step_summaries.append(
            {
                "policy_global_step": step,
                "policy_synced_global_step": step,
                "rows": len(block),
                "tasks": len(counts),
                "samples_per_task": GROUP_SIZE,
            }
        )
    _require(len(set(trajectory_ids)) == EXPECTED_ROLLOUTS, "trajectory ids are not unique")
    total_counts = Counter(_task_index(row) for row in rows)
    _require(
        total_counts == Counter({index: GROUP_SIZE for index in task_indices}),
        "representative600 one-pass K8 coverage mismatch",
    )
    micro_values = [row.get("policy_micro_step") for row in rows]
    boundaries: list[dict[str, int]] = []
    previous: int | None = None
    for position, value in enumerate(micro_values):
        if value is None:
            continue
        _require(type(value) is int, f"rollout {position} policy_micro_step invalid")
        if previous is not None and value < previous:
            boundaries.append({"before_row": position, "left": previous, "right": value})
        previous = value
    return step_summaries, {
        "missing_rows": sum(value is None for value in micro_values),
        "reset_boundaries": boundaries,
        "policy": "audit-only; resume-local micro-step cannot select an evaluation checkpoint",
    }


def _forbidden_run_paths(run_dir: Path) -> list[str]:
    forbidden = set(common._forbidden_transient_paths(run_dir))
    exact_names = {".next", ".tmp", "tmp", ".transient", "transient"}
    extra_suffixes = (".partial", ".incomplete")
    for path in run_dir.rglob("*"):
        if path.is_symlink() or path.name in exact_names or path.name.endswith(extra_suffixes):
            forbidden.add(str(path.relative_to(run_dir)))
    return sorted(forbidden)


def audit(
    run_dir: Path,
    tasks_path: Path,
    cohort_manifest_path: Path,
    experiment_config_path: Path,
    expected_implementation_lock_sha256: str,
) -> dict[str, Any]:
    _require(run_dir.is_dir() and not run_dir.is_symlink(), f"invalid run dir: {run_dir}")
    required = {
        "run_manifest": run_dir / "run_manifest.json",
        "implementation_lock": run_dir / "implementation_lock.json",
        "rollouts": run_dir / "rollouts.jsonl",
        "training_precision": run_dir / "training_precision.json",
        "checkpoint_state": run_dir / "checkpoint-20/trainer_state.json",
        "checkpoint_weights": run_dir / "checkpoint-20/adapter_model.safetensors",
        "checkpoint_config": run_dir / "checkpoint-20/adapter_config.json",
        "checkpoint_optimizer": run_dir / "checkpoint-20/optimizer.pt",
        "checkpoint_scheduler": run_dir / "checkpoint-20/scheduler.pt",
        "checkpoint_rng_state": run_dir / "checkpoint-20/rng_state.pth",
        "checkpoint_training_args": run_dir / "checkpoint-20/training_args.bin",
        "final_weights": run_dir / "final/adapter_model.safetensors",
        "final_config": run_dir / "final/adapter_config.json",
        "final_training_args": run_dir / "final/training_args.bin",
        "step5_gate": run_dir / "step5_gate.json",
        "step5_gate_invocation": run_dir / "checkpoint_gate_step5_invocation.json",
        "tasks": tasks_path,
        "cohort_manifest": cohort_manifest_path,
        "experiment_config": experiment_config_path,
    }
    for label, path in required.items():
        common._regular(path, label)
    _require(
        common.sha256_file(experiment_config_path) == EXPECTED_CONFIG_SHA256,
        "experiment config SHA mismatch",
    )
    task_indices, _, tasks = _validate_cohort(tasks_path, cohort_manifest_path)

    manifest = common.load_object(required["run_manifest"])
    _validate_manifest(
        manifest,
        run_dir=run_dir,
        tasks_manifest_path=cohort_manifest_path,
    )
    step5_gate = common.load_object(required["step5_gate"])
    initial_adapter_path = Path(str(manifest.get("adapter_path") or ""))
    _require(
        str(initial_adapter_path) == EXPECTED_INITIAL_ADAPTER_PATH,
        "run manifest initial adapter path mismatch",
    )
    recomputed_step5 = step5_audit.audit(
        run_dir,
        tasks_path,
        cohort_manifest_path,
        initial_adapter_path,
        allow_continued_run=True,
    )
    _require(
        step5_gate == recomputed_step5,
        "step5 gate receipt differs from its stable full recomputation",
    )
    _require(
        step5_gate.get("schema_version")
        == "vanilla-grpo-train600-step5-audit-v1",
        "step5 gate receipt schema mismatch",
    )
    _require(
        step5_gate.get("status")
        == {
            "outcome": "pass",
            "passes": True,
            "continuation_admitted": True,
            "exit_code": 0,
        },
        "step5 gate did not admit continuation",
    )
    checkpoint_gate = manifest.get("checkpoint_gate") or {}
    _require(
        step5_gate.get("auditor_sha256") == checkpoint_gate.get("script_sha256"),
        "step5 receipt/auditor source identity mismatch",
    )
    invocation = common.load_object(required["step5_gate_invocation"])
    _require_fields(
        invocation,
        {
            "schema_version": "trl-checkpoint-gate-invocation-v1",
            "step": 5,
            "verify_existing": False,
            "script_relative_path": checkpoint_gate.get("script_relative_path"),
            "script_sha256": checkpoint_gate.get("script_sha256"),
            "returncode": 0,
            "receipt": str((run_dir / "step5_gate.json").resolve()),
            "receipt_sha256": common.sha256_file(required["step5_gate"]),
        },
        "step5 gate invocation",
    )
    resume_path = manifest.get("resume_from_checkpoint")
    if resume_path is not None:
        match = CHECKPOINT_PATTERN.fullmatch(Path(str(resume_path)).name)
        _require(match is not None, "run manifest resume checkpoint path invalid")
        if int(match.group(1)) >= 5:
            verification_path = run_dir / "checkpoint_gate_step5_resume_verification.json"
            common._regular(verification_path, "step5_gate_resume_verification")
            verification = common.load_object(verification_path)
            _require_fields(
                verification,
                {
                    "schema_version": "trl-checkpoint-gate-invocation-v1",
                    "step": 5,
                    "verify_existing": True,
                    "script_sha256": checkpoint_gate.get("script_sha256"),
                    "returncode": 0,
                    "receipt": str((run_dir / "step5_gate.json").resolve()),
                    "receipt_sha256": common.sha256_file(required["step5_gate"]),
                },
                "step5 gate resume verification",
            )
    implementation_files = _validate_implementation_lock(
        run_dir,
        manifest,
        required["implementation_lock"],
        expected_implementation_lock_sha256,
    )
    primary_checkpoint, intermediate_checkpoints = _validate_checkpoint_inventory(run_dir)
    rows = common.load_jsonl(required["rollouts"])
    step_summaries, micro_step_audit = _validate_rollouts(rows, tasks)

    state = common.load_object(required["checkpoint_state"])
    _require(state.get("global_step") == EXPECTED_STEPS, "checkpoint global_step is not 20")
    _require(state.get("max_steps") == EXPECTED_STEPS, "checkpoint max_steps is not 20")
    trainer_metrics = _trainer_metrics(state)
    precision = common._precision_audit(required["training_precision"])
    checkpoint_adapter_sha = common.sha256_file(required["checkpoint_weights"])
    checkpoint_config_sha = common.sha256_file(required["checkpoint_config"])
    _require(
        checkpoint_adapter_sha == common.sha256_file(required["final_weights"]),
        "final/checkpoint-20 adapter weights differ",
    )
    _require(
        checkpoint_config_sha == common.sha256_file(required["final_config"]),
        "final/checkpoint-20 adapter config differs",
    )
    _require(
        common.sha256_file(required["checkpoint_training_args"])
        == common.sha256_file(required["final_training_args"]),
        "final/checkpoint-20 training args differ",
    )
    transients = _forbidden_run_paths(run_dir)
    _require(not transients, f"transient or symlink artifacts remain: {transients[:3]}")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": {"passes": True, "evaluation_admitted": True},
        "run_dir": str(run_dir),
        "artifact_sha256": {
            label: common.sha256_file(path) for label, path in required.items()
        },
        "identity_lock": {
            "implementation_lock_sha256": expected_implementation_lock_sha256,
            "implementation_files": implementation_files,
            "base_model_identity": EXPECTED_BASE_MODEL_IDENTITY,
            "protocol_version": EXPECTED_PROTOCOL_VERSION,
            "protocol_hash": EXPECTED_PROTOCOL_HASH,
            "task_sha256": EXPECTED_TASKS_SHA256,
            "task_manifest_sha256": EXPECTED_COHORT_MANIFEST_SHA256,
            "experiment_config_sha256": EXPECTED_CONFIG_SHA256,
        },
        "contract": {
            "records": EXPECTED_RECORDS,
            "passes": 1,
            "optimizer_steps": EXPECTED_STEPS,
            "prompts_per_step": PROMPTS_PER_STEP,
            "group_size": GROUP_SIZE,
            "rollout_rows": EXPECTED_ROLLOUTS,
            "reward": "binary-result-only",
            "process_rank_custom_credit": "disabled",
        },
        "rollout_schedule": {
            "steps": step_summaries,
            "coverage": "all 600 frozen tasks exactly K8 once",
            "policy_micro_step": micro_step_audit,
        },
        "checkpoint_policy": {
            "primary": "final",
            "checkpoint_alias": "checkpoint-20",
            "primary_adapter_sha256": checkpoint_adapter_sha,
            "primary_adapter_config_sha256": checkpoint_config_sha,
            "posthoc_checkpoint_selection_allowed": False,
            "evaluation_candidates": ["final"],
            "primary_checkpoint": primary_checkpoint,
            "intermediate_checkpoints": intermediate_checkpoints,
            "intermediate_checkpoint_policy": "resume-or-audit-only; never evaluation candidates",
        },
        "trainer_metrics": trainer_metrics,
        "training_precision": precision,
    }


def _publish_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    _require(path.is_absolute(), "completion audit output must be absolute")
    _require(not path.is_symlink(), "completion audit output may not be a symlink")
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        common._regular(path, "existing completion audit")
        _require(
            path.read_bytes() == encoded,
            "refusing to overwrite a different completion audit",
        )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".next", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            common._regular(path, "raced completion audit")
            _require(
                path.read_bytes() == encoded,
                "completion audit raced with different bytes",
            )
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
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-implementation-lock-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit(
            args.run_dir,
            args.tasks,
            args.cohort_manifest,
            args.experiment_config,
            args.expected_implementation_lock_sha256,
        )
    except Exception as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": {
                "passes": False,
                "evaluation_admitted": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        }
        try:
            _publish_immutable_json(args.output, result)
        except Exception as publish_exc:
            print(
                json.dumps(
                    {
                        "passes": False,
                        "evaluation_admitted": False,
                        "error": str(publish_exc),
                    },
                    ensure_ascii=False,
                )
            )
            return 2
        print(json.dumps(result["status"], ensure_ascii=False))
        return 2
    try:
        _publish_immutable_json(args.output, result)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "passes": False,
                    "evaluation_admitted": False,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(result["status"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
