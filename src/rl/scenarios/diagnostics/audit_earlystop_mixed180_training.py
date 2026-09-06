#!/usr/bin/env python3
"""Fail-closed completion audit for the earlystop mixed180 GRPO run."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from rl.diagnostics.io import read_jsonl as _diagnostic_read_jsonl
from rl.diagnostics.io import sha256_file as _diagnostic_sha256_file
from rl.diagnostics.validation import regular_file as _diagnostic_regular_file
from rl.diagnostics.validation import require as _diagnostic_require


SCHEMA_VERSION = "earlystop-mixed180-training-completion-audit-v1"
EXPECTED_TASKS_SHA256 = (
    "a015c6513b850576ff95388238d45ad5cc0130da53823bf5d43869668592fb5d"
)
EXPECTED_COHORT_MANIFEST_SHA256 = (
    "e940c996d854a1756ec9787d375517125d9b7b35872b6ce9251232cd7526eaad"
)
EXPECTED_CONFIG_SHA256 = (
    "58a9096030d242a36069ea026f5f27d3f12823b5912f25785756f217a100c6f6"
)
EXPECTED_BASE_MODEL_IDENTITY_SHA256 = (
    "85bd3b7d908acb3a9b9c7ec57b98d6b9e3b2fb427685ae808d1c43173279cecc"
)
EXPECTED_INITIAL_ADAPTER_SHA256 = (
    "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5"
)
EXPECTED_IMPLEMENTATION_LOCK_SHA256 = (
    "f3e52d1e5224955fe4ca436c2274cc205f61ae668c9687183c71b4a737b6ca68"
)


def sha256_file(path: Path) -> str:
    """Compatibility export backed by :mod:`rl.diagnostics.io`."""
    return _diagnostic_sha256_file(path)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Compatibility export backed by strict diagnostics JSONL parsing."""
    return [dict(row) for row in _diagnostic_read_jsonl(path, allow_blank=False)]


def _require(condition: bool, message: str) -> None:
    _diagnostic_require(condition, message)


def _regular(path: Path, label: str) -> None:
    try:
        _diagnostic_regular_file(path, label)
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {path}") from exc


def _task_index(row: dict[str, Any]) -> int:
    value = row.get("example_index")
    _require(type(value) is int, "training task lacks integer example_index")
    return int(value)


def _trainer_metrics(state: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [
        row
        for row in (state.get("log_history") or [])
        if isinstance(row, dict) and "step" in row and "grad_norm" in row
    ]
    by_step = {int(row["step"]): row for row in entries}
    _require(set(by_step) == set(range(1, 13)), "trainer metrics are not exact steps 1..12")
    for step, row in by_step.items():
        gradient = float(row["grad_norm"])
        _require(math.isfinite(gradient) and gradient > 0.0, f"step {step} grad_norm invalid")
    return [by_step[step] for step in range(1, 13)]


def _precision_audit(path: Path) -> dict[str, Any]:
    precision = load_object(path)
    _require(
        precision.get("schema_version") == "table-agent-trl-training-precision-v1",
        "training precision schema mismatch",
    )
    _require(precision.get("optimizer_name") == "adamw_torch", "optimizer mismatch")
    trainable = precision.get("trainable_parameters") or {}
    optimizer = precision.get("optimizer_state") or {}
    _require(int(trainable.get("trainable_tensors") or 0) > 0, "no trainable tensors")
    _require(not trainable.get("non_fp32_tensors"), "non-FP32 trainable tensor")
    _require(int(optimizer.get("moment_tensors") or 0) > 0, "no Adam moments")
    _require(not optimizer.get("non_fp32_moments"), "non-FP32 Adam moment")
    return precision


def _forbidden_transient_paths(run_dir: Path) -> list[str]:
    suffixes = (".next", ".tmp", ".temporary", ".resume-next")
    return sorted(
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.name.endswith(suffixes)
    )


def audit(
    run_dir: Path,
    tasks_path: Path,
    cohort_manifest_path: Path,
) -> dict[str, Any]:
    _require(run_dir.is_dir() and not run_dir.is_symlink(), f"invalid run dir: {run_dir}")
    required = {
        "run_manifest": run_dir / "run_manifest.json",
        "implementation_lock": run_dir / "implementation_lock.json",
        "rollouts": run_dir / "rollouts.jsonl",
        "training_precision": run_dir / "training_precision.json",
        "checkpoint_state": run_dir / "checkpoint-12/trainer_state.json",
        "checkpoint_weights": run_dir / "checkpoint-12/adapter_model.safetensors",
        "checkpoint_config": run_dir / "checkpoint-12/adapter_config.json",
        "checkpoint_optimizer": run_dir / "checkpoint-12/optimizer.pt",
        "checkpoint_scheduler": run_dir / "checkpoint-12/scheduler.pt",
        "checkpoint_rng_state": run_dir / "checkpoint-12/rng_state.pth",
        "final_weights": run_dir / "final/adapter_model.safetensors",
        "final_config": run_dir / "final/adapter_config.json",
        "tasks": tasks_path,
        "cohort_manifest": cohort_manifest_path,
    }
    for label, path in required.items():
        _regular(path, label)
    _require(sha256_file(tasks_path) == EXPECTED_TASKS_SHA256, "train180 SHA mismatch")
    _require(
        sha256_file(cohort_manifest_path) == EXPECTED_COHORT_MANIFEST_SHA256,
        "cohort manifest SHA mismatch",
    )
    tasks = load_jsonl(tasks_path)
    _require(len(tasks) == 180, "train180 row count mismatch")
    task_indices = [_task_index(row) for row in tasks]
    _require(len(set(task_indices)) == 180, "train180 example_index values are not unique")
    cohort = load_object(cohort_manifest_path)
    _require(
        cohort.get("schema_version") == "qwen3-v26-earlystop-mixed-grpo-cohort-v1",
        "cohort schema mismatch",
    )
    _require(
        cohort.get("status") == "frozen_operator_requested_earlystop_mixed180",
        "cohort status mismatch",
    )
    train_output = (cohort.get("outputs") or {}).get("train180") or {}
    _require(train_output.get("sha256") == EXPECTED_TASKS_SHA256, "cohort train SHA mismatch")
    _require(train_output.get("records") == 180, "cohort train records mismatch")
    _require(
        cohort.get("training_contract")
        == {
            "records": 180,
            "group_size": 8,
            "prompts_per_update": 30,
            "optimizer_steps": 12,
            "passes": 2,
            "fresh_online_trajectories": 2880,
            "reward": "binary-result-only",
            "kl_beta": 0.0,
        },
        "cohort training contract mismatch",
    )

    manifest = load_object(required["run_manifest"])
    expected_manifest = {
        "protocol_version": "version26",
        "protocol_hash": "4da19387399bd3a5",
        "student_prompt_sha256": "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316",
        "initial_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
        "examples_json_sha256": EXPECTED_TASKS_SHA256,
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
        "policy_reduction": "trajectory_token_mean",
        "records": 180,
        "expected_records": 180,
        "group_size": 8,
        "prompts_per_update": 30,
        "optimizer_steps": 12,
        "save_steps": 2,
        "save_total_limit": 1,
        "ppo_iterations": 1,
        "gradient_accumulation_steps": 1,
        "seed": 20260812,
        "optimizer_name": "adamw_torch",
        "learning_rate": 8e-7,
        "kl_beta": 0.0,
        "temperature": 0.8,
        "top_p": 1.0,
        "fixed_rollout_pool": None,
        "fixed_pool_manifest": None,
        "experiment_config_sha256": EXPECTED_CONFIG_SHA256,
    }
    for field, expected in expected_manifest.items():
        _require(manifest.get(field) == expected, f"run manifest {field} mismatch")
    model_identity = manifest.get("base_model_identity") or {}
    _require(
        model_identity.get("aggregate_sha256") == EXPECTED_BASE_MODEL_IDENTITY_SHA256,
        "base model identity mismatch",
    )
    lock = load_object(required["implementation_lock"])
    _require(
        sha256_file(required["implementation_lock"])
        == EXPECTED_IMPLEMENTATION_LOCK_SHA256,
        "implementation lock SHA mismatch",
    )
    _require(lock.get("schema_version") == "trl-implementation-lock-v1", "lock schema mismatch")
    _require(lock.get("base_model_identity") == model_identity, "lock model identity mismatch")
    _require(
        lock.get("files") == manifest.get("implementation_source_sha256"),
        "lock/manifest implementation files mismatch",
    )
    snapshot = run_dir / "implementation_source_snapshot"
    files = lock.get("files") or {}
    _require(len(files) > 0, "implementation lock is empty")
    source_mismatches = []
    for relative, expected in sorted(files.items()):
        path = snapshot / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            source_mismatches.append(relative)
    _require(not source_mismatches, f"implementation snapshot mismatch: {source_mismatches[:3]}")

    rows = load_jsonl(required["rollouts"])
    _require(len(rows) == 2880, f"rollout count mismatch: {len(rows)}")
    expected_task_set = set(task_indices)
    rows_per_step = 240
    step_summaries = []
    for step in range(12):
        block = rows[step * rows_per_step : (step + 1) * rows_per_step]
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
            len(counts) == 30 and set(counts.values()) == {8},
            f"step {step} is not 30 prompts x K8",
        )
        _require(set(counts) <= expected_task_set, f"step {step} contains unknown task")
        step_summaries.append(
            {
                "step": step,
                "rows": len(block),
                "prompts": len(counts),
                "policy_micro_steps": sorted(
                    {row.get("policy_micro_step") for row in block},
                    key=lambda value: str(value),
                ),
            }
        )
    pass_summaries = []
    for pass_index in range(2):
        block = rows[pass_index * 1440 : (pass_index + 1) * 1440]
        counts = Counter(_task_index(row) for row in block)
        _require(counts == Counter({index: 8 for index in task_indices}), f"pass {pass_index} coverage mismatch")
        pass_summaries.append({"pass": pass_index, "rows": len(block), "tasks": len(counts)})

    state = load_object(required["checkpoint_state"])
    _require(state.get("global_step") == 12, "checkpoint global_step is not 12")
    _require(state.get("max_steps") == 12, "checkpoint max_steps is not 12")
    trainer_metrics = _trainer_metrics(state)
    precision = _precision_audit(required["training_precision"])
    _require(
        sha256_file(required["checkpoint_weights"]) == sha256_file(required["final_weights"]),
        "final/checkpoint-12 adapter weights differ",
    )
    _require(
        sha256_file(required["checkpoint_config"]) == sha256_file(required["final_config"]),
        "final/checkpoint-12 adapter config differs",
    )
    checkpoints = sorted(
        path.name
        for path in run_dir.glob("checkpoint-*")
        if path.is_dir() and not path.is_symlink()
    )
    _require(checkpoints == ["checkpoint-12"], f"unexpected checkpoints: {checkpoints}")
    transients = _forbidden_transient_paths(run_dir)
    _require(not transients, f"uncommitted transient artifacts remain: {transients[:3]}")

    micro_values = [row.get("policy_micro_step") for row in rows]
    micro_missing_rows = sum(value is None for value in micro_values)
    micro_boundaries = []
    previous = None
    for position, value in enumerate(micro_values):
        if value is None:
            continue
        value = int(value)
        if previous is not None and value < previous:
            micro_boundaries.append(
                {"before_row": position, "left": previous, "right": value}
            )
        previous = value
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {"passes": True, "evaluation_admitted": True},
        "run_dir": str(run_dir),
        "artifact_sha256": {label: sha256_file(path) for label, path in required.items()},
        "contract": {"records": 180, "optimizer_steps": 12, "rollout_rows": 2880},
        "rollout_schedule": {
            "steps": step_summaries,
            "passes": pass_summaries,
            "policy_micro_step_reset_boundaries": micro_boundaries,
            "policy_micro_step_missing_rows": micro_missing_rows,
            "policy_micro_step_policy": "audit-only; resume may reset trainer-local micro-step and does not affect admission",
        },
        "trainer_metrics": trainer_metrics,
        "training_precision": precision,
        "final_adapter_sha256": sha256_file(required["final_weights"]),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(value, target, ensure_ascii=False, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit(args.run_dir, args.tasks, args.cohort_manifest)
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
        atomic_json(args.output, result)
        print(json.dumps(result["status"], ensure_ascii=False))
        return 2
    atomic_json(args.output, result)
    print(json.dumps(result["status"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
