from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.rl.diagnostics import audit_vanilla_grpo_train600_step5 as step5


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _precision_pass(checkpoint: Path) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint),
        "available": True,
        "adapter_tensor_count": 504,
        "adapter_dtype_counts": {"torch.float32": 504},
        "non_fp32_adapter_tensors": [],
        "optimizer_state_entries": 504,
        "optimizer_tensor_count": 1512,
        "optimizer_tensor_dtype_counts": {"torch.float32": 1512},
        "adam_moment_tensors": 1008,
        "adam_moment_dtype_counts": {"torch.float32": 1008},
        "non_fp32_adam_moments": [],
        "passes": True,
    }


def _movement_pass(initial: Path, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = str(checkpoint_path)
    return {
        "schema_version": "lora-checkpoint-update-comparison-v1",
        "reference": str(initial),
        "checkpoints": [checkpoint],
        "tensor_count": 504,
        "lora_module_count": 252,
        "raw_adapter": {
            checkpoint: {
                "reference_norm": 10.0,
                "checkpoint_norm": 10.01,
                "update_norm": 0.01,
                "update_over_reference": 0.001,
            }
        },
        "effective_lora": {
            checkpoint: {
                "reference_norm": 20.0,
                "update_norm": 0.02,
                "update_over_reference": 0.001,
                "modules": [],
            }
        },
    }


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, list[dict[str, Any]]]:
    tasks_path = tmp_path / "train600.jsonl"
    tasks = [
        {
            "dataset": "bird",
            "example_index": index,
            "example_id": f"bird_train_{index:05d}",
            "db_id": f"db_{index % 7}",
            "db_path": f"/frozen/db_{index % 7}.sqlite",
            "question": f"question {index}",
            "gold_sql": f"SELECT {index}",
        }
        for index in range(600)
    ]
    _write_jsonl(tasks_path, tasks)
    monkeypatch.setattr(step5, "EXPECTED_TASKS_SHA256", _sha(tasks_path))

    tasks_manifest_path = tmp_path / "train600.manifest.json"
    _write_json(
        tasks_manifest_path,
        {
            "schema_version": "bird-train-vanilla-grpo-cohort-v1",
            "status": "frozen_training_cohort",
            "all_acceptance_gates_passed": True,
            "output": {
                "records": 600,
                "sha256": _sha(tasks_path),
                "task_ids_in_frozen_order": [
                    task["example_id"] for task in tasks
                ],
            },
        },
    )
    monkeypatch.setattr(
        step5, "EXPECTED_TASKS_MANIFEST_SHA256", _sha(tasks_manifest_path)
    )

    initial = tmp_path / "sft1"
    initial.mkdir()
    (initial / "adapter_model.safetensors").write_bytes(b"synthetic sft1 fp32\n")
    _write_json(initial / "adapter_config.json", {"r": 8, "lora_alpha": 16})
    monkeypatch.setattr(
        step5,
        "EXPECTED_INITIAL_ADAPTER_SHA256",
        _sha(initial / "adapter_model.safetensors"),
    )
    monkeypatch.setattr(
        step5, "EXPECTED_EXPERIMENT_CONFIG_SHA256", "c" * 64
    )
    monkeypatch.setattr(
        step5, "EXPECTED_BASE_MODEL_IDENTITY_SHA256", "b" * 64
    )
    monkeypatch.setattr(step5, "_checkpoint_precision", _precision_pass)
    monkeypatch.setattr(step5, "_checkpoint_movement", _movement_pass)

    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoint-5"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(
        b"synthetic changed fp32 checkpoint\n"
    )
    _write_json(checkpoint / "adapter_config.json", {"r": 8, "lora_alpha": 16})
    (checkpoint / "optimizer.pt").write_bytes(b"synthetic adam fp32\n")
    (checkpoint / "scheduler.pt").write_bytes(b"synthetic scheduler\n")
    (checkpoint / "rng_state.pth").write_bytes(b"synthetic rng\n")

    base_identity = {"aggregate_sha256": "b" * 64, "files_sha256": {}}
    reference_policy = {"enabled": False, "kl_beta": 0.0}
    source = run_dir / "implementation_source_snapshot/src/rl/fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("SYNTHETIC = True\n", encoding="utf-8")
    implementation_files = {"src/rl/fixture.py": _sha(source)}
    checkpoint_gate = {
        "schema_version": "trl-synchronous-checkpoint-gate-v1",
        "step": 5,
        "script_relative_path": (
            "src/rl/diagnostics/audit_vanilla_grpo_train600_step5.py"
        ),
        "script_sha256": step5._auditor_sha256(),
        "receipt": str((run_dir / "step5_gate.json").resolve()),
        "tasks_manifest": str(tasks_manifest_path.resolve()),
        "tasks_manifest_sha256": _sha(tasks_manifest_path),
        "execution": "synchronous_on_save_before_next_optimizer_step",
        "resume_policy": (
            "checkpoint_at_or_after_gate_requires_verified_receipt"
        ),
    }
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": "table-agent-trl-transition-grpo-v2",
            "protocol_version": "version26",
            "protocol_hash": "4da19387399bd3a5",
            "student_prompt_sha256": (
                "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
            ),
            "initial_adapter_sha256": _sha(
                initial / "adapter_model.safetensors"
            ),
            "examples_json_sha256": _sha(tasks_path),
            "experiment_config_sha256": "c" * 64,
            "reward_mode": "result-only",
            "result_reward_profile": "binary",
            "policy_reduction": "trajectory_token_mean",
            "records": 600,
            "expected_records": 600,
            "group_size": 8,
            "prompts_per_update": 30,
            "optimizer_steps": 20,
            "save_steps": 1,
            "save_total_limit": 20,
            "ppo_iterations": 1,
            "gradient_accumulation_steps": 1,
            "seed": 20260812,
            "trainable_part": "all",
            "policy_loss_coefficient": 1.0,
            "rank_loss_coefficient": 0.0,
            "optimizer_name": "adamw_torch",
            "learning_rate": 8e-7,
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
            "base_model_identity": base_identity,
            "reference_policy": reference_policy,
            "implementation_source_sha256": implementation_files,
            "checkpoint_gate": checkpoint_gate,
            "experiment_config": {
                "experiment_name": (
                    "qwen3_8b_atomic_v26_vanilla_grpo_"
                    "representative600_onepass"
                ),
                "reward_type": "result",
                "result_reward_profile": "binary",
                "process_reward_config": None,
                "rank_loss": {"enabled": False, "coefficient": 0.0},
                "optimizer": {
                    "steps": 20,
                    "ppo_iterations": 1,
                    "kl_beta": 0.0,
                },
                "rollout": {"prompts_per_update": 30, "group_size": 8},
            },
        },
    )
    _write_json(
        run_dir / "implementation_lock.json",
        {
            "schema_version": "trl-implementation-lock-v1",
            "files": implementation_files,
            "initial_adapter_sha256": _sha(
                initial / "adapter_model.safetensors"
            ),
            "protocol_version": "version26",
            "protocol_hash": "4da19387399bd3a5",
            "student_prompt_sha256": (
                "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
            ),
            "base_model_identity": base_identity,
            "reference_policy": reference_policy,
            "experiment_config_sha256": "c" * 64,
            "checkpoint_gate": checkpoint_gate,
        },
    )

    log_history = [
        {
            "step": step,
            "loss": 0.0,
            "grad_norm": 0.04 + step / 1000,
            "learning_rate": 8e-7,
            "rollout/qlora_layers_synced": 252.0,
            "sampling/vllm_importance/log_ratio_abs_mean": 0.01,
            "sampling/vllm_importance/applied_ratio_mean": 1.0,
            "sampling/vllm_importance/cap_exceeded_fraction": 0.0,
        }
        for step in range(1, 6)
    ]
    _write_json(
        checkpoint / "trainer_state.json",
        {"global_step": 5, "max_steps": 20, "log_history": log_history},
    )

    rollout_rows: list[dict[str, Any]] = []
    for policy_step in range(5):
        for group_offset in range(30):
            example_index = policy_step * 30 + group_offset
            task = tasks[example_index]
            for sample_index in range(8):
                correct = sample_index < 4
                rollout_rows.append(
                    {
                        "protocol_version": "version26",
                        "protocol_hash": "4da19387399bd3a5",
                        "environment_implementation": "atomic-v26-isolated-v1",
                        "example_index": example_index,
                        "trajectory_id": (
                            f"rl_{example_index}_sample_{sample_index}"
                        ),
                        "db_id": task["db_id"],
                        "question": task["question"],
                        "gold_sql": task["gold_sql"],
                        "turns": [{"parsed": {"tool": "answer"}}],
                        "correct": correct,
                        "legal": True,
                        "steps": 1,
                        "errors": [],
                        "failure_type": None if correct else "wrong_answer",
                        "error_events": [],
                        "result_reward": {
                            "profile": "binary",
                            "correct": correct,
                            "executable_terminal": True,
                            "value": float(correct),
                        },
                        "policy_global_step": policy_step,
                        "policy_synced_global_step": policy_step,
                        "policy_micro_step": policy_step * 30 + group_offset,
                    }
                )
    _write_jsonl(run_dir / "rollouts.jsonl", rollout_rows)
    return run_dir, tasks_path, tasks_manifest_path, initial, rollout_rows


def _audit(paths: tuple[Path, Path, Path, Path, list[dict[str, Any]]]) -> dict[str, Any]:
    return step5.audit(paths[0], paths[1], paths[2], paths[3])


def _cli_args(
    paths: tuple[Path, Path, Path, Path, list[dict[str, Any]]], output: Path
) -> list[str]:
    return [
        "--run-dir",
        str(paths[0]),
        "--tasks",
        str(paths[1]),
        "--tasks-manifest",
        str(paths[2]),
        "--initial-adapter",
        str(paths[3]),
        "--output",
        str(output),
    ]


def _rewrite_rollouts(
    paths: tuple[Path, Path, Path, Path, list[dict[str, Any]]]
) -> None:
    _write_jsonl(paths[0] / "rollouts.jsonl", paths[4])


def _exclude_generation_length(row: dict[str, Any]) -> None:
    row["failure_type"] = "generation_length"
    row["generation_truncation"] = {
        "finish_reason": "length",
        "turn_index": 0,
    }
    row["optimization_exclusion"] = "nonsemantic_runtime_failure"
    row["result_reward"]["value"] = 0.0


def test_complete_step5_fixture_passes_all_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    result = _audit(paths)

    assert result["status"] == {
        "outcome": "pass",
        "passes": True,
        "continuation_admitted": True,
        "exit_code": 0,
    }
    assert all(result["checks"].values())
    assert result["observed"]["rollouts"] == 1200
    assert result["observed"]["unique_tasks_in_first_five_steps"] == 150
    assert result["observed"]["mixed_groups_by_step"] == [30] * 5
    assert result["precision"]["passes"]
    assert result["movement"]["lora_module_count"] == 252


def test_cli_writes_immutable_receipt_and_verify_recomputes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "receipts/step5.json"
    args = _cli_args(paths, output)

    assert step5.main(args) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == step5.SCHEMA_VERSION
    assert receipt["status"]["continuation_admitted"] is True
    assert step5.main(args + ["--verify-existing"]) == 0
    assert step5.main(args) == 2

    receipt["status"]["passes"] = False
    _write_json(output, receipt)
    assert step5.main(args + ["--verify-existing"]) == 2


def test_verify_existing_is_stable_after_training_continues_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = paths[0] / "step5_gate.json"
    args = _cli_args(paths, output)
    assert step5.main(args) == 0

    # A legal continuation grows the rollout log and creates later checkpoints.
    # A later process also records the explicit resume path in run_manifest.
    with (paths[0] / "rollouts.jsonl").open("a", encoding="utf-8") as target:
        for row in paths[4][:240]:
            continued = dict(row)
            continued["policy_global_step"] = 5
            continued["policy_synced_global_step"] = 5
            target.write(json.dumps(continued, sort_keys=True) + "\n")
    (paths[0] / "checkpoint-6").mkdir()
    manifest_path = paths[0] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resume_from_checkpoint"] = str(paths[0] / "checkpoint-5")
    _write_json(manifest_path, manifest)

    assert step5.main(args + ["--verify-existing"]) == 0


def test_malformed_input_is_exit2_with_fixed_receipt_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    (paths[0] / "rollouts.jsonl").write_text("{bad json\n", encoding="utf-8")
    output = tmp_path / "step5-input-error.json"

    assert step5.main(_cli_args(paths, output)) == 2
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert set(receipt) == {
        "schema_version",
        "auditor_sha256",
        "contract",
        "inputs",
        "artifact_sha256",
        "checks",
        "observed",
        "precision",
        "movement",
        "issues",
        "status",
    }
    assert receipt["status"]["outcome"] == "input_error"
    assert receipt["status"]["exit_code"] == 2


def test_gate_failure_is_exit1_and_preserves_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    state_path = paths[0] / "checkpoint-5/trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["log_history"][2][
        "sampling/vllm_importance/applied_ratio_mean"
    ] = 1.051
    _write_json(state_path, state)
    output = tmp_path / "step5-gate-failure.json"

    assert step5.main(_cli_args(paths, output)) == 1
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"]["outcome"] == "gate_failed"
    assert receipt["status"]["exit_code"] == 1
    assert not receipt["checks"]["importance_applied_mean_in_range"]
    assert receipt["observed"]["trainer_metrics"][2][
        "importance_applied_ratio_mean"
    ] == 1.051


@pytest.mark.parametrize(
    ("metric", "value", "failed_check"),
    [
        ("grad_norm", 1.0001, "positive_bounded_gradients"),
        (
            "sampling/vllm_importance/log_ratio_abs_mean",
            0.0501,
            "importance_log_ratio_abs_mean_bounded",
        ),
    ],
)
def test_optimizer_health_thresholds_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
    value: float,
    failed_check: str,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    state_path = paths[0] / "checkpoint-5/trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["log_history"][0][metric] = value
    _write_json(state_path, state)

    result = _audit(paths)
    assert not result["status"]["passes"]
    assert not result["checks"][failed_check]


def test_generation_length_boundary_is_inclusive_and_37th_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    # Exclude one incorrect sample in each of 36 groups.  Every group stays
    # mixed, and exactly 1164/1200 trajectories remain eligible.
    for group in range(36):
        _exclude_generation_length(paths[4][group * 8 + 7])
    _rewrite_rollouts(paths)
    result = _audit(paths)
    assert result["status"]["passes"]
    assert result["observed"]["eligible"] == 1164
    assert result["observed"]["generation_length_exclusions"] == 36

    _exclude_generation_length(paths[4][36 * 8 + 7])
    _rewrite_rollouts(paths)
    result = _audit(paths)
    assert not result["status"]["passes"]
    assert not result["checks"]["eligible_at_least_97_percent"]
    assert not result["checks"]["generation_length_exclusions_at_most_3_percent"]


def test_structured_state_preserving_timeout_is_allowed_but_bad_evidence_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    row = paths[4][0]
    event = {
        "error_type": "timeout_error",
        "error_code": "tool_execution_timeout",
        "state_before_hash": "same-state",
        "state_after_hash": "same-state",
        "details": {"state_preserved": True},
    }
    row["error_events"] = [event]
    row["optimization_exclusion"] = "nonsemantic_runtime_failure"
    row["result_reward"]["value"] = 0.0
    _rewrite_rollouts(paths)
    result = _audit(paths)
    assert result["status"]["passes"]
    assert result["observed"]["timeout_trajectories"] == 1
    assert result["observed"]["structured_timeout_events"] == 1

    event["details"]["state_preserved"] = False
    _rewrite_rollouts(paths)
    result = _audit(paths)
    assert not result["status"]["passes"]
    assert not result["checks"][
        "structured_timeouts_are_excluded_and_state_preserving"
    ]


def test_mixed_group_and_unique_task_gates_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    # Leave only 29 mixed groups, while retaining at least one in every step.
    keep_mixed = {0, 30, 60, 90, 120, *range(121, 145)}
    assert len(keep_mixed) == 29
    for group in range(150):
        if group in keep_mixed:
            continue
        for row in paths[4][group * 8 : (group + 1) * 8]:
            row["correct"] = True
            row["failure_type"] = None
            row["result_reward"]["correct"] = True
            row["result_reward"]["value"] = 1.0
    _rewrite_rollouts(paths)
    result = _audit(paths)
    assert not result["checks"]["at_least_30_mixed_groups"]
    assert result["checks"]["at_least_one_mixed_group_per_step"]

    # Copy a complete K=8 task group over another group in step 1.  The row
    # count is unchanged, but the audited five-step prefix has only 149 tasks.
    paths = _fixture(tmp_path / "second", monkeypatch)
    source_group = [dict(row) for row in paths[4][0:8]]
    paths[4][8:16] = source_group
    _rewrite_rollouts(paths)
    result = _audit(paths)
    assert not result["checks"]["five_blocks_are_30_by_k8"]
    assert not result["checks"]["first_five_blocks_have_150_unique_tasks"]


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("generation_oom", "no_generation_oom"),
        ("context_overflow", "no_context_overflow"),
        ("tokenization_warning", "no_tokenization_warning"),
        ("process_reward", "binary_terminal_reward_only"),
    ],
)
def test_forbidden_runtime_or_credit_evidence_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    failed_check: str,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    row = paths[4][0]
    if mutation in {"generation_oom", "context_overflow"}:
        row["failure_type"] = mutation
        row["optimization_exclusion"] = "nonsemantic_runtime_failure"
        row["result_reward"]["value"] = 0.0
    elif mutation == "tokenization_warning":
        row["rollout_tokenization_warning"] = True
    else:
        row["process_reward"] = {"total_reward": 1.0}
    _rewrite_rollouts(paths)

    result = _audit(paths)
    assert not result["status"]["passes"]
    assert not result["checks"][failed_check]


def test_precision_and_both_movement_views_are_mandatory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    bad_precision = _precision_pass(paths[0] / "checkpoint-5")
    bad_precision["passes"] = False
    bad_precision["non_fp32_adam_moments"] = [
        {"state": "exp_avg", "dtype": "torch.bfloat16"}
    ]
    monkeypatch.setattr(step5, "_checkpoint_precision", lambda _: bad_precision)
    result = _audit(paths)
    assert not result["checks"]["checkpoint_fp32_trainables_and_adam"]

    monkeypatch.setattr(step5, "_checkpoint_precision", _precision_pass)

    def zero_effective(initial: Path, checkpoint: Path) -> dict[str, Any]:
        value = _movement_pass(initial, checkpoint)
        value["effective_lora"][str(checkpoint)]["update_norm"] = 0.0
        return value

    monkeypatch.setattr(step5, "_checkpoint_movement", zero_effective)
    result = _audit(paths)
    assert not result["checks"]["checkpoint5_raw_and_effective_lora_moved"]
