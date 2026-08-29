from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.rl.diagnostics import (
    audit_representative600_onepass_training as completion_audit,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_policy() -> dict[str, object]:
    initial = completion_audit.EXPECTED_INITIAL_ADAPTER_SHA256
    return {
        "schema_version": "trl-frozen-reference-policy-v1",
        "enabled": False,
        "kl_beta": 0.0,
        "adapter_name": None,
        "adapter_path": None,
        "adapter_sha256": None,
        "expected_adapter_sha256": initial,
        "equals_initial_adapter": True,
        "load_audit": None,
        "after_trainer_init_audit": None,
    }


def _experiment_config() -> dict[str, object]:
    return {
        "schema_version": "table-agent-rl-experiment-v1",
        "experiment_name": (
            "qwen3_8b_atomic_v26_vanilla_grpo_representative600_onepass"
        ),
        "admission_status": "allowed_result_only_control",
        "reward_type": "result",
        "result_reward_profile": "binary",
        "policy_reduction": "trajectory_token_mean",
        "expected_records": 600,
        "process_reward_config": None,
        "process_loss": True,
        "process_admission_policy": "counterfactual-completeness",
        "trainable_part": "all",
        "rank_loss": {"enabled": False, "coefficient": 0.0, "beta": 0.1},
        "optimizer": {
            "name": "adamw_torch",
            "learning_rate": 8e-7,
            "weight_decay": 0.1,
            "steps": 20,
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
        "rollout": {
            "prompts_per_update": 30,
            "group_size": 8,
            "max_agent_steps": 30,
            "max_new_tokens": 2048,
            "max_context_tokens": 16384,
            "history_turns": 4,
            "temperature": 0.8,
            "top_p": 1.0,
            "top_k": 0,
            "enable_thinking": True,
        },
        "runtime_contract": {
            "runtime_root": completion_audit.EXPECTED_RUNTIME_ROOT,
            "runtime_content_tree_sha256": completion_audit.EXPECTED_RUNTIME_TREE_SHA256,
            "protocol_version": completion_audit.EXPECTED_PROTOCOL_VERSION,
            "protocol_hash": completion_audit.EXPECTED_PROTOCOL_HASH,
            "student_prompt_sha256": completion_audit.EXPECTED_STUDENT_PROMPT_SHA256,
            "initial_adapter_sha256": completion_audit.EXPECTED_INITIAL_ADAPTER_SHA256,
            "reference_adapter_sha256": completion_audit.EXPECTED_INITIAL_ADAPTER_SHA256,
            "base_model_identity": completion_audit.EXPECTED_BASE_MODEL_IDENTITY,
        },
    }


def _write_checkpoint(path: Path, step: int, *, primary: bool = False) -> None:
    path.mkdir(parents=True)
    adapter = b"primary adapter\n" if primary else f"adapter {step}\n".encode()
    (path / "adapter_model.safetensors").write_bytes(adapter)
    _write_json(path / "adapter_config.json", {"base_model_name_or_path": "synthetic"})
    (path / "optimizer.pt").write_bytes(f"optimizer {step}\n".encode())
    (path / "scheduler.pt").write_bytes(f"scheduler {step}\n".encode())
    (path / "rng_state.pth").write_bytes(f"rng {step}\n".encode())
    (path / "training_args.bin").write_bytes(b"training args\n")
    state: dict[str, object] = {
        "global_step": step,
        "max_steps": 20,
        "log_history": [],
    }
    if primary:
        state["log_history"] = [
            {
                "step": metric_step,
                "grad_norm": 0.01 + metric_step / 1000,
                "loss": 0.0,
                "learning_rate": 8e-7,
                "rollout/episodes": 240,
                "rollout/qlora_layers_synced": 252.0,
                "sampling/vllm_importance/log_ratio_abs_mean": 0.01,
                "sampling/vllm_importance/applied_ratio_mean": 1.0,
                "sampling/vllm_importance/cap_exceeded_fraction": 0.0,
            }
            for metric_step in range(1, 21)
        ]
    _write_json(path / "trainer_state.json", state)


def _synthetic_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, str]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tasks_path = tmp_path / "representative600.jsonl"
    tasks: list[dict[str, object]] = [
        {
            "example_id": f"bird_train_{index:05d}",
            "example_index": index,
            "db_id": f"db_{index % 69}",
            "question": f"question {index}",
            "gold_sql": f"SELECT {index}",
        }
        for index in range(600)
    ]
    _write_jsonl(tasks_path, tasks)
    monkeypatch.setattr(completion_audit, "EXPECTED_TASKS_SHA256", _sha(tasks_path))

    cohort_path = tmp_path / "representative600.manifest.json"
    _write_json(
        cohort_path,
        {
            "schema_version": "bird-train-vanilla-grpo-cohort-v1",
            "status": "frozen_training_cohort",
            "selection": {
                "count": 600,
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
            "output": {
                "sha256": _sha(tasks_path),
                "records": 600,
                "unique_databases": 69,
                "task_ids_in_frozen_order": [row["example_id"] for row in tasks],
            },
            "acceptance_gates": {
                "record_count_exact": True,
                "task_ids_unique": True,
                "representative": True,
            },
            "all_acceptance_gates_passed": True,
            "gold_visibility": (
                "gold SQL remains Harness-only and is never rendered to the actor"
            ),
        },
    )
    monkeypatch.setattr(
        completion_audit, "EXPECTED_COHORT_MANIFEST_SHA256", _sha(cohort_path)
    )

    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("synthetic: true\n", encoding="utf-8")
    monkeypatch.setattr(completion_audit, "EXPECTED_CONFIG_SHA256", _sha(config_path))

    snapshot = run_dir / "implementation_source_snapshot"
    implementation_files: dict[str, str] = {}
    for relative in sorted(completion_audit.REQUIRED_IMPLEMENTATION_FILES):
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic {relative}\n", encoding="utf-8")
        implementation_files[relative] = _sha(path)

    reference_policy = _reference_policy()
    runtime_identity = completion_audit._expected_runtime_identity()
    checkpoint_gate = {
        "schema_version": "trl-synchronous-checkpoint-gate-v1",
        "step": 5,
        "script_relative_path": (
            "src/rl/diagnostics/audit_vanilla_grpo_train600_step5.py"
        ),
        "script_sha256": implementation_files[
            "src/rl/diagnostics/audit_vanilla_grpo_train600_step5.py"
        ],
        "receipt": str((run_dir / "step5_gate.json").resolve()),
        "tasks_manifest": str(cohort_path.resolve()),
        "tasks_manifest_sha256": _sha(cohort_path),
        "execution": "synchronous_on_save_before_next_optimizer_step",
        "resume_policy": (
            "checkpoint_at_or_after_gate_requires_verified_receipt"
        ),
    }
    run_manifest = {
        **completion_audit.EXPECTED_RECORDED_PACKAGE_VERSIONS,
        "protocol_version": completion_audit.EXPECTED_PROTOCOL_VERSION,
        "protocol_hash": completion_audit.EXPECTED_PROTOCOL_HASH,
        "student_prompt_sha256": completion_audit.EXPECTED_STUDENT_PROMPT_SHA256,
        "tool_schema_sha256": completion_audit.EXPECTED_TOOL_SCHEMA_SHA256,
        "initial_adapter_sha256": completion_audit.EXPECTED_INITIAL_ADAPTER_SHA256,
        "adapter_path": completion_audit.EXPECTED_INITIAL_ADAPTER_PATH,
        "examples_json_sha256": _sha(tasks_path),
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
        "experiment_config_sha256": _sha(config_path),
        "base_model_identity": completion_audit.EXPECTED_BASE_MODEL_IDENTITY,
        "reference_policy": reference_policy,
        "runtime_identity_audit": {
            "schema_version": "trl-runtime-identity-audit-v1",
            "pinned": True,
            "expected": runtime_identity,
            "actual": runtime_identity,
            "reference": {
                "enabled": False,
                "expected_adapter_sha256": completion_audit.EXPECTED_INITIAL_ADAPTER_SHA256,
                "actual_adapter_sha256": completion_audit.EXPECTED_INITIAL_ADAPTER_SHA256,
                "equals_initial_adapter": True,
            },
        },
        "runtime_module_audit": {
            "runtime_root": completion_audit.EXPECTED_RUNTIME_ROOT,
            "tool_environment_factory_module": "tool_environment_v26",
            "module_paths": {
                "protocol": f"{completion_audit.EXPECTED_RUNTIME_ROOT}/src/sft/protocol.py",
                "rollout": f"{completion_audit.EXPECTED_RUNTIME_ROOT}/src/eval/rollout.py",
                "executor": f"{completion_audit.EXPECTED_RUNTIME_ROOT}/src/harness/executor.py",
                "tool_schemes": f"{completion_audit.EXPECTED_RUNTIME_ROOT}/src/sft/tool_schemes.py",
            },
        },
        "rollout_settings": {
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
        "experiment_config": _experiment_config(),
        "implementation_source_sha256": implementation_files,
        "checkpoint_gate": checkpoint_gate,
    }
    _write_json(run_dir / "run_manifest.json", run_manifest)
    lock_path = run_dir / "implementation_lock.json"
    _write_json(
        lock_path,
        {
            "schema_version": "trl-implementation-lock-v1",
            "files": implementation_files,
            "initial_adapter_sha256": completion_audit.EXPECTED_INITIAL_ADAPTER_SHA256,
            "protocol_version": completion_audit.EXPECTED_PROTOCOL_VERSION,
            "protocol_hash": completion_audit.EXPECTED_PROTOCOL_HASH,
            "student_prompt_sha256": completion_audit.EXPECTED_STUDENT_PROMPT_SHA256,
            "base_model_identity": completion_audit.EXPECTED_BASE_MODEL_IDENTITY,
            "reference_policy": reference_policy,
            "experiment_config_sha256": _sha(config_path),
            "checkpoint_gate": checkpoint_gate,
        },
    )

    step5_report = {
        "schema_version": "vanilla-grpo-train600-step5-audit-v1",
        "auditor_sha256": checkpoint_gate["script_sha256"],
        "contract": {"frozen": True},
        "inputs": {"tasks": {"sha256": _sha(tasks_path)}},
        "artifact_sha256": {"rollouts": "a" * 64},
        "checks": {"all_stable_step5_checks": True},
        "observed": {"rollouts": 1200},
        "precision": {"passes": True},
        "movement": {"lora_module_count": 252},
        "issues": [],
        "status": {
            "outcome": "pass",
            "passes": True,
            "continuation_admitted": True,
            "exit_code": 0,
        },
    }
    _write_json(run_dir / "step5_gate.json", step5_report)
    monkeypatch.setattr(
        completion_audit.step5_audit,
        "audit",
        lambda *args, **kwargs: step5_report,
    )
    _write_json(
        run_dir / "checkpoint_gate_step5_invocation.json",
        {
            "schema_version": "trl-checkpoint-gate-invocation-v1",
            "step": 5,
            "verify_existing": False,
            "script_relative_path": checkpoint_gate["script_relative_path"],
            "script_sha256": checkpoint_gate["script_sha256"],
            "returncode": 0,
            "stdout": "pass\n",
            "stderr": "",
            "receipt": checkpoint_gate["receipt"],
            "receipt_sha256": _sha(run_dir / "step5_gate.json"),
        },
    )

    rollout_rows: list[dict[str, object]] = []
    for step in range(20):
        for example_index in range(step * 30, (step + 1) * 30):
            for sample_index in range(8):
                correct = sample_index % 2 == 0
                rollout_rows.append(
                    {
                        "trajectory_id": f"{example_index}:{sample_index}",
                        "example_index": example_index,
                        "policy_global_step": step,
                        "policy_synced_global_step": step,
                        "policy_micro_step": step,
                        "protocol_version": completion_audit.EXPECTED_PROTOCOL_VERSION,
                        "protocol_hash": completion_audit.EXPECTED_PROTOCOL_HASH,
                        "tool_scheme": "atomic",
                        "environment_implementation": "atomic-v26-isolated-v1",
                        "context_mode": "rolling-legal-history",
                        "denotation_comparison": "bird-set",
                        "db_id": tasks[example_index]["db_id"],
                        "question": tasks[example_index]["question"],
                        "gold_sql": tasks[example_index]["gold_sql"],
                        "correct": correct,
                        "legal": True,
                        "failure_type": None if correct else "wrong_answer",
                        "error_events": [],
                        "result_reward": {
                            "profile": "binary",
                            "correct": correct,
                            "executable_terminal": True,
                            "value": 1.0 if correct else 0.0,
                        },
                    }
                )
    _write_jsonl(run_dir / "rollouts.jsonl", rollout_rows)
    _write_json(
        run_dir / "training_precision.json",
        {
            "schema_version": "table-agent-trl-training-precision-v1",
            "optimizer_name": "adamw_torch",
            "trainable_parameters": {
                "trainable_tensors": 2,
                "non_fp32_tensors": [],
            },
            "optimizer_state": {
                "moment_tensors": 4,
                "non_fp32_moments": [],
            },
        },
    )
    _write_checkpoint(run_dir / "checkpoint-5", 5)
    _write_checkpoint(run_dir / "diagnostic_checkpoint-15", 15)
    _write_checkpoint(run_dir / "checkpoint-20", 20, primary=True)
    final = run_dir / "final"
    final.mkdir()
    (final / "adapter_model.safetensors").write_bytes(b"primary adapter\n")
    _write_json(final / "adapter_config.json", {"base_model_name_or_path": "synthetic"})
    (final / "training_args.bin").write_bytes(b"training args\n")
    return run_dir, tasks_path, cohort_path, config_path, _sha(lock_path)


def _run_audit(
    fixture: tuple[Path, Path, Path, Path, str]
) -> dict[str, object]:
    run_dir, tasks, cohort, config, lock_sha = fixture
    return completion_audit.audit(run_dir, tasks, cohort, config, lock_sha)


def test_accepts_complete_onepass_fixture_and_excludes_intermediates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run_audit(_synthetic_fixture(tmp_path, monkeypatch))
    assert result["status"] == {"passes": True, "evaluation_admitted": True}
    assert result["contract"]["rollout_rows"] == 4800
    policy = result["checkpoint_policy"]
    assert policy["evaluation_candidates"] == ["final"]
    assert policy["checkpoint_alias"] == "checkpoint-20"
    assert {item["step"] for item in policy["intermediate_checkpoints"]} == {5, 15}
    assert all(
        item["evaluation_admitted"] is False
        for item in policy["intermediate_checkpoints"]
    )


@pytest.mark.parametrize(
    "name",
    ["optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"],
)
def test_rejects_incomplete_checkpoint20(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    fixture = _synthetic_fixture(tmp_path, monkeypatch)
    (fixture[0] / "checkpoint-20" / name).unlink()
    with pytest.raises(ValueError, match=name.replace(".", r"\.")):
        _run_audit(fixture)


def test_rejects_non_onepass_task_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _synthetic_fixture(tmp_path, monkeypatch)
    path = fixture[0] / "rollouts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["example_index"] = 0
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="step 19 is not 30 tasks x K8"):
        _run_audit(fixture)


def test_rejects_policy_sync_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _synthetic_fixture(tmp_path, monkeypatch)
    path = fixture[0] / "rollouts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[241]["policy_synced_global_step"] = 0
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="policy_synced_global_step mismatch at step 1"):
        _run_audit(fixture)


def test_rejects_process_credit_in_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _synthetic_fixture(tmp_path, monkeypatch)
    path = fixture[0] / "rollouts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["process_reward"] = {"steps": []}
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="contains non-vanilla credit"):
        _run_audit(fixture)


def test_rejects_rank_loss_enabled_in_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _synthetic_fixture(tmp_path, monkeypatch)
    path = fixture[0] / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["rank_loss_coefficient"] = 1.0
    _write_json(path, manifest)
    with pytest.raises(ValueError, match="rank_loss_coefficient mismatch"):
        _run_audit(fixture)


def test_rejects_final_adapter_different_from_checkpoint20(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _synthetic_fixture(tmp_path, monkeypatch)
    (fixture[0] / "final/adapter_model.safetensors").write_bytes(b"posthoc model\n")
    with pytest.raises(ValueError, match="adapter weights differ"):
        _run_audit(fixture)


def test_rejects_checkpoint_after_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _synthetic_fixture(tmp_path, monkeypatch)
    (fixture[0] / "checkpoint-5").rename(fixture[0] / "checkpoint-21")
    with pytest.raises(ValueError, match="checkpoint step out of range: 21"):
        _run_audit(fixture)


def test_rejects_transient_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _synthetic_fixture(tmp_path, monkeypatch)
    (fixture[0] / "status.next").write_text("not committed\n")
    with pytest.raises(ValueError, match="transient or symlink artifacts remain"):
        _run_audit(fixture)


def test_rejects_unpinned_implementation_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _synthetic_fixture(tmp_path, monkeypatch)
    run_dir, tasks, cohort, config, _ = fixture
    with pytest.raises(ValueError, match="implementation lock SHA mismatch"):
        completion_audit.audit(run_dir, tasks, cohort, config, "0" * 64)


def test_rejects_missing_step5_gate_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _synthetic_fixture(tmp_path, monkeypatch)
    (fixture[0] / "step5_gate.json").unlink()
    with pytest.raises(ValueError, match="step5_gate"):
        _run_audit(fixture)


def test_rejects_forged_or_mutated_step5_gate_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _synthetic_fixture(tmp_path, monkeypatch)
    receipt_path = fixture[0] / "step5_gate.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["checks"] = {"all_stable_step5_checks": False}
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="stable full recomputation"):
        _run_audit(fixture)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("task_binding", "frozen-task binding"),
        ("false_zero_reward", "binary reward/exclusion mismatch"),
        ("context_overflow", "forbidden runtime failure"),
        ("unknown_exclusion", "unknown optimization exclusion"),
        ("unexcluded_generation_length", "was not excluded"),
        ("importance_mismatch", "importance log-ratio mismatch"),
    ],
)
def test_rejects_bad_post_step5_rollout_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    fixture = _synthetic_fixture(tmp_path, monkeypatch)
    path = fixture[0] / "rollouts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = rows[1200]
    if mutation == "task_binding":
        row["question"] = "mutated"
    elif mutation == "false_zero_reward":
        row["correct"] = True
        row["result_reward"]["correct"] = True
        row["result_reward"]["value"] = 0.0
    elif mutation == "context_overflow":
        row["failure_type"] = "context_overflow"
        row["optimization_exclusion"] = "nonsemantic_runtime_failure"
    else:
        if mutation == "unknown_exclusion":
            row["optimization_exclusion"] = "mystery"
        elif mutation == "unexcluded_generation_length":
            row["failure_type"] = "generation_length"
            row["generation_truncation"] = {"finish_reason": "length"}
        else:
            state_path = fixture[0] / "checkpoint-20/trainer_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["log_history"][5][
                "sampling/vllm_importance/log_ratio_abs_mean"
            ] = 999.0
            _write_json(state_path, state)
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match=message):
        _run_audit(fixture)


def test_rejects_non_synchronous_save_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _synthetic_fixture(tmp_path, monkeypatch)
    path = fixture[0] / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["save_steps"] = 5
    _write_json(path, manifest)
    with pytest.raises(ValueError, match="save_steps mismatch"):
        _run_audit(fixture)


def test_cli_failure_writes_atomic_fail_closed_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _synthetic_fixture(tmp_path, monkeypatch)
    run_dir, tasks, cohort, config, lock_sha = fixture
    (run_dir / "checkpoint-20/optimizer.pt").unlink()
    output = tmp_path / "audit.json"
    code = completion_audit.main(
        [
            "--run-dir",
            str(run_dir),
            "--tasks",
            str(tasks),
            "--cohort-manifest",
            str(cohort),
            "--experiment-config",
            str(config),
            "--expected-implementation-lock-sha256",
            lock_sha,
            "--output",
            str(output),
        ]
    )
    assert code == 2
    result = json.loads(output.read_text())
    assert result["status"]["passes"] is False
    assert result["status"]["evaluation_admitted"] is False
    assert not list(output.parent.glob(f".{output.name}.*"))
    frozen = output.read_bytes()
    assert completion_audit.main(
        [
            "--run-dir",
            str(run_dir),
            "--tasks",
            str(tasks),
            "--cohort-manifest",
            str(cohort),
            "--experiment-config",
            str(config),
            "--expected-implementation-lock-sha256",
            lock_sha,
            "--output",
            str(output),
        ]
    ) == 2
    assert output.read_bytes() == frozen
