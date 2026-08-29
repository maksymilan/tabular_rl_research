from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from src.rl.diagnostics import audit_earlystop_mixed180_training as training_audit
from src.rl.diagnostics.summarize_earlystop_pass_pairing import exact_sign_p, summarize


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_audit_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoint-12"
    final = run_dir / "final"
    checkpoint.mkdir(parents=True)
    final.mkdir()

    tasks_path = tmp_path / "train180.jsonl"
    tasks = [{"example_index": index} for index in range(180)]
    tasks_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in tasks),
        encoding="utf-8",
    )
    tasks_sha = _sha(tasks_path)
    monkeypatch.setattr(training_audit, "EXPECTED_TASKS_SHA256", tasks_sha)

    cohort_path = tmp_path / "cohort.json"
    _write_json(
        cohort_path,
        {
            "schema_version": "qwen3-v26-earlystop-mixed-grpo-cohort-v1",
            "status": "frozen_operator_requested_earlystop_mixed180",
            "outputs": {"train180": {"sha256": tasks_sha, "records": 180}},
            "training_contract": {
                "records": 180,
                "group_size": 8,
                "prompts_per_update": 30,
                "optimizer_steps": 12,
                "passes": 2,
                "fresh_online_trajectories": 2880,
                "reward": "binary-result-only",
                "kl_beta": 0.0,
            },
        },
    )
    monkeypatch.setattr(
        training_audit, "EXPECTED_COHORT_MANIFEST_SHA256", _sha(cohort_path)
    )

    base_identity_sha = "b" * 64
    config_sha = "c" * 64
    initial_adapter_sha = "d" * 64
    monkeypatch.setattr(
        training_audit, "EXPECTED_BASE_MODEL_IDENTITY_SHA256", base_identity_sha
    )
    monkeypatch.setattr(training_audit, "EXPECTED_CONFIG_SHA256", config_sha)
    monkeypatch.setattr(
        training_audit, "EXPECTED_INITIAL_ADAPTER_SHA256", initial_adapter_sha
    )
    model_identity = {
        "aggregate_sha256": base_identity_sha,
        "identity_kind": "synthetic-test-fixture",
    }
    snapshot_file = run_dir / "implementation_source_snapshot/src/fixture.py"
    snapshot_file.parent.mkdir(parents=True)
    snapshot_file.write_text("SYNTHETIC = True\n", encoding="utf-8")
    implementation_files = {"src/fixture.py": _sha(snapshot_file)}
    lock_path = run_dir / "implementation_lock.json"
    _write_json(
        lock_path,
        {
            "schema_version": "trl-implementation-lock-v1",
            "base_model_identity": model_identity,
            "files": implementation_files,
        },
    )
    monkeypatch.setattr(
        training_audit, "EXPECTED_IMPLEMENTATION_LOCK_SHA256", _sha(lock_path)
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            "protocol_version": "version26",
            "protocol_hash": "4da19387399bd3a5",
            "student_prompt_sha256": (
                "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
            ),
            "initial_adapter_sha256": initial_adapter_sha,
            "examples_json_sha256": tasks_sha,
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
            "experiment_config_sha256": config_sha,
            "base_model_identity": model_identity,
            "implementation_source_sha256": implementation_files,
        },
    )

    rollout_rows = []
    for pass_index in range(2):
        for local_step in range(6):
            global_step = pass_index * 6 + local_step
            for task_index in range(local_step * 30, (local_step + 1) * 30):
                for sample_index in range(8):
                    rollout_rows.append(
                        {
                            "example_index": task_index,
                            "sample_index": sample_index,
                            "policy_global_step": global_step,
                            "policy_synced_global_step": global_step,
                            "policy_micro_step": global_step,
                        }
                    )
    (run_dir / "rollouts.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rollout_rows),
        encoding="utf-8",
    )
    _write_json(
        run_dir / "training_precision.json",
        {
            "schema_version": "table-agent-trl-training-precision-v1",
            "optimizer_name": "adamw_torch",
            "trainable_parameters": {
                "trainable_tensors": 2,
                "non_fp32_tensors": [],
            },
            "optimizer_state": {"moment_tensors": 4, "non_fp32_moments": []},
        },
    )
    _write_json(
        checkpoint / "trainer_state.json",
        {
            "global_step": 12,
            "max_steps": 12,
            "log_history": [
                {"step": step, "grad_norm": 1.0 + step / 100}
                for step in range(1, 13)
            ],
        },
    )
    adapter = b"synthetic adapter weights\n"
    adapter_config = {"base_model_name_or_path": "synthetic/base"}
    (checkpoint / "adapter_model.safetensors").write_bytes(adapter)
    (final / "adapter_model.safetensors").write_bytes(adapter)
    _write_json(checkpoint / "adapter_config.json", adapter_config)
    _write_json(final / "adapter_config.json", adapter_config)
    (checkpoint / "optimizer.pt").write_bytes(b"synthetic optimizer\n")
    (checkpoint / "scheduler.pt").write_bytes(b"synthetic scheduler\n")
    (checkpoint / "rng_state.pth").write_bytes(b"synthetic rng state\n")
    return run_dir, tasks_path, cohort_path


def test_completion_audit_accepts_complete_synthetic_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, tasks_path, cohort_path = _synthetic_audit_fixture(
        tmp_path, monkeypatch
    )
    result = training_audit.audit(run_dir, tasks_path, cohort_path)
    assert result["status"] == {"passes": True, "evaluation_admitted": True}
    assert result["contract"]["rollout_rows"] == 2880
    assert "checkpoint_optimizer" in result["artifact_sha256"]
    assert "checkpoint_scheduler" in result["artifact_sha256"]
    assert "checkpoint_rng_state" in result["artifact_sha256"]


@pytest.mark.parametrize(
    ("relative_path", "label"),
    [
        ("checkpoint-12/optimizer.pt", "checkpoint_optimizer"),
        ("checkpoint-12/scheduler.pt", "checkpoint_scheduler"),
        ("checkpoint-12/rng_state.pth", "checkpoint_rng_state"),
    ],
)
def test_completion_audit_rejects_each_missing_resume_state_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    label: str,
) -> None:
    run_dir, tasks_path, cohort_path = _synthetic_audit_fixture(
        tmp_path, monkeypatch
    )
    (run_dir / relative_path).unlink()
    with pytest.raises(ValueError, match=rf"invalid {label}"):
        training_audit.audit(run_dir, tasks_path, cohort_path)


def test_completion_audit_rejects_implementation_lock_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, tasks_path, cohort_path = _synthetic_audit_fixture(
        tmp_path, monkeypatch
    )
    lock_path = run_dir / "implementation_lock.json"
    lock_path.write_text(lock_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="implementation lock SHA mismatch"):
        training_audit.audit(run_dir, tasks_path, cohort_path)


def test_exact_sign_p_is_two_sided() -> None:
    assert exact_sign_p(0, 0) == 1.0
    assert exact_sign_p(2, 2) == 1.0
    assert exact_sign_p(10, 0) < 0.01


def test_pass_pairing_reports_task_level_change(tmp_path: Path) -> None:
    rows = []
    for pass_index in range(2):
        for task in range(180):
            for sample in range(8):
                correct = sample < (4 + pass_index) if task < 10 else sample < 4
                rows.append(
                    {
                        "example_index": task,
                        "correct": correct,
                        "legal": True,
                    }
                )
    path = tmp_path / "rollouts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = summarize(path)
    assert result["correct"]["improved_tasks"] == 10
    assert result["correct"]["regressed_tasks"] == 0
    assert result["correct"]["tied_tasks"] == 170
    assert result["correct"]["task_delta_sum"] == 10
    assert result["legal"]["tied_tasks"] == 180
    assert "not a generalization" in result["interpretation"]
