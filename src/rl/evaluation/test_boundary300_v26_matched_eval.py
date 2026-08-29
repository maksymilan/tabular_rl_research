from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from src.rl.evaluation import boundary300_v26_matched_eval as boundary
from src.rl.evaluation import formal_v26_matched_eval as formal


CONTRACT_PATH = (
    Path(__file__).resolve().parent
    / "qwen3_8b_v26_boundary300_formal_matched_contract.json"
)
LAUNCHER = (
    Path(__file__).resolve().parent
    / "run_qwen3_8b_v26_boundary300_formal_matched_eval.sh"
)


def load_template() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inert_paths() -> dict[str, str]:
    return {
        "boundary_manifest": (
            "/home/dengyan/tabular_rl_outputs/s2_combined_screen/"
            "boundary_selection/boundary_cohort_manifest.json"
        ),
        "train_tasks": (
            "/home/dengyan/tabular_rl_outputs/s2_combined_screen/"
            "boundary_selection/train300.jsonl"
        ),
        "training_run": (
            "/home/dengyan/tabular_rl_outputs/"
            "qwen3_8b_atomic_v26_boundary300_vanilla_grpo_20260812/"
            "train300_two_pass_seed20260812"
        ),
    }


def test_boundary_contract_is_independent_of_representative_train600() -> None:
    template = load_template()
    boundary.validate_boundary_template(template)
    assert template["schema_version"] == boundary.BOUNDARY_SCHEMA
    assert template["training_final"]["expected_manifest"]["records"] == 300
    assert template["training_final"]["expected_manifest"]["expected_records"] == 300
    assert template["training_final"]["experiment_config_sha256"] == (
        "f03674d0fe693442df4f60cf948279b1bc46c3510a43a9320e77c8200b4923c7"
    )
    assert "train600" not in json.dumps(template, sort_keys=True)
    assert template["host_paths"]["training_run"] == boundary.PATH_PLACEHOLDER


def test_dynamic_binding_is_explicit_and_injects_train300_hash() -> None:
    template = load_template()
    bindings = {field: f"{index + 1:064x}" for index, field in enumerate(boundary.SHA_FIELDS)}
    paths = inert_paths()
    bound = boundary.bind_contract(template, bindings, paths)
    assert bound["schema_version"] == boundary.GENERIC_SCHEMA
    assert bound["boundary_training"]["dynamic_sha256"] == bindings
    assert (
        bound["training_final"]["expected_manifest"]["examples_json_sha256"]
        == bindings["train300"]
    )
    formal.validate_contract_shape(bound)

    missing = dict(bindings)
    missing.pop("train300")
    with pytest.raises(ValueError, match="supplied dynamic binding fields"):
        boundary.bind_contract(template, missing, paths)
    uppercase = dict(bindings)
    uppercase["train300"] = "A" * 64
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        boundary.bind_contract(template, uppercase, paths)

    representative = dict(paths)
    representative["training_run"] = (
        "/home/dengyan/tabular_rl_outputs/"
        "qwen3_8b_atomic_v26_vanilla_grpo_20260812/train600_seed20260812"
    )
    with pytest.raises(ValueError, match="boundary300 training parent"):
        boundary.bind_contract(template, bindings, representative)


def make_boundary_fixture(tmp_path: Path) -> tuple[dict, Path, dict[str, str]]:
    template = load_template()
    selection_root = tmp_path / "outputs"
    selection_dir = selection_root / "s2_combined_screen" / "boundary_selection"
    train_path = selection_dir / "train300.jsonl"
    training_run = tmp_path / "boundary300_training" / "train300_two_pass_seed20260812"
    rows = [
        {
            "example_id": f"bird_train_{index:05d}",
            "example_index": index,
            "db_id": f"db_{index % 60}",
            "question": f"question {index}",
        }
        for index in range(300)
    ]
    train_path.parent.mkdir(parents=True)
    train_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    train_sha = sha(train_path)
    selection = {
        "schema_version": "policy-boundary-grpo-cohort-v1",
        "status": "frozen_boundary_training_cohort",
        "screen_pools": [{"stage": "S1"}, {"stage": "S2"}],
        "selection": {"screen_stage": "S1+S2"},
        "contract": {
            "boundary_records": 332,
            "train_records": 300,
            "validation_records": 32,
            "formal_training": {
                "optimizer_updates": 20,
                "prompts_per_update": 30,
                "group_size": 8,
                "train_passes": 2,
                "prompt_appearances": 600,
                "fresh_online_trajectories": 4800,
                "sampler": "trl-0.29-repeat-sampler-v1",
                "shuffle_dataset": True,
                "data_seed": 20260812,
                "task_order": "two deterministic data-seed shuffled passes",
                "per_pass_coverage": "each train300 identity exactly once",
                "reward_mode": "result-only",
                "result_reward_profile": "binary",
            },
            "primary_checkpoint": "final-step20-only",
        },
        "outputs": {
            "train": {"path": str(train_path), "sha256": train_sha, "records": 300}
        },
        "task_ids": {"train": [row["example_id"] for row in rows]},
    }
    selection_path = selection_dir / "boundary_cohort_manifest.json"
    write_json(selection_path, selection)

    implementation_files = {"src/rl/pinned.py": "1" * 64}
    run_manifest_path = training_run / "run_manifest.json"
    write_json(
        run_manifest_path,
        {
            "examples_json_sha256": train_sha,
            "implementation_source_sha256": implementation_files,
        },
    )
    lock_path = training_run / "implementation_lock.json"
    write_json(
        lock_path,
        {
            "schema_version": "trl-implementation-lock-v1",
            "files": implementation_files,
        },
    )
    adapter_payload = b"unique boundary300 step20 adapter"
    for path in (
        training_run / "checkpoint-20/adapter_model.safetensors",
        training_run / "final/adapter_model.safetensors",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(adapter_payload)

    bindings = {
        "boundary_manifest": sha(selection_path),
        "train300": train_sha,
        "training_run_manifest": sha(run_manifest_path),
        "training_implementation_lock": sha(lock_path),
        "final_checkpoint20_adapter": hashlib.sha256(adapter_payload).hexdigest(),
    }
    local_paths = {
        "boundary_manifest": str(selection_path),
        "train_tasks": str(train_path),
        "training_run": str(training_run),
    }
    contract = boundary.bind_contract(template, bindings, inert_paths())
    contract["host_paths"].update(local_paths)
    contract["boundary_training"]["allowed_artifact_roots"] = {
        "selection_runs": str(selection_root),
        "training_run_parent": str(training_run.parent),
    }
    return contract, training_run, bindings


def test_boundary_artifacts_bind_manifest_locks_and_unique_step20(tmp_path: Path) -> None:
    contract, training_run, bindings = make_boundary_fixture(tmp_path)
    report = boundary.verify_boundary_artifacts(contract, training_run)
    assert report["binding_status"] == "verified"
    assert report["observed_sha256"]["final_adapter"] == bindings[
        "final_checkpoint20_adapter"
    ]
    assert report["observed_sha256"]["checkpoint20_adapter"] == bindings[
        "final_checkpoint20_adapter"
    ]

    (training_run / "final/adapter_model.safetensors").write_bytes(b"drift")
    with pytest.raises(ValueError, match="final adapter SHA-256"):
        boundary.verify_boundary_artifacts(contract, training_run)


def test_boundary_artifacts_reject_any_secondary_checkpoint(tmp_path: Path) -> None:
    contract, training_run, _ = make_boundary_fixture(tmp_path)
    (training_run / "checkpoint-10").mkdir()
    with pytest.raises(ValueError, match="sole checkpoint directory"):
        boundary.verify_boundary_artifacts(contract, training_run)


def test_full_dev_decode_runtime_and_promotion_gate_are_frozen() -> None:
    template = load_template()
    assert template["arms"] == ["sft1", "final"]
    assert template["primary_candidate"] == "final"
    evaluation = template["evaluation"]
    serving = template["serving"]
    assert evaluation["questions"] == 1534
    assert evaluation["workers"] == evaluation["max_inflight_requests"] == 24
    assert evaluation["temperature"] == 0.0
    assert evaluation["top_p"] == 1.0
    assert evaluation["max_tokens"] == 2048
    assert evaluation["enable_thinking"] is True
    assert serving["max_model_len"] == 16384
    assert serving["max_num_seqs"] == 24
    assert template["promotion_gate"] == {
        "expected_count": 1534,
        "minimum_accuracy_gain_percentage_points": 1.0,
        "exact_mcnemar_alpha": 0.05,
        "legal_nonregression_required": True,
    }


def test_launcher_defaults_to_dry_run_and_has_no_remote_or_preemption_actions() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "MODE=${1:-dry-run}" in text
    assert "dry-run|preflight|run" in text
    for name in (
        "FROZEN_BOUNDARY_MANIFEST_SHA256",
        "FROZEN_TRAIN300_SHA256",
        "FROZEN_TRAIN_RUN_MANIFEST_SHA256",
        "FROZEN_TRAIN_IMPLEMENTATION_LOCK_SHA256",
        "FROZEN_FINAL_CHECKPOINT20_ADAPTER_SHA256",
        "BOUNDARY_MANIFEST_PATH",
        "TRAIN300_PATH",
        "TRAINING_RUN_PATH",
    ):
        assert name in text
    for forbidden in ("ssh ", "scp ", "pkill", "pgrep", "--resume"):
        assert forbidden not in text
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_launcher_dry_run_is_read_only_and_reports_pending_bindings() -> None:
    completed = subprocess.run(
        [str(LAUNCHER)], check=True, capture_output=True, text=True
    )
    plan = json.loads(completed.stdout)
    assert plan["status"] == "dry_run_waiting_for_explicit_artifact_bindings"
    assert plan["remote_operations_performed"] is False
    assert plan["primary_candidate"] == "final"
    assert plan["questions_per_arm"] == 1534
    assert set(plan["required_explicit_sha256"]) == set(boundary.SHA_FIELDS)
    assert set(plan["required_explicit_paths"]) == set(boundary.PATH_FIELDS)


def test_template_drift_to_representative_arm_fails_closed() -> None:
    template = copy.deepcopy(load_template())
    template["training_final"]["expected_manifest"]["records"] = 600
    with pytest.raises(ValueError, match="train600 contract"):
        boundary.validate_boundary_template(template)
