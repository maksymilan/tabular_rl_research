from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

from src.rl.diagnostics.test_validate_arm_b_training_inputs import _fixture as make_input_fixture
from src.rl.diagnostics.prepare_vanilla_grpo_confirmatory_arm_trigger import (
    DEFAULT_FORBIDDEN_ARM_A_PATHS,
    prepare_trigger,
)
from src.rl.evaluation import arm_b_v26_matched_eval as arm_b
from src.rl.evaluation import formal_v26_matched_eval as formal


HERE = Path(__file__).resolve().parent
OVERLAY_PATH = HERE / "qwen3_8b_v26_arm_b_formal_matched_contract.json"
BASE_PATH = HERE / "qwen3_8b_v26_boundary300_formal_matched_contract.json"
LAUNCHER = HERE / "run_qwen3_8b_v26_arm_b_formal_matched_eval.sh"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _templates() -> tuple[dict, dict]:
    return formal.load_object(OVERLAY_PATH), formal.load_object(BASE_PATH)


def _artifact_fixture(tmp_path: Path) -> tuple[dict, Path, dict[str, str]]:
    paths = make_input_fixture(tmp_path)
    selection_path = paths["selection"]
    train_path = paths["train"]
    validation_path = paths["validation_audit"]
    selection = formal.load_object(selection_path)
    cohort = formal.load_object(Path(selection["source_cohort_manifest"]["path"]))
    cohort_path = Path(selection["source_cohort_manifest"]["path"])

    training_run = tmp_path / "arm_b_training" / "train320_two_pass_k16_seed20260812"
    implementation_files = {"src/rl/pinned.py": "1" * 64}
    run_manifest_path = training_run / "run_manifest.json"
    _write_json(
        run_manifest_path,
        {
            "examples_json_sha256": _sha(train_path),
            "implementation_source_sha256": implementation_files,
        },
    )
    lock_path = training_run / "implementation_lock.json"
    _write_json(
        lock_path,
        {"schema_version": "trl-implementation-lock-v1", "files": implementation_files},
    )
    adapter_payload = b"unique Arm B final checkpoint32 adapter"
    checkpoint_adapter = training_run / "checkpoint-32/adapter_model.safetensors"
    final_adapter = training_run / "final/adapter_model.safetensors"
    for path in (checkpoint_adapter, final_adapter):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(adapter_payload)
    adapter_sha = hashlib.sha256(adapter_payload).hexdigest()

    overlay, base = _templates()
    trigger_path = tmp_path / "arm_b/confirmatory_arm_trigger.json"
    prepare_trigger(
        source_cohort_path=cohort_path,
        expected_source_cohort_sha256=_sha(cohort_path),
        selection_manifest_path=selection_path,
        expected_selection_manifest_sha256=_sha(selection_path),
        train320_path=train_path,
        expected_train320_sha256=_sha(train_path),
        validation64_audit_path=validation_path,
        expected_validation64_audit_sha256=_sha(validation_path),
        output_path=trigger_path,
        forbidden_existing_arm_a_paths=(tmp_path / "absent_arm_a",),
    )
    training_contract_path = training_run / "arm_b_training_contract.json"
    _write_json(
        training_contract_path,
        {
            "schema_version": "qwen3-v26-arm-b-training-contract-v1",
            "status": "completed_primary_final32",
            "arm": "arm_b",
            "inputs": {
                "selection_manifest": {"path": str(selection_path), "sha256": _sha(selection_path)},
                "train320": {"path": str(train_path), "sha256": _sha(train_path)},
                "validation64_audit": {"path": str(validation_path), "sha256": _sha(validation_path)},
                "source_cohort_manifest": {"path": str(cohort_path), "sha256": _sha(cohort_path)},
                "confirmatory_trigger": {"path": str(trigger_path), "sha256": _sha(trigger_path)},
            },
            "training": {
                "records": 320,
                "group_size": 16,
                "prompts_per_update": 20,
                "optimizer_steps": 32,
                "train_passes": 2,
                "prompt_appearances": 640,
                "fresh_online_trajectories": 10240,
                "learning_rate": 8e-7,
                "kl_beta": 0.0,
                "reward_mode": "result-only",
                "result_reward_profile": "binary",
                "experiment_config_sha256": overlay["training_final"]["experiment_config_sha256"],
            },
            "checkpoint_policy": overlay["arm_b_training"]["checkpoint_policy"],
            "outputs": {
                "run_manifest_sha256": _sha(run_manifest_path),
                "implementation_lock_sha256": _sha(lock_path),
                "checkpoint32_adapter_sha256": adapter_sha,
                "final_adapter_sha256": adapter_sha,
            },
        },
    )
    bindings = {
        "selection_manifest": _sha(selection_path),
        "train320": _sha(train_path),
        "validation64_audit": _sha(validation_path),
        "training_run_manifest": _sha(run_manifest_path),
        "training_implementation_lock": _sha(lock_path),
        "arm_b_training_contract": _sha(training_contract_path),
        "final_checkpoint32_adapter": adapter_sha,
        "confirmatory_trigger": _sha(trigger_path),
    }
    contract = arm_b.bind_contract(overlay, base, bindings, arm_b._inert_paths())
    contract["host_paths"].update(
        {
            "boundary_manifest": str(selection_path),
            "train_tasks": str(train_path),
            "training_run": str(training_run),
        }
    )
    contract["arm_b_paths"] = {
        "validation64_audit": str(validation_path),
        "confirmatory_trigger": str(trigger_path),
    }
    contract["arm_b_training"]["allowed_artifact_roots"] = {
        "selection_runs": str(tmp_path),
        "training_run_parent": str(training_run.parent),
    }
    contract["confirmatory_trigger_contract"]["forbidden_existing_arm_a_paths"] = [
        str(tmp_path / "absent_arm_a_training_marker"),
        str(tmp_path / "absent_arm_a_full_dev_marker"),
    ]
    return contract, training_run, bindings


def test_overlay_binds_arm_b_without_inheriting_arm_a_training_semantics() -> None:
    overlay, base = _templates()
    arm_b.validate_overlay(overlay, base)
    bindings = {field: f"{index + 1:064x}" for index, field in enumerate(arm_b.SHA_FIELDS)}
    bound = arm_b.bind_contract(overlay, base, bindings, arm_b._inert_paths())
    assert "boundary_training" not in bound
    assert bound["arm_b_training"]["dynamic_sha256"] == bindings
    assert bound["training_final"]["expected_manifest"]["examples_json_sha256"] == bindings["train320"]
    assert bound["training_final"]["checkpoint_name"] == "checkpoint-32"
    formal.validate_contract_shape(bound)


def test_active_paths_accept_the_canonical_f3_audit_name(tmp_path: Path) -> None:
    overlay, _ = _templates()
    root = tmp_path / "arm_b_screen"
    selection = root / "f2" / "selection" / "arm_b_selection_manifest.json"
    train = selection.with_name("train320.jsonl")
    validation = root / "f3" / "audit" / "arm_b_k16_validation_audit.json"
    trigger = validation.with_name("confirmatory_arm_trigger.json")
    training = tmp_path / "arm_b_training" / "train320_two_pass_k16_seed20260812"
    overlay["arm_b_training"]["allowed_artifact_roots"] = {
        "selection_runs": str(root),
        "training_run_parent": str(training.parent),
    }
    arm_b.validate_paths(
        overlay,
        {
            "selection_manifest": str(selection),
            "train_tasks": str(train),
            "validation64_audit": str(validation),
            "training_run": str(training),
            "confirmatory_trigger": str(trigger),
        },
    )


def test_arm_b_artifacts_bind_validation_trigger_and_sole_final32(tmp_path: Path) -> None:
    contract, training_run, bindings = _artifact_fixture(tmp_path)
    report = arm_b.verify_arm_b_artifacts(contract, training_run)
    assert report["binding_status"] == "verified"
    assert report["observed_sha256"]["final_adapter"] == bindings["final_checkpoint32_adapter"]
    assert report["confirmatory_trigger"]["confirmatory_arm_count"] == 1
    assert report["intermediate_checkpoints_evaluated"] is False


def test_arm_b_eval_rejects_any_intermediate_checkpoint(tmp_path: Path) -> None:
    contract, training_run, _ = _artifact_fixture(tmp_path)
    (training_run / "checkpoint-28").mkdir()
    with pytest.raises(ValueError, match="sole Arm B checkpoint directory"):
        arm_b.verify_arm_b_artifacts(contract, training_run)


def test_arm_b_eval_rejects_trigger_claiming_two_confirmatory_arms(tmp_path: Path) -> None:
    contract, training_run, _ = _artifact_fixture(tmp_path)
    trigger_path = Path(contract["arm_b_paths"]["confirmatory_trigger"])
    trigger = formal.load_object(trigger_path)
    trigger["confirmatory_arm_count"] = 2
    _write_json(trigger_path, trigger)
    contract["arm_b_training"]["dynamic_sha256"]["confirmatory_trigger"] = _sha(trigger_path)
    training_contract_path = training_run / "arm_b_training_contract.json"
    training_contract = formal.load_object(training_contract_path)
    training_contract["inputs"]["confirmatory_trigger"]["sha256"] = _sha(trigger_path)
    _write_json(training_contract_path, training_contract)
    contract["arm_b_training"]["dynamic_sha256"]["arm_b_training_contract"] = _sha(
        training_contract_path
    )
    with pytest.raises(ValueError, match="confirmatory trigger confirmatory_arm_count"):
        arm_b.verify_arm_b_artifacts(contract, training_run)


def test_trigger_preparer_rejects_partial_arm_a_training_directory(tmp_path: Path) -> None:
    paths = make_input_fixture(tmp_path)
    selection = formal.load_object(paths["selection"])
    cohort_path = Path(selection["source_cohort_manifest"]["path"])
    partial = tmp_path / "arm_a_train300_started"
    partial.mkdir()
    with pytest.raises(ValueError, match="Arm A training/evaluation marker already exists"):
        prepare_trigger(
            source_cohort_path=cohort_path,
            expected_source_cohort_sha256=_sha(cohort_path),
            selection_manifest_path=paths["selection"],
            expected_selection_manifest_sha256=_sha(paths["selection"]),
            train320_path=paths["train"],
            expected_train320_sha256=_sha(paths["train"]),
            validation64_audit_path=paths["validation_audit"],
            expected_validation64_audit_sha256=_sha(paths["validation_audit"]),
            output_path=tmp_path / "trigger.json",
            forbidden_existing_arm_a_paths=(partial,),
        )
    assert DEFAULT_FORBIDDEN_ARM_A_PATHS[0].name == "train300_two_pass_seed20260812"


def test_full_dev_order_decode_and_promotion_gate_are_preregistered() -> None:
    overlay, base = _templates()
    plan = arm_b.pending_plan(overlay, formal.verify_implementation(base))
    assert plan["execution_order"] == ["fresh_sft1", "final_checkpoint32"]
    assert plan["questions_per_arm"] == 1534
    assert plan["same_gpu_runtime_concurrency"] is True
    assert plan["runtime_and_decode"] == {
        "workers": 24,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "max_model_len": 16384,
        "enable_thinking": True,
    }
    assert plan["promotion_gate"] == {
        "expected_count": 1534,
        "minimum_accuracy_gain_percentage_points": 1.0,
        "exact_mcnemar_alpha": 0.05,
        "legal_nonregression_required": True,
    }


def test_eval_launcher_is_read_only_by_default_and_pins_all_bindings(tmp_path: Path) -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    assert LAUNCHER.stat().st_mode & stat.S_IXUSR
    env = dict(os.environ, RUN_DIR=str(tmp_path / "must_not_exist"))
    completed = subprocess.run([str(LAUNCHER)], check=True, capture_output=True, text=True, env=env)
    plan = json.loads(completed.stdout)
    assert plan["status"] == "dry_run_waiting_for_explicit_artifact_bindings"
    assert plan["remote_operations_performed"] is False
    assert not (tmp_path / "must_not_exist").exists()
    text = LAUNCHER.read_text(encoding="utf-8")
    for field in (
        "FROZEN_ARM_B_SELECTION_MANIFEST_SHA256",
        "FROZEN_TRAIN320_SHA256",
        "FROZEN_VALIDATION64_AUDIT_SHA256",
        "FROZEN_TRAIN_RUN_MANIFEST_SHA256",
        "FROZEN_TRAIN_IMPLEMENTATION_LOCK_SHA256",
        "FROZEN_ARM_B_TRAINING_CONTRACT_SHA256",
        "FROZEN_FINAL_CHECKPOINT32_ADAPTER_SHA256",
        "FROZEN_CONFIRMATORY_TRIGGER_SHA256",
    ):
        assert field in text
    for forbidden in ("ssh ", "scp ", "pkill", "pgrep", "--resume"):
        assert forbidden not in text


def test_eval_launcher_literal_source_hashes_are_current() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    pins = {
        "EXPECTED_ARM_B_CONTROLLER_SHA256": HERE / "arm_b_v26_matched_eval.py",
        "EXPECTED_FORMAL_CONTROLLER_SHA256": HERE / "formal_v26_matched_eval.py",
        "EXPECTED_WRAPPER_SHA256": HERE / "formal_v26_rollout_passk.py",
        "EXPECTED_OVERLAY_SHA256": OVERLAY_PATH,
        "EXPECTED_COMMON_BASE_SHA256": BASE_PATH,
        "EXPECTED_ANALYZER_SHA256": HERE.parent / "diagnostics/analyze_evaluation_results.py",
        "EXPECTED_GATE_SHA256": HERE.parent / "diagnostics/audit_vanilla_grpo_matched_gate.py",
    }
    for variable, path in pins.items():
        assert f"{variable}={_sha(path)}" in text
    assert re.search(r"__[A-Z0-9_]+__", text) is None
