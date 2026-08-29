from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from src.rl.evaluation import earlystop_mixed180_v26_matched_eval as earlystop
from src.rl.evaluation import formal_v26_matched_eval as formal


CONTRACT = (
    Path(__file__).resolve().parent
    / "qwen3_8b_v26_earlystop_mixed180_formal_matched_contract.json"
)
LAUNCHER = (
    Path(__file__).resolve().parent
    / "run_qwen3_8b_v26_earlystop_mixed180_formal_matched_eval.sh"
)


def load_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_contract_is_isolated_from_600_and_boundary300() -> None:
    contract = load_contract()
    earlystop.validate_contract(contract)
    assert contract["input"]["databases"]["debit_card_specializing"] == {
        "records": 64,
        "sha256": (
            "b3d149ad05746dbbe5116e229e17e18f09c39db43cf117d9ef3441753608b691"
        ),
    }
    generic = earlystop._generic_contract(contract)
    assert generic["training_final"]["checkpoint_name"] == "checkpoint-12"
    assert generic["training_final"]["expected_manifest"]["records"] == 180
    serialized = json.dumps(contract)
    assert "train600" not in serialized
    assert "train300_two_pass" not in serialized


def test_derived_input_hash_is_64_hex_and_rebuilds_frozen_input(
    tmp_path: Path,
) -> None:
    contract = load_contract()
    expected_hash = contract["input"]["derived_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", expected_hash)

    source = formal.PROJECT_ROOT / "data/eval_inputs/bird_dev_20240627.jsonl"
    rows = formal.load_jsonl(source)
    output = tmp_path / "bird_dev_20240627.newgnn.jsonl"
    manifest = tmp_path / "bird_dev_20240627.newgnn.manifest.json"
    prepared = formal.prepare_derived_input(
        contract,
        rows,
        Path(contract["host_paths"]["database_root"]),
        output,
        manifest,
    )

    assert prepared["derived_sha256"] == expected_hash
    assert formal.sha256_file(output) == expected_hash
    assert (
        formal.sha256_file(manifest)
        == contract["input"]["derived_manifest_sha256"]
    )


def test_launcher_pins_corrected_contract() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert (
        f"EXPECTED_CONTRACT_SHA256={formal.sha256_file(CONTRACT)}"
        in source
    )


def test_shared_final_verifier_accepts_parameterized_checkpoint_label(
    tmp_path: Path,
) -> None:
    contract = load_contract()
    contract["host_paths"]["training_run"] = str(tmp_path / "training")
    expected = contract["training_final"]
    assert expected["checkpoint_name"] == "checkpoint-12"
    assert expected["global_step"] == 12


def test_exact_artifact_gate_requires_final_equal_checkpoint12(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_contract()
    training = tmp_path / "training"
    cohort = tmp_path / "cohort.json"
    tasks = tmp_path / "train180.jsonl"
    contract["host_paths"]["training_run"] = str(training)
    contract["earlystop_training"]["cohort_manifest"] = str(cohort)
    contract["earlystop_training"]["train_tasks"] = str(tasks)
    tasks.write_bytes(b"tasks\n")
    cohort_payload = {
        "schema_version": "qwen3-v26-earlystop-mixed-grpo-cohort-v1",
        "status": "frozen_operator_requested_earlystop_mixed180",
        "selection": {"eligible_mixed": 193, "selected": 180},
        "outputs": {
            "train180": {
                "sha256": hashlib.sha256(tasks.read_bytes()).hexdigest(),
                "records": 180,
            }
        },
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
    }
    cohort.write_text(json.dumps(cohort_payload) + "\n")
    monkeypatch.setattr(earlystop, "EXPECTED_TASKS_SHA256", hashlib.sha256(tasks.read_bytes()).hexdigest())
    monkeypatch.setattr(
        earlystop,
        "EXPECTED_COHORT_MANIFEST_SHA256",
        hashlib.sha256(cohort.read_bytes()).hexdigest(),
    )
    implementation_files = {"src/rl/pinned.py": "a" * 64}
    write_json(
        training / "run_manifest.json",
        {
            "base_model_identity": contract["model"]["base_model_identity"],
            "implementation_source_sha256": implementation_files,
        },
    )
    write_json(
        training / "implementation_lock.json",
        {
            "base_model_identity": contract["model"]["base_model_identity"],
            "files": implementation_files,
        },
    )
    write_json(training / "training_precision.json", {})
    for name in ("checkpoint-12", "final"):
        directory = training / name
        directory.mkdir(parents=True)
        (directory / "adapter_model.safetensors").write_bytes(b"same")
        (directory / "adapter_config.json").write_bytes(b"same-config")
    write_json(training / "checkpoint-12/trainer_state.json", {"global_step": 12})
    report = earlystop.verify_earlystop_artifacts(contract)
    assert report["status"] == "verified"

    (training / "final/adapter_model.safetensors").write_bytes(b"drift")
    with pytest.raises(ValueError, match="final/checkpoint-12 adapter weights"):
        earlystop.verify_earlystop_artifacts(contract)


def test_plan_is_fresh_sft1_then_final_only() -> None:
    plan = earlystop.print_plan(load_contract(), {})
    assert plan["evaluation"] == {
        "arms": ["sft1", "final"],
        "order": "fresh SFT1 then final",
        "questions_per_arm": 1534,
        "no_resume": True,
    }
    assert plan["training"]["primary_policy"] == "final equals checkpoint-12"


def test_contract_implementation_pins_are_current() -> None:
    contract = load_contract()
    expected = contract["implementation_sha256"]
    for name, path in formal.implementation_paths().items():
        assert expected[name] == formal.sha256_file(path)
