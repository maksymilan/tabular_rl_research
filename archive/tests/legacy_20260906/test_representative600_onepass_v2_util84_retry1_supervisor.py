from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from rl.experiments import (
    supervise_qwen3_8b_atomic_v26_representative600_onepass as v1,
)
from rl.experiments import (
    supervise_qwen3_8b_atomic_v26_representative600_onepass_v2 as v2,
)
from rl.experiments import (
    supervise_qwen3_8b_atomic_v26_representative600_onepass_v2_util84_retry1
    as retry,
)


ROOT = Path(__file__).resolve().parents[3]
V2_CONFIG = (
    ROOT
    / "src/rl/configs/experiments/"
    "qwen3_8b_atomic_v26_vanilla_grpo_representative600_onepass_v2.yaml"
)
CONFIG = (
    ROOT
    / "src/rl/configs/experiments/"
    "qwen3_8b_atomic_v26_vanilla_grpo_representative600_"
    "onepass_v2_util84_retry1.yaml"
)
V2_CONTRACT = (
    ROOT
    / "src/rl/experiments/"
    "qwen3_8b_atomic_v26_representative600_onepass_v2_contract.json"
)
CONTRACT = (
    ROOT
    / "src/rl/experiments/"
    "qwen3_8b_atomic_v26_representative600_"
    "onepass_v2_util84_retry1_contract.json"
)


def _command_contract() -> dict:
    return {
        "paths": copy.deepcopy(retry.EXPECTED_PATHS),
        "resources": {
            "gpus": copy.deepcopy(retry.EXPECTED_GPUS),
            "vllm_port": 8076,
            "vllm_group_port": 51276,
        },
        "runtime": {
            "allocator_conf": "expandable_segments:True",
            "vllm_gpu_memory_utilization": 0.84,
            "max_model_len": retry.EXPECTED_MAX_CONTEXT_TOKENS,
        },
        "expected_implementation_lock_sha256": "a" * 64,
    }


def test_importing_retry_does_not_mutate_frozen_v1_or_v2_modules() -> None:
    assert v1.EXPECTED_EXPERIMENT_NAME.endswith("representative600_onepass")
    assert v2.EXPECTED_EXPERIMENT_NAME.endswith("representative600_onepass_v2")
    assert v2.EXPECTED_PATHS["train_output"].endswith(
        "train600_onepass_v2_seed20260812"
    )
    assert retry._engine is not v2._engine
    assert retry.EXPECTED_EXPERIMENT_NAME.endswith("v2_util84_retry1")


def test_frozen_retry_contract_shape_lock_and_read_only_plan() -> None:
    digest = retry.sha256_file(CONTRACT)
    assert digest == "5a8e85c18d7ba0051c0c1114b1a1a42c87bcc0c3f8f9e2e7ca6bba1325021166"
    contract = retry.load_contract(CONTRACT, digest)
    assert contract["expected_implementation_lock_sha256"] == (
        "7583871e23f9807a2ec4b841266a3511ca5a195a0b3c4b6cfe095b8baafef068"
    )
    assert retry.implementation_lock_sha256(contract) == (
        contract["expected_implementation_lock_sha256"]
    )
    plan = retry.plan_payload(contract)
    assert plan["mode"] == "plan_read_only"
    assert plan["engineering_retry"] == {
        "trigger": "vllm_kv_cache_startup_capacity_failure_before_rollout_or_trainer",
        "changed_field": "runtime.vllm_gpu_memory_utilization",
        "previous_value": 0.82,
        "retry_value": 0.84,
        "scientific_contract": "unchanged",
        "artifact_policy": "fresh_isolated_paths_no_resume_no_overwrite",
    }


def test_retry_config_changes_only_experiment_identity() -> None:
    base = yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))
    candidate = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    retry._validate_experiment_config(CONFIG)
    assert retry.sha256_file(CONFIG) == retry.EXPECTED_CONFIG_SHA256
    assert candidate.pop("experiment_name") == retry.EXPECTED_EXPERIMENT_NAME
    base.pop("experiment_name")
    assert candidate == base


def test_retry_contract_changes_only_runtime_fraction_and_isolated_identity() -> None:
    base = json.loads(V2_CONTRACT.read_text(encoding="utf-8"))
    candidate = retry.load_contract(CONTRACT, retry.sha256_file(CONTRACT))

    base_experiment = copy.deepcopy(base["experiment"])
    candidate_experiment = copy.deepcopy(candidate["experiment"])
    base_experiment.pop("name")
    candidate_experiment.pop("name")
    assert candidate_experiment == base_experiment

    base_runtime = copy.deepcopy(base["runtime"])
    candidate_runtime = copy.deepcopy(candidate["runtime"])
    assert base_runtime.pop("vllm_gpu_memory_utilization") == 0.82
    assert candidate_runtime.pop("vllm_gpu_memory_utilization") == 0.84
    assert candidate_runtime == base_runtime
    assert candidate["resources"] == base["resources"]

    for key in (
        "tasks_sha256",
        "tasks_manifest_sha256",
        "protocol_runtime_tree_sha256",
        "initial_adapter_sha256",
        "initial_adapter_config_sha256",
        "initial_adapter_state_sha256",
        "base_model_identity",
        "python_packages",
    ):
        assert candidate["identities"][key] == base["identities"][key]

    assert set(candidate["identities"]["runtime_source_sha256"]) == (
        set(base["identities"]["runtime_source_sha256"]) | {retry.SELF_SOURCE}
    )
    for key in ("project_root", "run_root", "train_output", "experiment_config"):
        assert candidate["paths"][key] != base["paths"][key]
        assert "util84_retry1" in candidate["paths"][key]


def test_contract_source_closure_matches_exact_local_bytes() -> None:
    contract = retry.load_contract(CONTRACT, retry.sha256_file(CONTRACT))
    source_hashes = contract["identities"]["runtime_source_sha256"]
    assert set(source_hashes) == retry.REQUIRED_RUNTIME_SOURCES
    for relative, expected in source_hashes.items():
        assert retry.sha256_file(ROOT / relative) == expected
    assert contract["identities"]["supervisor_source_sha256"] == (
        retry.sha256_file(Path(retry.__file__).resolve())
    )


def test_retry_vllm_command_changes_only_memory_fraction() -> None:
    command, environment = retry.vllm_command_and_environment(_command_contract())
    assert command[0] == "bash"
    assert environment["MAX_MODEL_LEN"] == "20480"
    assert environment["VLLM_GPU_MEMORY_UTILIZATION"] == "0.84"

    invalid = retry.load_contract(CONTRACT, retry.sha256_file(CONTRACT))
    invalid["runtime"]["vllm_gpu_memory_utilization"] = 0.82
    with pytest.raises(retry.ContractError, match="util84 retry"):
        retry.validate_contract_shape(invalid)


def test_retry_trainer_and_completion_stay_on_v2_scientific_gates() -> None:
    contract = _command_contract()
    command, environment = retry.trainer_command_and_environment(contract, None)
    gate_position = command.index("--checkpoint-gate-script") + 1
    assert command[gate_position].endswith(retry.GATE_AUDITOR)
    assert "--resume-from-checkpoint" not in command
    assert environment["OUTPUT_DIR"] == retry.EXPECTED_PATHS["train_output"]

    completion = retry.completion_command(contract, Path("/tmp/retry-completion.json"))
    assert completion[1].endswith(retry.COMPLETION_AUDITOR)
    assert completion[completion.index("--run-dir") + 1] == (
        retry.EXPECTED_PATHS["train_output"]
    )
