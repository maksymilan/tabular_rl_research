from __future__ import annotations

import copy
from pathlib import Path

import yaml

from src.rl.experiments import (
    supervise_qwen3_8b_atomic_v26_representative600_onepass as v1,
)
from src.rl.experiments import (
    supervise_qwen3_8b_atomic_v26_representative600_onepass_v2 as v2,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = (
    ROOT
    / "src/rl/configs/experiments/"
    "qwen3_8b_atomic_v26_vanilla_grpo_representative600_onepass_v2.yaml"
)
CONTRACT = (
    ROOT
    / "src/rl/experiments/"
    "qwen3_8b_atomic_v26_representative600_onepass_v2_contract.json"
)


def _command_contract() -> dict:
    return {
        "paths": copy.deepcopy(v2.EXPECTED_PATHS),
        "resources": {
            "gpus": copy.deepcopy(v2.EXPECTED_GPUS),
            "vllm_port": 8076,
            "vllm_group_port": 51276,
        },
        "runtime": {
            "allocator_conf": "expandable_segments:True",
            "vllm_gpu_memory_utilization": 0.82,
            "max_model_len": v2.EXPECTED_MAX_CONTEXT_TOKENS,
        },
        "expected_implementation_lock_sha256": "a" * 64,
    }


def test_importing_v2_does_not_mutate_frozen_v1_module() -> None:
    assert v1.EXPECTED_EXPERIMENT_NAME.endswith("representative600_onepass")
    assert v1.EXPECTED_CONFIG_SHA256 == (
        "445a1a326abef1bc2c7d16cc8489eb55c264ba1ee80a6151ea4de9e4c2d3398a"
    )
    assert v1.GATE_AUDITOR.endswith("audit_vanilla_grpo_train600_step5.py")
    assert v2._engine is not v1
    assert v2.EXPECTED_EXPERIMENT_NAME.endswith("representative600_onepass_v2")


def test_frozen_v2_contract_shape_lock_and_read_only_plan() -> None:
    digest = v2.sha256_file(CONTRACT)
    assert digest == "23b14d744d93e21f69ee67810aa84fa73a9949cbcd4ce9745cc1bde804e3777c"
    contract = v2.load_contract(CONTRACT, digest)
    assert contract["expected_implementation_lock_sha256"] == (
        "f0fd949bea063c925e861163e1fc46f18c0e1c8da647fe6e01fd56dd0b150465"
    )
    assert v2.implementation_lock_sha256(contract) == (
        contract["expected_implementation_lock_sha256"]
    )
    plan = v2.plan_payload(contract)
    assert plan["mode"] == "plan_read_only"
    assert plan["rollout_budget"] == {
        "max_new_tokens": 3072,
        "max_context_tokens": 20480,
        "max_model_len": 20480,
    }


def test_contract_source_closure_matches_exact_local_bytes() -> None:
    contract = v2.load_contract(CONTRACT, v2.sha256_file(CONTRACT))
    source_hashes = contract["identities"]["runtime_source_sha256"]
    assert set(source_hashes) == v2.REQUIRED_RUNTIME_SOURCES
    for relative, expected in source_hashes.items():
        assert v2.sha256_file(ROOT / relative) == expected


def test_v2_config_keeps_algorithm_and_changes_only_runtime_budget_identity() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    v2._validate_experiment_config(CONFIG)
    assert v2.sha256_file(CONFIG) == v2.EXPECTED_CONFIG_SHA256
    assert payload["optimizer"]["learning_rate"] == 8e-7
    assert payload["optimizer"]["steps"] == 20
    assert payload["optimizer"]["ppo_iterations"] == 1
    assert payload["optimizer"]["kl_beta"] == 0.0
    assert payload["rollout"]["group_size"] == 8
    assert payload["rollout"]["prompts_per_update"] == 30
    assert payload["rollout"]["max_new_tokens"] == 3072
    assert payload["rollout"]["max_context_tokens"] == 20480
    assert payload["process_reward_config"] is None
    assert payload["rank_loss"]["enabled"] is False


def test_v2_paths_are_fresh_and_cannot_resume_v1_checkpoint5() -> None:
    for key in ("project_root", "run_root", "train_output", "experiment_config"):
        assert v2.EXPECTED_PATHS[key] != v1.EXPECTED_PATHS[key]
    assert "representative600_onepass_v2" in v2.EXPECTED_PATHS["train_output"]
    assert v1.EXPECTED_PATHS["train_output"] not in v2.EXPECTED_PATHS["train_output"]


def test_v2_trainer_and_completion_commands_bind_only_v2_gate_and_auditor() -> None:
    contract = _command_contract()
    command, environment = v2.trainer_command_and_environment(contract, None)
    gate_position = command.index("--checkpoint-gate-script") + 1
    receipt_position = command.index("--checkpoint-gate-receipt") + 1
    assert command[gate_position].endswith(v2.GATE_AUDITOR)
    assert command[receipt_position].endswith("/step5_gate.json")
    assert "--resume-from-checkpoint" not in command
    assert environment["OUTPUT_DIR"] == v2.EXPECTED_PATHS["train_output"]

    completion = v2.completion_command(contract, Path("/tmp/v2-completion.json"))
    assert completion[1].endswith(v2.COMPLETION_AUDITOR)
    assert completion[completion.index("--run-dir") + 1] == (
        v2.EXPECTED_PATHS["train_output"]
    )


def test_v2_vllm_command_uses_20k_model_context_and_same_memory_fraction() -> None:
    command, environment = v2.vllm_command_and_environment(_command_contract())
    assert command[0] == "bash"
    assert environment["MAX_MODEL_LEN"] == "20480"
    assert environment["VLLM_GPU_MEMORY_UTILIZATION"] == "0.82"
