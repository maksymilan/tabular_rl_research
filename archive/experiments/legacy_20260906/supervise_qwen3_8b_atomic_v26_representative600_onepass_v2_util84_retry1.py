#!/usr/bin/env python3
"""Fail-closed supervisor for the isolated representative600-v2 util84 retry.

The preceding v2 attempt stopped during vLLM startup, before any rollout or
trainer process began, because a 20,480-token KV cache did not fit at GPU
memory utilization 0.82.  This entry point privately loads the frozen v2
supervisor and changes only the engineering memory fraction to 0.84 plus the
experiment/path identities needed for a clean retry.  All data, algorithm,
rollout, gate, completion, and evaluation contracts remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = (
    "qwen3-v26-representative600-onepass-v2-util84-retry1-"
    "supervisor-contract-v1"
)
EVIDENCE_SCHEMA_VERSION = (
    "qwen3-v26-representative600-v2-util84-retry1-supervisor-event-v1"
)
EXPECTED_EXPERIMENT_NAME = (
    "qwen3_8b_atomic_v26_vanilla_grpo_"
    "representative600_onepass_v2_util84_retry1"
)
EXPECTED_VLLM_GPU_MEMORY_UTILIZATION = 0.84
EXPECTED_MAX_NEW_TOKENS = 3072
EXPECTED_MAX_CONTEXT_TOKENS = 20480
EXPECTED_CONFIG_SHA256 = (
    "d753e30263534524c4b09f95a04ba2d3797bd7c51446905e0a980e3285af5ea1"
)

EXPECTED_PATHS = {
    "project_root": (
        "/home/dengyan/tabular_rl_outputs/"
        "rl_runtime_qwen3_8b_v26_representative600_onepass_"
        "v2_util84_retry1_20260820"
    ),
    "python": "/home/dengyan/miniconda3/envs/trl-table/bin/python3.11",
    "model": "/home/dengyan/models/Qwen3-8B-TrustSQL-baseline",
    "initial_adapter": (
        "/home/dengyan/tabular_rl_outputs/checkpoints/"
        "qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560"
    ),
    "tasks": (
        "/home/dengyan/tabular_rl_outputs/"
        "rl_runtime_qwen3_8b_v26_representative600_onepass_"
        "v2_util84_retry1_20260820/data/rl_inputs/"
        "qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.jsonl"
    ),
    "tasks_manifest": (
        "/home/dengyan/tabular_rl_outputs/"
        "rl_runtime_qwen3_8b_v26_representative600_onepass_"
        "v2_util84_retry1_20260820/data/rl_inputs/"
        "qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.manifest.json"
    ),
    "protocol_runtime": (
        "/home/dengyan/tabular_rl_outputs/"
        "runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de"
    ),
    "experiment_config": (
        "/home/dengyan/tabular_rl_outputs/"
        "rl_runtime_qwen3_8b_v26_representative600_onepass_"
        "v2_util84_retry1_20260820/src/rl/configs/experiments/"
        "qwen3_8b_atomic_v26_vanilla_grpo_representative600_"
        "onepass_v2_util84_retry1.yaml"
    ),
    "run_root": (
        "/home/dengyan/tabular_rl_outputs/"
        "qwen3_8b_atomic_v26_representative600_onepass_"
        "v2_util84_retry1_vanilla_grpo_20260820"
    ),
    "train_output": (
        "/home/dengyan/tabular_rl_outputs/"
        "qwen3_8b_atomic_v26_representative600_onepass_"
        "v2_util84_retry1_vanilla_grpo_20260820/"
        "train600_onepass_v2_util84_retry1_seed20260812"
    ),
}

GATE_AUDITOR = "src/rl/diagnostics/audit_vanilla_grpo_train600_v2_step5.py"
COMPLETION_AUDITOR = (
    "src/rl/diagnostics/audit_representative600_onepass_v2_training.py"
)
V1_ENGINE = (
    "src/rl/experiments/"
    "supervise_qwen3_8b_atomic_v26_representative600_onepass.py"
)
V2_ENGINE = (
    "src/rl/experiments/"
    "supervise_qwen3_8b_atomic_v26_representative600_onepass_v2.py"
)
SELF_SOURCE = (
    "src/rl/experiments/"
    "supervise_qwen3_8b_atomic_v26_representative600_"
    "onepass_v2_util84_retry1.py"
)


def _load_private_v2():
    source = Path(__file__).with_name(
        "supervise_qwen3_8b_atomic_v26_representative600_onepass_v2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_representative600_v2_util84_retry1_engine", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load v2 supervisor engine: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_v2 = _load_private_v2()
_engine = _v2._engine
_original_v2_validate_contract_shape = _v2.validate_contract_shape
_original_v2_plan_payload = _v2.plan_payload


def _validate_experiment_config(path: Path) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _engine._require(isinstance(payload, dict), "experiment config must be a mapping")
    required = {
        "schema_version": "table-agent-rl-experiment-v1",
        "experiment_name": EXPECTED_EXPERIMENT_NAME,
        "admission_status": "allowed_result_only_control",
        "reward_type": "result",
        "result_reward_profile": "binary",
        "policy_reduction": "trajectory_token_mean",
        "expected_records": 600,
        "process_reward_config": None,
        "trainable_part": "all",
    }
    for key, expected in required.items():
        _engine._require(payload.get(key) == expected, f"config {key} mismatch")
    _engine._require(
        payload.get("rank_loss")
        == {"enabled": False, "coefficient": 0.0, "beta": 0.1},
        "rank loss is not disabled",
    )
    optimizer = payload.get("optimizer") or {}
    for key, expected in {
        "name": "adamw_torch",
        "learning_rate": 8e-7,
        "steps": 20,
        "ppo_iterations": 1,
        "kl_beta": 0.0,
    }.items():
        _engine._require(optimizer.get(key) == expected, f"optimizer {key} mismatch")
    rollout = payload.get("rollout") or {}
    for key, expected in {
        "prompts_per_update": 30,
        "group_size": 8,
        "max_agent_steps": 30,
        "max_new_tokens": EXPECTED_MAX_NEW_TOKENS,
        "max_context_tokens": EXPECTED_MAX_CONTEXT_TOKENS,
        "history_turns": 4,
        "temperature": 0.8,
        "top_p": 1.0,
        "top_k": 0,
        "enable_thinking": True,
    }.items():
        _engine._require(rollout.get(key) == expected, f"rollout {key} mismatch")


def validate_contract_shape(contract: Mapping[str, Any]) -> None:
    """Require util84, then reuse the exact frozen v2 shape validator."""

    runtime = contract.get("runtime") or {}
    _engine._require(
        runtime.get("vllm_gpu_memory_utilization")
        == EXPECTED_VLLM_GPU_MEMORY_UTILIZATION,
        "util84 retry vLLM GPU memory utilization mismatch",
    )
    translated = copy.deepcopy(dict(contract))
    translated["runtime"]["vllm_gpu_memory_utilization"] = 0.82
    _original_v2_validate_contract_shape(translated)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("plan", "preflight", "run", "status"),
        default="plan",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name(
            "qwen3_8b_atomic_v26_representative600_"
            "onepass_v2_util84_retry1_contract.json"
        ),
    )
    parser.add_argument("--expected-contract-sha256")
    return parser.parse_args(argv)


def plan_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = _original_v2_plan_payload(contract)
    payload["purpose"] = (
        "retry the unchanged representative600-v2 one-pass binary result-only GRPO "
        "baseline after the preceding attempt failed closed during vLLM startup, "
        "before any rollout or trainer process began"
    )
    payload["engineering_retry"] = {
        "trigger": "vllm_kv_cache_startup_capacity_failure_before_rollout_or_trainer",
        "changed_field": "runtime.vllm_gpu_memory_utilization",
        "previous_value": 0.82,
        "retry_value": EXPECTED_VLLM_GPU_MEMORY_UTILIZATION,
        "scientific_contract": "unchanged",
        "artifact_policy": "fresh_isolated_paths_no_resume_no_overwrite",
    }
    payload["prohibited"].extend(
        [
            "resume or import any artifact from the failed util82 v2 attempt",
            "overwrite or remove the failed util82 v2 evidence",
            "change any scientific hyperparameter, cohort, threshold, or evaluation policy",
        ]
    )
    return payload


# Configure only the private v2/v1 engine.  Importing this module cannot mutate
# either separately importable frozen supervisor module.
_engine.__file__ = __file__
_engine.SCHEMA_VERSION = SCHEMA_VERSION
_engine.EVIDENCE_SCHEMA_VERSION = EVIDENCE_SCHEMA_VERSION
_engine.EXPECTED_EXPERIMENT_NAME = EXPECTED_EXPERIMENT_NAME
_engine.EXPECTED_PATHS = EXPECTED_PATHS
_engine.EXPECTED_CONFIG_SHA256 = EXPECTED_CONFIG_SHA256
_engine.GATE_AUDITOR = GATE_AUDITOR
_engine.SUPPORT_SOURCES = (
    GATE_AUDITOR,
    COMPLETION_AUDITOR,
    _v2.V1_COMPLETION_ENGINE,
    _v2.V1_GATE_ENGINE,
    V1_ENGINE,
    V2_ENGINE,
    SELF_SOURCE,
    "src/rl/__init__.py",
    "src/rl/frameworks/__init__.py",
    "src/rl/frameworks/trl/__init__.py",
    "src/rl/diagnostics/analyze_grpo_training.py",
    "src/rl/diagnostics/compare_lora_updates.py",
    "src/rl/diagnostics/audit_earlystop_mixed180_training.py",
)
_engine.REQUIRED_RUNTIME_SOURCES = frozenset(
    (*_engine.RUNNER_IMPLEMENTATION_SOURCES, *_engine.SUPPORT_SOURCES)
)
_engine._validate_experiment_config = _validate_experiment_config
_engine.validate_contract_shape = validate_contract_shape
_engine.parse_args = parse_args
_engine.plan_payload = plan_payload

# Public aliases used by tests and deployment tooling.
ContractError = _engine.ContractError
EXPECTED_GPUS = _engine.EXPECTED_GPUS
EXPECTED_TASK_SHA256 = _engine.EXPECTED_TASK_SHA256
EXPECTED_TASK_MANIFEST_SHA256 = _engine.EXPECTED_TASK_MANIFEST_SHA256
EXPECTED_PROTOCOL_TREE_SHA256 = _engine.EXPECTED_PROTOCOL_TREE_SHA256
EXPECTED_INITIAL_ADAPTER_SHA256 = _engine.EXPECTED_INITIAL_ADAPTER_SHA256
EXPECTED_BASE_MODEL_IDENTITY = _engine.EXPECTED_BASE_MODEL_IDENTITY
EXPECTED_PYTHON_PACKAGES = _engine.EXPECTED_PYTHON_PACKAGES
RUNNER_IMPLEMENTATION_SOURCES = _engine.RUNNER_IMPLEMENTATION_SOURCES
SUPPORT_SOURCES = _engine.SUPPORT_SOURCES
REQUIRED_RUNTIME_SOURCES = _engine.REQUIRED_RUNTIME_SOURCES

sha256_file = _engine.sha256_file
load_contract = _engine.load_contract
implementation_lock_payload = _engine.implementation_lock_payload
implementation_lock_sha256 = _engine.implementation_lock_sha256
trainer_command_and_environment = _engine.trainer_command_and_environment
vllm_command_and_environment = _engine.vllm_command_and_environment
completion_command = _engine.completion_command
run_training = _engine.run_training


def main(argv: Sequence[str] | None = None) -> int:
    return _engine.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
