#!/usr/bin/env python3
"""Durable fail-closed supervisor for the fresh representative600-v2 run.

This entry point deliberately loads the reviewed v1 process-ownership engine
into a private module namespace, then replaces only the experiment identity,
isolated paths, rollout budget, gate, and completion-audit bindings.  Loading
the engine privately is important: importing this module must not mutate the
frozen v1 supervisor used to audit the stopped checkpoint-5 run.

The values below are implementation candidates until a frozen JSON contract
with an explicit SHA-256 is published.  Non-plan modes still require that
exact contract SHA, so this source alone cannot launch training.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "qwen3-v26-representative600-onepass-v2-supervisor-contract-v1"
EVIDENCE_SCHEMA_VERSION = "qwen3-v26-representative600-v2-supervisor-event-v1"
EXPECTED_EXPERIMENT_NAME = (
    "qwen3_8b_atomic_v26_vanilla_grpo_representative600_onepass_v2"
)

# These two values are intentionally named constants rather than hidden inside
# command construction.  The final values must be bound by both the config SHA
# and the supervisor contract before launch.
EXPECTED_MAX_NEW_TOKENS = 3072
EXPECTED_MAX_CONTEXT_TOKENS = 20480
EXPECTED_CONFIG_SHA256 = (
    "c932d7f61ea55d20739fe433808aca1fe95117e0d68a3d7ee93717e5cd175a1e"
)

EXPECTED_PATHS = {
    "project_root": (
        "/home/dengyan/tabular_rl_outputs/"
        "rl_runtime_qwen3_8b_v26_representative600_onepass_v2_20260820"
    ),
    "python": "/home/dengyan/miniconda3/envs/trl-table/bin/python3.11",
    "model": "/home/dengyan/models/Qwen3-8B-TrustSQL-baseline",
    "initial_adapter": (
        "/home/dengyan/tabular_rl_outputs/checkpoints/"
        "qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560"
    ),
    "tasks": (
        "/home/dengyan/tabular_rl_outputs/"
        "rl_runtime_qwen3_8b_v26_representative600_onepass_v2_20260820/"
        "data/rl_inputs/qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.jsonl"
    ),
    "tasks_manifest": (
        "/home/dengyan/tabular_rl_outputs/"
        "rl_runtime_qwen3_8b_v26_representative600_onepass_v2_20260820/"
        "data/rl_inputs/"
        "qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.manifest.json"
    ),
    "protocol_runtime": (
        "/home/dengyan/tabular_rl_outputs/"
        "runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de"
    ),
    "experiment_config": (
        "/home/dengyan/tabular_rl_outputs/"
        "rl_runtime_qwen3_8b_v26_representative600_onepass_v2_20260820/"
        "src/rl/configs/experiments/"
        "qwen3_8b_atomic_v26_vanilla_grpo_representative600_onepass_v2.yaml"
    ),
    "run_root": (
        "/home/dengyan/tabular_rl_outputs/"
        "qwen3_8b_atomic_v26_representative600_onepass_v2_vanilla_grpo_20260820"
    ),
    "train_output": (
        "/home/dengyan/tabular_rl_outputs/"
        "qwen3_8b_atomic_v26_representative600_onepass_v2_vanilla_grpo_20260820/"
        "train600_onepass_v2_seed20260812"
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
V1_COMPLETION_ENGINE = (
    "src/rl/diagnostics/audit_representative600_onepass_training.py"
)
V1_GATE_ENGINE = "src/rl/diagnostics/audit_vanilla_grpo_train600_step5.py"
SELF_SOURCE = (
    "src/rl/experiments/"
    "supervise_qwen3_8b_atomic_v26_representative600_onepass_v2.py"
)


def _load_private_engine():
    source = Path(__file__).with_name(
        "supervise_qwen3_8b_atomic_v26_representative600_onepass.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_representative600_v2_supervisor_engine", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load supervisor engine: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_engine = _load_private_engine()
_original_validate_contract_shape = _engine.validate_contract_shape
_original_no_gpu_import_smoke = _engine._validate_no_gpu_import_smoke
_original_plan_payload = _engine.plan_payload


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
    """Validate v2, then reuse the reviewed exact-v1 structural checks."""

    runtime = contract.get("runtime") or {}
    _engine._require(
        runtime.get("max_model_len") == EXPECTED_MAX_CONTEXT_TOKENS,
        "v2 max_model_len mismatch",
    )
    translated = copy.deepcopy(dict(contract))
    translated["runtime"]["max_model_len"] = 16384
    _original_validate_contract_shape(translated)


def _validate_no_gpu_import_smoke(
    contract: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    result = _original_no_gpu_import_smoke(contract, paths)
    project = paths["project_root"]
    command = [
        str(paths["python"]),
        str(project / COMPLETION_AUDITOR),
        "--help",
    ]
    environment = _engine._base_environment(contract)
    environment.update(
        {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": "", "PYTHONNOUSERSITE": "1"}
    )
    completed = subprocess.run(
        command,
        cwd=Path(contract["runtime"]["import_smoke_cwd"]),
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    _engine._require(
        completed.returncode == 0,
        f"v2 completion auditor no-GPU help smoke failed: {completed.stderr}",
    )
    result["help_entry_points"].append(
        {"entry_point": command[1], "returncode": completed.returncode}
    )
    return result


def completion_command(contract: Mapping[str, Any], output: Path) -> list[str]:
    paths = contract["paths"]
    return [
        paths["python"],
        str(Path(paths["project_root"]) / COMPLETION_AUDITOR),
        "--run-dir",
        paths["train_output"],
        "--tasks",
        paths["tasks"],
        "--cohort-manifest",
        paths["tasks_manifest"],
        "--experiment-config",
        paths["experiment_config"],
        "--expected-implementation-lock-sha256",
        contract["expected_implementation_lock_sha256"],
        "--output",
        str(output),
    ]


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
            "qwen3_8b_atomic_v26_representative600_onepass_v2_contract.json"
        ),
    )
    parser.add_argument("--expected-contract-sha256")
    return parser.parse_args(argv)


def plan_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = _original_plan_payload(contract)
    payload["purpose"] = (
        "rerun the same representative600 one-pass binary result-only GRPO baseline "
        "with a preregistered larger generation/context budget and structured known-"
        "runtime exclusions"
    )
    payload["rollout_budget"] = {
        "max_new_tokens": EXPECTED_MAX_NEW_TOKENS,
        "max_context_tokens": EXPECTED_MAX_CONTEXT_TOKENS,
        "max_model_len": EXPECTED_MAX_CONTEXT_TOKENS,
    }
    payload["prohibited"].extend(
        [
            "resume or import any checkpoint/artifact from the stopped v1 run",
            "relax a gate after observing v2 outcomes",
        ]
    )
    return payload


# Configure only the private engine module.  The separately importable frozen
# v1 module and its tests retain their original globals.
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
    V1_COMPLETION_ENGINE,
    V1_GATE_ENGINE,
    V1_ENGINE,
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
_engine._validate_no_gpu_import_smoke = _validate_no_gpu_import_smoke
_engine.completion_command = completion_command
_engine.parse_args = parse_args
_engine.plan_payload = plan_payload

# Public aliases used by focused tests and deployment tooling.
ContractError = _engine.ContractError
EvidenceJournal = _engine.EvidenceJournal
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
validate_static_inputs = _engine.validate_static_inputs
trainer_command_and_environment = _engine.trainer_command_and_environment
vllm_command_and_environment = _engine.vllm_command_and_environment
validate_existing_run_identity = _engine.validate_existing_run_identity
prepare_resume = _engine.prepare_resume
run_training = _engine.run_training


def main(argv: Sequence[str] | None = None) -> int:
    return _engine.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
