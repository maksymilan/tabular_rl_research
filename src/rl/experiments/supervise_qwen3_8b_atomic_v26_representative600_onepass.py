#!/usr/bin/env python3
"""Durable, fail-closed supervisor for the representative600 vanilla-GRPO run.

The default ``plan`` mode is read-only: it does not create directories, inspect
GPUs, bind ports, or launch processes.  ``preflight`` and ``run`` require a
separately frozen JSON contract plus its explicit SHA-256.  The supervisor only
signals process groups that it created itself with ``start_new_session=True``.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml


SCHEMA_VERSION = "qwen3-v26-representative600-onepass-supervisor-contract-v1"
EVIDENCE_SCHEMA_VERSION = "qwen3-v26-representative600-supervisor-event-v1"
EXPECTED_EXPERIMENT_NAME = (
    "qwen3_8b_atomic_v26_vanilla_grpo_representative600_onepass"
)
EXPECTED_PATHS = {
    "project_root": (
        "/home/dengyan/tabular_rl_outputs/"
        "rl_runtime_qwen3_8b_v26_representative600_onepass_20260817"
    ),
    "python": "/home/dengyan/miniconda3/envs/trl-table/bin/python3.11",
    "model": "/home/dengyan/models/Qwen3-8B-TrustSQL-baseline",
    "initial_adapter": (
        "/home/dengyan/tabular_rl_outputs/checkpoints/"
        "qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560"
    ),
    "tasks": (
        "/home/dengyan/tabular_rl_outputs/"
        "rl_runtime_qwen3_8b_v26_representative600_onepass_20260817/"
        "data/rl_inputs/qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.jsonl"
    ),
    "tasks_manifest": (
        "/home/dengyan/tabular_rl_outputs/"
        "rl_runtime_qwen3_8b_v26_representative600_onepass_20260817/"
        "data/rl_inputs/"
        "qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.manifest.json"
    ),
    "protocol_runtime": (
        "/home/dengyan/tabular_rl_outputs/"
        "runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de"
    ),
    "experiment_config": (
        "/home/dengyan/tabular_rl_outputs/"
        "rl_runtime_qwen3_8b_v26_representative600_onepass_20260817/"
        "src/rl/configs/experiments/"
        "qwen3_8b_atomic_v26_vanilla_grpo_representative600_onepass.yaml"
    ),
    "run_root": (
        "/home/dengyan/tabular_rl_outputs/"
        "qwen3_8b_atomic_v26_representative600_onepass_vanilla_grpo_20260817"
    ),
    "train_output": (
        "/home/dengyan/tabular_rl_outputs/"
        "qwen3_8b_atomic_v26_representative600_onepass_vanilla_grpo_20260817/"
        "train600_onepass_seed20260812"
    ),
}
EXPECTED_GPUS = {
    "trainer": {
        "index": 0,
        "uuid": "GPU-6d56a672-b23f-982f-feea-156758ce441a",
        "name": "NVIDIA GeForce RTX 3090",
    },
    "vllm": {
        "index": 1,
        "uuid": "GPU-e9293dd5-89f1-1e52-3a53-99d1bd145017",
        "name": "NVIDIA GeForce RTX 3090",
    },
}
EXPECTED_TASK_SHA256 = (
    "b5a83c373e9be094ea7355c0bc23c9212249491491f44dfdcb3b09ea49b2457e"
)
EXPECTED_TASK_MANIFEST_SHA256 = (
    "9212d1f7ca4fb63576eb7e846a9b05f7d159e131fe992495b99e350b6aae8dd6"
)
EXPECTED_CONFIG_SHA256 = (
    "445a1a326abef1bc2c7d16cc8489eb55c264ba1ee80a6151ea4de9e4c2d3398a"
)
EXPECTED_PROTOCOL_TREE_SHA256 = (
    "5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab"
)
EXPECTED_INITIAL_ADAPTER_SHA256 = (
    "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5"
)
EXPECTED_INITIAL_ADAPTER_CONFIG_SHA256 = (
    "537dbc946a7131d59e294a8729ace4aa145d73e59f4cf8556360ea507ce4435a"
)
EXPECTED_INITIAL_ADAPTER_STATE_SHA256 = (
    "97e529475d8c4f68dc88bc1f80370dacab478a681629f5f99003c39c70e9600a"
)
EXPECTED_BASE_MODEL_IDENTITY = {
    "schema_version": "trl-base-model-identity-v1",
    "aggregate_sha256": (
        "85bd3b7d908acb3a9b9c7ec57b98d6b9e3b2fb427685ae808d1c43173279cecc"
    ),
    "files_sha256": {
        "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
        "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
        "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "model-00001-of-00005.safetensors": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
        "model-00002-of-00005.safetensors": "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
        "model-00003-of-00005.safetensors": "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
        "model-00004-of-00005.safetensors": "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
        "model-00005-of-00005.safetensors": "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
        "model.safetensors.index.json": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
        "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    },
}
EXPECTED_PYTHON_PACKAGES = {
    "PyYAML": "6.0.3",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.0",
    "datasets": "5.0.0",
    "peft": "0.19.1",
    "safetensors": "0.8.0",
    "torch": "2.9.0",
    "transformers": "4.57.6",
    "trl": "0.29.0",
    "vllm": "0.12.0",
}

# This order is the order used by run_transition_grpo.py to serialize its
# implementation lock.  Keep it explicit so the expected lock hash is fixed
# before any optimizer step is allowed to run.
RUNNER_IMPLEMENTATION_SOURCES = (
    "src/rl/counterfactual_suite.py",
    "src/rl/experiment_config.py",
    "src/rl/reference_result_filter.py",
    "src/rl/task_loader.py",
    "src/rl/frameworks/trl/fixed_rollout_pool.py",
    "src/rl/frameworks/trl/checkpoint_gate.py",
    "src/rl/diagnostics/analyze_grpo_training.py",
    "src/rl/diagnostics/compare_lora_updates.py",
    "src/rl/diagnostics/prepare_vanilla_grpo_resume.py",
    "src/rl/frameworks/trl/run_atomic_transition_grpo.sh",
    "src/rl/frameworks/trl/run_transition_grpo.py",
    "src/rl/frameworks/trl/start_vllm_server.sh",
    "src/rl/frameworks/trl/transition_grpo.py",
    "src/rl/frameworks/trl/transition_batch.py",
    "src/rl/frameworks/trl/rollout.py",
    "src/rl/frameworks/trl/tool_loss_mask.py",
    "src/rl/frameworks/trl/training_precision.py",
    "src/rl/frameworks/trl/trajectory_ranking.py",
    "src/rl/rollout_scoring.py",
    "src/rl/terminal_reward.py",
    "src/rl/tool_environment.py",
    "src/rl/tool_environment_v26.py",
)
GATE_AUDITOR = "src/rl/diagnostics/audit_vanilla_grpo_train600_step5.py"
SUPPORT_SOURCES = (
    GATE_AUDITOR,
    "src/rl/__init__.py",
    "src/rl/frameworks/__init__.py",
    "src/rl/frameworks/trl/__init__.py",
    "src/rl/diagnostics/analyze_grpo_training.py",
    "src/rl/diagnostics/compare_lora_updates.py",
    "src/rl/diagnostics/audit_representative600_onepass_training.py",
    "src/rl/diagnostics/audit_earlystop_mixed180_training.py",
    "src/rl/experiments/supervise_qwen3_8b_atomic_v26_representative600_onepass.py",
)
REQUIRED_RUNTIME_SOURCES = frozenset((*RUNNER_IMPLEMENTATION_SOURCES, *SUPPORT_SOURCES))
LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
OOM_PATTERN = re.compile(
    r"CUDA out of memory|torch\.OutOfMemoryError|OutOfMemoryError:\s*CUDA",
    re.IGNORECASE,
)
ATTEMPT_LOG_PATTERN = re.compile(r"(?:trainer|vllm)_attempt([0-9]+)\.log")


class ContractError(RuntimeError):
    """The frozen launch contract or a runtime gate failed."""


class SupervisorSignal(ContractError):
    """A termination signal received by the owning supervisor."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _require_sha(value: object, label: str) -> str:
    text = str(value or "")
    _require(LOWER_SHA256.fullmatch(text) is not None, f"invalid {label} SHA-256")
    return text


def _regular(path: Path, label: str) -> None:
    _require(path.is_file() and not path.is_symlink(), f"invalid {label}: {path}")


def _directory(path: Path, label: str) -> None:
    _require(path.is_dir() and not path.is_symlink(), f"invalid {label}: {path}")


def load_contract(path: Path, expected_sha256: str | None) -> dict[str, Any]:
    _regular(path, "supervisor contract")
    actual = sha256_file(path)
    if expected_sha256 is not None:
        _require_sha(expected_sha256, "expected contract")
        _require(actual == expected_sha256, f"contract SHA mismatch: {actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "supervisor contract must be an object")
    validate_contract_shape(payload)
    return payload


def validate_contract_shape(contract: Mapping[str, Any]) -> None:
    _require(contract.get("schema_version") == SCHEMA_VERSION, "contract schema mismatch")
    _require(contract.get("status") == "frozen_before_launch", "contract is not frozen")
    experiment = contract.get("experiment") or {}
    expected_experiment = {
        "name": EXPECTED_EXPERIMENT_NAME,
        "algorithm": "vanilla_grpo",
        "arm": "baseline_result_only",
        "reward": "binary_result_only",
        "process_credit": "disabled",
        "custom_credit": "disabled",
        "records": 600,
        "passes": 1,
        "optimizer_steps": 20,
        "prompts_per_update": 30,
        "group_size": 8,
        "ppo_iterations": 1,
        "learning_rate": 8e-7,
        "kl_beta": 0.0,
        "seed": 20260812,
        "save_steps": 1,
        "save_total_limit": 20,
        "checkpoint_gate_step": 5,
        "evaluation_policy": "final_byte_identical_to_checkpoint_20_only",
        "posthoc_checkpoint_selection": False,
    }
    _require(experiment == expected_experiment, "experiment contract is not exact baseline GRPO")
    _require(contract.get("paths") == EXPECTED_PATHS, "server path contract mismatch")
    resources = contract.get("resources") or {}
    _require(resources.get("gpus") == EXPECTED_GPUS, "GPU index/UUID contract mismatch")
    _require(resources.get("vllm_port") == 8076, "vLLM port mismatch")
    _require(resources.get("vllm_group_port") == 51276, "vLLM group port mismatch")
    _require(resources.get("stable_samples") == 2, "GPU stable sample count mismatch")
    _require(resources.get("stable_interval_seconds") == 5.0, "GPU sample interval mismatch")
    _require(resources.get("max_idle_memory_mib") == 512, "GPU idle threshold mismatch")
    _require(resources.get("gpu_wait_poll_seconds") == 30.0, "GPU wait interval mismatch")
    _require(resources.get("gpu_wait_timeout_seconds") == 172800, "GPU wait timeout mismatch")
    _require(
        resources.get("min_free_disk_bytes") == 100 * 1024**3,
        "minimum free-disk contract mismatch",
    )

    identities = contract.get("identities") or {}
    _require(identities.get("tasks_sha256") == EXPECTED_TASK_SHA256, "task identity mismatch")
    _require(
        identities.get("tasks_manifest_sha256") == EXPECTED_TASK_MANIFEST_SHA256,
        "task manifest identity mismatch",
    )
    _require(identities.get("config_sha256") == EXPECTED_CONFIG_SHA256, "config mismatch")
    _require(
        identities.get("protocol_runtime_tree_sha256") == EXPECTED_PROTOCOL_TREE_SHA256,
        "protocol tree identity mismatch",
    )
    _require(
        identities.get("initial_adapter_sha256") == EXPECTED_INITIAL_ADAPTER_SHA256,
        "initial adapter identity mismatch",
    )
    _require(
        identities.get("initial_adapter_config_sha256")
        == EXPECTED_INITIAL_ADAPTER_CONFIG_SHA256,
        "initial adapter config identity mismatch",
    )
    _require(
        identities.get("initial_adapter_state_sha256")
        == EXPECTED_INITIAL_ADAPTER_STATE_SHA256,
        "initial adapter state identity mismatch",
    )
    _require(
        identities.get("base_model_identity") == EXPECTED_BASE_MODEL_IDENTITY,
        "base model identity mismatch",
    )
    _require(
        identities.get("python_packages") == EXPECTED_PYTHON_PACKAGES,
        "Python package identity mismatch",
    )
    _require_sha(identities.get("supervisor_source_sha256"), "supervisor source")
    source_hashes = identities.get("runtime_source_sha256")
    _require(isinstance(source_hashes, dict), "runtime source hashes must be an object")
    _require(set(source_hashes) == REQUIRED_RUNTIME_SOURCES, "runtime source closure mismatch")
    for relative, digest in source_hashes.items():
        _require_sha(digest, f"runtime source {relative}")
    expected_lock = _require_sha(
        contract.get("expected_implementation_lock_sha256"),
        "expected implementation lock",
    )
    _require(
        implementation_lock_sha256(contract) == expected_lock,
        "pre-registered implementation lock SHA mismatch",
    )
    runtime = contract.get("runtime") or {}
    _require(
        runtime
        == {
            "max_oom_restarts": 2,
            "monitor_poll_seconds": 20.0,
            "vllm_ready_timeout_seconds": 480.0,
            "vllm_gpu_memory_utilization": 0.82,
            "max_model_len": 16384,
            "allocator_conf": "expandable_segments:True",
            "import_smoke_cwd": "/tmp",
        },
        "runtime retry/memory contract mismatch",
    )


def implementation_lock_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    identities = contract["identities"]
    source_hashes = identities["runtime_source_sha256"]
    files = {relative: source_hashes[relative] for relative in RUNNER_IMPLEMENTATION_SOURCES}
    files[GATE_AUDITOR] = source_hashes[GATE_AUDITOR]
    paths = contract["paths"]
    checkpoint_gate = {
        "schema_version": "trl-synchronous-checkpoint-gate-v1",
        "step": 5,
        "script_relative_path": GATE_AUDITOR,
        "script_sha256": source_hashes[GATE_AUDITOR],
        "receipt": str(Path(paths["train_output"]) / "step5_gate.json"),
        "tasks_manifest": paths["tasks_manifest"],
        "tasks_manifest_sha256": EXPECTED_TASK_MANIFEST_SHA256,
        "execution": "synchronous_on_save_before_next_optimizer_step",
        "resume_policy": "checkpoint_at_or_after_gate_requires_verified_receipt",
    }
    return {
        "schema_version": "trl-implementation-lock-v1",
        "files": files,
        "initial_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
        "protocol_version": "version26",
        "protocol_hash": "4da19387399bd3a5",
        "student_prompt_sha256": (
            "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
        ),
        "base_model_identity": EXPECTED_BASE_MODEL_IDENTITY,
        "reference_policy": {
            "schema_version": "trl-frozen-reference-policy-v1",
            "enabled": False,
            "kl_beta": 0.0,
            "adapter_name": None,
            "adapter_path": None,
            "adapter_sha256": None,
            "expected_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
            "equals_initial_adapter": True,
            "load_audit": None,
            "after_trainer_init_audit": None,
        },
        "experiment_config_sha256": EXPECTED_CONFIG_SHA256,
        "checkpoint_gate": checkpoint_gate,
    }


def implementation_lock_sha256(contract: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(implementation_lock_payload(contract), ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def plan_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": "plan_read_only",
        "experiment": contract["experiment"],
        "purpose": (
            "establish whether the most basic one-pass binary result-only GRPO baseline "
            "improves Qwen3-8B before any custom process-credit experiment"
        ),
        "gates": [
            "source/data/model/protocol/config/implementation-lock identity preflight",
            "two stable idle samples for exact table_rl GPU0 trainer and GPU1 vLLM",
            "synchronous checkpoint-5 behavior/precision gate before optimizer step 6",
            "strict checkpoint-20/final completion audit; final is the only evaluation policy",
        ],
        "prohibited": [
            "Arm B or custom/process credit",
            "old first32 probe admission or reuse",
            "post-hoc checkpoint selection",
            "reuse of an unowned server or port",
            "stopping or signaling any unowned process",
        ],
    }


def _runtime_tree_sha256(root: Path) -> str:
    lock_path = root / "runtime_lock.json"
    _regular(lock_path, "protocol runtime lock")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    exported = lock.get("exported_paths")
    _require(exported == ["src/eval", "src/sft", "src/harness"], "runtime export mismatch")
    files: list[Path] = []
    for relative in exported:
        directory = root / relative
        _directory(directory, f"protocol runtime {relative}")
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    _require(len(files) == int(lock.get("exported_file_count") or 0), "runtime file count drift")
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    actual = digest.hexdigest()
    _require(actual == lock.get("content_tree_sha256"), "protocol runtime lock/content drift")
    return actual


def _validate_experiment_config(path: Path) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "experiment config must be a mapping")
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
        _require(payload.get(key) == expected, f"config {key} mismatch")
    _require(payload.get("rank_loss") == {"enabled": False, "coefficient": 0.0, "beta": 0.1}, "rank loss is not disabled")
    optimizer = payload.get("optimizer") or {}
    for key, expected in {
        "name": "adamw_torch",
        "learning_rate": 8e-7,
        "steps": 20,
        "ppo_iterations": 1,
        "kl_beta": 0.0,
    }.items():
        _require(optimizer.get(key) == expected, f"optimizer {key} mismatch")
    rollout = payload.get("rollout") or {}
    for key, expected in {
        "prompts_per_update": 30,
        "group_size": 8,
        "max_new_tokens": 2048,
        "max_context_tokens": 16384,
        "temperature": 0.8,
        "top_p": 1.0,
        "top_k": 0,
        "enable_thinking": True,
    }.items():
        _require(rollout.get(key) == expected, f"rollout {key} mismatch")


def _validate_python_packages(python: Path) -> dict[str, str]:
    names = list(EXPECTED_PYTHON_PACKAGES)
    code = (
        "import importlib.metadata as m,json,sys;"
        "print(json.dumps({n:m.version(n) for n in sys.argv[1:]},sort_keys=True))"
    )
    completed = subprocess.run(
        [str(python), "-c", code, *names],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    _require(completed.returncode == 0, f"cannot audit Python packages: {completed.stderr}")
    observed = json.loads(completed.stdout)
    _require(observed == EXPECTED_PYTHON_PACKAGES, f"Python package drift: {observed}")
    return observed


def _expected_import_smoke_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    project = paths["project_root"]
    protocol = paths["protocol_runtime"]
    return {
        "runner": str(project / "src/rl/frameworks/trl/run_transition_grpo.py"),
        "project_root": str(project),
        "checkpoint_gate": str(project / "src/rl/frameworks/trl/checkpoint_gate.py"),
        "tool_environment_v26": str(project / "src/rl/tool_environment_v26.py"),
        "protocol": str(protocol / "src/sft/protocol.py"),
        "evaluator_rollout": str(protocol / "src/eval/rollout.py"),
        "executor": str(protocol / "src/harness/executor.py"),
    }


def _validate_no_gpu_import_smoke(
    contract: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    """Import every launch/audit entry point outside the project with no GPU."""

    project = paths["project_root"]
    protocol = paths["protocol_runtime"]
    config = paths["experiment_config"]
    runner = project / "src/rl/frameworks/trl/run_transition_grpo.py"
    configured_smoke_cwd = Path(contract["runtime"]["import_smoke_cwd"])
    try:
        smoke_cwd = configured_smoke_cwd.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"cannot resolve import smoke paths: {exc}") from exc
    project_resolved = project.resolve(strict=False)
    _directory(smoke_cwd, "import smoke working directory")
    _require(
        project_resolved not in (smoke_cwd, *smoke_cwd.parents),
        "import smoke cwd must be outside the isolated project",
    )
    environment = _base_environment(contract)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": "",
            "PYTHONNOUSERSITE": "1",
        }
    )
    help_commands = [
        [
            str(paths["python"]),
            str(runner),
            "--experiment-config",
            str(config),
            "--protocol-runtime-root",
            str(protocol),
            "--help",
        ],
        [
            str(paths["python"]),
            str(project / "src/rl/diagnostics/prepare_vanilla_grpo_resume.py"),
            "--help",
        ],
        [
            str(paths["python"]),
            str(project / GATE_AUDITOR),
            "--help",
        ],
        [
            str(paths["python"]),
            str(project / "src/rl/diagnostics/audit_representative600_onepass_training.py"),
            "--help",
        ],
    ]
    help_audit = []
    for command in help_commands:
        completed = subprocess.run(
            command,
            cwd=smoke_cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        _require(
            completed.returncode == 0,
            f"no-GPU import/help smoke failed: command={command} stderr={completed.stderr}",
        )
        help_audit.append({"entry_point": command[1], "returncode": 0})

    probe_code = r'''import importlib.util, inspect, json, pathlib, sys
runner, config, protocol = sys.argv[1:4]
sys.argv = [runner, "--experiment-config", config, "--protocol-runtime-root", protocol]
spec = importlib.util.spec_from_file_location("_representative600_runner_smoke", runner)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
payload = {
    "runner": str(pathlib.Path(module.__file__).resolve()),
    "project_root": str(pathlib.Path(module.ROOT).resolve()),
    "checkpoint_gate": str(pathlib.Path(inspect.getfile(module.build_checkpoint_gate_spec)).resolve()),
    "tool_environment_v26": str(pathlib.Path(sys.modules["tool_environment_v26"].__file__).resolve()),
    "protocol": str(pathlib.Path(module.protocol_runtime.__file__).resolve()),
    "evaluator_rollout": str(pathlib.Path(module.evaluator_runtime.__file__).resolve()),
    "executor": str(pathlib.Path(module.executor_runtime.__file__).resolve()),
    "implementation_sources": list(module.IMPLEMENTATION_SOURCE_FILES),
}
print("IMPORT_SMOKE_JSON=" + json.dumps(payload, sort_keys=True))'''
    probe = subprocess.run(
        [
            str(paths["python"]),
            "-c",
            probe_code,
            str(runner),
            str(config),
            str(protocol),
        ],
        cwd=smoke_cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    _require(probe.returncode == 0, f"runner import path probe failed: {probe.stderr}")
    marker = "IMPORT_SMOKE_JSON="
    marked_lines = [line for line in probe.stdout.splitlines() if line.startswith(marker)]
    _require(len(marked_lines) == 1, "runner import path probe emitted no unique audit")
    observed = json.loads(marked_lines[0][len(marker) :])
    expected_paths = _expected_import_smoke_paths(paths)
    for key, expected in expected_paths.items():
        _require(observed.get(key) == expected, f"import smoke path drift for {key}")
    _require(
        tuple(observed.get("implementation_sources") or ())
        == RUNNER_IMPLEMENTATION_SOURCES,
        "imported runner source order differs from the supervisor contract",
    )
    return {
        "cuda_visible_devices": "",
        "working_directory": str(smoke_cwd),
        "help_entry_points": help_audit,
        "import_paths": expected_paths,
        "implementation_source_count": len(RUNNER_IMPLEMENTATION_SOURCES),
    }


def validate_static_inputs(contract: Mapping[str, Any]) -> dict[str, Any]:
    paths = {key: Path(value) for key, value in contract["paths"].items()}
    project = paths["project_root"]
    _directory(project, "isolated project root")
    _regular(paths["python"], "Python executable")
    _require(os.access(paths["python"], os.X_OK), "Python runtime is not executable")
    _directory(paths["model"], "base model")
    _directory(paths["initial_adapter"], "initial adapter")
    _directory(paths["protocol_runtime"], "protocol runtime")
    free_disk_bytes = shutil.disk_usage(paths["run_root"]).free
    _require(
        free_disk_bytes >= contract["resources"]["min_free_disk_bytes"],
        f"insufficient free disk for 20 durable checkpoints: {free_disk_bytes}",
    )
    _regular(paths["tasks"], "representative600 tasks")
    _regular(paths["tasks_manifest"], "representative600 manifest")
    _regular(paths["experiment_config"], "experiment config")
    for forbidden in ("src/eval", "src/sft", "src/harness", "src/tool_modules"):
        _require(not (project / forbidden).exists(), f"isolated overlay shadows {forbidden}")
    identities = contract["identities"]
    for relative, expected in identities["runtime_source_sha256"].items():
        source = project / relative
        _regular(source, f"runtime source {relative}")
        _require(sha256_file(source) == expected, f"runtime source drift: {relative}")
    self_path = Path(__file__).resolve()
    _require(
        sha256_file(self_path) == identities["supervisor_source_sha256"],
        "executing supervisor differs from frozen source",
    )
    _require(sha256_file(paths["tasks"]) == EXPECTED_TASK_SHA256, "tasks SHA drift")
    _require(
        sha256_file(paths["tasks_manifest"]) == EXPECTED_TASK_MANIFEST_SHA256,
        "tasks manifest SHA drift",
    )
    _require(sha256_file(paths["experiment_config"]) == EXPECTED_CONFIG_SHA256, "config SHA drift")
    _require(
        sha256_file(paths["initial_adapter"] / "adapter_model.safetensors")
        == EXPECTED_INITIAL_ADAPTER_SHA256,
        "initial adapter weights drift",
    )
    _require(
        sha256_file(paths["initial_adapter"] / "adapter_config.json")
        == EXPECTED_INITIAL_ADAPTER_CONFIG_SHA256,
        "initial adapter config drift",
    )
    _require(
        sha256_file(paths["initial_adapter"] / "trainer_state.json")
        == EXPECTED_INITIAL_ADAPTER_STATE_SHA256,
        "initial adapter trainer state drift",
    )
    for filename, expected in EXPECTED_BASE_MODEL_IDENTITY["files_sha256"].items():
        path = paths["model"] / filename
        _regular(path, f"base model {filename}")
        _require(sha256_file(path) == expected, f"base model drift: {filename}")
    _require(
        _runtime_tree_sha256(paths["protocol_runtime"]) == EXPECTED_PROTOCOL_TREE_SHA256,
        "protocol runtime tree drift",
    )
    _validate_experiment_config(paths["experiment_config"])
    packages = _validate_python_packages(paths["python"])
    import_smoke = _validate_no_gpu_import_smoke(contract, paths)
    rows = []
    with paths["tasks"].open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            _require(bool(line.strip()), f"empty task row {line_number}")
            row = json.loads(line)
            _require(isinstance(row, dict), f"task row {line_number} is not an object")
            rows.append(row)
    indices = [row.get("example_index") for row in rows]
    ids = [row.get("example_id") or row.get("instance_id") for row in rows]
    _require(len(rows) == 600 and len(set(indices)) == 600, "task count/index contract failed")
    _require(len(set(ids)) == 600 and None not in ids, "task identity contract failed")
    _require(len({row.get("db_id") for row in rows}) == 69, "task DB coverage mismatch")
    for row in rows:
        database = Path(str(row.get("db_path") or ""))
        _regular(database, f"task database {row.get('db_id')}")
    expected_lock = contract["expected_implementation_lock_sha256"]
    _require(implementation_lock_sha256(contract) == expected_lock, "implementation lock drift")
    return {
        "tasks": len(rows),
        "databases": 69,
        "runtime_sources": len(identities["runtime_source_sha256"]),
        "base_model_files": len(EXPECTED_BASE_MODEL_IDENTITY["files_sha256"]),
        "python_packages": packages,
        "no_gpu_import_smoke": import_smoke,
        "implementation_lock_sha256": expected_lock,
        "free_disk_bytes": free_disk_bytes,
    }


def _publish_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _directory(path.parent, "immutable evidence directory")
    if path.exists():
        _regular(path, "immutable evidence")
        _require(path.read_bytes() == encoded, f"immutable evidence differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _regular(path, "immutable evidence")
            _require(path.read_bytes() == encoded, f"evidence race differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


class EvidenceJournal:
    def __init__(self, root: Path, contract_sha256: str):
        self.root = root
        self.contract_sha256 = contract_sha256
        self.sequence = 0
        _ensure_child_directory(root, "evidence")
        _ensure_child_directory(root / "evidence", "events")

    def record(self, state: str, detail: Mapping[str, Any] | None = None) -> Path:
        self.sequence += 1
        payload = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "sequence": self.sequence,
            "timestamp_ns": time.time_ns(),
            "supervisor_pid": os.getpid(),
            "contract_sha256": self.contract_sha256,
            "state": state,
            "detail": dict(detail or {}),
        }
        name = f"{payload['timestamp_ns']:020d}-{self.sequence:04d}-{state}.json"
        path = self.root / "evidence" / "events" / name
        _publish_immutable_json(path, payload)
        return path


@dataclass(frozen=True)
class GpuRecord:
    index: int
    uuid: str
    name: str
    memory_used_mib: int
    compute_pids: tuple[int, ...]


def sample_gpus() -> dict[int, GpuRecord]:
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    compute_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    compute: dict[str, list[int]] = {}
    for line in compute_result.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid_text = [part.strip() for part in line.split(",", 1)]
        compute.setdefault(uuid, []).append(int(pid_text))
    records = {}
    for line in gpu_result.stdout.splitlines():
        index_text, uuid, name, memory = [part.strip() for part in line.split(",", 3)]
        index = int(index_text)
        records[index] = GpuRecord(
            index=index,
            uuid=uuid,
            name=name,
            memory_used_mib=int(memory),
            compute_pids=tuple(sorted(compute.get(uuid, []))),
        )
    return records


def _idle_pair_once(contract: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    records = sample_gpus()
    resources = contract["resources"]
    detail: dict[str, Any] = {"gpus": {}}
    idle = True
    for role in ("trainer", "vllm"):
        expected = resources["gpus"][role]
        record = records.get(expected["index"])
        _require(record is not None, f"missing GPU index for {role}")
        _require(record.uuid == expected["uuid"], f"GPU UUID drift for {role}")
        _require(record.name == expected["name"], f"GPU model drift for {role}")
        role_idle = (
            record.memory_used_mib <= resources["max_idle_memory_mib"]
            and not record.compute_pids
        )
        idle &= role_idle
        detail["gpus"][role] = {
            "index": record.index,
            "uuid": record.uuid,
            "name": record.name,
            "memory_used_mib": record.memory_used_mib,
            "compute_pids": list(record.compute_pids),
            "idle": role_idle,
        }
    return idle, detail


def wait_for_stable_idle_pair(
    contract: Mapping[str, Any],
    journal: EvidenceJournal,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    resources = contract["resources"]
    deadline = time.monotonic() + float(resources["gpu_wait_timeout_seconds"])
    last_busy: str | None = None
    while time.monotonic() < deadline:
        idle, first = _idle_pair_once(contract)
        if idle:
            sleep(float(resources["stable_interval_seconds"]))
            second_idle, second = _idle_pair_once(contract)
            if second_idle:
                samples = [first, second]
                journal.record("gpu_pair_stably_idle", {"samples": samples})
                return samples
            first = second
        compact = json.dumps(first, sort_keys=True)
        if compact != last_busy:
            journal.record(
                "waiting_for_gpu_pair",
                {"sample": first, "policy": "never_stop_or_signal_unowned_processes"},
            )
            last_busy = compact
        sleep(float(resources["gpu_wait_poll_seconds"]))
    raise ContractError("timed out waiting for the exact idle GPU0/GPU1 pair")


def _port_is_free(port: int) -> bool:
    candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        candidate.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        candidate.close()


def require_ports_free(contract: Mapping[str, Any]) -> None:
    for key in ("vllm_port", "vllm_group_port"):
        port = int(contract["resources"][key])
        _require(_port_is_free(port), f"port {port} is occupied; unowned reuse is forbidden")


def _health(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return 200 <= int(response.status) < 300
    except (OSError, urllib.error.URLError):
        return False


def _listener_pids(port: int) -> set[int]:
    """Resolve Linux TCP listener socket inodes to visible owning PIDs."""

    inodes: set[str] = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table.read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port:
                inodes.add(fields[9])
    if not inodes:
        return set()
    owners: set[int] = set()
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            descriptors = (process_dir / "fd").iterdir()
            for descriptor in descriptors:
                try:
                    target = os.readlink(descriptor)
                except OSError:
                    continue
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    owners.add(int(process_dir.name))
                    break
        except OSError:
            continue
    return owners


def require_listener_owned(port: int, groups: set[int]) -> dict[str, Any]:
    pids = _listener_pids(port)
    _require(bool(pids), f"cannot prove ownership of listener on port {port}")
    records = []
    for pid in sorted(pids):
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError as exc:
            raise ContractError(f"listener PID vanished during ownership audit: {pid}") from exc
        records.append({"pid": pid, "pgid": pgid})
    foreign = [record for record in records if record["pgid"] not in groups]
    _require(not foreign, f"port {port} has an unowned listener: {foreign}")
    return {"port": port, "listeners": records}


@dataclass
class OwnedProcess:
    role: str
    process: subprocess.Popen[bytes]
    pgid: int
    start_time_ticks: int
    log_path: Path


def _open_new_no_follow(path: Path):
    _directory(path.parent, "owned log directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    return os.fdopen(descriptor, "ab", buffering=0)


def start_owned_process(
    *,
    role: str,
    command: Sequence[str],
    environment: Mapping[str, str],
    log_path: Path,
    journal: EvidenceJournal,
) -> OwnedProcess:
    log = _open_new_no_follow(log_path)
    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(environment),
            start_new_session=True,
        )
    finally:
        log.close()
    try:
        pgid = os.getpgid(process.pid)
        start_time = _process_start_time_ticks(process.pid)
        _require(pgid == process.pid, f"{role} did not start as a new owned process group")
        _require(start_time is not None, f"{role} exited before ownership was recorded")
    except Exception:
        # start_new_session makes process.pid the only possible group id.  If
        # launch evidence cannot be established, signal that newly created id
        # once and never discover or target any other process.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        raise
    owned = OwnedProcess(
        role=role,
        process=process,
        pgid=pgid,
        start_time_ticks=start_time,
        log_path=log_path,
    )
    try:
        journal.record(
            f"{role}_launched",
            {
                "pid": process.pid,
                "pgid": pgid,
                "leader_start_time_ticks": start_time,
                "command": list(command),
                "log": str(log_path),
                "start_new_session": True,
            },
        )
    except Exception:
        terminate_owned_process(owned)
        raise
    return owned


def _process_start_time_ticks(pid: int) -> int | None:
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii").split()
    except OSError:
        return None
    # /proc/<pid>/stat field 22 is process start time; split index 21.
    try:
        return int(fields[21])
    except (IndexError, ValueError):
        return None


def terminate_owned_process(owned: OwnedProcess, *, grace_seconds: float = 30.0) -> None:
    # pgid was captured immediately after Popen(start_new_session=True).  No PID
    # discovery, name-based broad kill, or external process signalling is used.
    if owned.pgid != owned.process.pid:
        raise ContractError(f"refusing cleanup of unproved process group: {owned.role}")
    current_start_time = _process_start_time_ticks(owned.process.pid)
    if current_start_time is None:
        # The recorded group leader is gone.  Do not risk a numeric PGID reuse.
        owned.process.poll()
        return
    if current_start_time != owned.start_time_ticks:
        raise ContractError(f"refusing cleanup after PID reuse: {owned.role}")
    try:
        os.killpg(owned.pgid, signal.SIGTERM)
    except ProcessLookupError:
        owned.process.poll()
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(owned.pgid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.25)
    else:
        try:
            os.killpg(owned.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        owned.process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def cleanup_owned_processes(processes: Sequence[OwnedProcess | None]) -> None:
    errors = []
    for owned in processes:
        if owned is None:
            continue
        try:
            terminate_owned_process(owned)
        except Exception as exc:  # Continue so every proved-owned group is handled.
            errors.append(f"{owned.role}: {type(exc).__name__}: {exc}")
    if errors:
        raise ContractError(f"owned process cleanup failed: {errors}")


def _base_environment(contract: Mapping[str, Any]) -> dict[str, str]:
    # Do not inherit experiment-control variables such as fixed-pool,
    # counterfactual, process-credit, host, port, model, adapter, or output
    # overrides.  Only generic process/runtime locations cross this boundary.
    allowed_parent_keys = (
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "CUDA_HOME",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "TRITON_CACHE_DIR",
        "TMPDIR",
    )
    environment = {
        key: os.environ[key]
        for key in allowed_parent_keys
        if key in os.environ
    }
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": contract["runtime"]["allocator_conf"],
            "PYTORCH_ALLOC_CONF": contract["runtime"]["allocator_conf"],
            "VLLM_HOST": "127.0.0.1",
        }
    )
    return environment


def vllm_command_and_environment(contract: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
    paths = contract["paths"]
    resources = contract["resources"]
    environment = _base_environment(contract)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(resources["gpus"]["vllm"]["index"]),
            "PYTHON_ENV": str(Path(paths["python"]).parent.parent),
            "MODEL_PATH": paths["model"],
            "VLLM_PORT": str(resources["vllm_port"]),
            "VLLM_GPU_MEMORY_UTILIZATION": str(
                contract["runtime"]["vllm_gpu_memory_utilization"]
            ),
            "MAX_MODEL_LEN": str(contract["runtime"]["max_model_len"]),
            "VLLM_ENFORCE_EAGER": "1",
        }
    )
    command = [
        "bash",
        str(Path(paths["project_root"]) / "src/rl/frameworks/trl/start_vllm_server.sh"),
    ]
    return command, environment


def trainer_command_and_environment(
    contract: Mapping[str, Any], resume_checkpoint: Path | None
) -> tuple[list[str], dict[str, str]]:
    paths = contract["paths"]
    resources = contract["resources"]
    project = Path(paths["project_root"])
    output = Path(paths["train_output"])
    environment = _base_environment(contract)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(resources["gpus"]["trainer"]["index"]),
            # run_atomic_transition_grpo.sh appends the incoming PYTHONPATH.
            # Keeping the source-locked project root provides a package-form
            # fallback.  The runner lock now includes analyze/compare/resume;
            # the gate auditor is the sole extra source snapshot.
            "PYTHONPATH": str(project),
            "PROJECT_DIR": str(project),
            "PYTHON": paths["python"],
            "MODEL_PATH": paths["model"],
            "ADAPTER_PATH": paths["initial_adapter"],
            "EXAMPLES_JSON": paths["tasks"],
            "OUTPUT_DIR": str(output),
            "EXPERIMENT_CONFIG": paths["experiment_config"],
            "VLLM_PORT": str(resources["vllm_port"]),
            "VLLM_GROUP_PORT": str(resources["vllm_group_port"]),
        }
    )
    command = [
        "bash",
        str(project / "src/rl/frameworks/trl/run_atomic_transition_grpo.sh"),
        "--optimizer-steps",
        "20",
        "--ppo-iterations",
        "1",
        "--prompts-per-update",
        "30",
        "--group-size",
        "8",
        "--kl-beta",
        "0",
        "--protocol-runtime-root",
        paths["protocol_runtime"],
        "--transition-micro-batch-size",
        "1",
        "--save-steps",
        "1",
        "--save-total-limit",
        "20",
        "--seed",
        "20260812",
        "--checkpoint-gate-step",
        "5",
        "--checkpoint-gate-script",
        str(project / GATE_AUDITOR),
        "--checkpoint-gate-receipt",
        str(output / "step5_gate.json"),
        "--checkpoint-gate-tasks-manifest",
        paths["tasks_manifest"],
    ]
    if resume_checkpoint is not None:
        command.extend(["--resume-from-checkpoint", str(resume_checkpoint)])
    return command, environment


def resume_command(contract: Mapping[str, Any], output: Path) -> list[str]:
    paths = contract["paths"]
    return [
        paths["python"],
        str(Path(paths["project_root"]) / "src/rl/diagnostics/prepare_vanilla_grpo_resume.py"),
        "--train-output",
        paths["train_output"],
        "--tasks",
        paths["tasks"],
        "--expected-records",
        "600",
        "--optimizer-steps",
        "20",
        "--prompts-per-update",
        "30",
        "--group-size",
        "8",
        "--save-steps",
        "1",
        "--retain-checkpoint-step",
        "5",
        "--output",
        str(output),
    ]


def completion_command(contract: Mapping[str, Any], output: Path) -> list[str]:
    paths = contract["paths"]
    return [
        paths["python"],
        str(
            Path(paths["project_root"])
            / "src/rl/diagnostics/audit_representative600_onepass_training.py"
        ),
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


def _next_attempt_path(directory: Path, stem: str) -> Path:
    for attempt in range(1, 1000):
        candidate = directory / f"{stem}_attempt{attempt}.json"
        if not candidate.exists():
            return candidate
    raise ContractError(f"exhausted immutable attempt names for {stem}")


def _next_named_path(directory: Path, stem: str, suffix: str) -> Path:
    for attempt in range(1, 1000):
        candidate = directory / f"{stem}_attempt{attempt}{suffix}"
        if not candidate.exists():
            return candidate
    raise ContractError(f"exhausted immutable attempt names for {stem}")


def _copy_immutable(source: Path, destination: Path) -> str:
    _regular(source, "resume source artifact")
    _directory(destination.parent, "resume evidence directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o444)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_file, os.fdopen(descriptor, "wb") as output_file:
            for block in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
                digest.update(block)
                output_file.write(block)
            output_file.flush()
            os.fsync(output_file.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    actual = sha256_file(destination)
    _require(actual == digest.hexdigest(), "resume evidence copy changed bytes")
    return actual


def validate_existing_run_identity(
    contract: Mapping[str, Any], train_output: Path
) -> dict[str, Any]:
    manifest_path = train_output / "run_manifest.json"
    lock_path = train_output / "implementation_lock.json"
    rollouts_path = train_output / "rollouts.jsonl"
    for path, label in (
        (manifest_path, "resume run manifest"),
        (lock_path, "resume implementation lock"),
        (rollouts_path, "resume rollout log"),
    ):
        _regular(path, label)
    expected_lock_sha = contract["expected_implementation_lock_sha256"]
    _require(sha256_file(lock_path) == expected_lock_sha, "resume implementation lock SHA drift")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(lock == implementation_lock_payload(contract), "resume implementation lock content drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = contract["paths"]
    expected_manifest = {
        "schema_version": "table-agent-trl-transition-grpo-v2",
        "protocol_version": "version26",
        "protocol_hash": "4da19387399bd3a5",
        "model_path": paths["model"],
        "base_model_identity": EXPECTED_BASE_MODEL_IDENTITY,
        "adapter_path": paths["initial_adapter"],
        "initial_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
        "reference_policy": lock["reference_policy"],
        "implementation_source_sha256": lock["files"],
        "experiment_config_sha256": EXPECTED_CONFIG_SHA256,
        "checkpoint_gate": lock["checkpoint_gate"],
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
        "transition_micro_batch_size": 1,
        "rank_loss_coefficient": 0.0,
        "optimizer_name": "adamw_torch",
        "learning_rate": 8e-7,
        "kl_beta": 0.0,
        "examples_json_sha256": EXPECTED_TASK_SHA256,
        "fixed_rollout_pool": None,
        "fixed_rollout_pool_sha256": None,
        "fixed_pool_manifest": None,
        "fixed_pool_manifest_sha256": None,
    }
    mismatches = {
        key: {"observed": manifest.get(key), "expected": expected}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    _require(not mismatches, f"resume run identity mismatch: {mismatches}")
    resume_from = manifest.get("resume_from_checkpoint")
    if resume_from is not None:
        resume_path = Path(str(resume_from)).resolve()
        _require(
            resume_path.parent == train_output.resolve()
            and re.fullmatch(r"checkpoint-[0-9]+", resume_path.name) is not None,
            "recorded resume checkpoint escaped the run",
        )
    snapshot_root = train_output / "implementation_source_snapshot"
    _directory(snapshot_root, "resume implementation snapshot")
    for relative, expected in lock["files"].items():
        snapshot = snapshot_root / relative
        _regular(snapshot, f"resume source snapshot {relative}")
        _require(sha256_file(snapshot) == expected, f"resume source snapshot drift: {relative}")
    complete_steps = []
    checkpoint_required = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "rng_state.pth",
        "training_args.bin",
    )
    for checkpoint in train_output.glob("checkpoint-*"):
        _require(checkpoint.is_dir() and not checkpoint.is_symlink(), "invalid resume checkpoint entry")
        match = re.fullmatch(r"checkpoint-([0-9]+)", checkpoint.name)
        _require(match is not None, f"invalid resume checkpoint name: {checkpoint.name}")
        step = int(match.group(1))
        if all((checkpoint / name).is_file() and not (checkpoint / name).is_symlink() for name in checkpoint_required):
            state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
            if state.get("global_step") == step and state.get("max_steps") in (None, 20):
                complete_steps.append(step)
    _require(bool(complete_steps), "resume output has no complete checkpoint")
    latest_complete = max(complete_steps)
    gate_receipt = train_output / "step5_gate.json"
    gate_invocation = train_output / "checkpoint_gate_step5_invocation.json"
    if latest_complete >= 5:
        _regular(gate_receipt, "resume step5 gate receipt")
        _regular(gate_invocation, "resume step5 gate invocation")
    else:
        _require(not gate_receipt.exists(), "gate receipt exists before latest complete step 5")
        _require(not gate_invocation.exists(), "gate invocation exists before latest complete step 5")
    return {
        "implementation_lock_sha256": expected_lock_sha,
        "manifest_sha256": sha256_file(manifest_path),
        "rollouts_sha256": sha256_file(rollouts_path),
        "latest_complete_checkpoint": latest_complete,
    }


def prepare_resume(contract: Mapping[str, Any], journal: EvidenceJournal) -> Path | None:
    train_output = Path(contract["paths"]["train_output"])
    if not train_output.exists():
        return None
    _directory(train_output, "training output")
    entries = list(train_output.iterdir())
    if not entries:
        return None
    identity = validate_existing_run_identity(contract, train_output)
    resume_evidence = _ensure_child_directory(journal.root / "evidence", "resume")
    preserved_rollouts = _next_named_path(
        resume_evidence, "rollouts_before_resume", ".jsonl"
    )
    preserved_sha = _copy_immutable(train_output / "rollouts.jsonl", preserved_rollouts)
    journal.record(
        "pre_resume_identity_and_rollouts_preserved",
        {
            "identity": identity,
            "preserved_rollouts": str(preserved_rollouts),
            "preserved_rollouts_sha256": preserved_sha,
        },
    )
    resume_records = _ensure_child_directory(train_output, "resume_records")
    record = _next_attempt_path(resume_records, "resume_preparation")
    command = resume_command(contract, record)
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=_base_environment(contract),
        check=False,
    )
    journal.record(
        "resume_preparer_finished",
        {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "record": str(record),
        },
    )
    _require(completed.returncode == 0, "pinned resume preparation failed")
    _regular(record, "resume preparation record")
    payload = json.loads(record.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == "vanilla-grpo-resume-preparation-v1", "resume schema mismatch")
    _require(payload.get("status") == "resume_ready", "resume is not ready")
    expected_contract = {
        "optimizer_steps": 20,
        "prompts_per_update": 30,
        "group_size": 8,
        "save_steps": 1,
        "committed_rollouts_per_step": 240,
    }
    _require(payload.get("contract") == expected_contract, "resume dimension contract mismatch")
    checkpoint = payload.get("checkpoint") or {}
    step = checkpoint.get("global_step")
    _require(type(step) is int and 0 < step < 20, "resume checkpoint step is invalid")
    path = Path(str(checkpoint.get("path") or "")).resolve()
    _require(path == train_output.resolve() / f"checkpoint-{step}", "resume checkpoint path mismatch")
    gate_receipt = train_output / "step5_gate.json"
    if step >= 5:
        _regular(gate_receipt, "step5 gate receipt required by resume")
    else:
        _require(not gate_receipt.exists(), "step5 receipt exists before resumed policy step 5")
    expected_retained = (
        [str((train_output.resolve() / "checkpoint-5"))]
        if step >= 5
        else []
    )
    _require(
        payload.get("retained_audit_checkpoints") == expected_retained,
        "resume helper did not preserve the exact checkpoint-5 audit anchor",
    )
    record.chmod(0o444)
    return path


def _gpu_processes_owned_by(groups: set[int]) -> tuple[bool, dict[str, Any]]:
    records = sample_gpus()
    detail: dict[str, Any] = {}
    valid = True
    for role, expected in EXPECTED_GPUS.items():
        record = records.get(expected["index"])
        _require(record is not None and record.uuid == expected["uuid"], f"GPU identity drift for {role}")
        foreign = []
        for pid in record.compute_pids:
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                continue
            if pgid not in groups:
                foreign.append({"pid": pid, "pgid": pgid})
        valid &= not foreign
        detail[role] = {
            "memory_used_mib": record.memory_used_mib,
            "compute_pids": list(record.compute_pids),
            "foreign": foreign,
        }
    return valid, detail


def require_vllm_owned_and_trainer_idle(
    contract: Mapping[str, Any], vllm_pgid: int
) -> dict[str, Any]:
    valid, detail = _gpu_processes_owned_by({vllm_pgid})
    _require(valid, f"foreign process entered GPU pair before trainer launch: {detail}")
    trainer = detail["trainer"]
    _require(
        not trainer["compute_pids"]
        and trainer["memory_used_mib"] <= contract["resources"]["max_idle_memory_mib"],
        f"trainer GPU is no longer idle: {trainer}",
    )
    return detail


def wait_for_vllm(
    contract: Mapping[str, Any], owned: OwnedProcess, journal: EvidenceJournal
) -> None:
    deadline = time.monotonic() + float(contract["runtime"]["vllm_ready_timeout_seconds"])
    port = int(contract["resources"]["vllm_port"])
    while time.monotonic() < deadline:
        code = owned.process.poll()
        _require(code is None, f"owned vLLM exited before readiness: {code}")
        if _health(port):
            listener = require_listener_owned(port, {owned.pgid})
            valid, detail = _gpu_processes_owned_by({owned.pgid})
            _require(valid, f"foreign GPU process appeared during vLLM startup: {detail}")
            trainer_gpu = detail["trainer"]
            _require(
                not trainer_gpu["compute_pids"]
                and trainer_gpu["memory_used_mib"]
                <= contract["resources"]["max_idle_memory_mib"],
                "trainer GPU lost ownership before trainer launch",
            )
            journal.record("vllm_ready_and_owned", {"listener": listener, **detail})
            return
        time.sleep(2)
    raise ContractError("owned vLLM failed readiness timeout")


def wait_for_trainer(
    contract: Mapping[str, Any], trainer: OwnedProcess, vllm: OwnedProcess, journal: EvidenceJournal
) -> int:
    interval = float(contract["runtime"]["monitor_poll_seconds"])
    while True:
        code = trainer.process.poll()
        if code is not None:
            journal.record("trainer_exited", {"returncode": code})
            return int(code)
        _require(vllm.process.poll() is None, "owned vLLM exited while trainer was active")
        vllm_port = int(contract["resources"]["vllm_port"])
        # Do not use an HTTP latency probe while a large rollout batch is in
        # flight.  vLLM's event loop can legitimately delay /health even while
        # its owned engine and listener are healthy and actively generating.
        # Process liveness, listener ownership, and GPU-process ownership are
        # the fail-closed runtime invariants during training.
        require_listener_owned(vllm_port, {vllm.pgid})
        valid, detail = _gpu_processes_owned_by({trainer.pgid, vllm.pgid})
        _require(valid, f"foreign compute process entered owned GPU pair: {detail}")
        time.sleep(interval)


def _confirmed_oom(log_path: Path) -> bool:
    if not log_path.is_file() or log_path.is_symlink():
        return False
    size = log_path.stat().st_size
    with log_path.open("rb") as source:
        source.seek(max(0, size - 2 * 1024 * 1024))
        text = source.read().decode("utf-8", errors="replace")
    return OOM_PATTERN.search(text) is not None


def _next_launch_number(log_root: Path) -> int:
    _directory(log_root, "owned log directory")
    numbers = []
    for path in log_root.iterdir():
        match = ATTEMPT_LOG_PATTERN.fullmatch(path.name)
        if match is not None:
            _regular(path, "prior owned launch log")
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _prior_confirmed_ooms(journal: EvidenceJournal) -> int:
    event_root = journal.root / "evidence" / "events"
    _directory(event_root, "supervisor event directory")
    return len(list(event_root.glob("*-confirmed_cuda_oom.json")))


def run_completion_audit(contract: Mapping[str, Any], journal: EvidenceJournal) -> Path:
    root = Path(contract["paths"]["run_root"])
    completion_root = _ensure_child_directory(root / "evidence", "completion")
    output = _next_attempt_path(completion_root, "completion_audit")
    command = completion_command(contract, output)
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=_base_environment(contract),
        check=False,
    )
    _regular(output, "completion audit receipt")
    payload = json.loads(output.read_text(encoding="utf-8"))
    passed = payload.get("status") == {"passes": True, "evaluation_admitted": True}
    output.chmod(0o444)
    journal.record(
        "completion_audit_finished",
        {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "receipt": str(output),
            "receipt_sha256": sha256_file(output),
            "passes": passed,
        },
    )
    _require(completed.returncode == 0 and passed, "strict checkpoint-20 completion audit failed")
    return output


def _has_final_candidate(contract: Mapping[str, Any]) -> bool:
    output = Path(contract["paths"]["train_output"])
    return (
        (output / "checkpoint-20/adapter_model.safetensors").is_file()
        and (output / "final/adapter_model.safetensors").is_file()
    )


def run_training(contract: Mapping[str, Any], journal: EvidenceJournal) -> Path:
    if _has_final_candidate(contract):
        journal.record("existing_final_candidate_found", {})
        return run_completion_audit(contract, journal)
    max_restarts = int(contract["runtime"]["max_oom_restarts"])
    prior_ooms = _prior_confirmed_ooms(journal)
    _require(prior_ooms <= max_restarts, "prior OOM evidence exceeds retry contract")
    remaining_launches = max_restarts - prior_ooms + 1
    for _ in range(remaining_launches):
        launch_number = _next_launch_number(Path(contract["paths"]["run_root"]) / "logs")
        wait_for_stable_idle_pair(contract, journal)
        # The GPU wait may last hours.  Re-hash every immutable input before
        # the resume helper is allowed to mutate/quarantine any run artifact.
        post_wait_audit = validate_static_inputs(contract)
        journal.record("post_wait_identity_revalidated_before_resume", post_wait_audit)
        resume_checkpoint = prepare_resume(contract, journal)
        # The helper itself is source-locked, but revalidate after its mutation
        # boundary before the final stable ownership/port admission.
        launch_audit = validate_static_inputs(contract)
        journal.record("post_resume_identity_revalidated", launch_audit)
        wait_for_stable_idle_pair(contract, journal)
        require_ports_free(contract)
        idle_now, immediate_sample = _idle_pair_once(contract)
        _require(idle_now, f"GPU pair changed immediately before vLLM spawn: {immediate_sample}")
        journal.record(
            "launch_attempt_admitted",
            {
                "launch_number": launch_number,
                "prior_confirmed_ooms": prior_ooms,
                "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            },
        )
        vllm: OwnedProcess | None = None
        trainer: OwnedProcess | None = None
        trainer_log = (
            Path(contract["paths"]["run_root"])
            / "logs"
            / f"trainer_attempt{launch_number}.log"
        )
        try:
            vllm_command, vllm_env = vllm_command_and_environment(contract)
            vllm = start_owned_process(
                role="vllm",
                command=vllm_command,
                environment=vllm_env,
                log_path=(
                    Path(contract["paths"]["run_root"])
                    / "logs"
                    / f"vllm_attempt{launch_number}.log"
                ),
                journal=journal,
            )
            owned_now, owned_detail = _gpu_processes_owned_by({vllm.pgid})
            _require(owned_now, f"foreign process raced vLLM spawn: {owned_detail}")
            wait_for_vllm(contract, vllm, journal)
            trainer_command, trainer_env = trainer_command_and_environment(
                contract, resume_checkpoint
            )
            pretrainer_detail = require_vllm_owned_and_trainer_idle(
                contract, vllm.pgid
            )
            journal.record("trainer_gpu_idle_immediately_pre_spawn", pretrainer_detail)
            trainer = start_owned_process(
                role="trainer",
                command=trainer_command,
                environment=trainer_env,
                log_path=trainer_log,
                journal=journal,
            )
            owned_now, owned_detail = _gpu_processes_owned_by(
                {trainer.pgid, vllm.pgid}
            )
            _require(owned_now, f"foreign process raced trainer spawn: {owned_detail}")
            code = wait_for_trainer(contract, trainer, vllm, journal)
        finally:
            cleanup_owned_processes((trainer, vllm))
        if code == 0:
            return run_completion_audit(contract, journal)
        if not _confirmed_oom(trainer_log):
            raise ContractError(f"trainer failed with non-OOM exit {code}; no automatic resume")
        prior_ooms += 1
        journal.record(
            "confirmed_cuda_oom",
            {
                "launch_number": launch_number,
                "confirmed_oom_count": prior_ooms,
                "trainer_log": str(trainer_log),
            },
        )
        if prior_ooms > max_restarts:
            raise ContractError("confirmed CUDA OOM exhausted pre-registered retry budget")
    raise AssertionError("unreachable")


def _acquire_lock(run_root: Path):
    run_root.mkdir(parents=True, exist_ok=True)
    _directory(run_root, "run root")
    path = run_root / "supervisor.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    _require(stat.S_ISREG(os.fstat(descriptor).st_mode), "supervisor lock is not regular")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise ContractError(f"another supervisor owns {path}")
    return descriptor


def _ensure_child_directory(parent: Path, name: str) -> Path:
    _directory(parent, "owned directory parent")
    path = parent / name
    try:
        os.mkdir(path, 0o755)
    except FileExistsError:
        pass
    _directory(path, f"owned directory {name}")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("plan", "preflight", "run", "status"), default="plan")
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name(
            "qwen3_8b_atomic_v26_representative600_onepass_contract.json"
        ),
    )
    parser.add_argument("--expected-contract-sha256")
    return parser.parse_args(argv)


def _status(contract: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(contract["paths"]["run_root"])
    events = sorted((root / "evidence" / "events").glob("*.json")) if root.exists() else []
    return {
        "run_root": str(root),
        "exists": root.exists(),
        "event_count": len(events),
        "latest_event": json.loads(events[-1].read_text(encoding="utf-8")) if events else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected_contract_sha = args.expected_contract_sha256
    if args.mode != "plan":
        _require(expected_contract_sha is not None, "non-plan modes require explicit contract SHA")
    contract = load_contract(args.contract, expected_contract_sha)
    if args.mode == "plan":
        print(json.dumps(plan_payload(contract), ensure_ascii=False, indent=2))
        return 0
    if args.mode == "status":
        print(json.dumps(_status(contract), ensure_ascii=False, indent=2))
        return 0
    run_root = Path(contract["paths"]["run_root"])
    lock_descriptor = _acquire_lock(run_root)
    _ensure_child_directory(run_root, "logs")
    contract_sha = sha256_file(args.contract)
    journal = EvidenceJournal(run_root, contract_sha)
    previous_handlers: dict[int, Any] = {}

    def receive_signal(signum: int, _frame: Any) -> None:
        raise SupervisorSignal(f"received signal {signal.Signals(signum).name}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, receive_signal)
    try:
        journal.record("validating_static_inputs", {"mode": args.mode})
        audit = validate_static_inputs(contract)
        journal.record("static_preflight_passed", audit)
        if args.mode == "preflight":
            print(json.dumps({"status": "preflight_passed", **audit}, ensure_ascii=False))
            return 0
        completion = run_training(contract, journal)
        journal.record(
            "complete",
            {
                "completion_audit": str(completion),
                "evaluation_policy": "final_only",
                "posthoc_checkpoint_selection": False,
            },
        )
        print(json.dumps({"status": "complete", "completion_audit": str(completion)}))
        return 0
    except Exception as exc:
        journal.record(
            "blocked",
            {"error_type": type(exc).__name__, "error": str(exc), "policy": "fail_closed"},
        )
        print(f"blocked: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        os.close(lock_descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
