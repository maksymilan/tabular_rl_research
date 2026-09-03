#!/usr/bin/env python3
"""Launch TRL-backed clipped policy optimization over exact table-agent turns."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[4]


def _bootstrap_protocol_runtime(argv: list[str]) -> Path | None:
    """Select the identity-bound protocol trees before project modules import."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--protocol-runtime-root", type=Path)
    known, _ = parser.parse_known_args(argv)
    config_runtime_root = None
    if known.experiment_config is not None:
        payload = yaml.safe_load(
            known.experiment_config.read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict):
            raise RuntimeError("experiment config must contain a YAML mapping")
        runtime_contract = payload.get("runtime_contract") or {}
        if runtime_contract.get("runtime_root"):
            config_runtime_root = Path(runtime_contract["runtime_root"])
    cli_runtime_root = known.protocol_runtime_root
    if cli_runtime_root is not None and config_runtime_root is not None:
        if cli_runtime_root.resolve() != config_runtime_root.resolve():
            raise RuntimeError(
                "CLI protocol runtime differs from the experiment contract: "
                f"{cli_runtime_root} != {config_runtime_root}"
            )
    runtime_root = cli_runtime_root or config_runtime_root
    if runtime_root is None:
        return None
    runtime_root = runtime_root.resolve()
    for relative in ("src/eval", "src/harness", "src/sft"):
        if not (runtime_root / relative).is_dir():
            raise RuntimeError(
                f"protocol runtime is missing {relative}: {runtime_root}"
            )
    return runtime_root


PROTOCOL_RUNTIME_ROOT_AT_IMPORT = _bootstrap_protocol_runtime(sys.argv[1:])
ACTIVE_RUNTIME_ROOT = PROTOCOL_RUNTIME_ROOT_AT_IMPORT or ROOT
ACTIVE_EVAL_DIR = ACTIVE_RUNTIME_ROOT / "src" / "eval"
ACTIVE_HARNESS_DIR = ACTIVE_RUNTIME_ROOT / "src" / "harness"
ACTIVE_SFT_DIR = ACTIVE_RUNTIME_ROOT / "src" / "sft"
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "src" / "rl"),
    str(ACTIVE_EVAL_DIR),
    str(ACTIVE_HARNESS_DIR),
    str(ACTIVE_SFT_DIR),
]

import protocol as protocol_runtime  # noqa: E402
import rollout as evaluator_runtime  # noqa: E402
import executor as executor_runtime  # noqa: E402
from tool_modules import registry as tool_schemes_runtime  # noqa: E402

import bitsandbytes as bnb
import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import GRPOConfig


IMPLEMENTATION_SOURCE_FILES = (
    "src/rl/counterfactual_suite.py",
    "src/rl/experiment_config.py",
    "src/rl/reference_result_filter.py",
    "src/rl/task_loader.py",
    "src/rl/frameworks/trl/fixed_rollout_pool.py",
    "src/rl/frameworks/trl/checkpoint_gate.py",
    "src/rl/diagnostics/analyze_grpo_training.py",
    "src/rl/diagnostics/compare_lora_updates.py",
    "src/rl/diagnostics/prepare_vanilla_grpo_resume.py",
    "src/rl/diagnostics/audit_saam_lineage_replay.py",
    "src/rl/frameworks/trl/run_atomic_transition_grpo.sh",
    "src/rl/frameworks/trl/run_transition_grpo.py",
    "src/rl/frameworks/trl/start_vllm_server.sh",
    "src/rl/frameworks/trl/transition_grpo.py",
    "src/rl/frameworks/trl/transition_batch.py",
    "src/rl/frameworks/trl/state_action_ambiguity.py",
    "src/rl/frameworks/trl/gradient_conflict.py",
    "src/rl/frameworks/trl/rollout.py",
    "src/rl/frameworks/trl/tool_loss_mask.py",
    "src/rl/frameworks/trl/training_precision.py",
    "src/rl/frameworks/trl/trajectory_ranking.py",
    "src/rl/experiments/run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_a100.sh",
    "src/rl/rollout_scoring.py",
    "src/rl/terminal_reward.py",
    "src/rl/tool_environment_v26.py",
    "src/tool_modules/registry.py",
)
from counterfactual_suite import (  # noqa: E402
    SCHEMA_VERSION as COUNTERFACTUAL_SCHEMA_VERSION,
    CounterfactualSuiteManifest,
    load_counterfactual_suite_manifest,
)
from experiment_config import (  # noqa: E402
    RLExperimentConfig,
    require_resume_base_model_identity,
    runtime_content_tree_sha256,
    verify_base_model_identity,
    validate_runtime_identity,
)
from reference_result_filter import filter_training_records  # noqa: E402
from task_loader import load_rl_task_records  # noqa: E402
from frameworks.trl.rollout import (  # noqa: E402
    RolloutSettings,
    TableAgentRolloutCollector,
    TOOL_ENVIRONMENT_FACTORY_MODULE,
)
from frameworks.trl.fixed_rollout_pool import FixedPoolRolloutCollector  # noqa: E402
from frameworks.trl.checkpoint_gate import (  # noqa: E402
    CheckpointGateSpec,
    build_checkpoint_gate_spec,
    run_checkpoint_gate,
)
from frameworks.trl.transition_grpo import TransitionGRPOTrainer  # noqa: E402
from frameworks.trl.training_precision import (  # noqa: E402
    optimizer_moment_precision_audit,
    promote_trainable_parameters_to_fp32,
    require_adam_moments_fp32,
    trainable_parameter_precision_audit,
    require_trainable_parameters_fp32,
)


FROZEN_REFERENCE_ADAPTER_NAME = "frozen_sft_reference"


def _distributed_env() -> tuple[int, int, int]:
    """Return ``(world_size, rank, local_rank)`` from torchrun's environment."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise RuntimeError(
            "invalid distributed environment: "
            f"WORLD_SIZE={world_size}, RANK={rank}, LOCAL_RANK={local_rank}"
        )
    return world_size, rank, local_rank


def _init_distributed() -> tuple[int, int, int]:
    """Select the local CUDA device before loading a per-rank model replica."""

    world_size, rank, local_rank = _distributed_env()
    if world_size == 1:
        return world_size, rank, local_rank
    if not torch.cuda.is_available():
        raise RuntimeError("multi-GPU training requires CUDA")
    torch.cuda.set_device(local_rank)
    return world_size, rank, local_rank


def _start_distributed_process_group() -> None:
    """Start NCCL only after PEFT has loaded the adapter state dict.

    PEFT 0.19 detects an initialized device mesh and attempts to use the
    Transformers tensor-parallel sharder. The host's Transformers 4.57.6 lacks
    one symbol that PEFT imports, so initializing NCCL before adapter loading
    turns ordinary DDP into an unrelated TP compatibility failure.
    """

    world_size, _, local_rank = _distributed_env()
    if world_size <= 1 or torch.distributed.is_initialized():
        return
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")


def _distributed_barrier() -> None:
    world_size, _, _ = _distributed_env()
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _is_main_process() -> bool:
    return _distributed_env()[1] == 0


class TrainingPrecisionGuardCallback(TrainerCallback):
    """Fail on the first completed update if mixed precision touched optimizer state."""

    def __init__(
        self,
        *,
        require_fp32_adam_moments: bool,
        require_fp32_trainable: bool = True,
    ):
        self.require_fp32_adam_moments = require_fp32_adam_moments
        self.require_fp32_trainable = require_fp32_trainable

    def on_step_end(self, args, state, control, **kwargs):
        if self.require_fp32_trainable:
            require_trainable_parameters_fp32(kwargs["model"])
        if self.require_fp32_adam_moments:
            optimizer_audit = optimizer_moment_precision_audit(kwargs["optimizer"])
            # A homogeneous zero-advantage group may legitimately create no Adam state.
            # Once any moment exists, however, it must be FP32 immediately.
            if optimizer_audit["moment_tensors"]:
                require_adam_moments_fp32(kwargs["optimizer"])
        return control


class SynchronousCheckpointGateCallback(TrainerCallback):
    """Block the next optimizer step until a source-locked checkpoint gate passes."""

    def __init__(self, spec: CheckpointGateSpec):
        self.spec = spec
        self.executed = False

    def on_save(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step < self.spec.step:
            return control
        if step > self.spec.step:
            if not self.spec.receipt.is_file() or self.spec.receipt.is_symlink():
                raise RuntimeError(
                    "training advanced beyond the checkpoint gate without a valid receipt"
                )
            return control
        if self.executed:
            raise RuntimeError("checkpoint gate was invoked more than once in one trainer process")
        run_checkpoint_gate(self.spec, verify_existing=False)
        self.executed = True
        return control


def parse_args() -> tuple[argparse.Namespace, RLExperimentConfig | None]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--experiment-config", type=Path)
    bootstrap_args, _ = bootstrap.parse_known_args()
    experiment_config = (
        RLExperimentConfig.load(bootstrap_args.experiment_config)
        if bootstrap_args.experiment_config is not None
        else None
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--examples-json", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--example-index",
        type=int,
        help="retain one exact example_index for deterministic debugging",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--reward-mode", choices=("result-only", "process"), default="result-only")
    parser.add_argument(
        "--result-reward-profile",
        choices=("binary", "execution-ladder", "four-level"),
        default="binary",
    )
    parser.add_argument(
        "--policy-reduction",
        choices=(
            "transition_mean",
            "trajectory_mean",
            "trajectory_token_mean",
        ),
        default="transition_mean",
    )
    parser.add_argument(
        "--credit-assignment",
        choices=("trajectory", "saam-strict", "saam-asymmetric-error"),
        default="trajectory",
        help=(
            "trajectory keeps vanilla terminal GRPO credit; saam-strict zeros only "
            "mixed-outcome exact state-action coefficients without renormalization; "
            "saam-asymmetric-error keeps correct shared actions, suppresses wrong "
            "shared actions, and applies local deterministic-error penalties"
        ),
    )
    parser.add_argument(
        "--error-penalty",
        type=float,
        default=1.0,
        help="absolute local advantage floor for deterministic Harness errors",
    )
    parser.add_argument(
        "--record-gradient-conflicts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="record full, reason-only, and tool-only pre-clip gradient geometry per update",
    )
    parser.add_argument(
        "--gradient-conflict-save-vectors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also persist full FP32 gradient vectors (default stores exact stats plus sketches)",
    )
    parser.add_argument(
        "--gradient-conflict-max-transitions",
        type=int,
        default=0,
        help=(
            "bound each diagnostic probe to this many deterministic transitions; "
            "0 keeps the full update (training remains full-batch)"
        ),
    )
    parser.add_argument(
        "--gradient-conflict-carrier-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "record only all-advantage reason/tool carrier gradients; skip the "
            "positive/negative full-response probes"
        ),
    )
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--protocol-runtime-root", type=Path)
    parser.add_argument("--expected-runtime-content-tree-sha256")
    parser.add_argument("--expected-protocol-version")
    parser.add_argument("--expected-protocol-hash")
    parser.add_argument("--expected-student-prompt-sha256")
    parser.add_argument("--expected-initial-adapter-sha256")
    parser.add_argument("--expected-reference-adapter-sha256")
    parser.add_argument(
        "--expected-base-model-identity",
        type=json.loads,
        help=(
            "JSON form of the exact 12-file base-model identity; normally supplied "
            "by the experiment config"
        ),
    )
    parser.add_argument("--process-reward-config", type=Path)
    parser.add_argument(
        "--process-admission-policy",
        choices=(
            "counterfactual-completeness",
            "counterfactual-screened",
            "rank-local-features",
            "dense-outcome",
            "denotation-nonempty",
        ),
        default="counterfactual-completeness",
    )
    parser.add_argument("--counterfactual-suite-manifest", type=Path)
    parser.add_argument("--exclude-empty-reference-results", action="store_true")
    parser.add_argument("--optimizer-steps", type=int, default=200)
    parser.add_argument("--ppo-iterations", type=int, default=2)
    parser.add_argument("--prompts-per-update", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument(
        "--transition-micro-batch-size",
        type=int,
        default=2,
        help=(
            "fixed transition row count; when --transition-micro-batch-tokens "
            "is set, this becomes the hard maximum row count"
        ),
    )
    parser.add_argument(
        "--transition-micro-batch-tokens",
        type=int,
        default=0,
        help=(
            "padded prompt+completion token budget per transition micro-batch; "
            "zero preserves fixed-row batching"
        ),
    )
    parser.add_argument("--policy-loss-coefficient", type=float, default=1.0)
    parser.add_argument("--train-turns", choices=("all", "last"), default="all")
    parser.add_argument("--trainable-part", choices=("all", "tool_only"), default="all")
    parser.add_argument(
        "--span-balance-alpha",
        type=float,
        default=None,
        help=(
            "when set with trainable-part=all, give alpha of each turn's loss "
            "to the tool span and 1-alpha to the reasoning span"
        ),
    )
    parser.add_argument("--rank-loss-coefficient", type=float, default=0.0)
    parser.add_argument("--rank-beta", type=float, default=0.1)
    parser.add_argument(
        "--rank-score-tokens",
        choices=("all", "tool_only"),
        default="all",
    )
    parser.add_argument(
        "--rank-update-scope",
        choices=("full_trajectory", "conservative_legal", "dense_outcome"),
        default="full_trajectory",
    )
    parser.add_argument(
        "--rank-score-scope",
        choices=("full_trajectory", "conservative_legal", "dense_outcome"),
        default="full_trajectory",
    )
    parser.add_argument(
        "--rank-score-reduction",
        choices=("sum_tokens", "mean_action"),
        default="sum_tokens",
    )
    parser.add_argument(
        "--optimizer-name",
        choices=("adamw_torch", "paged_adamw_8bit"),
        default="paged_adamw_8bit",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("constant", "cosine"),
        default="constant",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--kl-beta", type=float, default=0.001)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--clip-epsilon-high", type=float, default=0.2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--max-agent-steps", type=int, default=30)
    parser.add_argument("--max-batch-calls", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "explicitly bind a model chat template's thinking mode; omit for models "
            "whose template has no such switch"
        ),
    )
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--checkpoint-gate-step", type=int, default=0)
    parser.add_argument("--checkpoint-gate-script", type=Path)
    parser.add_argument("--checkpoint-gate-receipt", type=Path)
    parser.add_argument("--checkpoint-gate-tasks-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--fixed-rollout-pool", type=Path)
    parser.add_argument("--fixed-pool-manifest", type=Path)
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-group-port", type=int, default=51216)
    parser.add_argument(
        "--trainer-sharding",
        choices=("replicated", "fsdp"),
        default=os.environ.get("TABLE_RL_TRAINER_SHARDING", "replicated"),
        help=(
            "replicated keeps one complete model per trainer rank (DDP); fsdp "
            "uses one FULL_SHARD model across the trainer ranks"
        ),
    )
    parser.add_argument(
        "--fsdp-base-storage",
        choices=("bf16", "4bit"),
        default=os.environ.get("TABLE_RL_FSDP_BASE_STORAGE", "bf16"),
        help=(
            "base weight storage for fsdp; bf16 is the supported sharded path, "
            "while 4bit is rejected until Params4bit sharding is validated"
        ),
    )
    if experiment_config is not None:
        parser.set_defaults(**experiment_config.argparse_defaults(ROOT))
    return parser.parse_args(), experiment_config


def load_process_config(path: Path | None):
    from process_credit import ProcessRewardConfig

    if path is None:
        return ProcessRewardConfig()
    values = {
        key: value
        for key, value in json.loads(path.read_text(encoding="utf-8")).items()
        if not key.startswith("_")
    }
    config = ProcessRewardConfig(**values)
    config.validate()
    return config


def sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace an identity artifact without exposing a partially written file."""

    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def merge_distributed_rollouts(output_dir: Path, world_size: int) -> None:
    """Merge rank-local rollout logs after a successful distributed update."""

    if world_size <= 1:
        return
    sources = [
        output_dir / f"rollouts.rank{rank}.jsonl"
        for rank in range(world_size)
    ]
    temporary = output_dir / f".rollouts.jsonl.{os.getpid()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as target:
            for source in sources:
                if source.is_file():
                    target.write(source.read_text(encoding="utf-8"))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output_dir / "rollouts.jsonl")
    finally:
        temporary.unlink(missing_ok=True)


def implementation_source_sha256(
    extra_relative_paths: tuple[str, ...] = (),
) -> dict[str, str]:
    """Bind a training artifact to the exact executable RL implementation."""
    result = {}
    for relative_path in (*IMPLEMENTATION_SOURCE_FILES, *extra_relative_paths):
        if relative_path in result:
            continue
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing implementation source: {path}")
        digest = sha256_file(path)
        if digest is None:  # pragma: no cover - guarded by the path check
            raise RuntimeError(f"could not hash implementation source: {path}")
        result[relative_path] = digest
    return result


def protocol_module_path_audit(runtime_root: Path | None) -> dict[str, Any]:
    """Prove that imports resolved from the runtime selected before startup."""

    expected_root = (runtime_root or ROOT).resolve()
    modules = {
        "protocol": (
            protocol_runtime,
            expected_root / "src" / "sft" / "protocol.py",
        ),
        "rollout": (
            evaluator_runtime,
            expected_root / "src" / "eval" / "rollout.py",
        ),
        "executor": (
            executor_runtime,
            expected_root / "src" / "harness" / "executor.py",
        ),
        "tool_registry": (
            tool_schemes_runtime,
            ROOT / "src" / "tool_modules" / "registry.py",
        ),
    }
    module_paths = {}
    for name, (module, expected_path) in modules.items():
        actual_path = Path(inspect.getfile(module)).resolve()
        expected_path = expected_path.resolve()
        if actual_path != expected_path:
            raise RuntimeError(
                f"{name} import escaped its pinned runtime: "
                f"{actual_path} != {expected_path}"
            )
        module_paths[name] = str(actual_path)
    expected_factory = "tool_environment_v26"
    if TOOL_ENVIRONMENT_FACTORY_MODULE != expected_factory:
        raise RuntimeError(
            "rollout environment does not match the imported protocol: "
            f"{TOOL_ENVIRONMENT_FACTORY_MODULE} != {expected_factory}"
        )
    return {
        "runtime_root": str(expected_root),
        "module_paths": module_paths,
        "tool_environment_factory_module": TOOL_ENVIRONMENT_FACTORY_MODULE,
    }


def snapshot_implementation_sources(
    output_dir: Path,
    expected_hashes: dict[str, str],
) -> None:
    """Atomically preserve or verify the exact sources imported by a run."""
    snapshot_root = output_dir / "implementation_source_snapshot"
    for relative_path, expected_hash in expected_hashes.items():
        source = ROOT / relative_path
        destination = snapshot_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            actual_hash = sha256_file(destination)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    "implementation source changed across resume: "
                    f"{relative_path} expected={expected_hash} actual={actual_hash}"
                )
            continue
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != expected_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"implementation snapshot hash mismatch: {relative_path}")
        temporary.replace(destination)


def write_implementation_lock(
    output_dir: Path,
    manifest: dict[str, Any],
) -> None:
    """Atomically lock the identity-bearing sources and initialization contract."""
    lock = {
        "schema_version": "trl-implementation-lock-v1",
        "files": manifest["implementation_source_sha256"],
        "initial_adapter_sha256": manifest["initial_adapter_sha256"],
        "protocol_version": manifest["protocol_version"],
        "protocol_hash": manifest["protocol_hash"],
        "student_prompt_sha256": manifest["student_prompt_sha256"],
        "base_model_identity": manifest["base_model_identity"],
        "reference_policy": manifest["reference_policy"],
        "experiment_config_sha256": manifest["experiment_config_sha256"],
        "checkpoint_gate": manifest["checkpoint_gate"],
        "credit_assignment": manifest["credit_assignment"],
        "error_penalty": manifest.get("error_penalty", 1.0),
        "gradient_conflict_logging": manifest.get("gradient_conflict_logging"),
    }
    path = output_dir / "implementation_lock.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != lock:
            raise RuntimeError("implementation lock changed across resume")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _adapter_weight_path(adapter_path: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter_path / name
        if candidate.is_file():
            return candidate
    raise ValueError(f"adapter has no weight file: {adapter_path}")


def validate_fixed_pool_manifest(args: argparse.Namespace) -> dict[str, Any] | None:
    """Fail closed unless offline training consumes the exact frozen SFT2 K4 pool."""
    if args.fixed_pool_manifest is None:
        return None
    manifest_path = args.fixed_pool_manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_status = {
        "counterfactual-completeness": "frozen_passed",
        "counterfactual-screened": "frozen_process_screened",
        "rank-local-features": "frozen_rank_ready",
        "dense-outcome": "frozen_dense_ready",
    }.get(args.process_admission_policy)
    if expected_status is None:
        raise ValueError(
            "fixed-pool training requires a fixed-pool admission policy, got "
            f"{args.process_admission_policy!r}"
        )
    manifest_tasks = int(payload.get("tasks") or 0)
    manifest_trajectories = int(payload.get("trajectories") or 0)
    if manifest_tasks < 1 or manifest_trajectories != manifest_tasks * args.group_size:
        raise ValueError("fixed-pool manifest task/trajectory counts are inconsistent")
    expected_scalars = {
        "schema_version": "table-agent-fixed-rollout-pool-v1",
        "status": expected_status,
        "group_size": 4,
        "protocol_version": "version26",
        "temperature": 0.7,
        "top_p": 0.95,
        "max_steps": 30,
        "history_turns": 4,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"fixed-pool manifest {key} mismatch: {payload.get(key)!r} != {expected!r}"
            )
    runtime_scalars = {
        "group_size": args.group_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_steps": args.max_agent_steps,
        "history_turns": args.history_turns,
    }
    for key, actual in runtime_scalars.items():
        if actual != expected_scalars[key]:
            raise ValueError(
                f"offline runtime {key} mismatch: {actual!r} != {expected_scalars[key]!r}"
            )
    difficulty_counts = payload.get("difficulty_counts") or {}
    if set(difficulty_counts) != {"challenging", "moderate", "simple"}:
        raise ValueError("fixed-pool manifest is missing difficulty counts")
    if sum(int(value) for value in difficulty_counts.values()) != manifest_tasks:
        raise ValueError("fixed-pool difficulty counts do not sum to tasks")
    if len({int(value) for value in difficulty_counts.values()}) != 1:
        raise ValueError("fixed-pool manifest is not difficulty-balanced")
    reuse = payload.get("reuse_contract") or {}
    same_trajectories = reuse.get("same_trajectories") is True or (
        manifest_trajectories == 240 and reuse.get("same_240_trajectories") is True
    )
    if not same_trajectories or not all(
        reuse.get(key) is True
        for key in ("same_sequence", "online_resampling_forbidden")
    ):
        raise ValueError("fixed-pool reuse contract is incomplete")
    files = payload.get("files") or {}
    pool_record = files.get("validated_trajectories") or {}
    pool_path = args.fixed_rollout_pool.resolve()
    recorded_pool_path = Path(str(pool_record.get("path", ""))).resolve()
    if recorded_pool_path != pool_path:
        raise ValueError(
            f"fixed rollout path mismatch: {pool_path} != {recorded_pool_path}"
        )
    actual_pool_sha = sha256_file(pool_path)
    if pool_record.get("sha256") != actual_pool_sha:
        raise ValueError("fixed rollout pool SHA-256 does not match frozen manifest")
    process_record = payload.get("process_reward_config") or {}
    actual_process_sha = sha256_file(args.process_reward_config)
    if process_record.get("sha256") != actual_process_sha:
        raise ValueError("process reward config SHA-256 does not match frozen manifest")
    actual_adapter_sha = sha256_file(_adapter_weight_path(args.adapter_path))
    if payload.get("initial_adapter_sha256") != actual_adapter_sha:
        raise ValueError("training adapter is not the frozen SFT2 checkpoint-1682 adapter")
    if args.optimizer_steps != manifest_tasks or args.prompts_per_update != 1:
        raise ValueError(
            "controlled fixed-pool training requires exactly one optimizer step per task"
        )
    return payload


def require_clean_output(path: Path, resume: Path | None) -> None:
    """Create a shared output directory once when torchrun has multiple ranks."""

    world_size, rank, _ = _distributed_env()
    if world_size > 1 and rank != 0:
        for _ in range(300):
            if path.is_dir():
                break
            time.sleep(0.1)
        if not path.is_dir():
            raise SystemExit(
                f"distributed rank started before output directory was created: {path}"
            )
        return
    if resume is not None:
        if not path.is_dir():
            raise SystemExit(
                f"resume output directory does not exist: {path}"
            )
    else:
        if path.exists() and any(path.iterdir()):
            raise SystemExit(
                f"output directory is not empty: {path}; use an isolated directory"
            )
        path.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        _distributed_barrier()


def resume_checkpoint_global_step(checkpoint: Path) -> int:
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file() or state_path.is_symlink():
        raise RuntimeError(f"resume checkpoint has no regular trainer_state.json: {checkpoint}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    step = state.get("global_step")
    if type(step) is not int or step < 0:
        raise RuntimeError(f"resume checkpoint has invalid global_step: {checkpoint}")
    expected_name = f"checkpoint-{step}"
    if checkpoint.name != expected_name:
        raise RuntimeError(
            f"resume checkpoint name/global_step mismatch: {checkpoint.name} != {expected_name}"
        )
    return step


def frozen_reference_adapter_audit(model, adapter_name: str) -> dict[str, Any]:
    """Prove that the selected KL adapter exists and cannot receive gradients."""

    if adapter_name not in getattr(model, "peft_config", {}):
        raise RuntimeError(f"missing frozen KL reference adapter: {adapter_name}")
    marker = f".{adapter_name}."
    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if marker in name
    ]
    if not parameters:
        raise RuntimeError(
            f"frozen KL reference adapter has no parameters: {adapter_name}"
        )
    trainable = [name for name, parameter in parameters if parameter.requires_grad]
    if trainable:
        raise RuntimeError(
            "frozen KL reference adapter exposes trainable parameters: "
            f"{trainable[:5]}"
        )
    return {
        "adapter_name": adapter_name,
        "parameter_tensors": len(parameters),
        "parameters": sum(parameter.numel() for _, parameter in parameters),
        "trainable_parameter_tensors": 0,
    }


def load_counterfactual_suites(
    manifest_path: Path | None,
    records: list[dict[str, Any]],
) -> tuple[CounterfactualSuiteManifest | None, dict[str, Any]]:
    if manifest_path is None:
        return None, {}
    manifest = load_counterfactual_suite_manifest(manifest_path)
    suites = {}
    for record in records:
        metadata = record["environment"]
        suites[str(metadata["task_id"])] = manifest.suite_for(metadata)
    return manifest, suites


def load_qlora_model(
    args: argparse.Namespace,
    *,
    trainer_sharding: str = "replicated",
    fsdp_base_storage: str = "bf16",
):
    if trainer_sharding not in {"replicated", "fsdp"}:
        raise ValueError(f"unsupported trainer sharding: {trainer_sharding}")
    if trainer_sharding == "fsdp" and fsdp_base_storage != "bf16":
        raise ValueError(
            "FSDP sharding currently requires bf16 base storage; refusing "
            "unvalidated Params4bit sharding"
        )
    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if trainer_sharding == "replicated"
        else None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    _, _, local_rank = _distributed_env()
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        # ``device_map={"": 0}`` is correct for the historical single-GPU
        # launcher, but would put every torchrun rank on the first visible GPU.
        # LOCAL_RANK is relative to CUDA_VISIBLE_DEVICES and keeps one complete
        # initialization on each rank before FSDP shards it.
        "device_map": {"": local_rank},
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if quantization is not None:
        model_kwargs["quantization_config"] = quantization
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    base_model.config.use_cache = False
    # The FSDP path deliberately loads an unquantized BF16 base so floating
    # parameters can be sharded.  ``prepare_model_for_kbit_training`` is a
    # QLoRA helper: on a BF16 model it casts frozen parameters to FP32, which
    # creates a second full copy before FSDP wrapping and can OOM a 40 GB GPU.
    # Activation checkpointing is supplied by the FSDP plugin instead.
    if trainer_sharding == "replicated":
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True,
        )
    # PEFT 0.19 unconditionally imports EmbeddingParallel while loading an
    # adapter, but Transformers 4.57.6 does not export that class.  We do not
    # use a Transformers tensor-parallel plan here; provide a compatibility
    # placeholder so ordinary DDP/FSDP adapter loading can proceed.
    import transformers.integrations.tensor_parallel as tensor_parallel

    if not hasattr(tensor_parallel, "EmbeddingParallel"):
        class EmbeddingParallel:  # noqa: N801 - mirrors Transformers' public name
            pass

        tensor_parallel.EmbeddingParallel = EmbeddingParallel
    adapter_path = args.resume_from_checkpoint or args.adapter_path
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=True,
        autocast_adapter_dtype=True,
    )
    reference_adapter_audit = None
    if args.kl_beta != 0.0:
        # The trainable default adapter may come from a resumed optimizer
        # checkpoint. The KL reference must always be reloaded from the immutable
        # initial SFT path instead of being copied from that resumed policy.
        model.load_adapter(
            args.adapter_path,
            adapter_name=FROZEN_REFERENCE_ADAPTER_NAME,
            is_trainable=False,
        )
        model.set_adapter("default")
        reference_adapter_audit = frozen_reference_adapter_audit(
            model,
            FROZEN_REFERENCE_ADAPTER_NAME,
        )
    if trainer_sharding == "fsdp":
        # The adapter checkpoint is FP32 while the unquantized FSDP base is
        # BF16.  FSDP cannot flatten mixed dtypes inside one decoder block, so
        # use BF16 adapter parameters for this isolated sharded diagnostic.
        # AdamW still keeps its moment buffers in FP32.  The production
        # replicated QLoRA path below retains its FP32 adapter contract.
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
                    parameter.data = parameter.data.to(dtype=torch.bfloat16)
        precision_audit = trainable_parameter_precision_audit(model)
    else:
        precision_audit = promote_trainable_parameters_to_fp32(model)
    if trainer_sharding == "replicated":
        model.gradient_checkpointing_enable()
    return model, tokenizer, precision_audit, reference_adapter_audit


def main() -> None:
    args, experiment_config = parse_args()
    world_size, rank, _ = _init_distributed()
    if args.trainer_sharding == "fsdp" and world_size < 2:
        raise SystemExit("FSDP trainer sharding requires at least two trainer ranks")
    if args.trainer_sharding == "fsdp" and args.fsdp_base_storage != "bf16":
        raise SystemExit(
            "FSDP trainer sharding currently requires --fsdp-base-storage bf16"
        )
    require_clean_output(args.output_dir, args.resume_from_checkpoint)
    if args.prompts_per_update < 1 or args.group_size < 2:
        raise SystemExit("prompts-per-update must be positive and group-size must be at least 2")
    if args.ppo_iterations < 1:
        raise SystemExit("ppo-iterations must be positive")
    if args.gradient_accumulation_steps < 1:
        raise SystemExit("gradient-accumulation-steps must be positive")
    if args.transition_micro_batch_tokens < 0:
        raise SystemExit("transition-micro-batch-tokens must be non-negative")
    if args.save_steps < 1 or args.save_total_limit < 1:
        raise SystemExit("save-steps and save-total-limit must be positive")
    try:
        checkpoint_gate_spec = build_checkpoint_gate_spec(
            project_root=ROOT,
            output_dir=args.output_dir,
            optimizer_steps=args.optimizer_steps,
            save_steps=args.save_steps,
            step=args.checkpoint_gate_step,
            script=args.checkpoint_gate_script,
            receipt=args.checkpoint_gate_receipt,
            tasks=args.examples_json,
            tasks_manifest=args.checkpoint_gate_tasks_manifest,
            initial_adapter=args.adapter_path,
        )
    except ValueError as exc:
        raise SystemExit(f"checkpoint gate contract failed: {exc}") from exc
    if args.expected_records is not None and args.expected_records < 1:
        raise SystemExit("expected-records must be positive")
    if not 0 < args.adam_beta1 < 1 or not 0 < args.adam_beta2 < 1:
        raise SystemExit("Adam beta values must be in (0, 1)")
    if args.clip_epsilon_high < args.clip_epsilon:
        raise SystemExit("clip-epsilon-high must be >= clip-epsilon")
    if args.kl_beta < 0.0:
        raise SystemExit("kl-beta must be non-negative")
    if args.rank_loss_coefficient < 0:
        raise SystemExit("rank-loss-coefficient must be non-negative")
    if args.policy_loss_coefficient < 0:
        raise SystemExit("policy-loss-coefficient must be non-negative")
    if args.policy_loss_coefficient == 0.0 and args.rank_loss_coefficient == 0.0:
        raise SystemExit("at least one policy or rank loss must be enabled")
    if args.credit_assignment in {"saam-strict", "saam-asymmetric-error"}:
        credit_name = args.credit_assignment
        if args.reward_mode != "result-only":
            raise SystemExit(f"{credit_name} requires --reward-mode result-only")
        if credit_name == "saam-strict" and args.result_reward_profile != "binary":
            raise SystemExit("saam-strict requires binary terminal rewards")
        if credit_name == "saam-asymmetric-error" and args.result_reward_profile not in {
            "binary",
            "four-level",
        }:
            raise SystemExit(
                "saam-asymmetric-error requires binary or four-level terminal rewards"
            )
        if args.train_turns != "all":
            raise SystemExit(f"{credit_name} requires --train-turns all")
        if args.policy_loss_coefficient <= 0.0:
            raise SystemExit(f"{credit_name} requires an enabled policy loss")
        if args.rank_loss_coefficient != 0.0:
            raise SystemExit(f"{credit_name} cannot mix a trajectory ranking loss")
        # SAAM permits a future matched KL arm.  Nonzero KL still requires the
        # immutable reference adapter and complete runtime identity contract.
    if not math.isfinite(args.error_penalty) or args.error_penalty <= 0.0:
        raise SystemExit("error-penalty must be finite and positive")
    if args.record_gradient_conflicts:
        if args.rank_loss_coefficient != 0.0:
            raise SystemExit("gradient conflict recording cannot mix a rank loss")
        if args.kl_beta != 0.0:
            raise SystemExit("gradient conflict recording requires --kl-beta 0")
        if args.policy_loss_coefficient == 0.0:
            raise SystemExit("gradient conflict recording requires an enabled policy loss")
        if args.gradient_accumulation_steps != 1:
            raise SystemExit(
                "gradient conflict recording requires --gradient-accumulation-steps 1"
            )
    if (args.fixed_rollout_pool is None) != (args.fixed_pool_manifest is None):
        raise SystemExit(
            "--fixed-rollout-pool and --fixed-pool-manifest must be provided together"
        )
    fixed_pool_manifest = validate_fixed_pool_manifest(args)
    if args.rank_beta <= 0:
        raise SystemExit("rank-beta must be positive")
    if (
        args.rank_update_scope == "conservative_legal"
        and args.rank_score_tokens != "tool_only"
    ):
        raise SystemExit(
            "conservative_legal rank updates require --rank-score-tokens tool_only"
        )
    if (
        args.rank_score_scope == "conservative_legal"
        and args.rank_update_scope != "conservative_legal"
    ):
        raise SystemExit(
            "conservative_legal rank scores require "
            "--rank-update-scope conservative_legal"
        )
    if args.rank_update_scope == "dense_outcome" and args.rank_score_tokens != "all":
        raise SystemExit("dense_outcome rank updates require --rank-score-tokens all")
    if (
        args.rank_score_scope == "dense_outcome"
        and args.rank_update_scope != "dense_outcome"
    ):
        raise SystemExit(
            "dense_outcome rank scores require --rank-update-scope dense_outcome"
        )
    allowed_experiment_statuses = {
        "allowed_result_only_control",
        "allowed_process_after_gates",
        "allowed_process_screened",
        "allowed_process_unscreened",
        "allowed_rank_only_control",
    }
    if (
        experiment_config is not None
        and experiment_config.admission_status not in allowed_experiment_statuses
    ):
        raise SystemExit(
            "this experiment config is evaluation-only or blocked by the process "
            f"admission gates: {experiment_config.admission_status}"
        )
    if (
        experiment_config is not None
        and experiment_config.admission_status == "allowed_process_screened"
        and (
            args.reward_mode != "process"
            or args.process_admission_policy != "counterfactual-screened"
            or args.fixed_pool_manifest is None
        )
    ):
        raise SystemExit(
            "allowed_process_screened requires process reward and a frozen screened pool"
        )
    if (
        experiment_config is not None
        and experiment_config.admission_status == "allowed_rank_only_control"
        and (
            args.reward_mode != "process"
            or args.process_admission_policy != "rank-local-features"
            or args.policy_loss_coefficient != 0.0
            or args.rank_loss_coefficient <= 0.0
            or args.fixed_pool_manifest is None
        )
    ):
        raise SystemExit(
            "allowed_rank_only_control requires a frozen rank pool, disabled process "
            "loss, and enabled rank loss"
        )
    if (
        experiment_config is not None
        and experiment_config.admission_status == "allowed_process_unscreened"
        and (
            args.reward_mode != "process"
            or args.process_admission_policy != "dense-outcome"
            or args.fixed_pool_manifest is None
            or args.policy_loss_coefficient <= 0.0
            or args.rank_loss_coefficient <= 0.0
            or args.trainable_part != "all"
            or args.rank_score_tokens != "all"
            or args.rank_score_scope != "dense_outcome"
            or args.rank_update_scope != "dense_outcome"
        )
    ):
        raise SystemExit(
            "allowed_process_unscreened requires the frozen dense pool, enabled "
            "process/rank losses, and full-response dense_outcome scoring"
        )
    if (
        experiment_config is not None
        and experiment_config.admission_status == "allowed_process_after_gates"
        and (
            args.reward_mode != "process"
            or args.process_admission_policy != "counterfactual-completeness"
            or (
                args.counterfactual_suite_manifest is None
                and args.fixed_pool_manifest is None
            )
        )
    ):
        raise SystemExit(
            "allowed_process_after_gates requires process reward, "
            "counterfactual-completeness, and a passed counterfactual manifest"
        )
    if (
        args.reward_mode == "process"
        and args.process_admission_policy == "counterfactual-completeness"
        and args.counterfactual_suite_manifest is None
        and args.fixed_pool_manifest is None
    ):
        raise SystemExit(
            "counterfactual-completeness requires --counterfactual-suite-manifest"
        )
    if (
        args.reward_mode == "process"
        and args.process_admission_policy == "denotation-nonempty"
        and not args.exclude_empty_reference_results
    ):
        raise SystemExit(
            "denotation-nonempty requires --exclude-empty-reference-results"
        )

    if (
        args.expected_runtime_content_tree_sha256 is not None
        and args.protocol_runtime_root is None
    ):
        raise SystemExit(
            "expected-runtime-content-tree-sha256 requires protocol-runtime-root"
        )
    parsed_runtime_root = (
        args.protocol_runtime_root.resolve()
        if args.protocol_runtime_root is not None
        else None
    )
    if parsed_runtime_root != PROTOCOL_RUNTIME_ROOT_AT_IMPORT:
        raise SystemExit(
            "protocol runtime changed after identity-bearing modules were imported: "
            f"{parsed_runtime_root} != {PROTOCOL_RUNTIME_ROOT_AT_IMPORT}"
        )
    actual_runtime_content_tree_sha256 = (
        runtime_content_tree_sha256(parsed_runtime_root)
        if parsed_runtime_root is not None
        else None
    )
    runtime_module_audit = protocol_module_path_audit(parsed_runtime_root)
    base_model_identity = None
    if args.expected_base_model_identity is not None:
        try:
            base_model_identity = verify_base_model_identity(
                args.model_path,
                args.expected_base_model_identity,
            )
            require_resume_base_model_identity(
                output_dir=args.output_dir,
                resume_checkpoint=args.resume_from_checkpoint,
                base_model_identity=base_model_identity,
            )
        except ValueError as exc:
            raise SystemExit(f"base model identity contract failed: {exc}") from exc
    runtime_prompt = protocol_runtime.student_runtime_system_prompt(
        context_mode="rolling-legal-history",
        compact=False,
    )
    runtime_prompt_sha256 = hashlib.sha256(
        runtime_prompt.encode("utf-8")
    ).hexdigest()
    actual_initial_adapter_sha256 = sha256_file(
        _adapter_weight_path(args.adapter_path)
    )
    if actual_initial_adapter_sha256 is None:  # pragma: no cover - path is required
        raise RuntimeError("could not hash the initial SFT adapter")
    try:
        runtime_identity_audit = validate_runtime_identity(
            expected_protocol_version=args.expected_protocol_version,
            expected_protocol_hash=args.expected_protocol_hash,
            expected_student_prompt_sha256=args.expected_student_prompt_sha256,
            expected_initial_adapter_sha256=args.expected_initial_adapter_sha256,
            expected_reference_adapter_sha256=(
                args.expected_reference_adapter_sha256
            ),
            expected_runtime_content_tree_sha256=(
                args.expected_runtime_content_tree_sha256
            ),
            actual_runtime_content_tree_sha256=(
                actual_runtime_content_tree_sha256
            ),
            actual_protocol_version=protocol_runtime.PROTOCOL_VERSION,
            actual_protocol_hash=protocol_runtime.protocol_hash(runtime_prompt),
            actual_student_prompt_sha256=runtime_prompt_sha256,
            actual_initial_adapter_sha256=actual_initial_adapter_sha256,
            kl_beta=args.kl_beta,
        )
    except ValueError as exc:
        raise SystemExit(f"runtime identity contract failed: {exc}") from exc

    records = load_rl_task_records(
        ROOT,
        split="train",
        selection=args.selection,
        examples_json=args.examples_json,
        limit=args.limit,
        seed=args.seed,
        context_mode="rolling-legal-history",
        system_prompt=runtime_prompt,
    )
    if args.example_index is not None:
        records = [
            record
            for record in records
            if int(record["environment"]["example_index"]) == args.example_index
        ]
    reference_filter_audit = []
    if args.exclude_empty_reference_results:
        records, reference_filter_audit = filter_training_records(records)
    if not records:
        raise SystemExit("no training records selected")
    if args.expected_records is not None and len(records) != args.expected_records:
        raise SystemExit(
            "training record count violates the frozen experiment contract: "
            f"{len(records)} != {args.expected_records}"
        )

    counterfactual_manifest, counterfactual_suites = (
        (None, {})
        if args.fixed_rollout_pool is not None
        else load_counterfactual_suites(args.counterfactual_suite_manifest, records)
    )
    process_config = (
        load_process_config(args.process_reward_config)
        if args.reward_mode == "process"
        else None
    )
    rollout_settings = RolloutSettings(
        reward_mode=args.reward_mode,
        result_reward_profile=args.result_reward_profile,
        process_admission_policy=args.process_admission_policy,
        tool_scheme="atomic",
        max_steps=args.max_agent_steps,
        max_batch_calls=args.max_batch_calls,
        max_new_tokens=args.max_new_tokens,
        max_context_tokens=args.max_context_tokens,
        context_mode="rolling-legal-history",
        history_turns=args.history_turns,
        denotation_comparison="bird-set",
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_thinking=args.enable_thinking,
    )

    per_device_batch = args.prompts_per_update * args.group_size
    training_args_kwargs = dict(
        output_dir=str(args.output_dir),
        max_steps=args.optimizer_steps,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        steps_per_generation=1,
        num_generations=args.group_size,
        num_iterations=args.ppo_iterations,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        # Accelerate's FSDP preparation upcasts trainable FlatParameters to
        # FP32 whenever TrainingArguments.bf16=True.  The sharded diagnostic
        # already loads the base and adapter in BF16; leave the Trainer AMP
        # switch off so it does not create a second full-precision flat copy.
        bf16=args.trainer_sharding != "fsdp",
        gradient_checkpointing=args.trainer_sharding != "fsdp",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_dropout=True,
        max_completion_length=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        use_vllm=args.fixed_rollout_pool is None,
        vllm_mode="server",
        vllm_server_host=args.vllm_host,
        vllm_server_port=args.vllm_port,
        vllm_group_port=args.vllm_group_port,
        vllm_importance_sampling_correction=True,
        vllm_importance_sampling_mode="token_truncate",
        vllm_importance_sampling_cap=3.0,
        importance_sampling_level="token",
        beta=args.kl_beta,
        epsilon=args.clip_epsilon,
        epsilon_high=args.clip_epsilon_high,
        loss_type="grpo",
        scale_rewards="none",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        log_completions=False,
    )
    if args.trainer_sharding == "fsdp":
        # FSDP1 FULL_SHARD is intentionally selected for the first model-sharded
        # test.  The Qwen3 decoder blocks are the communication units; keeping
        # ``use_orig_params`` allows the custom LoRA optimizer and precision
        # audit to retain their original Parameter objects.
        training_args_kwargs.update(
            fsdp="full_shard auto_wrap",
            fsdp_config={
                "transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
                "use_orig_params": True,
                # Every rank loads the same local checkpoint before wrapping;
                # avoid an extra full-model broadcast on the first test.
                "sync_module_states": False,
                "activation_checkpointing": True,
                "state_dict_type": "SHARDED_STATE_DICT",
            },
        )
    training_args = GRPOConfig(**training_args_kwargs)
    (
        model,
        tokenizer,
        trainable_precision_promotion,
        reference_adapter_load_audit,
    ) = load_qlora_model(
        args,
        trainer_sharding=args.trainer_sharding,
        fsdp_base_storage=args.fsdp_base_storage,
    )
    # Adapter loading must happen before NCCL creates a device mesh (see the
    # compatibility note in _start_distributed_process_group).
    _start_distributed_process_group()
    rollout_log_path = (
        args.output_dir / "rollouts.jsonl"
        if world_size == 1
        else args.output_dir / f"rollouts.rank{rank}.jsonl"
    )
    rollout_collector = (
        FixedPoolRolloutCollector(
            args.fixed_rollout_pool,
            expected_group_size=args.group_size,
        )
        if args.fixed_rollout_pool is not None
        else TableAgentRolloutCollector(
            tokenizer,
            rollout_settings,
            process_config=process_config,
            counterfactual_suites=counterfactual_suites,
            rollout_log_path=rollout_log_path,
        )
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = (
        torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
        )
        if args.optimizer_name == "adamw_torch"
        else bnb.optim.PagedAdamW8bit(
            trainable,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
        )
    )
    dataset = Dataset.from_list(records)
    trainer = TransitionGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, None),
        rollout_collector=rollout_collector,
        reward_mode=args.reward_mode,
        train_turns=args.train_turns,
        trainable_part=args.trainable_part,
        trainer_sharding=args.trainer_sharding,
        rank_loss_coefficient=args.rank_loss_coefficient,
        rank_beta=args.rank_beta,
        rank_score_tokens=args.rank_score_tokens,
        rank_score_scope=args.rank_score_scope,
        rank_score_reduction=args.rank_score_reduction,
        rank_update_scope=args.rank_update_scope,
        policy_loss_coefficient=args.policy_loss_coefficient,
        policy_reduction=args.policy_reduction,
        credit_assignment=args.credit_assignment,
        error_penalty=args.error_penalty,
        span_balance_alpha=args.span_balance_alpha,
        record_gradient_conflicts=args.record_gradient_conflicts,
        gradient_conflict_dir=args.output_dir / "gradient_conflicts",
        gradient_conflict_save_vectors=args.gradient_conflict_save_vectors,
        gradient_conflict_max_transitions=args.gradient_conflict_max_transitions,
        gradient_conflict_carrier_only=args.gradient_conflict_carrier_only,
        reference_adapter_name=(
            FROZEN_REFERENCE_ADAPTER_NAME if args.kl_beta != 0.0 else None
        ),
        transition_micro_batch_size=args.transition_micro_batch_size,
        transition_micro_batch_tokens=args.transition_micro_batch_tokens,
        callbacks=[
            TrainingPrecisionGuardCallback(
                require_fp32_adam_moments=args.optimizer_name == "adamw_torch",
                require_fp32_trainable=args.trainer_sharding != "fsdp",
            )
        ],
    )
    # TRL deliberately casts trainable QLoRA adapters to BF16 inside
    # GRPOTrainer.__init__. Restore the low-learning-rate precision contract
    # after that upstream cast and before the first optimizer step. The
    # optimizer retains references to the same Parameter objects and has no
    # initialized state yet.
    trainable_precision_after_trainer_init = promote_trainable_parameters_to_fp32(
        trainer.model
    ) if args.trainer_sharding != "fsdp" else trainable_parameter_precision_audit(
        trainer.model
    )
    if args.trainer_sharding != "fsdp":
        require_trainable_parameters_fp32(trainer.model)
    reference_adapter_after_trainer_audit = (
        frozen_reference_adapter_audit(
            trainer.model,
            FROZEN_REFERENCE_ADAPTER_NAME,
        )
        if args.kl_beta != 0.0
        else None
    )
    implementation_hashes = implementation_source_sha256(
        (
            (checkpoint_gate_spec.script_relative_path,)
            if checkpoint_gate_spec is not None
            else ()
        )
    )
    manifest = {
        "schema_version": "table-agent-trl-transition-grpo-v2",
        "trainer_parallelism": {
            "mode": (
                "fsdp_full_shard"
                if args.trainer_sharding == "fsdp"
                else ("ddp" if world_size > 1 else "single")
            ),
            "world_size": world_size,
            "base_storage": args.fsdp_base_storage,
            "rollout_log": str(rollout_log_path),
        },
        "framework": "trl",
        "framework_version": importlib.metadata.version("trl"),
        "torch_version": importlib.metadata.version("torch"),
        "transformers_version": importlib.metadata.version("transformers"),
        "vllm_version": importlib.metadata.version("vllm"),
        "accelerate_version": importlib.metadata.version("accelerate"),
        "peft_version": importlib.metadata.version("peft"),
        "bitsandbytes_version": importlib.metadata.version("bitsandbytes"),
        "precision_contract": {
            "trainable_parameters": (
                "bfloat16 (FSDP sharded diagnostic)"
                if args.trainer_sharding == "fsdp"
                else "float32"
            ),
            "optimizer_moments": (
                "float32"
                if args.optimizer_name == "adamw_torch"
                else "backend-managed"
            ),
            "frozen_base_compute": "bfloat16",
            "runtime_guard": "after_each_optimizer_step",
            "promotion": trainable_precision_promotion,
            "after_trainer_init": trainable_precision_after_trainer_init,
        },
        "protocol_version": protocol_runtime.PROTOCOL_VERSION,
        "protocol_hash": protocol_runtime.protocol_hash(runtime_prompt),
        "runtime_identity_audit": runtime_identity_audit,
        "runtime_module_audit": runtime_module_audit,
        "model_path": str(args.model_path),
        "base_model_identity": base_model_identity,
        "adapter_path": str(args.adapter_path),
        "initial_adapter_sha256": actual_initial_adapter_sha256,
        "reference_policy": {
            "schema_version": "trl-frozen-reference-policy-v1",
            "enabled": args.kl_beta != 0.0,
            "kl_beta": args.kl_beta,
            "adapter_name": (
                FROZEN_REFERENCE_ADAPTER_NAME if args.kl_beta != 0.0 else None
            ),
            "adapter_path": str(args.adapter_path) if args.kl_beta != 0.0 else None,
            "adapter_sha256": (
                actual_initial_adapter_sha256 if args.kl_beta != 0.0 else None
            ),
            "expected_adapter_sha256": args.expected_reference_adapter_sha256,
            "equals_initial_adapter": bool(
                runtime_identity_audit["reference"]["equals_initial_adapter"]
            ),
            "load_audit": reference_adapter_load_audit,
            "after_trainer_init_audit": reference_adapter_after_trainer_audit,
        },
        "implementation_source_sha256": implementation_hashes,
        "resume_from_checkpoint": (
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        ),
        "experiment_config": (
            experiment_config.to_dict()
            if experiment_config is not None
            else None
        ),
        "experiment_config_sha256": sha256_file(args.experiment_config),
        "checkpoint_gate": (
            checkpoint_gate_spec.manifest()
            if checkpoint_gate_spec is not None
            else None
        ),
        "reward_mode": args.reward_mode,
        "result_reward_profile": args.result_reward_profile,
        "policy_reduction": args.policy_reduction,
        "credit_assignment": args.credit_assignment,
        "error_penalty": args.error_penalty,
        "gradient_conflict_logging": {
            "enabled": bool(args.record_gradient_conflicts),
            "schema_version": "gradient-conflict-record-v3"
            if args.record_gradient_conflicts
            else None,
            "directory": (
                str(args.output_dir / "gradient_conflicts")
                if args.record_gradient_conflicts
                else None
            ),
            "save_vectors": bool(args.gradient_conflict_save_vectors),
            "max_probe_transitions": int(args.gradient_conflict_max_transitions),
            "carrier_only": bool(args.gradient_conflict_carrier_only),
        },
        "process_admission_policy": args.process_admission_policy,
        "records": len(records),
        "expected_records": args.expected_records,
        "example_index_filter": args.example_index,
        "group_size": args.group_size,
        "prompts_per_update": args.prompts_per_update,
        "optimizer_steps": args.optimizer_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "ppo_iterations": args.ppo_iterations,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "transition_micro_batch_size": args.transition_micro_batch_size,
        "transition_micro_batch_tokens": args.transition_micro_batch_tokens,
        "train_turns": args.train_turns,
        "trainable_part": args.trainable_part,
        "span_balance_alpha": args.span_balance_alpha,
        "policy_loss_coefficient": args.policy_loss_coefficient,
        "rank_loss_coefficient": args.rank_loss_coefficient,
        "rank_beta": args.rank_beta,
        "rank_score_tokens": args.rank_score_tokens,
        "rank_score_scope": args.rank_score_scope,
        "rank_score_reduction": args.rank_score_reduction,
        "rank_update_scope": args.rank_update_scope,
        "optimizer_name": args.optimizer_name,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": 1.0,
        "kl_beta": args.kl_beta,
        "clip_epsilon": args.clip_epsilon,
        "clip_epsilon_high": args.clip_epsilon_high,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "vllm_mode": "server",
        "vllm_importance_sampling_mode": "token_truncate",
        "vllm_importance_sampling_cap": 3.0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "rollout_settings": rollout_settings.__dict__,
        "student_prompt_sha256": runtime_prompt_sha256,
        "tool_schema_sha256": protocol_runtime.tool_schema_hash(),
        "examples_json_sha256": sha256_file(args.examples_json),
        "selection_sha256": sha256_file(args.selection),
        "process_reward_config_sha256": sha256_file(args.process_reward_config),
        "counterfactual_manifest_sha256": sha256_file(
            args.counterfactual_suite_manifest
        ),
        "fixed_rollout_pool": (
            str(args.fixed_rollout_pool) if args.fixed_rollout_pool else None
        ),
        "fixed_rollout_pool_sha256": sha256_file(args.fixed_rollout_pool),
        "fixed_pool_manifest": (
            str(args.fixed_pool_manifest) if args.fixed_pool_manifest else None
        ),
        "fixed_pool_manifest_sha256": sha256_file(args.fixed_pool_manifest),
        "fixed_pool_status": (
            fixed_pool_manifest.get("status") if fixed_pool_manifest is not None else None
        ),
        "fixed_pool_reuse_contract": (
            fixed_pool_manifest.get("reuse_contract")
            if fixed_pool_manifest is not None
            else None
        ),
        "counterfactual_manifest_schema": (
            COUNTERFACTUAL_SCHEMA_VERSION
            if counterfactual_manifest is not None
            else None
        ),
    }
    if _is_main_process():
        atomic_write_json(args.output_dir / "run_manifest.json", manifest)
        write_implementation_lock(args.output_dir, manifest)
        snapshot_implementation_sources(args.output_dir, implementation_hashes)
    _distributed_barrier()
    if checkpoint_gate_spec is not None:
        snapshot_gate_script = (
            args.output_dir
            / "implementation_source_snapshot"
            / checkpoint_gate_spec.script_relative_path
        )
        runtime_gate_spec = checkpoint_gate_spec.with_script(snapshot_gate_script)
        if args.resume_from_checkpoint is not None:
            resume_step = resume_checkpoint_global_step(args.resume_from_checkpoint)
            if resume_step >= runtime_gate_spec.step:
                if (
                    not runtime_gate_spec.receipt.is_file()
                    or runtime_gate_spec.receipt.is_symlink()
                ):
                    raise RuntimeError(
                        "resume at or beyond the checkpoint gate requires its immutable receipt"
                    )
                run_checkpoint_gate(runtime_gate_spec, verify_existing=True)
            elif runtime_gate_spec.receipt.exists():
                raise RuntimeError(
                    "checkpoint gate receipt exists before the resumed policy reached its step"
                )
        trainer.add_callback(SynchronousCheckpointGateCallback(runtime_gate_spec))
    if reference_filter_audit and _is_main_process():
        (args.output_dir / "reference_result_filter.json").write_text(
            json.dumps(
                {
                    "schema_version": "rl-empty-reference-task-filter-v1",
                    "retained_count": len(records),
                    "excluded_count": sum(
                        audit.excluded for audit in reference_filter_audit
                    ),
                    "tasks": [audit.to_dict() for audit in reference_filter_audit],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        )
    )
    _distributed_barrier()
    if _is_main_process():
        merge_distributed_rollouts(args.output_dir, world_size)
        final_trainable_precision = (
            trainable_parameter_precision_audit(trainer.model)
            if args.trainer_sharding == "fsdp"
            else require_trainable_parameters_fp32(trainer.model)
        )
        optimizer_precision = optimizer_moment_precision_audit(trainer.optimizer)
        if args.optimizer_name == "adamw_torch":
            optimizer_precision = require_adam_moments_fp32(trainer.optimizer)
        (args.output_dir / "training_precision.json").write_text(
            json.dumps(
                {
                    "schema_version": "table-agent-trl-training-precision-v1",
                    "optimizer_name": args.optimizer_name,
                    "trainable_parameters": final_trainable_precision,
                    "optimizer_state": optimizer_precision,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        trainer.save_model(str(args.output_dir / "final"))
        tokenizer.save_pretrained(args.output_dir / "final")


if __name__ == "__main__":
    main()
