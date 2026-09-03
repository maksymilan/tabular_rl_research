#!/usr/bin/env python3
"""Strict YAML configuration for comparable table-agent RL experiments."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "table-agent-rl-experiment-v1"
REWARD_TYPES = {
    "result",
    "process",
    "process_no_backslice",
    "process_no_normalize",
}
TRAINABLE_PARTS = {"all", "tool_only"}
RESULT_REWARD_PROFILES = {"binary", "execution-ladder", "four-level"}
POLICY_REDUCTIONS = {
    "transition_mean",
    "trajectory_mean",
    "trajectory_token_mean",
}
CREDIT_ASSIGNMENTS = {"trajectory", "saam-strict", "saam-asymmetric-error"}
RUNTIME_CONTRACT_KEYS = {
    "runtime_root",
    "runtime_content_tree_sha256",
    "protocol_version",
    "protocol_hash",
    "student_prompt_sha256",
    "initial_adapter_sha256",
    "reference_adapter_sha256",
    "base_model_identity",
}
BASE_MODEL_IDENTITY_SCHEMA_VERSION = "trl-base-model-identity-v1"
BASE_MODEL_IDENTITY_KEYS = {
    "schema_version",
    "aggregate_sha256",
    "files_sha256",
}
QWEN3_8B_BASE_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
RANK_SCORE_TOKENS = {"all", "tool_only"}
RANK_SCORE_SCOPES = {"full_trajectory", "conservative_legal", "dense_outcome"}
RANK_SCORE_REDUCTIONS = {"sum_tokens", "mean_action"}
RANK_UPDATE_SCOPES = {"full_trajectory", "conservative_legal", "dense_outcome"}
ADMISSION_STATUSES = {
    "allowed_result_only_control",
    "allowed_process_after_gates",
    "allowed_process_screened",
    "allowed_process_unscreened",
    "allowed_rank_only_control",
    "evaluation_only_existing_checkpoint",
    "blocked_pending_process_gates",
}


def _strict_keys(value: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {context} keys: {unknown}")


def _require_lower_hex(value: object, *, length: int, field: str) -> str:
    text = str(value or "")
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be exactly {length} lowercase hexadecimal characters")
    return text


def base_model_aggregate_sha256(files_sha256: dict[str, str]) -> str:
    """Hash the canonical UTF-8 ``filename<TAB>sha256<LF>`` record stream."""

    digest = hashlib.sha256()
    for filename in sorted(files_sha256, key=lambda value: value.encode("utf-8")):
        digest.update(filename.encode("utf-8"))
        digest.update(b"\t")
        digest.update(files_sha256[filename].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_base_model_identity_contract(
    value: object,
    *,
    field: str = "base_model_identity",
) -> dict[str, Any]:
    """Validate and normalize the exact Qwen3-8B base-model content identity."""

    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    _strict_keys(value, BASE_MODEL_IDENTITY_KEYS, field)
    missing = sorted(BASE_MODEL_IDENTITY_KEYS - set(value))
    if missing:
        raise ValueError(f"{field} is missing keys: {missing}")
    if value.get("schema_version") != BASE_MODEL_IDENTITY_SCHEMA_VERSION:
        raise ValueError(
            f"{field}.schema_version must be "
            f"{BASE_MODEL_IDENTITY_SCHEMA_VERSION!r}"
        )
    files = value.get("files_sha256")
    if not isinstance(files, dict):
        raise ValueError(f"{field}.files_sha256 must be a mapping")
    required_files = set(QWEN3_8B_BASE_MODEL_FILES)
    actual_files = set(files)
    if actual_files != required_files:
        missing_files = sorted(required_files - actual_files)
        extra_files = sorted(actual_files - required_files)
        raise ValueError(
            f"{field}.files_sha256 must contain exactly the 12 frozen files; "
            f"missing={missing_files} extra={extra_files}"
        )
    normalized_files = {
        filename: _require_lower_hex(
            files[filename],
            length=64,
            field=f"{field}.files_sha256[{filename!r}]",
        )
        for filename in QWEN3_8B_BASE_MODEL_FILES
    }
    aggregate = _require_lower_hex(
        value.get("aggregate_sha256"),
        length=64,
        field=f"{field}.aggregate_sha256",
    )
    computed_aggregate = base_model_aggregate_sha256(normalized_files)
    if aggregate != computed_aggregate:
        raise ValueError(
            f"{field}.aggregate_sha256 does not match its canonical file records: "
            f"{aggregate} != {computed_aggregate}"
        )
    return {
        "schema_version": BASE_MODEL_IDENTITY_SCHEMA_VERSION,
        "aggregate_sha256": aggregate,
        "files_sha256": normalized_files,
    }


def verify_base_model_identity(
    model_root: Path,
    expected_identity: object,
) -> dict[str, Any]:
    """Stream-hash all 12 required files and fail on any model-content drift."""

    expected = validate_base_model_identity_contract(
        expected_identity,
        field="expected_base_model_identity",
    )
    model_root = model_root.resolve()
    actual_files: dict[str, str] = {}
    for filename in QWEN3_8B_BASE_MODEL_FILES:
        path = model_root / filename
        if not path.is_file():
            raise ValueError(f"base model is missing required file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        expected_file = expected["files_sha256"][filename]
        if actual != expected_file:
            raise ValueError(
                f"base model SHA-256 mismatch for {filename}: "
                f"{actual} != {expected_file}"
            )
        actual_files[filename] = actual
    actual_identity = {
        "schema_version": BASE_MODEL_IDENTITY_SCHEMA_VERSION,
        "aggregate_sha256": base_model_aggregate_sha256(actual_files),
        "files_sha256": actual_files,
    }
    if actual_identity != expected:
        raise ValueError(
            "base model aggregate identity mismatch after per-file verification: "
            f"{actual_identity['aggregate_sha256']} != "
            f"{expected['aggregate_sha256']}"
        )
    return actual_identity


def require_resume_base_model_identity(
    *,
    output_dir: Path,
    resume_checkpoint: Path | None,
    base_model_identity: dict[str, Any] | None,
) -> None:
    """Require both durable run locks to bind the same model before resume."""

    if resume_checkpoint is None or base_model_identity is None:
        return
    output_dir = output_dir.resolve()
    resume_checkpoint = resume_checkpoint.resolve()
    if not resume_checkpoint.is_dir():
        raise ValueError(f"resume checkpoint is not a directory: {resume_checkpoint}")
    if resume_checkpoint.parent != output_dir:
        raise ValueError(
            "resume checkpoint must be an immediate child of its output directory: "
            f"{resume_checkpoint.parent} != {output_dir}"
        )
    expected = validate_base_model_identity_contract(base_model_identity)
    for filename in ("run_manifest.json", "implementation_lock.json"):
        path = output_dir / filename
        if not path.is_file():
            raise ValueError(f"resume is missing identity artifact: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"resume identity artifact is invalid: {path}: {exc}") from exc
        recorded = validate_base_model_identity_contract(
            payload.get("base_model_identity"),
            field=f"{filename}.base_model_identity",
        )
        if recorded != expected:
            raise ValueError(
                f"resume base model identity changed since {filename}: "
                f"{recorded['aggregate_sha256']} != "
                f"{expected['aggregate_sha256']}"
            )


def runtime_content_tree_sha256(runtime_root: Path) -> str:
    """Verify and hash exactly the source files exported by a frozen runtime.

    Interpreter, test, and OS caches are side effects, not runtime source.
    They are ignored so an interrupted/resumed run has the same identity,
    while any other added, removed, or changed file fails the frozen hash.
    The digest includes every admitted relative path and its bytes, making an
    independent exact-file-count gate redundant.
    """

    runtime_root = runtime_root.resolve()
    lock_path = runtime_root / "runtime_lock.json"
    if not lock_path.is_file():
        raise ValueError(f"protocol runtime is missing runtime_lock.json: {runtime_root}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    exported_paths = lock.get("exported_paths")
    if exported_paths != ["src/eval", "src/sft", "src/harness"]:
        raise ValueError(
            f"unsupported protocol runtime export contract: {exported_paths!r}"
        )
    generated_directories = {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
    generated_names = {".DS_Store"}
    generated_suffixes = {".pyc", ".pyo"}
    files = []
    for relative in exported_paths:
        directory = runtime_root / relative
        if not directory.is_dir():
            raise ValueError(f"protocol runtime is missing {relative}: {runtime_root}")
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and not any(part in generated_directories for part in path.parts)
            and path.name not in generated_names
            and path.suffix not in generated_suffixes
        )
    digest = hashlib.sha256()
    for path in sorted(
        files,
        key=lambda value: value.relative_to(runtime_root).as_posix(),
    ):
        relative = path.relative_to(runtime_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    actual = digest.hexdigest()
    expected = str(lock.get("content_tree_sha256") or "")
    if actual != expected:
        raise ValueError(
            f"protocol runtime content tree mismatch: {actual} != {expected}"
        )
    return actual


def validate_runtime_identity(
    *,
    expected_protocol_version: str | None,
    expected_protocol_hash: str | None,
    expected_student_prompt_sha256: str | None,
    expected_initial_adapter_sha256: str | None,
    expected_reference_adapter_sha256: str | None,
    expected_runtime_content_tree_sha256: str | None = None,
    actual_runtime_content_tree_sha256: str | None = None,
    actual_protocol_version: str,
    actual_protocol_hash: str,
    actual_student_prompt_sha256: str,
    actual_initial_adapter_sha256: str,
    kl_beta: float,
) -> dict[str, Any]:
    """Fail closed on the identity-bearing inputs of an online GRPO run."""

    expected = {
        "runtime_content_tree_sha256": expected_runtime_content_tree_sha256,
        "protocol_version": expected_protocol_version,
        "protocol_hash": expected_protocol_hash,
        "student_prompt_sha256": expected_student_prompt_sha256,
        "initial_adapter_sha256": expected_initial_adapter_sha256,
    }
    required_expected = {
        key: value
        for key, value in expected.items()
        if key != "runtime_content_tree_sha256"
    }
    supplied = {
        key: value for key, value in required_expected.items() if value is not None
    }
    if supplied and len(supplied) != len(required_expected):
        missing = sorted(set(required_expected) - set(supplied))
        raise ValueError(f"runtime identity contract is incomplete; missing {missing}")

    actual = {
        "runtime_content_tree_sha256": actual_runtime_content_tree_sha256,
        "protocol_version": actual_protocol_version,
        "protocol_hash": actual_protocol_hash,
        "student_prompt_sha256": actual_student_prompt_sha256,
        "initial_adapter_sha256": actual_initial_adapter_sha256,
    }
    compared_expected = dict(supplied)
    if expected_runtime_content_tree_sha256 is not None:
        compared_expected["runtime_content_tree_sha256"] = (
            expected_runtime_content_tree_sha256
        )
    for key, expected_value in compared_expected.items():
        if actual[key] != expected_value:
            raise ValueError(
                f"runtime identity mismatch for {key}: "
                f"{actual[key]!r} != {expected_value!r}"
            )

    reference = {
        "enabled": float(kl_beta) != 0.0,
        "expected_adapter_sha256": expected_reference_adapter_sha256,
        "actual_adapter_sha256": actual_initial_adapter_sha256,
        "equals_initial_adapter": False,
    }
    if reference["enabled"]:
        if expected_reference_adapter_sha256 is None:
            raise ValueError(
                "kl_beta > 0 requires an expected frozen reference adapter SHA-256"
            )
        if not supplied:
            raise ValueError(
                "kl_beta > 0 requires a complete pinned runtime identity contract"
            )
        if expected_reference_adapter_sha256 != expected_initial_adapter_sha256:
            raise ValueError(
                "the frozen KL reference must equal the pinned initial SFT adapter"
            )
        if actual_initial_adapter_sha256 != expected_reference_adapter_sha256:
            raise ValueError(
                "the loaded frozen KL reference does not match the pinned initial SFT adapter"
            )
        reference["equals_initial_adapter"] = True
    elif expected_reference_adapter_sha256 is not None and supplied:
        reference["equals_initial_adapter"] = (
            expected_reference_adapter_sha256 == actual_initial_adapter_sha256
        )

    return {
        "schema_version": "trl-runtime-identity-audit-v1",
        "pinned": bool(supplied),
        "expected": expected,
        "actual": actual,
        "reference": reference,
    }


@dataclass(frozen=True)
class RLExperimentConfig:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "RLExperimentConfig":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("experiment config must be a YAML mapping")
        _strict_keys(
            payload,
            {
                "schema_version",
                "experiment_name",
                "admission_status",
                "reward_type",
                "result_reward_profile",
                "policy_reduction",
                "credit_assignment",
                "error_penalty",
                "record_gradient_conflicts",
                "gradient_conflict_save_vectors",
                "expected_records",
                "runtime_contract",
                "process_reward_config",
                "process_admission_policy",
                "exclude_empty_reference_results",
                "process_loss",
                "trainable_part",
                "rank_loss",
                "optimizer",
                "rollout",
            },
            "experiment",
        )
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SCHEMA_VERSION!r}"
            )
        if not payload.get("experiment_name"):
            raise ValueError("experiment_name must be non-empty")
        if payload.get("reward_type") not in REWARD_TYPES:
            raise ValueError(f"unsupported reward_type: {payload.get('reward_type')}")
        if payload.get("result_reward_profile", "binary") not in RESULT_REWARD_PROFILES:
            raise ValueError(
                "unsupported result_reward_profile: "
                f"{payload.get('result_reward_profile')}"
            )
        if payload.get("policy_reduction", "transition_mean") not in POLICY_REDUCTIONS:
            raise ValueError(
                f"unsupported policy_reduction: {payload.get('policy_reduction')}"
            )
        if payload.get("credit_assignment", "trajectory") not in CREDIT_ASSIGNMENTS:
            raise ValueError(
                f"unsupported credit_assignment: {payload.get('credit_assignment')}"
            )
        error_penalty = float(payload.get("error_penalty", 1.0))
        if not math.isfinite(error_penalty) or error_penalty <= 0.0:
            raise ValueError("error_penalty must be finite and positive")
        for field in (
            "record_gradient_conflicts",
            "gradient_conflict_save_vectors",
        ):
            if field in payload and not isinstance(payload[field], bool):
                raise ValueError(f"{field} must be boolean")
        if "expected_records" in payload and int(payload["expected_records"]) < 1:
            raise ValueError("expected_records must be positive")
        if (
            payload["reward_type"] != "result"
            and payload.get("result_reward_profile", "binary") != "binary"
        ):
            raise ValueError("non-binary result rewards are valid only for result-only RL")
        if payload.get("trainable_part") not in TRAINABLE_PARTS:
            raise ValueError(
                f"unsupported trainable_part: {payload.get('trainable_part')}"
            )
        if payload.get("admission_status") not in ADMISSION_STATUSES:
            raise ValueError(
                f"unsupported admission_status: {payload.get('admission_status')}"
            )
        if (
            payload["reward_type"] == "result"
            and payload["admission_status"] != "allowed_result_only_control"
        ):
            raise ValueError(
                "result reward must use allowed_result_only_control admission"
            )
        if (
            payload["reward_type"] != "result"
            and payload["admission_status"] == "allowed_result_only_control"
        ):
            raise ValueError(
                "process rewards cannot use result-only admission"
            )
        if (
            payload["reward_type"] == "result"
            and payload["admission_status"] == "allowed_process_after_gates"
        ):
            raise ValueError(
                "result reward cannot use process-gated admission"
            )
        if payload["admission_status"] == "allowed_process_after_gates":
            if payload.get("process_admission_policy") != "counterfactual-completeness":
                raise ValueError(
                    "process-gated admission requires counterfactual-completeness"
                )
            if bool(payload.get("exclude_empty_reference_results")):
                raise ValueError(
                    "process-gated admission cannot use the weaker empty-reference filter"
                )

        runtime_contract = payload.get("runtime_contract")
        if runtime_contract is not None:
            if not isinstance(runtime_contract, dict):
                raise ValueError("runtime_contract must be a mapping")
            _strict_keys(
                runtime_contract,
                RUNTIME_CONTRACT_KEYS,
                "runtime_contract",
            )
            missing = sorted(RUNTIME_CONTRACT_KEYS - set(runtime_contract))
            if missing:
                raise ValueError(f"runtime_contract is missing keys: {missing}")
            if not str(runtime_contract.get("protocol_version") or ""):
                raise ValueError("runtime_contract.protocol_version must be non-empty")
            if not str(runtime_contract.get("runtime_root") or ""):
                raise ValueError("runtime_contract.runtime_root must be non-empty")
            _require_lower_hex(
                runtime_contract.get("runtime_content_tree_sha256"),
                length=64,
                field="runtime_contract.runtime_content_tree_sha256",
            )
            _require_lower_hex(
                runtime_contract.get("protocol_hash"),
                length=16,
                field="runtime_contract.protocol_hash",
            )
            for field in (
                "student_prompt_sha256",
                "initial_adapter_sha256",
                "reference_adapter_sha256",
            ):
                _require_lower_hex(
                    runtime_contract.get(field),
                    length=64,
                    field=f"runtime_contract.{field}",
                )
            if (
                runtime_contract["reference_adapter_sha256"]
                != runtime_contract["initial_adapter_sha256"]
            ):
                raise ValueError(
                    "runtime_contract reference adapter must equal the initial adapter"
                )
            runtime_contract["base_model_identity"] = (
                validate_base_model_identity_contract(
                    runtime_contract.get("base_model_identity"),
                    field="runtime_contract.base_model_identity",
                )
            )

        rank = payload.get("rank_loss") or {}
        optimizer = payload.get("optimizer") or {}
        rollout = payload.get("rollout") or {}
        if not all(isinstance(value, dict) for value in (rank, optimizer, rollout)):
            raise ValueError("rank_loss, optimizer, and rollout must be mappings")
        _strict_keys(
            rank,
            {
                "enabled",
                "coefficient",
                "beta",
                "score_tokens",
                "score_scope",
                "score_reduction",
                "update_scope",
            },
            "rank_loss",
        )
        _strict_keys(
            optimizer,
            {
                "name",
                "learning_rate",
                "weight_decay",
                "steps",
                "ppo_iterations",
                "lr_scheduler_type",
                "warmup_ratio",
                "kl_beta",
                "clip_epsilon",
                "clip_epsilon_high",
                "gradient_accumulation_steps",
                "adam_beta1",
                "adam_beta2",
            },
            "optimizer",
        )
        _strict_keys(
            rollout,
            {
                "prompts_per_update",
                "group_size",
                "max_agent_steps",
                "max_new_tokens",
                "max_context_tokens",
                "history_turns",
                "temperature",
                "top_p",
                "top_k",
                "enable_thinking",
            },
            "rollout",
        )
        if bool(rank.get("enabled")) and float(rank.get("coefficient", 0.0)) <= 0:
            raise ValueError("enabled rank_loss requires coefficient > 0")
        if rank.get("score_tokens", "all") not in RANK_SCORE_TOKENS:
            raise ValueError(
                f"unsupported rank_loss.score_tokens: {rank.get('score_tokens')}"
            )
        if rank.get("score_scope", "full_trajectory") not in RANK_SCORE_SCOPES:
            raise ValueError(
                f"unsupported rank_loss.score_scope: {rank.get('score_scope')}"
            )
        if rank.get("score_reduction", "sum_tokens") not in RANK_SCORE_REDUCTIONS:
            raise ValueError(
                "unsupported rank_loss.score_reduction: "
                f"{rank.get('score_reduction')}"
            )
        if rank.get("update_scope", "full_trajectory") not in RANK_UPDATE_SCOPES:
            raise ValueError(
                f"unsupported rank_loss.update_scope: {rank.get('update_scope')}"
            )
        if (
            rank.get("update_scope", "full_trajectory") == "conservative_legal"
            and rank.get("score_tokens", "all") != "tool_only"
        ):
            raise ValueError(
                "conservative_legal rank updates require score_tokens=tool_only"
            )
        if (
            rank.get("score_scope", "full_trajectory") == "conservative_legal"
            and rank.get("update_scope", "full_trajectory")
            != "conservative_legal"
        ):
            raise ValueError(
                "conservative_legal rank scores require conservative_legal updates"
            )
        if rank.get("update_scope", "full_trajectory") == "dense_outcome" and (
            rank.get("score_tokens", "all") != "all"
        ):
            raise ValueError("dense_outcome rank updates require score_tokens=all")
        if rank.get("score_scope", "full_trajectory") == "dense_outcome" and (
            rank.get("update_scope", "full_trajectory") != "dense_outcome"
        ):
            raise ValueError("dense_outcome rank scores require dense_outcome updates")
        if int(optimizer.get("steps", 0)) <= 0:
            raise ValueError("optimizer.steps must be positive")
        if float(optimizer.get("kl_beta", 0.0)) < 0.0:
            raise ValueError("optimizer.kl_beta must be non-negative")
        if float(optimizer.get("kl_beta", 0.0)) > 0.0 and runtime_contract is None:
            raise ValueError(
                "optimizer.kl_beta > 0 requires a pinned runtime_contract"
            )
        if int(rollout.get("group_size", 0)) < 2:
            raise ValueError("rollout.group_size must be at least 2")
        if payload.get("credit_assignment", "trajectory") in {
            "saam-strict",
            "saam-asymmetric-error",
        }:
            credit_name = payload.get("credit_assignment")
            if payload["reward_type"] != "result":
                raise ValueError(f"{credit_name} requires result reward")
            profile = payload.get("result_reward_profile", "binary")
            if credit_name == "saam-strict" and profile != "binary":
                raise ValueError("saam-strict requires binary terminal reward")
            if credit_name == "saam-asymmetric-error" and profile not in {
                "binary",
                "four-level",
            }:
                raise ValueError(
                    "saam-asymmetric-error requires binary or four-level terminal reward"
                )
            if not bool(payload.get("process_loss", True)):
                raise ValueError(f"{credit_name} requires an enabled policy loss")
            if bool(rank.get("enabled")):
                raise ValueError(f"{credit_name} cannot mix a rank loss")
            # A future KL arm is a matched SAAM ablation.  Nonzero KL remains
            # fail-closed behind the pinned runtime/reference checks above;
            # the baseline config keeps kl_beta=0 until that arm is registered.
        if bool(payload.get("record_gradient_conflicts", False)):
            if bool(rank.get("enabled")):
                raise ValueError("gradient conflict recording cannot mix a rank loss")
            if float(optimizer.get("kl_beta", 0.0)) != 0.0:
                raise ValueError("gradient conflict recording requires optimizer.kl_beta=0")
            if int(optimizer.get("gradient_accumulation_steps", 1)) != 1:
                raise ValueError(
                    "gradient conflict recording requires gradient_accumulation_steps=1"
                )
        if not bool(payload.get("process_loss", True)) and not bool(rank.get("enabled")):
            raise ValueError("at least one of process_loss or rank_loss must be enabled")
        if payload["admission_status"] == "allowed_rank_only_control":
            if bool(payload.get("process_loss", True)) or not bool(rank.get("enabled")):
                raise ValueError(
                    "rank-only admission requires disabled process loss and enabled rank loss"
                )
            if payload.get("process_admission_policy") != "rank-local-features":
                raise ValueError(
                    "rank-only admission requires process_admission_policy=rank-local-features"
                )
        if payload["admission_status"] == "allowed_process_screened":
            if not bool(payload.get("process_loss", True)):
                raise ValueError("process-screened admission requires enabled process loss")
            if payload.get("process_admission_policy") != "counterfactual-screened":
                raise ValueError(
                    "process-screened admission requires counterfactual-screened policy"
                )
        if payload["admission_status"] == "allowed_process_unscreened":
            if not bool(payload.get("process_loss", True)):
                raise ValueError("process-unscreened admission requires enabled process loss")
            if payload.get("process_admission_policy") != "dense-outcome":
                raise ValueError(
                    "process-unscreened admission requires dense-outcome policy"
                )
            if payload.get("trainable_part") != "all":
                raise ValueError(
                    "dense-outcome process admission requires trainable_part=all"
                )
            if not bool(rank.get("enabled")):
                raise ValueError(
                    "dense-outcome process admission requires enabled rank loss"
                )
            if (
                rank.get("score_scope") != "dense_outcome"
                or rank.get("update_scope") != "dense_outcome"
            ):
                raise ValueError(
                    "dense-outcome admission requires dense_outcome rank score/update scopes"
                )
        return cls(path=path, payload=payload)

    def argparse_defaults(self, project_root: Path) -> dict[str, Any]:
        reward_type = self.payload["reward_type"]
        rank = self.payload.get("rank_loss") or {}
        optimizer = self.payload.get("optimizer") or {}
        rollout = self.payload.get("rollout") or {}
        runtime_contract = self.payload.get("runtime_contract") or {}
        process_config = self.payload.get("process_reward_config")
        if process_config:
            process_config = Path(process_config)
            if not process_config.is_absolute():
                process_config = project_root / process_config
        return {
            "reward_mode": "result-only" if reward_type == "result" else "process",
            "result_reward_profile": self.payload.get(
                "result_reward_profile", "binary"
            ),
            "policy_reduction": self.payload.get(
                "policy_reduction", "transition_mean"
            ),
            "credit_assignment": self.payload.get(
                "credit_assignment", "trajectory"
            ),
            "error_penalty": float(self.payload.get("error_penalty", 1.0)),
            "record_gradient_conflicts": bool(
                self.payload.get("record_gradient_conflicts", False)
            ),
            "gradient_conflict_save_vectors": bool(
                self.payload.get("gradient_conflict_save_vectors", False)
            ),
            "expected_records": (
                int(self.payload["expected_records"])
                if "expected_records" in self.payload
                else None
            ),
            "protocol_runtime_root": (
                Path(runtime_contract["runtime_root"])
                if runtime_contract.get("runtime_root")
                else None
            ),
            "expected_runtime_content_tree_sha256": runtime_contract.get(
                "runtime_content_tree_sha256"
            ),
            "expected_protocol_version": runtime_contract.get("protocol_version"),
            "expected_protocol_hash": runtime_contract.get("protocol_hash"),
            "expected_student_prompt_sha256": runtime_contract.get(
                "student_prompt_sha256"
            ),
            "expected_initial_adapter_sha256": runtime_contract.get(
                "initial_adapter_sha256"
            ),
            "expected_reference_adapter_sha256": runtime_contract.get(
                "reference_adapter_sha256"
            ),
            "expected_base_model_identity": runtime_contract.get(
                "base_model_identity"
            ),
            "process_reward_config": process_config,
            "process_admission_policy": self.payload.get(
                "process_admission_policy",
                "counterfactual-completeness",
            ),
            "exclude_empty_reference_results": bool(
                self.payload.get("exclude_empty_reference_results", False)
            ),
            "trainable_part": self.payload["trainable_part"],
            "policy_loss_coefficient": (
                1.0 if bool(self.payload.get("process_loss", True)) else 0.0
            ),
            "rank_loss_coefficient": (
                float(rank.get("coefficient", 0.0))
                if rank.get("enabled")
                else 0.0
            ),
            "rank_beta": float(rank.get("beta", 0.1)),
            "rank_score_tokens": rank.get("score_tokens", "all"),
            "rank_score_scope": rank.get(
                "score_scope",
                "full_trajectory",
            ),
            "rank_score_reduction": rank.get(
                "score_reduction",
                "sum_tokens",
            ),
            "rank_update_scope": rank.get(
                "update_scope",
                "full_trajectory",
            ),
            "optimizer_name": optimizer.get("name", "adamw_torch"),
            "learning_rate": float(optimizer.get("learning_rate", 1e-6)),
            "weight_decay": float(optimizer.get("weight_decay", 0.01)),
            "optimizer_steps": int(optimizer["steps"]),
            "ppo_iterations": int(optimizer.get("ppo_iterations", 2)),
            "lr_scheduler_type": optimizer.get("lr_scheduler_type", "cosine"),
            "warmup_ratio": float(optimizer.get("warmup_ratio", 0.03)),
            "kl_beta": float(optimizer.get("kl_beta", 0.0)),
            "clip_epsilon": float(optimizer.get("clip_epsilon", 0.2)),
            "clip_epsilon_high": float(
                optimizer.get("clip_epsilon_high", optimizer.get("clip_epsilon", 0.2))
            ),
            "gradient_accumulation_steps": int(
                optimizer.get("gradient_accumulation_steps", 1)
            ),
            "adam_beta1": float(optimizer.get("adam_beta1", 0.9)),
            "adam_beta2": float(optimizer.get("adam_beta2", 0.999)),
            "prompts_per_update": int(rollout.get("prompts_per_update", 1)),
            "group_size": int(rollout["group_size"]),
            "max_agent_steps": int(rollout.get("max_agent_steps", 30)),
            "max_new_tokens": int(rollout.get("max_new_tokens", 1024)),
            "max_context_tokens": int(rollout.get("max_context_tokens", 8192)),
            "history_turns": int(rollout.get("history_turns", 4)),
            "temperature": float(rollout.get("temperature", 0.7)),
            "top_p": float(rollout.get("top_p", 0.95)),
            "top_k": int(rollout.get("top_k", 0)),
            "enable_thinking": (
                bool(rollout["enable_thinking"])
                if "enable_thinking" in rollout
                else None
            ),
        }

    @property
    def admission_status(self) -> str:
        return str(self.payload["admission_status"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)
