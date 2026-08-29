#!/usr/bin/env python3
"""Bind preregistered Arm B and run the shared strict v26 matched evaluator."""
from __future__ import annotations

import argparse
import copy
import json
import re
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from src.rl.diagnostics import validate_arm_b_training_inputs as admission
from src.rl.evaluation import formal_v26_matched_eval as formal


HERE = Path(__file__).resolve().parent
DEFAULT_OVERLAY = HERE / "qwen3_8b_v26_arm_b_formal_matched_contract.json"
DEFAULT_BASE = HERE / "qwen3_8b_v26_boundary300_formal_matched_contract.json"
OVERLAY_SCHEMA = "qwen3-8b-v26-arm-b-formal-matched-eval-overlay-v1"
GENERIC_SCHEMA = "qwen3-8b-v26-vanilla-formal-matched-eval-v1"
PLACEHOLDER = "__REQUIRED_EXPLICIT_BINDING_AT_LAUNCH__"
PATH_PLACEHOLDER = "__REQUIRED_EXPLICIT_PATH_AT_LAUNCH__"
TRAIN320_PLACEHOLDER = "__BOUND_FROM_TRAIN320_AT_LAUNCH__"
SHA_PATTERN = re.compile(r"[0-9a-f]{64}")
SHA_FIELDS = (
    "selection_manifest",
    "train320",
    "validation64_audit",
    "training_run_manifest",
    "training_implementation_lock",
    "arm_b_training_contract",
    "final_checkpoint32_adapter",
    "confirmatory_trigger",
)
PATH_FIELDS = (
    "selection_manifest",
    "train_tasks",
    "validation64_audit",
    "training_run",
    "confirmatory_trigger",
)
FORMAL_TRAINING_CONTRACT = admission.FORMAL_TRAINING_CONTRACT


def arm_b_implementation_paths() -> dict[str, Path]:
    return {
        "vanilla_grpo_arm_b.py": HERE.parent / "vanilla_grpo_arm_b.py",
        "validate_arm_b_training_inputs.py": (
            HERE.parent / "diagnostics/validate_arm_b_training_inputs.py"
        ),
        "prepare_vanilla_grpo_confirmatory_arm_trigger.py": (
            HERE.parent
            / "diagnostics/prepare_vanilla_grpo_confirmatory_arm_trigger.py"
        ),
    }


def verify_arm_b_implementation(overlay: dict[str, Any]) -> dict[str, str]:
    expected = overlay.get("arm_b_implementation_sha256") or {}
    paths = arm_b_implementation_paths()
    _require(set(expected), set(paths), "Arm B implementation files")
    observed: dict[str, str] = {}
    for name, path in paths.items():
        _regular(path, f"Arm B implementation {name}")
        observed[name] = formal.sha256_file(path)
        _require(observed[name], expected[name], f"Arm B implementation {name} SHA")
    return observed


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected={expected!r}, actual={actual!r}")


def _regular(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")


def _task_id(row: Mapping[str, Any]) -> str:
    value = row.get("example_id") or row.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError("Arm B task lacks example_id/instance_id")
    return value


def _normalized_absolute(value: str, label: str) -> Path:
    if not isinstance(value, str) or not value or value != str(Path(value)):
        raise ValueError(f"{label} must be a normalized absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a normalized absolute path")
    return path


def validate_overlay(overlay: dict[str, Any], base: dict[str, Any]) -> None:
    _require(overlay.get("schema_version"), OVERLAY_SCHEMA, "Arm B overlay schema")
    _require(
        overlay.get("status"),
        "preregistered_waiting_for_explicit_artifact_bindings",
        "Arm B overlay status",
    )
    base_record = overlay.get("base_contract") or {}
    _require(base_record.get("file"), DEFAULT_BASE.name, "base contract file")
    _require(base_record.get("sha256"), formal.sha256_file(DEFAULT_BASE), "base contract SHA")
    _require(base, formal.load_object(DEFAULT_BASE), "canonical common base contract")
    verify_arm_b_implementation(overlay)
    formal.validate_contract_shape(
        _generic_contract(overlay, base, {field: "0" * 64 for field in SHA_FIELDS}, _inert_paths(), validate=False)
    )
    training = overlay.get("arm_b_training") or {}
    for field, expected in {
        "selection_schema_version": "vanilla-grpo-arm-b-k16-selection-v1",
        "selection_status": "frozen_arm_b_training_cohort",
        "selected_records": 384,
        "train_records": 320,
        "validation_records": 64,
        "validation_schema_version": "vanilla-grpo-arm-b-k16-validation-audit-v1",
        "validation_status": admission.VALIDATION_STATUS,
        "primary_checkpoint": "final-step32-only",
    }.items():
        _require(training.get(field), expected, f"arm_b_training.{field}")
    _require(training.get("formal_training"), FORMAL_TRAINING_CONTRACT, "Arm B formal training")
    _require(training.get("formal_validation"), admission.SELECTION_VALIDATION_CONTRACT, "Arm B formal validation")
    _require(
        training.get("validation_audit_contract"),
        admission.VALIDATION_AUDIT_CONTRACT,
        "Arm B validation audit contract",
    )
    _require(set(training.get("dynamic_sha256") or {}), set(SHA_FIELDS), "Arm B dynamic SHA fields")
    for field in SHA_FIELDS:
        _require(training["dynamic_sha256"].get(field), PLACEHOLDER, f"Arm B placeholder {field}")
    checkpoint = training.get("checkpoint_policy") or {}
    _require(checkpoint.get("save_steps"), 4, "Arm B save_steps")
    _require(checkpoint.get("save_total_limit"), 1, "Arm B save_total_limit")
    _require(checkpoint.get("intermediate_steps"), [4, 8, 12, 16, 20, 24, 28], "Arm B intermediate steps")
    _require(checkpoint.get("intermediate_evaluation_allowed"), False, "Arm B intermediate eval policy")
    _require(checkpoint.get("primary_checkpoint"), "checkpoint-32", "Arm B primary checkpoint")
    _require(checkpoint.get("primary_global_step"), 32, "Arm B primary step")
    final = overlay.get("training_final") or {}
    _require(final.get("checkpoint_name"), "checkpoint-32", "Arm B final checkpoint")
    _require(final.get("global_step"), 32, "Arm B final step")
    expected_manifest = final.get("expected_manifest") or {}
    for field, expected in {
        "examples_json_sha256": TRAIN320_PLACEHOLDER,
        "records": 320,
        "expected_records": 320,
        "optimizer_steps": 32,
        "save_steps": 4,
        "save_total_limit": 1,
        "prompts_per_update": 20,
        "group_size": 16,
        "seed": 20260812,
        "result_reward_profile": "binary",
        "policy_reduction": "trajectory_token_mean",
        "learning_rate": 8e-7,
        "kl_beta": 0.0,
    }.items():
        _require(expected_manifest.get(field), expected, f"Arm B training manifest {field}")
    _require(overlay.get("promotion_gate"), base.get("promotion_gate"), "Arm B promotion gate")
    _require(
        set(overlay.get("host_path_overrides") or {}),
        {"allowed_run_parent", *PATH_FIELDS},
        "Arm B host path override fields",
    )
    for field, value in (overlay.get("host_path_overrides") or {}).items():
        if field != "allowed_run_parent":
            _require(value, PATH_PLACEHOLDER, f"Arm B path placeholder {field}")


def _inert_paths() -> dict[str, str]:
    return {
        "selection_manifest": "/home/dengyan/tabular_rl_outputs/inert_arm_b/selection/arm_b_selection_manifest.json",
        "train_tasks": "/home/dengyan/tabular_rl_outputs/inert_arm_b/selection/train320.jsonl",
        "validation64_audit": "/home/dengyan/tabular_rl_outputs/inert_arm_b/selection/validation64/arm_b_k16_validation_audit.json",
        "training_run": "/home/dengyan/tabular_rl_outputs/qwen3_8b_atomic_v26_arm_b_train320_k16_vanilla_grpo_20260812/train320_two_pass_k16_seed20260812",
        "confirmatory_trigger": "/home/dengyan/tabular_rl_outputs/inert_arm_b/confirmatory_arm_trigger.json",
    }


def validate_paths(overlay: dict[str, Any], paths: Mapping[str, str]) -> None:
    _require(set(paths), set(PATH_FIELDS), "Arm B supplied path fields")
    converted = {field: _normalized_absolute(value, field) for field, value in paths.items()}
    selection = converted["selection_manifest"]
    train = converted["train_tasks"]
    validation = converted["validation64_audit"]
    trigger = converted["confirmatory_trigger"]
    _require(selection.name, "arm_b_selection_manifest.json", "selection manifest name")
    _require(train.name, "train320.jsonl", "train320 name")
    _require(selection.parent, train.parent, "selection/train directory")
    _require(validation.name, "arm_b_k16_validation_audit.json", "validation audit name")
    _require(trigger.name, "confirmatory_arm_trigger.json", "confirmatory trigger name")
    roots = overlay["arm_b_training"]["allowed_artifact_roots"]
    selection_root = Path(roots["selection_runs"])
    for label, artifact in {
        "selection": selection,
        "train320": train,
        "validation64 audit": validation,
        "confirmatory trigger": trigger,
    }.items():
        if selection_root not in artifact.parents:
            raise ValueError(f"Arm B {label} path is outside allowed output root")
    training = converted["training_run"]
    _require(training.parent, Path(roots["training_run_parent"]), "Arm B training parent")
    _require(training.name, "train320_two_pass_k16_seed20260812", "Arm B training run name")


def _generic_contract(
    overlay: dict[str, Any],
    base: dict[str, Any],
    bindings: Mapping[str, str],
    paths: Mapping[str, str],
    *,
    validate: bool = True,
) -> dict[str, Any]:
    if validate:
        validate_overlay(overlay, base)
    _require(set(bindings), set(SHA_FIELDS), "Arm B supplied SHA fields")
    for field, digest in bindings.items():
        if not isinstance(digest, str) or SHA_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"{field} must be an explicit lowercase SHA-256")
    validate_paths(overlay, paths)
    contract = copy.deepcopy(base)
    contract.pop("boundary_training", None)
    contract["schema_version"] = GENERIC_SCHEMA
    contract["status"] = "arm_b_explicitly_bound_for_preflight_or_run"
    contract["purpose"] = overlay["purpose"]
    contract["host_paths"]["allowed_run_parent"] = overlay["host_path_overrides"]["allowed_run_parent"]
    contract["host_paths"]["boundary_manifest"] = paths["selection_manifest"]
    contract["host_paths"]["train_tasks"] = paths["train_tasks"]
    contract["host_paths"]["training_run"] = paths["training_run"]
    contract["training_final"] = copy.deepcopy(overlay["training_final"])
    contract["training_final"]["expected_manifest"]["examples_json_sha256"] = bindings["train320"]
    contract["promotion_gate"] = copy.deepcopy(overlay["promotion_gate"])
    contract["arm_b_training"] = copy.deepcopy(overlay["arm_b_training"])
    contract["arm_b_training"]["dynamic_sha256"] = dict(bindings)
    contract["arm_b_training"]["binding_status"] = "explicitly_bound"
    contract["confirmatory_trigger_contract"] = copy.deepcopy(overlay["confirmatory_trigger_contract"])
    contract["arm_b_paths"] = {
        "validation64_audit": paths["validation64_audit"],
        "confirmatory_trigger": paths["confirmatory_trigger"],
    }
    formal.validate_contract_shape(contract)
    return contract


def bind_contract(
    overlay: dict[str, Any],
    base: dict[str, Any],
    bindings: Mapping[str, str],
    paths: Mapping[str, str],
) -> dict[str, Any]:
    return _generic_contract(overlay, base, bindings, paths)


def verify_confirmatory_trigger(contract: dict[str, Any]) -> dict[str, Any]:
    trigger_path = Path(contract["arm_b_paths"]["confirmatory_trigger"])
    _regular(trigger_path, "confirmatory trigger")
    dynamic = contract["arm_b_training"]["dynamic_sha256"]
    _require(formal.sha256_file(trigger_path), dynamic["confirmatory_trigger"], "confirmatory trigger SHA")
    trigger = formal.load_object(trigger_path)
    expected = contract["confirmatory_trigger_contract"]
    for field in ("schema_version", "status", "admitted_arm", "confirmatory_arm_count", "full_dev_policy"):
        _require(trigger.get(field), expected[field], f"confirmatory trigger {field}")
    if trigger.get("reason") not in expected["allowed_reasons"]:
        raise ValueError("confirmatory trigger reason is not preregistered")
    _require(trigger.get("arm_a_training_started"), False, "trigger Arm A training state")
    _require(trigger.get("arm_a_full_dev_started"), False, "trigger Arm A full-dev state")
    selection_record = trigger.get("arm_b_selection_manifest") or {}
    validation_record = trigger.get("arm_b_validation_audit") or {}
    _require(Path(str(selection_record.get("path"))).resolve(), Path(contract["host_paths"]["boundary_manifest"]).resolve(), "trigger selection path")
    _require(selection_record.get("sha256"), dynamic["selection_manifest"], "trigger selection SHA")
    _require(Path(str(validation_record.get("path"))).resolve(), Path(contract["arm_b_paths"]["validation64_audit"]).resolve(), "trigger validation path")
    _require(validation_record.get("sha256"), dynamic["validation64_audit"], "trigger validation SHA")
    failure = trigger.get("arm_a_failure_evidence") or {}
    selection = formal.load_object(Path(contract["host_paths"]["boundary_manifest"]))
    source_record = selection.get("source_cohort_manifest") or {}
    _require(failure, source_record, "trigger source-cohort failure evidence binding")
    failure_path = Path(str(source_record.get("path") or ""))
    _regular(failure_path, "Arm A failure evidence/source cohort")
    failure_sha = str(source_record.get("sha256") or "")
    if SHA_PATTERN.fullmatch(failure_sha) is None:
        raise ValueError("Arm A failure evidence/source cohort lacks an explicit SHA-256")
    _require(formal.sha256_file(failure_path), failure_sha, "Arm A failure evidence/source cohort SHA")
    source = formal.load_object(failure_path)
    _require(source.get("schema_version"), source_record.get("schema_version"), "source cohort schema binding")
    _require(source.get("status"), source_record.get("status"), "source cohort status binding")
    activation = admission._verify_activation(source)
    _require(trigger.get("reason"), activation["reason"], "trigger/source cohort failure reason")
    for forbidden in expected["forbidden_existing_arm_a_paths"]:
        if Path(forbidden).exists():
            raise ValueError(f"Arm A training/evaluation marker already exists: {forbidden}")
    return {
        "path": str(trigger_path),
        "sha256": dynamic["confirmatory_trigger"],
        "admitted_arm": "arm_b",
        "confirmatory_arm_count": 1,
        "reason": trigger["reason"],
        "arm_a_failure_evidence_path": str(failure_path.resolve()),
        "arm_a_failure_evidence_sha256": failure_sha,
        "negative_fact_boundary": "filesystem marker absence is checked at preflight/run and requires non-concurrent orchestration",
    }


def verify_arm_b_artifacts(contract: dict[str, Any], training_run: Path) -> dict[str, Any]:
    dynamic = contract["arm_b_training"]["dynamic_sha256"]
    selection_path = Path(contract["host_paths"]["boundary_manifest"])
    train_path = Path(contract["host_paths"]["train_tasks"])
    validation_path = Path(contract["arm_b_paths"]["validation64_audit"])
    run_manifest_path = training_run / "run_manifest.json"
    lock_path = training_run / "implementation_lock.json"
    training_contract_path = training_run / "arm_b_training_contract.json"
    checkpoint_adapter = training_run / "checkpoint-32/adapter_model.safetensors"
    final_adapter = training_run / "final/adapter_model.safetensors"
    for label, path in {
        "selection manifest": selection_path,
        "train320": train_path,
        "validation64 audit": validation_path,
        "run manifest": run_manifest_path,
        "implementation lock": lock_path,
        "training contract": training_contract_path,
        "checkpoint32 adapter": checkpoint_adapter,
        "final adapter": final_adapter,
    }.items():
        _regular(path, label)
    observed = {
        "selection_manifest": formal.sha256_file(selection_path),
        "train320": formal.sha256_file(train_path),
        "validation64_audit": formal.sha256_file(validation_path),
        "training_run_manifest": formal.sha256_file(run_manifest_path),
        "training_implementation_lock": formal.sha256_file(lock_path),
        "arm_b_training_contract": formal.sha256_file(training_contract_path),
        "checkpoint32_adapter": formal.sha256_file(checkpoint_adapter),
        "final_adapter": formal.sha256_file(final_adapter),
    }
    for field in (
        "selection_manifest", "train320", "validation64_audit",
        "training_run_manifest", "training_implementation_lock", "arm_b_training_contract",
    ):
        _require(observed[field], dynamic[field], f"bound Arm B {field} SHA")
    expected_adapter = dynamic["final_checkpoint32_adapter"]
    _require(observed["checkpoint32_adapter"], expected_adapter, "checkpoint32 adapter SHA")
    _require(observed["final_adapter"], expected_adapter, "final adapter SHA")
    admission_report = admission.validate(
        selection_manifest_path=selection_path,
        expected_selection_manifest_sha256=dynamic["selection_manifest"],
        train320_path=train_path,
        expected_train320_sha256=dynamic["train320"],
        validation64_audit_path=validation_path,
        expected_validation64_audit_sha256=dynamic["validation64_audit"],
    )

    selection = formal.load_object(selection_path)
    arm = contract["arm_b_training"]
    _require(selection.get("schema_version"), arm["selection_schema_version"], "selection schema")
    _require(selection.get("status"), arm["selection_status"], "selection status")
    selection_contract = selection.get("contract") or {}
    for field in ("selected_records", "train_records", "validation_records"):
        _require(selection_contract.get(field), arm[field], f"selection {field}")
    _require(selection_contract.get("formal_training"), FORMAL_TRAINING_CONTRACT, "selection formal training")
    _require(selection_contract.get("primary_checkpoint"), "final-step32-only", "selection primary checkpoint")
    train_record = (selection.get("outputs") or {}).get("train") or {}
    _require(Path(str(train_record.get("path"))).resolve(), train_path.resolve(), "selection train path")
    _require(train_record.get("records"), 320, "selection train records")
    _require(train_record.get("sha256"), dynamic["train320"], "selection train SHA")
    rows = formal.load_jsonl(train_path)
    _require(len(rows), 320, "train320 row count")
    ids = [_task_id(row) for row in rows]
    indices = [row.get("example_index") for row in rows]
    if len(set(ids)) != 320 or len(set(indices)) != 320 or any(type(value) is not int for value in indices):
        raise ValueError("train320 identities are not exact and unique")
    _require((selection.get("task_ids") or {}).get("train"), ids, "train320 frozen order")

    source_record = selection.get("source_cohort_manifest") or {}
    source_path = Path(str(source_record.get("path") or ""))
    _regular(source_path, "Arm B source cohort manifest")
    source_sha = formal.sha256_file(source_path)
    _require(source_record.get("sha256"), source_sha, "selection source cohort SHA")
    source = formal.load_object(source_path)
    for field, expected in {
        "schema_version": admission.COHORT_SCHEMA,
        "status": admission.COHORT_STATUS,
    }.items():
        _require(source.get(field), expected, f"Arm B source cohort {field}")
    activation = admission._verify_activation(source)
    if activation["reason"] not in contract["confirmatory_trigger_contract"]["allowed_reasons"]:
        raise ValueError("source cohort Arm A failure reason is not preregistered")

    validation = formal.load_object(validation_path)
    _require(validation.get("schema_version"), arm["validation_schema_version"], "validation schema")
    _require(validation.get("status"), arm["validation_status"], "validation status")
    _require(validation.get("contract"), arm["validation_audit_contract"], "formal validation contract")
    validation_output = (selection.get("outputs") or {}).get("validation") or {}
    validation_tasks = Path(str(validation_output.get("path") or ""))
    _regular(validation_tasks, "Arm B validation64 tasks")
    _require(validation_output.get("records"), 64, "selection validation records")
    _require(
        validation_output.get("sha256"),
        formal.sha256_file(validation_tasks),
        "selection validation tasks SHA",
    )
    validation_inputs = validation.get("inputs") or {}
    _require(
        Path(str(validation_inputs.get("selection_manifest") or "")).resolve(),
        selection_path.resolve(),
        "validation selection path",
    )
    _require(
        validation_inputs.get("selection_manifest_sha256"),
        dynamic["selection_manifest"],
        "validation selection SHA",
    )
    _require(
        Path(str(validation_inputs.get("tasks") or "")).resolve(),
        validation_tasks.resolve(),
        "validation tasks path",
    )
    _require(
        validation_inputs.get("tasks_sha256"),
        formal.sha256_file(validation_tasks),
        "validation tasks SHA",
    )
    for label in ("generation_manifest", "trajectories"):
        artifact = Path(str(validation_inputs.get(label) or ""))
        _regular(artifact, f"validation {label}")
        _require(
            validation_inputs.get(f"{label}_sha256"),
            formal.sha256_file(artifact),
            f"validation {label} SHA",
        )
    validation_observed = validation.get("observed") or {}
    _require(validation_observed.get("tasks"), 64, "validation tasks")
    _require(validation_observed.get("group_size"), 16, "validation group size")
    _require(validation_observed.get("trajectories"), 1024, "validation trajectories")
    _require(validation_observed.get("usable_groups"), 64, "validation runtime-clean groups")
    _require(validation_observed.get("contaminated_groups"), 0, "validation contaminated groups")
    if (
        type(validation_observed.get("mixed_boundary_groups")) is not int
        or validation_observed["mixed_boundary_groups"] < 56
    ):
        raise ValueError("Arm B validation mixed-outcome gate failed")
    if (
        type(validation_observed.get("core_boundary_groups")) is not int
        or validation_observed["core_boundary_groups"] < 48
    ):
        raise ValueError("Arm B validation core-outcome gate failed")
    validation_checks = validation.get("checks") or {}
    if not validation_checks or not all(value is True for value in validation_checks.values()):
        raise ValueError("Arm B validation checks are not all true")
    _require(validation_checks.get("exact_1024_runtime_clean_trajectories"), True, "validation runtime contamination")
    _require(validation_checks.get("initial_sft1_policy"), True, "validation policy identity")

    run_manifest = formal.load_object(run_manifest_path)
    _require(run_manifest.get("examples_json_sha256"), dynamic["train320"], "run train320 SHA")
    lock = formal.load_object(lock_path)
    _require(lock.get("schema_version"), "trl-implementation-lock-v1", "implementation lock schema")
    _require(lock.get("files"), run_manifest.get("implementation_source_sha256"), "implementation lock files")
    training_contract = formal.load_object(training_contract_path)
    _require(training_contract.get("schema_version"), "qwen3-v26-arm-b-training-contract-v1", "training contract schema")
    _require(training_contract.get("status"), "completed_primary_final32", "training contract status")
    _require(training_contract.get("arm"), "arm_b", "training contract arm")
    contract_inputs = training_contract.get("inputs") or {}
    for label, path, digest in (
        ("selection_manifest", selection_path, dynamic["selection_manifest"]),
        ("train320", train_path, dynamic["train320"]),
        ("validation64_audit", validation_path, dynamic["validation64_audit"]),
        ("source_cohort_manifest", source_path, source_sha),
        (
            "confirmatory_trigger",
            Path(contract["arm_b_paths"]["confirmatory_trigger"]),
            dynamic["confirmatory_trigger"],
        ),
    ):
        record = contract_inputs.get(label) or {}
        _require(
            Path(str(record.get("path") or "")).resolve(),
            path.resolve(),
            f"training contract {label} path",
        )
        _require(record.get("sha256"), digest, f"training contract {label} SHA")
    _require(
        training_contract.get("training"),
        {
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
            "experiment_config_sha256": contract["training_final"]["experiment_config_sha256"],
        },
        "completed Arm B training contract",
    )
    _require(training_contract.get("checkpoint_policy"), arm["checkpoint_policy"], "training checkpoint policy")
    outputs = training_contract.get("outputs") or {}
    for field, digest in {
        "run_manifest_sha256": dynamic["training_run_manifest"],
        "implementation_lock_sha256": dynamic["training_implementation_lock"],
        "checkpoint32_adapter_sha256": dynamic["final_checkpoint32_adapter"],
        "final_adapter_sha256": dynamic["final_checkpoint32_adapter"],
    }.items():
        _require(outputs.get(field), digest, f"training contract output {field}")
    checkpoint_entries = sorted(training_run.glob("checkpoint-*"), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_dir() for path in checkpoint_entries):
        raise ValueError("Arm B checkpoint entries must be real directories")
    checkpoints = [path.name for path in checkpoint_entries]
    _require(checkpoints, ["checkpoint-32"], "sole Arm B checkpoint directory")
    trigger = verify_confirmatory_trigger(contract)
    _require(trigger["reason"], activation["reason"], "trigger/source Arm A failure reason")
    _require(
        trigger["arm_a_failure_evidence_sha256"],
        source_sha,
        "trigger/source Arm A failure evidence SHA",
    )
    _require(
        Path(trigger["arm_a_failure_evidence_path"]).resolve(),
        source_path.resolve(),
        "trigger/source Arm A failure evidence path",
    )
    return {
        "binding_status": "verified",
        "observed_sha256": observed,
        "primary_policy": "final-equals-checkpoint-32",
        "intermediate_checkpoints_evaluated": False,
        "training_input_admission": admission_report,
        "confirmatory_trigger": trigger,
    }


@contextmanager
def install_arm_b_training_verifier(contract: dict[str, Any]) -> Iterator[None]:
    original = formal.verify_training_final

    def wrapped(
        active_contract: dict[str, Any],
        training_run: Path,
        base_model: Path,
        sft1: dict[str, Any],
    ) -> dict[str, Any]:
        report = original(active_contract, training_run, base_model, sft1)
        binding = verify_arm_b_artifacts(active_contract, training_run)
        _require(report["adapter_sha256"], active_contract["arm_b_training"]["dynamic_sha256"]["final_checkpoint32_adapter"], "verified Arm B final adapter")
        return {**report, "arm_b_training_binding": binding}

    formal.verify_training_final = wrapped
    try:
        yield
    finally:
        formal.verify_training_final = original


def pending_plan(overlay: dict[str, Any], implementation: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": "qwen3-v26-arm-b-formal-matched-eval-plan-v1",
        "status": "dry_run_waiting_for_explicit_artifact_bindings",
        "remote_operations_performed": False,
        "arms": ["sft1", "final"],
        "execution_order": ["fresh_sft1", "final_checkpoint32"],
        "primary_candidate": "final",
        "primary_policy": "unique final equal to checkpoint-32; checkpoints 4..28 forbidden",
        "questions_per_arm": 1534,
        "same_gpu_runtime_concurrency": True,
        "runtime_and_decode": {
            "workers": 24,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 2048,
            "max_model_len": 16384,
            "enable_thinking": True,
        },
        "promotion_gate": overlay["promotion_gate"],
        "confirmatory_policy": overlay["confirmatory_trigger_contract"],
        "required_explicit_sha256": list(SHA_FIELDS),
        "required_explicit_paths": list(PATH_FIELDS),
        "no_resume": True,
        "implementation_sha256": implementation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-plan", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--base-contract", type=Path, default=DEFAULT_BASE)
    for field in SHA_FIELDS:
        parser.add_argument(f"--{field.replace('_', '-')}-sha256")
    for field in PATH_FIELDS:
        parser.add_argument(f"--{field.replace('_', '-')}-path")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--maximum-used-mib", type=int, default=512)
    parser.add_argument("--model-ready-timeout", type=int, default=900)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overlay = formal.load_object(args.overlay)
    base = formal.load_object(args.base_contract)
    validate_overlay(overlay, base)
    inert = _generic_contract(overlay, base, {field: "0" * 64 for field in SHA_FIELDS}, _inert_paths())
    implementation = {
        **formal.verify_implementation(inert),
        **verify_arm_b_implementation(overlay),
    }
    if args.print_plan:
        print(json.dumps(pending_plan(overlay, implementation), ensure_ascii=False, indent=2))
        return 0
    bindings = {field: getattr(args, f"{field}_sha256") for field in SHA_FIELDS}
    paths = {field: getattr(args, f"{field}_path") for field in PATH_FIELDS}
    contract = bind_contract(overlay, base, bindings, paths)
    if args.gpu_id is None or args.gpu_id < 0:
        raise SystemExit("--gpu-id is required and must be non-negative")
    with install_arm_b_training_verifier(contract):
        with formal.exclusive_gpu_lock(args.gpu_id):
            if args.preflight:
                report = formal.verify_host_assets(contract, gpu_id=args.gpu_id, maximum_used_mib=args.maximum_used_mib)
                report.pop("source_rows", None)
                print(json.dumps({"status": "arm_b_preflight_ok", **report}, ensure_ascii=False, indent=2))
                return 0
            if args.run_dir is None:
                raise SystemExit("--run requires --run-dir")

            def terminate(signum: int, _frame: Any) -> None:
                raise formal.TerminationRequested(f"received {signal.Signals(signum).name}")

            signal.signal(signal.SIGTERM, terminate)
            signal.signal(signal.SIGINT, terminate)
            return formal.execute_run(
                contract,
                run_dir=args.run_dir,
                gpu_id=args.gpu_id,
                port=args.port,
                maximum_used_mib=args.maximum_used_mib,
                model_ready_timeout=args.model_ready_timeout,
                implementation=implementation,
            )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Arm B formal matched evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
