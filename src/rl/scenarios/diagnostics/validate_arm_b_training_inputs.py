#!/usr/bin/env python3
"""Fail-closed admission check for preregistered Qwen3 v26 GRPO Arm B.

This is a read-only artifact validator.  It binds the frozen K16 selection,
train320 file, fresh validation64 audit, and the earlier Arm-A failure evidence
that activated Arm B.  It does not generate trajectories or start training.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
from rl.diagnostics.io import sha256_file as _diagnostic_sha256_file
from rl.diagnostics.validation import require as _diagnostic_require

from rl.scenarios.data.vanilla_grpo_arm_b import (
    ARM_B_SEED_REGISTRY,
    F1_GENERATION_SEED,
    F2_GENERATION_SEED,
    F2_SELECTION_SEED,
    F3_AUDIT_SCHEMA,
    F3_GENERATION_SEED,
    FORMAL_TRAINING_CONTRACT,
    INITIAL_ADAPTER_SHA256,
    PROTOCOL_HASH,
    PROTOCOL_VERSION,
    SELECTION_NAMESPACE,
    SELECTION_SCHEMA,
    SELECTION_STATUS,
    SELECTION_VALIDATION_CONTRACT,
    STUDENT_PROMPT_SHA256,
    TRAIN_DATA_SEED,
    VALIDATION_AUDIT_CONTRACT,
    WIDE_COHORT_NAMESPACE,
    WIDE_COHORT_SCHEMA,
    WIDE_COHORT_STATUS,
    WIDE_SELECTION_SEED,
    parse_json_bytes,
    parse_jsonl_bytes,
    read_bound_bytes,
)


COHORT_SCHEMA = WIDE_COHORT_SCHEMA
COHORT_STATUS = WIDE_COHORT_STATUS
VALIDATION_SCHEMA = F3_AUDIT_SCHEMA
VALIDATION_STATUS = {
    "passes": True,
    "validation_admitted": True,
    "next_stage": "train_arm_b",
}
K16_SELECTION_SEED = F2_SELECTION_SEED
SELECTION_COHORT_NAMESPACE = SELECTION_NAMESPACE
TRAIN_SEED = TRAIN_DATA_SEED
F1_SEED = F1_GENERATION_SEED
F2_SEED = F2_GENERATION_SEED
F3_SEED = F3_GENERATION_SEED
EXPECTED_PROTOCOL_VERSION = PROTOCOL_VERSION
EXPECTED_PROTOCOL_HASH = PROTOCOL_HASH
EXPECTED_PROMPT_SHA256 = STUDENT_PROMPT_SHA256
EXPECTED_ADAPTER_SHA256 = INITIAL_ADAPTER_SHA256
SHA_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    _diagnostic_require(condition, message)


def sha256_file(path: Path) -> str:
    return _diagnostic_sha256_file(path)


def _regular_file(path: Path, label: str) -> None:
    _require(path.is_file() and not path.is_symlink(), f"{label} must be a regular non-symlink file: {path}")


def _explicit_sha(value: str, label: str) -> None:
    _require(SHA_PATTERN.fullmatch(value) is not None, f"{label} must be an explicit lowercase SHA-256")


def _object(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    value = json.loads(path.read_bytes())
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def _bound_object(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    return parse_json_bytes(
        read_bound_bytes(path, expected_sha256, role=label), path=path
    )


def _jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _regular_file(path, label)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_bytes().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        _require(isinstance(row, dict), f"{label} row {line_number} is not an object")
        rows.append(row)
    return rows


def _task_id(row: Mapping[str, Any]) -> str:
    value = row.get("example_id") or row.get("instance_id")
    _require(isinstance(value, str) and bool(value), "task row lacks example_id/instance_id")
    return value


def _declared_artifact(
    record: Mapping[str, Any], path: Path, *, records: int, label: str
) -> None:
    _require(Path(str(record.get("path"))).resolve() == path.resolve(), f"{label} declared path mismatch")
    _require(record.get("records") == records, f"{label} declared record count mismatch")
    _require(record.get("sha256") == sha256_file(path), f"{label} declared SHA-256 mismatch")


def _verify_activation(cohort: Mapping[str, Any]) -> dict[str, Any]:
    _require(cohort.get("schema_version") == COHORT_SCHEMA, "Arm B wide cohort schema mismatch")
    _require(cohort.get("status") == COHORT_STATUS, "Arm B wide cohort status mismatch")
    _require(cohort.get("cohort_namespace") == WIDE_COHORT_NAMESPACE, "Arm B wide cohort namespace mismatch")
    _require(
        cohort.get("seeds") == ARM_B_SEED_REGISTRY,
        "Arm B wide cohort seed registry mismatch",
    )
    _require(
        (cohort.get("selection") or {}).get("seed") == WIDE_SELECTION_SEED,
        "Arm B wide3000 public-field selection seed mismatch",
    )
    _require(
        (cohort.get("downstream_contract") or {}).get("seed") == F1_SEED,
        "Arm B F1 K8 generation seed mismatch",
    )
    activation = cohort.get("activation") or {}
    _require(activation.get("decision") == "arm_b_pretraining_fallback", "Arm B activation decision mismatch")
    _require(
        activation.get("reason")
        in {"arm_a_s1_s2_readiness_failed", "arm_a_validation_failed"},
        "Arm B activation reason is not preregistered",
    )
    _require(activation.get("arm_a_training_started") is False, "Arm A training had already started")
    _require(activation.get("arm_a_full_dev_started") is False, "Arm A full-dev had already started")
    _require(activation.get("confirmatory_full_dev_arms_allowed") == 1, "confirmatory arm count is not one")
    evidence = activation.get("arm_a_failure_evidence") or {}
    _require(
        evidence.get("schema_version") == "arm-b-pretraining-fallback-evidence-v1"
        and evidence.get("storage") == "inline-in-this-wide-cohort-manifest",
        "embedded Arm A failure evidence identity mismatch",
    )
    _require(evidence.get("reason") == activation.get("reason"), "embedded Arm A failure reason mismatch")
    recomputed = evidence.get("recomputed") or {}
    _require(recomputed.get("reason") == activation.get("reason"), "recomputed Arm A failure reason mismatch")
    if evidence.get("reason") == "arm_a_s1_s2_readiness_failed":
        combined = recomputed.get("combined") or {}
        thresholds = recomputed.get("thresholds") or {}
        _require(thresholds == {"mixed": 360, "core": 240}, "Arm A readiness thresholds mismatch")
        _require(
            type(combined.get("mixed")) is int
            and type(combined.get("core")) is int
            and (combined["mixed"] < 360 or combined["core"] < 240),
            "Arm A readiness evidence does not prove failure",
        )
        _require(len(recomputed.get("pool_counts") or []) == 2, "Arm A readiness evidence lacks S1+S2")
    else:
        bound_inputs = recomputed.get("bound_inputs")
        _require(isinstance(bound_inputs, dict) and bool(bound_inputs), "Arm A validation failure has no bound inputs")
        for role, record in bound_inputs.items():
            _require(isinstance(record, dict), f"Arm A failure bound input {role} is malformed")
            path = Path(str(record.get("path") or ""))
            digest = str(record.get("sha256") or "")
            _explicit_sha(digest, f"Arm A failure bound input {role} SHA-256")
            _regular_file(path, f"Arm A failure bound input {role}")
            _require(sha256_file(path) == digest, f"Arm A failure bound input {role} SHA mismatch")
        _require(
            (recomputed.get("recomputed_status") or {}).get("validation_admitted") is False,
            "Arm A validation evidence does not prove rejection",
        )
    pools = activation.get("actual_screen_pools")
    _require(isinstance(pools, list) and len(pools) in {1, 2}, "Arm A actual screen pools are absent")
    for index, pool in enumerate(pools, 1):
        _require(isinstance(pool, dict), f"Arm A screen pool {index} is malformed")
        for role in ("audit", "tasks"):
            path = Path(str(pool.get(f"{role}_path") or ""))
            digest = str(pool.get(f"{role}_sha256") or "")
            _explicit_sha(digest, f"Arm A screen pool {index} {role} SHA-256")
            _regular_file(path, f"Arm A screen pool {index} {role}")
            _require(sha256_file(path) == digest, f"Arm A screen pool {index} {role} SHA mismatch")
    evidence_inputs = evidence.get("bound_inputs") or {}
    _require(evidence_inputs.get("screen_pools") == pools, "inline evidence screen-pool binding mismatch")
    _require(evidence_inputs.get("validation") == recomputed.get("bound_inputs"), "inline validation evidence binding mismatch")
    return {
        "reason": activation["reason"],
        "decision": activation["decision"],
        "confirmatory_full_dev_arms_allowed": 1,
    }


def validate(
    *,
    selection_manifest_path: Path,
    expected_selection_manifest_sha256: str,
    train320_path: Path,
    expected_train320_sha256: str,
    validation64_audit_path: Path,
    expected_validation64_audit_sha256: str,
) -> dict[str, Any]:
    for value, label in (
        (expected_selection_manifest_sha256, "selection manifest SHA-256"),
        (expected_train320_sha256, "train320 SHA-256"),
        (expected_validation64_audit_sha256, "validation64 audit SHA-256"),
    ):
        _explicit_sha(value, label)
    selection_bytes = read_bound_bytes(
        selection_manifest_path,
        expected_selection_manifest_sha256,
        role="Arm B selection manifest",
    )
    train_bytes = read_bound_bytes(
        train320_path, expected_train320_sha256, role="Arm B train320"
    )
    validation_audit_bytes = read_bound_bytes(
        validation64_audit_path,
        expected_validation64_audit_sha256,
        role="Arm B validation64 audit",
    )
    selection = parse_json_bytes(selection_bytes, path=selection_manifest_path)
    _require(selection.get("schema_version") == SELECTION_SCHEMA, "Arm B selection schema mismatch")
    _require(selection.get("status") == SELECTION_STATUS, "Arm B selection status mismatch")
    _require(selection.get("cohort_namespace") == SELECTION_COHORT_NAMESPACE, "Arm B selection namespace mismatch")
    _require(
        selection.get("seeds")
        == {
            "wide_k8_generation": F1_SEED,
            "k16_confirmation_generation": F2_SEED,
            "k16_selection": K16_SELECTION_SEED,
            "validation64_generation": F3_SEED,
            "formal_training_data": TRAIN_SEED,
        },
        "Arm B selection seed registry mismatch",
    )
    contract = selection.get("contract") or {}
    _require(contract.get("selected_records") == 384, "Arm B selected count mismatch")
    _require(contract.get("train_records") == 320, "Arm B train count mismatch")
    _require(contract.get("validation_records") == 64, "Arm B validation count mismatch")
    _require(contract.get("formal_training") == FORMAL_TRAINING_CONTRACT, "Arm B formal training contract mismatch")
    _require(contract.get("validation") == SELECTION_VALIDATION_CONTRACT, "Arm B selection validation contract mismatch")
    _require(contract.get("primary_checkpoint") == "final-step32-only", "Arm B primary checkpoint mismatch")
    selection_policy = selection.get("selection") or {}
    _require(selection_policy.get("seed") == K16_SELECTION_SEED, "Arm B selected384 tie-break seed mismatch")
    _require(selection_policy.get("selection_seed") == K16_SELECTION_SEED, "Arm B explicit selection seed mismatch")
    _require(selection_policy.get("confirmation_generation_seed") == F2_SEED, "Arm B explicit F2 generation seed mismatch")

    confirmation_record = selection.get("k16_confirmation_audit") or {}
    confirmation_path = Path(str(confirmation_record.get("path") or ""))
    _regular_file(confirmation_path, "Arm B K16 confirmation audit")
    _require(confirmation_record.get("sha256") == sha256_file(confirmation_path), "K16 confirmation audit SHA mismatch")
    confirmation = _object(confirmation_path, "Arm B K16 confirmation audit")
    _require((confirmation.get("contract") or {}).get("seed") == F2_SEED, "Arm B F2 K16 generation seed mismatch")
    generation_record = (selection.get("confirmation_generation") or {}).get("generation_manifest") or {}
    generation_path = Path(str(generation_record.get("path") or ""))
    _regular_file(generation_path, "Arm B K16 generation manifest")
    _require(generation_record.get("sha256") == sha256_file(generation_path), "K16 generation manifest SHA mismatch")
    _require(_object(generation_path, "Arm B K16 generation manifest").get("seed") == F2_SEED, "Arm B F2 rollout generation seed mismatch")

    outputs = selection.get("outputs") or {}
    task_ids = selection.get("task_ids") or {}
    selected_record = outputs.get("selected") or {}
    train_record = outputs.get("train") or {}
    validation_record = outputs.get("validation") or {}
    selected_path = Path(str(selected_record.get("path") or ""))
    validation_path = Path(str(validation_record.get("path") or ""))
    _regular_file(selected_path, "Arm B selected384")
    _regular_file(validation_path, "Arm B validation64")
    _declared_artifact(selected_record, selected_path, records=384, label="selected384")
    _declared_artifact(train_record, train320_path, records=320, label="train320")
    _declared_artifact(validation_record, validation_path, records=64, label="validation64")
    rows_by_name = {
        "selected": _jsonl(selected_path, "selected384"),
        "train": parse_jsonl_bytes(train_bytes, path=train320_path),
        "validation": _jsonl(validation_path, "validation64"),
    }
    ids = {name: [_task_id(row) for row in rows] for name, rows in rows_by_name.items()}
    for name, expected in (("selected", 384), ("train", 320), ("validation", 64)):
        _require(len(ids[name]) == len(set(ids[name])) == expected, f"Arm B {name} identities are not exact/unique")
        _require(task_ids.get(name) == ids[name], f"Arm B {name} frozen order mismatch")
    _require(set(ids["train"]).isdisjoint(ids["validation"]), "Arm B train/validation overlap")
    _require(set(ids["train"]) | set(ids["validation"]) == set(ids["selected"]), "Arm B selected split coverage mismatch")
    selected_by_id = {_task_id(row): row for row in rows_by_name["selected"]}
    _require(
        all(selected_by_id[_task_id(row)] == row for row in rows_by_name["train"] + rows_by_name["validation"]),
        "Arm B train/validation rows differ from selected384",
    )

    cohort_record = selection.get("source_cohort_manifest") or {}
    cohort_path = Path(str(cohort_record.get("path") or ""))
    _regular_file(cohort_path, "Arm B source cohort manifest")
    cohort_sha = str(cohort_record.get("sha256") or "")
    _require(cohort_record.get("schema_version") == COHORT_SCHEMA, "Arm B source cohort declared schema mismatch")
    _require(cohort_record.get("status") == COHORT_STATUS, "Arm B source cohort declared status mismatch")
    activation = _verify_activation(
        _bound_object(cohort_path, cohort_sha, "Arm B source cohort manifest")
    )

    audit = parse_json_bytes(validation_audit_bytes, path=validation64_audit_path)
    _require(audit.get("schema_version") == VALIDATION_SCHEMA, "Arm B validation schema mismatch")
    _require(audit.get("status") == VALIDATION_STATUS, "Arm B validation did not pass")
    validation_contract = audit.get("contract") or {}
    _require(
        validation_contract == VALIDATION_AUDIT_CONTRACT,
        "Arm B validation contract mismatch",
    )
    observed = audit.get("observed") or {}
    _require(observed.get("tasks") == 64, "Arm B validation task count mismatch")
    _require(observed.get("group_size") == 16, "Arm B validation group size mismatch")
    _require(observed.get("trajectories") == 1024, "Arm B validation trajectory count mismatch")
    _require(observed.get("usable_groups") == 64, "Arm B validation lacks 64 runtime-clean groups")
    _require(observed.get("contaminated_groups") == 0, "Arm B validation has runtime contamination")
    _require(type(observed.get("mixed_boundary_groups")) is int and observed["mixed_boundary_groups"] >= 56, "Arm B validation mixed gate failed")
    _require(type(observed.get("core_boundary_groups")) is int and observed["core_boundary_groups"] >= 48, "Arm B validation core gate failed")
    checks = audit.get("checks") or {}
    _require(checks and all(value is True for value in checks.values()), "Arm B validation checks are not all true")
    _require(checks.get("exact_1024_runtime_clean_trajectories") is True, "Arm B validation has runtime contamination")
    _require(checks.get("initial_sft1_policy") is True, "Arm B validation policy is not initial SFT1")
    inputs = audit.get("inputs") or {}
    _require(Path(str(inputs.get("selection_manifest") or "")).resolve() == selection_manifest_path.resolve(), "validation selection manifest path mismatch")
    _require(inputs.get("selection_manifest_sha256") == expected_selection_manifest_sha256, "validation selection manifest SHA mismatch")
    _require(Path(str(inputs.get("tasks") or "")).resolve() == validation_path.resolve(), "validation task path mismatch")
    _require(inputs.get("tasks_sha256") == sha256_file(validation_path), "validation task SHA mismatch")
    for label in ("generation_manifest", "trajectories"):
        path_value = Path(str(inputs.get(label) or ""))
        _regular_file(path_value, f"validation {label}")
        _require(inputs.get(f"{label}_sha256") == sha256_file(path_value), f"validation {label} SHA mismatch")
    validation_generation = _object(
        Path(str(inputs["generation_manifest"])), "validation generation manifest"
    )
    for field, expected in {
        "seed": F3_SEED,
        "adapter_sha256": EXPECTED_ADAPTER_SHA256,
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "student_prompt_sha256": EXPECTED_PROMPT_SHA256,
        "tasks": 64,
        "group_size": 16,
        "trajectories": 1024,
    }.items():
        _require(validation_generation.get(field) == expected, f"validation generation {field} mismatch")

    return {
        "schema_version": "vanilla-grpo-arm-b-training-input-admission-v1",
        "status": "training_inputs_admitted",
        "selection_manifest": {"path": str(selection_manifest_path.resolve()), "sha256": expected_selection_manifest_sha256},
        "train320": {"path": str(train320_path.resolve()), "sha256": expected_train320_sha256, "records": 320},
        "validation64_audit": {"path": str(validation64_audit_path.resolve()), "sha256": expected_validation64_audit_sha256},
        "activation": activation,
        "training_contract": FORMAL_TRAINING_CONTRACT,
        "primary_checkpoint": "final-step32-only",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--expected-selection-manifest-sha256", required=True)
    parser.add_argument("--train320", type=Path, required=True)
    parser.add_argument("--expected-train320-sha256", required=True)
    parser.add_argument("--validation64-audit", type=Path, required=True)
    parser.add_argument("--expected-validation64-audit-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate(
            selection_manifest_path=args.selection_manifest,
            expected_selection_manifest_sha256=args.expected_selection_manifest_sha256,
            train320_path=args.train320,
            expected_train320_sha256=args.expected_train320_sha256,
            validation64_audit_path=args.validation64_audit,
            expected_validation64_audit_sha256=args.expected_validation64_audit_sha256,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Arm B training input admission blocked: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
