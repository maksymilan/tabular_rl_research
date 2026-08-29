#!/usr/bin/env python3
"""Atomically admit exactly Arm B after its frozen F3 validation passes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.rl.diagnostics.validate_arm_b_training_inputs import validate
from src.rl.vanilla_grpo_arm_b import (
    F3_AUDIT_SCHEMA,
    SELECTION_SCHEMA,
    SELECTION_STATUS,
    WIDE_COHORT_SCHEMA,
    WIDE_COHORT_STATUS,
    parse_json_bytes,
    publish_exact,
    read_bound_bytes,
)


SCHEMA_VERSION = "vanilla-grpo-confirmatory-arm-trigger-v1"
STATUS = "arm_b_admitted_before_training"
DEFAULT_FORBIDDEN_ARM_A_PATHS = (
    Path(
        "/home/dengyan/tabular_rl_outputs/"
        "qwen3_8b_atomic_v26_boundary300_vanilla_grpo_20260812/"
        "train300_two_pass_seed20260812"
    ),
    Path(
        "/home/dengyan/tabular_rl_outputs/evaluations/"
        "qwen3_8b_atomic_v26_boundary300_vanilla_formal"
    ),
)


def prepare_trigger(
    *,
    source_cohort_path: Path,
    expected_source_cohort_sha256: str,
    selection_manifest_path: Path,
    expected_selection_manifest_sha256: str,
    train320_path: Path,
    expected_train320_sha256: str,
    validation64_audit_path: Path,
    expected_validation64_audit_sha256: str,
    output_path: Path,
    verify_existing: bool = False,
    forbidden_existing_arm_a_paths: Sequence[Path] = DEFAULT_FORBIDDEN_ARM_A_PATHS,
) -> dict[str, Any]:
    for path in forbidden_existing_arm_a_paths:
        if path.exists():
            raise ValueError(f"Arm A training/evaluation marker already exists: {path}")
    source_bytes = read_bound_bytes(
        source_cohort_path,
        expected_source_cohort_sha256,
        role="arm_b_source_wide_cohort",
    )
    selection_bytes = read_bound_bytes(
        selection_manifest_path,
        expected_selection_manifest_sha256,
        role="arm_b_selection_manifest",
    )
    validation_bytes = read_bound_bytes(
        validation64_audit_path,
        expected_validation64_audit_sha256,
        role="arm_b_validation64_audit",
    )
    source = parse_json_bytes(source_bytes, path=source_cohort_path)
    selection = parse_json_bytes(selection_bytes, path=selection_manifest_path)
    if (
        source.get("schema_version") != WIDE_COHORT_SCHEMA
        or source.get("status") != WIDE_COHORT_STATUS
    ):
        raise ValueError("source wide cohort schema/status mismatch")
    if (
        selection.get("schema_version") != SELECTION_SCHEMA
        or selection.get("status") != SELECTION_STATUS
    ):
        raise ValueError("selection schema/status mismatch")
    source_binding = selection.get("source_cohort_manifest") or {}
    expected_source_binding = {
        "path": str(source_cohort_path.resolve()),
        "sha256": expected_source_cohort_sha256,
        "schema_version": WIDE_COHORT_SCHEMA,
        "status": WIDE_COHORT_STATUS,
    }
    if source_binding != expected_source_binding:
        raise ValueError("selection does not bind the supplied source wide cohort")
    admission = validate(
        selection_manifest_path=selection_manifest_path,
        expected_selection_manifest_sha256=expected_selection_manifest_sha256,
        train320_path=train320_path,
        expected_train320_sha256=expected_train320_sha256,
        validation64_audit_path=validation64_audit_path,
        expected_validation64_audit_sha256=expected_validation64_audit_sha256,
    )
    activation = source.get("activation") or {}
    reason = activation.get("reason")
    if reason not in {
        "arm_a_s1_s2_readiness_failed",
        "arm_a_validation_failed",
    }:
        raise ValueError("source cohort lacks a preregistered Arm A failure reason")
    if admission.get("activation", {}).get("reason") != reason:
        raise ValueError("training admission and source cohort failure reasons differ")
    audit = parse_json_bytes(validation_bytes, path=validation64_audit_path)
    if audit.get("schema_version") != F3_AUDIT_SCHEMA:
        raise ValueError("F3 validation schema mismatch")
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "admitted_arm": "arm_b",
        "reason": reason,
        "arm_a_failure_evidence": expected_source_binding,
        "arm_a_training_started": False,
        "arm_a_full_dev_started": False,
        "arm_b_selection_manifest": {
            "path": str(selection_manifest_path.resolve()),
            "sha256": expected_selection_manifest_sha256,
            "schema_version": SELECTION_SCHEMA,
            "status": SELECTION_STATUS,
        },
        "arm_b_validation_audit": {
            "path": str(validation64_audit_path.resolve()),
            "sha256": expected_validation64_audit_sha256,
            "schema_version": F3_AUDIT_SCHEMA,
            "status": audit["status"],
        },
        "confirmatory_arm_count": 1,
        "full_dev_policy": "exactly-one-admitted-arm",
        "negative_fact_boundary": (
            "canonical Arm-A markers were absent at trigger publication; "
            "non-concurrent orchestration remains required"
        ),
    }
    payload = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode()
    if verify_existing:
        if (
            not output_path.is_file()
            or output_path.is_symlink()
            or output_path.read_bytes() != payload
        ):
            raise ValueError("existing confirmatory trigger differs from canonical bytes")
    else:
        publish_exact(((output_path, payload),))
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cohort", type=Path, required=True)
    parser.add_argument("--expected-source-cohort-sha256", required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--expected-selection-manifest-sha256", required=True)
    parser.add_argument("--train320", type=Path, required=True)
    parser.add_argument("--expected-train320-sha256", required=True)
    parser.add_argument("--validation64-audit", type=Path, required=True)
    parser.add_argument("--expected-validation64-audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)
    try:
        record = prepare_trigger(
            source_cohort_path=args.source_cohort,
            expected_source_cohort_sha256=args.expected_source_cohort_sha256,
            selection_manifest_path=args.selection_manifest,
            expected_selection_manifest_sha256=args.expected_selection_manifest_sha256,
            train320_path=args.train320,
            expected_train320_sha256=args.expected_train320_sha256,
            validation64_audit_path=args.validation64_audit,
            expected_validation64_audit_sha256=args.expected_validation64_audit_sha256,
            output_path=args.output,
            verify_existing=args.verify_existing,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Arm-B confirmatory trigger blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.resolve()), **record}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
