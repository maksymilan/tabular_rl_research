#!/usr/bin/env python3
"""Audit Arm B's fresh held-out validation64 K16 admission gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from rl.scenarios.data.vanilla_grpo_arm_b import (
    F2_GENERATION_SEED,
    F3_AUDIT_SCHEMA,
    F3_GENERATION_SEED,
    FORMAL_TRAINING_CONTRACT,
    GENERATION_SEED_SCHEME,
    INITIAL_ADAPTER_SHA256,
    PROTOCOL_HASH,
    PROTOCOL_VERSION,
    SELECTION_SCHEMA as SHARED_SELECTION_SCHEMA,
    SELECTION_STATUS as SHARED_SELECTION_STATUS,
    SELECTION_VALIDATION_CONTRACT,
    STUDENT_PROMPT_SHA256,
    VALIDATION_AUDIT_CONTRACT,
    audit_fixed_policy_groups,
    parse_json_bytes,
    parse_jsonl_bytes,
    publish_exact,
    read_bound_bytes,
    sha256_bytes,
    task_id,
)


SCHEMA_VERSION = F3_AUDIT_SCHEMA
SELECTION_SCHEMA = SHARED_SELECTION_SCHEMA
SELECTION_STATUS = SHARED_SELECTION_STATUS
EXPECTED_TASKS = 64
GROUP_SIZE = 16
EXPECTED_TRAJECTORIES = EXPECTED_TASKS * GROUP_SIZE
VALIDATION_SEED = F3_GENERATION_SEED
CONFIRMATION_SEED = F2_GENERATION_SEED
MIN_MIXED = 56
MIN_CORE = 48


def audit_validation(
    *,
    selection_manifest: dict[str, Any],
    generation_manifest: dict[str, Any],
    tasks: Sequence[dict[str, Any]],
    trajectories: Sequence[dict[str, Any]],
    expected_selection_manifest_sha256: str,
    selection_manifest_sha256: str,
    generation_manifest_sha256: str,
    tasks_sha256: str,
    trajectories_sha256: str,
    selection_manifest_path: Path,
    generation_manifest_path: Path,
    tasks_path: Path,
    trajectories_path: Path,
    expected_tasks: int = EXPECTED_TASKS,
    minimum_mixed: int = MIN_MIXED,
    minimum_core: int = MIN_CORE,
) -> dict[str, Any]:
    if selection_manifest_sha256 != expected_selection_manifest_sha256:
        raise ValueError("explicit Arm-B selection manifest digest mismatch")
    if (
        selection_manifest.get("schema_version") != SELECTION_SCHEMA
        or selection_manifest.get("status") != SELECTION_STATUS
    ):
        raise ValueError("unsupported Arm-B selection schema/status")
    source = selection_manifest.get("source_cohort_manifest") or {}
    source_path = Path(str(source.get("path") or ""))
    source_bytes = read_bound_bytes(
        source_path, source.get("sha256"), role="source_wide_cohort"
    )
    source_manifest = parse_json_bytes(source_bytes, path=source_path)
    activation = source_manifest.get("activation") or {}
    if (
        source_manifest.get("schema_version") != source.get("schema_version")
        or source_manifest.get("status") != source.get("status")
        or activation.get("decision") != "arm_b_pretraining_fallback"
        or activation.get("arm_a_training_started") is not False
        or activation.get("arm_a_full_dev_started") is not False
        or activation.get("confirmatory_full_dev_arms_allowed") != 1
    ):
        raise ValueError("source cohort no longer proves Arm-B-only pre-training activation")

    contract = selection_manifest.get("contract") or {}
    if contract != {
        "selected_records": 384,
        "train_records": 320,
        "validation_records": 64,
        "formal_training": FORMAL_TRAINING_CONTRACT,
        "validation": SELECTION_VALIDATION_CONTRACT,
        "primary_checkpoint": "final-step32-only",
    }:
        raise ValueError("selection manifest training/validation contract differs")
    outputs = selection_manifest.get("outputs") or {}
    task_ids = selection_manifest.get("task_ids") or {}
    validation_output = outputs.get("validation") or {}
    ordered_ids = [task_id(row) for row in tasks]
    if (
        validation_output.get("path") != str(tasks_path.resolve())
        or validation_output.get("sha256") != tasks_sha256
        or validation_output.get("records") != expected_tasks
        or task_ids.get("validation") != ordered_ids
    ):
        raise ValueError("validation64 bytes are not bound by the selection manifest")

    observed = audit_fixed_policy_groups(
        generation_manifest=generation_manifest,
        tasks=tasks,
        trajectories=trajectories,
        tasks_sha256=tasks_sha256,
        trajectories_sha256=trajectories_sha256,
        tasks_path=tasks_path,
        expected_tasks=expected_tasks,
        group_size=GROUP_SIZE,
        seed=VALIDATION_SEED,
    )
    groups = observed.pop("groups")
    all_clean = observed["usable_groups"] == expected_tasks
    mixed = observed["mixed_boundary_groups"]
    core = observed["core_boundary_groups"]
    observed.update(
        {
            "groups": expected_tasks,
            "eligible_trajectories": observed["usable_groups"] * GROUP_SIZE,
            "mixed_outcome_groups": mixed,
            "core_outcome_groups": core,
            "tokenization_warnings": sum(
                "tokenization_warning" in group["contamination"] for group in groups
            ),
        }
    )
    checks = {
        "explicit_selection_manifest_digest": True,
        "frozen_selection_contract": True,
        "validation_tasks_byte_bound": True,
        "independent_validation_seed": generation_manifest.get("seed")
        == VALIDATION_SEED
        and generation_manifest.get("seed") != CONFIRMATION_SEED,
        "initial_sft1_policy": generation_manifest.get("adapter_sha256")
        == INITIAL_ADAPTER_SHA256,
        "generation_manifest_contract": True,
        "exact_trajectory_task_binding": True,
        "exact_1024_runtime_clean_trajectories": all_clean
        and observed["trajectories"] == expected_tasks * GROUP_SIZE,
        "no_runtime_contamination": all_clean
        and observed["eligible_trajectories"] == expected_tasks * GROUP_SIZE
        and observed["tokenization_warnings"] == 0,
        "at_least_56_mixed_outcome_groups": mixed >= minimum_mixed,
        "at_least_48_core_outcome_groups": core >= minimum_core,
        "no_gold_or_dev_selection": (selection_manifest.get("selection") or {}).get(
            "gold_or_dev_used"
        )
        is False,
    }
    passes = all(checks.values())
    failed = [name for name, value in checks.items() if not value]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": dict(VALIDATION_AUDIT_CONTRACT),
        "inputs": {
            "source_cohort_manifest": source,
            "selection_manifest": str(selection_manifest_path.resolve()),
            "selection_manifest_sha256": selection_manifest_sha256,
            "expected_selection_manifest_sha256": expected_selection_manifest_sha256,
            "generation_manifest": str(generation_manifest_path.resolve()),
            "generation_manifest_sha256": generation_manifest_sha256,
            "tasks": str(tasks_path.resolve()),
            "tasks_sha256": tasks_sha256,
            "trajectories": str(trajectories_path.resolve()),
            "trajectories_sha256": trajectories_sha256,
        },
        "observed": observed,
        "groups": groups,
        "checks": checks,
        "issues": [
            {"code": "validation_gate_failed", "location": name, "detail": "false"}
            for name in failed
        ],
        "issue_counts": ({"validation_gate_failed": len(failed)} if failed else {}),
        "status": {
            "passes": passes,
            "validation_admitted": passes,
            "next_stage": "train_arm_b" if passes else "stop_arm_b",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--expected-selection-manifest-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        selection_bytes = read_bound_bytes(
            args.selection_manifest,
            args.expected_selection_manifest_sha256,
            role="arm_b_selection_manifest",
        )
        selection_sha = sha256_bytes(selection_bytes)
        generation_bytes = args.manifest.read_bytes()
        task_bytes = args.tasks.read_bytes()
        trajectory_bytes = args.trajectories.read_bytes()
        result = audit_validation(
            selection_manifest=parse_json_bytes(
                selection_bytes, path=args.selection_manifest
            ),
            generation_manifest=parse_json_bytes(generation_bytes, path=args.manifest),
            tasks=parse_jsonl_bytes(task_bytes, path=args.tasks),
            trajectories=parse_jsonl_bytes(trajectory_bytes, path=args.trajectories),
            expected_selection_manifest_sha256=args.expected_selection_manifest_sha256,
            selection_manifest_sha256=selection_sha,
            generation_manifest_sha256=sha256_bytes(generation_bytes),
            tasks_sha256=sha256_bytes(task_bytes),
            trajectories_sha256=sha256_bytes(trajectory_bytes),
            selection_manifest_path=args.selection_manifest,
            generation_manifest_path=args.manifest,
            tasks_path=args.tasks,
            trajectories_path=args.trajectories,
        )
        result_bytes = (
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        ).encode()
        publish_exact(((args.output, result_bytes),))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Arm-B K16 validation blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.resolve()), **result["status"]}))
    return 0 if result["status"]["validation_admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
