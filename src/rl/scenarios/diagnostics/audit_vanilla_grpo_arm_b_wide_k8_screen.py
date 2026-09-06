#!/usr/bin/env python3
"""Audit Arm B's wide3000 K8 screen and freeze 640 confirmation tasks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from rl.scenarios.data.vanilla_grpo_arm_b import (
    ARM_B_SEED_REGISTRY,
    F1_AUDIT_SCHEMA,
    F1_GENERATION_SEED,
    F1_SELECTION_SEED,
    INITIAL_ADAPTER_SHA256,
    PROTOCOL_HASH,
    PROTOCOL_VERSION,
    WIDE_COHORT_NAMESPACE,
    WIDE_COHORT_SCHEMA,
    WIDE_COHORT_STATUS,
    audit_fixed_policy_groups,
    balanced_uncertainty_select,
    jsonl_bytes,
    parse_json_bytes,
    parse_jsonl_bytes,
    publish_exact,
    read_bound_bytes,
    sha256_bytes,
    task_id,
)


SCHEMA_VERSION = F1_AUDIT_SCHEMA
COHORT_SCHEMA = WIDE_COHORT_SCHEMA
COHORT_STATUS = WIDE_COHORT_STATUS
EXPECTED_TASKS = 3000
GROUP_SIZE = 8
SCREEN_SEED = F1_GENERATION_SEED
MIN_CLEAN_MIXED = 640
CONFIRMATION_TASKS = 640
CONFIRMATION_SELECTION_SEED = F1_SELECTION_SEED
AUDIT_NAME = "arm_b_wide_k8_screen_audit.json"
CONFIRMATION_NAME = "confirmation640.jsonl"


def audit_wide_screen(
    *,
    cohort_manifest: dict[str, Any],
    generation_manifest: dict[str, Any],
    tasks: Sequence[dict[str, Any]],
    trajectories: Sequence[dict[str, Any]],
    expected_cohort_manifest_sha256: str,
    cohort_manifest_sha256: str,
    generation_manifest_sha256: str,
    tasks_sha256: str,
    trajectories_sha256: str,
    cohort_manifest_path: Path,
    generation_manifest_path: Path,
    tasks_path: Path,
    trajectories_path: Path,
    output_dir: Path,
    expected_tasks: int = EXPECTED_TASKS,
    minimum_clean_mixed: int = MIN_CLEAN_MIXED,
    confirmation_tasks: int = CONFIRMATION_TASKS,
) -> tuple[dict[str, Any], bytes | None]:
    if cohort_manifest_sha256 != expected_cohort_manifest_sha256:
        raise ValueError("explicit Arm-B wide cohort manifest digest mismatch")
    if (
        cohort_manifest.get("schema_version") != COHORT_SCHEMA
        or cohort_manifest.get("status") != COHORT_STATUS
        or cohort_manifest.get("cohort_namespace")
        != WIDE_COHORT_NAMESPACE
        or cohort_manifest.get("all_acceptance_gates_passed") is not True
        or not all((cohort_manifest.get("acceptance_gates") or {}).values())
    ):
        raise ValueError("wide cohort manifest is not frozen/admitted for screening")
    if cohort_manifest.get("seeds") != ARM_B_SEED_REGISTRY:
        raise ValueError("wide cohort seed registry differs from Arm-B preregistration")
    activation = cohort_manifest.get("activation") or {}
    if (
        activation.get("decision") != "arm_b_pretraining_fallback"
        or activation.get("arm_a_training_started") is not False
        or activation.get("arm_a_full_dev_started") is not False
        or activation.get("confirmatory_full_dev_arms_allowed") != 1
        or activation.get("reason")
        not in {"arm_a_s1_s2_readiness_failed", "arm_a_validation_failed"}
    ):
        raise ValueError("wide cohort lacks an exact pre-training Arm-B activation")
    output = cohort_manifest.get("output") or {}
    ordered_ids = [task_id(row) for row in tasks]
    if (
        output.get("path") != str(tasks_path.resolve())
        or output.get("sha256") != tasks_sha256
        or output.get("records") != expected_tasks
        or output.get("task_ids_in_frozen_order") != ordered_ids
    ):
        raise ValueError("wide3000 task bytes are not bound by the cohort manifest")
    downstream = cohort_manifest.get("downstream_contract") or {}
    if (
        downstream.get("screen_policy") != "fresh initial-SFT1"
        or downstream.get("group_size") != GROUP_SIZE
        or downstream.get("seed") != SCREEN_SEED
        or downstream.get("minimum_clean_mixed_groups") != minimum_clean_mixed
    ):
        raise ValueError("wide cohort downstream K8 contract mismatch")

    observed = audit_fixed_policy_groups(
        generation_manifest=generation_manifest,
        tasks=tasks,
        trajectories=trajectories,
        tasks_sha256=tasks_sha256,
        trajectories_sha256=trajectories_sha256,
        tasks_path=tasks_path,
        expected_tasks=expected_tasks,
        group_size=GROUP_SIZE,
        seed=SCREEN_SEED,
    )
    groups = observed.pop("groups")
    tasks_by_id = {task_id(row): row for row in tasks}
    candidates = [
        {
            "task_id": group["task_id"],
            "task": tasks_by_id[group["task_id"]],
            "correct_count": group["correct_count"],
            "uncertainty": group["uncertainty"],
            "core_boundary": group["core_boundary"],
        }
        for group in groups
        if group["usable"] and group["mixed_boundary"]
    ]
    ready = len(candidates) >= minimum_clean_mixed
    selected: list[dict[str, Any]] = []
    distribution: dict[str, Any] | None = None
    confirmation_bytes: bytes | None = None
    confirmation_path = output_dir / CONFIRMATION_NAME
    if ready:
        selected, distribution = balanced_uncertainty_select(
            candidates,
            count=confirmation_tasks,
            seed=CONFIRMATION_SELECTION_SEED,
            reference=tasks,
        )
        confirmation_bytes = jsonl_bytes(item["task"] for item in selected)

    acceptance_gates = {
        "structural_audit_passes": True,
        "at_least_640_clean_mixed_groups": len(candidates) >= minimum_clean_mixed,
        "confirmation_count_exact": (not ready) or len(selected) == confirmation_tasks,
        "confirmation_uses_only_clean_mixed_groups": (not ready)
        or all(
            item["correct_count"] in range(1, GROUP_SIZE)
            for item in selected
        ),
    }
    output_binding = (
        {
            "path": str(confirmation_path.resolve()),
            "sha256": sha256_bytes(confirmation_bytes),
            "records": confirmation_tasks,
        }
        if confirmation_bytes is not None
        else None
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "policy": "fresh initial-SFT1",
            "initial_adapter_sha256": INITIAL_ADAPTER_SHA256,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": PROTOCOL_HASH,
            "tasks": expected_tasks,
            "group_size": GROUP_SIZE,
            "trajectories": expected_tasks * GROUP_SIZE,
            "seed": SCREEN_SEED,
            "optimizer_updates": 0,
            "reward_mode": "result-only",
            "result_reward_profile": "binary",
            "minimum_clean_mixed_groups": minimum_clean_mixed,
            "confirmation_records": confirmation_tasks,
            "contaminated_group_policy": "retain audit evidence and exclude entire group",
            "replacement_sampling": False,
        },
        "inputs": {
            "cohort_manifest": str(cohort_manifest_path.resolve()),
            "cohort_manifest_sha256": cohort_manifest_sha256,
            "expected_cohort_manifest_sha256": expected_cohort_manifest_sha256,
            "generation_manifest": str(generation_manifest_path.resolve()),
            "generation_manifest_sha256": generation_manifest_sha256,
            "tasks": str(tasks_path.resolve()),
            "tasks_sha256": tasks_sha256,
            "trajectories": str(trajectories_path.resolve()),
            "trajectories_sha256": trajectories_sha256,
        },
        "observed": observed,
        "groups": groups,
        "selection": {
            "seed": CONFIRMATION_SELECTION_SEED,
            "rule": (
                "clean mixed K8 only; uncertainty c*(8-c) descending; public "
                "DB/question-length/external-knowledge deficit balance; SHA tie-break"
            ),
            "gold_or_dev_used": False,
            "distribution": distribution,
            "selected_metadata": [
                {
                    "task_id": item["task_id"],
                    "correct_count": item["correct_count"],
                    "uncertainty": item["uncertainty"],
                }
                for item in selected
            ],
        },
        "outputs": {"confirmation": output_binding},
        "task_ids": {"confirmation": [item["task_id"] for item in selected]},
        "acceptance_gates": acceptance_gates,
        "issues": [],
        "issue_counts": {},
        "status": {
            "audit_passes": True,
            "confirmation_ready": ready,
            "next_stage": "confirm_k16" if ready else "stop_arm_b",
        },
    }
    return audit, confirmation_bytes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--expected-cohort-manifest-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    audit_path = args.output_dir / AUDIT_NAME
    confirmation_path = args.output_dir / CONFIRMATION_NAME
    try:
        cohort_bytes = read_bound_bytes(
            args.cohort_manifest,
            args.expected_cohort_manifest_sha256,
            role="arm_b_wide_cohort_manifest",
        )
        cohort_sha = sha256_bytes(cohort_bytes)
        generation_bytes = args.manifest.read_bytes()
        tasks_bytes = args.tasks.read_bytes()
        trajectories_bytes = args.trajectories.read_bytes()
        audit, confirmation_bytes = audit_wide_screen(
            cohort_manifest=parse_json_bytes(cohort_bytes, path=args.cohort_manifest),
            generation_manifest=parse_json_bytes(generation_bytes, path=args.manifest),
            tasks=parse_jsonl_bytes(tasks_bytes, path=args.tasks),
            trajectories=parse_jsonl_bytes(trajectories_bytes, path=args.trajectories),
            expected_cohort_manifest_sha256=args.expected_cohort_manifest_sha256,
            cohort_manifest_sha256=cohort_sha,
            generation_manifest_sha256=sha256_bytes(generation_bytes),
            tasks_sha256=sha256_bytes(tasks_bytes),
            trajectories_sha256=sha256_bytes(trajectories_bytes),
            cohort_manifest_path=args.cohort_manifest,
            generation_manifest_path=args.manifest,
            tasks_path=args.tasks,
            trajectories_path=args.trajectories,
            output_dir=args.output_dir,
        )
        audit_bytes = (json.dumps(audit, ensure_ascii=False, indent=2) + "\n").encode()
        outputs = []
        if confirmation_bytes is not None:
            outputs.append((confirmation_path, confirmation_bytes))
        outputs.append((audit_path, audit_bytes))
        publish_exact(outputs)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Arm-B wide K8 screen audit blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(audit_path.resolve()), **audit["status"]}))
    return 0 if audit["status"]["confirmation_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
