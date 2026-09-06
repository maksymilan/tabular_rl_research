#!/usr/bin/env python3
"""Audit Arm-B fresh K16 confirmation and freeze 384 -> 320/64."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from rl.scenarios.data.vanilla_grpo_arm_b import (
    F1_AUDIT_SCHEMA,
    F2_AUDIT_SCHEMA,
    F2_GENERATION_SEED,
    F2_SELECTION_SEED,
    F3_GENERATION_SEED,
    FORMAL_TRAINING_CONTRACT,
    INITIAL_ADAPTER_SHA256,
    PROTOCOL_HASH,
    PROTOCOL_VERSION,
    SELECTION_NAMESPACE,
    SELECTION_SCHEMA as SHARED_SELECTION_SCHEMA,
    SELECTION_STATUS as SHARED_SELECTION_STATUS,
    SELECTION_VALIDATION_CONTRACT,
    TRAIN_DATA_SEED,
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
    stable_rank,
    task_id,
)


AUDIT_SCHEMA = F2_AUDIT_SCHEMA
SELECTION_SCHEMA = SHARED_SELECTION_SCHEMA
SELECTION_STATUS = SHARED_SELECTION_STATUS
WIDE_AUDIT_SCHEMA = F1_AUDIT_SCHEMA
COHORT_SCHEMA = WIDE_COHORT_SCHEMA
COHORT_STATUS = WIDE_COHORT_STATUS
EXPECTED_TASKS = 640
GROUP_SIZE = 16
CONFIRMATION_SEED = F2_GENERATION_SEED
MIN_ROBUST_CORE = 384
SELECTED_SIZE = 384
TRAIN_SIZE = 320
VALIDATION_SIZE = 64
SELECTION_SEED = F2_SELECTION_SEED
VALIDATION_SEED = F3_GENERATION_SEED
AUDIT_NAME = "arm_b_k16_confirmation_audit.json"
SELECTED_NAME = "selected384.jsonl"
TRAIN_NAME = "train320.jsonl"
VALIDATION_NAME = "validation64.jsonl"
MANIFEST_NAME = "arm_b_selection_manifest.json"


def _validate_wide_audit(
    *,
    audit: dict[str, Any],
    audit_path: Path,
    audit_sha256: str,
    expected_audit_sha256: str,
    tasks: Sequence[dict[str, Any]],
    tasks_path: Path,
    tasks_sha256: str,
) -> dict[str, Any]:
    if audit_sha256 != expected_audit_sha256:
        raise ValueError("explicit wide-screen audit digest mismatch")
    status = audit.get("status") or {}
    if (
        audit.get("schema_version") != WIDE_AUDIT_SCHEMA
        or status
        != {
            "audit_passes": True,
            "confirmation_ready": True,
            "next_stage": "confirm_k16",
        }
        or audit.get("issues") != []
        or audit.get("issue_counts") != {}
        or not all((audit.get("acceptance_gates") or {}).values())
    ):
        raise ValueError("wide K8 audit did not admit confirmation640")
    contract = audit.get("contract") or {}
    if (
        contract.get("group_size") != 8
        or contract.get("confirmation_records") != EXPECTED_TASKS
        or contract.get("minimum_clean_mixed_groups") != 640
        or contract.get("optimizer_updates") != 0
        or contract.get("reward_mode") != "result-only"
        or contract.get("result_reward_profile") != "binary"
    ):
        raise ValueError("wide-screen confirmation contract mismatch")
    output = (audit.get("outputs") or {}).get("confirmation")
    ordered_ids = [task_id(row) for row in tasks]
    if (
        not isinstance(output, dict)
        or output.get("path") != str(tasks_path.resolve())
        or output.get("sha256") != tasks_sha256
        or output.get("records") != EXPECTED_TASKS
        or (audit.get("task_ids") or {}).get("confirmation") != ordered_ids
    ):
        raise ValueError("confirmation640 bytes are not bound by wide K8 audit")
    inputs = audit.get("inputs") or {}
    cohort_path = Path(str(inputs.get("cohort_manifest") or ""))
    cohort_sha = inputs.get("cohort_manifest_sha256")
    cohort_bytes = read_bound_bytes(
        cohort_path, cohort_sha, role="source_wide_cohort_manifest"
    )
    cohort = parse_json_bytes(cohort_bytes, path=cohort_path)
    if (
        cohort.get("schema_version") != COHORT_SCHEMA
        or cohort.get("status") != COHORT_STATUS
        or (cohort.get("activation") or {}).get("arm_a_training_started") is not False
        or (cohort.get("activation") or {}).get("arm_a_full_dev_started") is not False
    ):
        raise ValueError("source wide cohort activation is invalid")
    return {
        "path": str(cohort_path.resolve()),
        "sha256": cohort_sha,
        "schema_version": COHORT_SCHEMA,
        "status": COHORT_STATUS,
    }


def confirm_and_select(
    *,
    wide_audit: dict[str, Any],
    generation_manifest: dict[str, Any],
    tasks: Sequence[dict[str, Any]],
    trajectories: Sequence[dict[str, Any]],
    expected_wide_audit_sha256: str,
    wide_audit_sha256: str,
    generation_manifest_sha256: str,
    tasks_sha256: str,
    trajectories_sha256: str,
    wide_audit_path: Path,
    generation_manifest_path: Path,
    tasks_path: Path,
    trajectories_path: Path,
    output_dir: Path,
    minimum_robust_core: int = MIN_ROBUST_CORE,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    source_cohort = _validate_wide_audit(
        audit=wide_audit,
        audit_path=wide_audit_path,
        audit_sha256=wide_audit_sha256,
        expected_audit_sha256=expected_wide_audit_sha256,
        tasks=tasks,
        tasks_path=tasks_path,
        tasks_sha256=tasks_sha256,
    )
    observed = audit_fixed_policy_groups(
        generation_manifest=generation_manifest,
        tasks=tasks,
        trajectories=trajectories,
        tasks_sha256=tasks_sha256,
        trajectories_sha256=trajectories_sha256,
        tasks_path=tasks_path,
        expected_tasks=EXPECTED_TASKS,
        group_size=GROUP_SIZE,
        seed=CONFIRMATION_SEED,
    )
    groups = observed.pop("groups")
    tasks_by_id = {task_id(row): row for row in tasks}
    candidates = [
        {
            "task_id": group["task_id"],
            "task": tasks_by_id[group["task_id"]],
            "correct_count": group["correct_count"],
            "uncertainty": group["uncertainty"],
        }
        for group in groups
        if group["usable"] and group["core_boundary"]
    ]
    ready = len(candidates) >= minimum_robust_core
    selected: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    distribution: dict[str, Any] | None = None
    rendered: dict[str, bytes] = {}
    if ready:
        selected, distribution = balanced_uncertainty_select(
            candidates,
            count=SELECTED_SIZE,
            seed=SELECTION_SEED,
            reference=tasks,
        )
        validation_ids = {
            item["task_id"]
            for item in sorted(
                selected,
                key=lambda item: stable_rank(
                    SELECTION_SEED, "validation-split", str(item["task_id"])
                ),
            )[:VALIDATION_SIZE]
        }
        validation = [item for item in selected if item["task_id"] in validation_ids]
        train = [item for item in selected if item["task_id"] not in validation_ids]
        if len(train) != TRAIN_SIZE or len(validation) != VALIDATION_SIZE:
            raise RuntimeError("deterministic Arm-B train/validation split is malformed")
        rendered = {
            "selected": jsonl_bytes(item["task"] for item in selected),
            "train": jsonl_bytes(item["task"] for item in train),
            "validation": jsonl_bytes(item["task"] for item in validation),
        }

    paths = {
        "selected": output_dir / SELECTED_NAME,
        "train": output_dir / TRAIN_NAME,
        "validation": output_dir / VALIDATION_NAME,
    }
    output_bindings = {
        role: {
            "path": str(paths[role].resolve()),
            "sha256": sha256_bytes(payload),
            "records": {"selected": SELECTED_SIZE, "train": TRAIN_SIZE, "validation": VALIDATION_SIZE}[role],
        }
        for role, payload in rendered.items()
    }
    acceptance_gates = {
        "structural_audit_passes": True,
        "at_least_384_clean_robust_core_groups": len(candidates) >= minimum_robust_core,
        "selected384_exact": (not ready) or len(selected) == SELECTED_SIZE,
        "train320_exact": (not ready) or len(train) == TRAIN_SIZE,
        "validation64_exact": (not ready) or len(validation) == VALIDATION_SIZE,
        "selected_only_clean_c2_to_c14": (not ready)
        or all(2 <= item["correct_count"] <= 14 for item in selected),
    }
    audit = {
        "schema_version": AUDIT_SCHEMA,
        "contract": {
            "policy": "fresh initial-SFT1",
            "initial_adapter_sha256": INITIAL_ADAPTER_SHA256,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": PROTOCOL_HASH,
            "tasks": EXPECTED_TASKS,
            "group_size": GROUP_SIZE,
            "trajectories": EXPECTED_TASKS * GROUP_SIZE,
            "seed": CONFIRMATION_SEED,
            "optimizer_updates": 0,
            "reward_mode": "result-only",
            "result_reward_profile": "binary",
            "robust_core": "2<=correct_count<=14 and 16/16 runtime-clean",
            "minimum_robust_core_groups": minimum_robust_core,
            "replacement_sampling": False,
        },
        "inputs": {
            "source_cohort_manifest": source_cohort,
            "wide_screen_audit": {
                "path": str(wide_audit_path.resolve()),
                "sha256": wide_audit_sha256,
            },
            "generation_manifest": {
                "path": str(generation_manifest_path.resolve()),
                "sha256": generation_manifest_sha256,
            },
            "tasks": {"path": str(tasks_path.resolve()), "sha256": tasks_sha256},
            "trajectories": {
                "path": str(trajectories_path.resolve()),
                "sha256": trajectories_sha256,
            },
        },
        "observed": observed,
        "groups": groups,
        "selection": {
            "seed": SELECTION_SEED,
            "rule": (
                "clean robust-core K16 only; uncertainty c*(16-c) descending; public "
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
        "outputs": output_bindings,
        "task_ids": {
            "selected": [item["task_id"] for item in selected],
            "train": [item["task_id"] for item in train],
            "validation": [item["task_id"] for item in validation],
        },
        "acceptance_gates": acceptance_gates,
        "issues": [],
        "issue_counts": {},
        "status": {
            "audit_passes": True,
            "selection_ready": ready,
            "next_stage": "validate_k16" if ready else "stop_arm_b",
        },
    }
    return audit, rendered


def _selection_manifest(
    *,
    audit: dict[str, Any],
    audit_path: Path,
    audit_sha256: str,
    rendered: dict[str, bytes],
    output_dir: Path,
) -> dict[str, Any]:
    paths = {
        "selected": output_dir / SELECTED_NAME,
        "train": output_dir / TRAIN_NAME,
        "validation": output_dir / VALIDATION_NAME,
    }
    task_ids = audit["task_ids"]
    outputs = {
        role: {
            "path": str(paths[role].resolve()),
            "sha256": sha256_bytes(rendered[role]),
            "records": {"selected": SELECTED_SIZE, "train": TRAIN_SIZE, "validation": VALIDATION_SIZE}[role],
        }
        for role in ("selected", "train", "validation")
    }
    return {
        "schema_version": SELECTION_SCHEMA,
        "status": SELECTION_STATUS,
        "cohort_namespace": SELECTION_NAMESPACE,
        "seeds": {
            "wide_k8_generation": 20260814,
            "k16_confirmation_generation": CONFIRMATION_SEED,
            "k16_selection": SELECTION_SEED,
            "validation64_generation": VALIDATION_SEED,
            "formal_training_data": TRAIN_DATA_SEED,
        },
        "purpose": "independent Arm-B vanilla binary-GRPO train320 cohort",
        "source_cohort_manifest": audit["inputs"]["source_cohort_manifest"],
        "wide_screen_audit": audit["inputs"]["wide_screen_audit"],
        "k16_confirmation_audit": {
            "path": str(audit_path.resolve()),
            "sha256": audit_sha256,
            "schema_version": AUDIT_SCHEMA,
            "status": audit["status"],
        },
        "contract": {
            "selected_records": SELECTED_SIZE,
            "train_records": TRAIN_SIZE,
            "validation_records": VALIDATION_SIZE,
            "formal_training": dict(FORMAL_TRAINING_CONTRACT),
            "validation": dict(SELECTION_VALIDATION_CONTRACT),
            "primary_checkpoint": "final-step32-only",
        },
        "confirmation_generation": audit["inputs"],
        "selection": {
            **audit["selection"],
            "confirmation_generation_seed": CONFIRMATION_SEED,
            "selection_seed": SELECTION_SEED,
            "acceptance_gates": audit["acceptance_gates"],
            "split_rule": "fixed SHA-256 validation namespace over selected384",
        },
        "outputs": outputs,
        "task_ids": task_ids,
        "gold_visibility": "no gold/dev field used for screen selection, split, or ordering",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-screen-audit", type=Path, required=True)
    parser.add_argument("--expected-wide-screen-audit-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    audit_path = args.output_dir / AUDIT_NAME
    manifest_path = args.output_dir / MANIFEST_NAME
    try:
        wide_bytes = read_bound_bytes(
            args.wide_screen_audit,
            args.expected_wide_screen_audit_sha256,
            role="arm_b_wide_k8_screen_audit",
        )
        wide_sha = sha256_bytes(wide_bytes)
        generation_bytes = args.manifest.read_bytes()
        tasks_bytes = args.tasks.read_bytes()
        trajectories_bytes = args.trajectories.read_bytes()
        audit, rendered = confirm_and_select(
            wide_audit=parse_json_bytes(wide_bytes, path=args.wide_screen_audit),
            generation_manifest=parse_json_bytes(generation_bytes, path=args.manifest),
            tasks=parse_jsonl_bytes(tasks_bytes, path=args.tasks),
            trajectories=parse_jsonl_bytes(trajectories_bytes, path=args.trajectories),
            expected_wide_audit_sha256=args.expected_wide_screen_audit_sha256,
            wide_audit_sha256=wide_sha,
            generation_manifest_sha256=sha256_bytes(generation_bytes),
            tasks_sha256=sha256_bytes(tasks_bytes),
            trajectories_sha256=sha256_bytes(trajectories_bytes),
            wide_audit_path=args.wide_screen_audit,
            generation_manifest_path=args.manifest,
            tasks_path=args.tasks,
            trajectories_path=args.trajectories,
            output_dir=args.output_dir,
        )
        audit_bytes = (json.dumps(audit, ensure_ascii=False, indent=2) + "\n").encode()
        outputs: list[tuple[Path, bytes]] = []
        if audit["status"]["selection_ready"]:
            for role, filename in (
                ("selected", SELECTED_NAME),
                ("train", TRAIN_NAME),
                ("validation", VALIDATION_NAME),
            ):
                outputs.append((args.output_dir / filename, rendered[role]))
        outputs.append((audit_path, audit_bytes))
        if audit["status"]["selection_ready"]:
            selection = _selection_manifest(
                audit=audit,
                audit_path=audit_path,
                audit_sha256=sha256_bytes(audit_bytes),
                rendered=rendered,
                output_dir=args.output_dir,
            )
            selection_bytes = (
                json.dumps(selection, ensure_ascii=False, indent=2) + "\n"
            ).encode()
            outputs.append((manifest_path, selection_bytes))
        publish_exact(outputs)
    except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Arm-B K16 confirmation blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(audit_path.resolve()), **audit["status"]}))
    return 0 if audit["status"]["selection_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
