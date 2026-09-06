#!/usr/bin/env python3
"""Freeze Arm B's trigger-bound, public-field-selected wide3000 screen.

Arm B is not a continuation of Arm A.  This preparer re-audits immutable Arm-A
artifacts and opens the fallback only after a *pre-training* readiness or
validation failure.  It never reads dev outcomes and never uses gold SQL or
gold values for selection or ordering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from eval.select_bird_train_baseline import (
    FORBIDDEN_SELECTION_FIELDS,
    PUBLIC_SELECTION_FIELDS,
    distribution_audit,
    length_cutpoints,
    select_rows,
)
from rl.scenarios.diagnostics.audit_vanilla_grpo_boundary_validation import (
    audit_boundary_validation,
)
from rl.scenarios.data.select_policy_boundary_grpo_tasks import _load_pool
from rl.scenarios.data.vanilla_grpo_arm_b import (
    ARM_B_SEED_REGISTRY,
    WIDE_COHORT_NAMESPACE,
    WIDE_COHORT_SCHEMA,
    WIDE_COHORT_STATUS,
    WIDE_SELECTION_SEED,
    jsonl_bytes,
    parse_json_bytes,
    parse_jsonl_bytes,
    publish_exact,
    read_bound_bytes,
    require_file_digest,
    sha256_bytes,
    sha256_file,
    task_id,
)


SCHEMA_VERSION = WIDE_COHORT_SCHEMA
STATUS = WIDE_COHORT_STATUS
SELECTION_SEED = WIDE_SELECTION_SEED
EXPECTED_COUNT = 3000
MIN_ARM_A_MIXED = 360
MIN_ARM_A_CORE = 240
CANONICAL_HASHES = {
    "reference": "24d21a349c430df549f92724dd07de1c59b2365cd72802957120afba13c7a2e1",
    "eligible": "c8ea3c6419ac57f05021eca470c7519843456e8ac02b04cca9af39e779ec7545",
    "baseline_eval300": "87b37b307ea2710acdff7d03bdc474a6ba2cdb8ed51b005cff0fe8fa54c67c03",
    "sft1_index": "aa6b8a72a1cfbcf44e37702e0498c9a85f615edc201ba3f4172b65f701e0cda5",
    "old_mixed60": "ef97b6977735996fde505aaad4e00215570963b37a1c82d890c34c4b414f30ae",
}
CANONICAL_COUNTS = {
    "reference": (6601, 6601),
    "eligible": (5915, 5915),
    "baseline_eval300": (300, 300),
    "sft1_index": (4471, 678),
    "old_mixed60": (60, 60),
}
EXCLUSION_ID_FIELDS = ("example_id", "instance_id", "source_episode_id")


def _index(rows: Sequence[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    examples: set[int] = set()
    for position, row in enumerate(rows, start=1):
        identifier = task_id(row)
        example_index = row.get("example_index")
        if identifier in indexed:
            raise ValueError(f"{source}:{position}: duplicate task id {identifier}")
        if source != "sft1_index" and (
            type(example_index) is not int or example_index in examples
        ):
            raise ValueError(f"{source}:{position}: invalid/duplicate example_index")
        indexed[identifier] = row
        if type(example_index) is int:
            examples.add(example_index)
    return indexed


def _exclusion_ids(
    path: Path, rows: Sequence[dict[str, Any]], *, role: str, digest: str
) -> tuple[set[str], dict[str, Any]]:
    identifiers: set[str] = set()
    example_indices: set[int] = set()
    for position, row in enumerate(rows, start=1):
        value = next(
            (
                row.get(field)
                for field in EXCLUSION_ID_FIELDS
                if isinstance(row.get(field), str) and row.get(field)
            ),
            None,
        )
        if not isinstance(value, str):
            raise ValueError(f"{role}:{position}: exclusion task identity is absent")
        identifiers.add(value)
        example_indices.add(_example_index(row, value))
    return identifiers, {
        "role": role,
        "path": str(path.resolve()),
        "sha256": digest,
        "records": len(rows),
        "unique_task_ids": len(identifiers),
        "task_ids": sorted(identifiers),
        "example_indices": sorted(example_indices),
    }


def _example_index(row: Mapping[str, Any], identifier: str) -> int:
    value = row.get("example_index")
    if type(value) is int and value >= 0:
        return value
    match = re.fullmatch(r"bird_train_([0-9]{5})", identifier)
    if match is None:
        raise ValueError(f"cannot derive BIRD example_index from {identifier!r}")
    return int(match.group(1))


def _load_screen_pools(
    *,
    audit_paths: Sequence[Path],
    expected_audit_hashes: Sequence[str],
    task_paths: Sequence[Path],
    expected_task_hashes: Sequence[str],
) -> list[dict[str, Any]]:
    if not (
        len(audit_paths)
        == len(expected_audit_hashes)
        == len(task_paths)
        == len(expected_task_hashes)
    ) or len(task_paths) not in {1, 2}:
        raise ValueError("pass one exact audit/hash/tasks/hash tuple for each actual Arm-A pool")
    pools: list[dict[str, Any]] = []
    all_ids: set[str] = set()
    all_examples: set[int] = set()
    for index, (audit, audit_sha, tasks, tasks_sha) in enumerate(
        zip(
            audit_paths,
            expected_audit_hashes,
            task_paths,
            expected_task_hashes,
            strict=True,
        ),
        start=1,
    ):
        if not audit.is_file() or audit.is_symlink() or not tasks.is_file() or tasks.is_symlink():
            raise ValueError(f"Arm-A pool {index} inputs must be regular non-symlink files")
        pool = _load_pool(audit, tasks)
        if pool["audit_sha256"] != audit_sha or pool["tasks_sha256"] != tasks_sha:
            raise ValueError(f"Arm-A pool {index} digest binding changed during validation")
        identifiers = set(pool["tasks"])
        example_indices = {
            int(row["example_index"]) for row in pool["tasks"].values()
        }
        overlap = all_ids & identifiers
        example_overlap = all_examples & example_indices
        if overlap or example_overlap:
            raise ValueError(
                "Arm-A screen pools overlap by task id or example_index: "
                f"ids={sorted(overlap)[:5]} examples={sorted(example_overlap)[:5]}"
            )
        all_ids.update(identifiers)
        all_examples.update(example_indices)
        pools.append(pool)
    return pools


def _validate_readiness_failure(pools: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(pools) != 2:
        raise ValueError("readiness fallback requires completed, disjoint S1 and S2 pools")
    counts = []
    for pool in pools:
        groups = list(pool["groups"].values())
        counts.append(
            {
                "mixed": sum(int(group["mixed_boundary"]) for group in groups),
                "core": sum(int(group["core_boundary"]) for group in groups),
                "usable": sum(int(group["usable"]) for group in groups),
            }
        )
    if counts[0]["mixed"] >= MIN_ARM_A_MIXED and counts[0]["core"] >= MIN_ARM_A_CORE:
        raise ValueError("S2 was not authorized because S1 already met Arm-A readiness")
    combined_mixed = sum(row["mixed"] for row in counts)
    combined_core = sum(row["core"] for row in counts)
    if combined_mixed >= MIN_ARM_A_MIXED and combined_core >= MIN_ARM_A_CORE:
        raise ValueError("Arm A is training-ready; Arm B fallback is forbidden")
    return {
        "reason": "arm_a_s1_s2_readiness_failed",
        "pool_counts": counts,
        "combined": {"mixed": combined_mixed, "core": combined_core},
        "thresholds": {"mixed": MIN_ARM_A_MIXED, "core": MIN_ARM_A_CORE},
    }


def _validate_validation_failure(
    *,
    pools: Sequence[dict[str, Any]],
    selection_manifest_path: Path,
    expected_selection_manifest_sha256: str,
    generation_manifest_path: Path,
    expected_generation_manifest_sha256: str,
    validation_tasks_path: Path,
    expected_validation_tasks_sha256: str,
    trajectories_path: Path,
    expected_trajectories_sha256: str,
) -> dict[str, Any]:
    selection_bytes = read_bound_bytes(
        selection_manifest_path,
        expected_selection_manifest_sha256,
        role="arm_a_selection_manifest",
    )
    generation_bytes = read_bound_bytes(
        generation_manifest_path,
        expected_generation_manifest_sha256,
        role="arm_a_validation_generation_manifest",
    )
    task_bytes = read_bound_bytes(
        validation_tasks_path,
        expected_validation_tasks_sha256,
        role="arm_a_validation_tasks",
    )
    trajectory_bytes = read_bound_bytes(
        trajectories_path,
        expected_trajectories_sha256,
        role="arm_a_validation_trajectories",
    )
    selection_sha = sha256_bytes(selection_bytes)
    generation_sha = sha256_bytes(generation_bytes)
    tasks_sha = sha256_bytes(task_bytes)
    trajectories_sha = sha256_bytes(trajectory_bytes)
    selection = parse_json_bytes(selection_bytes, path=selection_manifest_path)
    generation = parse_json_bytes(generation_bytes, path=generation_manifest_path)
    tasks = parse_jsonl_bytes(task_bytes, path=validation_tasks_path)
    trajectories = parse_jsonl_bytes(trajectory_bytes, path=trajectories_path)

    declared_pools = selection.get("screen_pools")
    if not isinstance(declared_pools, list) or len(declared_pools) != len(pools):
        raise ValueError("Arm-A selection does not bind the supplied actual screen pools")
    for declared, pool in zip(declared_pools, pools, strict=True):
        if not isinstance(declared, dict) or any(
            declared.get(key) != pool[key]
            for key in ("audit_path", "audit_sha256", "tasks_path", "tasks_sha256")
        ):
            raise ValueError("Arm-A selection screen-pool binding mismatch")

    result = audit_boundary_validation(
        selection_manifest=selection,
        generation_manifest=generation,
        tasks=tasks,
        trajectories=trajectories,
        expected_selection_manifest_sha256=expected_selection_manifest_sha256,
        selection_manifest_sha256=selection_sha,
        generation_manifest_sha256=generation_sha,
        tasks_sha256=tasks_sha,
        trajectories_sha256=trajectories_sha,
        selection_manifest_path=selection_manifest_path,
        generation_manifest_path=generation_manifest_path,
        tasks_path=validation_tasks_path,
        trajectories_path=trajectories_path,
    )
    checks = result.get("checks") or {}
    required_true = (
        "explicit_selection_manifest_digest",
        "frozen_selection_contract",
        "validation_tasks_byte_bound",
        "independent_validation_seed",
        "initial_sft1_policy",
        "generation_manifest_contract",
        "exact_trajectory_task_binding",
    )
    failed_required = [name for name in required_true if checks.get(name) is not True]
    if failed_required:
        raise ValueError(
            "Arm-A validation evidence is structurally/identity invalid: "
            f"{failed_required}"
        )
    if result.get("observed", {}).get("trajectories") != 256:
        raise ValueError("Arm-A validation failure is not one complete 32xK8 attempt")
    runtime_pass = checks.get("no_runtime_contamination") is True
    mixed_pass = checks.get("at_least_20_mixed_outcome_groups") is True
    if result.get("status", {}).get("validation_admitted") is True or (
        runtime_pass and mixed_pass
    ):
        raise ValueError("Arm-A validation passed; Arm B fallback is forbidden")
    return {
        "reason": "arm_a_validation_failed",
        "recomputed_status": result["status"],
        "failed_terminal_gates": {
            "no_runtime_contamination": runtime_pass,
            "at_least_20_mixed_outcome_groups": mixed_pass,
        },
        "bound_inputs": {
            "selection_manifest": {
                "path": str(selection_manifest_path.resolve()),
                "sha256": selection_sha,
            },
            "generation_manifest": {
                "path": str(generation_manifest_path.resolve()),
                "sha256": generation_sha,
            },
            "tasks": {"path": str(validation_tasks_path.resolve()), "sha256": tasks_sha},
            "trajectories": {
                "path": str(trajectories_path.resolve()),
                "sha256": trajectories_sha,
            },
        },
    }


def _remap(rows: Sequence[dict[str, Any]], remote_db_root: Path) -> list[dict[str, Any]]:
    remapped: list[dict[str, Any]] = []
    for row in rows:
        identifier = task_id(row)
        db_id = row.get("db_id")
        source = Path(str(row.get("db_path") or ""))
        if not isinstance(db_id, str) or not db_id or source.parent.name != db_id:
            raise ValueError(f"{identifier}: db_id/db_path identity mismatch")
        retained = dict(row)
        retained["db_path"] = str(remote_db_root / db_id / source.name)
        metadata = dict(retained.get("metadata") or {})
        metadata["rl_training_cohort"] = SCHEMA_VERSION
        metadata["policy_boundary_screen_stage"] = "Arm-B-wide-K8"
        retained["metadata"] = metadata
        remapped.append(retained)
    return remapped


def freeze_wide_cohort(
    *,
    trigger_mode: str,
    screen_audit_paths: Sequence[Path],
    expected_screen_audit_hashes: Sequence[str],
    screen_task_paths: Sequence[Path],
    expected_screen_task_hashes: Sequence[str],
    reference_path: Path,
    eligible_path: Path,
    baseline_eval300_path: Path,
    sft1_index_path: Path,
    old_mixed60_path: Path,
    output_path: Path,
    manifest_path: Path,
    remote_db_root: Path,
    arm_a_selection_manifest_path: Path | None = None,
    expected_arm_a_selection_manifest_sha256: str | None = None,
    arm_a_validation_manifest_path: Path | None = None,
    expected_arm_a_validation_manifest_sha256: str | None = None,
    arm_a_validation_tasks_path: Path | None = None,
    expected_arm_a_validation_tasks_sha256: str | None = None,
    arm_a_validation_trajectories_path: Path | None = None,
    expected_arm_a_validation_trajectories_sha256: str | None = None,
    expected_count: int = EXPECTED_COUNT,
    expected_source_hashes: Mapping[str, str] = CANONICAL_HASHES,
    expected_source_counts: Mapping[str, tuple[int, int]] = CANONICAL_COUNTS,
) -> dict[str, Any]:
    source_paths = {
        "reference": reference_path,
        "eligible": eligible_path,
        "baseline_eval300": baseline_eval300_path,
        "sft1_index": sft1_index_path,
        "old_mixed60": old_mixed60_path,
    }
    if set(source_paths) != set(expected_source_hashes):
        raise ValueError("canonical Arm-B source hash roles are incomplete")
    source_bytes = {
        role: read_bound_bytes(path, expected_source_hashes[role], role=role)
        for role, path in source_paths.items()
    }
    actual_hashes = {role: sha256_bytes(data) for role, data in source_bytes.items()}
    rows_by_role: dict[str, list[dict[str, Any]]] = {}
    ids_by_role: dict[str, set[str]] = {}
    exclusion_artifacts: list[dict[str, Any]] = []
    for role, path in source_paths.items():
        rows = parse_jsonl_bytes(source_bytes[role], path=path)
        rows_by_role[role] = rows
        if role in {"baseline_eval300", "sft1_index", "old_mixed60"}:
            identifiers, artifact = _exclusion_ids(
                path, rows, role=role, digest=actual_hashes[role]
            )
            ids_by_role[role] = identifiers
            exclusion_artifacts.append(artifact)
        else:
            ids_by_role[role] = set(_index(rows, source=role))
        expected = expected_source_counts.get(role)
        if expected is None or (len(rows), len(ids_by_role[role])) != expected:
            raise ValueError(
                f"{role}: records/unique ids {(len(rows), len(ids_by_role[role]))}; "
                f"expected {expected}"
            )
    if any(
        (row.get("metadata") or {}).get("tool_round_trip") != "verified"
        for row in rows_by_role["eligible"]
    ):
        raise ValueError("eligible universe contains a non-verified tool round-trip")
    if not ids_by_role["eligible"] <= ids_by_role["reference"]:
        raise ValueError("eligible universe is not a subset of the reference population")

    pools = _load_screen_pools(
        audit_paths=screen_audit_paths,
        expected_audit_hashes=expected_screen_audit_hashes,
        task_paths=screen_task_paths,
        expected_task_hashes=expected_screen_task_hashes,
    )
    if trigger_mode == "readiness":
        activation_evidence = _validate_readiness_failure(pools)
        unexpected = [
            arm_a_selection_manifest_path,
            arm_a_validation_manifest_path,
            arm_a_validation_tasks_path,
            arm_a_validation_trajectories_path,
        ]
        if any(value is not None for value in unexpected):
            raise ValueError("readiness trigger must not receive validation artifacts")
    elif trigger_mode == "validation":
        required = (
            arm_a_selection_manifest_path,
            expected_arm_a_selection_manifest_sha256,
            arm_a_validation_manifest_path,
            expected_arm_a_validation_manifest_sha256,
            arm_a_validation_tasks_path,
            expected_arm_a_validation_tasks_sha256,
            arm_a_validation_trajectories_path,
            expected_arm_a_validation_trajectories_sha256,
        )
        if any(value is None for value in required):
            raise ValueError("validation trigger requires all eight validation path/hash inputs")
        activation_evidence = _validate_validation_failure(
            pools=pools,
            selection_manifest_path=arm_a_selection_manifest_path,  # type: ignore[arg-type]
            expected_selection_manifest_sha256=expected_arm_a_selection_manifest_sha256,  # type: ignore[arg-type]
            generation_manifest_path=arm_a_validation_manifest_path,  # type: ignore[arg-type]
            expected_generation_manifest_sha256=expected_arm_a_validation_manifest_sha256,  # type: ignore[arg-type]
            validation_tasks_path=arm_a_validation_tasks_path,  # type: ignore[arg-type]
            expected_validation_tasks_sha256=expected_arm_a_validation_tasks_sha256,  # type: ignore[arg-type]
            trajectories_path=arm_a_validation_trajectories_path,  # type: ignore[arg-type]
            expected_trajectories_sha256=expected_arm_a_validation_trajectories_sha256,  # type: ignore[arg-type]
        )
    else:
        raise ValueError("trigger_mode must be readiness or validation")

    screen_ids = set().union(*(set(pool["tasks"]) for pool in pools))
    screen_examples = set().union(
        *(
            {int(row["example_index"]) for row in pool["tasks"].values()}
            for pool in pools
        )
    )
    for index, pool in enumerate(pools, start=1):
        artifact = {
            "role": f"arm_a_screen_pool_{index}",
            "path": str(Path(pool["tasks_path"]).resolve()),
            "sha256": pool["tasks_sha256"],
            "records": len(pool["tasks"]),
            "unique_task_ids": len(pool["tasks"]),
            "task_ids": sorted(pool["tasks"]),
            "example_indices": sorted(
                {int(row["example_index"]) for row in pool["tasks"].values()}
            ),
        }
        exclusion_artifacts.append(artifact)
    excluded_source_ids = set().union(
        ids_by_role["baseline_eval300"],
        ids_by_role["sft1_index"],
        ids_by_role["old_mixed60"],
        screen_ids,
    )
    excluded_examples = set(screen_examples)
    for artifact in exclusion_artifacts:
        excluded_examples.update(int(value) for value in artifact["example_indices"])
    eligible_index = _index(rows_by_role["eligible"], source="eligible_alias_guard")
    alias_excluded_ids = {
        identifier
        for identifier, row in eligible_index.items()
        if int(row["example_index"]) in excluded_examples
    }
    excluded = excluded_source_ids | alias_excluded_ids
    selected, details = select_rows(
        rows_by_role["reference"],
        rows_by_role["eligible"],
        excluded,
        count=expected_count,
        seed=SELECTION_SEED,
    )
    selected_ids = [task_id(row) for row in selected]
    selected_id_set = set(selected_ids)
    selected_examples = {int(row["example_index"]) for row in selected}
    cutpoints = length_cutpoints(rows_by_role["reference"])
    distribution = distribution_audit(
        rows_by_role["reference"], selected, cutpoints=cutpoints
    )
    remapped = _remap(selected, remote_db_root)
    output_bytes = jsonl_bytes(remapped)
    output_sha = sha256_bytes(output_bytes)

    gates = {
        "arm_a_pretraining_failure_recomputed": activation_evidence["reason"] in {
            "arm_a_s1_s2_readiness_failed",
            "arm_a_validation_failed",
        },
        "record_count_exact": len(selected) == expected_count,
        "task_ids_unique": len(selected_id_set) == expected_count,
        "overlap_with_all_exclusions_zero": not bool(selected_id_set & excluded),
        "example_index_overlap_with_all_exclusions_zero": not bool(
            selected_examples & excluded_examples
        ),
        "overlap_with_actual_arm_a_screens_zero": not bool(selected_id_set & screen_ids),
        "overlap_with_baseline_eval300_zero": not bool(
            selected_id_set & ids_by_role["baseline_eval300"]
        ),
        "overlap_with_sft1_train678_zero": not bool(
            selected_id_set & ids_by_role["sft1_index"]
        ),
        "overlap_with_old_mixed60_zero": not bool(
            selected_id_set & ids_by_role["old_mixed60"]
        ),
        "all_reference_databases_covered": (
            {str(row.get("db_id") or "") for row in selected}
            == {str(row.get("db_id") or "") for row in rows_by_role["reference"]}
        ),
        "database_tv_at_most_0_04": (
            distribution["axes"]["database"]["total_variation_distance"] <= 0.04
        ),
        "length_quintile_tv_at_most_0_03": (
            distribution["axes"]["question_length_quintile"]["total_variation_distance"]
            <= 0.03
        ),
        "knowledge_rate_delta_at_most_0_02": (
            distribution["axes"]["external_knowledge"]["total_variation_distance"]
            <= 0.02
        ),
        "joint_knowledge_length_tv_at_most_0_04": (
            distribution["axes"]["knowledge_x_length"]["total_variation_distance"]
            <= 0.04
        ),
        "question_length_mean_delta_at_most_0_05": (
            distribution["question_length_characters"]["relative_mean_difference"] <= 0.05
        ),
        "remote_paths_bound": all(
            Path(str(row["db_path"])).is_relative_to(remote_db_root) for row in remapped
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"Arm-B wide3000 gates failed: {[k for k, v in gates.items() if not v]}")

    pool_bindings = [
        {
            "audit_path": str(Path(pool["audit_path"]).resolve()),
            "audit_sha256": pool["audit_sha256"],
            "tasks_path": str(Path(pool["tasks_path"]).resolve()),
            "tasks_sha256": pool["tasks_sha256"],
            "records": len(pool["tasks"]),
            "schema_version": "vanilla-grpo-boundary-screen-audit-v1",
        }
        for pool in pools
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "cohort_namespace": WIDE_COHORT_NAMESPACE,
        "seeds": dict(ARM_B_SEED_REGISTRY),
        "purpose": "independent Arm-B wide K8 screen; no training admission",
        "activation": {
            "decision": "arm_b_pretraining_fallback",
            **activation_evidence,
            "arm_a_failure_evidence": {
                "schema_version": "arm-b-pretraining-fallback-evidence-v1",
                "storage": "inline-in-this-wide-cohort-manifest",
                "reason": activation_evidence["reason"],
                "recomputed": activation_evidence,
                "bound_inputs": {
                    "screen_pools": pool_bindings,
                    "validation": activation_evidence.get("bound_inputs"),
                },
            },
            "actual_screen_pools": pool_bindings,
            "arm_a_training_started": False,
            "arm_a_full_dev_started": False,
            "confirmatory_full_dev_arms_allowed": 1,
            "external_state_limitation": (
                "immutable artifacts prove the pre-training failure; absence of an "
                "external concurrent launch is additionally enforced by the staged launcher"
            ),
        },
        "reference_population": {
            "path": str(reference_path.resolve()),
            "sha256": actual_hashes["reference"],
            "records": len(rows_by_role["reference"]),
        },
        "eligibility_population": {
            "path": str(eligible_path.resolve()),
            "sha256": actual_hashes["eligible"],
            "records": len(rows_by_role["eligible"]),
            "rule": "tool_round_trip=verified and frozen non-empty BIRD-train input",
        },
        "exclusions": exclusion_artifacts,
        "selection": {
            "count": expected_count,
            "seed": SELECTION_SEED,
            "public_fields_used": list(PUBLIC_SELECTION_FIELDS),
            "forbidden_fields_not_used": list(FORBIDDEN_SELECTION_FIELDS),
            "gold_used_for_selection_or_order": False,
            "algorithm": (
                "public DB proportional allocation plus external-knowledge x question-length "
                "stratification and fixed SHA representative order"
            ),
            "details": details,
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": output_sha,
            "records": expected_count,
            "task_ids_in_frozen_order": selected_ids,
            "remote_db_root": str(remote_db_root.resolve()),
        },
        "distribution_audit": distribution,
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": True,
        "downstream_contract": {
            "screen_policy": "fresh initial-SFT1",
            "group_size": 8,
            "seed": 20260814,
            "minimum_clean_mixed_groups": 640,
            "contaminated_groups": "audited and excluded; no replacement sampling",
            "training_admission": "none until fresh K16 confirmation and validation64 pass",
        },
        "gold_visibility": (
            "selection and ordering use no gold/dev field; hidden denotation is Harness-only"
        ),
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    publish_exact(((output_path, output_bytes), (manifest_path, manifest_bytes)))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger-mode", choices=("readiness", "validation"), required=True)
    parser.add_argument("--screen-audit", type=Path, action="append", required=True)
    parser.add_argument("--expected-screen-audit-sha256", action="append", required=True)
    parser.add_argument("--screen-tasks", type=Path, action="append", required=True)
    parser.add_argument("--expected-screen-tasks-sha256", action="append", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--baseline-eval300", type=Path, required=True)
    parser.add_argument("--sft1-index", type=Path, required=True)
    parser.add_argument("--old-mixed60", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--remote-db-root", type=Path, required=True)
    parser.add_argument("--arm-a-selection-manifest", type=Path)
    parser.add_argument("--expected-arm-a-selection-manifest-sha256")
    parser.add_argument("--arm-a-validation-manifest", type=Path)
    parser.add_argument("--expected-arm-a-validation-manifest-sha256")
    parser.add_argument("--arm-a-validation-tasks", type=Path)
    parser.add_argument("--expected-arm-a-validation-tasks-sha256")
    parser.add_argument("--arm-a-validation-trajectories", type=Path)
    parser.add_argument("--expected-arm-a-validation-trajectories-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = freeze_wide_cohort(
            trigger_mode=args.trigger_mode,
            screen_audit_paths=args.screen_audit,
            expected_screen_audit_hashes=args.expected_screen_audit_sha256,
            screen_task_paths=args.screen_tasks,
            expected_screen_task_hashes=args.expected_screen_tasks_sha256,
            reference_path=args.reference,
            eligible_path=args.eligible,
            baseline_eval300_path=args.baseline_eval300,
            sft1_index_path=args.sft1_index,
            old_mixed60_path=args.old_mixed60,
            output_path=args.output,
            manifest_path=args.manifest,
            remote_db_root=args.remote_db_root,
            arm_a_selection_manifest_path=args.arm_a_selection_manifest,
            expected_arm_a_selection_manifest_sha256=args.expected_arm_a_selection_manifest_sha256,
            arm_a_validation_manifest_path=args.arm_a_validation_manifest,
            expected_arm_a_validation_manifest_sha256=args.expected_arm_a_validation_manifest_sha256,
            arm_a_validation_tasks_path=args.arm_a_validation_tasks,
            expected_arm_a_validation_tasks_sha256=args.expected_arm_a_validation_tasks_sha256,
            arm_a_validation_trajectories_path=args.arm_a_validation_trajectories,
            expected_arm_a_validation_trajectories_sha256=(
                args.expected_arm_a_validation_trajectories_sha256
            ),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Arm-B wide3000 preparation blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": manifest["output"], "activation": manifest["activation"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
