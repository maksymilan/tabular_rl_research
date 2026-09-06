#!/usr/bin/env python3
"""Prepare the gated public-field-only S2 extra600 screening cohort.

S2 is not a second training cohort.  It is an additional initial-SFT1 K=8
screening pool that may be materialized only after the frozen S1 boundary audit
has admitted its own artifacts but returned ``next_stage=screen_s2``.  Selection
reuses the representative public-field allocator from the vanilla-GRPO cohort;
gold SQL remains Harness-only and cannot affect selection or order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl.scenarios.data.select_vanilla_grpo_tasks import (
    FORBIDDEN_SELECTION_FIELDS,
    PUBLIC_SELECTION_FIELDS,
    distribution_audit,
    exclusion_ids,
    jsonl_bytes,
    length_cutpoints,
    read_jsonl,
    remap_database_paths,
    select_rows,
    sha256_file,
    task_id,
    write_atomic,
)


SCHEMA_VERSION = "bird-train-policy-boundary-s2-extra600-v1"
S1_AUDIT_SCHEMA = "vanilla-grpo-boundary-screen-audit-v1"
S1_COHORT_SCHEMA = "bird-train-vanilla-grpo-cohort-v1"
DEFAULT_SEED = "qwen3-v26-policy-boundary-s2-extra600-v1-20260812"
EXPECTED_INITIAL_ADAPTER_SHA256 = (
    "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5"
)
EXPECTED_PROTOCOL_VERSION = "version26"
EXPECTED_PROTOCOL_HASH = "4da19387399bd3a5"
EXPECTED_SCREEN_SEED = 20260812
EXPECTED_COUNT = 600

# Every input whose bytes are already frozen is literal here.  The S1 screen
# audit does not exist until S1 finishes; callers must pass its newly frozen
# digest explicitly rather than allowing an unknown placeholder to pass.
CANONICAL_INPUT_SHA256 = {
    "reference": "24d21a349c430df549f92724dd07de1c59b2365cd72802957120afba13c7a2e1",
    "eligible": "c8ea3c6419ac57f05021eca470c7519843456e8ac02b04cca9af39e779ec7545",
    "s1_tasks": "b5a83c373e9be094ea7355c0bc23c9212249491491f44dfdcb3b09ea49b2457e",
    "s1_cohort_manifest": "9212d1f7ca4fb63576eb7e846a9b05f7d159e131fe992495b99e350b6aae8dd6",
    "baseline_eval300": "87b37b307ea2710acdff7d03bdc474a6ba2cdb8ed51b005cff0fe8fa54c67c03",
    "sft1_index": "aa6b8a72a1cfbcf44e37702e0498c9a85f615edc201ba3f4172b65f701e0cda5",
    "old_mixed60": "ef97b6977735996fde505aaad4e00215570963b37a1c82d890c34c4b414f30ae",
}
CANONICAL_INPUT_COUNTS = {
    "reference": (6601, 6601),
    "eligible": (5915, 5915),
    "s1_tasks": (600, 600),
    "baseline_eval300": (300, 300),
    "sft1_index": (4471, 678),
    "old_mixed60": (60, 60),
}
EXCLUSION_ROLES = (
    "s1_tasks",
    "baseline_eval300",
    "sft1_index",
    "old_mixed60",
)


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def _require_digest(path: Path, expected: str, *, role: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{role}: missing/non-regular input {path}")
    if not _valid_sha256(expected):
        raise ValueError(f"{role}: expected SHA-256 is not 64 lowercase hex")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{role}: SHA-256 mismatch for {path}: {actual} != {expected}"
        )
    return actual


def _index_task_identity(
    rows: Sequence[dict[str, Any]], *, source: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    example_indices: dict[int, str] = {}
    for position, row in enumerate(rows):
        identifier = task_id(row, source=f"{source}:{position + 1}")
        example_index = row.get("example_index")
        if type(example_index) is not int or example_index < 0:
            raise ValueError(f"{source}:{position + 1}: invalid example_index")
        expected_identifier = f"bird_train_{example_index:05d}"
        if identifier != expected_identifier or row.get("instance_id") != identifier:
            raise ValueError(
                f"{source}:{position + 1}: task/example identity mismatch "
                f"({identifier!r}, {example_index!r})"
            )
        if identifier in indexed or example_index in example_indices:
            raise ValueError(f"{source}: duplicate task/example identity {identifier}")
        indexed[identifier] = row
        example_indices[example_index] = identifier
    return indexed


def _validate_s1_activation(
    *,
    audit_path: Path,
    expected_audit_sha256: str,
    s1_tasks_path: Path,
    s1_tasks: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit_sha = _require_digest(
        audit_path, expected_audit_sha256, role="s1_requires_s2_audit"
    )
    audit = _load_json(audit_path)
    if audit.get("schema_version") != S1_AUDIT_SCHEMA:
        raise ValueError("S1 audit has an unsupported schema")
    if audit.get("issues") != [] or audit.get("issue_counts") != {}:
        raise ValueError("S1 audit contains unresolved issues")
    status = audit.get("status")
    if not isinstance(status, dict) or status != {
        "audit_passes": True,
        "pool_admitted": True,
        "selection_ready_without_s2": False,
        "next_stage": "screen_s2",
    }:
        raise ValueError(
            "S2 is forbidden unless the frozen S1 audit decision is requires_s2 "
            "(next_stage=screen_s2)"
        )
    contract = audit.get("contract")
    required_contract = {
        "tasks": 600,
        "group_size": 8,
        "optimizer_updates": 0,
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "initial_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
        "screen_seed": EXPECTED_SCREEN_SEED,
        "generation_seed_scheme": "sha256-task-sample-turn-v1",
        "minimum_mixed_groups": 360,
        "minimum_core_groups": 240,
    }
    if not isinstance(contract, dict) or any(
        contract.get(key) != expected for key, expected in required_contract.items()
    ):
        raise ValueError("S1 audit contract differs from the frozen boundary screen")

    observed = audit.get("observed")
    if not isinstance(observed, dict):
        raise ValueError("S1 audit has no aggregate observations")
    mixed_count = observed.get("mixed_boundary_groups")
    core_count = observed.get("core_boundary_groups")
    if (
        observed.get("tasks") != 600
        or observed.get("trajectories") != 4800
        or type(mixed_count) is not int
        or type(core_count) is not int
        or not (mixed_count < 360 or core_count < 240)
    ):
        raise ValueError("S1 aggregate evidence does not justify requires_s2")

    inputs = audit.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("S1 audit has no bound input record")
    for key in ("manifest", "trajectories", "tasks"):
        recorded = inputs.get(key)
        expected_digest = inputs.get(f"{key}_sha256")
        if not isinstance(recorded, str) or not _valid_sha256(expected_digest):
            raise ValueError(f"S1 audit input binding is incomplete: {key}")
        recorded_path = Path(recorded)
        if not recorded_path.is_file() or recorded_path.is_symlink():
            raise ValueError(f"S1 audit bound input is unavailable: {recorded_path}")
        if sha256_file(recorded_path) != expected_digest:
            raise ValueError(f"S1 audit bound input changed after audit: {recorded_path}")
    if (
        Path(str(inputs["tasks"])).resolve() != s1_tasks_path.resolve()
        or inputs["tasks_sha256"] != sha256_file(s1_tasks_path)
    ):
        raise ValueError("S1 audit is not bound to the supplied current S1 600")

    s1_index = _index_task_identity(s1_tasks, source="current S1 tasks")
    groups = audit.get("groups")
    if not isinstance(groups, list) or len(groups) != len(s1_tasks):
        raise ValueError("S1 audit does not contain exactly 600 group summaries")
    seen: set[str] = set()
    recomputed_mixed = 0
    recomputed_core = 0
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("S1 group summary must be an object")
        identifier = group.get("task_id")
        correct_count = group.get("correct_count")
        usable = group.get("usable")
        contamination = group.get("contamination")
        mixed = group.get("mixed_boundary")
        core = group.get("core_boundary")
        if (
            not isinstance(identifier, str)
            or identifier not in s1_index
            or identifier in seen
            or group.get("example_index") != s1_index[identifier]["example_index"]
            or group.get("trajectories") != 8
            or type(correct_count) is not int
            or not 0 <= correct_count <= 8
            or type(usable) is not bool
            or not isinstance(contamination, list)
            or bool(contamination) == usable
            or type(mixed) is not bool
            or type(core) is not bool
            or mixed != (usable and 1 <= correct_count <= 7)
            or core != (usable and 2 <= correct_count <= 6)
        ):
            raise ValueError(f"invalid S1 group/task identity: {identifier!r}")
        seen.add(identifier)
        recomputed_mixed += int(mixed)
        recomputed_core += int(core)
    if seen != set(s1_index):
        raise ValueError("S1 audit group identities do not match current S1 tasks")
    if (recomputed_mixed, recomputed_core) != (mixed_count, core_count):
        raise ValueError("S1 group summaries do not reconcile with aggregate evidence")

    return audit, {
        "decision": "requires_s2",
        "path": str(audit_path.resolve()),
        "sha256": audit_sha,
        "schema_version": S1_AUDIT_SCHEMA,
        "status": status,
        "bound_inputs": {
            key: {
                "path": str(Path(inputs[key]).resolve()),
                "sha256": inputs[f"{key}_sha256"],
            }
            for key in ("manifest", "trajectories", "tasks")
        },
    }


def _validate_s1_cohort_manifest(
    *,
    manifest_path: Path,
    s1_tasks_path: Path,
    source_paths: Mapping[str, Path],
    expected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema_version") != S1_COHORT_SCHEMA
        or manifest.get("status") != "frozen_training_cohort"
        or manifest.get("all_acceptance_gates_passed") is not True
    ):
        raise ValueError("current S1 cohort manifest is not frozen/admitted")
    reference = manifest.get("reference_population") or {}
    eligibility = manifest.get("eligibility_population") or {}
    if (
        reference.get("sha256") != expected_hashes["reference"]
        or eligibility.get("sha256") != expected_hashes["eligible"]
    ):
        raise ValueError("S1 did not use the same frozen reference/eligible universe")
    output = manifest.get("output") or {}
    s1_rows = read_jsonl(s1_tasks_path)
    s1_ids = [task_id(row) for row in s1_rows]
    if (
        output.get("sha256") != expected_hashes["s1_tasks"]
        or output.get("records") != 600
        or output.get("task_ids_in_frozen_order") != s1_ids
    ):
        raise ValueError("S1 cohort manifest/output binding mismatch")

    expected_exclusions = {
        expected_hashes[role]: role
        for role in ("baseline_eval300", "sft1_index", "old_mixed60")
    }
    observed: dict[str, dict[str, Any]] = {}
    for artifact in manifest.get("exclusions") or []:
        if not isinstance(artifact, dict):
            raise ValueError("S1 cohort exclusion entry is not an object")
        role = expected_exclusions.get(artifact.get("sha256"))
        if role is not None:
            if role in observed:
                raise ValueError(f"duplicate S1 exclusion role: {role}")
            rows = read_jsonl(source_paths[role])
            ids, _ = exclusion_ids([source_paths[role]])
            if (
                artifact.get("records") != len(rows)
                or artifact.get("unique_task_ids") != len(ids)
                or artifact.get("task_ids") != sorted(ids)
            ):
                raise ValueError(f"S1 manifest exclusion binding mismatch: {role}")
            observed[role] = artifact
    if set(observed) != set(expected_exclusions.values()):
        raise ValueError("S1 manifest does not bind all three original exclusions")
    return manifest


def prepare_s2_pool(
    *,
    s1_audit_path: Path,
    expected_s1_audit_sha256: str,
    reference_path: Path,
    eligible_path: Path,
    s1_tasks_path: Path,
    s1_cohort_manifest_path: Path,
    baseline_eval300_path: Path,
    sft1_index_path: Path,
    old_mixed60_path: Path,
    output_path: Path,
    manifest_path: Path,
    remote_db_root: Path,
    seed: str = DEFAULT_SEED,
    count: int = EXPECTED_COUNT,
    expected_hashes: Mapping[str, str] = CANONICAL_INPUT_SHA256,
    expected_counts: Mapping[str, tuple[int, int]] = CANONICAL_INPUT_COUNTS,
) -> dict[str, Any]:
    """Validate the requires-S2 decision, select extra tasks, and freeze output."""
    if count <= 0:
        raise ValueError("S2 count must be positive")
    if output_path.exists() or manifest_path.exists():
        raise ValueError("refusing to overwrite an existing S2 output or manifest")
    if output_path.resolve() == manifest_path.resolve():
        raise ValueError("S2 output and manifest paths must differ")

    source_paths = {
        "reference": reference_path,
        "eligible": eligible_path,
        "s1_tasks": s1_tasks_path,
        "s1_cohort_manifest": s1_cohort_manifest_path,
        "baseline_eval300": baseline_eval300_path,
        "sft1_index": sft1_index_path,
        "old_mixed60": old_mixed60_path,
    }
    if set(expected_hashes) != set(source_paths):
        raise ValueError("expected input hash roles are incomplete or unknown")
    actual_hashes = {
        role: _require_digest(path, expected_hashes[role], role=role)
        for role, path in source_paths.items()
    }

    rows_by_role = {
        role: read_jsonl(path)
        for role, path in source_paths.items()
        if role != "s1_cohort_manifest"
    }
    unique_ids_by_role: dict[str, set[str]] = {}
    for role, rows in rows_by_role.items():
        if role == "sft1_index":
            ids, _ = exclusion_ids([source_paths[role]])
        else:
            indexed = _index_task_identity(rows, source=role)
            ids = set(indexed)
        unique_ids_by_role[role] = ids
        expected = expected_counts.get(role)
        if expected is None or (len(rows), len(ids)) != expected:
            raise ValueError(
                f"{role}: observed records/unique ids {(len(rows), len(ids))}; "
                f"expected {expected}"
            )

    reference_index = _index_task_identity(
        rows_by_role["reference"], source="reference"
    )
    invalid_eligible = [
        task_id(row)
        for row in rows_by_role["eligible"]
        if (row.get("metadata") or {}).get("tool_round_trip") != "verified"
    ]
    if invalid_eligible:
        raise ValueError(
            "eligible universe contains tasks without verified tool round-trip: "
            f"{invalid_eligible[:5]}"
        )
    for role, ids in unique_ids_by_role.items():
        if role == "reference":
            continue
        unknown = sorted(ids - set(reference_index))
        if unknown:
            raise ValueError(f"{role}: task ids absent from reference: {unknown[:5]}")
    for identifier in unique_ids_by_role["sft1_index"]:
        expected_identifier = (
            f"bird_train_{int(identifier.rsplit('_', 1)[-1]):05d}"
            if re.fullmatch(r"bird_train_[0-9]{5}", identifier)
            else None
        )
        if identifier != expected_identifier:
            raise ValueError(f"sft1_index: invalid source task identity {identifier!r}")

    _s1_manifest = _validate_s1_cohort_manifest(
        manifest_path=s1_cohort_manifest_path,
        s1_tasks_path=s1_tasks_path,
        source_paths=source_paths,
        expected_hashes=expected_hashes,
    )
    _s1_audit, activation = _validate_s1_activation(
        audit_path=s1_audit_path,
        expected_audit_sha256=expected_s1_audit_sha256,
        s1_tasks_path=s1_tasks_path,
        s1_tasks=rows_by_role["s1_tasks"],
    )

    exclusion_paths = [source_paths[role] for role in EXCLUSION_ROLES]
    excluded, exclusion_artifacts = exclusion_ids(exclusion_paths)
    for role, artifact in zip(EXCLUSION_ROLES, exclusion_artifacts, strict=True):
        artifact["role"] = role
        if artifact["sha256"] != actual_hashes[role]:
            raise ValueError(f"exclusion hash changed during selection: {role}")
    selected, details = select_rows(
        rows_by_role["reference"],
        rows_by_role["eligible"],
        excluded,
        count=count,
        seed=seed,
    )
    selected_ids = [task_id(row) for row in selected]
    selected_id_set = set(selected_ids)
    cutpoints = length_cutpoints(rows_by_role["reference"])
    distribution = distribution_audit(
        rows_by_role["reference"], selected, cutpoints=cutpoints
    )
    remapped = remap_database_paths(selected, remote_db_root)
    for row in remapped:
        metadata = dict(row.get("metadata") or {})
        metadata["rl_training_cohort"] = SCHEMA_VERSION
        metadata["policy_boundary_screen_stage"] = "S2"
        row["metadata"] = metadata
    rendered = jsonl_bytes(remapped)
    output_sha = hashlib.sha256(rendered).hexdigest()

    gates = {
        "s1_requires_s2": activation["decision"] == "requires_s2",
        "record_count_exact": len(selected) == count,
        "task_ids_unique": len(selected_id_set) == count,
        "task_example_identity_bound": all(
            identifier == f"bird_train_{row['example_index']:05d}"
            for identifier, row in zip(selected_ids, remapped, strict=True)
        ),
        "same_eligible_universe": (
            actual_hashes["reference"] == expected_hashes["reference"]
            and actual_hashes["eligible"] == expected_hashes["eligible"]
        ),
        "overlap_with_all_exclusions_zero": not bool(selected_id_set & excluded),
        "overlap_with_current_s1_zero": not bool(
            selected_id_set & unique_ids_by_role["s1_tasks"]
        ),
        "overlap_with_baseline_eval300_zero": not bool(
            selected_id_set & unique_ids_by_role["baseline_eval300"]
        ),
        "overlap_with_sft1_train678_zero": not bool(
            selected_id_set & unique_ids_by_role["sft1_index"]
        ),
        "overlap_with_old_mixed60_zero": not bool(
            selected_id_set & unique_ids_by_role["old_mixed60"]
        ),
        "all_reference_databases_covered": (
            {str(row.get("db_id") or "") for row in selected}
            == {
                str(row.get("db_id") or "")
                for row in rows_by_role["reference"]
            }
        ),
        "database_tv_at_most_0_04": (
            distribution["axes"]["database"]["total_variation_distance"] <= 0.04
        ),
        "length_quintile_tv_at_most_0_03": (
            distribution["axes"]["question_length_quintile"]
            ["total_variation_distance"]
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
            distribution["question_length_characters"]["relative_mean_difference"]
            <= 0.05
        ),
        "remote_paths_bound": all(
            Path(str(row["db_path"])).is_relative_to(remote_db_root)
            for row in remapped
        ),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise ValueError(f"S2 extra600 acceptance gates failed: {failed}")

    overlap_matrix: dict[str, dict[str, int]] = {}
    for left in EXCLUSION_ROLES:
        overlap_matrix[left] = {
            right: len(unique_ids_by_role[left] & unique_ids_by_role[right])
            for right in EXCLUSION_ROLES
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_s2_screening_cohort",
        "purpose": (
            "additional public-field-selected initial-SFT1 K8 screening pool; "
            "not an admitted training cohort"
        ),
        "activation_gate": activation,
        "reference_population": {
            "path": str(reference_path.resolve()),
            "sha256": actual_hashes["reference"],
            "records": len(rows_by_role["reference"]),
        },
        "eligibility_population": {
            "path": str(eligible_path.resolve()),
            "sha256": actual_hashes["eligible"],
            "records": len(rows_by_role["eligible"]),
            "rule": "same frozen tool_round_trip=verified non-empty BIRD-train universe as S1",
        },
        "current_s1_cohort_manifest": {
            "path": str(s1_cohort_manifest_path.resolve()),
            "sha256": actual_hashes["s1_cohort_manifest"],
        },
        "exclusions": exclusion_artifacts,
        "exclusion_overlap_matrix": overlap_matrix,
        "selection": {
            "count": count,
            "seed": seed,
            "public_fields_used": list(PUBLIC_SELECTION_FIELDS),
            "forbidden_fields_not_used": list(FORBIDDEN_SELECTION_FIELDS),
            "gold_used_for_selection_or_order": False,
            "algorithm": (
                "shared public-field proportional database allocation with "
                "knowledge x question-length stratification and deterministic "
                "representative order"
            ),
            "details": details,
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": output_sha,
            "records": len(remapped),
            "unique_databases": len({row["db_id"] for row in remapped}),
            "remote_db_root": str(remote_db_root),
            "task_ids_in_frozen_order": selected_ids,
            "task_example_identity_in_frozen_order": [
                {
                    "task_id": identifier,
                    "example_index": int(row["example_index"]),
                    "db_id": str(row["db_id"]),
                }
                for identifier, row in zip(selected_ids, remapped, strict=True)
            ],
        },
        "distribution_audit": distribution,
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": True,
        "downstream_contract": {
            "screen_policy": "fresh initial-SFT1",
            "screen_group_size": 8,
            "screen_seed": EXPECTED_SCREEN_SEED,
            "merge_rule": "S1+S2 audits only; no S3 and no threshold relaxation",
            "training_admission": "none_until_existing_boundary332_selection_and_validation32_gates_pass",
        },
        "gold_visibility": "gold SQL remains Harness-only and is never used for selection/order",
    }
    encoded_manifest = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode()
    # No path is touched before every source, activation, selection, and output
    # gate passes.  Publish the task bytes first and the manifest last.
    write_atomic(output_path, rendered)
    write_atomic(manifest_path, encoded_manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-audit", type=Path, required=True)
    parser.add_argument("--expected-s1-audit-sha256", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--s1-tasks", type=Path, required=True)
    parser.add_argument("--s1-cohort-manifest", type=Path, required=True)
    parser.add_argument("--baseline-eval300", type=Path, required=True)
    parser.add_argument("--sft1-index", type=Path, required=True)
    parser.add_argument("--old-mixed60", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--remote-db-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = prepare_s2_pool(
            s1_audit_path=args.s1_audit,
            expected_s1_audit_sha256=args.expected_s1_audit_sha256,
            reference_path=args.reference,
            eligible_path=args.eligible,
            s1_tasks_path=args.s1_tasks,
            s1_cohort_manifest_path=args.s1_cohort_manifest,
            baseline_eval300_path=args.baseline_eval300,
            sft1_index_path=args.sft1_index,
            old_mixed60_path=args.old_mixed60,
            output_path=args.output,
            manifest_path=args.manifest,
            remote_db_root=args.remote_db_root,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"S2 preparation blocked: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": manifest["output"],
                "activation_gate": manifest["activation_gate"],
                "acceptance_gates": manifest["acceptance_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
