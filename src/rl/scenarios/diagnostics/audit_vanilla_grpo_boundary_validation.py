#!/usr/bin/env python3
"""Audit the fresh held-out validation32 gate for boundary vanilla GRPO.

This is intentionally a thin, stricter layer over the frozen 32 x K8 rollout
probe audit.  It adds byte-level binding to the policy-boundary selection,
requires the independent validation seed, verifies every trajectory against
the selected task row, and rejects *any* runtime-ineligible trajectory.  Its
output is a boundary-validation admission and can never be substituted by the
earlier representative600/first32 probe admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Script execution from diagnostics/.
    from rl.scenarios.diagnostics.audit_vanilla_grpo_rollout_probe import audit as audit_base_probe
except ImportError:  # Package import in CPU tests.
    from rl.scenarios.diagnostics.audit_vanilla_grpo_rollout_probe import (
        audit as audit_base_probe,
    )


SCHEMA_VERSION = "vanilla-grpo-boundary-validation-audit-v1"
SELECTION_SCHEMA = "policy-boundary-grpo-cohort-v1"
PENDING_MANIFEST_SCHEMA = "table-agent-fixed-rollout-pool-pending-v1"
TRAJECTORY_SCHEMA = "table-agent-fixed-policy-episode-v1"
EXPECTED_PROTOCOL_VERSION = "version26"
EXPECTED_PROTOCOL_HASH = "4da19387399bd3a5"
EXPECTED_STUDENT_PROMPT_SHA256 = (
    "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
)
EXPECTED_RUNTIME_TREE_SHA256 = (
    "5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab"
)
EXPECTED_INITIAL_ADAPTER_SHA256 = (
    "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5"
)
EXPECTED_SELECTION_SEED = "qwen3-v26-boundary332-v1-20260812"
SCREEN_SEED = 20260812
VALIDATION_SEED = 20260813
EXPECTED_TASKS = 32
EXPECTED_GROUP_SIZE = 8
EXPECTED_TRAJECTORIES = EXPECTED_TASKS * EXPECTED_GROUP_SIZE
MIN_MIXED_GROUPS = 20


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_json(data: bytes, *, path: Path) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def _parse_jsonl(data: bytes, *, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be a JSON object")
        rows.append(value)
    return rows


def _task_id(row: Mapping[str, Any]) -> str:
    value = row.get("example_id") or row.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError("validation task is missing example_id/instance_id")
    return value


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def audit_boundary_validation(
    *,
    selection_manifest: Mapping[str, Any],
    generation_manifest: dict[str, Any],
    tasks: Sequence[dict[str, Any]],
    trajectories: Sequence[dict[str, Any]],
    expected_selection_manifest_sha256: str,
    selection_manifest_sha256: str,
    generation_manifest_sha256: str,
    tasks_sha256: str,
    trajectories_sha256: str,
    selection_manifest_path: Path | None = None,
    generation_manifest_path: Path | None = None,
    tasks_path: Path | None = None,
    trajectories_path: Path | None = None,
) -> dict[str, Any]:
    """Return a complete fail-closed boundary validation audit."""

    issues: list[dict[str, str]] = []
    issue_counts: Counter[str] = Counter()

    def issue(code: str, location: str, detail: str) -> None:
        issue_counts[code] += 1
        issues.append({"code": code, "location": location, "detail": detail})

    if not _valid_sha256(expected_selection_manifest_sha256):
        issue(
            "selection_manifest_expected_digest_invalid",
            "expected_selection_manifest_sha256",
            "an explicit 64-character lowercase SHA-256 is required",
        )
    elif selection_manifest_sha256 != expected_selection_manifest_sha256:
        issue(
            "selection_manifest_digest_mismatch",
            "selection_manifest",
            (
                f"actual {selection_manifest_sha256}; "
                f"expected {expected_selection_manifest_sha256}"
            ),
        )

    if (
        selection_manifest.get("schema_version") != SELECTION_SCHEMA
        or selection_manifest.get("status") != "frozen_boundary_training_cohort"
    ):
        issue(
            "selection_manifest_contract_mismatch",
            "selection_manifest",
            "unsupported schema/status",
        )

    selection = selection_manifest.get("selection")
    contract = selection_manifest.get("contract")
    outputs = selection_manifest.get("outputs")
    task_ids = selection_manifest.get("task_ids")
    if not all(isinstance(value, dict) for value in (selection, contract, outputs, task_ids)):
        issue(
            "selection_manifest_contract_mismatch",
            "selection_manifest",
            "selection, contract, outputs, and task_ids must be objects",
        )
        selection = selection if isinstance(selection, dict) else {}
        contract = contract if isinstance(contract, dict) else {}
        outputs = outputs if isinstance(outputs, dict) else {}
        task_ids = task_ids if isinstance(task_ids, dict) else {}

    formal_contract = {
        "optimizer_updates": 20,
        "prompts_per_update": 30,
        "group_size": 8,
        "train_passes": 2,
        "prompt_appearances": 600,
        "fresh_online_trajectories": 4800,
        "sampler": "trl-0.29-repeat-sampler-v1",
        "shuffle_dataset": True,
        "data_seed": 20260812,
        "task_order": "two deterministic data-seed shuffled passes",
        "per_pass_coverage": "each train300 identity exactly once",
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
    }
    validation_contract = {
        "policy": "fresh initial-SFT1",
        "records": 32,
        "group_size": 8,
        "seed": VALIDATION_SEED,
        "screen_seed_must_differ": SCREEN_SEED,
        "generation_seed_scheme": "sha256-task-sample-turn-v1",
        "gate": "same >=20/32 mixed probe gate",
        "runtime_contamination_allowed": False,
        "screen_trajectories_reused": False,
    }
    expected_contract = {
        "boundary_records": 332,
        "train_records": 300,
        "validation_records": 32,
        "formal_training": formal_contract,
        "validation": validation_contract,
        "primary_checkpoint": "final-step20-only",
    }
    if contract != expected_contract:
        issue(
            "selection_manifest_contract_mismatch",
            "selection_manifest.contract",
            "formal training/validation contract differs from the frozen boundary design",
        )
    gates = selection.get("acceptance_gates")
    if (
        selection.get("seed") != EXPECTED_SELECTION_SEED
        or not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
    ):
        issue(
            "selection_manifest_contract_mismatch",
            "selection_manifest.selection",
            "selection seed or acceptance gates are not frozen/passing",
        )

    task_rows_valid = True
    observed_task_ids: list[str] = []
    example_indices: set[int] = set()
    try:
        for row in tasks:
            task_id = _task_id(row)
            example_index = row.get("example_index")
            if (
                task_id in observed_task_ids
                or type(example_index) is not int
                or example_index in example_indices
            ):
                raise ValueError(f"duplicate/invalid task identity: {task_id}")
            if not all(
                isinstance(row.get(field), str) and bool(str(row.get(field)).strip())
                for field in ("db_id", "db_path", "question")
            ):
                raise ValueError(f"task lacks required public/runtime fields: {task_id}")
            observed_task_ids.append(task_id)
            example_indices.add(example_index)
    except ValueError as exc:
        task_rows_valid = False
        issue("validation_tasks_invalid", "tasks", str(exc))
    if len(tasks) != EXPECTED_TASKS:
        task_rows_valid = False
        issue(
            "validation_task_count_mismatch",
            "tasks",
            f"observed {len(tasks)}; expected {EXPECTED_TASKS}",
        )

    validation_output = outputs.get("validation")
    if not isinstance(validation_output, dict):
        issue(
            "selection_validation_output_invalid",
            "selection_manifest.outputs.validation",
            "missing validation output binding",
        )
    else:
        if (
            validation_output.get("records") != EXPECTED_TASKS
            or validation_output.get("sha256") != tasks_sha256
            or task_ids.get("validation") != observed_task_ids
        ):
            issue(
                "selection_validation_output_invalid",
                "selection_manifest.outputs.validation",
                "validation records, digest, or ordered task ids do not match",
            )
        if tasks_path is not None:
            declared_path = validation_output.get("path")
            if (
                not isinstance(declared_path, str)
                or Path(declared_path).resolve() != tasks_path.resolve()
            ):
                issue(
                    "selection_validation_output_invalid",
                    "selection_manifest.outputs.validation.path",
                    "declared validation path differs from the audited task file",
                )

    expected_generation = {
        "schema_version": PENDING_MANIFEST_SCHEMA,
        "status": "generated_pending_counterfactual_validation",
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "student_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
        "protocol_runtime_content_tree_sha256": EXPECTED_RUNTIME_TREE_SHA256,
        "adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
        "tasks_sha256": tasks_sha256,
        "tasks": EXPECTED_TASKS,
        "group_size": EXPECTED_GROUP_SIZE,
        "trajectories": EXPECTED_TRAJECTORIES,
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
        "temperature": 0.8,
        "top_p": 1.0,
        "max_steps": 30,
        "max_new_tokens": 2048,
        "max_context_tokens": 16384,
        "history_turns": 4,
        "enable_thinking": True,
        "seed": VALIDATION_SEED,
        "generation_seed_scheme": "sha256-task-sample-turn-v1",
        "denotation_comparison": "bird-set",
        "trajectories_sha256": trajectories_sha256,
    }
    for key, expected in expected_generation.items():
        if generation_manifest.get(key) != expected:
            issue(
                "generation_manifest_contract_mismatch",
                f"generation_manifest.{key}",
                f"observed {generation_manifest.get(key)!r}; expected {expected!r}",
            )
    if generation_manifest.get("seed") == SCREEN_SEED:
        issue(
            "validation_seed_not_independent",
            "generation_manifest.seed",
            "boundary validation must not reuse the screening/probe seed",
        )
    if tasks_path is not None:
        declared_tasks_path = generation_manifest.get("tasks_path")
        if (
            not isinstance(declared_tasks_path, str)
            or Path(declared_tasks_path).resolve() != tasks_path.resolve()
        ):
            issue(
                "generation_manifest_contract_mismatch",
                "generation_manifest.tasks_path",
                "generator was not run on the byte-bound validation32 file",
            )

    base = audit_base_probe(
        generation_manifest,
        trajectories,
        trajectories_sha256=trajectories_sha256,
    )
    if not base["status"]["passes"]:
        issue(
            "base_rollout_contract_failed",
            "base_audit",
            json.dumps(base.get("issue_counts") or {}, sort_keys=True),
        )

    # Bind every trajectory to the selected validation row and its exact K8
    # position.  The generic probe gate intentionally does not know a task file.
    trajectory_binding_valid = task_rows_valid and len(trajectories) == EXPECTED_TRAJECTORIES
    tokenization_warnings = 0
    if trajectory_binding_valid:
        for position, row in enumerate(trajectories):
            task = tasks[position // EXPECTED_GROUP_SIZE]
            expected_id = observed_task_ids[position // EXPECTED_GROUP_SIZE]
            expected_sample = position % EXPECTED_GROUP_SIZE
            environment = row.get("environment")
            sample = row.get("sample")
            record = sample.get("audit_record") if isinstance(sample, dict) else None
            expected_environment = {
                "dataset_split": "train",
                "example_index": task.get("example_index"),
                "task_id": expected_id,
                "db_id": task.get("db_id"),
                "db_path": task.get("db_path"),
                "question": task.get("question"),
                "gold_sql": task.get("gold_sql") or task.get("query"),
                "external_knowledge": task.get("external_knowledge"),
            }
            valid = (
                row.get("schema_version") == TRAJECTORY_SCHEMA
                and row.get("sequence") == position
                and isinstance(environment, dict)
                and all(environment.get(key) == value for key, value in expected_environment.items())
                and isinstance(record, dict)
                and record.get("example_index") == task.get("example_index")
                and record.get("sample_index") == expected_sample
                and record.get("trajectory_id")
                == f"rl_{task.get('example_index')}_sample_{expected_sample}"
            )
            if not valid:
                trajectory_binding_valid = False
                issue(
                    "trajectory_task_binding_invalid",
                    f"trajectories[{position}]",
                    "row does not match its frozen validation task/sample position",
                )
            if isinstance(record, dict) and record.get("rollout_tokenization_warning") is True:
                tokenization_warnings += 1
    elif len(trajectories) != EXPECTED_TRAJECTORIES:
        issue(
            "trajectory_task_binding_invalid",
            "trajectories",
            f"observed {len(trajectories)} rows; expected {EXPECTED_TRAJECTORIES}",
        )

    no_runtime_contamination = (
        base["observed"].get("eligible_trajectories") == EXPECTED_TRAJECTORIES
        and not base["observed"].get("forbidden_counts")
        and tokenization_warnings == 0
    )
    if not no_runtime_contamination:
        issue(
            "runtime_contamination",
            "trajectories",
            (
                f"eligible={base['observed'].get('eligible_trajectories')}/"
                f"{EXPECTED_TRAJECTORIES}, forbidden="
                f"{base['observed'].get('forbidden_counts')}, "
                f"tokenization_warnings={tokenization_warnings}"
            ),
        )

    mixed_groups = int(base["observed"].get("mixed_outcome_groups") or 0)
    checks = {
        "explicit_selection_manifest_digest": (
            _valid_sha256(expected_selection_manifest_sha256)
            and selection_manifest_sha256 == expected_selection_manifest_sha256
        ),
        "frozen_selection_contract": not any(
            code.startswith("selection_") for code in issue_counts
        ),
        "validation_tasks_byte_bound": (
            task_rows_valid
            and isinstance(validation_output, dict)
            and validation_output.get("sha256") == tasks_sha256
            and task_ids.get("validation") == observed_task_ids
        ),
        "independent_validation_seed": generation_manifest.get("seed") == VALIDATION_SEED,
        "initial_sft1_policy": (
            generation_manifest.get("adapter_sha256")
            == EXPECTED_INITIAL_ADAPTER_SHA256
        ),
        "generation_manifest_contract": not any(
            code == "generation_manifest_contract_mismatch" for code in issue_counts
        ),
        "generic_k8_binary_probe_contract": base["status"]["passes"],
        "exact_trajectory_task_binding": trajectory_binding_valid,
        "no_runtime_contamination": no_runtime_contamination,
        "at_least_20_mixed_outcome_groups": mixed_groups >= MIN_MIXED_GROUPS,
    }
    passes = all(checks.values()) and not issues
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "selection_schema": SELECTION_SCHEMA,
            "selection_seed": EXPECTED_SELECTION_SEED,
            "validation_seed": VALIDATION_SEED,
            "screen_seed_must_differ": SCREEN_SEED,
            "policy": "fresh initial-SFT1",
            "initial_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
            "protocol_version": EXPECTED_PROTOCOL_VERSION,
            "protocol_hash": EXPECTED_PROTOCOL_HASH,
            "tasks": EXPECTED_TASKS,
            "group_size": EXPECTED_GROUP_SIZE,
            "trajectories": EXPECTED_TRAJECTORIES,
            "minimum_mixed_groups": MIN_MIXED_GROUPS,
            "runtime_contamination_allowed": False,
            "optimizer_updates": 0,
        },
        "inputs": {
            "selection_manifest": str(selection_manifest_path.resolve())
            if selection_manifest_path is not None
            else None,
            "selection_manifest_sha256": selection_manifest_sha256,
            "expected_selection_manifest_sha256": expected_selection_manifest_sha256,
            "generation_manifest": str(generation_manifest_path.resolve())
            if generation_manifest_path is not None
            else None,
            "generation_manifest_sha256": generation_manifest_sha256,
            "tasks": str(tasks_path.resolve()) if tasks_path is not None else None,
            "tasks_sha256": tasks_sha256,
            "trajectories": str(trajectories_path.resolve())
            if trajectories_path is not None
            else None,
            "trajectories_sha256": trajectories_sha256,
        },
        "observed": {
            **base["observed"],
            "tokenization_warnings": tokenization_warnings,
        },
        "checks": checks,
        "groups": base["groups"],
        "base_audit": {
            "schema_version": base["schema_version"],
            "checks": base["checks"],
            "issue_counts": base["issue_counts"],
            "status": base["status"],
        },
        "issues": issues,
        "issue_counts": dict(sorted(issue_counts.items())),
        "status": {"passes": passes, "validation_admitted": passes},
    }


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".next")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-selection-manifest-sha256",
        required=True,
        help="explicit frozen digest printed by boundary-screen finalize",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.output.exists() and not args.overwrite:
        print(f"output exists; pass --overwrite: {args.output}", file=sys.stderr)
        return 2
    if args.output.exists():
        try:
            args.output.unlink()
        except OSError as exc:
            print(f"cannot revoke stale output {args.output}: {exc}", file=sys.stderr)
            return 2

    try:
        selection_bytes = args.selection_manifest.read_bytes()
        generation_bytes = args.manifest.read_bytes()
        tasks_bytes = args.tasks.read_bytes()
        trajectories_bytes = args.trajectories.read_bytes()
        result = audit_boundary_validation(
            selection_manifest=_parse_json(
                selection_bytes, path=args.selection_manifest
            ),
            generation_manifest=_parse_json(generation_bytes, path=args.manifest),
            tasks=_parse_jsonl(tasks_bytes, path=args.tasks),
            trajectories=_parse_jsonl(
                trajectories_bytes, path=args.trajectories
            ),
            expected_selection_manifest_sha256=(
                args.expected_selection_manifest_sha256
            ),
            selection_manifest_sha256=_sha256(selection_bytes),
            generation_manifest_sha256=_sha256(generation_bytes),
            tasks_sha256=_sha256(tasks_bytes),
            trajectories_sha256=_sha256(trajectories_bytes),
            selection_manifest_path=args.selection_manifest,
            generation_manifest_path=args.manifest,
            tasks_path=args.tasks,
            trajectories_path=args.trajectories,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"boundary validation input error: {exc}", file=sys.stderr)
        return 2

    if not result["status"]["passes"]:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "checks": result["checks"],
                    "issue_counts": result["issue_counts"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    _write_atomic(args.output, result)
    print(json.dumps({"output": str(args.output.resolve()), **result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
