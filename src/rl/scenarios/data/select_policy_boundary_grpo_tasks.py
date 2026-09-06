#!/usr/bin/env python3
"""Freeze the preregistered policy-boundary cohort for vanilla binary GRPO."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "policy-boundary-grpo-cohort-v1"
EXPECTED_AUDIT_SCHEMA = "vanilla-grpo-boundary-screen-audit-v1"
EXPECTED_INITIAL_ADAPTER_SHA256 = (
    "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5"
)
EXPECTED_POOL_SIZE = 600
BOUNDARY_SIZE = 332
TRAIN_SIZE = 300
VALIDATION_SIZE = 32
MIN_MIXED_GROUPS = 360
MIN_CORE_GROUPS = 240
MIN_UNIQUE_DATABASES = 50
MAX_DATABASE_TV = 0.15
MAX_LENGTH_TV = 0.10
MAX_KNOWLEDGE_TV = 0.05
DEFAULT_SEED = "qwen3-v26-boundary332-v1-20260812"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_jsonl(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    data = path.read_bytes()
    rows = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return data, rows


def _task_id(row: Mapping[str, Any]) -> str:
    value = row.get("example_id") or row.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError("task is missing example_id/instance_id")
    return value


def _stable_rank(seed: str, namespace: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{task_id}".encode()).hexdigest()


def _quantile(values: Sequence[int], probability: float) -> int:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute quantile of an empty pool")
    return int(ordered[max(0, math.ceil(probability * len(ordered)) - 1)])


def _length_cutpoints(rows: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    values = [len(str(row.get("question") or "").strip()) for row in rows]
    if any(value < 1 for value in values):
        raise ValueError("every task must have a nonempty question")
    return tuple(_quantile(values, p) for p in (0.2, 0.4, 0.6, 0.8))


def _length_bin(row: Mapping[str, Any], cutpoints: Sequence[int]) -> str:
    length = len(str(row.get("question") or "").strip())
    for index, boundary in enumerate(cutpoints, start=1):
        if length <= boundary:
            return f"q{index}"
    return "q5"


def _knowledge(row: Mapping[str, Any]) -> str:
    value = row.get("external_knowledge")
    return "present" if isinstance(value, str) and value.strip() else "absent"


def _tv(reference: Sequence[str], sample: Sequence[str]) -> float:
    left = Counter(reference)
    right = Counter(sample)
    categories = set(left) | set(right)
    return 0.5 * sum(
        abs(left[key] / len(reference) - right[key] / len(sample))
        for key in categories
    )


def _load_pool(audit_path: Path, tasks_path: Path) -> dict[str, Any]:
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    if not isinstance(audit, dict) or audit.get("schema_version") != EXPECTED_AUDIT_SCHEMA:
        raise ValueError(f"unsupported boundary audit: {audit_path}")
    status = audit.get("status") or {}
    contract = audit.get("contract") or {}
    inputs = audit.get("inputs") or {}
    if (
        status.get("audit_passes") is not True
        or status.get("pool_admitted") is not True
        or audit.get("issues") != []
        or audit.get("issue_counts") != {}
    ):
        raise ValueError(f"screen audit is not admitted: {audit_path}")
    if (
        Path(str(inputs.get("tasks") or "")).resolve() != tasks_path.resolve()
        or not isinstance(inputs.get("manifest_sha256"), str)
        or len(str(inputs.get("manifest_sha256"))) != 64
        or not isinstance(inputs.get("trajectories_sha256"), str)
        or len(str(inputs.get("trajectories_sha256"))) != 64
    ):
        raise ValueError(f"screen audit source binding is incomplete: {audit_path}")
    if (
        contract.get("tasks") != EXPECTED_POOL_SIZE
        or contract.get("group_size") != 8
        or contract.get("optimizer_updates") != 0
        or contract.get("initial_adapter_sha256")
        != EXPECTED_INITIAL_ADAPTER_SHA256
        or contract.get("reward_mode") != "result-only"
        or contract.get("result_reward_profile") != "binary"
    ):
        raise ValueError(f"screen audit contract mismatch: {audit_path}")
    task_bytes, tasks = _read_jsonl(tasks_path)
    if len(tasks) != EXPECTED_POOL_SIZE or inputs.get("tasks_sha256") != _sha(task_bytes):
        raise ValueError(f"screen audit/tasks binding mismatch: {tasks_path}")
    indexed: dict[str, dict[str, Any]] = {}
    example_indices: set[int] = set()
    for row in tasks:
        task_id = _task_id(row)
        example_index = row.get("example_index")
        if task_id in indexed or type(example_index) is not int or example_index in example_indices:
            raise ValueError(f"duplicate/invalid task identity in {tasks_path}: {task_id}")
        indexed[task_id] = row
        example_indices.add(example_index)
    groups = audit.get("groups")
    if not isinstance(groups, list) or len(groups) != EXPECTED_POOL_SIZE:
        raise ValueError(f"audit must contain exactly 600 group summaries: {audit_path}")
    summaries: dict[str, dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("group summary must be an object")
        task_id = group.get("task_id")
        correct = group.get("correct_count")
        if (
            not isinstance(task_id, str)
            or task_id not in indexed
            or task_id in summaries
            or group.get("example_index") != indexed[task_id].get("example_index")
            or type(correct) is not int
            or not 0 <= correct <= 8
            or group.get("uncertainty")
            != (correct * (8 - correct) if 1 <= correct <= 7 else 0)
            or type(group.get("usable")) is not bool
            or type(group.get("mixed_boundary")) is not bool
            or type(group.get("core_boundary")) is not bool
            or group.get("mixed_boundary")
            != (group.get("usable") and 1 <= correct <= 7)
            or group.get("core_boundary")
            != (group.get("usable") and 2 <= correct <= 6)
            or not isinstance(group.get("contamination"), list)
            or bool(group.get("contamination")) == bool(group.get("usable"))
        ):
            raise ValueError(f"invalid group summary in {audit_path}: {task_id}")
        summaries[task_id] = group
    return {
        "audit_path": str(audit_path),
        "audit_sha256": _sha(audit_bytes),
        "tasks_path": str(tasks_path),
        "tasks_sha256": _sha(task_bytes),
        "tasks": indexed,
        "groups": summaries,
    }


def _distribution_balanced_select(
    candidates: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: str,
    cutpoints: Sequence[int],
    reference: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select by uncertainty first, then minimize public-axis TV within a bucket."""

    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} boundary tasks available; need {count}")
    features = (
        lambda row: str(row.get("db_id") or ""),
        lambda row: _length_bin(row, cutpoints),
        _knowledge,
    )
    reference_proportions: list[dict[str, float]] = []
    for feature in features:
        counts = Counter(feature(row) for row in reference)
        reference_proportions.append(
            {key: value / len(reference) for key, value in counts.items()}
        )
    selected_counts = [Counter() for _ in features]
    by_uncertainty: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_uncertainty.setdefault(int(candidate["uncertainty"]), []).append(candidate)

    selected: list[dict[str, Any]] = []
    for uncertainty in sorted(by_uncertainty, reverse=True):
        available = list(by_uncertainty[uncertainty])
        while available and len(selected) < count:
            next_size = len(selected) + 1

            def score(candidate: dict[str, Any]) -> tuple[float, str]:
                row = candidate["task"]
                objective = 0.0
                for axis, feature in enumerate(features):
                    candidate_value = feature(row)
                    categories = (
                        set(reference_proportions[axis])
                        | set(selected_counts[axis])
                        | {candidate_value}
                    )
                    objective += 0.5 * sum(
                        abs(
                            (
                                selected_counts[axis][category]
                                + int(category == candidate_value)
                            )
                            / next_size
                            - reference_proportions[axis].get(category, 0.0)
                        )
                        for category in categories
                    )
                return (
                    objective,
                    _stable_rank(seed, "public-distribution", candidate["task_id"]),
                )

            chosen = min(available, key=score)
            available.remove(chosen)
            selected.append(chosen)
            for axis, feature in enumerate(features):
                selected_counts[axis][feature(chosen["task"])] += 1
        if len(selected) == count:
            break
    return selected


def select_boundary_cohort(
    pools: Sequence[dict[str, Any]], *, seed: str = DEFAULT_SEED
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if len(pools) not in {1, 2}:
        raise ValueError("boundary selection accepts exactly S1 or S1+S2")
    all_tasks: dict[str, dict[str, Any]] = {}
    all_example_indices: dict[int, str] = {}
    candidates: list[dict[str, Any]] = []
    pool_counts: list[dict[str, int]] = []
    for pool_index, pool in enumerate(pools, start=1):
        pool_mixed = 0
        pool_core = 0
        for task_id, task in pool["tasks"].items():
            if task_id in all_tasks:
                raise ValueError(f"S1/S2 pools overlap at {task_id}")
            example_index = int(task["example_index"])
            if example_index in all_example_indices:
                raise ValueError(
                    "S1/S2 pools overlap at example_index "
                    f"{example_index}: {all_example_indices[example_index]} / {task_id}"
                )
            all_tasks[task_id] = task
            all_example_indices[example_index] = task_id
            group = pool["groups"][task_id]
            if group.get("usable") and group.get("mixed_boundary"):
                pool_mixed += 1
                pool_core += int(bool(group.get("core_boundary")))
                candidates.append(
                    {
                        "task_id": task_id,
                        "task": task,
                        "correct_count": int(group["correct_count"]),
                        "uncertainty": int(group["uncertainty"]),
                        "core_boundary": bool(group.get("core_boundary")),
                        "screen_pool": f"S{pool_index}",
                    }
                )
        pool_counts.append({"mixed": pool_mixed, "core": pool_core})
    s1_ready = (
        pool_counts[0]["mixed"] >= MIN_MIXED_GROUPS
        and pool_counts[0]["core"] >= MIN_CORE_GROUPS
    )
    if len(pools) == 2 and s1_ready:
        raise ValueError("S2 is forbidden because S1 already meets the frozen thresholds")
    mixed_count = len(candidates)
    core_count = sum(item["core_boundary"] for item in candidates)
    if mixed_count < MIN_MIXED_GROUPS or core_count < MIN_CORE_GROUPS:
        raise ValueError(
            f"boundary readiness failed: mixed={mixed_count}/{MIN_MIXED_GROUPS}, "
            f"core={core_count}/{MIN_CORE_GROUPS}"
        )
    # Distribution gates stay anchored to the preregistered representative S1
    # population even if S2 is needed only to add policy-boundary candidates.
    reference = list(pools[0]["tasks"].values())
    cutpoints = _length_cutpoints(reference)
    selected = _distribution_balanced_select(
        candidates,
        count=BOUNDARY_SIZE,
        seed=seed,
        cutpoints=cutpoints,
        reference=reference,
    )
    selected_tasks = [item["task"] for item in selected]
    database_tv = _tv(
        [str(row.get("db_id") or "") for row in reference],
        [str(row.get("db_id") or "") for row in selected_tasks],
    )
    length_tv = _tv(
        [_length_bin(row, cutpoints) for row in reference],
        [_length_bin(row, cutpoints) for row in selected_tasks],
    )
    knowledge_tv = _tv(
        [_knowledge(row) for row in reference],
        [_knowledge(row) for row in selected_tasks],
    )
    unique_databases = len({str(row.get("db_id") or "") for row in selected_tasks})
    gates = {
        "exact_boundary332": len(selected) == BOUNDARY_SIZE,
        "mixed_at_least_360": mixed_count >= MIN_MIXED_GROUPS,
        "core_at_least_240": core_count >= MIN_CORE_GROUPS,
        "unique_databases_at_least_50": unique_databases >= MIN_UNIQUE_DATABASES,
        "database_tv_at_most_0_15": database_tv <= MAX_DATABASE_TV,
        "length_tv_at_most_0_10": length_tv <= MAX_LENGTH_TV,
        "knowledge_tv_at_most_0_05": knowledge_tv <= MAX_KNOWLEDGE_TV,
    }
    if not all(gates.values()):
        raise ValueError(f"boundary cohort gates failed: {[k for k, v in gates.items() if not v]}")

    # The 332 identities are fixed before splitting.  A separate namespace makes
    # the validation allocation deterministic and independent of list position.
    validation_ids = {
        item["task_id"]
        for item in sorted(
            selected,
            key=lambda item: _stable_rank(seed, "validation-split", item["task_id"]),
        )[:VALIDATION_SIZE]
    }
    validation = [item for item in selected if item["task_id"] in validation_ids]
    train = [item for item in selected if item["task_id"] not in validation_ids]
    assert len(train) == TRAIN_SIZE and len(validation) == VALIDATION_SIZE
    details = {
        "seed": seed,
        "eligible_mixed_groups": mixed_count,
        "eligible_core_groups": core_count,
        "screen_stage": "S1" if len(pools) == 1 else "S1+S2",
        "pool_eligible_counts": pool_counts,
        "length_quintile_cutpoints_characters": list(cutpoints),
        "selection_rule": (
            "uncertainty c*(8-c): 4 > 3/5 > 2/6 > 1/7; deterministic public "
            "database + question-length + external-knowledge TV-deficit balancing "
            "within each uncertainty bucket; SHA tie-break"
        ),
        "split_rule": "fixed SHA-256 validation namespace over the selected 332 identities",
        "distribution_reference": "S1 representative600 public fields only",
        "distribution": {
            "unique_databases": unique_databases,
            "database_tv": database_tv,
            "length_tv": length_tv,
            "knowledge_tv": knowledge_tv,
        },
        "acceptance_gates": gates,
        "selected_metadata": [
            {
                key: item[key]
                for key in (
                    "task_id",
                    "correct_count",
                    "uncertainty",
                    "core_boundary",
                    "screen_pool",
                )
            }
            for item in selected
        ],
    }
    return selected, train, validation, details


def _jsonl_bytes(items: Sequence[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(item["task"], ensure_ascii=False, separators=(",", ":")) + "\n"
        for item in items
    ).encode()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".next")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-audit", type=Path, action="append", required=True)
    parser.add_argument("--tasks", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if len(args.screen_audit) != len(args.tasks) or len(args.tasks) not in {1, 2}:
        parser.error("pass one tasks file per screen audit, for S1 or S1+S2")
    output_paths = {
        "boundary": args.output_dir / "boundary332.jsonl",
        "train": args.output_dir / "train300.jsonl",
        "validation": args.output_dir / "validation32.jsonl",
        "manifest": args.output_dir / "boundary_cohort_manifest.json",
    }
    if any(path.exists() for path in output_paths.values()):
        print("refusing to overwrite an existing boundary selection", file=sys.stderr)
        return 2
    try:
        pools = [
            _load_pool(audit, tasks)
            for audit, tasks in zip(args.screen_audit, args.tasks, strict=True)
        ]
        all_selected, train, validation, details = select_boundary_cohort(
            pools, seed=args.seed
        )
        boundary_bytes = _jsonl_bytes(all_selected)
        train_bytes = _jsonl_bytes(train)
        validation_bytes = _jsonl_bytes(validation)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "frozen_boundary_training_cohort",
            "purpose": "policy-specific boundary cohort for vanilla binary GRPO",
            "screen_pools": [
                {key: pool[key] for key in ("audit_path", "audit_sha256", "tasks_path", "tasks_sha256")}
                for pool in pools
            ],
            "contract": {
                "boundary_records": BOUNDARY_SIZE,
                "train_records": TRAIN_SIZE,
                "validation_records": VALIDATION_SIZE,
                "formal_training": {
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
                },
                "validation": {
                    "policy": "fresh initial-SFT1",
                    "records": 32,
                    "group_size": 8,
                    "seed": 20260813,
                    "screen_seed_must_differ": 20260812,
                    "generation_seed_scheme": "sha256-task-sample-turn-v1",
                    "gate": "same >=20/32 mixed probe gate",
                    "runtime_contamination_allowed": False,
                    "screen_trajectories_reused": False,
                },
                "primary_checkpoint": "final-step20-only",
            },
            "selection": details,
            "outputs": {
                "boundary": {"path": str(output_paths["boundary"]), "sha256": _sha(boundary_bytes), "records": BOUNDARY_SIZE},
                "train": {"path": str(output_paths["train"]), "sha256": _sha(train_bytes), "records": TRAIN_SIZE},
                "validation": {"path": str(output_paths["validation"]), "sha256": _sha(validation_bytes), "records": VALIDATION_SIZE},
            },
            "task_ids": {
                "boundary": [item["task_id"] for item in all_selected],
                "train": [item["task_id"] for item in train],
                "validation": [item["task_id"] for item in validation],
            },
        }
        manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
        _write(output_paths["boundary"], boundary_bytes)
        _write(output_paths["train"], train_bytes)
        _write(output_paths["validation"], validation_bytes)
        _write(output_paths["manifest"], manifest_bytes)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"boundary selection failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "train": 300, "validation": 32}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
