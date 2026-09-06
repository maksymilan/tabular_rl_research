#!/usr/bin/env python3
"""Freeze disjoint continuation assignments from complete atomic K8 groups."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "qwen3-v26-boundary-screen-continuation-plan-v1"
EXPECTED_TASKS_SHA256 = (
    "b5a83c373e9be094ea7355c0bc23c9212249491491f44dfdcb3b09ea49b2457e"
)
GROUP_SIZE = 8
EXPECTED_RECORDS = 600


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"tasks must be a regular file: {path}")
    if _sha(path) != EXPECTED_TASKS_SHA256:
        raise ValueError("train600 task SHA-256 differs from the frozen S1 cohort")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids: list[str] = []
    for row in rows:
        task_id = row.get("example_id") or row.get("instance_id")
        index = row.get("example_index")
        if (
            not isinstance(task_id, str)
            or type(index) is not int
            or task_id != f"bird_train_{index:05d}"
            or row.get("instance_id") != task_id
        ):
            raise ValueError(f"invalid task/example identity: {task_id!r}")
        ids.append(task_id)
    if len(ids) != len(set(ids)) or len(ids) != EXPECTED_RECORDS:
        raise ValueError(
            f"tasks must contain exactly {EXPECTED_RECORDS} unique identities"
        )
    return rows


def _complete_groups(
    group_dirs: Sequence[Path], tasks: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    order = {
        str(row["example_id"]): (position, int(row["example_index"]))
        for position, row in enumerate(tasks)
    }
    complete: dict[str, dict[str, Any]] = {}
    for directory in group_dirs:
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"group directory must be regular: {directory}")
        unknown = [path for path in directory.iterdir() if path.suffix != ".json"]
        if unknown:
            raise ValueError(f"group directory contains non-JSON entries: {unknown[:3]}")
        for path in sorted(directory.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"group entry must be a regular file: {path}")
            task_id = path.stem
            if task_id not in order or task_id in complete:
                raise ValueError(f"unknown/duplicate completed task: {task_id}")
            position, example_index = order[task_id]
            rows = json.loads(path.read_text())
            if not isinstance(rows, list) or len(rows) != GROUP_SIZE:
                raise ValueError(f"completed group is not K8: {path}")
            for sample_index, row in enumerate(rows):
                audit = ((row.get("sample") or {}).get("audit_record") or {})
                environment = row.get("environment") or {}
                if (
                    row.get("sequence") != position * GROUP_SIZE + sample_index
                    or environment.get("task_id") != task_id
                    or audit.get("example_index") != example_index
                    or audit.get("sample_index") != sample_index
                ):
                    raise ValueError(f"completed group identity/sequence mismatch: {path}")
            complete[task_id] = {
                "path": str(path.resolve()),
                "sha256": _sha(path),
                "rows": GROUP_SIZE,
            }
    return complete


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".next")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def prepare(
    *,
    tasks_path: Path,
    group_dirs: Sequence[Path],
    output_dir: Path,
    shards: int,
    restricted_db_ids: Sequence[str] = (),
    restricted_shards: int = 0,
) -> dict[str, Any]:
    if shards < 1:
        raise ValueError("shards must be positive")
    restricted_db_ids = tuple(sorted(set(restricted_db_ids)))
    if restricted_db_ids and not 0 < restricted_shards <= shards:
        raise ValueError("restricted DBs require 1..shards capable prefix shards")
    if not restricted_db_ids and restricted_shards != 0:
        raise ValueError("restricted_shards requires at least one restricted DB")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    tasks = _read_tasks(tasks_path)
    complete = _complete_groups(group_dirs, tasks)
    ordered_ids = [str(row["example_id"]) for row in tasks]
    remaining = [task_id for task_id in ordered_ids if task_id not in complete]
    if not remaining:
        raise ValueError("no continuation tasks remain")
    db_by_task = {
        str(row["example_id"]): str(row.get("db_id") or "") for row in tasks
    }
    restricted = [
        task_id for task_id in remaining
        if db_by_task[task_id] in set(restricted_db_ids)
    ]
    general = [task_id for task_id in remaining if task_id not in set(restricted)]
    assignments = [general[index::shards] for index in range(shards)]
    for index, task_id in enumerate(restricted):
        assignments[index % restricted_shards].append(task_id)
    if set().union(*map(set, assignments)) != set(remaining):
        raise RuntimeError("continuation assignment coverage mismatch")
    if sum(map(len, assignments)) != len(remaining):
        raise RuntimeError("continuation assignments overlap")

    assignment_records = []
    payloads: list[tuple[Path, bytes]] = []
    for shard, ids in enumerate(assignments):
        path = output_dir / f"remaining.shard-{shard:02d}-of-{shards:02d}.txt"
        payload = "".join(f"{task_id}\n" for task_id in ids).encode()
        payloads.append((path, payload))
        assignment_records.append(
            {
                "shard": shard,
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "tasks": len(ids),
                "task_ids": ids,
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_continuation_plan",
        "tasks": {
            "path": str(tasks_path.resolve()),
            "sha256": EXPECTED_TASKS_SHA256,
            "records": EXPECTED_RECORDS,
        },
        "group_size": GROUP_SIZE,
        "generation_contract": {
            "initial_policy": "qwen3-8b-atomic-v26-sft1-checkpoint560",
            "optimizer_updates": 0,
            "temperature": 0.8,
            "top_p": 1.0,
            "seed": 20260812,
            "max_new_tokens": 2048,
            "max_context_tokens": 16384,
            "enable_thinking": True,
        },
        "completed": {
            "tasks": len(complete),
            "trajectories": len(complete) * GROUP_SIZE,
            "groups": complete,
        },
        "remaining": {
            "tasks": len(remaining),
            "trajectories": len(remaining) * GROUP_SIZE,
        },
        "host_capability_partition": {
            "restricted_db_ids": list(restricted_db_ids),
            "restricted_tasks": restricted,
            "capable_prefix_shards": restricted_shards,
            "rule": (
                "restricted DB tasks are assigned only to prefix shards; "
                "all other tasks remain deterministic round-robin"
            ),
        },
        "assignments": assignment_records,
        "acceptance_gates": {
            "completed_and_remaining_cover_600": (
                len(complete) + len(remaining) == EXPECTED_RECORDS
            ),
            "assignments_cover_remaining_once": sum(map(len, assignments)) == len(remaining),
            "first32_already_complete": set(ordered_ids[:32]) <= set(complete),
        },
    }
    if not all(manifest["acceptance_gates"].values()):
        raise ValueError(f"continuation plan gates failed: {manifest['acceptance_gates']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in payloads:
        _atomic(path, payload)
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    _atomic(output_dir / "continuation_plan.json", encoded)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--group-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--restricted-db-id", action="append", default=[])
    parser.add_argument("--restricted-shards", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        manifest = prepare(
            tasks_path=args.tasks,
            group_dirs=args.group_dir,
            output_dir=args.output_dir,
            shards=args.shards,
            restricted_db_ids=args.restricted_db_id,
            restricted_shards=args.restricted_shards,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"continuation preparation blocked: {exc}", file=__import__("sys").stderr)
        return 1
    print(json.dumps({"completed": manifest["completed"]["tasks"], "remaining": manifest["remaining"]["tasks"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
