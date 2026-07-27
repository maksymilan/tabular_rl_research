#!/usr/bin/env python3
"""Partition failed student rollouts without waiting for success replay.

This is the latency-sensitive first half of ``prepare_batch2_rollout_lanes``.  It deliberately
does not accept or export any successful student trajectory.  Failed tasks are split into:

* real, state-preserving error candidates for external anchor selection; and
* immediate teacher-from-scratch tasks when no eligible error anchor exists.

Selected recovery prefixes are replayed later by ``generate_recovery_teacher_rollouts`` before
the external teacher is called.  Successful student trajectories remain subject to the full
deterministic replay and quality gates in ``prepare_batch2_rollout_lanes``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_bird_sft2_dataset import preferred_failed_sample
from prepare_batch2_rollout_lanes import (
    choose_success_sample,
    read_jsonl,
    recovery_package,
    sha256_file,
    task_id,
    task_map,
    write_jsonl_atomic,
)


def build(
    tasks_path: Path,
    rollout_paths: list[Path],
    recovery_candidates_out: Path,
    scratch_tasks_out: Path,
    manifest_path: Path,
    *,
    max_recovery_anchor_action: int,
) -> dict[str, Any]:
    tasks = read_jsonl(tasks_path)
    by_index = task_map(tasks)
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in rollout_paths:
        records.extend((path, row) for row in read_jsonl(path))
    if len(records) != len(tasks):
        raise ValueError(f"rollout records {len(records)} != tasks {len(tasks)}")

    seen_indices: set[int] = set()
    successful_ids: set[str] = set()
    recovery: list[dict[str, Any]] = []
    scratch: list[dict[str, Any]] = []
    recovery_error_types: Counter[str] = Counter()

    for source_path, record in records:
        index = int(record["example_index"])
        if index in seen_indices:
            raise ValueError(f"duplicate rollout example_index {index}")
        seen_indices.add(index)
        if index not in by_index:
            raise ValueError(f"rollout example_index {index} is outside the task cohort")
        task = by_index[index]
        if choose_success_sample(record) is not None:
            successful_ids.add(task_id(task))
            continue

        failed_sample = preferred_failed_sample(record)
        package = recovery_package(
            task,
            failed_sample,
            rollout_source=source_path,
            max_anchor_action=max_recovery_anchor_action,
        )
        if package is None:
            scratch.append(task)
            continue
        recovery.append(package)
        for candidate in package["candidates"]:
            recovery_error_types[candidate["last_tool_error"]["error"]["type"]] += 1

    missing = set(by_index) - seen_indices
    if missing:
        raise ValueError(f"missing rollout records for task indices: {sorted(missing)[:5]}")
    recovery_ids = {package["task"]["example_id"] for package in recovery}
    scratch_ids = {task_id(task) for task in scratch}
    if successful_ids & recovery_ids or successful_ids & scratch_ids or recovery_ids & scratch_ids:
        raise RuntimeError("fast failure lanes overlap")
    expected_ids = {task_id(task) for task in tasks}
    if successful_ids | recovery_ids | scratch_ids != expected_ids:
        raise RuntimeError("fast failure lanes do not exactly cover the rollout cohort")

    write_jsonl_atomic(recovery_candidates_out, recovery)
    write_jsonl_atomic(scratch_tasks_out, scratch)
    manifest = {
        "method": "batch2_failed_student_fast_partition",
        "tasks": {
            "path": str(tasks_path),
            "sha256": sha256_file(tasks_path),
            "records": len(tasks),
        },
        "rollout_inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in rollout_paths
        ],
        "coverage": {
            "raw_student_success_pending_replay": len(successful_ids),
            "failed_tasks": len(recovery) + len(scratch),
            "pending_recovery_evaluator": len(recovery),
            "teacher_from_scratch_immediate": len(scratch),
            "total": len(successful_ids | recovery_ids | scratch_ids),
            "lane_overlap": 0,
        },
        "recovery_candidates": {
            "tasks": len(recovery),
            "anchors": sum(len(row["candidates"]) for row in recovery),
            "error_types": dict(sorted(recovery_error_types.items())),
        },
        "safety": {
            "successful_student_trajectories_exported": False,
            "success_replay_runs_in_parallel": True,
            "selected_recovery_prefix_replayed_before_teacher": True,
            "gold_sql_visible_to_evaluator": False,
            "rejected_actions_are_sft_targets": False,
        },
        "outputs": {
            "recovery_candidates": str(recovery_candidates_out),
            "teacher_from_scratch_immediate": str(scratch_tasks_out),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--rollout-all", type=Path, action="append", required=True)
    parser.add_argument("--recovery-candidates-out", type=Path, required=True)
    parser.add_argument("--scratch-tasks-out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-recovery-anchor-action", type=int, default=24)
    args = parser.parse_args()
    manifest = build(
        args.tasks.resolve(),
        [path.resolve() for path in args.rollout_all],
        args.recovery_candidates_out.resolve(),
        args.scratch_tasks_out.resolve(),
        args.manifest.resolve(),
        max_recovery_anchor_action=args.max_recovery_anchor_action,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
