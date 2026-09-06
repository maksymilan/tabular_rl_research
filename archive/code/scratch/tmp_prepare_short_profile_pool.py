#!/usr/bin/env python3
"""Prepare an isolated diagnostic manifest/tasks file for a fresh K8 pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--tasks-source", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=8)
    args = parser.parse_args()
    pool = args.pool.resolve()
    trajectory_path = pool / "trajectories.jsonl"
    pending_path = pool / "manifest.pending.json"
    if not trajectory_path.is_file() or not pending_path.is_file():
        raise SystemExit("short pool is incomplete")
    rows = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_rows = args.tasks * args.group_size
    if len(rows) != expected_rows or [int(row["sequence"]) for row in rows] != list(range(expected_rows)):
        raise SystemExit(f"expected exactly {expected_rows} contiguous trajectories, got {len(rows)}")
    task_lines = [line for line in args.tasks_source.read_text(encoding="utf-8").splitlines() if line.strip()][:args.tasks]
    if len(task_lines) != args.tasks:
        raise SystemExit(f"tasks source does not contain {args.tasks} tasks")
    tasks_path = pool / f"tasks{args.tasks}.jsonl"
    tasks_path.write_text("\n".join(task_lines) + "\n", encoding="utf-8")
    manifest = json.loads(pending_path.read_text(encoding="utf-8"))
    pool_sha = sha256(trajectory_path)
    manifest.update(
        {
            "schema_version": "table-agent-diagnostic-rollout-pool-v1",
            "status": "diagnostic_generated",
            "result_reward_profile": "four-level",
            "tasks_path": str(tasks_path),
            "tasks_sha256": sha256(tasks_path),
            "tasks": args.tasks,
            "group_size": args.group_size,
            "trajectories": expected_rows,
            "trajectories_sha256": pool_sha,
            "correct_trajectories": sum(bool(row["sample"]["correct"]) for row in rows),
            "files": {"validated_trajectories": {"path": str(trajectory_path), "sha256": pool_sha}},
            "generated_at_epoch": time.time(),
            "reuse_contract": {
                "same_trajectories": True,
                "same_sequence": True,
                "online_resampling_forbidden": True,
            },
            "diagnostic_note": "Fresh two-task K8 pool with one causal model turn per trajectory; isolated profiling only.",
        }
    )
    (pool / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    json.loads((pool / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({"pool": str(pool), "tasks": str(tasks_path), "rows": len(rows), "sha256": pool_sha}))


if __name__ == "__main__":
    main()
