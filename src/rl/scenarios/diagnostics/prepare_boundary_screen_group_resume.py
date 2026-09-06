#!/usr/bin/env python3
"""Quarantine only known interrupted fixed-pool group writes.

The fixed-pool generator publishes ``TASK.json`` through ``TASK.json.next``.
An interruption after the temporary write must not make an otherwise resumable
worker permanently unusable. This helper moves only regular, non-symlink
``.json.next`` files for the exact assigned task identities. Unknown entries
fail before any move; completed ``.json`` groups are never modified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


def _task_ids(tasks: Path, task_id_file: Path | None) -> list[str]:
    rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
    all_ids = [str(row.get("example_id") or row.get("instance_id")) for row in rows]
    if not all_ids or "None" in all_ids or len(all_ids) != len(set(all_ids)):
        raise ValueError("tasks must contain unique nonempty task identities")
    if task_id_file is None:
        return all_ids
    assigned = [line for line in task_id_file.read_text().splitlines() if line]
    if not assigned or len(assigned) != len(set(assigned)):
        raise ValueError("task-id-file must contain unique nonempty identities")
    if not set(assigned).issubset(all_ids):
        raise ValueError("task-id-file contains identities absent from tasks")
    return assigned


def prepare_group_resume(
    *,
    tasks: Path,
    groups_dir: Path,
    quarantine_dir: Path,
    task_id_file: Path | None = None,
) -> dict[str, object]:
    for path, label in ((tasks, "tasks"), (task_id_file, "task-id-file")):
        if path is not None and (not path.is_file() or path.is_symlink()):
            raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if not groups_dir.is_dir() or groups_dir.is_symlink():
        raise ValueError(f"groups-dir must be a regular directory: {groups_dir}")
    if quarantine_dir.exists() and (
        not quarantine_dir.is_dir() or quarantine_dir.is_symlink()
    ):
        raise ValueError(
            f"quarantine-dir must be a regular directory: {quarantine_dir}"
        )

    ids = _task_ids(tasks, task_id_file)
    expected_final = {f"{task_id}.json": task_id for task_id in ids}
    expected_partial = {f"{task_id}.json.next": task_id for task_id in ids}
    partials: list[tuple[Path, str]] = []
    for entry in groups_dir.iterdir():
        if entry.name in expected_final:
            if not entry.is_file() or entry.is_symlink():
                raise ValueError(f"completed group is not a regular file: {entry}")
        elif entry.name in expected_partial:
            if not entry.is_file() or entry.is_symlink():
                raise ValueError(f"partial group is not a regular file: {entry}")
            partials.append((entry, expected_partial[entry.name]))
        else:
            raise ValueError(f"unknown group entry blocks resume: {entry}")

    destinations: list[Path] = []
    reserved: set[Path] = set()
    for source, _task_id in sorted(partials, key=lambda item: item[0].name):
        destination = quarantine_dir / source.name
        suffix = 1
        while destination.exists() or destination in reserved:
            destination = quarantine_dir / f"{source.name}.{suffix}"
            suffix += 1
        destinations.append(destination)
        reserved.add(destination)

    if partials:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for (source, task_id), destination in zip(
        sorted(partials, key=lambda item: item[0].name), destinations, strict=True
    ):
        os.replace(source, destination)
        moved.append(
            {
                "task_id": task_id,
                "from": str(source.resolve()),
                "to": str(destination.resolve()),
            }
        )

    remaining = list(groups_dir.iterdir())
    if any(
        entry.name not in expected_final
        or not entry.is_file()
        or entry.is_symlink()
        for entry in remaining
    ):
        raise RuntimeError("group resume preparation left a non-final entry")
    return {
        "status": "resume_ready",
        "assigned_tasks": len(ids),
        "completed_groups": len(remaining),
        "quarantined_partials": moved,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--task-id-file", type=Path)
    parser.add_argument("--groups-dir", type=Path, required=True)
    parser.add_argument("--quarantine-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare_group_resume(
            tasks=args.tasks,
            task_id_file=args.task_id_file,
            groups_dir=args.groups_dir,
            quarantine_dir=args.quarantine_dir,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"boundary group resume preparation blocked: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
