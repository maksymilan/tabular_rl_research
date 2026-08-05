#!/usr/bin/env python3
"""Validate and merge one completed mixed-pool shard without overwriting groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def task_id(row: dict[str, Any]) -> str:
    value = row.get("example_id") or row.get("instance_id") or row.get("trajectory_id")
    if not value:
        raise ValueError("task is missing a stable id")
    return str(value)


def write_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--task-id-file", required=True, type=Path)
    parser.add_argument("--source-groups", required=True, type=Path)
    parser.add_argument("--target-groups", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=4)
    args = parser.parse_args()

    tasks = load_jsonl(args.tasks)
    order = {task_id(row): index for index, row in enumerate(tasks)}
    assigned = [line.strip() for line in args.task_id_file.read_text().splitlines() if line.strip()]
    if len(set(assigned)) != len(assigned):
        raise ValueError("assigned task ids are not unique")
    unknown = set(assigned) - set(order)
    if unknown:
        raise ValueError(f"assigned task ids are absent from the candidate set: {sorted(unknown)}")
    unexpected = sorted(path.stem for path in args.source_groups.glob("*.json") if path.stem not in set(assigned))
    if unexpected:
        raise ValueError(f"source shard contains unassigned groups: {unexpected[:20]}")

    args.target_groups.mkdir(parents=True, exist_ok=True)
    copied = identical = 0
    for identity in assigned:
        source = args.source_groups / f"{identity}.json"
        if not source.exists():
            raise ValueError(f"completed shard is missing {source.name}")
        rows = json.loads(source.read_text(encoding="utf-8"))
        if len(rows) != args.group_size:
            raise ValueError(f"incomplete group: {source}")
        for sample_index, row in enumerate(rows):
            expected_sequence = order[identity] * args.group_size + sample_index
            if int(row["sequence"]) != expected_sequence:
                raise ValueError(f"bad sequence in {source}: expected {expected_sequence}")
            audit = row["sample"]["audit_record"]
            if int(audit["sample_index"]) != sample_index:
                raise ValueError(f"bad sample_index in {source}")
        target = args.target_groups / source.name
        if target.exists():
            if json.loads(target.read_text(encoding="utf-8")) != rows:
                raise ValueError(f"refusing to overwrite inconsistent group: {target}")
            identical += 1
        else:
            write_atomic(target, rows)
            copied += 1
    print(json.dumps({"assigned": len(assigned), "copied": copied, "identical": identical}))


if __name__ == "__main__":
    main()
