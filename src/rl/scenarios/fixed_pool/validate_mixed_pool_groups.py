#!/usr/bin/env python3
"""Fail closed on completeness and sequence invariants of a mixed K4 search."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from rl.fixed_pool.io import load_jsonl, sha256_file, task_id
except ModuleNotFoundError:
    from fixed_pool.io import load_jsonl, sha256_file, task_id



identity = task_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--groups-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=4)
    args = parser.parse_args()

    tasks = load_jsonl(args.tasks)
    task_ids = [identity(row) for row in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("candidate task ids are not unique")
    expected = set(task_ids)
    actual = {path.stem for path in args.groups_dir.glob("*.json")}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"group coverage mismatch missing={missing[:5]} extra={extra[:5]}")

    correct_counts: Counter[int] = Counter()
    for task_index, task_id in enumerate(task_ids):
        path = args.groups_dir / f"{task_id}.json"
        samples = json.loads(path.read_text(encoding="utf-8"))
        if len(samples) != args.group_size:
            raise ValueError(f"{task_id} has {len(samples)} samples, expected {args.group_size}")
        for sample_index, row in enumerate(samples):
            expected_sequence = task_index * args.group_size + sample_index
            if int(row.get("sequence", -1)) != expected_sequence:
                raise ValueError(
                    f"{task_id} sample {sample_index} sequence mismatch: "
                    f"{row.get('sequence')} != {expected_sequence}"
                )
            environment = row.get("environment") or {}
            if str(environment.get("task_id")) != task_id:
                raise ValueError(f"{task_id} sample {sample_index} task_id mismatch")
            audit = (row.get("sample") or {}).get("audit_record") or {}
            if int(audit.get("sample_index", -1)) != sample_index:
                raise ValueError(f"{task_id} sample_index mismatch at {sample_index}")
        correct_counts[sum(bool(row["sample"]["correct"]) for row in samples)] += 1

    payload = {
        "schema_version": "mixed-k4-group-integrity-v1",
        "status": "passed",
        "tasks": len(tasks),
        "group_size": args.group_size,
        "trajectories": len(tasks) * args.group_size,
        "correct_count_distribution": {
            str(key): value for key, value in sorted(correct_counts.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
        raise FileExistsError(f"refusing to replace a different integrity audit: {args.output}")
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
