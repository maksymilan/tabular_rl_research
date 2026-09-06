#!/usr/bin/env python3
"""Summarize K4 rollout groups and freeze a balanced mixed-outcome task set."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LEVELS = ("simple", "moderate", "challenging")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def task_id(row: dict[str, Any]) -> str:
    return str(row.get("example_id") or row.get("instance_id") or row.get("trajectory_id"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--groups-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--selected-tasks", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--required-per-level", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="freeze only after every candidate group is present",
    )
    args = parser.parse_args()

    tasks = load_jsonl(args.tasks)
    task_map = {task_id(row): row for row in tasks}
    distributions: dict[str, Counter[int]] = {level: Counter() for level in LEVELS}
    mixed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    malformed: list[str] = []
    complete = 0
    for identity, row in task_map.items():
        path = args.groups_dir / f"{identity}.json"
        if not path.exists():
            continue
        samples = json.loads(path.read_text(encoding="utf-8"))
        if len(samples) != args.group_size:
            malformed.append(identity)
            continue
        complete += 1
        level = str((row.get("metadata") or {}).get("fixed_pool_difficulty"))
        if level not in LEVELS:
            raise ValueError(f"unsupported difficulty for {identity}: {level}")
        correct = sum(bool(sample["sample"]["correct"]) for sample in samples)
        distributions[level][correct] += 1
        if 0 < correct < args.group_size:
            mixed[level].append(row)

    enough_mixed = all(len(mixed[level]) >= args.required_per_level for level in LEVELS)
    ready = enough_mixed and (not args.require_complete or complete == len(tasks))
    summary = {
        "schema_version": "mixed-k4-search-summary-v1",
        "status": "ready" if ready else "collecting",
        "candidate_tasks": len(tasks),
        "complete_groups": complete,
        "malformed_groups": malformed,
        "group_size": args.group_size,
        "required_per_level": args.required_per_level,
        "correct_count_distribution": {
            level: {str(k): v for k, v in sorted(distributions[level].items())}
            for level in LEVELS
        },
        "mixed_counts": {level: len(mixed[level]) for level in LEVELS},
        "remaining_needed": {
            level: max(0, args.required_per_level - len(mixed[level])) for level in LEVELS
        },
        "enough_mixed": enough_mixed,
        "require_complete": args.require_complete,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if not ready:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2

    selected = [row for level in LEVELS for row in mixed[level][: args.required_per_level]]
    with args.selected_tasks.open("w", encoding="utf-8") as target:
        for row in selected:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        **summary,
        "status": "frozen",
        "difficulty_counts": {
            "challenging": args.required_per_level,
            "moderate": args.required_per_level,
            "simple": args.required_per_level,
        },
        "selection_rule": "first deterministic candidate-order tasks with 1..K-1 correct samples",
        "selected_tasks": len(selected),
        "selected_task_ids": [task_id(row) for row in selected],
        "selected_tasks_sha256": sha256_file(args.selected_tasks),
    }
    args.selection_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
