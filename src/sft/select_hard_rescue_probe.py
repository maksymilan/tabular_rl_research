#!/usr/bin/env python3
"""Select a reproducible task probe from a rollout failure bucket."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--rollout-all", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--failure-type", default="protocol_error")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    failures = [
        row
        for row in read_jsonl(args.rollout_all)
        if row.get("difficulty") == args.difficulty
        and row.get("failure_type") == args.failure_type
    ]
    failure_ids = {str(row["trajectory_id"]) for row in failures}
    tasks_by_id = {
        str(row.get("example_id") or row.get("instance_id")): row
        for row in read_jsonl(args.tasks)
    }
    missing = sorted(failure_ids - tasks_by_id.keys())
    if missing:
        raise ValueError(f"{len(missing)} failure ids are absent from task input")

    candidate_ids = sorted(failure_ids)
    random.Random(args.seed).shuffle(candidate_ids)
    selected_ids = candidate_ids[: args.limit]
    if len(selected_ids) < args.limit:
        raise ValueError(f"requested {args.limit} tasks, only {len(selected_ids)} candidates")
    selected = [tasks_by_id[task_id] for task_id in selected_ids]
    write_jsonl(args.output, selected)

    manifest = {
        "tasks": str(args.tasks.resolve()),
        "rollout_all": str(args.rollout_all.resolve()),
        "output": str(args.output.resolve()),
        "selection": {
            "difficulty": args.difficulty,
            "failure_type": args.failure_type,
            "seed": args.seed,
            "candidate_count": len(candidate_ids),
            "selected_count": len(selected),
        },
        "selected_ids": selected_ids,
        "database_count": len({row.get("db_id") for row in selected}),
        "database_histogram": dict(sorted(Counter(row.get("db_id") for row in selected).items())),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
