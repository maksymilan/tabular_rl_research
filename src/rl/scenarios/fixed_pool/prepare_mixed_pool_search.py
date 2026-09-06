#!/usr/bin/env python3
"""Prepare disjoint rollout shards for a larger mixed-outcome K4 search.

Existing groups may be reused only when their task id occurs in the new candidate
set.  ``sequence`` is pool-local metadata, so it is rewritten to the new immutable
task order; model prompts, responses, token ids, log probabilities, and outcomes
are copied without modification.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LEVELS = ("simple", "moderate", "challenging")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def task_id(row: dict[str, Any]) -> str:
    value = row.get("example_id") or row.get("instance_id") or row.get("trajectory_id")
    if not value:
        raise ValueError("task record is missing a stable id")
    return str(value)


def difficulty(row: dict[str, Any]) -> str:
    value = str((row.get("metadata") or {}).get("fixed_pool_difficulty") or "")
    if value not in LEVELS:
        raise ValueError(f"{task_id(row)} has unsupported fixed-pool difficulty {value!r}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reuse-groups", type=Path)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--shards", type=int, default=3)
    args = parser.parse_args()

    if args.group_size < 1 or args.shards < 1:
        raise ValueError("group size and shard count must be positive")
    tasks = load_jsonl(args.tasks)
    order = {task_id(row): index for index, row in enumerate(tasks)}
    if len(order) != len(tasks):
        raise ValueError("candidate task ids must be unique")
    counts = Counter(difficulty(row) for row in tasks)

    groups_dir = args.output_dir / "groups"
    shards_dir = args.output_dir / "shards"
    groups_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    reused: list[str] = []
    if args.reuse_groups:
        for source in sorted(args.reuse_groups.glob("*.json")):
            source_id = source.stem
            if source_id not in order:
                continue
            rows = json.loads(source.read_text(encoding="utf-8"))
            if len(rows) != args.group_size:
                raise ValueError(f"incomplete reusable group: {source}")
            rewritten = copy.deepcopy(rows)
            for sample_index, row in enumerate(rewritten):
                row["sequence"] = order[source_id] * args.group_size + sample_index
                audit = row["sample"]["audit_record"]
                audit["sample_index"] = sample_index
            target = groups_dir / source.name
            if target.exists():
                existing = json.loads(target.read_text(encoding="utf-8"))
                if existing != rewritten:
                    raise ValueError(f"refusing to overwrite inconsistent group: {target}")
            else:
                atomic_json(target, rewritten)
            reused.append(source_id)

    reused_set = set(reused)
    pending_by_level: dict[str, list[str]] = defaultdict(list)
    for row in tasks:
        identity = task_id(row)
        if identity not in reused_set:
            pending_by_level[difficulty(row)].append(identity)

    shards: list[list[str]] = [[] for _ in range(args.shards)]
    cursor = 0
    for level in LEVELS:
        for identity in pending_by_level[level]:
            shards[cursor % args.shards].append(identity)
            cursor += 1
    for index, identities in enumerate(shards):
        path = shards_dir / f"shard-{index:02d}.ids"
        path.write_text("".join(f"{identity}\n" for identity in identities), encoding="utf-8")

    manifest = {
        "schema_version": "mixed-k4-search-preparation-v1",
        "tasks": len(tasks),
        "difficulty_counts": dict(sorted(counts.items())),
        "tasks_sha256": sha256_file(args.tasks),
        "group_size": args.group_size,
        "reused_groups": len(reused),
        "pending_groups": len(tasks) - len(reused),
        "shards": [
            {
                "index": index,
                "tasks": len(identities),
                "path": str(shards_dir / f"shard-{index:02d}.ids"),
                "sha256": sha256_file(shards_dir / f"shard-{index:02d}.ids"),
            }
            for index, identities in enumerate(shards)
        ],
        "reused_task_ids": reused,
    }
    manifest_path = args.output_dir / "preparation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
