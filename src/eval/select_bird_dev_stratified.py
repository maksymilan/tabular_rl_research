#!/usr/bin/env python3
"""Select a deterministic difficulty-stratified BIRD dev cohort by task index."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_indices(path: Path | None) -> set[int]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("indices", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError(f"{path}: expected an index list")
    return {int(value) for value in values}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-indices", type=Path)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--total", type=int, default=100)
    args = parser.parse_args()
    if args.total <= 0 or args.total % 10:
        parser.error("--total must be a positive multiple of 10")

    quotas = {
        "simple": args.total * 4 // 10,
        "moderate": args.total * 3 // 10,
        "challenging": args.total * 3 // 10,
    }
    excluded = read_indices(args.exclude_indices)
    tasks = read_jsonl(args.tasks)
    buckets = {difficulty: [] for difficulty in quotas}
    for index, task in enumerate(tasks):
        difficulty = (task.get("metadata") or {}).get("difficulty")
        if index not in excluded and difficulty in buckets:
            buckets[difficulty].append(index)

    rng = random.Random(args.seed)
    selected = []
    for difficulty, quota in quotas.items():
        candidates = buckets[difficulty]
        rng.shuffle(candidates)
        if len(candidates) < quota:
            raise ValueError(f"only {len(candidates)} {difficulty} tasks for quota {quota}")
        selected.extend(candidates[:quota])
    selected.sort()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "tasks": str(args.tasks),
        "indices": str(args.out),
        "seed": args.seed,
        "excluded_indices": sorted(excluded),
        "total": len(selected),
        "difficulty": dict(Counter(
            (tasks[index].get("metadata") or {}).get("difficulty")
            for index in selected
        )),
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
