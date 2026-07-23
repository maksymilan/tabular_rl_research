#!/usr/bin/env python3
"""Create the frozen difficulty-stratified BIRD-dev cohort for tool evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--total", type=int, default=200)
    args = parser.parse_args()
    if args.total <= 0 or args.total % 10:
        parser.error("--total must be a positive multiple of 10")

    tasks = [
        json.loads(line)
        for line in args.tasks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    quotas = {
        "simple": args.total * 4 // 10,
        "moderate": args.total * 3 // 10,
        "challenging": args.total * 3 // 10,
    }
    buckets = {difficulty: [] for difficulty in quotas}
    for source_index, task in enumerate(tasks):
        example_index = int(task.get("example_index", source_index))
        if example_index != source_index:
            raise ValueError(
                f"source row {source_index} has unexpected example_index {example_index}"
            )
        difficulty = (task.get("metadata") or {}).get("difficulty")
        if difficulty in buckets:
            buckets[difficulty].append(source_index)

    rng = random.Random(args.seed)
    selected_indices = []
    for difficulty, quota in quotas.items():
        candidates = buckets[difficulty]
        rng.shuffle(candidates)
        if len(candidates) < quota:
            raise ValueError(
                f"only {len(candidates)} {difficulty} tasks available for quota {quota}"
            )
        selected_indices.extend(candidates[:quota])
    selected_indices.sort()
    selected = [tasks[index] for index in selected_indices]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for task in selected:
            handle.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "purpose": "held-out historical-tool SFT ablation",
        "dataset": "BIRD dev 20240627",
        "source": str(args.tasks),
        "source_sha256": file_sha256(args.tasks),
        "seed": args.seed,
        "total": len(selected),
        "difficulty": dict(
            sorted(
                Counter(
                    (task.get("metadata") or {}).get("difficulty") for task in selected
                ).items()
            )
        ),
        "source_example_indices": selected_indices,
        "db_ids": len({task["db_id"] for task in selected}),
        "output": str(args.out),
        "output_sha256": file_sha256(args.out),
        "denotation_comparison": "strict-multiset",
        "note": "The 40/30/30 cohort oversamples challenging tasks; report only paired deltas.",
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
