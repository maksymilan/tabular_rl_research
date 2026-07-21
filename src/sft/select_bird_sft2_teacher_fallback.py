#!/usr/bin/env python3
"""Select a frozen, stratified cohort from SFT-1 student pass@K failures."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--passk-all", type=Path, action="append", required=True)
    parser.add_argument("--include-index", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index-out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--total", type=int, default=20)
    parser.add_argument(
        "--all-failures", action="store_true",
        help="select every pass@K-failed task; preserves the residual on-policy difficulty distribution",
    )
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--per-db-cap", type=int, default=2)
    args = parser.parse_args()
    if not args.all_failures and (args.total <= 0 or args.total % 10):
        parser.error("--total must be a positive multiple of 10")

    unit = args.total // 10
    quotas = {"easy": 4 * unit, "medium": 3 * unit, "hard": 3 * unit}
    tasks = {int(row["example_index"]): row for row in read_jsonl(args.tasks)}
    failures: dict[int, dict[str, Any]] = {}
    source_by_index: dict[int, str] = {}
    for path in args.passk_all:
        for row in read_jsonl(path):
            index = int(row["example_index"])
            if index in failures:
                raise ValueError(f"duplicate pass@K example {index}")
            if not row.get("correct"):
                failures[index] = row
                source_by_index[index] = str(path)

    required_by_difficulty: dict[str, list[int]] = collections.defaultdict(list)
    if args.include_index:
        for row in read_jsonl(args.include_index):
            index = int(row["example_index"])
            if index not in failures:
                raise ValueError(f"required example {index} is not a pass@K failure")
            difficulty = str((tasks[index].get("metadata") or {}).get("difficulty_proxy"))
            required_by_difficulty[difficulty].append(index)

    if args.all_failures:
        selected_indices = []
        required_flat = [item for values in required_by_difficulty.values() for item in values]
        selected_indices.extend(dict.fromkeys(required_flat))
        remaining = sorted(index for index in failures if index not in selected_indices)
        rng = random.Random(args.seed)
        rng.shuffle(remaining)
        selected_indices.extend(remaining)
        db_counts = collections.Counter(str(tasks[index]["db_id"]) for index in selected_indices)
        quotas = dict(collections.Counter(
            str((tasks[index].get("metadata") or {}).get("difficulty_proxy"))
            for index in selected_indices
        ))
    else:
        rng = random.Random(args.seed)
        selected_indices = []
        db_counts: collections.Counter[str] = collections.Counter()
        for difficulty, quota in quotas.items():
            required = list(dict.fromkeys(required_by_difficulty.get(difficulty, [])))
            if len(required) > quota:
                raise ValueError(f"required {difficulty} examples exceed quota {quota}")
            for index in required:
                selected_indices.append(index)
                db_counts[str(tasks[index]["db_id"])] += 1
            candidates = [
                index for index in failures
                if index not in selected_indices
                and (tasks[index].get("metadata") or {}).get("difficulty_proxy") == difficulty
            ]
            candidates.sort()
            rng.shuffle(candidates)
            while sum(
                (tasks[index].get("metadata") or {}).get("difficulty_proxy") == difficulty
                for index in selected_indices
            ) < quota:
                chosen = next((
                    index for index in candidates
                    if not args.per_db_cap or db_counts[str(tasks[index]["db_id"])] < args.per_db_cap
                ), None)
                if chosen is None:
                    chosen = candidates[0] if candidates else None
                if chosen is None:
                    raise ValueError(f"not enough {difficulty} pass@K failures for quota {quota}")
                candidates.remove(chosen)
                selected_indices.append(chosen)
                db_counts[str(tasks[chosen]["db_id"])] += 1

    selected_tasks = [tasks[index] for index in selected_indices]
    index_rows = [{
        "example_id": tasks[index].get("example_id"),
        "example_index": index,
        "difficulty": (tasks[index].get("metadata") or {}).get("difficulty_proxy"),
        "db_id": tasks[index]["db_id"],
        "student_passk_source": source_by_index[index],
        "student_pass4_correct": False,
        "required_from_prior_pilot": index in {
            item for values in required_by_difficulty.values() for item in values
        },
    } for index in selected_indices]
    write_jsonl(args.out, selected_tasks)
    write_jsonl(args.index_out, index_rows)
    manifest = {
        "method": "stratified_sft1_student_pass4_failures",
        "tasks": str(args.tasks),
        "passk_inputs": [str(path) for path in args.passk_all],
        "include_index": str(args.include_index) if args.include_index else None,
        "seed": args.seed,
        "all_failures": args.all_failures,
        "total": len(selected_tasks),
        "quotas": quotas,
        "counts": dict(collections.Counter(row["difficulty"] for row in index_rows)),
        "required_prior_pilot": sum(row["required_from_prior_pilot"] for row in index_rows),
        "unique_databases": len({row["db_id"] for row in index_rows}),
        "database_counts": dict(sorted(db_counts.items())),
        "output": str(args.out),
        "output_sha256": file_sha256(args.out),
        "index_output": str(args.index_out),
        "selected_example_indices": selected_indices,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
