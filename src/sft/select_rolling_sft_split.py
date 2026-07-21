#!/usr/bin/env python3
"""Create a deterministic, difficulty-stratified train/holdout split of verified rolling episodes."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quotas(groups: dict[str, list[dict]], total: int) -> dict[str, int]:
    available = sum(len(rows) for rows in groups.values())
    if not 0 < total < available:
        raise ValueError(f"--train-count must be between 1 and {available - 1}")
    raw = {difficulty: total * len(rows) / available for difficulty, rows in groups.items()}
    result = {difficulty: int(value) for difficulty, value in raw.items()}
    for difficulty in sorted(raw, key=lambda item: (raw[item] - result[item], item), reverse=True):
        if sum(result.values()) == total:
            break
        if result[difficulty] < len(groups[difficulty]):
            result[difficulty] += 1
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--holdout-out", type=Path, required=True)
    parser.add_argument("--holdout-tasks-out", type=Path,
                        help="DatasetTask JSONL for closed-loop evaluation of the held-out episodes")
    parser.add_argument("--train-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if not rows or any(row.get("label_status") != "verified" for row in rows):
        raise ValueError("input must contain only verified episodes")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("difficulty") or "unknown")].append(row)
    selected = quotas(groups, args.train_count)
    rng = random.Random(args.seed)
    train, holdout = [], []
    for difficulty in sorted(groups):
        candidates = sorted(groups[difficulty], key=lambda row: row["trajectory_id"])
        rng.shuffle(candidates)
        train.extend(candidates[:selected[difficulty]])
        holdout.extend(candidates[selected[difficulty]:])
    train.sort(key=lambda row: row["trajectory_id"])
    holdout.sort(key=lambda row: row["trajectory_id"])
    train_ids = {row["trajectory_id"] for row in train}
    holdout_ids = {row["trajectory_id"] for row in holdout}
    if train_ids & holdout_ids or len(train) != args.train_count or len(train) + len(holdout) != len(rows):
        raise AssertionError("invalid split")
    write_jsonl(args.train_out, train)
    write_jsonl(args.holdout_out, holdout)
    if args.holdout_tasks_out:
        tasks = []
        for row in holdout:
            source = row["source"]
            tasks.append({
                "dataset": source.get("dataset"),
                "split": source.get("split"),
                "example_id": source.get("example_id"),
                "db_id": source["db_id"],
                "db_path": source["db_path"],
                "question": row["question"],
                "gold_sql": source["gold_sql"],
                "external_knowledge": source.get("external_knowledge"),
                "metadata": {"difficulty_proxy": row.get("difficulty")},
            })
        write_jsonl(args.holdout_tasks_out, tasks)
    manifest = {
        "input": str(args.input),
        "seed": args.seed,
        "total": len(rows),
        "train_count": len(train),
        "holdout_count": len(holdout),
        "train_difficulties": dict(sorted(Counter(row.get("difficulty") for row in train).items())),
        "holdout_difficulties": dict(sorted(Counter(row.get("difficulty") for row in holdout).items())),
        "train_trajectory_ids": [row["trajectory_id"] for row in train],
        "holdout_trajectory_ids": [row["trajectory_id"] for row in holdout],
        "holdout_tasks": str(args.holdout_tasks_out) if args.holdout_tasks_out else None,
    }
    manifest_path = args.train_out.with_suffix(".split_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
