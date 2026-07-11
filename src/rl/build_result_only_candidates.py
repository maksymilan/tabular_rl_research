#!/usr/bin/env python3
"""Export verified external-rollout training tasks for result-only RL selection.

The exported JSON is accepted by ``src/eval/rollout_passk.py --examples-json``.  It contains
Spider training questions only; developer/evaluation examples are deliberately excluded.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def train_index(trajectory_id: str) -> int:
    prefix = "spider_train_"
    if not trajectory_id.startswith(prefix):
        raise ValueError(f"not a Spider training trajectory id: {trajectory_id!r}")
    return int(trajectory_id[len(prefix):])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    examples = []
    seen: set[int] = set()
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            trajectory = json.loads(line)
            if trajectory.get("label_status") != "verified":
                continue
            index = train_index(str(trajectory["trajectory_id"]))
            if index in seen:
                raise ValueError(f"duplicate training index {index}")
            seen.add(index)
            origin = trajectory["source"]
            examples.append({
                "example_index": index,
                "dataset_split": "train",
                "db_id": origin["db_id"],
                "question": trajectory["question"],
                "query": origin["gold_sql"],
                "trajectory_id": trajectory["trajectory_id"],
            })

    random.Random(args.seed).shuffle(examples)
    if args.limit:
        examples = examples[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "source": str(args.input),
        "dataset_split": "train",
        "count": len(examples),
        "examples": examples,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(examples), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
