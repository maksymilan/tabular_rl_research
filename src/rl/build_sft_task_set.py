#!/usr/bin/env python3
"""Build the exact BIRD RL task cohort represented by a retained SFT index."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_example_index(trajectory: dict[str, Any]) -> int:
    source = trajectory.get("source") or {}
    stable_id = str(
        source.get("example_id")
        or trajectory.get("trajectory_id")
        or ""
    )
    match = re.search(r"(\d+)$", stable_id)
    if not match:
        raise ValueError(f"trajectory has no numeric example suffix: {stable_id!r}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-index", type=Path, required=True)
    parser.add_argument("--trajectory-input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    index = read_jsonl(args.sft_index)
    retained_ids = {str(row["source_episode_id"]) for row in index}
    trajectories: dict[str, dict[str, Any]] = {}
    for path in args.trajectory_input:
        for trajectory in read_jsonl(path):
            trajectory_id = str(trajectory["trajectory_id"])
            if trajectory_id in trajectories:
                raise ValueError(f"duplicate trajectory_id: {trajectory_id}")
            trajectories[trajectory_id] = trajectory
    missing = retained_ids - trajectories.keys()
    if missing:
        raise ValueError(f"{len(missing)} retained SFT episodes lack source trajectories")

    examples = []
    seen_indices: set[int] = set()
    for trajectory_id in sorted(retained_ids):
        trajectory = trajectories[trajectory_id]
        source = trajectory.get("source") or {}
        example_index = numeric_example_index(trajectory)
        if example_index in seen_indices:
            raise ValueError(f"duplicate numeric example_index: {example_index}")
        seen_indices.add(example_index)
        examples.append(
            {
                "dataset": source.get("dataset", "bird-sql"),
                "split": source.get("split", "train"),
                "example_id": source.get("example_id") or trajectory_id,
                "trajectory_id": trajectory_id,
                "example_index": example_index,
                "db_id": source["db_id"],
                "db_path": source["db_path"],
                "question": trajectory["question"],
                "gold_sql": source["gold_sql"],
                "external_knowledge": source.get("external_knowledge"),
                "difficulty": trajectory.get("difficulty"),
                "denotation_comparison": "bird-set",
            }
        )

    payload = {
        "schema_version": "sft-index-rl-task-set-v1",
        "dataset_split": "train",
        "denotation_comparison": "bird-set",
        "context_mode": "rolling-legal-history",
        "history_turns": 4,
        "sft_index": {
            "path": str(args.sft_index.resolve()),
            "sha256": sha256(args.sft_index),
            "records": len(index),
            "episodes": len(retained_ids),
        },
        "trajectory_inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in args.trajectory_input
        ],
        "count": len(examples),
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "examples": len(examples),
                "denotation_comparison": "bird-set",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
