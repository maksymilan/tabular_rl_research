#!/usr/bin/env python3
"""Assemble an offline semantic-ranking pool from existing BIRD-train rollout shards.

This command never calls a model.  It binds the already-recorded ``verified.all.jsonl`` rows to
their local SQLite databases and marks which correct trajectories were ultimately admitted to
SFT, so reward diagnostics can report the full generation population and the training-admitted
subset separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def trajectory_id(row: dict[str, Any]) -> str:
    value = row.get("trajectory_id") or row.get("example_id") or row.get("source_episode_id")
    if not value:
        raise ValueError("rollout record lacks trajectory_id/example_id")
    return str(value)


def admitted_ids(path: Path) -> set[str]:
    return {trajectory_id(row) for row in read_jsonl(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", action="append", type=Path, required=True)
    parser.add_argument("--admitted", action="append", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if len(args.rollout_root) != len(args.admitted):
        parser.error("provide exactly one --admitted file per --rollout-root")

    output_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_files: list[dict[str, Any]] = []
    lane_summaries: list[dict[str, Any]] = []
    for lane_index, (root, admitted_path) in enumerate(
        zip(args.rollout_root, args.admitted, strict=True), start=1
    ):
        admitted = admitted_ids(admitted_path)
        lane_counts: Counter[str] = Counter()
        lane_ids: set[str] = set()
        shard_paths = sorted(root.glob("shard_*/verified.all.jsonl"))
        if not shard_paths:
            raise ValueError(f"no verified.all.jsonl shards under {root}")
        for shard_path in shard_paths:
            shard_rows = read_jsonl(shard_path)
            source_files.append({
                "path": str(shard_path.resolve()),
                "sha256": sha256_file(shard_path),
                "records": len(shard_rows),
            })
            for record in shard_rows:
                task_id = trajectory_id(record)
                if task_id in seen:
                    raise ValueError(f"duplicate trajectory id {task_id!r}")
                seen.add(task_id)
                lane_ids.add(task_id)
                if not task_id.startswith("bird_train_"):
                    raise ValueError(f"non-train trajectory id {task_id!r}")
                example_index = record.get("example_index")
                if example_index is None:
                    example_index = int(task_id.rsplit("_", 1)[-1])
                db_id = str(record.get("db_id") or "")
                db_path = args.database_root / db_id / f"{db_id}.sqlite"
                if not db_path.is_file():
                    raise ValueError(f"missing database for {task_id}: {db_path}")
                correct = bool(record.get("correct"))
                is_admitted = task_id in admitted
                if is_admitted and not correct:
                    raise ValueError(f"incorrect trajectory was marked admitted: {task_id}")
                lane_counts["records"] += 1
                lane_counts["correct"] += int(correct)
                lane_counts["incorrect"] += int(not correct)
                lane_counts["admitted_correct"] += int(is_admitted)
                lane_counts[f"failure:{record.get('failure_type')}"] += int(not correct)
                output_rows.append({
                    "environment": {
                        "dataset": "bird-sql",
                        "split": "train",
                        "task_id": task_id,
                        "example_id": task_id,
                        "example_index": int(example_index),
                        "db_id": db_id,
                        "db_path": str(db_path.resolve()),
                        "question": record.get("question"),
                        "gold_sql": record.get("gold_sql"),
                        "external_knowledge": record.get("external_knowledge"),
                        "denotation_comparison": "bird-set",
                        "metadata": {
                            "existing_rollout_lane": lane_index,
                            "sft_admitted": is_admitted,
                        },
                    },
                    "sample": {
                        "correct": correct,
                        "audit_record": record,
                    },
                    "offline_source": {
                        "rollout_root": str(root.resolve()),
                        "shard": str(shard_path.resolve()),
                        "sft_admitted": is_admitted,
                    },
                })
        missing_admitted = admitted - lane_ids
        if missing_admitted:
            raise ValueError(
                f"{len(missing_admitted)} admitted IDs are absent from {root}; "
                f"first={sorted(missing_admitted)[:3]}"
            )
        lane_summaries.append({
            "lane": lane_index,
            "rollout_root": str(root.resolve()),
            "admitted": str(admitted_path.resolve()),
            "admitted_sha256": sha256_file(admitted_path),
            **dict(sorted(lane_counts.items())),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for row in output_rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "existing-bird-train-semantic-rank-pool-v1",
        "operation": "offline-bind-existing-rollouts",
        "model_calls": 0,
        "optimizer_updates": 0,
        "dataset_split": "train",
        "database_root": str(args.database_root.resolve()),
        "records": len(output_rows),
        "unique_trajectories": len(seen),
        "correct": sum(bool(row["sample"]["correct"]) for row in output_rows),
        "incorrect": sum(not bool(row["sample"]["correct"]) for row in output_rows),
        "sft_admitted_correct": sum(
            bool(row["environment"]["metadata"]["sft_admitted"])
            for row in output_rows
        ),
        "lanes": lane_summaries,
        "source_files": source_files,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
