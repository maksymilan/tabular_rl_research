#!/usr/bin/env python3
"""Materialize every unused mixed SFT2-K4 task from a frozen candidate pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


LEVELS = ("simple", "moderate", "challenging")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.next.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def task_id(row: dict[str, Any]) -> str:
    return str(row.get("example_id") or row.get("instance_id"))


def verified_correct(row: dict[str, Any], *, protocol_version: str) -> bool:
    sample = row.get("sample") or {}
    audit = sample.get("audit_record") or {}
    if audit.get("protocol_version") != protocol_version:
        raise ValueError(
            f"unexpected protocol_version={audit.get('protocol_version')!r}"
        )
    return bool(sample.get("correct") and audit.get("correct") and audit.get("legal"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-tasks", required=True, type=Path)
    parser.add_argument("--groups-dir", required=True, type=Path)
    parser.add_argument("--existing-eligibility", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--protocol-version", default="version26")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"refusing existing expansion output: {args.output_dir}")
    tasks = load_jsonl(args.candidate_tasks)
    existing = {
        int(row["example_index"]) for row in load_jsonl(args.existing_eligibility)
    }
    selected_tasks = []
    rollout_rows = []
    difficulty_counts: Counter[str] = Counter()
    correct_counts: Counter[int] = Counter()
    for task in tasks:
        example_index = int(task["example_index"])
        if example_index in existing:
            continue
        level = str((task.get("metadata") or {}).get("fixed_pool_difficulty"))
        if level not in LEVELS:
            raise ValueError(f"example {example_index} has unsupported difficulty {level!r}")
        group_path = args.groups_dir / f"{task_id(task)}.json"
        if not group_path.is_file():
            raise ValueError(f"missing frozen SFT2 group: {group_path}")
        group = json.loads(group_path.read_text(encoding="utf-8"))
        if not isinstance(group, list) or len(group) != args.group_size:
            raise ValueError(f"malformed frozen SFT2 group: {group_path}")
        sample_indices = []
        for row in group:
            environment = row.get("environment") or {}
            audit = (row.get("sample") or {}).get("audit_record") or {}
            if int(environment.get("example_index", -1)) != example_index:
                raise ValueError(f"group identity mismatch: {group_path}")
            sample_indices.append(int(audit.get("sample_index", -1)))
        if sorted(sample_indices) != list(range(args.group_size)):
            raise ValueError(f"group sample indices drifted: {group_path}")
        correct = sum(
            verified_correct(row, protocol_version=args.protocol_version) for row in group
        )
        if not 0 < correct < args.group_size:
            continue
        selected_tasks.append(task)
        difficulty_counts[level] += 1
        correct_counts[correct] += 1
        for row in sorted(
            group,
            key=lambda value: int(
                ((value.get("sample") or {}).get("audit_record") or {})["sample_index"]
            ),
        ):
            retained = dict(row)
            retained["sequence"] = len(rollout_rows)
            rollout_rows.append(retained)

    args.output_dir.mkdir(parents=True)
    tasks_path = args.output_dir / "tasks.jsonl"
    rollouts_path = args.output_dir / "sft2_trajectories.jsonl"
    write_jsonl_atomic(tasks_path, selected_tasks)
    write_jsonl_atomic(rollouts_path, rollout_rows)
    manifest = {
        "schema_version": "teacher-union-expansion-pool-v1",
        "status": "frozen",
        "candidate_tasks": str(args.candidate_tasks.resolve()),
        "candidate_tasks_sha256": sha256_file(args.candidate_tasks),
        "groups_dir": str(args.groups_dir.resolve()),
        "existing_eligibility": str(args.existing_eligibility.resolve()),
        "existing_eligibility_sha256": sha256_file(args.existing_eligibility),
        "protocol_version": args.protocol_version,
        "group_size": args.group_size,
        "tasks": len(selected_tasks),
        "trajectories": len(rollout_rows),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "sft2_correct_count_distribution": {
            str(key): value for key, value in sorted(correct_counts.items())
        },
        "tasks_path": str(tasks_path.resolve()),
        "tasks_sha256": sha256_file(tasks_path),
        "sft2_rollouts_path": str(rollouts_path.resolve()),
        "sft2_rollouts_sha256": sha256_file(rollouts_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
