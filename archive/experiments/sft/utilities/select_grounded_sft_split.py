#!/usr/bin/env python3
"""Select a deterministic episode-level SFT split from a process-grounding report."""
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
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "targets": sum(len(row.get("steps") or []) for row in rows),
        "difficulty": dict(sorted(collections.Counter(
            row.get("difficulty") or "unknown" for row in rows
        ).items())),
        "feedback_recovery_targets": sum(
            bool(step.get("feedback_recovery"))
            for row in rows
            for step in row.get("steps") or []
        ),
        "databases": len({(row.get("source") or {}).get("db_id") for row in rows}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--eval-out", type=Path, required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if not 0 <= args.eval_fraction < 0.5:
        parser.error("--eval-fraction must be between 0 (inclusive) and 0.5")

    trajectories_path = args.trajectories.resolve()
    scored_path = args.scored.resolve()
    by_id = {row["trajectory_id"]: row for row in read_jsonl(trajectories_path)}
    scored = read_jsonl(scored_path)
    score_ids = {row["trajectory_id"] for row in scored}
    if score_ids != set(by_id):
        raise ValueError("trajectory ids and scored ids differ")

    eligible: list[dict[str, Any]] = []
    rejected = collections.Counter()
    for score in scored:
        diagnostics = score.get("diagnostics") or {}
        if not score.get("correct"):
            rejected["replay_incorrect"] += 1
            continue
        if not diagnostics.get("deterministic_grounding_complete"):
            if not diagnostics.get("final_value_grounding_complete"):
                rejected["unsupported_final_value"] += 1
            if not diagnostics.get("action_literal_grounding_complete"):
                rejected["unsupported_action_literal"] += 1
            continue
        eligible.append(by_id[score["trajectory_id"]])

    rng = random.Random(args.seed)
    train: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for difficulty in sorted({row.get("difficulty") or "unknown" for row in eligible}):
        bucket = sorted(
            [row for row in eligible if (row.get("difficulty") or "unknown") == difficulty],
            key=lambda row: row["trajectory_id"],
        )
        rng.shuffle(bucket)
        eval_count = 0 if args.eval_fraction == 0 else max(1, round(len(bucket) * args.eval_fraction))
        evaluation.extend(bucket[:eval_count])
        train.extend(bucket[eval_count:])

    train.sort(key=lambda row: row["trajectory_id"])
    evaluation.sort(key=lambda row: row["trajectory_id"])
    train_out = args.train_out.resolve()
    eval_out = args.eval_out.resolve()
    write_jsonl(train_out, train)
    write_jsonl(eval_out, evaluation)
    manifest = {
        "trajectories": str(trajectories_path),
        "trajectories_sha256": sha256(trajectories_path),
        "scored": str(scored_path),
        "scored_sha256": sha256(scored_path),
        "selection": {
            "replay_correct": True,
            "deterministic_grounding_complete": True,
            "episode_level_split": True,
            "seed": args.seed,
            "eval_fraction": args.eval_fraction,
        },
        "eligible": stats(eligible),
        "train": {**stats(train), "path": str(train_out), "sha256": sha256(train_out)},
        "eval": {**stats(evaluation), "path": str(eval_out), "sha256": sha256(eval_out)},
        "episode_overlap": sorted(
            {row["trajectory_id"] for row in train} & {row["trajectory_id"] for row in evaluation}
        ),
        "rejected_issue_counts": dict(sorted(rejected.items())),
    }
    manifest_path = train_out.with_suffix(".split_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
