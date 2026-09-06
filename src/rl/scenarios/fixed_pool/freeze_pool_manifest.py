#!/usr/bin/env python3
"""Validate, hash, and mark a fixed rollout pool read-only for controlled training."""
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
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", required=True, type=Path)
    parser.add_argument("--process-reward-config", required=True, type=Path)
    parser.add_argument("--counterfactual-manifest", type=Path)
    parser.add_argument(
        "--admission-mode",
        choices=("rank-only", "dense-outcome", "process-screened", "process-required"),
        default="process-required",
    )
    args = parser.parse_args()

    output = args.pool_dir / "manifest.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {output}")
    pending = json.loads((args.pool_dir / "manifest.pending.json").read_text())
    selection = json.loads((args.pool_dir / "task_selection_manifest.json").read_text())
    validation = json.loads((args.pool_dir / "validation_summary.json").read_text())
    counterfactual = (
        json.loads(args.counterfactual_manifest.read_text())
        if args.counterfactual_manifest is not None
        else None
    )
    paths = {
        "tasks": args.pool_dir / "tasks.jsonl",
        "trajectories": args.pool_dir / "trajectories.jsonl",
        "validated_trajectories": args.pool_dir / "validated_trajectories.jsonl",
        "transitions": args.pool_dir / "transitions.jsonl",
        "process_features": args.pool_dir / "process_features.jsonl",
        "counterfactual_results": args.pool_dir / "counterfactual_results.jsonl",
        "validation_summary": args.pool_dir / "validation_summary.json",
        "task_selection_manifest": args.pool_dir / "task_selection_manifest.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_tasks = int(selection.get("selected_tasks") or selection.get("tasks") or 0)
    if expected_tasks < 1:
        raise ValueError("task selection manifest has no positive selected task count")
    difficulty_counts = selection.get("difficulty_counts") or {}
    if set(difficulty_counts) != {"challenging", "moderate", "simple"}:
        raise ValueError("fixed pool must declare all three difficulty levels")
    if sum(int(value) for value in difficulty_counts.values()) != expected_tasks:
        raise ValueError("difficulty counts do not sum to the selected task count")
    if len({int(value) for value in difficulty_counts.values()}) != 1:
        raise ValueError("fixed pool is not difficulty-balanced")
    expected_trajectories = expected_tasks * 4
    rows = load_jsonl(paths["validated_trajectories"])
    if len(rows) != expected_trajectories or [row["sequence"] for row in rows] != list(
        range(expected_trajectories)
    ):
        raise ValueError(
            "validated pool must contain ordered sequences "
            f"0..{expected_trajectories - 1}"
        )
    group_counts = Counter(int(row["environment"]["example_index"]) for row in rows)
    if len(group_counts) != expected_tasks or set(group_counts.values()) != {4}:
        raise ValueError(
            f"validated pool must contain {expected_tasks} exact K4 groups"
        )
    correct = sum(row["sample"]["correct"] for row in rows)
    mixed_counts = Counter(
        sum(bool(row["sample"]["correct"]) for row in rows if int(row["environment"]["example_index"]) == example_index)
        for example_index in group_counts
    )
    if set(mixed_counts) - {1, 2, 3}:
        raise ValueError("fixed rank pool must contain only mixed K4 groups with 1..3 correct")
    counterfactual_rows = load_jsonl(paths["counterfactual_results"])
    if args.admission_mode in {"rank-only", "dense-outcome"}:
        if args.counterfactual_manifest is not None or counterfactual_rows:
            raise ValueError(
                f"{args.admission_mode} pool must not depend on counterfactual artifacts"
            )
        frozen_status = (
            "frozen_rank_ready"
            if args.admission_mode == "rank-only"
            else "frozen_dense_ready"
        )
    else:
        if args.counterfactual_manifest is None or counterfactual is None:
            raise ValueError(f"{args.admission_mode} requires a counterfactual manifest")
        if len(counterfactual_rows) != correct:
            raise ValueError("every correct trajectory must have a counterfactual screening result")
        if args.admission_mode == "process-required" and not all(
            row["passed"] for row in counterfactual_rows
        ):
            raise ValueError("every correct trajectory must pass required counterfactual replay")
        frozen_status = (
            "frozen_process_screened"
            if args.admission_mode == "process-screened"
            else "frozen_passed"
        )
    if pending.get("protocol_version") != "version26":
        raise ValueError("fixed pool protocol must be version26")
    if pending.get("adapter_sha256") != "d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e":
        raise ValueError("fixed pool did not originate from the frozen SFT2 adapter")
    if (
        validation.get("trajectories") != expected_trajectories
        or validation.get("admission_mode") != args.admission_mode
    ):
        raise ValueError("process validation summary is incomplete")
    if args.admission_mode == "process-required" and (
        (counterfactual.get("quality_gate") or {}).get("status") != "passed"
        or validation.get("counterfactual_passed") != correct
    ):
        raise ValueError("counterfactual manifest did not pass all consistency tests")
    if args.admission_mode == "process-screened" and (
        (counterfactual.get("quality_gate") or {}).get("status")
        not in {"passed", "screening_partial"}
    ):
        raise ValueError("counterfactual screening manifest has an invalid status")

    manifest = {
        "schema_version": "table-agent-fixed-rollout-pool-v1",
        "status": frozen_status,
        "admission_mode": args.admission_mode,
        "tasks": expected_tasks,
        "group_size": 4,
        "trajectories": expected_trajectories,
        "difficulty_counts": difficulty_counts,
        "protocol_version": "version26",
        "temperature": 0.7,
        "top_p": 0.95,
        "max_steps": 30,
        "history_turns": 4,
        "seed": 101,
        "initial_adapter_sha256": pending["adapter_sha256"],
        "process_reward_config": file_record(args.process_reward_config),
        "counterfactual_manifest": (
            file_record(args.counterfactual_manifest)
            if args.counterfactual_manifest is not None
            else None
        ),
        "counterfactual_quality_audit_sha256": (
            counterfactual["quality_gate"]["audit_sha256"]
            if counterfactual is not None
            else None
        ),
        "files": {name: file_record(path) for name, path in paths.items()},
        "validation": validation,
        "reuse_contract": {
            "same_trajectories": True,
            "trajectory_count": expected_trajectories,
            "same_240_trajectories": expected_trajectories == 240,
            "same_sequence": True,
            "online_resampling_forbidden": True,
            "training_initialization": "SFT2 checkpoint-1682 only",
        },
    }
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    immutable = [*paths.values(), output]
    if args.counterfactual_manifest is not None:
        immutable.append(args.counterfactual_manifest)
    for path in immutable:
        path.chmod(0o444)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
