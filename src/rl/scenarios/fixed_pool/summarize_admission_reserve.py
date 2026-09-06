#!/usr/bin/env python3
"""Summarize causal admission of a deterministic fixed-pool reserve cohort."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(
    tasks: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    counterfactual_results: list[dict[str, Any]],
    *,
    required_per_level: int,
) -> dict[str, Any]:
    task_rows = {
        str(row.get("example_id") or row.get("instance_id")): row for row in tasks
    }
    correct_by_task: dict[str, int] = defaultdict(int)
    for row in trajectories:
        if row["sample"]["correct"]:
            correct_by_task[str(row["environment"]["task_id"])] += 1
    cf_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in counterfactual_results:
        cf_by_task[str(row["task_id"])].append(row)

    records = []
    for task_id, task in task_rows.items():
        metadata = task.get("metadata") or {}
        level = str(metadata["fixed_pool_difficulty"])
        correct = correct_by_task.get(task_id, 0)
        cf_rows = cf_by_task.get(task_id, [])
        passed = sum(bool(row["passed"]) for row in cf_rows)
        admitted = len(cf_rows) == correct and passed == correct
        records.append(
            {
                "task_id": task_id,
                "difficulty": level,
                "reserve_rank": int(metadata["fixed_pool_reserve_rank"]),
                "correct_trajectories": correct,
                "counterfactual_results": len(cf_rows),
                "counterfactual_passed": passed,
                "admitted": admitted,
                "reason": (
                    "passed"
                    if admitted and correct > 0
                    else "no_correct_trajectory_vacuous_gate_pass"
                    if admitted
                    else "counterfactual_incomplete_or_failed"
                ),
            }
        )

    selected: dict[str, list[str]] = {}
    for level in ("simple", "moderate", "challenging"):
        eligible = sorted(
            (row for row in records if row["difficulty"] == level and row["admitted"]),
            key=lambda row: (
                row["correct_trajectories"] == 0,
                row["reserve_rank"],
                row["task_id"],
            ),
        )
        selected[level] = [row["task_id"] for row in eligible[:required_per_level]]
    sufficient = all(len(values) == required_per_level for values in selected.values())
    return {
        "schema_version": "fixed-pool-admission-reserve-audit-v1",
        "status": "passed" if sufficient else "insufficient",
        "required_per_level": required_per_level,
        "tasks": len(records),
        "admitted_tasks": sum(row["admitted"] for row in records),
        "positive_bearing_admitted_tasks": sum(
            row["admitted"] and row["correct_trajectories"] > 0 for row in records
        ),
        "admission_rule": (
            "every source-correct trajectory must have a passing counterfactual result; "
            "tasks with zero correct trajectories pass this universal condition vacuously "
            "but are ranked after positive-bearing admitted tasks"
        ),
        "selected_replacements": selected,
        "records": sorted(records, key=lambda row: (row["difficulty"], row["reserve_rank"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--counterfactual-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--required-per-level", type=int, default=2)
    args = parser.parse_args()
    summary = summarize(
        load_jsonl(args.tasks),
        load_jsonl(args.trajectories),
        load_jsonl(args.counterfactual_results),
        required_per_level=args.required_per_level,
    )
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "passed":
        raise SystemExit("reserve cohort does not contain enough causally admitted replacements")


if __name__ == "__main__":
    main()
