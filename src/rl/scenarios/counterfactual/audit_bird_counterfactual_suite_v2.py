#!/usr/bin/env python3
"""Replay source-correct BIRD trajectories against a pending counterfactual suite."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rl.runtime.trajectory_replay import evaluate_counterfactual_suite


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def trajectory_task_id(trajectory: dict[str, Any]) -> str:
    source = trajectory.get("source") or {}
    value = source.get("example_id")
    if isinstance(value, str) and value:
        return value
    identifier = str(trajectory.get("trajectory_id") or "")
    marker = identifier.find("_passk")
    if marker > 0:
        return identifier[:marker]
    raise ValueError(f"trajectory has no stable task id: {identifier!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pending-manifest", type=Path, required=True)
    parser.add_argument("--trajectories-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.pending_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "process-counterfactual-suite-v2":
        raise ValueError("unexpected counterfactual manifest schema")
    if (manifest.get("quality_gate") or {}).get("status") != "pending":
        raise ValueError("audit input must be a pending counterfactual manifest")
    suites_by_trajectory_id = {
        str(task["trajectory_task_id"]): {
            "trainer_task_id": trainer_task_id,
            "database_paths": [
                str(item["path"]) for item in task["databases"]
            ],
            "min_informative_databases": int(task["min_informative_databases"]),
        }
        for trainer_task_id, task in manifest["tasks"].items()
    }

    trajectories = [
        trajectory
        for trajectory in read_jsonl(args.trajectories_jsonl)
        if bool((trajectory.get("failure_audit") or {}).get("recorded_correct"))
        and trajectory_task_id(trajectory) in suites_by_trajectory_id
    ]
    observed_task_ids = {trajectory_task_id(item) for item in trajectories}
    missing = sorted(set(suites_by_trajectory_id) - observed_task_ids)
    if missing:
        raise ValueError(f"no source-correct trajectory for tasks: {missing}")

    completed: dict[str, dict[str, Any]] = {}
    if args.resume and args.output_jsonl.exists():
        for record in read_jsonl(args.output_jsonl):
            completed[str(record["trajectory_id"])] = record
    elif args.output_jsonl.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_jsonl}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if completed else "w"
    with args.output_jsonl.open(mode, encoding="utf-8") as sink:
        for ordinal, trajectory in enumerate(trajectories, 1):
            trajectory_id = str(trajectory["trajectory_id"])
            if trajectory_id in completed:
                continue
            task_id = trajectory_task_id(trajectory)
            suite = suites_by_trajectory_id[task_id]
            try:
                result = evaluate_counterfactual_suite(
                    trajectory,
                    suite["database_paths"],
                    min_informative_databases=suite["min_informative_databases"],
                    denotation_comparison="bird-set",
                )
                record = {
                    "trajectory_id": trajectory_id,
                    "trajectory_task_id": task_id,
                    "trainer_task_id": suite["trainer_task_id"],
                    "source_recorded_correct": True,
                    "replay": result.to_dict(),
                    "audit_error": None,
                }
            except Exception as exc:
                record = {
                    "trajectory_id": trajectory_id,
                    "trajectory_task_id": task_id,
                    "trainer_task_id": suite["trainer_task_id"],
                    "source_recorded_correct": True,
                    "replay": None,
                    "audit_error": f"{type(exc).__name__}: {exc}",
                }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            completed[trajectory_id] = record
            print(
                json.dumps(
                    {
                        "event": "trajectory_complete",
                        "ordinal": ordinal,
                        "total": len(trajectories),
                        "trajectory_id": trajectory_id,
                        "passed": bool(
                            record["replay"] and record["replay"]["passed"]
                        ),
                        "audit_error": record["audit_error"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in completed.values():
        by_task[str(record["trainer_task_id"])].append(record)
    task_summaries = {}
    for trainer_task_id in sorted(manifest["tasks"]):
        records = by_task[trainer_task_id]
        passed = [
            record
            for record in records
            if record["replay"] is not None and record["replay"]["passed"]
        ]
        errors = [record for record in records if record["audit_error"]]
        task_summaries[trainer_task_id] = {
            "trajectory_task_id": manifest["tasks"][trainer_task_id]["trajectory_task_id"],
            "source_correct_trajectories": len(records),
            "counterfactual_passed_trajectories": len(passed),
            "counterfactual_pass_rate": (
                len(passed) / len(records) if records else 0.0
            ),
            "audit_errors": len(errors),
            "has_known_correct_passing_program": bool(passed),
            "passing_trajectory_ids": [
                record["trajectory_id"] for record in passed
            ],
            "failing_trajectory_ids": [
                record["trajectory_id"]
                for record in records
                if record not in passed
            ],
        }
    tasks_without_passing_program = [
        task_id
        for task_id, record in task_summaries.items()
        if not record["has_known_correct_passing_program"]
    ]
    total_passed = sum(
        record["counterfactual_passed_trajectories"]
        for record in task_summaries.values()
    )
    summary = {
        "schema_version": "process-counterfactual-known-correct-replay-audit-v1",
        "pending_manifest": str(args.pending_manifest.resolve()),
        "trajectories_jsonl": str(args.trajectories_jsonl.resolve()),
        "source_correct_trajectories": len(completed),
        "counterfactual_passed_trajectories": total_passed,
        "counterfactual_pass_rate": (
            total_passed / len(completed) if completed else 0.0
        ),
        "tasks": task_summaries,
        "tasks_without_passing_program": tasks_without_passing_program,
        "provisional_known_correct_gate_passed": not tasks_without_passing_program,
    }
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "source_correct_trajectories": len(completed),
                "passed": total_passed,
                "tasks_without_passing_program": tasks_without_passing_program,
                "summary": str(args.summary_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
