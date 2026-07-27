#!/usr/bin/env python3
"""Replay student rollout successes and partition failures for batch-2 SFT.

The recovery lane contains only evaluator candidates anchored to real, state-preserving harness
errors.  Candidate context ends at the selected error: rejected assistant text and all future
turns are excluded from the continuation prefix.  Gold SQL is retained only in the separate task
file used by the harness and is never copied into evaluator candidate records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(HERE),
]

from bird_sft1_teacher import replay_success_trajectory  # noqa: E402
from build_bird_sft2_dataset import preferred_failed_sample, replay_sample  # noqa: E402
from select_verified_rollouts import quality_reason  # noqa: E402


RECOVERABLE_ERROR_TYPES = {
    "protocol_error",
    "argument_validation_error",
    "execution_error",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def task_id(task: dict[str, Any]) -> str:
    value = task.get("example_id") or task.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"task {task.get('example_index')}: missing example_id")
    return value


def task_map(tasks: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    ids: set[str] = set()
    for task in tasks:
        index = int(task["example_index"])
        identity = task_id(task)
        if index in result or identity in ids:
            raise ValueError(f"duplicate task identity: {index} / {identity}")
        result[index] = task
        ids.add(identity)
    return result


def visible_task_context(task: dict[str, Any]) -> dict[str, Any]:
    """Return evaluator-visible identity/context without harness-only answer data."""
    return {
        "example_id": task_id(task),
        "example_index": int(task["example_index"]),
        "db_id": task["db_id"],
        "question": task["question"],
        "external_knowledge": task.get("external_knowledge"),
        "difficulty": (task.get("metadata") or {}).get("difficulty_proxy"),
    }


def legal_prefix_before(
    turns: list[dict[str, Any]],
    action_index: int,
) -> list[dict[str, Any]]:
    """Keep only successful structured actions before an error, never rejected text."""
    prefix: list[dict[str, Any]] = []
    for ordinal, turn in enumerate(turns, start=1):
        turn_action = int(turn.get("turn_index", ordinal - 1)) + 1
        if turn_action >= action_index:
            break
        if turn.get("execution_error_type") or turn.get("error_event"):
            continue
        parsed = turn.get("parsed")
        if not isinstance(parsed, dict) or not parsed.get("tool"):
            continue
        if "tool_output" not in turn:
            continue
        prefix.append(
            {
                "action_index": turn_action,
                "tool": parsed["tool"],
                "arguments": parsed.get("arguments") or {},
                "tool_output": turn["tool_output"],
            }
        )
    return prefix


def recoverable_error_events(
    sample: dict[str, Any],
    *,
    max_anchor_action: int | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_actions: set[int] = set()
    for event in sample.get("error_events") or []:
        error_type = str(event.get("error_type") or "")
        before = event.get("state_before_hash")
        after = event.get("state_after_hash")
        if (
            error_type not in RECOVERABLE_ERROR_TYPES
            or not isinstance(before, str)
            or not before
            or before != after
        ):
            continue
        action_index = int(event["action_index"])
        if max_anchor_action is not None and action_index > max_anchor_action:
            continue
        if action_index in seen_actions:
            raise ValueError(f"duplicate error action_index {action_index}")
        seen_actions.add(action_index)
        candidates.append(dict(event))
    return sorted(candidates, key=lambda event: int(event["action_index"]))


def recovery_package(
    task: dict[str, Any],
    sample: dict[str, Any],
    *,
    rollout_source: Path,
    max_anchor_action: int | None = None,
) -> dict[str, Any] | None:
    events = recoverable_error_events(
        sample,
        max_anchor_action=max_anchor_action,
    )
    if not events:
        return None
    turns = sample.get("turns") or []
    sample_index = int(sample.get("sample_index", 0))
    identity = task_id(task)
    candidates = []
    for event in events:
        action_index = int(event["action_index"])
        candidates.append(
            {
                "candidate_id": f"{identity}:sample_{sample_index}:action_{action_index}",
                "anchor_action_index": action_index,
                "legal_prefix": legal_prefix_before(turns, action_index),
                "last_tool_error": {
                    "step_id": str(event.get("step_id") or f"step_{action_index}"),
                    "status": "error",
                    "error": {
                        "type": event["error_type"],
                        "message": str(event.get("message") or ""),
                    },
                },
                "attempted_action": {
                    "tool": event.get("attempted_tool"),
                    "arguments": event.get("attempted_arguments"),
                },
                "state_before_hash": event["state_before_hash"],
                "state_after_hash": event["state_after_hash"],
            }
        )
    package = {
        "task": visible_task_context(task),
        "student_rollout": {
            "source": str(rollout_source),
            "sample_index": sample_index,
            "failure_type": sample.get("failure_type"),
            "legal": bool(sample.get("legal")),
            "steps": int(sample.get("steps") or 0),
            "errors": int(sample.get("errors") or 0),
        },
        "selection_status": "pending_external_evaluator",
        "selection_contract": {
            "allowed_decisions": ["select_candidate", "route_to_teacher_from_scratch"],
            "gold_sql_visible": False,
            "future_suffix_visible": False,
            "rejected_action_is_sft_target": False,
        },
        "candidates": candidates,
    }
    gold_sql = str(task.get("gold_sql") or task.get("query") or "")
    if gold_sql and gold_sql in json.dumps(package, ensure_ascii=False):
        raise ValueError(f"{identity}: gold SQL leaked into recovery evaluator package")
    return package


def choose_success_sample(record: dict[str, Any]) -> dict[str, Any] | None:
    successes = [sample for sample in record.get("samples") or [] if sample.get("correct")]
    if not successes:
        return None
    return min(
        successes,
        key=lambda sample: (
            int(sample.get("errors") or 0),
            int(sample.get("steps") or 10**6),
            int(sample.get("sample_index") or 0),
        ),
    )


def build(
    tasks_path: Path,
    rollout_paths: list[Path],
    accepted_success_out: Path,
    recovery_candidates_out: Path,
    scratch_tasks_out: Path,
    rejected_success_out: Path,
    manifest_path: Path,
    *,
    student_model: str,
    max_steps: int,
    max_think_words: int,
    max_recovery_anchor_action: int,
) -> dict[str, Any]:
    tasks = read_jsonl(tasks_path)
    by_index = task_map(tasks)
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in rollout_paths:
        records.extend((path, row) for row in read_jsonl(path))
    if len(records) != len(tasks):
        raise ValueError(f"rollout records {len(records)} != tasks {len(tasks)}")

    seen_indices: set[int] = set()
    accepted: list[dict[str, Any]] = []
    recovery: list[dict[str, Any]] = []
    scratch: list[dict[str, Any]] = []
    rejected_success: list[dict[str, Any]] = []
    scratch_reasons: Counter[str] = Counter()
    recovery_error_types: Counter[str] = Counter()
    replay_rejections: Counter[str] = Counter()

    for source_path, record in records:
        index = int(record["example_index"])
        if index in seen_indices:
            raise ValueError(f"duplicate rollout example_index {index}")
        seen_indices.add(index)
        if index not in by_index:
            raise ValueError(f"rollout example_index {index} is outside the task cohort")
        task = by_index[index]
        sample = choose_success_sample(record)
        if sample is not None:
            trajectory_id = f"bird_batch2_student_{task_id(task)}_sample_{sample.get('sample_index', 0)}"
            try:
                trajectory = replay_sample(
                    task,
                    sample,
                    require_correct=True,
                    trajectory_id=trajectory_id,
                )
                trajectory["schema_version"] = "bird-batch2-student-onpolicy-v1"
                generation = trajectory["rollout_generation"]
                generation.update(
                    {
                        "method": "batch2_selected_student_closed_loop",
                        "model": student_model,
                        "denotation_comparison": "bird-set",
                        "error_actions_are_sft_targets": False,
                    }
                )
                replay_ok, replay_error = replay_success_trajectory(
                    trajectory,
                    denotation_comparison="bird-set",
                )
                if not replay_ok:
                    raise ValueError(str(replay_error))
                reason = quality_reason(
                    trajectory,
                    max_steps=max_steps,
                    max_think_words=max_think_words,
                )
                if reason:
                    raise ValueError(f"quality:{reason}")
            except Exception as exc:  # noqa: BLE001
                detail = str(exc)
                category = detail if detail.startswith("quality:") else type(exc).__name__
                replay_rejections[category] += 1
                rejected_success.append(
                    {
                        "example_id": task_id(task),
                        "example_index": index,
                        "trajectory_id": trajectory_id,
                        "reason": category,
                        "detail": detail,
                    }
                )
                scratch.append(task)
                scratch_reasons["student_success_rejected_by_replay_or_quality"] += 1
            else:
                accepted.append(trajectory)
            continue

        failed_sample = preferred_failed_sample(record)
        package = recovery_package(
            task,
            failed_sample,
            rollout_source=source_path,
            max_anchor_action=max_recovery_anchor_action,
        )
        if package is None:
            scratch.append(task)
            scratch_reasons["no_state_preserving_error_anchor"] += 1
        else:
            recovery.append(package)
            for candidate in package["candidates"]:
                recovery_error_types[candidate["last_tool_error"]["error"]["type"]] += 1

    missing = set(by_index) - seen_indices
    if missing:
        raise ValueError(f"missing rollout records for task indices: {sorted(missing)[:5]}")
    accepted_ids = {trajectory["source"]["example_id"] for trajectory in accepted}
    recovery_ids = {package["task"]["example_id"] for package in recovery}
    scratch_ids = {task_id(task) for task in scratch}
    if accepted_ids & recovery_ids or accepted_ids & scratch_ids or recovery_ids & scratch_ids:
        raise RuntimeError("batch-2 lanes overlap")
    covered = accepted_ids | recovery_ids | scratch_ids
    expected = {task_id(task) for task in tasks}
    if covered != expected:
        raise RuntimeError("batch-2 lanes do not exactly cover the rollout cohort")

    write_jsonl_atomic(accepted_success_out, accepted)
    write_jsonl_atomic(recovery_candidates_out, recovery)
    write_jsonl_atomic(scratch_tasks_out, scratch)
    write_jsonl_atomic(rejected_success_out, rejected_success)
    manifest = {
        "method": "batch2_student_success_replay_and_failure_partition",
        "tasks": {
            "path": str(tasks_path),
            "sha256": sha256_file(tasks_path),
            "records": len(tasks),
        },
        "rollout_inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in rollout_paths
        ],
        "student_model": student_model,
        "denotation_comparison": "bird-set",
        "quality_gates": {
            "fresh_deterministic_replay": True,
            "max_legal_steps": max_steps,
            "max_think_words": max_think_words,
            "repeated_identical_calls": "reject",
            "gold_sql_visible_to_actor_or_evaluator": False,
            "rejected_actions_are_sft_targets": False,
            "max_recovery_anchor_action": max_recovery_anchor_action,
        },
        "coverage": {
            "tasks": len(tasks),
            "accepted_student_success": len(accepted),
            "pending_recovery_evaluator": len(recovery),
            "teacher_from_scratch_immediate": len(scratch),
            "total": len(covered),
            "lane_overlap": 0,
        },
        "accepted_student_targets": sum(len(row["steps"]) for row in accepted),
        "accepted_student_feedback_recovery_targets": sum(
            bool(step.get("feedback_recovery"))
            for row in accepted
            for step in row["steps"]
        ),
        "recovery_candidates": {
            "tasks": len(recovery),
            "anchors": sum(len(row["candidates"]) for row in recovery),
            "error_types": dict(sorted(recovery_error_types.items())),
            "policy": (
                "external evaluator selects only among real state-preserving error events; "
                "all rejected candidates fall back to teacher-from-scratch"
            ),
        },
        "scratch_reasons": dict(sorted(scratch_reasons.items())),
        "rejected_student_success": dict(sorted(replay_rejections.items())),
        "outputs": {
            "accepted_student_success": str(accepted_success_out),
            "recovery_candidates": str(recovery_candidates_out),
            "teacher_from_scratch_tasks": str(scratch_tasks_out),
            "rejected_student_success": str(rejected_success_out),
        },
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--rollout-all", type=Path, action="append", required=True)
    parser.add_argument("--accepted-success-out", type=Path, required=True)
    parser.add_argument("--recovery-candidates-out", type=Path, required=True)
    parser.add_argument("--scratch-tasks-out", type=Path, required=True)
    parser.add_argument("--rejected-success-out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-think-words", type=int, default=300)
    parser.add_argument(
        "--max-recovery-anchor-action",
        type=int,
        default=24,
        help="latest eligible error action; default leaves six actions in a 30-action episode",
    )
    args = parser.parse_args()
    manifest = build(
        args.tasks.resolve(),
        [path.resolve() for path in args.rollout_all],
        args.accepted_success_out.resolve(),
        args.recovery_candidates_out.resolve(),
        args.scratch_tasks_out.resolve(),
        args.rejected_success_out.resolve(),
        args.manifest.resolve(),
        student_model=args.student_model,
        max_steps=args.max_steps,
        max_think_words=args.max_think_words,
        max_recovery_anchor_action=args.max_recovery_anchor_action,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
