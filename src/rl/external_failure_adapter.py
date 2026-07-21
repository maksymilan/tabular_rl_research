#!/usr/bin/env python3
"""Normalize audited external-rollout failures for process-reward replay.

The external generator deliberately keeps rejected actions in ``turns/error_events`` instead of
trajectory ``steps``. This adapter preserves that trust boundary while converting successful legal
turns into the normalized trajectory shape consumed by ``process_reward``. API transport failures
are excluded because they are not model-semantic transitions.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def _task_id(record: dict[str, Any]) -> str | None:
    return record.get("example_id") or record.get("instance_id") or record.get("trajectory_id")


def load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    tasks = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        task_id = _task_id(task)
        if not task_id:
            raise ValueError("task record has no stable id")
        if task_id in tasks:
            raise ValueError(f"duplicate task id {task_id!r}")
        tasks[task_id] = task
    return tasks


def normalize_failure_record(
    record: dict[str, Any], task: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    failure_type = record.get("failure_type")
    if failure_type == "api_error":
        return None, "api_transport_failure"

    trajectory_id = str(record.get("trajectory_id") or _task_id(task) or "unknown")
    legal_steps: list[dict[str, Any]] = []
    error_events: list[dict[str, Any]] = []
    legal_indices: set[int] = set()
    error_indices: set[int] = set()

    for turn in record.get("turns") or []:
        action_index = int(turn.get("turn_index", len(legal_steps) + len(error_events))) + 1
        parsed = turn.get("parsed") or {}
        error_type = turn.get("execution_error_type")
        if error_type:
            event = dict(turn.get("error_event") or {})
            event.setdefault("action_index", action_index)
            event.setdefault("step_id", f"step_{action_index}")
            event.setdefault("error_type", error_type)
            event.setdefault("message", turn.get("execution_error") or error_type)
            if parsed.get("tool"):
                event.setdefault("attempted_tool", parsed["tool"])
                event.setdefault("attempted_arguments", parsed.get("arguments") or {})
            if action_index in error_indices or action_index in legal_indices:
                raise ValueError(f"duplicate action index {action_index} in {trajectory_id}")
            error_indices.add(action_index)
            error_events.append(event)
            continue

        tool = parsed.get("tool")
        arguments = parsed.get("arguments")
        if not tool or not isinstance(arguments, dict):
            continue
        if action_index in legal_indices or action_index in error_indices:
            raise ValueError(f"duplicate action index {action_index} in {trajectory_id}")
        legal_indices.add(action_index)
        output = turn.get("tool_output")
        if tool == "answer_from_context" and not isinstance(output, dict):
            output = {"final_answer": arguments.get("answer")}
        legal_steps.append({
            "step_id": f"step_{action_index}",
            "think": parsed.get("think") or "",
            "think_source": turn.get("think_source", "model"),
            "tool_call": {"tool": tool, "arguments": arguments},
            "tool_output": output or {},
            "feedback_recovery": bool(turn.get("feedback_recovery")),
            "recovered_from_error_type": turn.get("recovered_from_error_type"),
        })

    if not legal_steps and not error_events:
        return None, "no_semantic_actions"

    source = {
        "dataset": task.get("dataset", "bird-sql"),
        "split": task.get("split", "train"),
        "example_id": _task_id(task) or trajectory_id,
        "db_id": task.get("db_id") or record.get("db_id"),
        "db_path": task.get("db_path"),
        "external_knowledge": task.get("external_knowledge"),
        "gold_sql": task.get("gold_sql") or task.get("query") or record.get("gold_sql"),
    }
    if not source["db_path"] or not source["gold_sql"]:
        raise ValueError(f"task {trajectory_id} lacks db_path or gold_sql")

    normalized = {
        "trajectory_id": trajectory_id,
        "schema_version": "v4-external-rollout-failure-adapter",
        "source": source,
        "question": task.get("question") or record.get("question"),
        "difficulty": record.get("difficulty") or (task.get("metadata") or {}).get("difficulty"),
        "label_status": "rollout_failure",
        "steps": legal_steps,
        "rollout_generation": {
            "method": "external_llm_closed_loop",
            "outcome": record.get("outcome") or failure_type,
            "failure_type": failure_type,
            "error_events": error_events,
            "errors": len(error_events),
            "action_count": len(legal_indices) + len(error_indices),
            "error_actions_are_sft_targets": False,
            "adapter": "external_failure_adapter_v1",
        },
        "failure_audit": {
            "recorded_correct": bool(record.get("correct")),
            "recorded_legal": bool(record.get("legal")),
            "recorded_failure_type": failure_type,
            "raw_turn_count": len(record.get("turns") or []),
        },
    }
    return normalized, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tasks-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--excluded-output", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks_json)
    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    normalized = []
    excluded = []
    counts = collections.Counter()
    seen: set[str] = set()
    for record in records:
        trajectory_id = str(record.get("trajectory_id") or "")
        if not trajectory_id or trajectory_id in seen:
            raise ValueError(f"missing or duplicate trajectory id {trajectory_id!r}")
        seen.add(trajectory_id)
        task = tasks.get(trajectory_id)
        if not task:
            raise ValueError(f"no task metadata for {trajectory_id}")
        converted, reason = normalize_failure_record(record, task)
        if converted is None:
            excluded.append({
                "trajectory_id": trajectory_id,
                "reason": reason,
                "failure_type": record.get("failure_type"),
            })
            counts[f"excluded:{reason}"] += 1
            continue
        normalized.append(converted)
        counts[f"included:{record.get('failure_type')}"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized),
        encoding="utf-8",
    )
    args.excluded_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in excluded),
        encoding="utf-8",
    )
    summary = {
        "input_records": len(records),
        "normalized_semantic_failures": len(normalized),
        "excluded_nonsemantic_failures": len(excluded),
        "counts": dict(sorted(counts.items())),
        "output": str(args.output.resolve()),
        "excluded_output": str(args.excluded_output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
