#!/usr/bin/env python3
"""Select causal recovery anchors from persistent adjacent-repeat failures.

This selector is intentionally narrower than the general batch-2 failure partition:

* the student sample must be terminally incorrect;
* provider/transport/context and tool-timeout failures are excluded;
* at least ``min_repeat_errors`` state-preserving
  ``adjacent_identical_action`` rejections must occur in the same sample; and
* the recovery anchor is the first such rejection, whose attempted action must exactly match the
  immediately preceding structured student action, whether that preceding action executed or was
  itself rejected after parsing.

The resulting rows use the input contract consumed by
``generate_recovery_teacher_rollouts.py``.  Gold verifier fields never leave the task file.  The
student legal prefix is context-only, and the rejected repeated action is audit/error context only.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from protocol import PROTOCOL_VERSION
from tool_modules.registry import ATOMIC_TOOL_SCHEME


INFRASTRUCTURE_FAILURE_TYPES = {
    "api_error",
    "provider_carrier_error",
    "transport_error",
    "context_overflow",
}
ADJACENT_REPEAT_ERROR_TYPE = "no_progress_error"
ADJACENT_REPEAT_ERROR_CODE = "adjacent_identical_action"
TRAINING_ADMISSION = "diagnostic_only_pending_protocol_scale_gate"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_identity(task: dict[str, Any]) -> str:
    value = task.get("example_id") or task.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"task {task.get('example_index')}: missing example_id")
    return value


def visible_task_context(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": task_identity(task),
        "example_index": int(task["example_index"]),
        "db_id": task["db_id"],
        "question": task["question"],
        "external_knowledge": task.get("external_knowledge"),
        "difficulty": (task.get("metadata") or {}).get("difficulty_proxy"),
    }


def canonical_action(tool: Any, arguments: Any) -> str:
    if not isinstance(tool, str) or not isinstance(arguments, dict):
        raise ValueError("adjacent-repeat event lacks a structured attempted action")
    return json.dumps(
        {"tool": tool, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def error_from_event(event: dict[str, Any]) -> dict[str, Any]:
    action_index = int(event["action_index"])
    error = {
        "type": str(event["error_type"]),
        "message": str(event.get("message") or ""),
    }
    if event.get("error_code"):
        error["code"] = str(event["error_code"])
    if event.get("details") is not None:
        error["details"] = event["details"]
    result: dict[str, Any] = {
        "step_id": str(event.get("step_id") or f"step_{action_index}"),
        "status": "error",
        "error": error,
    }
    if event.get("attempted_tool"):
        result["attempted_action"] = {
            "tool": event["attempted_tool"],
            "arguments": event.get("attempted_arguments") or {},
        }
    return result


def legal_prefix_before(
    turns: list[dict[str, Any]],
    action_index: int,
) -> list[dict[str, Any]]:
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


def repeat_events(sample: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for event in sample.get("error_events") or []:
        if (
            event.get("error_type") == ADJACENT_REPEAT_ERROR_TYPE
            and event.get("error_code") == ADJACENT_REPEAT_ERROR_CODE
            and event.get("state_before_hash")
            and event.get("state_before_hash") == event.get("state_after_hash")
        ):
            events.append(dict(event))
    return sorted(events, key=lambda event: int(event["action_index"]))


def has_infrastructure_failure(sample: dict[str, Any]) -> bool:
    if sample.get("failure_type") in INFRASTRUCTURE_FAILURE_TYPES:
        return True
    for event in sample.get("error_events") or []:
        code = str(event.get("error_code") or "")
        message = str(event.get("message") or "").lower()
        if (
            code == "ToolExecutionTimeoutError"
            or "transport error" in message
            or "connection error" in message
        ):
            return True
    return False


def turn_by_action(sample: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(turn.get("turn_index", ordinal - 1)) + 1: turn
        for ordinal, turn in enumerate(sample.get("turns") or [], start=1)
    }


def structured_action_at(
    sample: dict[str, Any],
    action_index: int,
    turn: dict[str, Any],
) -> tuple[Any, Any]:
    parsed = turn.get("parsed")
    if isinstance(parsed, dict):
        return parsed.get("tool"), parsed.get("arguments")
    matches = [
        event
        for event in sample.get("error_events") or []
        if int(event.get("action_index", -1)) == action_index
    ]
    if len(matches) == 1:
        event = matches[0]
        return event.get("attempted_tool"), event.get("attempted_arguments")
    raise ValueError(
        f"repeat anchor action {action_index + 1}: preceding action "
        f"{action_index} lacks one structured parsed or rejected action"
    )


def validate_first_repeat(
    sample: dict[str, Any],
    event: dict[str, Any],
) -> None:
    action_index = int(event["action_index"])
    if action_index <= 1:
        raise ValueError("an adjacent-repeat anchor cannot be the first action")
    turns = turn_by_action(sample)
    previous = turns.get(action_index - 1)
    current = turns.get(action_index)
    if previous is None or current is None:
        raise ValueError(f"repeat anchor action {action_index}: source turns are incomplete")
    previous_tool, previous_arguments = structured_action_at(
        sample,
        action_index - 1,
        previous,
    )
    if canonical_action(
        previous_tool,
        previous_arguments,
    ) != canonical_action(
        event.get("attempted_tool"),
        event.get("attempted_arguments"),
    ):
        raise ValueError(
            f"repeat anchor action {action_index}: attempted action does not exactly match "
            "the immediately preceding parsed action"
        )
    if current.get("execution_error_type") != ADJACENT_REPEAT_ERROR_TYPE:
        raise ValueError(
            f"repeat anchor action {action_index}: source turn lacks no_progress_error"
        )
    details = event.get("details") or {}
    previous_step_id = details.get("previous_step_id")
    if previous_step_id is not None and previous_step_id != f"step_{action_index - 1}":
        raise ValueError(
            f"repeat anchor action {action_index}: previous_step_id does not name the "
            "immediately preceding action"
        )


def choose_repeated_failure_sample(
    record: dict[str, Any],
    *,
    min_repeat_errors: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    candidates: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for sample in record.get("samples") or []:
        if sample.get("correct") or has_infrastructure_failure(sample):
            continue
        events = repeat_events(sample)
        if len(events) >= min_repeat_errors:
            validate_first_repeat(sample, events[0])
            candidates.append((sample, events))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            -len(item[1]),
            -int(item[0].get("steps") or 0),
            int(item[0].get("sample_index") or 0),
        ),
    )


def selected_anchor(
    task: dict[str, Any],
    record: dict[str, Any],
    rollout_source: Path,
    *,
    min_repeat_errors: int,
) -> dict[str, Any] | None:
    chosen = choose_repeated_failure_sample(
        record,
        min_repeat_errors=min_repeat_errors,
    )
    if chosen is None:
        return None
    sample, events = chosen
    event = events[0]
    action_index = int(event["action_index"])
    identity = task_identity(task)
    selected_candidate = {
        "candidate_id": (
            f"{identity}:sample_{int(sample.get('sample_index') or 0)}:"
            f"adjacent_repeat_action_{action_index}"
        ),
        "anchor_action_index": action_index,
        "legal_prefix": legal_prefix_before(sample.get("turns") or [], action_index),
        "last_tool_error": error_from_event(event),
        "attempted_action": {
            "tool": event.get("attempted_tool"),
            "arguments": event.get("attempted_arguments") or {},
        },
        "state_before_hash": event["state_before_hash"],
        "state_after_hash": event["state_after_hash"],
    }
    package = {
        "task": visible_task_context(task),
        "student_rollout": {
            "source": str(rollout_source),
            "sample_index": int(sample.get("sample_index") or 0),
            "failure_type": sample.get("failure_type"),
            "legal": bool(sample.get("legal")),
            "steps": int(sample.get("steps") or 0),
            "errors": int(sample.get("errors") or 0),
            "adjacent_repeat_errors": len(events),
        },
        "selected_candidate": selected_candidate,
        "selection_audit": {
            "model": "deterministic_adjacent_repeat_selector",
            "decision": "select_candidate",
            "rationale": (
                f"sample contains {len(events)} verified state-preserving adjacent identical "
                f"action rejections; continue from the first rejection at action {action_index}"
            ),
            "usage": {},
            "rationale_is_teacher_visible": False,
            "rationale_is_sft_target": False,
            "gold_sql_visible": False,
            "selection_is_deterministic": True,
        },
    }
    gold_sql = str(task.get("gold_sql") or task.get("query") or "")
    if gold_sql and gold_sql in json.dumps(package, ensure_ascii=False):
        raise ValueError(f"{identity}: gold SQL leaked into selected recovery package")
    return package


def rollout_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"rollout manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("tool_scheme") != ATOMIC_TOOL_SCHEME:
        raise ValueError(f"{manifest_path}: rollout tool scheme is not atomic")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"{manifest_path}: expected protocol {PROTOCOL_VERSION}, "
            f"got {manifest.get('protocol_version')}"
        )
    if manifest.get("denotation_comparison") != "bird-set":
        raise ValueError(f"{manifest_path}: denotation comparison is not bird-set")
    if manifest.get("sample_detail") != "full":
        raise ValueError(f"{manifest_path}: full sample detail is required for recovery")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "model": manifest.get("model"),
        "protocol_version": manifest.get("protocol_version"),
        "protocol_hash": manifest.get("protocol_hash"),
        "tool_scheme": manifest.get("tool_scheme"),
        "assistant_carrier": manifest.get("assistant_carrier"),
        "denotation_comparison": manifest.get("denotation_comparison"),
        "n_samples": manifest.get("n_samples"),
        "temperature": manifest.get("temperature"),
        "top_p": manifest.get("top_p"),
    }


def pilot_selection(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    if size <= 0:
        return []
    ordered = sorted(
        rows,
        key=lambda row: (
            -int(row["student_rollout"]["adjacent_repeat_errors"]),
            {"hard": 0, "medium": 1, "easy": 2}.get(
                row["task"].get("difficulty"),
                3,
            ),
            int(row["task"]["example_index"]),
        ),
    )
    diverse: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    seen_databases: set[str] = set()
    for row in ordered:
        database = str(row["task"]["db_id"])
        if database in seen_databases:
            deferred.append(row)
        else:
            diverse.append(row)
            seen_databases.add(database)
    return (diverse + deferred)[:size]


def build(
    tasks_path: Path,
    rollout_paths: list[Path],
    selected_out: Path,
    pilot_out: Path,
    manifest_path: Path,
    *,
    min_repeat_errors: int,
    pilot_size: int,
) -> dict[str, Any]:
    if min_repeat_errors < 1:
        raise ValueError("min_repeat_errors must be positive")
    tasks = read_jsonl(tasks_path)
    tasks_by_index = {int(task["example_index"]): task for task in tasks}
    if len(tasks_by_index) != len(tasks):
        raise ValueError("task cohort contains duplicate example_index values")
    identities = [task_identity(task) for task in tasks]
    if len(set(identities)) != len(identities):
        raise ValueError("task cohort contains duplicate example ids")

    records: list[tuple[Path, dict[str, Any]]] = []
    input_manifests = []
    for path in rollout_paths:
        input_manifests.append(rollout_manifest(path))
        records.extend((path, row) for row in read_jsonl(path))
    if len(records) != len(tasks):
        raise ValueError(f"rollout records {len(records)} != tasks {len(tasks)}")

    seen_indices: set[int] = set()
    selected: list[dict[str, Any]] = []
    skipped = Counter()
    for source, record in records:
        index = int(record["example_index"])
        if index in seen_indices:
            raise ValueError(f"duplicate rollout example_index {index}")
        seen_indices.add(index)
        task = tasks_by_index.get(index)
        if task is None:
            raise ValueError(f"rollout example_index {index} is outside the task cohort")
        package = selected_anchor(
            task,
            record,
            source,
            min_repeat_errors=min_repeat_errors,
        )
        if package is None:
            skipped["not_persistent_adjacent_repeat_failure"] += 1
        else:
            selected.append(package)
    if seen_indices != set(tasks_by_index):
        missing = sorted(set(tasks_by_index) - seen_indices)
        raise ValueError(f"rollout cohort is incomplete; first missing indices: {missing[:5]}")

    selected.sort(key=lambda row: int(row["task"]["example_index"]))
    pilot = pilot_selection(selected, pilot_size)
    write_jsonl_atomic(selected_out, selected)
    write_jsonl_atomic(pilot_out, pilot)
    manifest = {
        "method": "persistent_adjacent_repeat_recovery_selection",
        "training_admission": TRAINING_ADMISSION,
        "tasks": {
            "path": str(tasks_path),
            "sha256": sha256_file(tasks_path),
            "records": len(tasks),
        },
        "rollout_inputs": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "manifest": input_manifest,
            }
            for path, input_manifest in zip(rollout_paths, input_manifests)
        ],
        "selection_policy": {
            "minimum_adjacent_repeat_errors_in_one_failed_sample": min_repeat_errors,
            "error_type": ADJACENT_REPEAT_ERROR_TYPE,
            "error_code": ADJACENT_REPEAT_ERROR_CODE,
            "anchor": "first_verified_adjacent_repeat_rejection",
            "require_state_preservation": True,
            "require_exact_match_to_immediately_preceding_structured_action": True,
            "exclude_correct_samples": True,
            "exclude_infrastructure_failures": True,
            "student_prefix_steps_are_sft_targets": False,
            "rejected_repeat_action_is_sft_target": False,
            "teacher_receives_future_student_suffix": False,
            "gold_sql_visible_to_selector_or_teacher": False,
        },
        "counts": {
            "tasks": len(tasks),
            "selected_repeated_failures": len(selected),
            "diagnostic_pilot": len(pilot),
            "skipped": dict(sorted(skipped.items())),
            "selected_by_difficulty": dict(
                sorted(Counter(row["task"].get("difficulty") for row in selected).items())
            ),
            "selected_by_database": len({row["task"]["db_id"] for row in selected}),
            "repeat_events": sum(
                int(row["student_rollout"]["adjacent_repeat_errors"])
                for row in selected
            ),
        },
        "outputs": {
            "selected_all": str(selected_out),
            "diagnostic_pilot": str(pilot_out),
        },
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--rollout-all", type=Path, action="append", required=True)
    parser.add_argument("--selected-out", type=Path, required=True)
    parser.add_argument("--pilot-out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--min-repeat-errors", type=int, default=3)
    parser.add_argument("--pilot-size", type=int, default=20)
    args = parser.parse_args()
    manifest = build(
        args.tasks.resolve(),
        [path.resolve() for path in args.rollout_all],
        args.selected_out.resolve(),
        args.pilot_out.resolve(),
        args.manifest.resolve(),
        min_repeat_errors=args.min_repeat_errors,
        pilot_size=args.pilot_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
