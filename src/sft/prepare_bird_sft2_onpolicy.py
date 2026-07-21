#!/usr/bin/env python3
"""Replay pass@K student successes and select balanced teacher-fallback tasks."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src" / "eval"), str(ROOT / "src" / "harness"), str(HERE)]

from bird_sft1_teacher import replay_success_trajectory  # noqa: E402
from executor import Harness  # noqa: E402
from rollout import execute_tool, new_ctx, overview, score, state_digest  # noqa: E402
from select_verified_rollouts import quality_reason  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def task_key(record: dict[str, Any]) -> str:
    value = record.get("example_id") or record.get("instance_id")
    if value:
        return str(value)
    return f"example_index:{int(record['example_index'])}"


def tasks_by_index(tasks: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for task in tasks:
        index = int(task["example_index"])
        if index in result:
            raise ValueError(f"duplicate task example_index {index}")
        result[index] = task
    return result


def action_signature(trajectory: dict[str, Any]) -> str:
    actions = [step["tool_call"] for step in trajectory["steps"]]
    return json.dumps(actions, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _error_message(turn: dict[str, Any]) -> str:
    event = turn.get("error_event") or {}
    return str(turn.get("execution_error") or event.get("message") or turn.get("execution_error_type"))


def replay_sample(
    task: dict[str, Any],
    sample: dict[str, Any],
    *,
    require_correct: bool,
    trajectory_id: str,
) -> dict[str, Any]:
    """Rebuild authoritative steps from raw pass@K turns on a fresh harness."""
    harness = Harness(str(task["db_path"]))
    try:
        catalog = overview(harness)
        ctx = new_ctx(catalog)
        created: set[str] = set()
        steps: list[dict[str, Any]] = []
        error_events: list[dict[str, Any]] = []
        last_error: dict[str, Any] | None = None
        final_correct = False
        terminal_seen = False

        for ordinal, turn in enumerate(sample.get("turns") or [], 1):
            action_index = int(turn.get("turn_index", ordinal - 1)) + 1
            parsed = turn.get("parsed") or {}
            error_type = turn.get("execution_error_type")
            state_before = ctx["environment"].snapshot()
            if error_type:
                event = dict(turn.get("error_event") or {})
                event.setdefault("action_index", action_index)
                event.setdefault("step_id", f"step_{action_index}")
                event.setdefault("error_type", error_type)
                event.setdefault("message", _error_message(turn))
                event.setdefault("state_before_hash", state_digest(state_before))
                event.setdefault("state_after_hash", state_digest(state_before))
                if event["state_before_hash"] != event["state_after_hash"]:
                    raise ValueError(f"state-changing error in {trajectory_id} action {action_index}")
                error_events.append(event)
                # Historical v2i Harness._new increments its hidden handle counter before SQL
                # validation.  Re-execute audited execution errors so later materialized handle
                # names remain byte-identical (for example filter_002 after a failed filter_001).
                # The model-visible EnvironmentState must still remain unchanged.
                if error_type == "execution_error" and parsed.get("tool") and isinstance(parsed.get("arguments"), dict):
                    visible_before = state_digest(state_before)
                    try:
                        execute_tool(
                            harness,
                            parsed["tool"],
                            parsed["arguments"],
                            ctx,
                            f"step_{action_index}",
                        )
                    except Exception:  # expected historical execution failure
                        pass
                    else:
                        raise ValueError(
                            f"audited execution error now succeeds in {trajectory_id} action {action_index}"
                        )
                    if state_digest(ctx["environment"].snapshot()) != visible_before:
                        raise ValueError(
                            f"execution error changed visible state in {trajectory_id} action {action_index}"
                        )
                last_error = {
                    "step_id": f"step_{action_index}",
                    "status": "error",
                    "error": {"type": error_type, "message": event["message"]},
                }
                continue

            tool = parsed.get("tool")
            arguments = parsed.get("arguments")
            think = parsed.get("think")
            if not tool or not isinstance(arguments, dict) or not isinstance(think, str) or not think.strip():
                raise ValueError(f"missing parsed legal action in {trajectory_id} action {action_index}")
            step_id = f"step_{action_index}"
            recovery = bool(last_error)
            if recovery != bool(turn.get("feedback_recovery")):
                raise ValueError(f"feedback_recovery mismatch in {trajectory_id} action {action_index}")

            if tool == "answer_from_context":
                terminal_seen = True
                final_correct, _, _ = score(harness, str(task["gold_sql"]), arguments, created)
                output = {"final_answer": arguments.get("answer")}
                state_after = ctx["environment"].snapshot()
            else:
                output, table_name = execute_tool(harness, tool, arguments, ctx, step_id)
                if table_name:
                    created.add(table_name)
                state_after = ctx["environment"].snapshot()

            steps.append({
                "step_id": step_id,
                "think": think,
                "think_source": "sft1_student",
                "tool_call": {"tool": tool, "arguments": arguments},
                "tool_output": output,
                "environment_state_before": state_before,
                "environment_state": state_after,
                "last_tool_error_before": last_error,
                "feedback_recovery": recovery,
                "recovered_from_error_type": (last_error or {}).get("error", {}).get("type"),
            })
            last_error = None
            if terminal_seen:
                break

        if require_correct and (not terminal_seen or not final_correct):
            raise ValueError(f"recorded success does not replay correct: {trajectory_id}")
        difficulty = (task.get("metadata") or {}).get("difficulty_proxy") or "unknown"
        return {
            "trajectory_id": trajectory_id,
            "schema_version": "bird-sft2-student-onpolicy-v1",
            "source": {
                "dataset": task.get("dataset", "bird-sql"),
                "split": task.get("split", "train"),
                "example_id": task.get("example_id"),
                "example_index": int(task["example_index"]),
                "db_id": task["db_id"],
                "db_path": task["db_path"],
                "external_knowledge": task.get("external_knowledge"),
                "gold_sql": task["gold_sql"],
            },
            "question": task["question"],
            "difficulty": difficulty,
            "label_status": "verified" if final_correct else "rollout_failure",
            "initial_state": {"dataset_overview": catalog},
            "steps": steps,
            "rollout_generation": {
                "method": "sft1_student_passk",
                "model": "qwen2.5-7b-bird-sft1-grounded-651-epoch1",
                "context_mode": "rolling-legal-history",
                "history_turns": 4,
                "rolling_prompt_variant": "full",
                "rolling_observation_style": "resident",
                "error_actions_are_sft_targets": False,
                "action_count": len(sample.get("turns") or []),
                "error_events": error_events,
                "outcome": sample.get("outcome") or sample.get("failure_type"),
                "sample_index": int(sample.get("sample_index", 0)),
            },
        }
    finally:
        harness.conn.close()


def preferred_failed_sample(record: dict[str, Any]) -> dict[str, Any]:
    samples = [sample for sample in record.get("samples") or [] if not sample.get("correct")]
    if not samples:
        raise ValueError(f"no failed sample for example {record.get('example_index')}")
    order = {
        "wrong_answer": 0,
        "execution_error": 1,
        "argument_validation_error": 2,
        "protocol_error": 3,
        "max_steps": 4,
        "context_overflow": 5,
        "api_error": 6,
    }
    return min(samples, key=lambda sample: (
        order.get(str(sample.get("failure_type")), 99),
        int(sample.get("errors") or 0),
        int(sample.get("steps") or 10**6),
        int(sample.get("sample_index") or 0),
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--passk-all", type=Path, action="append", required=True)
    parser.add_argument("--raw-success-out", type=Path, required=True)
    parser.add_argument("--accepted-success-out", type=Path, required=True)
    parser.add_argument("--rejected-out", type=Path, required=True)
    parser.add_argument("--fallback-tasks-out", type=Path, required=True)
    parser.add_argument("--fallback-index-out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fallback-per-difficulty", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-think-words", type=int, default=300)
    args = parser.parse_args()

    tasks = read_jsonl(args.tasks)
    task_map = tasks_by_index(tasks)
    rollout_records: list[tuple[Path, dict[str, Any]]] = []
    for path in args.passk_all:
        rollout_records.extend((path, row) for row in read_jsonl(path))

    raw: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    seen_actions: set[tuple[int, str]] = set()
    rejection_counts: collections.Counter[str] = collections.Counter()
    replay_errors: collections.Counter[str] = collections.Counter()

    for source_path, record in rollout_records:
        task = task_map[int(record["example_index"])]
        for sample in record.get("samples") or []:
            if not sample.get("correct"):
                continue
            trajectory_id = f"bird_sft2_student_{task['example_index']}_sample_{sample['sample_index']}"
            try:
                trajectory = replay_sample(
                    task, sample, require_correct=True, trajectory_id=trajectory_id,
                )
                replay_ok, replay_error = replay_success_trajectory(trajectory)
                if not replay_ok:
                    raise ValueError(str(replay_error))
            except Exception as exc:  # noqa: BLE001
                replay_errors[type(exc).__name__] += 1
                rejected.append({
                    "trajectory_id": trajectory_id,
                    "reason": "replay_error",
                    "detail": str(exc),
                    "source": str(source_path),
                })
                continue
            raw.append(trajectory)
            dedupe_key = (int(task["example_index"]), action_signature(trajectory))
            if dedupe_key in seen_actions:
                rejection_counts["exact_action_duplicate"] += 1
                rejected.append({"trajectory_id": trajectory_id, "reason": "exact_action_duplicate"})
                continue
            seen_actions.add(dedupe_key)
            reason = quality_reason(
                trajectory, max_steps=args.max_steps, max_think_words=args.max_think_words,
            )
            if reason:
                rejection_counts[reason] += 1
                rejected.append({"trajectory_id": trajectory_id, "reason": reason})
                continue
            accepted.append(trajectory)

    failures_by_difficulty: dict[str, list[tuple[dict[str, Any], dict[str, Any], Path]]] = (
        collections.defaultdict(list)
    )
    for source_path, record in rollout_records:
        if record.get("correct"):
            continue
        task = task_map[int(record["example_index"])]
        difficulty = (task.get("metadata") or {}).get("difficulty_proxy") or "unknown"
        failures_by_difficulty[difficulty].append((task, record, source_path))

    rng = random.Random(args.seed)
    fallback_tasks: list[dict[str, Any]] = []
    fallback_index: list[dict[str, Any]] = []
    for difficulty in ("easy", "medium", "hard"):
        candidates = sorted(failures_by_difficulty.get(difficulty, []), key=lambda item: int(item[0]["example_index"]))
        rng.shuffle(candidates)
        selected = candidates[: args.fallback_per_difficulty]
        for task, record, source_path in selected:
            sample = preferred_failed_sample(record)
            fallback_tasks.append(task)
            fallback_index.append({
                "example_id": task.get("example_id"),
                "example_index": int(task["example_index"]),
                "difficulty": difficulty,
                "db_id": task["db_id"],
                "student_passk_source": str(source_path),
                "student_sample_index": int(sample.get("sample_index", 0)),
                "student_failure_type": sample.get("failure_type"),
                "student_legal": bool(sample.get("legal")),
            })

    write_jsonl(args.raw_success_out, raw)
    write_jsonl(args.accepted_success_out, accepted)
    write_jsonl(args.rejected_out, rejected)
    write_jsonl(args.fallback_tasks_out, fallback_tasks)
    write_jsonl(args.fallback_index_out, fallback_index)
    manifest = {
        "protocol": "draft/bird_sft2_onpolicy_data_protocol.md",
        "tasks": str(args.tasks),
        "passk_inputs": [str(path) for path in args.passk_all],
        "student_correct_samples": len(raw),
        "student_accepted_episodes": len(accepted),
        "student_accepted_targets": sum(len(row["steps"]) for row in accepted),
        "student_accepted_recovery_targets": sum(
            bool(step.get("feedback_recovery")) for row in accepted for step in row["steps"]
        ),
        "student_rejected": dict(rejection_counts),
        "student_replay_errors": dict(replay_errors),
        "teacher_fallback_tasks": len(fallback_tasks),
        "teacher_fallback_by_difficulty": dict(collections.Counter(
            row["difficulty"] for row in fallback_index
        )),
        "teacher_policy": {
            "primary": "deepseek-v4-flash",
            "fallback": "deepseek-v4-pro only after matched pilot quality gate",
            "eligibility": "no correct SFT-1 student sample after pass@4",
        },
        "outputs": {
            "raw_success": str(args.raw_success_out),
            "accepted_success": str(args.accepted_success_out),
            "rejected": str(args.rejected_out),
            "fallback_tasks": str(args.fallback_tasks_out),
            "fallback_index": str(args.fallback_index_out),
        },
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
