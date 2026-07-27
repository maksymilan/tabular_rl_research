#!/usr/bin/env python3
"""Run one causal, human/agent-authored teacher episode against the live harness.

The interactive teacher sees only the same bounded model-visible context used by online rollout.
Gold SQL remains inside the terminal verifier and the persisted audit artifact.  Rejected actions
are recorded as error context but never become SFT targets.  A successful episode is written only
after an independent ``bird-set`` replay and the active structural/quality gates pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(HERE),
]

from bird_sft1_teacher import replay_success_trajectory  # noqa: E402
from executor import Harness  # noqa: E402
from generate_teacher_rollouts import (  # noqa: E402
    DATA_GENERATION_SUFFIX,
    PLAN_POLICY_OPTIONAL,
    ResidentPlanPolicyTracker,
    error_event,
    protocol_failure_type,
    retain_in_rolling_history,
    sft_export_eligible,
)
from protocol import (  # noqa: E402
    AdjacentActionGuard,
    PROTOCOL_VERSION,
    ProtocolError,
    get_system_prompt,
    parse_assistant_strict,
    protocol_hash,
    rolling_legal_history_messages,
    rolling_system_prompt,
    teacher_system_prompt,
    tool_error_message,
    tool_output_message,
    tool_schema_hash,
)
from rollout import (  # noqa: E402
    execute_tool,
    format_tool_error,
    new_ctx,
    overview,
    score,
    state_digest,
    task_db_path,
    task_gold_sql,
)
from select_verified_rollouts import quality_reason  # noqa: E402
from tool_schemes import (  # noqa: E402
    ATOMIC_ASSISTANT_CARRIER,
    ATOMIC_TOOL_SCHEME,
    TOOL_SCHEME_REGISTRY_VERSION,
)

TRAINING_ADMISSION = "diagnostic_only_pending_protocol_scale_gate"


def load_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"task input must contain a JSON array or JSONL rows: {path}")
    return value


def choose_task(
    tasks: list[dict[str, Any]],
    *,
    example_id: str | None,
    example_index: int | None,
) -> dict[str, Any]:
    matches = []
    for position, task in enumerate(tasks):
        identity = task.get("example_id") or task.get("instance_id")
        index = task.get("example_index", position)
        if example_id is not None and identity == example_id:
            matches.append(task)
        elif example_id is None and example_index is not None and int(index) == example_index:
            matches.append(task)
    if len(matches) != 1:
        target = example_id if example_id is not None else example_index
        raise ValueError(f"expected exactly one task for {target!r}, found {len(matches)}")
    return matches[0]


def emit(event: str, **payload: Any) -> None:
    print(
        json.dumps({"event": event, **payload}, ensure_ascii=False, default=str),
        flush=True,
    )


def read_assistant() -> str:
    line = sys.stdin.readline()
    if not line:
        raise EOFError("teacher input closed before terminal action")
    value = json.loads(line)
    if not isinstance(value, dict) or set(value) != {"assistant"}:
        raise ValueError('input must be one JSON object with exactly the key "assistant"')
    assistant = value["assistant"]
    if not isinstance(assistant, str) or not assistant.strip():
        raise ValueError('"assistant" must be a non-empty string')
    return assistant


def make_step(
    *,
    step_id: str,
    think: str,
    tool: str,
    arguments: dict[str, Any],
    output: dict[str, Any],
    state_before: dict[str, Any],
    state_after: dict[str, Any],
    last_error: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "think": think,
        "think_source": "codex_manual_teacher",
        "tool_call": {"tool": tool, "arguments": arguments},
        "tool_output": output,
        "environment_state_before": state_before,
        "environment_state": state_after,
        "last_tool_error_before": last_error,
        "feedback_recovery": bool(last_error),
        "recovered_from_error_type": (last_error or {}).get("error", {}).get("type"),
        "sft_target_eligible": True,
    }


def write_jsonl_atomic(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(row, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-json", type=Path, required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--example-id")
    selector.add_argument("--example-index", type=int)
    parser.add_argument("--trajectory-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-errors-per-type", type=int, default=3)
    parser.add_argument("--max-think-words", type=int, default=300)
    parser.add_argument("--table-output-rows", type=int, default=0)
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_errors_per_type < 1 or args.max_think_words < 1:
        parser.error("step, error, and think limits must be positive")
    for path in (args.trajectory_out, args.audit_out):
        if path.exists():
            parser.error(f"refusing to overwrite existing artifact: {path}")

    task = choose_task(
        load_tasks(args.tasks_json.resolve()),
        example_id=args.example_id,
        example_index=args.example_index,
    )
    task_path = task_db_path(task)
    gold_sql = task_gold_sql(task)
    if not gold_sql:
        parser.error("selected task has no hidden verifier SQL")

    harness = Harness(task_path)
    catalog = overview(harness)
    ctx = new_ctx(catalog)
    created: set[str] = set()
    legal_history: list[dict[str, str]] = []
    last_error: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    error_events: list[dict[str, Any]] = []
    error_counts: Counter[str] = Counter()
    adjacent_guard = AdjacentActionGuard()
    plan_tracker = ResidentPlanPolicyTracker(PLAN_POLICY_OPTIONAL)

    student_prompt = rolling_system_prompt(get_system_prompt(), compact=False)
    system_prompt = teacher_system_prompt(student_prompt) + DATA_GENERATION_SUFFIX
    prompt_audit = {
        "teacher_prompt_role": "teacher-generation-manual-agent",
        "teacher_canonical_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "student_runtime_prompt_sha256": hashlib.sha256(
            student_prompt.encode("utf-8")
        ).hexdigest(),
        "tool_schema_sha256": tool_schema_hash(),
    }
    started = time.monotonic()
    correct = False
    legal = False
    failure_type: str | None = None
    final_action_count = 0

    emit(
        "session_started",
        example_id=task.get("example_id") or task.get("instance_id"),
        example_index=task.get("example_index"),
        difficulty=(task.get("metadata") or {}).get("difficulty_proxy"),
        db_id=task.get("db_id"),
        hidden_fields=["gold_sql", "query"],
        input_format={
            "assistant": '<think>one non-empty reason</think>{"tool":"...","arguments":{...}}'
        },
    )

    try:
        for action_count in range(1, args.max_steps + 1):
            final_action_count = action_count
            state_before = ctx["environment"].snapshot()
            model_input = rolling_legal_history_messages(
                system_prompt,
                catalog,
                task["question"],
                state_before,
                last_error,
                task.get("external_knowledge") or None,
                legal_history,
                4,
            )
            emit("model_input", action_index=action_count, messages=model_input)
            try:
                text = read_assistant()
            except (EOFError, json.JSONDecodeError, ValueError) as exc:
                failure_type = "teacher_transport_error"
                emit("session_failed", failure_type=failure_type, message=str(exc))
                break

            turn: dict[str, Any] = {
                "turn_index": len(turns),
                "model_input": deepcopy(model_input),
                "model_output": text,
            }
            try:
                step_id = f"step_{action_count}"
                think, tool, arguments = parse_assistant_strict(
                    text,
                    adjacent_guard=adjacent_guard,
                    step_id=step_id,
                )
                turn["parsed"] = {
                    "think": think,
                    "tool": tool,
                    "arguments": arguments,
                }
                turn["feedback_recovery"] = bool(last_error)
                turn["recovered_from_error_type"] = (
                    (last_error or {}).get("error", {}).get("type")
                )
                plan_tracker.validate_before_execution(tool, arguments)

                if tool == "answer_from_context":
                    legal = True
                    correct, pred_sample, _ = score(
                        harness,
                        gold_sql,
                        arguments,
                        created,
                        denotation_comparison="bird-set",
                    )
                    final_output = {"final_answer": arguments.get("answer")}
                    steps.append(
                        make_step(
                            step_id=step_id,
                            think=think,
                            tool=tool,
                            arguments=arguments,
                            output=final_output,
                            state_before=state_before,
                            state_after=ctx["environment"].snapshot(),
                            last_error=last_error,
                        )
                    )
                    turn["answer_score"] = {
                        "correct": correct,
                        "pred_sample": pred_sample,
                        "gold_sample_redacted": True,
                    }
                    turns.append(turn)
                    if not correct:
                        failure_type = "wrong_answer"
                    emit(
                        "terminal_result",
                        correct=correct,
                        legal=True,
                        failure_type=failure_type,
                        pred_sample=pred_sample,
                    )
                    break

                output, table_name = execute_tool(
                    harness,
                    tool,
                    arguments,
                    ctx,
                    step_id,
                    table_output_rows=args.table_output_rows,
                )
                state_after = ctx["environment"].snapshot()
                steps.append(
                    make_step(
                        step_id=step_id,
                        think=think,
                        tool=tool,
                        arguments=arguments,
                        output=output,
                        state_before=state_before,
                        state_after=state_after,
                        last_error=last_error,
                    )
                )
                turn["tool_output"] = output
                turns.append(turn)
                adjacent_guard.mark_last("success")
                plan_tracker.record_success(tool, arguments)
                last_error = None
                if table_name:
                    created.add(table_name)
                observation = tool_output_message(step_id, output)
                if retain_in_rolling_history(tool, PLAN_POLICY_OPTIONAL):
                    legal_history.append(
                        {"assistant": text, "observation": observation}
                    )
                emit(
                    "tool_result",
                    action_index=action_count,
                    step_id=step_id,
                    status="success",
                    output=output,
                )
            except Exception as exc:  # Every rejected action becomes bounded feedback.
                state_after = ctx["environment"].snapshot()
                parsed = turn.get("parsed") or {}
                attempted_tool = parsed.get("tool") or getattr(
                    exc, "attempted_tool", None
                )
                attempted_arguments = (
                    parsed.get("arguments")
                    if parsed.get("tool")
                    else getattr(exc, "attempted_arguments", None)
                )
                error_type = (
                    protocol_failure_type(exc)
                    if isinstance(exc, ProtocolError)
                    else "execution_error"
                )
                adjacent_guard.mark_last("rejected")
                if (
                    error_type == "execution_error"
                    and state_digest(state_after) != state_digest(state_before)
                ):
                    error_type = "nonrecoverable_execution_error"
                message = format_tool_error(
                    exc,
                    harness,
                    attempted_tool,
                    attempted_arguments,
                )
                event = error_event(
                    action_count,
                    error_type,
                    message,
                    state_before,
                    state_after,
                    attempted_tool,
                    attempted_arguments,
                    getattr(exc, "code", type(exc).__name__),
                    getattr(exc, "details", None),
                )
                error_events.append(event)
                turn.update(
                    {
                        "execution_error": message,
                        "execution_error_type": error_type,
                        "error_event": event,
                    }
                )
                turns.append(turn)
                if error_type == "nonrecoverable_execution_error":
                    failure_type = error_type
                    emit(
                        "session_failed",
                        failure_type=failure_type,
                        message=message,
                    )
                    break
                error_counts[error_type] += 1
                if error_counts[error_type] >= args.max_errors_per_type:
                    failure_type = error_type
                    emit(
                        "session_failed",
                        failure_type=failure_type,
                        message=(
                            f"aborted after {error_counts[error_type]} "
                            f"{error_type} events: {message}"
                        ),
                    )
                    break
                error_observation = tool_error_message(
                    f"step_{action_count}",
                    error_type,
                    message,
                    error_code=getattr(exc, "code", type(exc).__name__),
                    details=getattr(exc, "details", None),
                    attempted_tool=attempted_tool,
                    attempted_arguments=attempted_arguments,
                )
                last_error = json.loads(error_observation)
                emit(
                    "tool_result",
                    action_index=action_count,
                    step_id=f"step_{action_count}",
                    status="error",
                    error=last_error,
                )
        else:
            failure_type = "max_steps"
            emit("session_failed", failure_type=failure_type, message="max_steps")

        identity = task.get("example_id") or task.get("instance_id")
        trajectory: dict[str, Any] | None = None
        replay_ok = False
        replay_error: str | None = None
        quality_gate: str | None = None
        if correct:
            format_gate_passed = sft_export_eligible(
                context_mode="rolling-legal-history",
                history_turns=4,
                rolling_prompt_variant="full",
                denotation_comparison="bird-set",
            )
            trajectory = {
                "tool_scheme": ATOMIC_TOOL_SCHEME,
                "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
                "assistant_carrier": ATOMIC_ASSISTANT_CARRIER,
                "protocol_version": PROTOCOL_VERSION,
                "protocol_hash": protocol_hash(system_prompt),
                "trajectory_id": f"bird_manual_teacher_{identity}",
                "schema_version": "bird-manual-teacher-v1",
                "source": {
                    "dataset": task.get("dataset", "bird-sql"),
                    "split": task.get("split", "train"),
                    "example_id": identity,
                    "example_index": task.get("example_index"),
                    "db_id": task.get("db_id"),
                    "db_path": task_path,
                    "external_knowledge": task.get("external_knowledge") or None,
                    "gold_sql": gold_sql,
                },
                "question": task["question"],
                "difficulty": (task.get("metadata") or {}).get(
                    "difficulty_proxy"
                ),
                "label_status": "verified",
                "initial_state": {"dataset_overview": catalog},
                "steps": steps,
                "rollout_generation": {
                    "tool_scheme": ATOMIC_TOOL_SCHEME,
                    "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
                    "assistant_carrier": ATOMIC_ASSISTANT_CARRIER,
                    "protocol_version": PROTOCOL_VERSION,
                    "method": "codex_manual_teacher_closed_loop",
                    "model": "codex-agent",
                    "protocol_hash": protocol_hash(system_prompt),
                    "prompt_contract": prompt_audit,
                    "context_mode": "rolling-legal-history",
                    "history_turns": 4,
                    "rolling_prompt_variant": "full",
                    "policy_prompt_variant": "canonical",
                    "plan_policy": PLAN_POLICY_OPTIONAL,
                    "denotation_comparison": "bird-set",
                    # The format/context gate passes independently of the current protocol's
                    # experiment-level scale gate.  Keep manual sessions diagnostic-only until
                    # that latter gate is explicitly promoted in the shared project contract.
                    "sft_format_gate_passed": format_gate_passed,
                    "sft_export_eligible": False,
                    "training_admission": TRAINING_ADMISSION,
                    "error_actions_are_sft_targets": False,
                    "errors": len(error_events),
                    "successful_tool_steps": len(steps) - 1,
                    "action_count": final_action_count,
                    "error_events": error_events,
                    "error_counts": dict(error_counts),
                    "outcome": (
                        "recovered_success" if error_events else "clean_success"
                    ),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "usage": {},
                },
            }
            replay_ok, replay_error = replay_success_trajectory(
                trajectory,
                denotation_comparison="bird-set",
            )
            quality_gate = quality_reason(
                trajectory,
                max_steps=args.max_steps,
                max_think_words=args.max_think_words,
            )
            if replay_ok and quality_gate is None:
                write_jsonl_atomic(args.trajectory_out.resolve(), trajectory)
            else:
                failure_type = "replay_mismatch" if not replay_ok else quality_gate

        audit = {
            "method": "codex_manual_teacher_closed_loop",
            "tasks_json": str(args.tasks_json.resolve()),
            "example_id": identity,
            "example_index": task.get("example_index"),
            "db_id": task.get("db_id"),
            "difficulty": (task.get("metadata") or {}).get("difficulty_proxy"),
            "gold_visible_to_teacher": False,
            "correct": correct,
            "legal": legal,
            "failure_type": failure_type,
            "actions": final_action_count,
            "legal_steps": len(steps),
            "error_events": error_events,
            "replay": {
                "denotation_comparison": "bird-set",
                "passed": replay_ok,
                "error": replay_error,
            },
            "quality_gate": {
                "passed": quality_gate is None if trajectory else False,
                "reason": quality_gate,
                "max_steps": args.max_steps,
                "max_think_words": args.max_think_words,
            },
            "training_admission": TRAINING_ADMISSION,
            "trajectory_written": bool(
                trajectory and replay_ok and quality_gate is None
            ),
            "trajectory_out": str(args.trajectory_out.resolve()),
            "prompt_contract": prompt_audit,
            "turns": turns,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        write_json_atomic(args.audit_out.resolve(), audit)
        emit(
            "session_complete",
            correct=correct,
            replay_passed=replay_ok,
            quality_gate_passed=quality_gate is None if trajectory else False,
            trajectory_written=audit["trajectory_written"],
            trajectory_out=str(args.trajectory_out.resolve()),
            audit_out=str(args.audit_out.resolve()),
        )
        return 0 if audit["trajectory_written"] else 1
    finally:
        harness.conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
