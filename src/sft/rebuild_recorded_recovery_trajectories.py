#!/usr/bin/env python3
"""Rebuild length-rejected recovery successes from their recorded causal turns.

This performs no model requests. It replays the selected student prefix and every recorded
teacher turn against a fresh harness, keeps rejected actions as audit-only error events, and emits
only bird-set-correct, deterministically replayable trajectories. Student prefix steps remain
context-only, and only teacher-authored legal continuation steps are eligible SFT targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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

from bird_sft1_teacher import compact, replay_success_trajectory  # noqa: E402
from executor import Harness  # noqa: E402
from generate_recovery_teacher_rollouts import (  # noqa: E402
    continuation_quality_reason,
    error_from_event,
    replay_prefix,
    source_record,
    task_identity,
)
from generate_teacher_rollouts import DATA_GENERATION_SUFFIX  # noqa: E402
from provider_adapter import (  # noqa: E402
    DEEPSEEK_CARRIER_JSON_OUTPUT,
    provider_system_prompt,
)
from protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    get_system_prompt,
    parse_assistant_strict,
    protocol_hash,
    rolling_system_prompt,
    teacher_system_prompt,
    tool_schema_hash,
)
from rollout import execute_tool, score, task_db_path, task_gold_sql  # noqa: E402
from select_verified_rollouts import quality_reason  # noqa: E402
from tool_schemes import (  # noqa: E402
    ATOMIC_ASSISTANT_CARRIER,
    ATOMIC_TOOL_SCHEME,
    TOOL_SCHEME_REGISTRY_VERSION,
)


REBUILDABLE_REASONS = {"teacher_step_limit", "teacher_think_limit"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_contract(model: str) -> tuple[str, dict[str, str]]:
    student_prompt = rolling_system_prompt(get_system_prompt(), compact=False)
    canonical_teacher_prompt = teacher_system_prompt(student_prompt) + DATA_GENERATION_SUFFIX
    provider_prompt = provider_system_prompt(
        model,
        canonical_teacher_prompt,
        carrier=DEEPSEEK_CARRIER_JSON_OUTPUT,
    )
    audit = {
        "teacher_prompt_role": "teacher-generation-recovery",
        "teacher_canonical_prompt_sha256": hashlib.sha256(
            canonical_teacher_prompt.encode("utf-8")
        ).hexdigest(),
        "teacher_provider_prompt_sha256": hashlib.sha256(
            provider_prompt.encode("utf-8")
        ).hexdigest(),
        "student_runtime_prompt_sha256": hashlib.sha256(
            student_prompt.encode("utf-8")
        ).hexdigest(),
        "tool_schema_sha256": tool_schema_hash(),
    }
    return provider_prompt, audit


def strict_protocol_reason(trajectory: dict[str, Any]) -> str | None:
    """Apply current protocol/reference checks without conflating context-only repetition."""
    for step in trajectory.get("steps") or []:
        reason = quality_reason(
            {"label_status": "verified", "steps": [step]},
            max_steps=None,
            max_think_words=None,
        )
        if reason:
            return reason
    target_only = [
        step
        for step in trajectory.get("steps") or []
        if step.get("sft_target_eligible", True) is not False
    ]
    return quality_reason(
        {"label_status": "verified", "steps": target_only},
        max_steps=None,
        max_think_words=None,
    )


def replay_recorded_teacher_error(
    harness: Harness,
    ctx: dict[str, Any],
    event: dict[str, Any],
) -> None:
    """Verify a rejected teacher action without requiring legacy diagnostic hashes.

    Some teacher error events hashed the prompt-rendering snapshot rather than the canonical
    resident snapshot. The two recorded hashes must still agree, and an execution error must still
    fail and preserve the freshly rebuilt canonical state. Final trajectory replay independently
    repeats the same execution-error check.
    """
    if event.get("state_before_hash") != event.get("state_after_hash"):
        raise ValueError("recorded rejected teacher action changed environment state")
    state_before = deepcopy(ctx["environment"].snapshot())
    if event.get("error_type") == "execution_error":
        tool = event.get("attempted_tool")
        arguments = event.get("attempted_arguments")
        if not isinstance(tool, str) or not isinstance(arguments, dict):
            raise ValueError("recorded execution error lacks attempted tool/arguments")
        try:
            execute_tool(
                harness,
                tool,
                arguments,
                ctx,
                str(event.get("step_id") or f"step_{event['action_index']}"),
            )
        except Exception:  # expected audited rejection
            pass
        else:
            raise ValueError("recorded execution error now succeeds")
    if compact(ctx["environment"].snapshot()) != compact(state_before):
        raise ValueError("rejected teacher action changed freshly rebuilt environment state")


def rebuild_one(
    *,
    task: dict[str, Any],
    sample: dict[str, Any],
    selected: dict[str, Any],
    attempt: dict[str, Any],
    model: str,
    provider_prompt: str,
    prompt_audit: dict[str, str],
    history_turns: int,
) -> dict[str, Any]:
    identity = task_identity(task)
    harness = Harness(task_db_path(task))
    try:
        seed = replay_prefix(harness, task, sample, selected["selected_candidate"])
        catalog = seed["catalog"]
        ctx = seed["ctx"]
        created = seed["created"]
        steps = seed["prefix_steps"]
        error_events = seed["error_events"]
        last_error = seed["last_error"]
        previous_action = int(seed["anchor_action_index"])
        teacher_legal_steps = 0
        terminal_seen = False
        pred_sample = gold_sample = None

        for turn in attempt.get("turns") or []:
            action_index = int(turn["turn_index"]) + 1
            if action_index != previous_action + 1:
                raise ValueError(
                    f"recorded continuation action {action_index} does not follow {previous_action}"
                )
            if terminal_seen:
                raise ValueError("recorded continuation has turns after the terminal action")
            previous_action = action_index
            state_before = ctx["environment"].snapshot()
            model_input_text = "\n".join(
                str(message.get("content") or "")
                for message in (turn.get("model_input") or [])
                if isinstance(message, dict)
            )
            if last_error and compact(last_error) not in model_input_text:
                raise ValueError(f"action {action_index}: LAST TOOL ERROR was not model-visible")
            gold_sql = task_gold_sql(task)
            if gold_sql and gold_sql in model_input_text:
                raise ValueError(f"action {action_index}: gold SQL leaked into teacher input")
            event = turn.get("error_event")
            if event is not None:
                if not isinstance(event, dict):
                    raise ValueError(f"action {action_index}: malformed error event")
                if int(event.get("action_index", -1)) != action_index:
                    raise ValueError(f"action {action_index}: error event index mismatch")
                replay_recorded_teacher_error(harness, ctx, event)
                error_events.append(deepcopy(event))
                last_error = error_from_event(event)
                continue

            parsed = turn.get("parsed")
            if not isinstance(parsed, dict):
                raise ValueError(f"action {action_index}: recorded legal turn lacks parsed action")
            think, tool, arguments = parse_assistant_strict(str(turn.get("model_output") or ""))
            if parsed != {"think": think, "tool": tool, "arguments": arguments}:
                raise ValueError(f"action {action_index}: parsed action differs from recorded output")

            step_id = f"step_{action_index}"
            if tool == "answer_from_context":
                correct, pred_sample, gold_sample = score(
                    harness,
                    task_gold_sql(task),
                    arguments,
                    created,
                    denotation_comparison="bird-set",
                )
                if not correct:
                    raise ValueError("recorded terminal action is not bird-set correct")
                output = {"final_answer": arguments.get("answer")}
                terminal_seen = True
            else:
                output, table_name = execute_tool(
                    harness,
                    tool,
                    arguments,
                    ctx,
                    step_id,
                )
                if compact(output) != compact(turn.get("tool_output")):
                    raise ValueError(f"action {action_index}: tool output differs on fresh replay")
                if table_name:
                    created.add(table_name)

            steps.append(
                {
                    "step_id": step_id,
                    "think": think,
                    "think_source": "external_teacher_recovery",
                    "tool_call": {"tool": tool, "arguments": arguments},
                    "tool_output": output,
                    "environment_state_before": state_before,
                    "environment_state": ctx["environment"].snapshot(),
                    "last_tool_error_before": last_error,
                    "feedback_recovery": bool(last_error),
                    "recovered_from_error_type": (
                        (last_error or {}).get("error") or {}
                    ).get("type"),
                    "sft_target_eligible": True,
                }
            )
            teacher_legal_steps += 1
            last_error = None

        if not terminal_seen:
            raise ValueError("recorded continuation did not terminate")

        trajectory = {
            "tool_scheme": ATOMIC_TOOL_SCHEME,
            "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
            "assistant_carrier": ATOMIC_ASSISTANT_CARRIER,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash(provider_prompt),
            "trajectory_id": f"bird_batch2_recovery_{identity}",
            "schema_version": "bird-batch2-recovery-v1",
            "source": {
                "dataset": task.get("dataset", "bird-sql"),
                "split": task.get("split", "train"),
                "example_id": identity,
                "example_index": int(task["example_index"]),
                "db_id": task["db_id"],
                "db_path": task_db_path(task),
                "external_knowledge": task.get("external_knowledge"),
                "gold_sql": task_gold_sql(task),
            },
            "question": task["question"],
            "difficulty": (task.get("metadata") or {}).get("difficulty_proxy"),
            "label_status": "verified",
            "initial_state": {"dataset_overview": catalog},
            "steps": steps,
            "rollout_generation": {
                "method": "external_teacher_selected_error_continuation",
                "model": model,
                "context_mode": "rolling-legal-history",
                "history_turns": history_turns,
                "rolling_prompt_variant": "full",
                "rolling_observation_style": "resident",
                "denotation_comparison": "bird-set",
                "error_actions_are_sft_targets": False,
                "student_prefix_steps_are_sft_targets": False,
                "action_count": previous_action,
                "prefix_anchor_action_index": seed["anchor_action_index"],
                "prefix_legal_steps": len(seed["prefix_steps"]),
                "teacher_legal_steps": teacher_legal_steps,
                "error_events": error_events,
                "outcome": "recovery_teacher_success",
                "prompt_contract": prompt_audit,
                "selection_audit": selected["selection_audit"],
                "usage": attempt.get("usage") or {},
                "rebuild_audit": {
                    "method": "fresh_harness_replay_of_recorded_causal_turns",
                    "original_post_replay_rejection": attempt["failure_type"],
                    "external_model_request": False,
                    "length_limits_applied": False,
                    "repeated_identical_teacher_calls_rejected": True,
                },
            },
        }
        replay_ok, replay_message = replay_success_trajectory(
            trajectory,
            denotation_comparison="bird-set",
        )
        if not replay_ok:
            raise ValueError(f"deterministic replay failed: {replay_message}")
        quality = continuation_quality_reason(
            trajectory,
            max_teacher_steps=None,
            max_think_words=None,
        )
        if quality:
            raise ValueError(quality)
        protocol_reason = strict_protocol_reason(trajectory)
        if protocol_reason:
            raise ValueError(protocol_reason)
        return {
            "example_id": identity,
            "example_index": int(task["example_index"]),
            "correct": True,
            "legal": True,
            "original_failure_type": attempt["failure_type"],
            "teacher_legal_steps": teacher_legal_steps,
            "pred_sample": pred_sample,
            "gold_sample": gold_sample,
            "trajectory": trajectory,
        }
    finally:
        harness.conn.close()


def build(
    *,
    selected_path: Path,
    tasks_path: Path,
    attempts_path: Path,
    out_path: Path,
    failures_path: Path,
    model: str,
    history_turns: int,
) -> dict[str, Any]:
    selected_rows = read_jsonl(selected_path)
    selected_by_id = {
        str(row["task"]["example_id"]): row
        for row in selected_rows
    }
    if len(selected_by_id) != len(selected_rows):
        raise ValueError("selected recovery anchors contain duplicate task ids")
    tasks = read_jsonl(tasks_path)
    tasks_by_id = {task_identity(task): task for task in tasks}
    if len(tasks_by_id) != len(tasks):
        raise ValueError("task input contains duplicate task ids")
    attempts = [
        row
        for row in read_jsonl(attempts_path)
        if row.get("failure_type") in REBUILDABLE_REASONS
    ]
    attempt_ids = [str(row.get("example_id") or "") for row in attempts]
    if not all(attempt_ids) or len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("rebuildable attempts must have unique non-empty example ids")

    provider_prompt, prompt_audit = prompt_contract(model)
    source_cache: dict[Path, dict[int, dict[str, Any]]] = {}
    success: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for attempt in attempts:
        identity = str(attempt["example_id"])
        try:
            selected = selected_by_id[identity]
            task = tasks_by_id[identity]
            _, sample = source_record(selected, source_cache)
            result = rebuild_one(
                task=task,
                sample=sample,
                selected=selected,
                attempt=attempt,
                model=model,
                provider_prompt=provider_prompt,
                prompt_audit=prompt_audit,
                history_turns=history_turns,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "example_id": identity,
                    "example_index": attempt.get("example_index"),
                    "original_failure_type": attempt.get("failure_type"),
                    "failure_type": "recorded_recovery_rebuild_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            success.append(result["trajectory"])

    write_jsonl_atomic(out_path, success)
    write_jsonl_atomic(failures_path, failures)
    counts = Counter(row["original_failure_type"] for row in failures)
    manifest = {
        "method": "fresh_harness_replay_of_recorded_causal_recovery_turns",
        "model": model,
        "inputs": {
            "selected_anchors": str(selected_path),
            "selected_anchors_sha256": sha256(selected_path),
            "tasks": str(tasks_path),
            "tasks_sha256": sha256(tasks_path),
            "attempts": str(attempts_path),
            "attempts_sha256": sha256(attempts_path),
        },
        "eligible_attempts": len(attempts),
        "verified_success": len(success),
        "failures": len(failures),
        "failure_source_reasons": dict(sorted(counts.items())),
        "quality_policy": {
            "bird_set_reverified": True,
            "deterministic_replay": True,
            "current_protocol_and_reference_safety": True,
            "length_limits": None,
            "repeated_identical_teacher_calls": "reject",
            "student_prefix_steps_are_sft_targets": False,
            "error_actions_are_sft_targets": False,
            "external_model_requests": 0,
        },
        "outputs": {
            "verified_success": str(out_path),
            "verified_success_sha256": sha256(out_path),
            "failures": str(failures_path),
            "failures_sha256": sha256(failures_path),
        },
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-anchors", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--failures-out", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--history-turns", type=int, default=4)
    args = parser.parse_args()
    if args.history_turns <= 0:
        parser.error("--history-turns must be positive")
    for path in (args.out, args.failures_out):
        if path.exists():
            parser.error(f"{path} exists; choose a fresh output path")
    manifest = build(
        selected_path=args.selected_anchors.resolve(),
        tasks_path=args.tasks.resolve(),
        attempts_path=args.attempts.resolve(),
        out_path=args.out.resolve(),
        failures_path=args.failures_out.resolve(),
        model=args.model,
        history_turns=args.history_turns,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
