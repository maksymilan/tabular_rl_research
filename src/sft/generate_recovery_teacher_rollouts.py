#!/usr/bin/env python3
"""Continue selected failed student prefixes with an external teacher.

The harness replays every pre-anchor action from scratch, including state-preserving execution
errors needed for deterministic handle allocation. Student legal prefix steps are stored as
context-only. The selected rejected action is audit/error context only. Only teacher-authored legal
continuation steps are eligible SFT targets, and a trajectory is emitted only after fresh bird-set
terminal verification and deterministic replay.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    ProviderCarrierError,
    add_usage,
    chat_with_retries,
    compact_json,
    error_event,
    protocol_failure_type,
)
from provider_adapter import (  # noqa: E402
    DEEPSEEK_CARRIER_JSON_OUTPUT,
    adapt_provider_response,
    provider_default_max_tokens,
    provider_rejection_message,
    provider_request_messages,
    provider_request_options,
    provider_system_prompt,
)
from provider_client import load_api_config  # noqa: E402
from protocol import (  # noqa: E402
    AdjacentActionGuard,
    PROTOCOL_VERSION,
    ProtocolError,
    assistant_message,
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
    ContextOverflowError,
    execute_tool,
    format_tool_error,
    new_ctx,
    overview,
    score,
    state_digest,
    task_db_path,
    task_gold_sql,
)
from tool_schemes import (  # noqa: E402
    ATOMIC_ASSISTANT_CARRIER,
    ATOMIC_TOOL_SCHEME,
    TOOL_SCHEME_REGISTRY_VERSION,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def task_identity(task: dict[str, Any]) -> str:
    value = task.get("example_id") or task.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError("task is missing example_id")
    return value


def error_from_event(event: dict[str, Any]) -> dict[str, Any]:
    action_index = int(event["action_index"])
    error = {
        "type": str(event["error_type"]),
        "message": str(event.get("message") or ""),
    }
    if event.get("error_code"):
        error["code"] = str(event["error_code"])
    if event.get("details"):
        error["details"] = deepcopy(event["details"])
    result = {
        "step_id": str(event.get("step_id") or f"step_{action_index}"),
        "status": "error",
        "error": error,
    }
    if event.get("attempted_tool"):
        result["attempted_action"] = {
            "tool": event["attempted_tool"],
            "arguments": deepcopy(event.get("attempted_arguments") or {}),
        }
    return result


def source_record(
    selected: dict[str, Any],
    cache: dict[Path, dict[int, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(selected["student_rollout"]["source"]).resolve()
    if path not in cache:
        cache[path] = {
            int(row["example_index"]): row
            for row in read_jsonl(path)
        }
    index = int(selected["task"]["example_index"])
    record = cache[path].get(index)
    if record is None:
        raise ValueError(f"{path}: missing rollout example_index {index}")
    sample_index = int(selected["student_rollout"]["sample_index"])
    sample = next(
        (
            item
            for item in record.get("samples") or []
            if int(item.get("sample_index", 0)) == sample_index
        ),
        None,
    )
    if sample is None:
        raise ValueError(f"{path}: missing sample_index {sample_index} for example {index}")
    return record, sample


def matching_error_event(
    sample: dict[str, Any],
    action_index: int,
) -> dict[str, Any]:
    matches = [
        event
        for event in sample.get("error_events") or []
        if int(event.get("action_index", -1)) == action_index
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one error event at action {action_index}, got {len(matches)}")
    event = dict(matches[0])
    if event.get("state_before_hash") != event.get("state_after_hash"):
        raise ValueError(f"selected error action {action_index} changed environment state")
    return event


def replay_error(
    harness: Harness,
    ctx: dict[str, Any],
    event: dict[str, Any],
) -> None:
    """Reproduce execution errors when possible and always require visible state preservation."""
    state_before = ctx["environment"].snapshot()
    before_hash = state_digest(state_before)
    if before_hash != event.get("state_before_hash"):
        raise ValueError(
            f"error action {event.get('action_index')}: replay state hash differs before error"
        )
    if event.get("error_type") == "execution_error":
        tool = event.get("attempted_tool")
        arguments = event.get("attempted_arguments")
        if not isinstance(tool, str) or not isinstance(arguments, dict):
            raise ValueError("execution recovery anchor lacks attempted tool/arguments")
        try:
            execute_tool(
                harness,
                tool,
                arguments,
                ctx,
                str(event.get("step_id") or f"step_{event['action_index']}"),
            )
        except Exception:  # expected audited failure
            pass
        else:
            raise ValueError(
                f"execution error action {event.get('action_index')} now succeeds"
            )
    after_hash = state_digest(ctx["environment"].snapshot())
    if after_hash != before_hash or after_hash != event.get("state_after_hash"):
        raise ValueError(
            f"error action {event.get('action_index')}: visible state changed during replay"
        )


def replay_prefix(
    harness: Harness,
    task: dict[str, Any],
    sample: dict[str, Any],
    selected_candidate: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild the exact legal prefix and selected LAST TOOL ERROR from raw rollout turns."""
    catalog = overview(harness)
    ctx = new_ctx(catalog)
    created: set[str] = set()
    legal_history: list[dict[str, str]] = []
    prefix_steps: list[dict[str, Any]] = []
    replayed_errors: list[dict[str, Any]] = []
    last_error: dict[str, Any] | None = None
    anchor_index = int(selected_candidate["anchor_action_index"])
    selected_event = matching_error_event(sample, anchor_index)
    selected_error = selected_candidate["last_tool_error"]
    if error_from_event(selected_event) != selected_error:
        raise ValueError("selected recovery error does not match the source rollout event")

    turns_by_action = {
        int(turn.get("turn_index", ordinal - 1)) + 1: turn
        for ordinal, turn in enumerate(sample.get("turns") or [], start=1)
    }
    for action_index in range(1, anchor_index + 1):
        turn = turns_by_action.get(action_index)
        if turn is None:
            raise ValueError(f"source rollout is missing action {action_index}")
        event_type = turn.get("execution_error_type")
        if event_type:
            event = matching_error_event(sample, action_index)
            replay_error(harness, ctx, event)
            replayed_errors.append(event)
            last_error = error_from_event(event)
            if action_index == anchor_index:
                break
            continue
        if action_index == anchor_index:
            raise ValueError("selected anchor action is not an error turn")

        parsed = turn.get("parsed")
        if not isinstance(parsed, dict):
            raise ValueError(f"action {action_index}: legal prefix turn lacks parsed action")
        think = parsed.get("think")
        tool = parsed.get("tool")
        arguments = parsed.get("arguments")
        if (
            not isinstance(think, str)
            or not think.strip()
            or not isinstance(tool, str)
            or not isinstance(arguments, dict)
        ):
            raise ValueError(f"action {action_index}: malformed legal prefix action")
        if tool == "answer_from_context":
            raise ValueError("terminal action appears before selected recovery anchor")
        state_before = ctx["environment"].snapshot()
        output, table_name = execute_tool(
            harness,
            tool,
            arguments,
            ctx,
            f"step_{action_index}",
        )
        state_after = ctx["environment"].snapshot()
        step = {
            "step_id": f"step_{action_index}",
            "think": think,
            "think_source": "student_context_only",
            "tool_call": {"tool": tool, "arguments": arguments},
            "tool_output": output,
            "environment_state_before": state_before,
            "environment_state": state_after,
            "last_tool_error_before": last_error,
            "feedback_recovery": bool(last_error),
            "recovered_from_error_type": (last_error or {}).get("error", {}).get("type"),
            "sft_target_eligible": False,
        }
        prefix_steps.append(step)
        legal_history.append(
            {
                "assistant": assistant_message(think, tool, arguments),
                "observation": tool_output_message(f"step_{action_index}", output),
            }
        )
        if table_name:
            created.add(table_name)
        last_error = None

    if last_error != selected_error:
        raise ValueError("selected LAST TOOL ERROR was not resident after prefix replay")
    return {
        "catalog": catalog,
        "ctx": ctx,
        "created": created,
        "legal_history": legal_history,
        "prefix_steps": prefix_steps,
        "error_events": replayed_errors,
        "last_error": last_error,
        "anchor_action_index": anchor_index,
    }


def continuation_quality_reason(
    trajectory: dict[str, Any],
    *,
    max_teacher_steps: int | None,
    max_think_words: int | None,
) -> str | None:
    targets = [
        step
        for step in trajectory.get("steps") or []
        if step.get("sft_target_eligible", True) is not False
    ]
    if not targets or (
        max_teacher_steps is not None and len(targets) > max_teacher_steps
    ):
        return "teacher_step_limit"
    seen: set[str] = set()
    for step in targets:
        if (
            max_think_words is not None
            and len(str(step.get("think") or "").split()) > max_think_words
        ):
            return "teacher_think_limit"
        call = step.get("tool_call") or {}
        signature = json.dumps(call, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if signature in seen:
            return "teacher_repeated_call"
        seen.add(signature)
    return None


def run_recovery(
    *,
    task: dict[str, Any],
    sample: dict[str, Any],
    selected: dict[str, Any],
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    prompt_audit: dict[str, str],
    max_total_steps: int,
    max_teacher_steps: int,
    max_think_words: int,
    max_tokens: int,
    api_retries: int,
    api_timeout: int,
    max_errors_per_type: int,
    history_turns: int,
) -> dict[str, Any]:
    harness = Harness(task_db_path(task))
    started = time.time()
    try:
        seed = replay_prefix(
            harness,
            task,
            sample,
            selected["selected_candidate"],
        )
        catalog = seed["catalog"]
        ctx = seed["ctx"]
        created = seed["created"]
        legal_history = seed["legal_history"]
        steps = seed["prefix_steps"]
        error_events = seed["error_events"]
        last_error = seed["last_error"]
        action_count = seed["anchor_action_index"]
        prefix_error_count = len(error_events)
        teacher_error_counts: Counter[str] = Counter()
        teacher_errors = 0
        teacher_legal_steps = 0
        turns: list[dict[str, Any]] = []
        adjacent_guard = AdjacentActionGuard()
        if error_events:
            preceding_event = error_events[-1]
            if preceding_event.get("attempted_tool"):
                adjacent_guard.prime(
                    preceding_event["attempted_tool"],
                    preceding_event.get("attempted_arguments") or {},
                    step_id=str(
                        preceding_event.get("step_id")
                        or f"step_{preceding_event['action_index']}"
                    ),
                    status="rejected",
                    error_context=(last_error or {}).get("error"),
                )
        usage: Counter[str] = Counter()
        correct = legal = False
        failure_type: str | None = None
        pred_sample = gold_sample = None

        while action_count < max_total_steps and teacher_legal_steps < max_teacher_steps:
            action_count += 1
            state_before = ctx["environment"].snapshot()
            model_input = rolling_legal_history_messages(
                system_prompt,
                catalog,
                task["question"],
                state_before,
                last_error,
                task.get("external_knowledge") or None,
                legal_history,
                history_turns,
            )
            model_input = provider_request_messages(
                model,
                model_input,
                carrier=DEEPSEEK_CARRIER_JSON_OUTPUT,
            )
            turn: dict[str, Any] = {
                "turn_index": action_count - 1,
                "model_input": deepcopy(model_input),
                "provider_request_options": provider_request_options(
                    model,
                    carrier=DEEPSEEK_CARRIER_JSON_OUTPUT,
                ),
            }
            try:
                raw_text, call_usage, reasoning = chat_with_retries(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=model_input,
                    max_tokens=max_tokens,
                    timeout=api_timeout,
                    retries=api_retries,
                    deepseek_carrier=DEEPSEEK_CARRIER_JSON_OUTPUT,
                )
                add_usage(usage, call_usage)
            except ProviderCarrierError as exc:
                add_usage(usage, exc.usage)
                turn["api_error"] = f"{type(exc).__name__}: {exc}"
                turns.append(turn)
                failure_type = "provider_carrier_error"
                break
            except ContextOverflowError as exc:
                turn["api_error"] = f"{type(exc).__name__}: {exc}"
                turns.append(turn)
                failure_type = "context_overflow"
                break
            except Exception as exc:  # noqa: BLE001
                turn["api_error"] = f"{type(exc).__name__}: {exc}"
                turns.append(turn)
                failure_type = "api_error"
                break

            text, adapter = adapt_provider_response(
                model,
                raw_text,
                reasoning,
                carrier=DEEPSEEK_CARRIER_JSON_OUTPUT,
            )
            turn["raw_model_output"] = raw_text
            turn["response_adapter"] = adapter
            turn["model_output"] = text
            if reasoning:
                turn["provider_reasoning_content"] = reasoning
            try:
                rejection = provider_rejection_message(adapter)
                if rejection:
                    adjacent_guard.clear()
                    raise ProtocolError(rejection)
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
                    (last_error or {}).get("error") or {}
                ).get("type")

                if tool == "answer_from_context":
                    legal = True
                    correct, pred_sample, gold_sample = score(
                        harness,
                        task_gold_sql(task),
                        arguments,
                        created,
                        denotation_comparison="bird-set",
                    )
                    step = {
                        "step_id": step_id,
                        "think": think,
                        "think_source": "external_teacher_recovery",
                        "tool_call": {"tool": tool, "arguments": arguments},
                        "tool_output": {"final_answer": arguments.get("answer")},
                        "environment_state_before": state_before,
                        "environment_state": ctx["environment"].snapshot(),
                        "last_tool_error_before": last_error,
                        "feedback_recovery": bool(last_error),
                        "recovered_from_error_type": (
                            (last_error or {}).get("error") or {}
                        ).get("type"),
                        "sft_target_eligible": True,
                    }
                    steps.append(step)
                    teacher_legal_steps += 1
                    turns.append(turn)
                    failure_type = None if correct else "wrong_answer"
                    break

                output, table_name = execute_tool(
                    harness,
                    tool,
                    arguments,
                    ctx,
                    step_id,
                )
                turn["tool_output"] = output
                step = {
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
                steps.append(step)
                teacher_legal_steps += 1
                turns.append(turn)
                adjacent_guard.mark_last("success")
                legal_history.append(
                    {
                        "assistant": assistant_message(think, tool, arguments),
                        "observation": tool_output_message(step_id, output),
                    }
                )
                if table_name:
                    created.add(table_name)
                last_error = None
            except Exception as exc:  # noqa: BLE001
                teacher_errors += 1
                parsed = turn.get("parsed") or {}
                attempted_tool = parsed.get("tool") or getattr(exc, "attempted_tool", None)
                attempted_arguments = (
                    parsed.get("arguments")
                    if parsed.get("tool")
                    else getattr(exc, "attempted_arguments", None)
                )
                state_after = ctx["environment"].snapshot()
                error_type = (
                    protocol_failure_type(exc)
                    if isinstance(exc, ProtocolError)
                    else "execution_error"
                )
                if (
                    error_type == "execution_error"
                    and state_digest(state_after) != state_digest(state_before)
                ):
                    error_type = "nonrecoverable_execution_error"
                adjacent_guard.mark_last("rejected")
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
                turn["execution_error"] = message
                turn["execution_error_type"] = error_type
                turn["error_event"] = event
                turns.append(turn)
                error_events.append(event)
                if error_type == "nonrecoverable_execution_error":
                    failure_type = error_type
                    break
                teacher_error_counts[error_type] += 1
                if teacher_error_counts[error_type] >= max_errors_per_type:
                    failure_type = error_type
                    break
                last_error = json.loads(tool_error_message(
                    str(event.get("step_id") or f"step_{action_count}"),
                    error_type,
                    message,
                    error_code=getattr(exc, "code", type(exc).__name__),
                    details=getattr(exc, "details", None),
                    attempted_tool=attempted_tool,
                    attempted_arguments=attempted_arguments,
                ))
                adjacent_guard.mark_last("rejected", last_error["error"])
        else:
            failure_type = (
                "max_total_steps"
                if action_count >= max_total_steps
                else "max_teacher_steps"
            )

        elapsed = round(time.time() - started, 3)
        identity = task_identity(task)
        result: dict[str, Any] = {
            "example_id": identity,
            "example_index": int(task["example_index"]),
            "trajectory_id": f"bird_batch2_recovery_{identity}",
            "db_id": task["db_id"],
            "question": task["question"],
            "correct": correct,
            "legal": legal,
            "failure_type": failure_type,
            "anchor_action_index": seed["anchor_action_index"],
            "prefix_legal_steps": len(seed["prefix_steps"]),
            "prefix_error_events": prefix_error_count,
            "teacher_legal_steps": teacher_legal_steps,
            "teacher_error_events": teacher_errors,
            "turns": turns,
            "usage": dict(usage),
            "elapsed_seconds": elapsed,
            "pred_sample": pred_sample,
            "gold_sample": gold_sample,
        }
        if correct:
            trajectory = {
                "tool_scheme": ATOMIC_TOOL_SCHEME,
                "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
                "assistant_carrier": ATOMIC_ASSISTANT_CARRIER,
                "protocol_version": PROTOCOL_VERSION,
                "protocol_hash": protocol_hash(system_prompt),
                "trajectory_id": result["trajectory_id"],
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
                    "action_count": action_count,
                    "prefix_anchor_action_index": seed["anchor_action_index"],
                    "prefix_legal_steps": len(seed["prefix_steps"]),
                    "teacher_legal_steps": teacher_legal_steps,
                    "error_events": error_events,
                    "outcome": "recovery_teacher_success",
                    "prompt_contract": prompt_audit,
                    "selection_audit": selected["selection_audit"],
                    "usage": dict(usage),
                },
            }
            replay_ok, replay_error_message = replay_success_trajectory(
                trajectory,
                denotation_comparison="bird-set",
            )
            if not replay_ok:
                result["correct"] = False
                result["failure_type"] = "deterministic_replay_failed"
                result["replay_error"] = replay_error_message
            else:
                quality = continuation_quality_reason(
                    trajectory,
                    max_teacher_steps=max_teacher_steps,
                    max_think_words=max_think_words,
                )
                if quality:
                    result["correct"] = False
                    result["failure_type"] = quality
                else:
                    result["trajectory"] = trajectory
        return result
    finally:
        harness.conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-anchors", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--failures-out", type=Path, required=True)
    parser.add_argument("--all-out", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-total-steps", type=int, default=30)
    parser.add_argument("--max-teacher-steps", type=int, default=20)
    parser.add_argument("--max-think-words", type=int, default=300)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--api-timeout", type=int, default=300)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--max-errors-per-type", type=int, default=3)
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    selected_rows = read_jsonl(args.selected_anchors.resolve())
    tasks = read_jsonl(args.tasks.resolve())
    tasks_by_id = {task_identity(task): task for task in tasks}
    if len(tasks_by_id) != len(tasks):
        parser.error("task file contains duplicate example ids")
    selected_ids = [row["task"]["example_id"] for row in selected_rows]
    if len(set(selected_ids)) != len(selected_ids):
        parser.error("selected anchor file contains duplicate tasks")
    missing = set(selected_ids) - set(tasks_by_id)
    if missing:
        parser.error(f"selected anchors are missing from tasks: {sorted(missing)[:5]}")

    api_key, base_url = load_api_config()
    if not api_key or not base_url:
        parser.error("api.md must define API_KEY and BASE_URL")
    max_tokens = args.max_tokens or provider_default_max_tokens(args.model, 1024)
    student_prompt = rolling_system_prompt(get_system_prompt(), compact=False)
    canonical_teacher_prompt = teacher_system_prompt(student_prompt) + DATA_GENERATION_SUFFIX
    system_prompt = provider_system_prompt(
        args.model,
        canonical_teacher_prompt,
        carrier=DEEPSEEK_CARRIER_JSON_OUTPUT,
    )
    prompt_audit = {
        "teacher_prompt_role": "teacher-generation-recovery",
        "teacher_canonical_prompt_sha256": hashlib.sha256(
            canonical_teacher_prompt.encode("utf-8")
        ).hexdigest(),
        "teacher_provider_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "student_runtime_prompt_sha256": hashlib.sha256(
            student_prompt.encode("utf-8")
        ).hexdigest(),
        "tool_schema_sha256": tool_schema_hash(),
    }

    for path in (args.out, args.failures_out, args.all_out):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.resume:
            parser.error(f"{path} exists; use --resume or choose a new output")
    completed: set[str] = set()
    if args.resume and args.all_out.exists():
        completed = {
            row["example_id"]
            for row in read_jsonl(args.all_out)
        }
    work = [row for row in selected_rows if row["task"]["example_id"] not in completed]
    source_cache: dict[Path, dict[int, dict[str, Any]]] = {}
    source_cache_lock = __import__("threading").Lock()

    def process(selected: dict[str, Any]) -> dict[str, Any]:
        with source_cache_lock:
            _, sample = source_record(selected, source_cache)
        task = tasks_by_id[selected["task"]["example_id"]]
        return run_recovery(
            task=task,
            sample=sample,
            selected=selected,
            base_url=base_url,
            api_key=api_key,
            model=args.model,
            system_prompt=system_prompt,
            prompt_audit=prompt_audit,
            max_total_steps=args.max_total_steps,
            max_teacher_steps=args.max_teacher_steps,
            max_think_words=args.max_think_words,
            max_tokens=max_tokens,
            api_retries=args.api_retries,
            api_timeout=args.api_timeout,
            max_errors_per_type=args.max_errors_per_type,
            history_turns=args.history_turns,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, row): row for row in work}
        for future in as_completed(futures):
            selected = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "example_id": selected["task"]["example_id"],
                    "example_index": selected["task"]["example_index"],
                    "correct": False,
                    "failure_type": "recovery_runner_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            append_jsonl(args.all_out, result)
            if result.get("trajectory"):
                append_jsonl(args.out, result["trajectory"])
            else:
                append_jsonl(args.failures_out, result)

    all_rows = read_jsonl(args.all_out)
    counts: Counter[str] = Counter()
    for row in all_rows:
        if row.get("trajectory"):
            counts["success"] += 1
        else:
            counts[f"failure:{row.get('failure_type') or 'unknown'}"] += 1
    manifest = {
        "method": "external_teacher_selected_error_continuation",
        "model": args.model,
        "selected_anchor_tasks": len(selected_rows),
        "completed": len(all_rows),
        "counts": dict(sorted(counts.items())),
        "context_contract": {
            "history_turns": args.history_turns,
            "max_total_steps": args.max_total_steps,
            "max_teacher_steps": args.max_teacher_steps,
            "student_prefix_steps_are_sft_targets": False,
            "error_actions_are_sft_targets": False,
            "evaluator_rationale_visible_to_teacher": False,
            "denotation_comparison": "bird-set",
        },
        "prompt_contract": prompt_audit,
        "outputs": {
            "success": str(args.out),
            "failures": str(args.failures_out),
            "all": str(args.all_out),
        },
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
