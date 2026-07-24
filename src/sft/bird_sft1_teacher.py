#!/usr/bin/env python3
"""One-attempt, closed-loop external-teacher pilot for fixed BIRD train tasks.

Gold SQL remains harness-only: it is never placed in ``model_input``.  Successful episodes are
replayed from their initial state before they are admitted to the canonical success JSONL.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src" / "eval"), str(ROOT / "src" / "harness"), str(HERE)]

from provider_client import load_api_config  # noqa: E402
from rollout import execute_tool, format_tool_error, new_ctx, overview, score  # noqa: E402
from executor import Harness  # noqa: E402
from protocol import (  # noqa: E402
    ProtocolError,
    assistant_message,
    get_system_prompt,
    model_context_messages,
    parse_assistant_strict,
    protocol_hash,
)
from generate_teacher_rollouts import add_usage, chat_with_retries  # noqa: E402

DEFAULT_SELECTION = ROOT / "data" / "eval_inputs" / "bird_train_sft1_pilot30.jsonl"
DEFAULT_ALL = ROOT / "data" / "trajectories" / "bird_sft1_teacher_pilot30_all.jsonl"
DEFAULT_SUCCESS = ROOT / "data" / "trajectories" / "bird_sft1_teacher_pilot30_success.jsonl"
DEFAULT_FAILURES = ROOT / "data" / "trajectories" / "bird_sft1_teacher_pilot30_failures.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "trajectories" / "bird_sft1_teacher_pilot30.summary.json"
DEFAULT_MODEL = "deepseek-v4-flash"
MAX_STEPS = 30
MAX_ERRORS_PER_TYPE = 3
DATA_GENERATION_SUFFIX = (
    "\n\nTEACHER DEMONSTRATION STRICTNESS\n"
    "Produce exactly one legal tool call per turn. Every turn must contain a non-empty <think> "
    "block followed by one <tool_call> block. Use only the current task context, CURRENT "
    "ENVIRONMENT STATE, and any LAST TOOL ERROR; do not invent prior observations or results."
)
STEP_REF = re.compile(r"^step_(\d+)$")


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(compact(value).encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def first_future_reference(value: Any, current_step_index: int) -> str | None:
    """Reject only structured step references, not ordinary prose that happens to contain a token."""
    if isinstance(value, list):
        for item in value:
            found = first_future_reference(item, current_step_index)
            if found:
                return found
        return None
    if not isinstance(value, dict):
        return None
    for key, item in value.items():
        if key in {"value_ref", "evidence", "evidence_step_id", "source_step_id"} and isinstance(item, str):
            match = STEP_REF.fullmatch(item)
            if match and int(match.group(1)) >= current_step_index:
                return item
        found = first_future_reference(item, current_step_index)
        if found:
            return found
    return None


def make_step(
    step_id: str,
    think: str,
    tool: str,
    arguments: dict,
    output: dict,
    state_before: dict,
    state_after: dict,
    last_error: dict | None,
) -> dict:
    return {
        "step_id": step_id,
        "think": think,
        "tool_call": {"tool": tool, "arguments": arguments},
        "tool_output": output,
        "environment_state_before": state_before,
        "environment_state": state_after,
        "last_tool_error_before": last_error,
        "feedback_recovery": bool(last_error),
        "recovered_from_error_type": (last_error or {}).get("error", {}).get("type"),
    }


def protocol_failure_type(exc: ProtocolError) -> str:
    """Keep schema failures distinct from malformed/missing protocol blocks in audit data."""
    text = str(exc).lower()
    if any(marker in text for marker in (
        "arguments", "unexpected", "requires", "missing required", "must be", "must contain",
    )):
        return "argument_validation_error"
    return "protocol_error"


def record_model_error(
    record: dict,
    turn: dict,
    *,
    action_index: int,
    error_type: str,
    message: str,
    state_before: dict,
    state_after: dict | None = None,
) -> dict:
    """Append an auditable error while deliberately leaving EnvironmentState untouched."""
    state_after = state_before if state_after is None else state_after
    event = {
        "action_index": action_index,
        "step_id": f"step_{action_index}",
        "error_type": error_type,
        "message": message,
        "state_before_hash": stable_hash(state_before),
        "state_after_hash": stable_hash(state_after),
    }
    record["error_events"].append(event)
    if record["first_error_type"] is None:
        record["first_error_type"] = error_type
    turn.update({"failure_type": error_type, "error": message, "error_event": event})
    record["turns"].append(turn)
    return {
        "step_id": event["step_id"],
        "status": "error",
        "error": {"type": error_type, "message": message},
    }


def run_episode(
    task: dict,
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_steps: int,
    max_tokens: int,
    api_timeout: int,
    api_retries: int,
    max_errors_per_type: int,
    saved_first_turn: dict | None = None,
) -> dict:
    h = Harness(task["db_path"])
    catalog = overview(h)
    ctx = new_ctx(catalog)
    external_knowledge = task.get("external_knowledge") or None
    episode_id = f"bird_sft1_{task['example_id']}"
    source = {
        "dataset": task.get("dataset", "bird-sql"),
        "split": task.get("split", "train"),
        "example_id": task["example_id"],
        "example_index": task["example_index"],
        "db_id": task["db_id"],
        "db_path": task["db_path"],
        "external_knowledge": external_knowledge,
        "gold_sql": task["gold_sql"],
    }
    record: dict[str, Any] = {
        "episode_id": episode_id,
        "task_id": task["example_id"],
        "db_id": task["db_id"],
        "difficulty": task.get("metadata", {}).get("difficulty_proxy"),
        "correct": False,
        "failure_type": None,
        "first_error_type": None,
        "outcome": None,
        "steps": [],
        "turns": [],
        "error_events": [],
        "source": source,
        "usage": {},
    }
    usage = collections.Counter()
    created: set[str] = set()
    last_error: dict | None = None
    error_counts: collections.Counter[str] = collections.Counter()
    started = time.monotonic()
    system_prompt = get_system_prompt() + DATA_GENERATION_SUFFIX
    try:
        for step_index in range(1, max_steps + 1):
            state_before = ctx["environment"].snapshot()
            model_input = model_context_messages(
                system_prompt,
                catalog,
                task["question"],
                state_before,
                last_error,
                external_knowledge,
            )
            if step_index == 1 and saved_first_turn:
                model_input = saved_first_turn["model_input"]
            turn: dict[str, Any] = {"turn_index": step_index - 1, "model_input": model_input}
            try:
                if step_index == 1 and saved_first_turn:
                    text = saved_first_turn.get("model_output", "")
                    reasoning_content = saved_first_turn.get("reasoning_content", "")
                    turn["reused_first_model_output"] = True
                else:
                    text, call_usage, reasoning_content = chat_with_retries(
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        messages=model_input,
                        max_tokens=max_tokens,
                        timeout=api_timeout,
                        retries=api_retries,
                    )
                    add_usage(usage, call_usage)
            except Exception as exc:  # The client distinguishes context overflow in its exception name.
                failure_type = "context_limit" if "context" in type(exc).__name__.lower() else "api_error"
                record["failure_type"] = failure_type
                if record["first_error_type"] is None:
                    record["first_error_type"] = failure_type
                turn.update({"api_error": f"{type(exc).__name__}: {exc}", "failure_type": failure_type})
                record["turns"].append(turn)
                break

            turn["model_output"] = text
            if reasoning_content:
                turn["reasoning_content"] = reasoning_content
            try:
                think, tool, arguments = parse_assistant_strict(text)
                future = first_future_reference(arguments, step_index)
                if future:
                    raise ValueError(f"future reference {future!r} at {step_index}")
                turn["parsed"] = {"think": think, "tool": tool, "arguments": arguments}
            except ProtocolError as exc:
                error_type = protocol_failure_type(exc)
                last_error = record_model_error(
                    record, turn, action_index=step_index, error_type=error_type,
                    message=str(exc), state_before=state_before,
                )
                error_counts[error_type] += 1
                if error_counts[error_type] >= max_errors_per_type:
                    record["failure_type"] = error_type
                    break
                continue
            except ValueError as exc:
                last_error = record_model_error(
                    record, turn, action_index=step_index, error_type="argument_validation_error",
                    message=str(exc), state_before=state_before,
                )
                error_counts["argument_validation_error"] += 1
                if error_counts["argument_validation_error"] >= max_errors_per_type:
                    record["failure_type"] = "argument_validation_error"
                    break
                continue

            if tool == "answer_from_context":
                correct, pred_sample, gold_sample = score(h, task["gold_sql"], arguments, created)
                final_step = make_step(
                    f"step_{step_index}", think, tool, arguments,
                    {"final_answer": arguments.get("answer")}, state_before,
                    ctx["environment"].snapshot(), last_error,
                )
                record["steps"].append(final_step)
                turn.update({"answer_score": {"correct": correct, "pred_sample": pred_sample,
                                               "gold_sample": gold_sample}})
                record["turns"].append(turn)
                if not correct:
                    record["failure_type"] = "wrong_answer"
                    if record["first_error_type"] is None:
                        record["first_error_type"] = "wrong_answer"
                else:
                    record["correct"] = True
                    record["outcome"] = (
                        "recovered_success" if record["error_events"] else "clean_success"
                    )
                break

            try:
                output, created_table = execute_tool(h, tool, arguments, ctx, f"step_{step_index}")
            except Exception as exc:  # Argument validation happens inside parse_assistant; these are harness errors.
                error = format_tool_error(exc, h, tool, arguments)
                state_after = ctx["environment"].snapshot()
                error_type = (
                    "execution_error"
                    if stable_hash(state_after) == stable_hash(state_before)
                    else "nonrecoverable_execution_error"
                )
                last_error = record_model_error(
                    record, turn, action_index=step_index, error_type=error_type,
                    message=error, state_before=state_before, state_after=state_after,
                )
                if error_type != "execution_error":
                    record["failure_type"] = error_type
                    break
                error_counts[error_type] += 1
                if error_counts[error_type] >= max_errors_per_type:
                    record["failure_type"] = error_type
                    break
                continue

            step = make_step(
                f"step_{step_index}", think, tool, arguments, output, state_before,
                ctx["environment"].snapshot(), last_error,
            )
            record["steps"].append(step)
            turn["tool_output"] = output
            record["turns"].append(turn)
            if created_table:
                created.add(created_table)
            last_error = None
        else:
            record["failure_type"] = "max_steps"
            if record["first_error_type"] is None:
                record["first_error_type"] = "max_steps"
    finally:
        h.conn.close()

    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    record["usage"] = dict(usage)
    if record["correct"]:
        trajectory = {
            "trajectory_id": episode_id,
            "schema_version": "bird-sft1-teacher-v1",
            "source": source,
            "question": task["question"],
            "difficulty": record["difficulty"],
            "label_status": "verified",
            "initial_state": {"dataset_overview": catalog},
            "steps": record["steps"],
            "rollout_generation": {
                "method": "external_teacher_closed_loop",
                "model": model,
                "semantic_attempts": 1,
                "protocol_hash": protocol_hash(),
                "elapsed_seconds": record["elapsed_seconds"],
                "usage": record["usage"],
                "first_error_type": record["first_error_type"],
                "reused_first_model_output": bool(saved_first_turn),
                "outcome": record["outcome"],
                "error_events": record["error_events"],
                "error_counts": dict(error_counts),
            },
        }
        record["trajectory"] = trajectory
    return record


def replay_success_trajectory(
    trajectory: dict,
    *,
    denotation_comparison: str | None = None,
) -> tuple[bool, str | None]:
    """Re-execute accepted steps from scratch and verify state snapshots and final denotation."""
    source = trajectory["source"]
    generation = trajectory.get("rollout_generation") or {}
    recorded_comparison = generation.get("denotation_comparison")
    if (
        denotation_comparison
        and recorded_comparison
        and denotation_comparison != recorded_comparison
    ):
        return (
            False,
            "replay_mismatch: denotation comparison override "
            f"{denotation_comparison!r} conflicts with recorded {recorded_comparison!r}",
        )
    comparison = recorded_comparison or denotation_comparison or "bird-set"
    h = Harness(str(source.get("db_path") or ROOT / "data" / "bird" / "train" / "train_databases" /
                    source["db_id"] / f"{source['db_id']}.sqlite"))
    try:
        catalog = overview(h)
        if compact(catalog) != compact(trajectory["initial_state"]["dataset_overview"]):
            return False, "replay_mismatch: catalog differs"
        ctx = new_ctx(catalog)
        created: set[str] = set()
        error_events = {
            int(event["action_index"]): event
            for event in (trajectory.get("rollout_generation") or {}).get("error_events", [])
            if event.get("action_index") is not None
        }
        for index, step in enumerate(trajectory["steps"], 1):
            action_index = int(step["step_id"].rsplit("_", 1)[1])
            previous_action = (
                int(trajectory["steps"][index - 2]["step_id"].rsplit("_", 1)[1])
                if index > 1 else 0
            )
            for error_index in range(previous_action + 1, action_index):
                event = error_events.get(error_index)
                if not event or event.get("error_type") != "execution_error":
                    continue
                tool = event.get("attempted_tool")
                arguments = event.get("attempted_arguments")
                if not tool or not isinstance(arguments, dict):
                    return False, f"replay_mismatch: incomplete execution error action {error_index}"
                visible_before = compact(ctx["environment"].snapshot())
                try:
                    execute_tool(h, tool, arguments, ctx, f"step_{error_index}")
                except Exception:  # expected audited execution failure
                    pass
                else:
                    return False, f"replay_mismatch: execution error action {error_index} now succeeds"
                if compact(ctx["environment"].snapshot()) != visible_before:
                    return False, f"replay_mismatch: execution error action {error_index} changed state"
            if compact(ctx["environment"].snapshot()) != compact(step["environment_state_before"]):
                return False, f"replay_mismatch: state_before step_{index}"
            tool_call = step["tool_call"]
            future = first_future_reference(tool_call["arguments"], action_index)
            if future:
                return False, f"future_reference: {future} at step_{index}"
            if tool_call["tool"] == "answer_from_context":
                correct, _, _ = score(
                    h,
                    source["gold_sql"],
                    tool_call["arguments"],
                    created,
                    denotation_comparison=comparison,
                )
                if not correct:
                    return False, "replay_mismatch: final denotation"
            else:
                _, created_table = execute_tool(h, tool_call["tool"], tool_call["arguments"], ctx, step["step_id"])
                if created_table:
                    created.add(created_table)
            if compact(ctx["environment"].snapshot()) != compact(step["environment_state"]):
                return False, f"replay_mismatch: state_after step_{index}"
        return True, None
    finally:
        h.conn.close()


def summary(records: list[dict]) -> dict:
    counts = collections.Counter()
    by_difficulty: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for record in records:
        difficulty = record.get("difficulty") or "unknown"
        by_difficulty[difficulty]["candidates"] += 1
        if record.get("correct"):
            counts["success"] += 1
            by_difficulty[difficulty]["success"] += 1
            outcome = record.get("outcome") or "clean_success"
            counts[outcome] += 1
            by_difficulty[difficulty][outcome] += 1
        else:
            counts["failure"] += 1
            counts[f"failure:{record.get('failure_type') or 'other'}"] += 1
            by_difficulty[difficulty][f"failure:{record.get('failure_type') or 'other'}"] += 1
    counts["candidates"] = len(records)
    counts["error_events"] = sum(len(record.get("error_events", [])) for record in records)
    counts["feedback_recovery_steps"] = sum(
        sum(1 for step in record.get("steps", []) if step.get("feedback_recovery"))
        for record in records
    )
    return {"counts": dict(counts), "by_difficulty": {key: dict(value) for key, value in by_difficulty.items()}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--all-out", type=Path, default=DEFAULT_ALL)
    parser.add_argument("--success-out", type=Path, default=DEFAULT_SUCCESS)
    parser.add_argument("--failures-out", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--api-timeout", type=int, default=300)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--max-errors-per-type", type=int, default=MAX_ERRORS_PER_TYPE,
                        help="terminate an episode after this many recoverable errors of one class")
    parser.add_argument("--start", type=int, default=0,
                        help="start offset within the fixed 30-task selection")
    parser.add_argument("--limit", type=int, default=0,
                        help="optional number of tasks to run from --start (0 = all remaining)")
    parser.add_argument(
        "--reuse-first-turns-from",
        type=Path,
        default=None,
        help="continue from saved first model outputs without reissuing those first requests",
    )
    args = parser.parse_args()
    if args.max_steps < 1 or args.workers < 1 or args.max_errors_per_type < 1 or args.start < 0 or args.limit < 0:
        parser.error("--max-steps, --workers, --max-errors-per-type, --start, and --limit must be non-negative")

    all_tasks = load_jsonl(args.selection)
    if len(all_tasks) != 30:
        parser.error(f"selection must contain exactly 30 fixed candidates, got {len(all_tasks)}")
    stop = args.start + args.limit if args.limit else len(all_tasks)
    tasks = all_tasks[args.start:stop]
    if not tasks:
        parser.error("--start/--limit select no tasks")
    api_key, base_url = load_api_config()
    if not api_key or not base_url:
        parser.error("api.md must define API_KEY and BASE_URL")
    for path in (args.all_out, args.success_out, args.failures_out, args.summary_out):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            parser.error(f"refusing to overwrite existing pilot artifact: {path}")

    saved_first_turns: dict[str, dict] = {}
    if args.reuse_first_turns_from:
        saved_records = load_jsonl(args.reuse_first_turns_from)
        saved_first_turns = {
            record["task_id"]: record["turns"][0]
            for record in saved_records
            if record.get("turns") and record.get("task_id")
        }
        if set(saved_first_turns) != {task["example_id"] for task in tasks}:
            parser.error("--reuse-first-turns-from must contain exactly one first turn for every selected task")

    started = time.monotonic()
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_episode, task, base_url=base_url, api_key=api_key, model=args.model,
                max_steps=args.max_steps, max_tokens=args.max_tokens, api_timeout=args.api_timeout,
                api_retries=args.api_retries, max_errors_per_type=args.max_errors_per_type,
                saved_first_turn=saved_first_turns.get(task["example_id"]),
            ): task["example_id"]
            for task in tasks
        }
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            if record.get("correct"):
                ok, error = replay_success_trajectory(record["trajectory"])
                if not ok:
                    record["correct"] = False
                    record["failure_type"] = "future_reference" if error and error.startswith("future_reference") else "replay_mismatch"
                    record["replay_error"] = error
                    record.pop("trajectory", None)
            append_jsonl(args.all_out, record)
            if record.get("correct"):
                append_jsonl(args.success_out, record["trajectory"])
            else:
                append_jsonl(args.failures_out, record)
            records.append(record)
            print(f"[{done}/{len(tasks)}] {'OK' if record.get('correct') else 'ERR'} "
                  f"{record['episode_id']} type={record.get('failure_type')} steps={len(record['steps'])}")

    report = summary(records)
    report.update({
        "generator": "src/sft/bird_sft1_teacher.py",
        "selection": str(args.selection),
        "selection_window": {"start": args.start, "limit": args.limit or len(tasks), "count": len(tasks)},
        "model": args.model,
        "semantic_attempts_per_candidate": 1,
        "teacher_parser": "strict_no_repair",
        "recovery_policy": {
            "same_episode": True,
            "max_errors_per_type": args.max_errors_per_type,
            "error_actions_are_sft_targets": False,
            "api_transport_retries_per_request": args.api_retries,
        },
        "reused_first_model_outputs": bool(args.reuse_first_turns_from),
        "reused_first_turns_from": str(args.reuse_first_turns_from) if args.reuse_first_turns_from else None,
        "protocol_hash": protocol_hash(),
        "all_output": str(args.all_out),
        "success_output": str(args.success_out),
        "failures_output": str(args.failures_out),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    })
    args.summary_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
