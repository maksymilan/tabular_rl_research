#!/usr/bin/env python3
"""Causal official-DeepSeek diagnostic runner for checkpoint-relalg-v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "src" / "sft"),
    str(PROJECT_ROOT / "src" / "eval"),
]

from artifacts import ArtifactWriter  # noqa: E402
from denotation import compare_denotations  # noqa: E402
from provider_client import load_api_config  # noqa: E402
from tool_modules.checkpoint_relalg.audit import audit_result_dir  # noqa: E402
from tool_modules.checkpoint_relalg.protocol import (  # noqa: E402
    ADMISSION_STATUS,
    ATOMIC_TOOLS,
    BACKEND,
    CHECKPOINT_POLICY_VERSION,
    DIALECT,
    ENVIRONMENT_RENDERER_VERSION,
    EXECUTOR_VERSION,
    MODES,
    PROTOCOL_VERSION,
    SCHEME,
    capability_manifest,
    get_system_prompt,
    prompt_hash,
    provider_tool_definitions,
    tool_schema_hash,
)
from tool_modules.checkpoint_relalg.executors import (  # noqa: E402
    ARTIFACT_BYTE_ACCOUNTING_VERSION,
    ResultSizeTracker,
)
from tool_modules.checkpoint_relalg.provider import (  # noqa: E402
    DeepSeekNativeClient,
    ProviderContextOverflow,
    ProviderError,
    tool_result_message,
)
from tool_modules.checkpoint_relalg.provider_tools import (  # noqa: E402
    NativeToolCallError,
    attempted_action_from_native_message,
    validate_native_assistant_message,
)
from tool_modules.checkpoint_relalg.runtime import (  # noqa: E402
    CheckpointRelalgRuntime,
    RuntimeConfig,
)
from tool_modules.checkpoint_relalg.strict_artifact_audit import (  # noqa: E402
    STRICT_AUDIT_VERSION,
    compare_strict_artifacts,
    has_top_level_order_by,
)
from tool_modules.registry import (  # noqa: E402
    TOOL_SCHEME_REGISTRY_VERSION,
    build_checkpoint_relalg_tool_scheme,
)


DEFAULT_TASKS = PROJECT_ROOT / "data/eval_inputs/bird_train_sft1_protocol_terminal10.jsonl"
DEFAULT_MODEL = "deepseek-v4-flash"


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError("tasks file must contain a JSON array or JSONL records")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_read_only(path: str) -> sqlite3.Connection:
    absolute = str(Path(path).expanduser().resolve())
    connection = sqlite3.connect(f"file:{quote(absolute, safe='/')}?mode=ro", uri=True)
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, bytes):
        return {"$blob_hex": value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _rows_sha256(columns: list[str], rows: list[list[Any]]) -> str:
    encoded = json.dumps(
        {"columns": columns, "rows": _json_safe(rows)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_db_path(task: Mapping[str, Any]) -> str:
    value = task.get("db_path")
    if not isinstance(value, str) or not value:
        raise ValueError("checkpoint-relalg tasks require an explicit db_path")
    return value


def _task_reference_sql(task: Mapping[str, Any]) -> str:
    value = task.get("gold_sql") or task.get("query")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("task has no hidden reference SQL for post-terminal scoring")
    return value


def _hidden_reference_rows(
    task: Mapping[str, Any],
    *,
    runtime_config: RuntimeConfig,
) -> list[list[Any]]:
    _, rows, _ = _hidden_reference_artifact(task, runtime_config=runtime_config)
    return rows


def _hidden_reference_artifact(
    task: Mapping[str, Any],
    *,
    runtime_config: RuntimeConfig,
) -> tuple[list[str], list[list[Any]], bool]:
    connection = _open_read_only(_task_db_path(task))
    if hasattr(connection, "setlimit") and hasattr(sqlite3, "SQLITE_LIMIT_LENGTH"):
        current_limit = connection.getlimit(sqlite3.SQLITE_LIMIT_LENGTH)
        connection.setlimit(
            sqlite3.SQLITE_LIMIT_LENGTH,
            min(current_limit, max(runtime_config.max_artifact_bytes, 64 * 1024)),
        )
    deadline = time.monotonic() + runtime_config.sql_timeout_seconds
    connection.set_progress_handler(
        lambda: 1 if time.monotonic() >= deadline else 0,
        1_000,
    )
    try:
        reference_sql = _task_reference_sql(task)
        cursor = connection.execute(reference_sql)
        columns = [item[0] for item in (cursor.description or ())]
        tracker = ResultSizeTracker(
            max_rows=runtime_config.max_artifact_rows,
            max_artifact_bytes=runtime_config.max_artifact_bytes,
            max_cell_bytes=runtime_config.max_cell_bytes,
            operation="hidden_reference",
        )
        rows: list[list[Any]] = []
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            tracker.add_row(row)
            rows.append(list(row))
        return columns, rows, has_top_level_order_by(reference_sql)
    finally:
        connection.set_progress_handler(None, 0)
        connection.close()


def _phase_execution_styles(turns: list[Mapping[str, Any]]) -> dict[str, str]:
    tools_by_phase: dict[str, list[str]] = defaultdict(list)
    for turn in turns:
        action = turn.get("action")
        if isinstance(action, Mapping) and isinstance(action.get("tool"), str):
            tools_by_phase[str(turn.get("phase_id_before"))].append(action["tool"])
    result: dict[str, str] = {}
    for phase_id, tools in tools_by_phase.items():
        relevant = [tool for tool in tools if tool == "execute_sql" or tool in ATOMIC_TOOLS]
        kinds = ["sql" if tool == "execute_sql" else "atomic" for tool in relevant]
        if not kinds:
            style = "perception_or_control_only"
        elif set(kinds) == {"sql"}:
            style = "sql_only"
        elif set(kinds) == {"atomic"}:
            style = "atomic_only"
        elif kinds == sorted(kinds, key=lambda value: 0 if value == "sql" else 1):
            style = "sql_to_atomic"
        elif kinds == sorted(kinds, key=lambda value: 0 if value == "atomic" else 1):
            style = "atomic_to_sql"
        else:
            style = "mixed"
        result[phase_id] = style
    return result


def _runtime_config_payload(config: RuntimeConfig, *, max_model_turns: int) -> dict[str, Any]:
    return {
        "max_model_turns": max_model_turns,
        "max_primitive_calls": config.max_primitive_calls,
        "max_checkpoints": config.max_checkpoints,
        "max_restores": config.max_restores,
        "sql_timeout_seconds": config.sql_timeout_seconds,
        "max_artifact_rows": config.max_artifact_rows,
        "max_artifact_bytes": config.max_artifact_bytes,
        "max_cell_bytes": config.max_cell_bytes,
        "artifact_byte_accounting": ARTIFACT_BYTE_ACCOUNTING_VERSION,
    }


def _process_metrics(
    turns: list[Mapping[str, Any]],
    *,
    final_runtime: Mapping[str, Any],
    mode: str,
    failure_type: str | None,
    provider_usage: Mapping[str, int],
) -> dict[str, Any]:
    """Compute spec-defined process metrics without inventing token splits."""

    phase_lengths: Counter[str] = Counter()
    tool_errors: Counter[str] = Counter()
    calls: list[tuple[str, str]] = []
    sql_calls = 0
    atomic_calls = 0
    for turn in turns:
        phase_id = turn.get("phase_id_before")
        if isinstance(phase_id, str):
            phase_lengths[phase_id] += 1
        result = turn.get("result")
        if not isinstance(result, Mapping):
            continue
        action = turn.get("action")
        tool = action.get("tool") if isinstance(action, Mapping) else None
        checkpoint_id = turn.get("checkpoint_id_before")
        calls.append((str(checkpoint_id), str(tool or "native_rejection")))
        if tool == "execute_sql":
            sql_calls += 1
        elif tool in ATOMIC_TOOLS:
            atomic_calls += 1
        error = result.get("error")
        if isinstance(error, Mapping) and isinstance(error.get("code"), str):
            tool_errors[error["code"]] += 1

    active_checkpoints = {
        str(item) for item in final_runtime.get("active_checkpoint_path", [])
    }
    active_path_length = sum(
        1 for checkpoint_id, _ in calls if checkpoint_id in active_checkpoints
    )
    abandoned_path_cost = len(calls) - active_path_length
    prompt_tokens = provider_usage.get("prompt_tokens")
    total_prompt_tokens = prompt_tokens if isinstance(prompt_tokens, int) else None
    return {
        "model_turns": len(turns),
        "primitive_calls": len(calls),
        "tool_errors": dict(sorted(tool_errors.items())),
        "sql_calls": sql_calls,
        "atomic_calls": atomic_calls,
        "checkpoint_count": int(final_runtime.get("checkpoint_count", 0)),
        "restore_count": int(final_runtime.get("restore_count", 0)),
        "phase_lengths": dict(sorted(phase_lengths.items())),
        "phase_length_unit": "model_turns",
        "active_path_length": active_path_length,
        "abandoned_path_cost": abandoned_path_cost,
        "path_cost_unit": "primitive_calls",
        "history_tokens": None,
        "checkpoint_history_tokens": None,
        "environment_tokens": None,
        "total_prompt_tokens": total_prompt_tokens,
        "token_metric_availability": {
            "history_tokens": "unavailable_without_provider_subsection_tokenizer",
            "checkpoint_history_tokens": "unavailable_without_provider_subsection_tokenizer",
            "environment_tokens": "unavailable_without_provider_subsection_tokenizer",
            "total_prompt_tokens": (
                "provider_reported" if total_prompt_tokens is not None else "unavailable"
            ),
        },
        "context_overflow_rate": 1.0 if failure_type == "context_length_exceeded" else 0.0,
        "final_mode": mode,
    }


def run_episode(
    task: Mapping[str, Any],
    *,
    task_position: int,
    mode: str,
    client: DeepSeekNativeClient,
    runtime_config: RuntimeConfig,
    max_model_turns: int,
    max_tokens: int,
    max_completion_tokens: int,
    api_retries: int,
) -> dict[str, Any]:
    started = time.monotonic()
    connection = _open_read_only(_task_db_path(task))
    runtime = CheckpointRelalgRuntime(
        connection,
        mode=mode,
        config=runtime_config,
    )
    system_prompt = get_system_prompt(mode, teacher=True)
    tools = provider_tool_definitions(mode)
    phase_history: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    provider_usage: Counter[str] = Counter()
    failure_type: str | None = None
    legal = False
    correct = False
    answer_columns: list[str] = []
    answer_rows: list[list[Any]] = []
    scorer_error_type: str | None = None
    strict_artifact_audit: dict[str, Any] | None = None

    try:
        for model_turn in range(1, max_model_turns + 1):
            context = runtime.render_context(
                str(task.get("question") or ""),
                task.get("external_knowledge"),
            )
            current_user_message = {"role": "user", "content": context}
            model_input = [
                {"role": "system", "content": system_prompt},
                *deepcopy(phase_history),
                deepcopy(current_user_message),
            ]
            before_hash = runtime.state.logical_hash()
            phase_before = runtime.state.phase_id
            checkpoint_before = runtime.state.checkpoint_id
            turn: dict[str, Any] = {
                "model_turn_index": model_turn,
                "model_input": deepcopy(model_input),
                "phase_id_before": phase_before,
                "checkpoint_id_before": checkpoint_before,
                "environment_state_hash_before": before_hash,
            }
            try:
                response = client.request_turn_with_retries(
                    messages=model_input,
                    tools=tools,
                    max_tokens=max_tokens,
                    retries=api_retries,
                    max_completion_tokens=max_completion_tokens,
                )
            except ProviderContextOverflow as exc:
                provider_usage.update(exc.accumulated_usage)
                failure_type = "context_length_exceeded"
                turn["provider_error"] = {
                    "type": failure_type,
                    "message": str(exc),
                }
                turn["provider_retry_events"] = deepcopy(exc.retry_events)
                turn["provider_failed_usage"] = dict(sorted(exc.accumulated_usage.items()))
                turns.append(turn)
                break
            except ProviderError as exc:
                provider_usage.update(exc.accumulated_usage)
                failure_type = "provider_error"
                turn["provider_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                turn["provider_retry_events"] = deepcopy(exc.retry_events)
                turn["provider_failed_usage"] = dict(sorted(exc.accumulated_usage.items()))
                turns.append(turn)
                break

            provider_usage.update(
                {key: value for key, value in response.usage.items() if isinstance(value, int)}
            )
            turn["assistant_message"] = deepcopy(response.message)
            turn["finish_reason"] = response.finish_reason
            turn["provider_response_metadata"] = deepcopy(response.response_metadata)
            turn["provider_retry_events"] = deepcopy(response.retry_events)

            try:
                action = validate_native_assistant_message(mode, response.message)
                canonical_action = {
                    "tool": action["tool"],
                    "arguments": deepcopy(action["arguments"]),
                    "tool_call_id": action["tool_call_id"],
                }
                turn["action"] = canonical_action
                result = runtime.apply(action["tool"], action["arguments"])
            except NativeToolCallError as exc:
                attempted = attempted_action_from_native_message(response.message)
                result = runtime.reject_native_turn(
                    code=exc.code,
                    message=exc.message,
                    details={"argument_path": exc.path, "call_count": len(response.calls)},
                    attempted_tool=attempted["tool"],
                    attempted_arguments=attempted["arguments"],
                )
                turn["native_rejection"] = {
                    "code": exc.code,
                    "message": exc.message,
                    "argument_path": exc.path,
                    "call_count": len(response.calls),
                }

            result_messages = [
                tool_result_message(call.call_id, result) for call in response.calls
            ]
            turn["result"] = deepcopy(result)
            turn["tool_result_messages"] = deepcopy(result_messages)
            turn["environment_state_hash_after"] = runtime.state.logical_hash()
            turn["phase_id_after"] = runtime.state.phase_id
            turn["checkpoint_id_after"] = runtime.state.checkpoint_id
            turns.append(turn)

            if result.get("status") == "success" and result.get("phase_transition"):
                phase_history = []
                turn["provider_phase_history_reset"] = True
            elif runtime.done:
                turn["provider_phase_history_reset"] = False
            else:
                phase_history.extend(
                    [
                        deepcopy(current_user_message),
                        deepcopy(response.message),
                        *deepcopy(result_messages),
                    ]
                )

            if runtime.done:
                if runtime.terminal_table is not None:
                    legal = True
                    try:
                        answer_columns, answer_rows = runtime.answer_rows()
                        reference_columns, reference_rows, reference_ordered = (
                            _hidden_reference_artifact(
                                task,
                                runtime_config=runtime_config,
                            )
                        )
                        correct = compare_denotations(
                            answer_rows,
                            reference_rows,
                            comparison="bird-set",
                        )
                        strict_artifact_audit = compare_strict_artifacts(
                            answer_columns,
                            answer_rows,
                            reference_columns,
                            reference_rows,
                            ordered=reference_ordered,
                        )
                        failure_type = None if correct else "wrong_answer"
                    except Exception as exc:  # noqa: BLE001
                        # Hidden-reference failures are evaluation infrastructure
                        # events.  Preserve the causal trajectory without storing
                        # the private SQL or feeding verifier details to the model.
                        scorer_error_type = type(exc).__name__
                        failure_type = "hidden_verifier_error"
                else:
                    failure_type = runtime.failure_type or "runtime_terminated"
                break
        else:
            failure_type = "max_model_turns"
    finally:
        final_runtime = runtime.audit_state()
        connection.close()

    example_index = task.get("example_index", task.get("index", task_position))
    if isinstance(example_index, bool) or not isinstance(example_index, int):
        example_index = task_position
    scheme = build_checkpoint_relalg_tool_scheme(mode=mode)
    process_metrics = _process_metrics(
        turns,
        final_runtime=final_runtime,
        mode=mode,
        failure_type=failure_type,
        provider_usage=provider_usage,
    )
    record = {
        **scheme.manifest_fields(),
        "capability_manifest": capability_manifest(mode),
        "backend": BACKEND,
        "dialect": DIALECT,
        "environment_renderer_version": ENVIRONMENT_RENDERER_VERSION,
        "checkpoint_policy_version": CHECKPOINT_POLICY_VERSION,
        "executor_version": EXECUTOR_VERSION,
        "tool_schema_hash": tool_schema_hash(mode),
        "prompt_hash": prompt_hash(mode, teacher=True),
        "example_index": example_index,
        "task_position": task_position,
        "example_id": task.get("example_id") or task.get("instance_id"),
        "db_id": task.get("db_id"),
        "question": task.get("question"),
        "external_knowledge": task.get("external_knowledge"),
        "mode": mode,
        "teacher_prompt_sha256": prompt_hash(mode, teacher=True),
        "runtime_config": _runtime_config_payload(
            runtime_config,
            max_model_turns=max_model_turns,
        ),
        "provider_request_options": client.request_audit_options,
        "turns": turns,
        "phase_execution_styles": _phase_execution_styles(turns),
        "final_runtime": final_runtime,
        "correct": correct,
        "official_execution_accuracy": correct,
        "strict_artifact_audit": strict_artifact_audit,
        "strict_artifact_accuracy": (
            strict_artifact_audit["strict_artifact_accuracy"]
            if strict_artifact_audit is not None
            else None
        ),
        "schema_match": (
            strict_artifact_audit["schema_match"]
            if strict_artifact_audit is not None
            else None
        ),
        "legal": legal,
        "failure_type": failure_type,
        "scorer_error_type": scorer_error_type,
        "steps": len(turns),
        "primitive_calls": runtime.primitive_calls,
        "errors": runtime.error_count,
        "answer_columns": answer_columns,
        "answer_row_count": len(answer_rows) if legal else None,
        "answer_relation_sha256": (
            _rows_sha256(answer_columns, answer_rows) if legal else None
        ),
        "provider_usage": dict(sorted(provider_usage.items())),
        "process_metrics": process_metrics,
        **process_metrics,
        "denotation_comparison": "bird-set",
        "strict_artifact_audit_version": STRICT_AUDIT_VERSION,
        "sft_export_eligible": False,
        "rl_admission_eligible": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    return record


def build_manifest(args: argparse.Namespace, *, provider_verification: Mapping[str, Any]) -> dict[str, Any]:
    scheme = build_checkpoint_relalg_tool_scheme(mode=args.mode)
    config = RuntimeConfig(
        max_primitive_calls=args.max_primitive_calls,
        max_checkpoints=args.max_checkpoints,
        max_restores=args.max_restores,
        sql_timeout_seconds=args.sql_timeout_seconds,
        max_artifact_rows=args.max_artifact_rows,
        max_artifact_bytes=args.max_artifact_bytes,
        max_cell_bytes=args.max_cell_bytes,
    )
    return {
        **scheme.manifest_fields(),
        "capability_manifest": capability_manifest(args.mode),
        "backend": BACKEND,
        "dialect": DIALECT,
        "environment_renderer_version": ENVIRONMENT_RENDERER_VERSION,
        "checkpoint_policy_version": CHECKPOINT_POLICY_VERSION,
        "executor_version": EXECUTOR_VERSION,
        "tool_schema_hash": tool_schema_hash(args.mode),
        "prompt_hash": prompt_hash(args.mode, teacher=True),
        "runner": "checkpoint-relalg-causal-official-deepseek-v1",
        "dataset": str(args.tasks_json.resolve()),
        "dataset_sha256": _file_sha256(args.tasks_json),
        "start": args.start,
        "requested_size": args.n,
        "model": args.model,
        "provider_verification": dict(provider_verification),
        "provider_request_options": {
            "endpoint": "https://api.deepseek.com/chat/completions",
            "tool_choice": "auto",
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        },
        "teacher_prompt_sha256": prompt_hash(args.mode, teacher=True),
        "runtime_config": _runtime_config_payload(config, max_model_turns=args.max_model_turns),
        "max_tokens": args.max_tokens,
        "max_completion_tokens": args.max_completion_tokens,
        "api_retries": args.api_retries,
        "api_timeout_seconds": args.api_timeout_seconds,
        "workers": 1,
        "denotation_comparison": "bird-set",
        "strict_artifact_audit_version": STRICT_AUDIT_VERSION,
        "causal_model_harness_loop": True,
        "hidden_reference_visible_to_model": False,
        "sft_export_eligible": False,
        "rl_admission_eligible": False,
        "admission_status": ADMISSION_STATUS,
        "process_metric_schema": {
            "phase_lengths": "model_turns_by_phase",
            "active_path_length": "primitive_calls",
            "abandoned_path_cost": "primitive_calls",
            "subsection_tokens": "nullable_when_provider_does_not_report_splits",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run causal checkpoint-relalg-v1 diagnostics with official DeepSeek."
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--tasks-json", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-config", type=Path, default=PROJECT_ROOT / "api.md")
    parser.add_argument("--api-timeout-seconds", type=int, default=300)
    parser.add_argument("--api-retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--max-model-turns", type=int, default=20)
    parser.add_argument("--max-primitive-calls", type=int, default=30)
    parser.add_argument("--max-checkpoints", type=int, default=8)
    parser.add_argument("--max-restores", type=int, default=3)
    parser.add_argument("--sql-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--max-artifact-rows", type=int, default=100_000)
    parser.add_argument("--max-artifact-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-cell-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.start < 0 or args.n < 1:
        raise SystemExit("--start must be non-negative and --n must be positive")
    if args.workers != 1:
        raise SystemExit("checkpoint-relalg-v1 diagnostic runner currently requires --workers 1")
    if args.max_model_turns < 1 or args.api_retries < 1:
        raise SystemExit("model turns and API retries must be positive")
    if args.api_timeout_seconds < 1:
        raise SystemExit("--api-timeout-seconds must be positive")
    if args.max_tokens < 1 or args.max_completion_tokens < args.max_tokens:
        raise SystemExit("completion token bounds are invalid")
    try:
        RuntimeConfig(
            max_primitive_calls=args.max_primitive_calls,
            max_checkpoints=args.max_checkpoints,
            max_restores=args.max_restores,
            sql_timeout_seconds=args.sql_timeout_seconds,
            max_artifact_rows=args.max_artifact_rows,
            max_artifact_bytes=args.max_artifact_bytes,
            max_cell_bytes=args.max_cell_bytes,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid runtime resource limits: {exc}") from exc
    tasks = _load_tasks(args.tasks_json)
    selected = list(enumerate(tasks))[args.start : args.start + args.n]
    if not selected:
        raise SystemExit("selected task slice is empty")

    if args.dry_run:
        print(json.dumps({
            "tool_scheme": SCHEME,
            "protocol_version": PROTOCOL_VERSION,
            "mode": args.mode,
            "tasks": [position for position, _ in selected],
            "tool_schema_sha256": tool_schema_hash(args.mode),
            "teacher_prompt_sha256": prompt_hash(args.mode, teacher=True),
            "result_dir": str(args.result_dir),
            "admission_status": ADMISSION_STATUS,
        }, ensure_ascii=False, sort_keys=True))
        return 0

    api_key, base_url = load_api_config(args.api_config)
    client = DeepSeekNativeClient(
        api_key=api_key,
        base_url=base_url,
        model=args.model,
        timeout_seconds=args.api_timeout_seconds,
    )
    verification = client.verify_model()
    manifest = build_manifest(args, provider_verification=verification)
    writer = ArtifactWriter(str(args.result_dir), manifest, args.resume)
    runtime_config = RuntimeConfig(
        max_primitive_calls=args.max_primitive_calls,
        max_checkpoints=args.max_checkpoints,
        max_restores=args.max_restores,
        sql_timeout_seconds=args.sql_timeout_seconds,
        max_artifact_rows=args.max_artifact_rows,
        max_artifact_bytes=args.max_artifact_bytes,
        max_cell_bytes=args.max_cell_bytes,
    )
    for position, task in selected:
        example_index = task.get("example_index", task.get("index", position))
        if example_index in writer.completed:
            continue
        record = run_episode(
            task,
            task_position=position,
            mode=args.mode,
            client=client,
            runtime_config=runtime_config,
            max_model_turns=args.max_model_turns,
            max_tokens=args.max_tokens,
            max_completion_tokens=args.max_completion_tokens,
            api_retries=args.api_retries,
        )
        writer.append(record)
        flag = "PASS" if record["correct"] else "FAIL"
        print(
            f"[{flag}] {record.get('example_id') or record['example_index']} "
            f"mode={args.mode} turns={record['steps']} errors={record['errors']}"
        )
    summary = writer.summarize()
    audit = audit_result_dir(args.result_dir)
    print(json.dumps({"summary": summary, "audit": audit}, ensure_ascii=False, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
