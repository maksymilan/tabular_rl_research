#!/usr/bin/env python3
"""Structural, provider-history, and no-leak gates for checkpoint-relalg artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote

from .protocol import (
    ADMISSION_STATUS,
    BACKEND,
    CHECKPOINT_POLICY_VERSION,
    DIALECT,
    ENVIRONMENT_RENDERER_VERSION,
    EXECUTOR_VERSION,
    PROTOCOL_VERSION,
    SCHEME,
    capability_manifest,
    get_system_prompt,
    prompt_hash,
    tool_schema_hash,
)
from .provider_tools import (
    NativeToolCallError,
    attempted_action_from_native_message,
    validate_native_assistant_message,
)


FORBIDDEN_MODEL_KEYS = frozenset(
    {
        "gold_sql",
        "gold_rows",
        "gold_result",
        "gold_exec_results",
        "reference_sql",
        "reference_rows",
        "reference_answer",
        "verifier_result",
        "correct",
    }
)


def _walk(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def no_leak_issues(value: Any, *, root: str = "model_input") -> list[str]:
    """Reject verifier/gold fields anywhere in model-visible provider input."""

    issues: list[str] = []
    for path, child in _walk(value, root):
        if not isinstance(child, Mapping):
            continue
        forbidden = sorted(str(key) for key in child if str(key) in FORBIDDEN_MODEL_KEYS)
        if forbidden:
            issues.append(f"{path} contains forbidden model-visible keys: {forbidden}")
    return issues


def provider_history_issues(messages: Any, *, root: str) -> list[str]:
    """Check native assistant-call/tool-result order in one actual request prefix."""

    if not isinstance(messages, list):
        return [f"{root} is not a message list"]
    issues: list[str] = []
    outstanding: list[str] = []
    seen: set[str] = set()
    for index, message in enumerate(messages):
        path = f"{root}[{index}]"
        if not isinstance(message, Mapping):
            issues.append(f"{path} is not an object")
            continue
        role = message.get("role")
        if role == "assistant":
            if outstanding:
                issues.append(f"{path} starts before tool results for {outstanding}")
            raw_calls = message.get("tool_calls") or []
            if not isinstance(raw_calls, list):
                issues.append(f"{path}.tool_calls is not a list")
                continue
            outstanding = []
            for call_index, call in enumerate(raw_calls):
                call_id = call.get("id") if isinstance(call, Mapping) else None
                if not isinstance(call_id, str) or not call_id:
                    issues.append(f"{path}.tool_calls[{call_index}] has no call id")
                elif call_id in seen:
                    issues.append(f"{path}.tool_calls[{call_index}] reuses call id {call_id}")
                else:
                    seen.add(call_id)
                    outstanding.append(call_id)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not outstanding:
                issues.append(f"{path} has an unexpected tool result")
            elif call_id != outstanding[0]:
                issues.append(
                    f"{path} result id {call_id!r} does not match provider order {outstanding[0]!r}"
                )
                if call_id in outstanding:
                    outstanding.remove(call_id)
            else:
                outstanding.pop(0)
        elif outstanding:
            issues.append(f"{path} interrupts tool results for {outstanding}")
    if outstanding:
        issues.append(f"{root} ends before tool results for {outstanding}")
    return issues


def _runtime_identity_issues(record: Mapping[str, Any], *, root: str) -> list[str]:
    """Bind an artifact to the executable backend and capability surface."""

    issues: list[str] = []
    expected_fields = {
        "backend": BACKEND,
        "dialect": DIALECT,
        "environment_renderer_version": ENVIRONMENT_RENDERER_VERSION,
        "checkpoint_policy_version": CHECKPOINT_POLICY_VERSION,
        "executor_version": EXECUTOR_VERSION,
    }
    for field, expected in expected_fields.items():
        if record.get(field) != expected:
            issues.append(f"{root}.{field} does not match the current runtime")
    mode = record.get("mode")
    if mode in {"direct", "atomic", "hybrid"}:
        if _canonical(record.get("capability_manifest")) != _canonical(
            capability_manifest(mode)
        ):
            issues.append(f"{root}.capability_manifest does not match the current mode")
    return issues


def audit_record(record: Mapping[str, Any], *, record_index: int = 0) -> dict[str, Any]:
    root = f"record[{record_index}]"
    issues: list[str] = []
    if record.get("tool_scheme") != SCHEME:
        issues.append(f"{root}.tool_scheme is not {SCHEME!r}")
    if record.get("protocol_version") != PROTOCOL_VERSION:
        issues.append(f"{root}.protocol_version is not {PROTOCOL_VERSION!r}")
    mode = record.get("mode")
    if mode not in {"direct", "atomic", "hybrid"}:
        issues.append(f"{root}.mode is invalid: {mode!r}")
    else:
        if record.get("tool_schema_sha256") != tool_schema_hash(mode):
            issues.append(f"{root}.tool_schema_sha256 does not match current protocol")
        if record.get("teacher_prompt_sha256") != prompt_hash(mode, teacher=True):
            issues.append(f"{root}.teacher_prompt_sha256 does not match current protocol")
    issues.extend(_runtime_identity_issues(record, root=root))
    if record.get("admission_status") != ADMISSION_STATUS:
        issues.append(f"{root}.admission_status must remain diagnostic-only")
    request_options = record.get("provider_request_options")
    if not isinstance(request_options, Mapping):
        issues.append(f"{root}.provider_request_options is missing")
    else:
        if request_options.get("endpoint") != "https://api.deepseek.com/chat/completions":
            issues.append(f"{root}.provider_request_options is not the official endpoint")
        if request_options.get("tool_choice") != "auto":
            issues.append(f"{root}.provider_request_options.tool_choice is not auto")
        if request_options.get("thinking") != {"type": "enabled"}:
            issues.append(f"{root}.provider_request_options.thinking is not enabled")
        if request_options.get("reasoning_effort") != "high":
            issues.append(f"{root}.provider_request_options.reasoning_effort is not high")
        if not isinstance(request_options.get("model"), str) or not request_options.get("model"):
            issues.append(f"{root}.provider_request_options.model is missing")

    turns = record.get("turns")
    if not isinstance(turns, list):
        issues.append(f"{root}.turns is not a list")
        turns = []
    step_ids: set[str] = set()
    expected_phase_history: list[Mapping[str, Any]] = []
    primitive_turns = 0
    runtime_config = record.get("runtime_config") or {}
    try:
        max_primitive_calls = int(runtime_config.get("max_primitive_calls", 30))
        if max_primitive_calls < 1:
            raise ValueError
    except (TypeError, ValueError):
        issues.append(f"{root}.runtime_config.max_primitive_calls is invalid")
        max_primitive_calls = 30
    for turn_index, turn in enumerate(turns):
        path = f"{root}.turns[{turn_index}]"
        if not isinstance(turn, Mapping):
            issues.append(f"{path} is not an object")
            continue
        model_input = turn.get("model_input")
        issues.extend(no_leak_issues(model_input, root=f"{path}.model_input"))
        issues.extend(provider_history_issues(model_input, root=f"{path}.model_input"))
        if isinstance(model_input, list) and mode in {"direct", "atomic", "hybrid"}:
            if len(model_input) < 2:
                issues.append(f"{path}.model_input must contain system and current user context")
            else:
                system = model_input[0]
                if not isinstance(system, Mapping) or system.get("role") != "system":
                    issues.append(f"{path}.model_input does not start with system")
                elif system.get("content") != get_system_prompt(mode, teacher=True):
                    issues.append(f"{path}.model_input system prompt differs from current teacher prompt")
                if (
                    not isinstance(model_input[-1], Mapping)
                    or model_input[-1].get("role") != "user"
                ):
                    issues.append(f"{path}.model_input does not end with current user context")
                if _canonical(model_input[1:-1]) != _canonical(expected_phase_history):
                    issues.append(f"{path}.model_input phase history differs from causal transcript")
        assistant = turn.get("assistant_message")
        if assistant is not None:
            primitive_turns += 1
            calls = assistant.get("tool_calls") if isinstance(assistant, Mapping) else None
            if not isinstance(calls, list) or not calls:
                issues.append(f"{path}.assistant_message has no native calls")
                calls = []
            result_messages = turn.get("tool_result_messages")
            if not isinstance(result_messages, list):
                issues.append(f"{path}.tool_result_messages is not a list")
                result_messages = []
            issues.extend(
                provider_history_issues(
                    [assistant, *result_messages],
                    root=f"{path}.authored_turn",
                )
            )
            result = turn.get("result")
            result_code = (
                (result.get("error") or {}).get("code")
                if isinstance(result, Mapping)
                else None
            )
            try:
                lowered = validate_native_assistant_message(str(mode), assistant)
            except NativeToolCallError as exc:
                if turn.get("action") is not None:
                    issues.append(f"{path} executes an action from a rejected native call")
                rejection = turn.get("native_rejection")
                if not isinstance(rejection, Mapping):
                    issues.append(f"{path} rejected native call has no rejection record")
                else:
                    expected_rejection = {
                        "code": exc.code,
                        "message": exc.message,
                        "argument_path": exc.path,
                        "call_count": len(calls),
                    }
                    if _canonical(rejection) != _canonical(expected_rejection):
                        issues.append(f"{path}.native_rejection differs from native validator")
                attempted = attempted_action_from_native_message(assistant)
                if not isinstance(result, Mapping) or _canonical(
                    result.get("attempted_action")
                ) != _canonical(attempted):
                    issues.append(f"{path}.result attempted_action differs from authored calls")
                budget_override = (
                    primitive_turns > max_primitive_calls
                    and result_code == "primitive_call_limit_reached"
                )
                if not budget_override and result_code != exc.code:
                    issues.append(f"{path}.result error differs from native validator")
                if isinstance(result, Mapping) and not budget_override:
                    expected_error = {
                        "type": "protocol_error",
                        "code": exc.code,
                        "message": exc.message,
                        "details": {
                            "argument_path": exc.path,
                            "call_count": len(calls),
                        },
                    }
                    if _canonical(result.get("error")) != _canonical(expected_error):
                        issues.append(f"{path}.result rejection envelope is not deterministic")
            except Exception as exc:  # invalid record metadata must not crash the audit
                issues.append(
                    f"{path}.assistant_message could not be audited: {type(exc).__name__}"
                )
            else:
                expected_action = {
                    "tool": lowered["tool"],
                    "arguments": lowered["arguments"],
                    "tool_call_id": lowered["tool_call_id"],
                }
                if _canonical(turn.get("action")) != _canonical(expected_action):
                    issues.append(f"{path}.action differs from the authored native call")
                if turn.get("native_rejection") is not None:
                    issues.append(f"{path} records a rejection for a valid native call")
            response_metadata = turn.get("provider_response_metadata")
            if not isinstance(response_metadata, Mapping) or not isinstance(
                response_metadata.get("model"), str
            ):
                issues.append(f"{path}.provider_response_metadata.model is missing")
            elif isinstance(request_options, Mapping):
                response_model = response_metadata.get("model")
                requested_model = request_options.get("model")
                if response_model is not None and requested_model is not None and response_model != requested_model:
                    issues.append(f"{path}.provider response model differs from requested model")
            for result_index, result_message in enumerate(result_messages):
                result_path = f"{path}.tool_result_messages[{result_index}]"
                content = result_message.get("content") if isinstance(result_message, Mapping) else None
                if not isinstance(content, str):
                    issues.append(f"{result_path}.content is not encoded JSON")
                    continue
                try:
                    decoded = json.loads(content)
                except json.JSONDecodeError:
                    issues.append(f"{result_path}.content is invalid JSON")
                    continue
                if _canonical(decoded) != _canonical(result):
                    issues.append(f"{result_path}.content differs from turn.result")
            rejection = turn.get("native_rejection")
            if (
                isinstance(rejection, Mapping)
                and rejection.get("code") != result_code
                and not (
                    primitive_turns > max_primitive_calls
                    and result_code == "primitive_call_limit_reached"
                )
            ):
                issues.append(f"{path}.native_rejection code does not match result error")
            if (
                isinstance(result, Mapping)
                and result.get("status") == "success"
                and result.get("phase_transition")
            ):
                expected_phase_history = []
                if turn.get("provider_phase_history_reset") is not True:
                    issues.append(f"{path} phase transition did not record provider history reset")
            else:
                current_user = (
                    model_input[-1]
                    if isinstance(model_input, list) and model_input
                    else None
                )
                if isinstance(current_user, Mapping):
                    expected_phase_history.extend(
                        [current_user, assistant, *result_messages]
                    )
        result = turn.get("result")
        if isinstance(result, Mapping):
            step_id = result.get("step_id")
            if isinstance(step_id, str):
                if step_id in step_ids:
                    issues.append(f"{path}.result reuses step id {step_id}")
                step_ids.add(step_id)
            if result.get("status") == "error" and (
                turn.get("environment_state_hash_before")
                != turn.get("environment_state_hash_after")
            ):
                issues.append(f"{path} error mutated EnvironmentState")
    if any(key in record for key in ("gold_sql", "gold_rows", "gold_result")):
        issues.append(f"{root} stores hidden gold content")
    return {"record_index": record_index, "passed": not issues, "issues": issues}


def audit_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    reports = [audit_record(record, record_index=index) for index, record in enumerate(records)]
    issues = [issue for report in reports for issue in report["issues"]]
    return {
        "schema_version": "checkpoint-relalg-audit-v1",
        "records": len(reports),
        "passed_records": sum(report["passed"] for report in reports),
        "failed_records": sum(not report["passed"] for report in reports),
        "issue_count": len(issues),
        "issue_types": dict(sorted(Counter(issue.split(" ", 1)[-1] for issue in issues).items())),
        "passed": not issues,
        "records_detail": reports,
    }


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _row_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"$blob_hex": value.hex()}
    if isinstance(value, Mapping):
        return {str(key): _row_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_row_value(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _answer_hash(columns: list[str], rows: list[list[Any]]) -> str:
    payload = json.dumps(
        {"columns": columns, "rows": _row_value(rows)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fresh_replay_record(
    record: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    record_index: int = 0,
) -> dict[str, Any]:
    """Replay only recorded model actions; never consult hidden scoring content."""

    from .runtime import CheckpointRelalgRuntime, RuntimeConfig

    issues: list[str] = []
    issues.extend(
        _runtime_identity_issues(record, root=f"record[{record_index}]")
    )
    db_path = task.get("db_path")
    if not isinstance(db_path, str) or not db_path:
        return {
            "record_index": record_index,
            "passed": False,
            "issues": ["task has no db_path for fresh replay"],
        }
    absolute = str(Path(db_path).expanduser().resolve())
    connection = sqlite3.connect(f"file:{quote(absolute, safe='/')}?mode=ro", uri=True)
    runtime_payload = record.get("runtime_config") or {}
    try:
        for field in ("question", "external_knowledge", "db_id"):
            if _canonical(record.get(field)) != _canonical(task.get(field)):
                issues.append(
                    f"record[{record_index}].{field} differs from the replay task"
                )
        config = RuntimeConfig(
            max_primitive_calls=int(runtime_payload.get("max_primitive_calls", 30)),
            max_checkpoints=int(runtime_payload.get("max_checkpoints", 8)),
            max_restores=int(runtime_payload.get("max_restores", 3)),
            sql_timeout_seconds=float(runtime_payload.get("sql_timeout_seconds", 20.0)),
            max_artifact_rows=int(runtime_payload.get("max_artifact_rows", 100_000)),
            max_artifact_bytes=int(
                runtime_payload.get("max_artifact_bytes", 64 * 1024 * 1024)
            ),
            max_cell_bytes=int(
                runtime_payload.get("max_cell_bytes", 4 * 1024 * 1024)
            ),
        )
        runtime = CheckpointRelalgRuntime(
            connection,
            mode=str(record.get("mode")),
            config=config,
        )
        for turn_index, turn in enumerate(record.get("turns") or []):
            path = f"record[{record_index}].turns[{turn_index}]"
            if not isinstance(turn, Mapping):
                continue
            if turn.get("environment_state_hash_before") != runtime.state.logical_hash():
                issues.append(
                    f"{path}.environment_state_hash_before differs under fresh replay"
                )
            if turn.get("phase_id_before") != runtime.state.phase_id:
                issues.append(f"{path}.phase_id_before differs under fresh replay")
            if turn.get("checkpoint_id_before") != runtime.state.checkpoint_id:
                issues.append(f"{path}.checkpoint_id_before differs under fresh replay")
            expected_context = runtime.render_context(
                str(task.get("question") or ""),
                task.get("external_knowledge"),
            )
            model_input = turn.get("model_input")
            current_user = (
                model_input[-1]
                if isinstance(model_input, list) and model_input
                else None
            )
            if _canonical(current_user) != _canonical(
                {"role": "user", "content": expected_context}
            ):
                issues.append(
                    f"{path}.model_input current context differs from replay state"
                )
            expected = turn.get("result")
            assistant = turn.get("assistant_message")
            if isinstance(assistant, Mapping):
                try:
                    authored = validate_native_assistant_message(
                        str(record.get("mode")), assistant
                    )
                except NativeToolCallError as exc:
                    attempted = attempted_action_from_native_message(assistant)
                    call_count = len(assistant.get("tool_calls") or [])
                    actual = runtime.reject_native_turn(
                        code=exc.code,
                        message=exc.message,
                        details={
                            "argument_path": exc.path,
                            "call_count": call_count,
                        },
                        attempted_tool=attempted["tool"],
                        attempted_arguments=attempted["arguments"],
                    )
                else:
                    actual = runtime.apply(authored["tool"], authored["arguments"])
            elif isinstance(turn.get("native_rejection"), Mapping) and isinstance(expected, Mapping):
                # A native rejection without its raw authored assistant message
                # cannot establish what was actually attempted.
                issues.append(f"{path} native rejection has no authored assistant message")
                continue
            else:
                # Provider failures have no authored executable action, but the
                # exact context request above is still audited.
                continue
            if _canonical(actual) != _canonical(expected):
                issues.append(f"{path}.result differs under fresh replay")
            expected_hash = turn.get("environment_state_hash_after")
            if expected_hash != runtime.state.logical_hash():
                issues.append(f"{path}.environment_state_hash_after differs under fresh replay")
            if turn.get("phase_id_after") != runtime.state.phase_id:
                issues.append(f"{path}.phase_id_after differs under fresh replay")
            if turn.get("checkpoint_id_after") != runtime.state.checkpoint_id:
                issues.append(f"{path}.checkpoint_id_after differs under fresh replay")
        final_runtime = record.get("final_runtime") or {}
        replay_runtime = runtime.audit_state()
        if _canonical(final_runtime) != _canonical(replay_runtime):
            issues.append(f"record[{record_index}].final_runtime differs under fresh replay")
        if record.get("legal"):
            columns, rows = runtime.answer_rows()
            if record.get("answer_relation_sha256") != _answer_hash(columns, rows):
                issues.append(f"record[{record_index}] terminal answer relation differs under replay")
    except BaseException as exc:  # noqa: BLE001
        issues.append(
            f"record[{record_index}] fresh replay raised {type(exc).__name__}: {exc}"
        )
    finally:
        connection.close()
    return {"record_index": record_index, "passed": not issues, "issues": issues}


def fresh_replay_records(
    records: list[Mapping[str, Any]],
    tasks: list[Mapping[str, Any]],
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        position = record.get("task_position")
        if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < len(tasks):
            details.append({
                "record_index": index,
                "passed": False,
                "issues": [f"invalid task_position {position!r}"],
            })
            continue
        details.append(fresh_replay_record(record, tasks[position], record_index=index))
    issues = [issue for detail in details for issue in detail["issues"]]
    return {
        "records": len(details),
        "passed_records": sum(detail["passed"] for detail in details),
        "failed_records": sum(not detail["passed"] for detail in details),
        "issue_count": len(issues),
        "passed": not issues,
        "records_detail": details,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_result_dir(
    result_dir: Path,
    *,
    tasks_json: Path | None = None,
) -> dict[str, Any]:
    records_path = result_dir / "all.jsonl"
    records = load_jsonl(records_path)
    report = audit_records(records)
    if tasks_json is None:
        manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
        dataset = manifest.get("dataset")
        tasks_json = Path(dataset) if isinstance(dataset, str) and dataset else None
    if tasks_json is None or not tasks_json.exists():
        replay = {
            "passed": False,
            "records": len(records),
            "passed_records": 0,
            "failed_records": len(records),
            "issue_count": 1,
            "records_detail": [],
            "issues": ["tasks JSON is unavailable for fresh replay"],
        }
    else:
        with tasks_json.open(encoding="utf-8") as handle:
            if tasks_json.suffix == ".jsonl":
                tasks = [json.loads(line) for line in handle if line.strip()]
            else:
                tasks = json.load(handle)
        replay = fresh_replay_records(records, tasks)
    report["fresh_replay"] = replay
    report["passed"] = bool(report["passed"] and replay["passed"])
    report_path = result_dir / "checkpoint_relalg_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    report = audit_result_dir(args.result_dir)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
