#!/usr/bin/env python3
"""Local Qwen3 closed-loop evaluator for the frozen Atomic-v24 profile.

This is deliberately separate from the official-DeepSeek runner.  The model
authors one ordinary text completion containing ``<think>...</think>`` plus a
raw JSON action.  The action is executed by the exact same
``CheckpointRelalgRuntime`` used by the DeepSeek source trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "src" / "eval")]

from denotation import compare_denotations  # noqa: E402
from tool_modules.checkpoint_relalg.executors import ResultSizeTracker  # noqa: E402
from tool_modules.checkpoint_relalg.protocol import (  # noqa: E402
    ATOMIC_OPERATOR_PROFILE_FROZEN_V24,
    CARRIER_TEXT_JSON,
    CHECKPOINT_GUIDANCE_PROFILE_DISABLED,
    trim_provider_phase_history,
)
from tool_modules.checkpoint_relalg.qwen3_carrier import (  # noqa: E402
    QWEN3_INLINE_CARRIER_VERSION,
    Qwen3ActionError,
    decode_qwen3_action,
    qwen3_student_prompt_sha256,
    qwen3_student_system_prompt,
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
from tool_modules.checkpoint_relalg.text_json_carrier import (  # noqa: E402
    text_json_result_message,
)
from tool_modules.registry import build_checkpoint_relalg_tool_scheme  # noqa: E402


EVALUATOR_VERSION = "checkpoint-relalg-qwen3-local-eval-v1"
LOCAL_HISTORY_POLICY = "qwen3-json-action-only-recent-4-v1"
LOCAL_INFERENCE_TRANSPORT = "vllm-openai-chat-completions-inline-qwen3-v1"
MODE = "atomic"
PROFILE = ATOMIC_OPERATOR_PROFILE_FROZEN_V24
GUIDANCE = CHECKPOINT_GUIDANCE_PROFILE_DISABLED
SOURCE_CARRIER = CARRIER_TEXT_JSON


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, bytes):
        return {"$blob_hex": value.hex()}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value cannot enter local evaluation artifacts")
        return value
    return str(value)


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            values = [json.loads(line) for line in handle if line.strip()]
        else:
            values = json.load(handle)
    if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
        raise ValueError("tasks must be a JSON array or JSONL objects")
    return values


def _resolve_db_path(task: Mapping[str, Any], db_root: Path | None) -> Path:
    raw = task.get("db_path")
    if isinstance(raw, str) and raw:
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    db_id = task.get("db_id")
    if db_root is not None and isinstance(db_id, str) and db_id:
        candidate = db_root / db_id / f"{db_id}.sqlite"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"database unavailable for db_id={db_id!r}")


def _open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{quote(str(path), safe='/')}?mode=ro", uri=True)
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _reference_artifact(
    task: Mapping[str, Any],
    db_path: Path,
    config: RuntimeConfig,
) -> tuple[list[str], list[list[Any]], bool]:
    sql = task.get("gold_sql") or task.get("query")
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("task has no hidden reference SQL")
    connection = _open_read_only(db_path)
    deadline = time.monotonic() + config.sql_timeout_seconds
    connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1_000)
    try:
        cursor = connection.execute(sql)
        columns = [item[0] for item in (cursor.description or ())]
        tracker = ResultSizeTracker(
            max_rows=config.max_artifact_rows,
            max_artifact_bytes=config.max_artifact_bytes,
            max_cell_bytes=config.max_cell_bytes,
            operation="hidden_reference",
        )
        rows: list[list[Any]] = []
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            tracker.add_row(row)
            rows.append(list(row))
        return columns, rows, has_top_level_order_by(sql)
    finally:
        connection.set_progress_handler(None, 0)
        connection.close()


@dataclass(frozen=True)
class LocalCompletion:
    text: str
    finish_reason: str
    response_model: str
    usage: dict[str, int]
    attempts: int
    elapsed_seconds: float


class CompletionClient(Protocol):
    model: str

    def complete(self, messages: Sequence[Mapping[str, Any]], *, max_tokens: int) -> LocalCompletion:
        ...


class LocalInferenceError(RuntimeError):
    pass


class VLLMChatClient:
    """Minimal strict client for an owned local vLLM OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 600.0,
        retries: int = 2,
    ) -> None:
        if not base_url.startswith("http://127.0.0.1:") and not base_url.startswith(
            "http://localhost:"
        ):
            raise ValueError("local evaluator accepts only a loopback vLLM endpoint")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.retries = int(retries)

    def complete(self, messages: Sequence[Mapping[str, Any]], *, max_tokens: int) -> LocalCompletion:
        payload = {
            "model": self.model,
            "messages": [deepcopy(dict(item)) for item in messages],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": int(max_tokens),
            "seed": 0,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        started = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    value = json.loads(response.read())
                response_model = value.get("model")
                if response_model != self.model:
                    raise LocalInferenceError(
                        f"response model mismatch: expected {self.model!r}, got {response_model!r}"
                    )
                choices = value.get("choices")
                if not isinstance(choices, list) or len(choices) != 1:
                    raise LocalInferenceError("local vLLM response must contain one choice")
                choice = choices[0]
                message = choice.get("message")
                text = message.get("content") if isinstance(message, Mapping) else None
                if not isinstance(text, str):
                    raise LocalInferenceError("local vLLM response omitted text content")
                raw_usage = value.get("usage")
                usage: dict[str, int] = {}
                if isinstance(raw_usage, Mapping):
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        item = raw_usage.get(key)
                        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                            usage[key] = item
                if set(usage) != {"prompt_tokens", "completion_tokens", "total_tokens"}:
                    raise LocalInferenceError("local vLLM response omitted token usage")
                if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
                    raise LocalInferenceError("local vLLM token usage is inconsistent")
                return LocalCompletion(
                    text=text,
                    finish_reason=str(choice.get("finish_reason")),
                    response_model=response_model,
                    usage=usage,
                    attempts=attempt,
                    elapsed_seconds=round(time.monotonic() - started, 6),
                )
            except urllib.error.HTTPError as exc:
                retryable = exc.code in {408, 409, 425, 429} or exc.code >= 500
                if retryable and attempt <= self.retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                raise LocalInferenceError(f"local vLLM HTTP {exc.code}") from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                if attempt <= self.retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                raise LocalInferenceError(type(exc).__name__) from exc


def build_qwen3_model_input(
    phase_history: Sequence[Mapping[str, Any]],
    current_context: str,
) -> list[dict[str, str]]:
    """Project recent provider-style triples to the trained local chat shape."""

    raw = [
        {"role": "system", "content": qwen3_student_system_prompt()},
        *[deepcopy(dict(item)) for item in phase_history],
        {"role": "user", "content": current_context},
    ]
    projected: list[dict[str, str]] = []
    for item in raw:
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("local model history has an invalid message")
        if role == "user" and projected and projected[-1]["role"] == "user":
            projected[-1]["content"] += "\n\n" + content
        else:
            projected.append({"role": role, "content": content})
    if projected[0]["role"] != "system" or projected[-1]["role"] != "user":
        raise ValueError("local model input boundary is invalid")
    for index, item in enumerate(projected[1:], start=1):
        expected = "user" if index % 2 else "assistant"
        if item["role"] != expected:
            raise ValueError("local model input does not alternate user/assistant roles")
    return projected


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        max_primitive_calls=50,
        max_checkpoints=0,
        max_restores=0,
        sql_timeout_seconds=20.0,
        max_artifact_rows=100_000,
        max_artifact_bytes=64 * 1024 * 1024,
        max_cell_bytes=4 * 1024 * 1024,
    )


def _runtime_payload(config: RuntimeConfig, max_model_turns: int) -> dict[str, Any]:
    return {
        "max_model_turns": max_model_turns,
        "max_primitive_calls": config.max_primitive_calls,
        "max_checkpoints": config.max_checkpoints,
        "max_restores": config.max_restores,
        "sql_timeout_seconds": config.sql_timeout_seconds,
        "max_artifact_rows": config.max_artifact_rows,
        "max_artifact_bytes": config.max_artifact_bytes,
        "max_cell_bytes": config.max_cell_bytes,
    }


def run_episode(
    task: Mapping[str, Any],
    *,
    task_position: int,
    db_root: Path | None,
    client: CompletionClient,
    max_model_turns: int = 30,
    max_tokens: int = 8_192,
) -> dict[str, Any]:
    started = time.monotonic()
    config = _runtime_config()
    db_path = _resolve_db_path(task, db_root)
    connection = _open_read_only(db_path)
    runtime = CheckpointRelalgRuntime(
        connection,
        mode=MODE,
        atomic_operator_profile=PROFILE,
        config=config,
    )
    history: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    usage: Counter[str] = Counter()
    failure_type: str | None = None
    legal = False
    correct = False
    strict_audit: dict[str, Any] | None = None
    answer_columns: list[str] = []
    answer_rows: list[list[Any]] = []
    try:
        for turn_index in range(1, max_model_turns + 1):
            context = runtime.render_context(
                str(task.get("question") or ""), task.get("external_knowledge")
            )
            model_input = build_qwen3_model_input(history, context)
            before_hash = runtime.state.logical_hash()
            turn: dict[str, Any] = {
                "model_turn_index": turn_index,
                "model_input": deepcopy(model_input),
                "environment_state_hash_before": before_hash,
            }
            try:
                completion = client.complete(model_input, max_tokens=max_tokens)
            except LocalInferenceError as exc:
                turn["local_inference_error"] = {"type": type(exc).__name__, "message": str(exc)}
                turns.append(turn)
                failure_type = "local_inference_error"
                break
            usage.update(completion.usage)
            turn.update(
                {
                    "assistant_text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "response_model": completion.response_model,
                    "usage": completion.usage,
                    "request_attempts": completion.attempts,
                    "generation_elapsed_seconds": completion.elapsed_seconds,
                }
            )
            if completion.finish_reason != "stop":
                turn["local_carrier_error"] = {
                    "code": "completion_truncated"
                    if completion.finish_reason == "length"
                    else "invalid_finish_reason",
                    "finish_reason": completion.finish_reason,
                }
                turns.append(turn)
                failure_type = "local_carrier_error"
                break
            try:
                decoded = decode_qwen3_action(completion.text)
            except Qwen3ActionError as exc:
                turn["local_carrier_error"] = {
                    "code": exc.code,
                    "path": exc.path,
                    "message": exc.message,
                }
                turns.append(turn)
                failure_type = "local_carrier_error"
                break
            raw_action = decoded["raw_action"]
            result = runtime.apply(raw_action["tool"], raw_action["arguments"])
            turn.update(
                {
                    "reasoning_sha256": hashlib.sha256(
                        decoded["reasoning"].encode("utf-8")
                    ).hexdigest(),
                    "reasoning_characters": len(decoded["reasoning"]),
                    "action_text": decoded["action_text"],
                    "action": deepcopy(raw_action),
                    "result": _json_safe(result),
                    "environment_state_hash_after": runtime.state.logical_hash(),
                }
            )
            turns.append(turn)
            if runtime.done:
                if runtime.terminal_table is not None:
                    legal = True
                    try:
                        answer_columns, answer_rows = runtime.answer_rows()
                        reference_columns, reference_rows, ordered = _reference_artifact(
                            task, db_path, config
                        )
                        correct = compare_denotations(
                            answer_rows, reference_rows, comparison="bird-set"
                        )
                        strict_audit = compare_strict_artifacts(
                            answer_columns,
                            answer_rows,
                            reference_columns,
                            reference_rows,
                            ordered=ordered,
                        )
                        failure_type = None if correct else "wrong_answer"
                    except Exception as exc:  # noqa: BLE001
                        failure_type = "hidden_verifier_error"
                        turn["scorer_error_type"] = type(exc).__name__
                else:
                    failure_type = runtime.failure_type or "runtime_terminated"
                break
            history.extend(
                [
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": decoded["action_text"]},
                    text_json_result_message(result),
                ]
            )
            history = trim_provider_phase_history(history, PROFILE)
        else:
            failure_type = "max_model_turns"
    finally:
        final_runtime = runtime.audit_state()
        connection.close()

    scheme = build_checkpoint_relalg_tool_scheme(
        mode=MODE,
        carrier=SOURCE_CARRIER,
        atomic_operator_profile=PROFILE,
    )
    example_index = task.get("example_index", task.get("index", task_position))
    if isinstance(example_index, bool) or not isinstance(example_index, int):
        example_index = task_position
    record = {
        "schema_version": "checkpoint-relalg-qwen3-local-eval-record-v1",
        "evaluator_version": EVALUATOR_VERSION,
        "local_inference_transport": LOCAL_INFERENCE_TRANSPORT,
        "local_history_policy": LOCAL_HISTORY_POLICY,
        "qwen3_inline_carrier_version": QWEN3_INLINE_CARRIER_VERSION,
        "qwen3_student_prompt_sha256": qwen3_student_prompt_sha256(),
        "source_tool_scheme": "checkpoint-relalg",
        "source_mode": MODE,
        "source_atomic_operator_profile": PROFILE,
        "source_carrier": SOURCE_CARRIER,
        "source_checkpoint_guidance_profile": GUIDANCE,
        "source_protocol_hash": scheme.protocol_hash,
        "source_student_prompt_sha256": scheme.student_prompt_hash,
        "source_tool_schema_sha256": scheme.tool_schema_hash,
        "model": client.model,
        "task_position": task_position,
        "example_index": example_index,
        "example_id": task.get("example_id") or task.get("instance_id"),
        "db_id": task.get("db_id"),
        "question": task.get("question"),
        "external_knowledge": task.get("external_knowledge"),
        "runtime_config": _runtime_payload(config, max_model_turns),
        "generation_config": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "seed": 0,
            "thinking": True,
        },
        "turns": turns,
        "final_runtime": _json_safe(final_runtime),
        "correct": correct,
        "official_execution_accuracy": correct,
        "strict_artifact_audit": _json_safe(strict_audit),
        "strict_artifact_accuracy": (
            strict_audit.get("strict_artifact_accuracy") if strict_audit else None
        ),
        "schema_match": strict_audit.get("schema_match") if strict_audit else None,
        "legal": legal,
        "failure_type": failure_type,
        "steps": len(turns),
        "primitive_calls": runtime.primitive_calls,
        "errors": runtime.error_count,
        "checkpoint_count": final_runtime["checkpoint_count"],
        "restore_count": final_runtime["restore_count"],
        "answer_columns": answer_columns,
        "answer_row_count": len(answer_rows) if legal else None,
        "answer_relation_sha256": (
            _sha256_value({"columns": answer_columns, "rows": answer_rows}) if legal else None
        ),
        "usage": dict(sorted(usage.items())),
        "denotation_comparison": "bird-set",
        "strict_artifact_audit_version": STRICT_AUDIT_VERSION,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "sft_export_eligible": False,
        "rl_admission_eligible": False,
    }
    issues = fresh_replay_record(record, task, db_root=db_root)
    record["audit"] = {
        "structure_passed": not issues,
        "fresh_replay_passed": not issues,
        "issues": issues,
    }
    return record


def fresh_replay_record(
    record: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    db_root: Path | None,
) -> list[str]:
    """Rebuild every model input and execute every authored action from scratch."""

    issues: list[str] = []
    expected_identity = {
        "evaluator_version": EVALUATOR_VERSION,
        "local_inference_transport": LOCAL_INFERENCE_TRANSPORT,
        "local_history_policy": LOCAL_HISTORY_POLICY,
        "qwen3_inline_carrier_version": QWEN3_INLINE_CARRIER_VERSION,
        "qwen3_student_prompt_sha256": qwen3_student_prompt_sha256(),
        "source_mode": MODE,
        "source_atomic_operator_profile": PROFILE,
        "source_carrier": SOURCE_CARRIER,
        "source_checkpoint_guidance_profile": GUIDANCE,
    }
    for key, value in expected_identity.items():
        if record.get(key) != value:
            issues.append(f"identity:{key}")
    runtime_payload = record.get("runtime_config")
    if not isinstance(runtime_payload, Mapping):
        return [*issues, "runtime_config"]
    if runtime_payload.get("max_checkpoints") != 0 or runtime_payload.get("max_restores") != 0:
        issues.append("checkpoint_runtime_not_disabled")
    config = _runtime_config()
    db_path = _resolve_db_path(task, db_root)
    connection = _open_read_only(db_path)
    runtime = CheckpointRelalgRuntime(
        connection, mode=MODE, atomic_operator_profile=PROFILE, config=config
    )
    history: list[dict[str, Any]] = []
    try:
        turns = record.get("turns")
        if not isinstance(turns, list) or not turns:
            return [*issues, "turns"]
        for index, turn in enumerate(turns):
            if not isinstance(turn, Mapping):
                issues.append(f"turn[{index}]:shape")
                break
            context = runtime.render_context(
                str(task.get("question") or ""), task.get("external_knowledge")
            )
            expected_input = build_qwen3_model_input(history, context)
            if _canonical(turn.get("model_input")) != _canonical(expected_input):
                issues.append(f"turn[{index}]:model_input")
            if turn.get("environment_state_hash_before") != runtime.state.logical_hash():
                issues.append(f"turn[{index}]:before_hash")
            action = turn.get("action")
            if not isinstance(action, Mapping):
                if index != len(turns) - 1 or not (
                    isinstance(turn.get("local_carrier_error"), Mapping)
                    or isinstance(turn.get("local_inference_error"), Mapping)
                ):
                    issues.append(f"turn[{index}]:missing_action")
                break
            assistant_text = turn.get("assistant_text")
            try:
                decoded = decode_qwen3_action(assistant_text)
            except Qwen3ActionError:
                issues.append(f"turn[{index}]:assistant_carrier")
                break
            if _canonical(decoded["raw_action"]) != _canonical(action):
                issues.append(f"turn[{index}]:action")
            result = runtime.apply(action.get("tool"), action.get("arguments"))
            if _canonical(result) != _canonical(turn.get("result")):
                issues.append(f"turn[{index}]:result")
            if turn.get("environment_state_hash_after") != runtime.state.logical_hash():
                issues.append(f"turn[{index}]:after_hash")
            if runtime.done:
                if index != len(turns) - 1:
                    issues.append("turn_after_terminal")
                break
            history.extend(
                [
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": decoded["action_text"]},
                    text_json_result_message(result),
                ]
            )
            history = trim_provider_phase_history(history, PROFILE)
        if _canonical(runtime.audit_state()) != _canonical(record.get("final_runtime")):
            issues.append("final_runtime")
        legal = runtime.done and runtime.terminal_table is not None
        if bool(record.get("legal")) != legal:
            issues.append("legal")
        if legal:
            answer_columns, answer_rows = runtime.answer_rows()
            _, reference_rows, _ = _reference_artifact(task, db_path, config)
            correct = compare_denotations(answer_rows, reference_rows, comparison="bird-set")
            if bool(record.get("correct")) != correct:
                issues.append("correct")
        gold = task.get("gold_sql") or task.get("query")
        if isinstance(gold, str) and gold.strip():
            for index, turn in enumerate(turns):
                if gold in _canonical(turn.get("model_input")):
                    issues.append(f"turn[{index}]:gold_leak")
    finally:
        connection.close()
    return issues


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _summary(records: Sequence[Mapping[str, Any]], requested: int) -> dict[str, Any]:
    failures = Counter(str(item.get("failure_type")) for item in records if item.get("failure_type"))
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "requested": requested,
        "completed": len(records),
        "correct": sum(item.get("correct") is True for item in records),
        "legal": sum(item.get("legal") is True for item in records),
        "strict_artifact_correct": sum(
            item.get("strict_artifact_accuracy") is True for item in records
        ),
        "schema_match": sum(item.get("schema_match") is True for item in records),
        "fresh_replay_passed": sum(
            (item.get("audit") or {}).get("fresh_replay_passed") is True for item in records
        ),
        "failures": dict(sorted(failures.items())),
        "model_turns": sum(int(item.get("steps") or 0) for item in records),
        "total_tokens": sum(int((item.get("usage") or {}).get("total_tokens") or 0) for item in records),
    }


def _selection_identity(tasks: Sequence[Mapping[str, Any]], start: int) -> dict[str, Any]:
    public = [
        {
            "task_position": start + offset,
            "example_index": task.get("example_index", task.get("index", start + offset)),
            "example_id": task.get("example_id") or task.get("instance_id"),
            "db_id": task.get("db_id"),
            "question_sha256": hashlib.sha256(
                str(task.get("question") or "").encode("utf-8")
            ).hexdigest(),
            "external_knowledge_sha256": hashlib.sha256(
                _canonical(task.get("external_knowledge")).encode("utf-8")
            ).hexdigest(),
        }
        for offset, task in enumerate(tasks)
    ]
    return {"records": len(public), "sha256": _sha256_value(public)}


def build_manifest(args: argparse.Namespace, selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tasks_path = Path(args.tasks_json).resolve()
    adapter_path = Path(args.adapter_path).resolve()
    scheme = build_checkpoint_relalg_tool_scheme(
        mode=MODE, carrier=SOURCE_CARRIER, atomic_operator_profile=PROFILE
    )
    config = {
        "evaluator_version": EVALUATOR_VERSION,
        "local_inference_transport": LOCAL_INFERENCE_TRANSPORT,
        "local_history_policy": LOCAL_HISTORY_POLICY,
        "qwen3_inline_carrier_version": QWEN3_INLINE_CARRIER_VERSION,
        "qwen3_student_prompt_sha256": qwen3_student_prompt_sha256(),
        "source_tool_scheme": "checkpoint-relalg",
        "source_mode": MODE,
        "source_atomic_operator_profile": PROFILE,
        "source_carrier": SOURCE_CARRIER,
        "source_checkpoint_guidance_profile": GUIDANCE,
        "source_protocol_hash": scheme.protocol_hash,
        "source_student_prompt_sha256": scheme.student_prompt_hash,
        "source_tool_schema_sha256": scheme.tool_schema_hash,
        "tasks_json": str(tasks_path),
        "tasks_sha256": _sha256_file(tasks_path),
        "selection": _selection_identity(selected, args.start),
        "start": args.start,
        "requested_size": len(selected),
        "db_root": str(Path(args.db_root).resolve()) if args.db_root else None,
        "base_model": args.base_model,
        "adapter_path": str(adapter_path),
        "adapter_model_sha256": _sha256_file(adapter_path / "adapter_model.safetensors"),
        "served_model": args.model,
        "endpoint": args.base_url,
        "generation_config": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "seed": 0,
            "thinking": True,
        },
        "runtime_config": _runtime_payload(_runtime_config(), args.max_model_turns),
        "workers": args.workers,
    }
    return {
        "schema_version": "checkpoint-relalg-qwen3-local-eval-manifest-v1",
        **config,
        "config_sha256": _sha256_value(config),
        "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_batch(args: argparse.Namespace) -> int:
    tasks = _read_tasks(Path(args.tasks_json).resolve())
    selected = tasks[args.start : args.start + args.n]
    if len(selected) != args.n:
        raise ValueError("requested task slice exceeds dataset")
    result_dir = Path(args.result_dir).resolve()
    manifest_path = result_dir / "manifest.json"
    all_path = result_dir / "all.jsonl"
    summary_path = result_dir / "summary.json"
    manifest = build_manifest(args, selected)
    if result_dir.exists() and not args.resume:
        raise FileExistsError(f"result directory already exists: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("config_sha256") != manifest["config_sha256"]:
            raise ValueError("resume manifest/config mismatch")
        manifest = existing_manifest
    else:
        _atomic_write_json(manifest_path, manifest)
    existing = _load_records(all_path)
    completed_positions = {int(item["task_position"]) for item in existing}
    if len(completed_positions) != len(existing):
        raise ValueError("duplicate task positions in existing journal")
    pending = [
        (args.start + offset, task)
        for offset, task in enumerate(selected)
        if args.start + offset not in completed_positions
    ]
    client = VLLMChatClient(
        args.base_url,
        args.model,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )
    db_root = Path(args.db_root).resolve() if args.db_root else None
    lock = threading.Lock()
    records = list(existing)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_episode,
                task,
                task_position=position,
                db_root=db_root,
                client=client,
                max_model_turns=args.max_model_turns,
                max_tokens=args.max_tokens,
            ): position
            for position, task in pending
        }
        for future in as_completed(futures):
            record = future.result()
            with lock:
                with all_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                records.append(record)
                summary = _summary(records, len(selected))
                _atomic_write_json(summary_path, summary)
                print(
                    f"[{len(records)}/{len(selected)}] "
                    f"{'OK' if record['correct'] else 'WRONG'} "
                    f"legal={record['legal']} steps={record['steps']} "
                    f"position={record['task_position']}",
                    flush=True,
                )
    summary = _summary(records, len(selected))
    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["all_records_sha256"] = _sha256_file(all_path)
    _atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if len(records) == len(selected) and summary["fresh_replay_passed"] == len(records) else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-json", required=True)
    parser.add_argument("--db-root")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-model-turns", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for name in ("n", "workers", "max_model_turns", "max_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.start < 0 or args.retries < 0 or args.timeout_seconds <= 0:
        parser.error("start/retries/timeout are invalid")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return run_batch(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

