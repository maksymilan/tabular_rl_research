#!/usr/bin/env python3
"""Causal official-DeepSeek diagnostic runner for checkpoint-relalg-v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
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
from tool_modules.checkpoint_relalg.checkpoint_store import (  # noqa: E402
    CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
)
from tool_modules.checkpoint_relalg.protocol import (  # noqa: E402
    ADMISSION_STATUS,
    ATOMIC_OPERATOR_PROFILES,
    ATOMIC_TOOLS,
    SEMANTIC_ATOMIC_TOOLS,
    BACKEND,
    CARRIERS,
    CARRIER_ABLATION_PROTOCOL_VERSION,
    CARRIER_NATIVE_TOOL_CALLS,
    CARRIER_POLICY_VERSION,
    CHECKPOINT_GUIDANCE_PROFILES,
    CHECKPOINT_POLICY_VERSION,
    DEFAULT_CHECKPOINT_GUIDANCE_PROFILE,
    DEFAULT_ATOMIC_OPERATOR_PROFILE,
    DEFAULT_CARRIER,
    DIALECT,
    ENVIRONMENT_RENDERER_VERSION,
    EXECUTOR_VERSION,
    MODES,
    PROTOCOL_VERSION,
    SCHEME,
    capability_manifest,
    carrier_experiment_arm,
    checkpoint_commit_eligibility_for_guidance_profile,
    get_system_prompt,
    normalize_checkpoint_guidance_profile,
    normalize_atomic_operator_profile,
    normalize_carrier,
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
    OFFICIAL_DEEPSEEK_BASE_URL,
    ProviderContextOverflow,
    ProviderError,
    provider_request_audit_options,
    tool_result_message,
)
from tool_modules.checkpoint_relalg.provider_tools import (  # noqa: E402
    NativeToolCallError,
    attempted_action_from_native_message,
    native_authored_action_count,
    native_carrier_envelope_valid,
    validate_native_assistant_message,
)
from tool_modules.checkpoint_relalg.runtime import (  # noqa: E402
    CheckpointRelalgRuntime,
    RuntimeConfig,
)
from tool_modules.checkpoint_relalg.text_json_carrier import (  # noqa: E402
    TextJSONActionError,
    attempted_action_from_text_json_message,
    text_json_result_message,
    text_json_authored_action_count,
    text_json_carrier_envelope_valid,
    text_json_exact_single_action,
    validate_text_json_assistant_message,
)
from tool_modules.checkpoint_relalg.strict_artifact_audit import (  # noqa: E402
    STRICT_AUDIT_VERSION,
    compare_strict_artifacts,
    has_top_level_order_by,
)
from tool_modules.registry import (  # noqa: E402
    build_checkpoint_relalg_tool_scheme,
)


DEFAULT_TASKS = PROJECT_ROOT / "data/eval_inputs/bird_train_sft1_protocol_terminal10.jsonl"
DEFAULT_MODEL = "deepseek-v4-flash"
RUNNER_VERSION = "checkpoint-relalg-causal-official-deepseek-carrier-ab-v2"
LEGACY_RUNNER_VERSION = "checkpoint-relalg-causal-official-deepseek-carrier-ab-v1"
BATCH_CONTROL_VERSION = "checkpoint-relalg-batch-control-v1"
SELECTION_IDENTITY_VERSION = "checkpoint-relalg-selection-identity-v1"
DEFAULT_MAX_BATCH_PROVIDER_ATTEMPTS = 1_200
DEFAULT_MAX_BATCH_PROVIDER_TOKENS = 15_000_000
DEFAULT_MAX_BATCH_WALL_SECONDS = 7_200.0
DEFAULT_MAX_CONSECUTIVE_PROVIDER_FAILURES = 2
DEFAULT_MAX_TOTAL_PROVIDER_FAILURES = 3
DEFAULT_MAX_CONSECUTIVE_SEMANTIC_FAILURES = 5
PUBLIC_TASK_IDENTITY_FIELDS = (
    "position",
    "example_id",
    "example_index",
    "db_id",
    "question_sha256",
    "external_knowledge_sha256",
)
INFRASTRUCTURE_FAILURE_TYPES = frozenset(
    {
        "provider_error",
        "context_length_exceeded",
        "hidden_verifier_error",
        "batch_limit_reached",
    }
)
BATCH_STATUS_FIELDS = frozenset(
    {
        "batch_control_version",
        "state",
        "stop_code",
        "stop_detail",
        "requested_size",
        "completed_records",
        "counters",
        "limits",
        "manifest_config_sha256",
        "all_jsonl_sha256",
        "recorded_at_utc",
    }
)
BATCH_COUNTER_FIELDS = frozenset(
    {
        "provider_attempts",
        "provider_tokens",
        "total_provider_failures",
        "consecutive_provider_failures",
        "consecutive_semantic_failures",
        "semantic_failures",
        "total_wall_seconds",
    }
)
OPERATIONAL_RESUME_FIELDS = frozenset({"run_started_at_utc"})
OPERATIONAL_RESUME_POLICY_VERSION = (
    "checkpoint-relalg-preserve-original-run-start-v1"
)


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


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _task_example_index(task: Mapping[str, Any], position: int) -> int:
    value = task.get("example_index", task.get("index", position))
    if isinstance(value, bool) or not isinstance(value, int):
        return position
    return value


def _resolve_source_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _dataset_source_identity(source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    outputs = source_manifest.get("outputs")
    harness_tasks = outputs.get("harness_tasks") if isinstance(outputs, Mapping) else None
    selection = source_manifest.get("selection")
    preserved = (
        selection.get("preserved_cohort")
        if isinstance(selection, Mapping)
        else None
    )
    source_path = (
        harness_tasks.get("path")
        if isinstance(harness_tasks, Mapping)
        else None
    )
    identity: dict[str, Any] = {
        "schema_version": source_manifest.get("schema_version"),
        "status": source_manifest.get("status"),
        "harness_source_path": (
            str(_resolve_source_path(source_path))
            if isinstance(source_path, str) and source_path
            else None
        ),
        "harness_source_sha256": (
            harness_tasks.get("sha256")
            if isinstance(harness_tasks, Mapping)
            else None
        ),
        "harness_source_records": (
            harness_tasks.get("records")
            if isinstance(harness_tasks, Mapping)
            else None
        ),
        "selection_algorithm": (
            selection.get("algorithm") if isinstance(selection, Mapping) else None
        ),
        "order_preserved": (
            preserved.get("task_ids_and_order_preserved")
            if isinstance(preserved, Mapping)
            else None
        ),
    }
    identity["dataset_source_identity_sha256"] = _canonical_sha256(identity)
    return identity


def _selection_identity(
    tasks: list[Mapping[str, Any]],
    *,
    start: int,
    requested_size: int,
) -> dict[str, Any]:
    selected = list(enumerate(tasks))[start : start + requested_size]
    position_ids: list[dict[str, Any]] = []
    public_tasks: list[dict[str, Any]] = []
    for position, task in selected:
        example_id = task.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("selected tasks require nonempty example_id values")
        position_ids.append({"position": position, "example_id": example_id})
        public_tasks.append(
            {
                "position": position,
                "example_id": example_id,
                "example_index": _task_example_index(task, position),
                "db_id": (
                    task.get("db_id") if isinstance(task.get("db_id"), str) else None
                ),
                "question_sha256": _canonical_sha256(task.get("question")),
                "external_knowledge_sha256": _canonical_sha256(
                    task.get("external_knowledge")
                ),
            }
        )
    identity: dict[str, Any] = {
        "identity_version": SELECTION_IDENTITY_VERSION,
        "start": start,
        "requested_size": requested_size,
        "selected_count": len(selected),
        "position_example_id_sequence_sha256": _canonical_sha256(position_ids),
        "public_task_identity_sha256": _canonical_sha256(public_tasks),
        "public_identity_fields": list(PUBLIC_TASK_IDENTITY_FIELDS),
    }
    identity["selection_identity_sha256"] = _canonical_sha256(identity)
    return identity


def _build_dataset_identity(
    tasks_path: Path,
    dataset_manifest_path: Path,
    tasks: list[Mapping[str, Any]],
    *,
    start: int,
    requested_size: int,
) -> dict[str, Any]:
    resolved_tasks = tasks_path.expanduser().resolve()
    resolved_manifest = dataset_manifest_path.expanduser().resolve()
    try:
        source_manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("dataset manifest is unavailable or malformed") from exc
    if not isinstance(source_manifest, Mapping):
        raise ValueError("dataset manifest root must be an object")
    source_identity = _dataset_source_identity(source_manifest)
    actual_tasks_sha256 = _file_sha256(resolved_tasks)
    if source_identity["harness_source_path"] != str(resolved_tasks):
        raise ValueError("dataset manifest does not identify --tasks-json")
    if source_identity["harness_source_sha256"] != actual_tasks_sha256:
        raise ValueError("dataset manifest task hash does not match --tasks-json")
    if source_identity["harness_source_records"] != len(tasks):
        raise ValueError("dataset manifest task count does not match --tasks-json")
    if source_identity["order_preserved"] is not True:
        raise ValueError("dataset manifest does not certify preserved task order")
    selection = _selection_identity(
        tasks,
        start=start,
        requested_size=requested_size,
    )
    if selection["selected_count"] != requested_size:
        raise ValueError("requested task slice extends beyond the certified dataset")
    return {
        "dataset_manifest": str(resolved_manifest),
        "dataset_manifest_sha256": _file_sha256(resolved_manifest),
        "dataset_source_identity": source_identity,
        "selection_identity": selection,
    }


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _batch_limits_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_provider_attempts": _positive_int(
            getattr(
                args,
                "max_batch_provider_attempts",
                DEFAULT_MAX_BATCH_PROVIDER_ATTEMPTS,
            ),
            name="max_batch_provider_attempts",
        ),
        "max_provider_tokens": _positive_int(
            getattr(
                args,
                "max_batch_provider_tokens",
                DEFAULT_MAX_BATCH_PROVIDER_TOKENS,
            ),
            name="max_batch_provider_tokens",
        ),
        "max_wall_seconds": _positive_float(
            getattr(
                args,
                "max_batch_wall_seconds",
                DEFAULT_MAX_BATCH_WALL_SECONDS,
            ),
            name="max_batch_wall_seconds",
        ),
        "max_consecutive_provider_failures": _positive_int(
            getattr(
                args,
                "max_consecutive_provider_failures",
                DEFAULT_MAX_CONSECUTIVE_PROVIDER_FAILURES,
            ),
            name="max_consecutive_provider_failures",
        ),
        "max_total_provider_failures": _positive_int(
            getattr(
                args,
                "max_total_provider_failures",
                DEFAULT_MAX_TOTAL_PROVIDER_FAILURES,
            ),
            name="max_total_provider_failures",
        ),
        "max_consecutive_semantic_failures": _positive_int(
            getattr(
                args,
                "max_consecutive_semantic_failures",
                DEFAULT_MAX_CONSECUTIVE_SEMANTIC_FAILURES,
            ),
            name="max_consecutive_semantic_failures",
        ),
    }


def _load_committed_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"authoritative journal is malformed at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"authoritative journal record {line_number} is not an object"
                )
            records.append(value)
    return records


def _record_provider_attempts(record: Mapping[str, Any]) -> int:
    attempts = 0
    turns = record.get("turns")
    if not isinstance(turns, list):
        return attempts
    for turn in turns:
        if not isinstance(turn, Mapping):
            continue
        events = turn.get("provider_attempt_events")
        if isinstance(events, list):
            attempts += len(events)
    return attempts


def _record_provider_tokens(record: Mapping[str, Any]) -> int:
    usage = record.get("provider_usage")
    value = usage.get("total_tokens") if isinstance(usage, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("record provider_usage.total_tokens must be a nonnegative integer")
    return value


def _record_wall_seconds(record: Mapping[str, Any]) -> float:
    value = record.get("elapsed_seconds")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError("record elapsed_seconds must be a nonnegative finite number")
    return float(value)


def _is_provider_failure(record: Mapping[str, Any]) -> bool:
    return record.get("failure_type") in {
        "provider_error",
        "context_length_exceeded",
    }


def _is_semantic_failure(record: Mapping[str, Any]) -> bool:
    if record.get("correct") is True:
        return False
    return record.get("failure_type") not in INFRASTRUCTURE_FAILURE_TYPES


def _batch_counters_from_records(
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    provider_attempts = 0
    provider_tokens = 0
    total_provider_failures = 0
    consecutive_provider_failures = 0
    semantic_failures = 0
    consecutive_semantic_failures = 0
    total_wall_seconds = 0.0
    for record in records:
        provider_attempts += _record_provider_attempts(record)
        provider_tokens += _record_provider_tokens(record)
        total_wall_seconds += _record_wall_seconds(record)
        if _is_provider_failure(record):
            total_provider_failures += 1
            consecutive_provider_failures += 1
        else:
            consecutive_provider_failures = 0
        if _is_semantic_failure(record):
            semantic_failures += 1
            consecutive_semantic_failures += 1
        else:
            consecutive_semantic_failures = 0
    return {
        "provider_attempts": provider_attempts,
        "provider_tokens": provider_tokens,
        "total_provider_failures": total_provider_failures,
        "consecutive_provider_failures": consecutive_provider_failures,
        "consecutive_semantic_failures": consecutive_semantic_failures,
        "semantic_failures": semantic_failures,
        "total_wall_seconds": round(total_wall_seconds, 6),
    }


def _last_provider_error(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    turns = record.get("turns")
    if not isinstance(turns, list):
        return None
    for turn in reversed(turns):
        if not isinstance(turn, Mapping):
            continue
        error = turn.get("provider_error")
        if isinstance(error, Mapping):
            return error
    return None


def _non_retryable_provider_stop_detail(
    records: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not records or not _is_provider_failure(records[-1]):
        return None
    error = _last_provider_error(records[-1])
    if not isinstance(error, Mapping) or error.get("retryable") is not False:
        return None
    status = error.get("http_status")
    if status is not None and (
        isinstance(status, bool) or not isinstance(status, int)
    ):
        raise ValueError("provider_error.http_status must be an integer or null")
    return {"http_status": status, "retryable": False}


def _batch_stop_decision(
    records: list[Mapping[str, Any]],
    counters: Mapping[str, Any],
    limits: Mapping[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    provider_detail = _non_retryable_provider_stop_detail(records)
    if provider_detail is not None:
        return "non_retryable_provider_http", provider_detail
    comparisons = (
        ("max_provider_attempts", "provider_attempts"),
        ("max_provider_tokens", "provider_tokens"),
        ("max_wall_seconds", "total_wall_seconds"),
        (
            "max_consecutive_provider_failures",
            "consecutive_provider_failures",
        ),
        ("max_total_provider_failures", "total_provider_failures"),
        (
            "max_consecutive_semantic_failures",
            "consecutive_semantic_failures",
        ),
    )
    for stop_code, counter_name in comparisons:
        actual = counters[counter_name]
        limit = limits[stop_code]
        if actual >= limit:
            return stop_code, {"counter": actual, "limit": limit}
    return None


class BatchRequestGuard:
    """Run-wide pre-request guard with resume-aware provider accounting."""

    def __init__(
        self,
        *,
        counters: Mapping[str, Any],
        limits: Mapping[str, Any],
        base_wall_seconds: float | None = None,
    ) -> None:
        self.limits = dict(limits)
        self.provider_attempts = int(counters["provider_attempts"])
        self.provider_tokens = int(counters["provider_tokens"])
        recorded_wall = float(counters["total_wall_seconds"])
        self.base_wall_seconds = max(
            recorded_wall,
            float(base_wall_seconds) if base_wall_seconds is not None else recorded_wall,
        )
        self.started = time.monotonic()

    def total_wall_seconds(self) -> float:
        return round(
            self.base_wall_seconds + max(0.0, time.monotonic() - self.started),
            6,
        )

    def provider_counters(self) -> dict[str, Any]:
        return {
            "provider_attempts": self.provider_attempts,
            "provider_tokens": self.provider_tokens,
            "total_wall_seconds": self.total_wall_seconds(),
        }

    def stop_decision(self) -> tuple[str, dict[str, Any]] | None:
        for stop_code, counter_name in (
            ("max_provider_attempts", "provider_attempts"),
            ("max_provider_tokens", "provider_tokens"),
            ("max_wall_seconds", "total_wall_seconds"),
        ):
            actual = self.provider_counters()[counter_name]
            limit = self.limits[stop_code]
            if actual >= limit:
                return stop_code, {"counter": actual, "limit": limit}
        return None

    def allowed_retries(self, requested_retries: int) -> int:
        remaining = self.limits["max_provider_attempts"] - self.provider_attempts
        return min(requested_retries, max(0, remaining))

    def _observe(self, value: Any) -> None:
        events = getattr(value, "provider_attempt_events", None)
        event_count = len(events) if isinstance(events, list) else 0
        attempt_count = getattr(value, "provider_attempt_count", None)
        if isinstance(attempt_count, bool) or not isinstance(attempt_count, int):
            attempt_count = event_count
        if attempt_count < 0:
            raise ValueError("provider attempt count cannot be negative")
        usage = getattr(value, "usage", None)
        if usage is None:
            usage = getattr(value, "accumulated_usage", None)
        total_tokens = usage.get("total_tokens", 0) if isinstance(usage, Mapping) else 0
        if (
            isinstance(total_tokens, bool)
            or not isinstance(total_tokens, int)
            or total_tokens < 0
        ):
            raise ValueError("provider total token count must be a nonnegative integer")
        self.provider_attempts += attempt_count
        self.provider_tokens += total_tokens

    def observe_response(self, response: Any) -> None:
        self._observe(response)

    def observe_error(self, error: BaseException) -> None:
        self._observe(error)


class BatchRunLimitReached(RuntimeError):
    def __init__(self, stop_code: str, stop_detail: Mapping[str, Any]) -> None:
        super().__init__(stop_code)
        self.stop_code = stop_code
        self.stop_detail = dict(stop_detail)


def _ensure_journal(path: Path) -> None:
    if path.exists():
        return
    with path.open("xb") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp-",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _batch_status_payload(
    *,
    state: str,
    stop_code: str | None,
    stop_detail: Mapping[str, Any] | None,
    requested_size: int,
    completed_records: int,
    counters: Mapping[str, Any],
    limits: Mapping[str, Any],
    manifest_config_sha256: str,
    all_jsonl_sha256: str,
) -> dict[str, Any]:
    if state not in {"running", "stopped", "completed", "incomplete"}:
        raise ValueError("invalid batch state")
    payload = {
        "batch_control_version": BATCH_CONTROL_VERSION,
        "state": state,
        "stop_code": stop_code,
        "stop_detail": dict(stop_detail) if stop_detail is not None else None,
        "requested_size": requested_size,
        "completed_records": completed_records,
        "counters": dict(counters),
        "limits": dict(limits),
        "manifest_config_sha256": manifest_config_sha256,
        "all_jsonl_sha256": all_jsonl_sha256,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if set(payload) != BATCH_STATUS_FIELDS or set(payload["counters"]) != BATCH_COUNTER_FIELDS:
        raise AssertionError("batch status schema drifted")
    return payload


def _write_batch_status(
    result_dir: Path,
    *,
    state: str,
    stop_code: str | None,
    stop_detail: Mapping[str, Any] | None,
    requested_size: int,
    records: list[Mapping[str, Any]],
    counters: Mapping[str, Any],
    limits: Mapping[str, Any],
    manifest_config_sha256: str,
) -> dict[str, Any]:
    all_path = result_dir / "all.jsonl"
    _ensure_journal(all_path)
    payload = _batch_status_payload(
        state=state,
        stop_code=stop_code,
        stop_detail=stop_detail,
        requested_size=requested_size,
        completed_records=len(records),
        counters=counters,
        limits=limits,
        manifest_config_sha256=manifest_config_sha256,
        all_jsonl_sha256=_file_sha256(all_path),
    )
    _atomic_write_json(result_dir / "batch_status.json", payload)
    return payload


def _record_batch_identity_fields(
    dataset_identity: Mapping[str, Any],
    *,
    start: int,
    requested_size: int,
) -> dict[str, Any]:
    source_identity = dataset_identity["dataset_source_identity"]
    selection_identity = dataset_identity["selection_identity"]
    if not isinstance(source_identity, Mapping) or not isinstance(
        selection_identity, Mapping
    ):
        raise ValueError("dataset identities must be objects")
    return {
        "runner": RUNNER_VERSION,
        "dataset_manifest_sha256": dataset_identity["dataset_manifest_sha256"],
        "dataset_source_identity_sha256": source_identity[
            "dataset_source_identity_sha256"
        ],
        "selection_identity_sha256": selection_identity[
            "selection_identity_sha256"
        ],
        "selection_start": start,
        "selection_requested_size": requested_size,
        "batch_control_version": BATCH_CONTROL_VERSION,
    }


def _validate_committed_records(
    records: list[Mapping[str, Any]],
    selected: list[tuple[int, Mapping[str, Any]]],
    *,
    identity_fields: Mapping[str, Any],
    batch_limits: Mapping[str, Any],
) -> None:
    if len(records) > len(selected):
        raise ValueError("authoritative journal exceeds the selected cohort")
    expected_positions = [position for position, _ in selected[: len(records)]]
    actual_positions = [record.get("task_position") for record in records]
    if actual_positions != expected_positions:
        raise ValueError("authoritative journal is not an ordered cohort prefix")
    for record, (_, task) in zip(records, selected):
        if record.get("example_id") != task.get("example_id"):
            raise ValueError("authoritative journal example identity differs from cohort")
        for field, expected in identity_fields.items():
            if record.get(field) != expected:
                raise ValueError(f"authoritative journal {field} differs from manifest")
        if record.get("batch_limits") != dict(batch_limits):
            raise ValueError("authoritative journal batch limits differ from manifest")


def _stored_manifest_config_sha256(path: Path) -> str:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("stored manifest is unavailable or malformed") from exc
    value = manifest.get("config_sha256") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("stored manifest config hash is invalid")
    return value


def _journal_prefix_sha256(path: Path, record_count: int) -> str:
    lines = [line for line in path.read_bytes().splitlines(keepends=True) if line.strip()]
    if record_count < 0 or record_count > len(lines):
        raise ValueError("journal prefix record count is invalid")
    return hashlib.sha256(b"".join(lines[:record_count])).hexdigest()


def _load_batch_status(
    result_dir: Path,
    *,
    records: list[Mapping[str, Any]],
    requested_size: int,
    limits: Mapping[str, Any],
    manifest_config_sha256: str,
) -> dict[str, Any] | None:
    path = result_dir / "batch_status.json"
    if not path.exists():
        if records:
            raise ValueError("committed records exist without batch_status.json")
        return None
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("batch_status.json is unavailable or malformed") from exc
    if not isinstance(status, Mapping) or set(status) != BATCH_STATUS_FIELDS:
        raise ValueError("batch_status.json schema differs from batch-control policy")
    if status.get("batch_control_version") != BATCH_CONTROL_VERSION:
        raise ValueError("batch_status.json control version differs")
    if status.get("requested_size") != requested_size:
        raise ValueError("batch_status.json requested size differs")
    if status.get("limits") != dict(limits):
        raise ValueError("batch_status.json limits differ")
    if status.get("manifest_config_sha256") != manifest_config_sha256:
        raise ValueError("batch_status.json manifest hash differs")
    counters = status.get("counters")
    if not isinstance(counters, Mapping) or set(counters) != BATCH_COUNTER_FIELDS:
        raise ValueError("batch_status.json counters differ from policy")
    state = status.get("state")
    if state not in {"running", "stopped", "completed", "incomplete"}:
        raise ValueError("batch_status.json state is invalid")
    completed_records = status.get("completed_records")
    if (
        isinstance(completed_records, bool)
        or not isinstance(completed_records, int)
        or completed_records < 0
        or completed_records > len(records)
        or len(records) - completed_records > 1
    ):
        raise ValueError("batch_status.json is not a recoverable journal prefix")
    recomputed = _batch_counters_from_records(records)
    if completed_records == len(records):
        for field in BATCH_COUNTER_FIELDS - {"total_wall_seconds"}:
            if counters.get(field) != recomputed[field]:
                raise ValueError(f"batch_status.json counter {field} differs from journal")
        wall = counters.get("total_wall_seconds")
        if (
            isinstance(wall, bool)
            or not isinstance(wall, (int, float))
            or not math.isfinite(float(wall))
            or float(wall) < recomputed["total_wall_seconds"]
        ):
            raise ValueError("batch_status.json wall time differs from journal")
        if status.get("all_jsonl_sha256") != _file_sha256(
            result_dir / "all.jsonl"
        ):
            raise ValueError("batch_status.json journal hash differs")
    else:
        if state != "running":
            raise ValueError("only running status may lag one committed record")
        prior_records = records[:completed_records]
        prior_counters = _batch_counters_from_records(prior_records)
        for field in BATCH_COUNTER_FIELDS - {"total_wall_seconds"}:
            if counters.get(field) != prior_counters[field]:
                raise ValueError(
                    f"lagging batch_status.json counter {field} differs from journal prefix"
                )
        wall = counters.get("total_wall_seconds")
        if (
            isinstance(wall, bool)
            or not isinstance(wall, (int, float))
            or not math.isfinite(float(wall))
            or float(wall) < prior_counters["total_wall_seconds"]
        ):
            raise ValueError("lagging batch_status.json wall time differs from journal prefix")
        if status.get("all_jsonl_sha256") != _journal_prefix_sha256(
            result_dir / "all.jsonl",
            completed_records,
        ):
            raise ValueError("lagging batch_status.json hash differs from journal prefix")
    if state == "completed" and len(records) != requested_size:
        raise ValueError("completed batch_status.json has an incomplete cohort")
    return dict(status)


def _merge_wall_counter(
    counters: Mapping[str, Any],
    guard: BatchRequestGuard,
) -> dict[str, Any]:
    result = dict(counters)
    result["total_wall_seconds"] = max(
        float(result["total_wall_seconds"]),
        guard.total_wall_seconds(),
    )
    return result


def _open_artifact_writer(
    result_dir: Path,
    manifest: Mapping[str, Any],
    *,
    resume: bool,
) -> ArtifactWriter:
    """Open artifacts while preserving the first run's immutable identity.

    A resumed process necessarily computes a new wall-clock start timestamp.
    That single operational field may differ, but ArtifactWriter retains the
    original manifest and records the resume event.  Protocol, carrier, model,
    envelope, cohort, and resource identities remain fail-closed.
    """

    return ArtifactWriter(
        str(result_dir),
        dict(manifest),
        resume,
        operational_resume_fields=set(OPERATIONAL_RESUME_FIELDS),
        operational_resume_metadata={
            "policy_version": OPERATIONAL_RESUME_POLICY_VERSION,
        },
    )


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
        relevant = [
            tool
            for tool in tools
            if tool == "execute_sql" or tool in {*ATOMIC_TOOLS, *SEMANTIC_ATOMIC_TOOLS}
        ]
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
    payload = {
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
    if (
        config.checkpoint_commit_eligibility_policy
        != CHECKPOINT_COMMIT_ELIGIBILITY_NONE
    ):
        payload["checkpoint_commit_eligibility_policy"] = (
            config.checkpoint_commit_eligibility_policy
        )
    return payload


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
        elif tool in {*ATOMIC_TOOLS, *SEMANTIC_ATOMIC_TOOLS}:
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


def _provider_error_payload(error: ProviderError, *, error_type: str) -> dict[str, Any]:
    http_status = getattr(error, "http_status", None)
    if http_status is not None and (
        isinstance(http_status, bool) or not isinstance(http_status, int)
    ):
        raise ValueError("provider error http_status must be an integer or null")
    retryable = getattr(error, "retryable", True)
    if not isinstance(retryable, bool):
        raise ValueError("provider error retryable must be a boolean")
    return {
        "type": error_type,
        "message": str(error),
        "finish_reason": getattr(error, "finish_reason", None),
        "response_envelope_sha256": getattr(
            error,
            "response_envelope_sha256",
            None,
        ),
        "response_model": getattr(error, "response_model", None),
        "http_status": http_status,
        "retryable": retryable,
    }


def run_episode(
    task: Mapping[str, Any],
    *,
    task_position: int,
    mode: str,
    client: DeepSeekNativeClient,
    carrier: str = DEFAULT_CARRIER,
    atomic_operator_profile: str = DEFAULT_ATOMIC_OPERATOR_PROFILE,
    checkpoint_guidance_profile: str = DEFAULT_CHECKPOINT_GUIDANCE_PROFILE,
    experiment_arm: str | None = None,
    within_batch_order: str | None = None,
    runtime_config: RuntimeConfig,
    max_model_turns: int,
    max_tokens: int,
    max_completion_tokens: int,
    api_retries: int,
    artifact_identity_fields: Mapping[str, Any] | None = None,
    batch_limits: Mapping[str, Any] | None = None,
    batch_guard: BatchRequestGuard | None = None,
) -> dict[str, Any]:
    active_carrier = normalize_carrier(carrier)
    active_operator_profile = normalize_atomic_operator_profile(atomic_operator_profile)
    active_checkpoint_guidance = normalize_checkpoint_guidance_profile(
        checkpoint_guidance_profile
    )
    active_commit_eligibility = checkpoint_commit_eligibility_for_guidance_profile(
        active_checkpoint_guidance
    )
    if (
        runtime_config.checkpoint_commit_eligibility_policy
        != active_commit_eligibility
    ):
        raise ValueError(
            "runtime checkpoint commit eligibility does not match guidance profile"
        )
    active_experiment_arm = experiment_arm or carrier_experiment_arm(active_carrier)
    if active_experiment_arm != carrier_experiment_arm(active_carrier):
        raise ValueError("experiment arm does not match the frozen carrier mapping")
    if within_batch_order not in {None, "A_then_B", "B_then_A"}:
        raise ValueError("invalid within-batch carrier order")
    if getattr(client, "carrier", DEFAULT_CARRIER) != active_carrier:
        raise ValueError("runner carrier and provider client carrier must match")
    started = time.monotonic()
    connection = _open_read_only(_task_db_path(task))
    runtime = CheckpointRelalgRuntime(
        connection,
        mode=mode,
        atomic_operator_profile=active_operator_profile,
        config=runtime_config,
    )
    system_prompt = get_system_prompt(
        mode,
        teacher=True,
        carrier=active_carrier,
        checkpoint_guidance_profile=active_checkpoint_guidance,
        atomic_operator_profile=active_operator_profile,
    )
    tools = provider_tool_definitions(mode, active_operator_profile)
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
    batch_stop_code: str | None = None
    batch_stop_detail: dict[str, Any] | None = None

    try:
        for model_turn in range(1, max_model_turns + 1):
            if batch_guard is not None:
                stop = batch_guard.stop_decision()
                if stop is not None:
                    batch_stop_code, batch_stop_detail = stop
                    if not turns:
                        raise BatchRunLimitReached(
                            batch_stop_code,
                            batch_stop_detail,
                        )
                    failure_type = "batch_limit_reached"
                    break
                effective_retries = batch_guard.allowed_retries(api_retries)
                if effective_retries < 1:
                    batch_stop_code = "max_provider_attempts"
                    batch_stop_detail = {
                        "counter": batch_guard.provider_attempts,
                        "limit": batch_guard.limits["max_provider_attempts"],
                    }
                    if not turns:
                        raise BatchRunLimitReached(
                            batch_stop_code,
                            batch_stop_detail,
                        )
                    failure_type = "batch_limit_reached"
                    break
            else:
                effective_retries = api_retries
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
                "carrier": active_carrier,
                "carrier_metrics": {
                    "carrier": active_carrier,
                    "provider_response_present": False,
                    "authored_action_count": 0,
                    "exact_single_action": False,
                    "carrier_envelope_valid": False,
                    "carrier_error_code": None,
                    "action_validation_error_code": None,
                    "provider_attempt_count": None,
                    "provider_elapsed_seconds": None,
                },
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
                    retries=effective_retries,
                    max_completion_tokens=max_completion_tokens,
                )
            except ProviderContextOverflow as exc:
                if batch_guard is not None:
                    batch_guard.observe_error(exc)
                failed_usage = dict(exc.accumulated_usage)
                failed_attempt_events = deepcopy(exc.provider_attempt_events)
                if batch_limits is not None:
                    failed_usage.setdefault("total_tokens", 0)
                    for event in failed_attempt_events:
                        if isinstance(event, dict) and isinstance(event.get("usage"), dict):
                            event["usage"].setdefault("total_tokens", 0)
                provider_usage.update(failed_usage)
                failure_type = "context_length_exceeded"
                turn["provider_error"] = _provider_error_payload(
                    exc,
                    error_type=failure_type,
                )
                turn["provider_retry_events"] = deepcopy(exc.retry_events)
                turn["provider_attempt_events"] = failed_attempt_events
                turn["provider_failed_usage"] = dict(sorted(failed_usage.items()))
                turn["provider_usage"] = dict(sorted(failed_usage.items()))
                turn["carrier_metrics"]["provider_attempt_count"] = (
                    exc.provider_attempt_count
                )
                turn["carrier_metrics"]["provider_elapsed_seconds"] = (
                    exc.provider_elapsed_seconds
                )
                turns.append(turn)
                if batch_guard is not None:
                    stop = batch_guard.stop_decision()
                    if stop is not None:
                        batch_stop_code, batch_stop_detail = stop
                break
            except ProviderError as exc:
                if batch_guard is not None:
                    batch_guard.observe_error(exc)
                failed_usage = dict(exc.accumulated_usage)
                failed_attempt_events = deepcopy(exc.provider_attempt_events)
                if batch_limits is not None:
                    failed_usage.setdefault("total_tokens", 0)
                    for event in failed_attempt_events:
                        if isinstance(event, dict) and isinstance(event.get("usage"), dict):
                            event["usage"].setdefault("total_tokens", 0)
                provider_usage.update(failed_usage)
                failure_type = "provider_error"
                turn["provider_error"] = _provider_error_payload(
                    exc,
                    error_type=type(exc).__name__,
                )
                turn["provider_retry_events"] = deepcopy(exc.retry_events)
                turn["provider_attempt_events"] = failed_attempt_events
                turn["provider_failed_usage"] = dict(sorted(failed_usage.items()))
                turn["provider_usage"] = dict(sorted(failed_usage.items()))
                turn["carrier_metrics"]["provider_attempt_count"] = (
                    exc.provider_attempt_count
                )
                turn["carrier_metrics"]["provider_elapsed_seconds"] = (
                    exc.provider_elapsed_seconds
                )
                turns.append(turn)
                if batch_guard is not None:
                    stop = batch_guard.stop_decision()
                    if stop is not None:
                        batch_stop_code, batch_stop_detail = stop
                break

            if batch_guard is not None:
                batch_guard.observe_response(response)
            provider_usage.update(
                {
                    key: value
                    for key, value in response.usage.items()
                    if isinstance(value, int) and not isinstance(value, bool)
                }
            )
            turn["assistant_message"] = deepcopy(response.message)
            turn["finish_reason"] = response.finish_reason
            turn["provider_response_metadata"] = deepcopy(response.response_metadata)
            turn["provider_retry_events"] = deepcopy(response.retry_events)
            turn["provider_attempt_events"] = deepcopy(
                response.provider_attempt_events
            )
            turn["provider_usage"] = dict(sorted(
                (key, value)
                for key, value in response.usage.items()
                if isinstance(value, int) and not isinstance(value, bool)
            ))
            carrier_metrics = turn["carrier_metrics"]
            carrier_metrics["provider_response_present"] = True
            carrier_metrics["provider_attempt_count"] = response.provider_attempt_count
            carrier_metrics["provider_elapsed_seconds"] = (
                response.provider_elapsed_seconds
            )
            if active_carrier == CARRIER_NATIVE_TOOL_CALLS:
                authored_action_count = native_authored_action_count(response.message)
                carrier_metrics["authored_action_count"] = authored_action_count
                carrier_metrics["exact_single_action"] = authored_action_count == 1
                carrier_metrics["carrier_envelope_valid"] = (
                    native_carrier_envelope_valid(response.message)
                )
            else:
                carrier_metrics["authored_action_count"] = (
                    text_json_authored_action_count(response.message)
                )
                carrier_metrics["exact_single_action"] = (
                    text_json_exact_single_action(response.message)
                )
                carrier_metrics["carrier_envelope_valid"] = (
                    text_json_carrier_envelope_valid(response.message)
                )

            try:
                if active_carrier == CARRIER_NATIVE_TOOL_CALLS:
                    action = validate_native_assistant_message(
                        mode,
                        response.message,
                        atomic_operator_profile=active_operator_profile,
                    )
                else:
                    action = validate_text_json_assistant_message(
                        mode,
                        response.message,
                        atomic_operator_profile=active_operator_profile,
                    )
                canonical_action = {
                    "tool": action["tool"],
                    "arguments": deepcopy(action["arguments"]),
                    "tool_call_id": action.get("tool_call_id"),
                }
                turn["action"] = canonical_action
                result = runtime.apply(action["tool"], action["arguments"])
            except (NativeToolCallError, TextJSONActionError) as exc:
                if carrier_metrics["carrier_envelope_valid"]:
                    carrier_metrics["action_validation_error_code"] = exc.code
                else:
                    carrier_metrics["carrier_error_code"] = exc.code
                if active_carrier == CARRIER_NATIVE_TOOL_CALLS:
                    attempted = attempted_action_from_native_message(response.message)
                    rejection_details = {
                        "argument_path": exc.path,
                        "call_count": len(response.calls),
                    }
                    rejection_field = "native_rejection"
                else:
                    attempted = attempted_action_from_text_json_message(response.message)
                    rejection_details = {
                        "argument_path": exc.path,
                        "carrier": active_carrier,
                    }
                    rejection_field = "text_json_rejection"
                result = runtime.reject_native_turn(
                    code=exc.code,
                    message=exc.message,
                    details=deepcopy(rejection_details),
                    attempted_tool=attempted["tool"],
                    attempted_arguments=attempted["arguments"],
                )
                turn[rejection_field] = {
                    "code": exc.code,
                    "message": exc.message,
                    **rejection_details,
                }

            if active_carrier == CARRIER_NATIVE_TOOL_CALLS:
                result_messages = [
                    tool_result_message(call.call_id, result) for call in response.calls
                ]
                turn["tool_result_messages"] = deepcopy(result_messages)
            else:
                result_messages = [text_json_result_message(result)]
                turn["text_result_messages"] = deepcopy(result_messages)
            turn["result"] = deepcopy(result)
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
            if batch_guard is not None:
                stop = batch_guard.stop_decision()
                if stop is not None:
                    batch_stop_code, batch_stop_detail = stop
                    failure_type = "batch_limit_reached"
                    break
        else:
            failure_type = "max_model_turns"
    finally:
        final_runtime = runtime.audit_state()
        connection.close()

    example_index = task.get("example_index", task.get("index", task_position))
    if isinstance(example_index, bool) or not isinstance(example_index, int):
        example_index = task_position
    scheme = build_checkpoint_relalg_tool_scheme(
        mode=mode,
        carrier=active_carrier,
        atomic_operator_profile=active_operator_profile,
        checkpoint_commit_eligibility_policy=active_commit_eligibility,
    )
    process_metrics = _process_metrics(
        turns,
        final_runtime=final_runtime,
        mode=mode,
        failure_type=failure_type,
        provider_usage=provider_usage,
    )
    record = {
        **(dict(artifact_identity_fields) if artifact_identity_fields else {}),
        **scheme.manifest_fields(),
        "capability_manifest": capability_manifest(
            mode,
            active_carrier,
            active_operator_profile,
            active_commit_eligibility,
        ),
        "backend": BACKEND,
        "dialect": DIALECT,
        "environment_renderer_version": ENVIRONMENT_RENDERER_VERSION,
        "checkpoint_policy_version": CHECKPOINT_POLICY_VERSION,
        "checkpoint_guidance_profile": active_checkpoint_guidance,
        "executor_version": EXECUTOR_VERSION,
        "tool_schema_hash": tool_schema_hash(mode, active_operator_profile),
        "carrier_ablation_protocol_version": CARRIER_ABLATION_PROTOCOL_VERSION,
        "carrier_policy_version": CARRIER_POLICY_VERSION,
        "experiment_arm": active_experiment_arm,
        "within_batch_order": within_batch_order,
        "carrier": active_carrier,
        "prompt_hash": prompt_hash(
            mode,
            teacher=True,
            carrier=active_carrier,
            checkpoint_guidance_profile=active_checkpoint_guidance,
            atomic_operator_profile=active_operator_profile,
        ),
        "example_index": example_index,
        "task_position": task_position,
        "example_id": task.get("example_id") or task.get("instance_id"),
        "db_id": task.get("db_id"),
        "question": task.get("question"),
        "external_knowledge": task.get("external_knowledge"),
        "mode": mode,
        "teacher_prompt_sha256": prompt_hash(
            mode,
            teacher=True,
            carrier=active_carrier,
            checkpoint_guidance_profile=active_checkpoint_guidance,
            atomic_operator_profile=active_operator_profile,
        ),
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
        "batch_stop_code": batch_stop_code,
        "batch_stop_detail": batch_stop_detail,
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
    if batch_limits is not None:
        record["batch_limits"] = deepcopy(dict(batch_limits))
    if active_commit_eligibility != CHECKPOINT_COMMIT_ELIGIBILITY_NONE:
        record["checkpoint_commit_eligibility_policy"] = active_commit_eligibility
    return record


def build_manifest(
    args: argparse.Namespace,
    *,
    provider_verification: Mapping[str, Any],
    dataset_identity: Mapping[str, Any] | None = None,
    batch_limits: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active_carrier = normalize_carrier(getattr(args, "carrier", DEFAULT_CARRIER))
    active_operator_profile = normalize_atomic_operator_profile(
        getattr(args, "atomic_operator_profile", DEFAULT_ATOMIC_OPERATOR_PROFILE)
    )
    active_checkpoint_guidance = normalize_checkpoint_guidance_profile(
        getattr(
            args,
            "checkpoint_guidance_profile",
            DEFAULT_CHECKPOINT_GUIDANCE_PROFILE,
        )
    )
    active_commit_eligibility = checkpoint_commit_eligibility_for_guidance_profile(
        active_checkpoint_guidance
    )
    experiment_arm = getattr(args, "experiment_arm", None) or carrier_experiment_arm(
        active_carrier
    )
    within_batch_order = getattr(args, "within_batch_order", None)
    if experiment_arm != carrier_experiment_arm(active_carrier):
        raise ValueError("manifest experiment arm does not match carrier")
    if within_batch_order not in {None, "A_then_B", "B_then_A"}:
        raise ValueError("manifest within-batch order is invalid")
    scheme = build_checkpoint_relalg_tool_scheme(
        mode=args.mode,
        carrier=active_carrier,
        atomic_operator_profile=active_operator_profile,
        checkpoint_commit_eligibility_policy=active_commit_eligibility,
    )
    config = RuntimeConfig(
        max_primitive_calls=args.max_primitive_calls,
        max_checkpoints=args.max_checkpoints,
        max_restores=args.max_restores,
        sql_timeout_seconds=args.sql_timeout_seconds,
        max_artifact_rows=args.max_artifact_rows,
        max_artifact_bytes=args.max_artifact_bytes,
        max_cell_bytes=args.max_cell_bytes,
        checkpoint_commit_eligibility_policy=active_commit_eligibility,
    )
    strict_batch = dataset_identity is not None
    manifest = {
        **scheme.manifest_fields(),
        "capability_manifest": capability_manifest(
            args.mode,
            active_carrier,
            active_operator_profile,
            active_commit_eligibility,
        ),
        "backend": BACKEND,
        "dialect": DIALECT,
        "environment_renderer_version": ENVIRONMENT_RENDERER_VERSION,
        "checkpoint_policy_version": CHECKPOINT_POLICY_VERSION,
        "checkpoint_guidance_profile": active_checkpoint_guidance,
        "executor_version": EXECUTOR_VERSION,
        "tool_schema_hash": tool_schema_hash(args.mode, active_operator_profile),
        "carrier_ablation_protocol_version": CARRIER_ABLATION_PROTOCOL_VERSION,
        "carrier_policy_version": CARRIER_POLICY_VERSION,
        "experiment_arm": experiment_arm,
        "within_batch_order": within_batch_order,
        "carrier": active_carrier,
        "prompt_hash": prompt_hash(
            args.mode,
            teacher=True,
            carrier=active_carrier,
            checkpoint_guidance_profile=active_checkpoint_guidance,
            atomic_operator_profile=active_operator_profile,
        ),
        "runner": RUNNER_VERSION if strict_batch else LEGACY_RUNNER_VERSION,
        "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.tasks_json.resolve()),
        "dataset_sha256": _file_sha256(args.tasks_json),
        "start": args.start,
        "requested_size": args.n,
        "model": args.model,
        "provider_verification": dict(provider_verification),
        "provider_request_options": provider_request_audit_options(
            base_url=OFFICIAL_DEEPSEEK_BASE_URL,
            model=args.model,
            carrier=active_carrier,
        ),
        "teacher_prompt_sha256": prompt_hash(
            args.mode,
            teacher=True,
            carrier=active_carrier,
            checkpoint_guidance_profile=active_checkpoint_guidance,
            atomic_operator_profile=active_operator_profile,
        ),
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
    if strict_batch:
        if batch_limits is None:
            raise ValueError("v2 batch manifest requires batch limits")
        manifest.update(deepcopy(dict(dataset_identity)))
        manifest["batch_control_version"] = BATCH_CONTROL_VERSION
        manifest["batch_limits"] = deepcopy(dict(batch_limits))
    if active_commit_eligibility != CHECKPOINT_COMMIT_ELIGIBILITY_NONE:
        manifest["checkpoint_commit_eligibility_policy"] = active_commit_eligibility
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run causal checkpoint-relalg-v1 diagnostics with official DeepSeek."
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--atomic-operator-profile",
        choices=ATOMIC_OPERATOR_PROFILES,
        default=DEFAULT_ATOMIC_OPERATOR_PROFILE,
    )
    parser.add_argument("--carrier", choices=CARRIERS, default=DEFAULT_CARRIER)
    parser.add_argument(
        "--checkpoint-guidance-profile",
        choices=CHECKPOINT_GUIDANCE_PROFILES,
        default=DEFAULT_CHECKPOINT_GUIDANCE_PROFILE,
    )
    parser.add_argument("--experiment-arm", choices=("A", "B"))
    parser.add_argument(
        "--within-batch-order",
        choices=("A_then_B", "B_then_A"),
    )
    parser.add_argument("--tasks-json", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--dataset-manifest", type=Path)
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
    parser.add_argument(
        "--max-batch-provider-attempts",
        type=int,
        default=DEFAULT_MAX_BATCH_PROVIDER_ATTEMPTS,
    )
    parser.add_argument(
        "--max-batch-provider-tokens",
        type=int,
        default=DEFAULT_MAX_BATCH_PROVIDER_TOKENS,
    )
    parser.add_argument(
        "--max-batch-wall-seconds",
        type=float,
        default=DEFAULT_MAX_BATCH_WALL_SECONDS,
    )
    parser.add_argument(
        "--max-consecutive-provider-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_PROVIDER_FAILURES,
    )
    parser.add_argument(
        "--max-total-provider-failures",
        type=int,
        default=DEFAULT_MAX_TOTAL_PROVIDER_FAILURES,
    )
    parser.add_argument(
        "--max-consecutive-semantic-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_SEMANTIC_FAILURES,
    )
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
    if args.dataset_manifest is None:
        raise SystemExit("--dataset-manifest is required for v2 diagnostic runs")
    if args.atomic_operator_profile != DEFAULT_ATOMIC_OPERATOR_PROFILE and args.mode != "atomic":
        raise SystemExit("semantic-v2 is currently isolated to --mode atomic")
    expected_arm = carrier_experiment_arm(args.carrier)
    if args.experiment_arm is not None and args.experiment_arm != expected_arm:
        raise SystemExit(
            f"--experiment-arm must be {expected_arm} for carrier {args.carrier}"
        )
    try:
        runtime_config = RuntimeConfig(
            max_primitive_calls=args.max_primitive_calls,
            max_checkpoints=args.max_checkpoints,
            max_restores=args.max_restores,
            sql_timeout_seconds=args.sql_timeout_seconds,
            max_artifact_rows=args.max_artifact_rows,
            max_artifact_bytes=args.max_artifact_bytes,
            max_cell_bytes=args.max_cell_bytes,
            checkpoint_commit_eligibility_policy=(
                checkpoint_commit_eligibility_for_guidance_profile(
                    args.checkpoint_guidance_profile
                )
            ),
        )
        batch_limits = _batch_limits_payload(args)
    except ValueError as exc:
        raise SystemExit(f"invalid resource limits: {exc}") from exc
    tasks = _load_tasks(args.tasks_json)
    selected = list(enumerate(tasks))[args.start : args.start + args.n]
    if len(selected) != args.n:
        raise SystemExit("selected task slice does not contain --n certified tasks")
    selected_indices = [
        _task_example_index(task, position) for position, task in selected
    ]
    if len(set(selected_indices)) != len(selected_indices):
        raise SystemExit("selected task slice has duplicate example_index values")
    try:
        dataset_identity = _build_dataset_identity(
            args.tasks_json,
            args.dataset_manifest,
            tasks,
            start=args.start,
            requested_size=args.n,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid dataset identity: {exc}") from exc
    identity_fields = _record_batch_identity_fields(
        dataset_identity,
        start=args.start,
        requested_size=args.n,
    )

    if args.dry_run:
        print(json.dumps({
            "tool_scheme": SCHEME,
            "protocol_version": PROTOCOL_VERSION,
            "mode": args.mode,
            "atomic_operator_profile": args.atomic_operator_profile,
            "carrier": args.carrier,
            "carrier_ablation_protocol_version": CARRIER_ABLATION_PROTOCOL_VERSION,
            "carrier_policy_version": CARRIER_POLICY_VERSION,
            "checkpoint_guidance_profile": args.checkpoint_guidance_profile,
            "experiment_arm": args.experiment_arm or expected_arm,
            "within_batch_order": args.within_batch_order,
            "tasks": [position for position, _ in selected],
            "dataset_manifest_sha256": dataset_identity[
                "dataset_manifest_sha256"
            ],
            "dataset_source_identity_sha256": dataset_identity[
                "dataset_source_identity"
            ]["dataset_source_identity_sha256"],
            "selection_identity_sha256": dataset_identity[
                "selection_identity"
            ]["selection_identity_sha256"],
            "batch_control_version": BATCH_CONTROL_VERSION,
            "batch_limits": batch_limits,
            "tool_schema_sha256": tool_schema_hash(
                args.mode, args.atomic_operator_profile
            ),
            "teacher_prompt_sha256": prompt_hash(
                args.mode,
                teacher=True,
                carrier=args.carrier,
                checkpoint_guidance_profile=args.checkpoint_guidance_profile,
                atomic_operator_profile=args.atomic_operator_profile,
            ),
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
        carrier=args.carrier,
    )
    verification = client.verify_model()
    manifest = build_manifest(
        args,
        provider_verification=verification,
        dataset_identity=dataset_identity,
        batch_limits=batch_limits,
    )
    writer = _open_artifact_writer(
        args.result_dir,
        manifest,
        resume=args.resume,
    )
    _ensure_journal(writer.all_path)
    records = _load_committed_records(writer.all_path)
    _validate_committed_records(
        records,
        selected,
        identity_fields=identity_fields,
        batch_limits=batch_limits,
    )
    manifest_config_sha256 = _stored_manifest_config_sha256(writer.manifest_path)
    existing_status = _load_batch_status(
        args.result_dir,
        records=records,
        requested_size=args.n,
        limits=batch_limits,
        manifest_config_sha256=manifest_config_sha256,
    )
    counters = _batch_counters_from_records(records)
    prior_wall = (
        existing_status["counters"]["total_wall_seconds"]
        if existing_status is not None
        else counters["total_wall_seconds"]
    )
    guard = BatchRequestGuard(
        counters=counters,
        limits=batch_limits,
        base_wall_seconds=prior_wall,
    )
    counters = _merge_wall_counter(counters, guard)

    if existing_status is not None and existing_status["state"] == "completed":
        summary = writer.summarize()
        audit = audit_result_dir(args.result_dir)
        print(json.dumps({"summary": summary, "audit": audit}, ensure_ascii=False, sort_keys=True))
        return 0 if audit["passed"] else 1

    stop = _batch_stop_decision(records, counters, batch_limits)
    if existing_status is not None and existing_status["state"] == "stopped":
        if stop is None or stop[0] != existing_status.get("stop_code"):
            raise SystemExit("stored stopped batch is not supported by committed records")
    if stop is not None:
        _write_batch_status(
            args.result_dir,
            state="stopped",
            stop_code=stop[0],
            stop_detail=stop[1],
            requested_size=args.n,
            records=records,
            counters=counters,
            limits=batch_limits,
            manifest_config_sha256=manifest_config_sha256,
        )
        summary = writer.summarize()
        audit = audit_result_dir(args.result_dir)
        print(json.dumps({"summary": summary, "audit": audit}, ensure_ascii=False, sort_keys=True))
        return 2

    _write_batch_status(
        args.result_dir,
        state="running",
        stop_code=None,
        stop_detail=None,
        requested_size=args.n,
        records=records,
        counters=counters,
        limits=batch_limits,
        manifest_config_sha256=manifest_config_sha256,
    )
    stop = None
    try:
        for position, task in selected[len(records) :]:
            pre_request_stop = guard.stop_decision()
            if pre_request_stop is not None:
                stop = pre_request_stop
                break
            try:
                record = run_episode(
                    task,
                    task_position=position,
                    mode=args.mode,
                    atomic_operator_profile=args.atomic_operator_profile,
                    client=client,
                    carrier=args.carrier,
                    checkpoint_guidance_profile=args.checkpoint_guidance_profile,
                    experiment_arm=args.experiment_arm,
                    within_batch_order=args.within_batch_order,
                    runtime_config=runtime_config,
                    max_model_turns=args.max_model_turns,
                    max_tokens=args.max_tokens,
                    max_completion_tokens=args.max_completion_tokens,
                    api_retries=args.api_retries,
                    artifact_identity_fields=identity_fields,
                    batch_limits=batch_limits,
                    batch_guard=guard,
                )
            except BatchRunLimitReached as exc:
                stop = (exc.stop_code, exc.stop_detail)
                break
            writer.append(record)
            records = _load_committed_records(writer.all_path)
            counters = _merge_wall_counter(
                _batch_counters_from_records(records),
                guard,
            )
            stop = _batch_stop_decision(records, counters, batch_limits)
            flag = "PASS" if record["correct"] else "FAIL"
            print(
                f"[{flag}] {record.get('example_id') or record['example_index']} "
                f"mode={args.mode} carrier={args.carrier} "
                f"turns={record['steps']} errors={record['errors']}"
            )
            if stop is not None and len(records) < args.n:
                break
            _write_batch_status(
                args.result_dir,
                state="running",
                stop_code=None,
                stop_detail=None,
                requested_size=args.n,
                records=records,
                counters=counters,
                limits=batch_limits,
                manifest_config_sha256=manifest_config_sha256,
            )
    except BaseException as exc:
        records = _load_committed_records(writer.all_path)
        counters = _merge_wall_counter(
            _batch_counters_from_records(records),
            guard,
        )
        _write_batch_status(
            args.result_dir,
            state="incomplete",
            stop_code="runner_interrupted",
            stop_detail={"exception_type": type(exc).__name__},
            requested_size=args.n,
            records=records,
            counters=counters,
            limits=batch_limits,
            manifest_config_sha256=manifest_config_sha256,
        )
        raise

    records = _load_committed_records(writer.all_path)
    counters = _merge_wall_counter(_batch_counters_from_records(records), guard)
    completed = len(records) == args.n and not any(
        record.get("failure_type") == "batch_limit_reached" for record in records
    )
    if completed:
        stop = None
        state = "completed"
    else:
        if stop is None:
            stop = _batch_stop_decision(records, counters, batch_limits)
        if stop is None:
            raise RuntimeError("batch ended without completion or a stop condition")
        state = "stopped"
    _write_batch_status(
        args.result_dir,
        state=state,
        stop_code=stop[0] if stop is not None else None,
        stop_detail=stop[1] if stop is not None else None,
        requested_size=args.n,
        records=records,
        counters=counters,
        limits=batch_limits,
        manifest_config_sha256=manifest_config_sha256,
    )
    summary = writer.summarize()
    audit = audit_result_dir(args.result_dir)
    print(json.dumps({"summary": summary, "audit": audit}, ensure_ascii=False, sort_keys=True))
    if state != "completed":
        return 2
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
