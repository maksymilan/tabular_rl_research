#!/usr/bin/env python3
"""Deterministic, content-blind matched carrier A/B result aggregation.

The analyzer intentionally never inspects questions, gold data, assistant
reasoning/content, tool arguments, answer rows, or database values.  Text-JSON
turns must provide the normalized, content-free ``carrier_metrics`` written by
their runner.  Native turns can additionally be summarized from tool-call
cardinality, without looking inside any call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .protocol import (
    CARRIER_NATIVE_TOOL_CALLS,
    CARRIER_TEXT_JSON,
    assistant_carrier_protocol,
    capability_manifest,
    carrier_experiment_arm,
    prompt_hash,
    tool_schema_hash,
)
from .provider import provider_response_envelope_version
from ..registry import build_checkpoint_relalg_tool_scheme


SCHEMA_VERSION = "checkpoint-carrier-ab-analysis-v1"
TOKEN_FIELDS = (
    "prompt_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "completion_tokens",
    "completion_tokens_details.reasoning_tokens",
    "total_tokens",
)
COMMON_MANIFEST_FIELDS = (
    "dataset_sha256",
    "tool_scheme_registry_version",
    "tool_scheme",
    "protocol_version",
    "mode",
    "model",
    "runner",
    "backend",
    "dialect",
    "environment_renderer_version",
    "checkpoint_policy_version",
    "executor_version",
    "runtime_config",
    "max_tokens",
    "max_completion_tokens",
    "api_retries",
    "api_timeout_seconds",
    "workers",
    "denotation_comparison",
    "strict_artifact_audit_version",
    "carrier_ablation_protocol_version",
    "carrier_policy_version",
    "causal_model_harness_loop",
    "hidden_reference_visible_to_model",
    "admission_status",
    "sft_export_eligible",
    "rl_admission_eligible",
)
_CAPABILITY_CARRIER_FIELDS = frozenset(
    {
        "assistant_carrier",
        "single_tool_call_per_turn",
        "max_calls_per_turn",
        "min_tool_calls_per_turn",
        "max_tool_calls_per_turn",
        "single_action_per_turn",
        "min_actions_per_turn",
        "max_actions_per_turn",
        "student_prompt_sha256",
        "teacher_prompt_sha256",
        "carrier_ablation_protocol_version",
        "carrier_policy_version",
        "experiment_arm",
        "carrier",
    }
)
RECORD_MANIFEST_FIELDS = (
    "tool_scheme_registry_version",
    "tool_scheme",
    "protocol_version",
    "protocol_hash",
    "assistant_carrier",
    "top_level_tools",
    "atomic_tools",
    "max_batch_calls",
    "mode",
    "tool_schema_sha256",
    "student_prompt_sha256",
    "teacher_prompt_sha256",
    "capability_manifest",
    "backend",
    "dialect",
    "environment_renderer_version",
    "checkpoint_policy_version",
    "executor_version",
    "tool_schema_hash",
    "carrier_ablation_protocol_version",
    "carrier_policy_version",
    "provider_response_envelope_version",
    "experiment_arm",
    "carrier",
    "prompt_hash",
    "provider_request_options",
    "runtime_config",
    "denotation_comparison",
    "strict_artifact_audit_version",
)
PROVIDER_ATTEMPT_CATEGORIES = (
    "accepted_response",
    "missing_tool_calls",
    "empty_text",
    "provider_shape",
    "completion_length",
    "transport",
    "context_overflow",
    "model_mismatch",
    "insufficient_system_resource",
    "content_filter",
)


class CarrierAnalysisError(ValueError):
    """Input artifacts are incomplete, unaudited, or not truly matched."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_int(value: Any, *, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CarrierAnalysisError(f"{path} must be an integer >= {minimum}")
    return value


def _optional_number(value: Any, *, path: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CarrierAnalysisError(f"{path} must be numeric or null")
    if not math.isfinite(float(value)) or value < 0:
        raise CarrierAnalysisError(f"{path} must be finite and non-negative")
    return value


def _nested_optional_value(
    value: Mapping[str, Any],
    dotted_path: str,
    *,
    path: str,
) -> Any:
    # Provider/runner audit artifacts use flattened dotted counters.  Retain
    # nested compatibility for older/synthetic artifacts without ever walking
    # unrelated provider response payloads.
    if dotted_path in value:
        return value[dotted_path]
    current: Any = value
    traversed: list[str] = []
    for field in dotted_path.split("."):
        if not isinstance(current, Mapping):
            location = ".".join(traversed)
            raise CarrierAnalysisError(f"{path}.{location} must be an object")
        if field not in current:
            return None
        current = current[field]
        traversed.append(field)
    return current


def _tool_schema_hash(manifest: Mapping[str, Any]) -> Any:
    return manifest.get("tool_schema_hash", manifest.get("tool_schema_sha256"))


def _common_capability(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _common_capability(child)
            for key, child in value.items()
            if key not in _CAPABILITY_CARRIER_FIELDS
        }
    if isinstance(value, list):
        return [_common_capability(child) for child in value]
    return value


def _manifest_signature(manifest: Mapping[str, Any]) -> dict[str, Any]:
    signature = {field: manifest.get(field) for field in COMMON_MANIFEST_FIELDS}
    signature["tool_schema_hash"] = _tool_schema_hash(manifest)
    signature["capability_manifest"] = _common_capability(
        manifest.get("capability_manifest")
    )
    options = manifest.get("provider_request_options")
    signature["provider_request_options"] = (
        {
            key: value
            for key, value in options.items()
            if key
            not in {
                "carrier",
                "assistant_carrier",
                "response_format",
                "tool_choice",
                "tools",
            }
        }
        if isinstance(options, Mapping)
        else options
    )
    return signature


def _validate_carrier_manifest(
    manifest: Mapping[str, Any],
    *,
    label: str,
    carrier: str,
    expected_mode: str,
    expected_model: str,
) -> None:
    expected_response_envelope_version = provider_response_envelope_version(carrier)
    if (
        manifest.get("provider_response_envelope_version")
        != expected_response_envelope_version
    ):
        raise CarrierAnalysisError(
            f"arm {label} provider_response_envelope_version differs from "
            "the current carrier response envelope"
        )
    expected_scheme = build_checkpoint_relalg_tool_scheme(
        mode=expected_mode,
        carrier=carrier,
    ).manifest_fields()
    for field, expected in expected_scheme.items():
        if _canonical(manifest.get(field)) != _canonical(expected):
            raise CarrierAnalysisError(
                f"arm {label} manifest.{field} differs from current tool scheme"
            )
    if _canonical(manifest.get("capability_manifest")) != _canonical(
        capability_manifest(expected_mode, carrier)
    ):
        raise CarrierAnalysisError(
            f"arm {label} capability_manifest differs from current protocol"
        )
    if manifest.get("tool_schema_hash") != tool_schema_hash(expected_mode):
        raise CarrierAnalysisError(
            f"arm {label} tool_schema_hash differs from current protocol"
        )
    expected_teacher_hash = prompt_hash(
        expected_mode,
        teacher=True,
        carrier=carrier,
    )
    if (
        manifest.get("prompt_hash") != expected_teacher_hash
        or manifest.get("teacher_prompt_sha256") != expected_teacher_hash
    ):
        raise CarrierAnalysisError(
            f"arm {label} teacher prompt hash differs from current protocol"
        )
    options = manifest.get("provider_request_options")
    if not isinstance(options, Mapping):
        raise CarrierAnalysisError(f"arm {label} has no provider_request_options")
    expected_common = {
        "endpoint": "https://api.deepseek.com/chat/completions",
        "model": expected_model,
        "carrier": carrier,
        "assistant_carrier": assistant_carrier_protocol(carrier),
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }
    for key, expected in expected_common.items():
        if options.get(key) != expected:
            raise CarrierAnalysisError(
                f"arm {label} provider_request_options.{key} differs from carrier policy"
            )
    if carrier == CARRIER_TEXT_JSON:
        if options.get("response_format") != {"type": "json_object"}:
            raise CarrierAnalysisError("arm A must use JSON Output")
        if "tool_choice" in options:
            raise CarrierAnalysisError("arm A must not send tool_choice")
    else:
        if options.get("tool_choice") != "auto":
            raise CarrierAnalysisError("arm B must use tool_choice=auto")
        if "response_format" in options:
            raise CarrierAnalysisError("arm B must not send response_format")
    verification = manifest.get("provider_verification")
    if (
        not isinstance(verification, Mapping)
        or verification.get("authenticated") is not True
        or verification.get("model_available") is not True
        or verification.get("model") != expected_model
    ):
        raise CarrierAnalysisError(f"arm {label} provider model was not verified")


def _validate_audit(directory: Path, *, expected_records: int) -> str:
    path = directory / "checkpoint_relalg_audit.json"
    if not path.is_file():
        raise CarrierAnalysisError(f"missing audit artifact: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    fresh = audit.get("fresh_replay")
    binding = audit.get("manifest_binding")
    expected_hashes = {
        "manifest.json": _file_sha256(directory / "manifest.json"),
        "all.jsonl": _file_sha256(directory / "all.jsonl"),
    }
    if (
        audit.get("passed") is not True
        or audit.get("issue_count") != 0
        or audit.get("records") != expected_records
        or audit.get("artifact_sha256") != expected_hashes
        or not isinstance(binding, Mapping)
        or binding.get("passed") is not True
        or binding.get("issue_count") != 0
        or not isinstance(fresh, Mapping)
        or fresh.get("passed") is not True
        or fresh.get("issue_count") != 0
        or fresh.get("records") != expected_records
    ):
        raise CarrierAnalysisError(f"audit/fresh replay did not pass exactly: {path}")
    return _file_sha256(path)


def _validate_record_manifest_binding(
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    path: str,
) -> None:
    """Bind semantic/runtime identity without trusting scheduling labels."""

    for field in RECORD_MANIFEST_FIELDS:
        if _canonical(record.get(field)) != _canonical(manifest.get(field)):
            raise CarrierAnalysisError(f"{path}.{field} differs from manifest")


def _run_current_audit(
    directory: Path,
    records: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> str:
    """Run the current structural and replay auditors without rewriting artifacts."""

    from .audit import audit_records, fresh_replay_records

    structural = audit_records(records)
    dataset = manifest.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise CarrierAnalysisError("current fresh replay requires manifest.dataset")
    tasks_path = Path(dataset)
    if not tasks_path.is_file():
        raise CarrierAnalysisError("current fresh replay dataset is unavailable")
    expected_dataset_hash = manifest.get("dataset_sha256")
    if (
        not isinstance(expected_dataset_hash, str)
        or _file_sha256(tasks_path) != expected_dataset_hash
    ):
        raise CarrierAnalysisError("fresh replay dataset hash differs from manifest")
    with tasks_path.open(encoding="utf-8") as handle:
        if tasks_path.suffix == ".jsonl":
            tasks = [json.loads(line) for line in handle if line.strip()]
        else:
            tasks = json.load(handle)
    replay = fresh_replay_records(records, tasks)
    if (
        structural.get("passed") is not True
        or structural.get("issue_count") != 0
        or structural.get("records") != len(records)
        or replay.get("passed") is not True
        or replay.get("issue_count") != 0
        or replay.get("records") != len(records)
    ):
        # Do not surface audit issue text: it can contain model-authored data.
        raise CarrierAnalysisError(
            f"current structural audit/fresh replay failed for {directory}"
        )
    payload = {"structural": structural, "fresh_replay": replay}
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EpisodeMetrics:
    task_position: int
    task_id: str
    correct: bool
    legal: bool
    strict_correct: bool | None
    schema_match: bool | None
    model_turns: int
    authored_responses: int
    provider_attempts: int
    provider_attempt_categories: Mapping[str, int]
    primitive_calls: int
    successful_actions: int
    error_actions: int
    carrier_rejections: int
    semantic_protocol_errors: int
    argument_validation_errors: int
    state_validation_errors: int
    execution_errors: int
    protocol_errors: int
    protocol_error_codes: Mapping[str, int]
    multi_action_turns: int
    zero_action_turns: int
    exact_single_action_turns: int
    carrier_valid_turns: int
    recovered_after_carrier_rejection: bool
    carrier_rejection_then_legal: bool
    carrier_rejection_then_correct: bool
    restore_count: int
    provider_failure: bool
    context_overflow: bool
    usage: Mapping[str, int | None]
    provider_elapsed_seconds: float | None
    elapsed_seconds: float | None

    @property
    def carrier_clean(self) -> bool:
        return self.carrier_rejections == 0


@dataclass(frozen=True)
class ArmData:
    label: str
    carrier: str
    records: tuple[EpisodeMetrics, ...]
    manifest_signature: Mapping[str, Any]
    input_hashes: tuple[Mapping[str, str], ...]
    batches: tuple[Mapping[str, Any], ...]


def _carrier_turn_metrics(
    turn: Mapping[str, Any],
    *,
    carrier: str,
    path: str,
) -> tuple[bool, int, bool, bool, str | None]:
    """Return present/count/exact-one/envelope-valid/error-code without content access."""

    normalized = turn.get("carrier_metrics")
    if not isinstance(normalized, Mapping):
        raise CarrierAnalysisError(f"{path} is missing content-free carrier_metrics")
    if normalized.get("carrier") != carrier:
        raise CarrierAnalysisError(f"{path}.carrier_metrics.carrier differs from manifest")
    present = normalized.get("provider_response_present") is True
    count = _required_int(
        normalized.get("authored_action_count", 0),
        path=f"{path}.carrier_metrics.authored_action_count",
    )
    exact_one = normalized.get("exact_single_action") is True
    valid = normalized.get("carrier_envelope_valid") is True
    code = normalized.get("carrier_error_code")
    if code is not None and not isinstance(code, str):
        raise CarrierAnalysisError(
            f"{path}.carrier_metrics.carrier_error_code must be a string or null"
        )
    return present, count, exact_one, valid, code


def _episode_metrics(record: Mapping[str, Any], *, carrier: str, path: str) -> EpisodeMetrics:
    position = _required_int(record.get("task_position"), path=f"{path}.task_position")
    raw_task_id = record.get("example_id", record.get("example_index"))
    if isinstance(raw_task_id, bool) or not isinstance(raw_task_id, (str, int)):
        raise CarrierAnalysisError(f"{path} requires a string/integer task id")
    task_id = str(raw_task_id)
    turns = record.get("turns")
    if not isinstance(turns, list):
        raise CarrierAnalysisError(f"{path}.turns must be an array")

    authored_responses = 0
    provider_attempts = 0
    provider_attempt_categories: Counter[str] = Counter()
    successful_actions = 0
    error_actions = 0
    carrier_rejections = 0
    semantic_protocol_errors = 0
    argument_validation_errors = 0
    state_validation_errors = 0
    execution_errors = 0
    protocol_errors = 0
    protocol_codes: Counter[str] = Counter()
    multi_action_turns = 0
    zero_action_turns = 0
    exact_single_turns = 0
    carrier_valid_turns = 0
    saw_rejection = False
    recovered = False
    provider_elapsed_total = 0.0
    provider_elapsed_available = True
    saw_provider_attempt = False

    for turn_index, raw_turn in enumerate(turns):
        turn_path = f"{path}.turns[{turn_index}]"
        if not isinstance(raw_turn, Mapping):
            raise CarrierAnalysisError(f"{turn_path} must be an object")
        events = raw_turn.get("provider_retry_events", [])
        if not isinstance(events, list):
            raise CarrierAnalysisError(f"{turn_path}.provider_retry_events must be an array")
        normalized_metrics = raw_turn.get("carrier_metrics")
        if not isinstance(normalized_metrics, Mapping):
            raise CarrierAnalysisError(
                f"{turn_path} is missing content-free carrier_metrics"
            )
        exact_attempts = _required_int(
            normalized_metrics.get("provider_attempt_count"),
            path=f"{turn_path}.carrier_metrics.provider_attempt_count",
            minimum=1,
        )
        if exact_attempts != len(events) + 1:
            raise CarrierAnalysisError(
                f"{turn_path}.carrier_metrics.provider_attempt_count differs from retry history"
            )
        attempt_events = raw_turn.get("provider_attempt_events")
        if not isinstance(attempt_events, list) or len(attempt_events) != exact_attempts:
            raise CarrierAnalysisError(
                f"{turn_path}.provider_attempt_events does not match provider attempts"
            )
        for event_index, event in enumerate(attempt_events):
            if not isinstance(event, Mapping):
                raise CarrierAnalysisError(
                    f"{turn_path}.provider_attempt_events[{event_index}] must be an object"
                )
            category = event.get("shape_category")
            if category not in PROVIDER_ATTEMPT_CATEGORIES:
                raise CarrierAnalysisError(
                    f"{turn_path}.provider_attempt_events[{event_index}].shape_category is invalid"
                )
            provider_attempt_categories[category] += 1
        provider_attempts += exact_attempts
        saw_provider_attempt = True
        raw_provider_elapsed = normalized_metrics.get("provider_elapsed_seconds")
        provider_elapsed = _optional_number(
            raw_provider_elapsed,
            path=f"{turn_path}.carrier_metrics.provider_elapsed_seconds",
        )
        if provider_elapsed is None:
            provider_elapsed_available = False
        else:
            provider_elapsed_total += float(provider_elapsed)

        present, action_count, exact_one, carrier_valid, carrier_code = (
            _carrier_turn_metrics(raw_turn, carrier=carrier, path=turn_path)
        )
        if present:
            authored_responses += 1
            multi_action_turns += int(action_count > 1)
            zero_action_turns += int(action_count == 0)
            exact_single_turns += int(exact_one)
            carrier_valid_turns += int(carrier_valid)
        result = raw_turn.get("result")
        if isinstance(result, Mapping):
            if result.get("status") == "success":
                successful_actions += 1
            elif result.get("status") == "error":
                error_actions += 1
            error = result.get("error")
            if isinstance(error, Mapping) and error.get("type") == "protocol_error":
                protocol_errors += 1
                code = error.get("code")
                if isinstance(code, str):
                    protocol_codes[code] += 1
                if carrier_valid:
                    semantic_protocol_errors += 1
            if isinstance(error, Mapping):
                error_type = error.get("type")
                argument_validation_errors += int(error_type == "argument_validation_error")
                state_validation_errors += int(error_type == "state_validation_error")
                execution_errors += int(
                    error_type in {"execution_error", "nonrecoverable_execution_error"}
                )
        # A carrier rejection is exclusively an authored response whose
        # transport envelope/cardinality is invalid.  A valid one-action
        # envelope with unknown tool/arguments is a semantic protocol error.
        rejected = present and not carrier_valid
        if rejected:
            carrier_rejections += 1
            saw_rejection = True
        elif saw_rejection and carrier_valid:
            recovered = True

    usage_value = record.get("provider_usage")
    usage = usage_value if isinstance(usage_value, Mapping) else {}
    normalized_usage: dict[str, int | None] = {}
    for field in TOKEN_FIELDS:
        value = _nested_optional_value(
            usage,
            field,
            path=f"{path}.provider_usage",
        )
        if value is None:
            normalized_usage[field] = None
        else:
            normalized_usage[field] = _required_int(
                value, path=f"{path}.provider_usage.{field}"
            )
    elapsed = _optional_number(record.get("elapsed_seconds"), path=f"{path}.elapsed_seconds")
    strict = record.get("strict_artifact_accuracy")
    schema_match = record.get("schema_match")
    if strict is not None and not isinstance(strict, bool):
        raise CarrierAnalysisError(f"{path}.strict_artifact_accuracy must be boolean/null")
    if schema_match is not None and not isinstance(schema_match, bool):
        raise CarrierAnalysisError(f"{path}.schema_match must be boolean/null")
    correct = record.get("correct")
    legal = record.get("legal")
    if not isinstance(correct, bool):
        raise CarrierAnalysisError(f"{path}.correct must be boolean")
    if not isinstance(legal, bool):
        raise CarrierAnalysisError(f"{path}.legal must be boolean")
    return EpisodeMetrics(
        task_position=position,
        task_id=task_id,
        correct=correct,
        legal=legal,
        strict_correct=strict,
        schema_match=schema_match,
        model_turns=len(turns),
        authored_responses=authored_responses,
        provider_attempts=provider_attempts,
        provider_attempt_categories={
            category: provider_attempt_categories.get(category, 0)
            for category in PROVIDER_ATTEMPT_CATEGORIES
        },
        primitive_calls=_required_int(
            record.get("primitive_calls", 0), path=f"{path}.primitive_calls"
        ),
        successful_actions=successful_actions,
        error_actions=error_actions,
        carrier_rejections=carrier_rejections,
        semantic_protocol_errors=semantic_protocol_errors,
        argument_validation_errors=argument_validation_errors,
        state_validation_errors=state_validation_errors,
        execution_errors=execution_errors,
        protocol_errors=protocol_errors,
        protocol_error_codes=dict(sorted(protocol_codes.items())),
        multi_action_turns=multi_action_turns,
        zero_action_turns=zero_action_turns,
        exact_single_action_turns=exact_single_turns,
        carrier_valid_turns=carrier_valid_turns,
        recovered_after_carrier_rejection=recovered,
        carrier_rejection_then_legal=saw_rejection and recovered and legal,
        carrier_rejection_then_correct=saw_rejection and recovered and correct,
        restore_count=_required_int(
            record.get("restore_count", 0), path=f"{path}.restore_count"
        ),
        provider_failure=record.get("failure_type") == "provider_error",
        context_overflow=record.get("failure_type") == "context_length_exceeded",
        usage=normalized_usage,
        provider_elapsed_seconds=(
            provider_elapsed_total
            if saw_provider_attempt and provider_elapsed_available
            else None
        ),
        elapsed_seconds=float(elapsed) if elapsed is not None else None,
    )


def load_arm(
    label: str,
    directories: Sequence[Path],
    *,
    expected_mode: str,
    expected_model: str,
) -> ArmData:
    if not directories:
        raise CarrierAnalysisError(f"arm {label} has no result directories")
    records: list[EpisodeMetrics] = []
    hashes: list[Mapping[str, str]] = []
    batches: list[Mapping[str, Any]] = []
    signature: Mapping[str, Any] | None = None
    carrier: str | None = None
    prior_position = -1
    expected_carrier = CARRIER_TEXT_JSON if label == "A" else CARRIER_NATIVE_TOOL_CALLS
    for batch_index, directory in enumerate(directories):
        manifest_path = directory / "manifest.json"
        records_path = directory / "all.jsonl"
        if not manifest_path.is_file() or not records_path.is_file():
            raise CarrierAnalysisError(f"incomplete result directory: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_signature = _manifest_signature(manifest)
        if signature is None:
            signature = current_signature
        elif _canonical(signature) != _canonical(current_signature):
            raise CarrierAnalysisError(f"arm {label} batch manifests are not matched")
        current_carrier = manifest.get("carrier")
        if current_carrier != expected_carrier:
            raise CarrierAnalysisError(
                f"arm {label} must declare manifest.carrier={expected_carrier!r}"
            )
        expected_assistant_protocol = assistant_carrier_protocol(current_carrier)
        if manifest.get("assistant_carrier") != expected_assistant_protocol:
            raise CarrierAnalysisError(
                f"arm {label} assistant_carrier does not match manifest.carrier"
            )
        if manifest.get("experiment_arm") != carrier_experiment_arm(current_carrier):
            raise CarrierAnalysisError(
                f"arm {label} experiment_arm does not match manifest.carrier"
            )
        _validate_carrier_manifest(
            manifest,
            label=label,
            carrier=current_carrier,
            expected_mode=expected_mode,
            expected_model=expected_model,
        )
        if carrier is None:
            carrier = current_carrier
        elif carrier != current_carrier:
            raise CarrierAnalysisError(f"arm {label} mixes carrier identities")
        if manifest.get("mode") != expected_mode or manifest.get("model") != expected_model:
            raise CarrierAnalysisError(
                f"arm {label} must use mode={expected_mode!r}, model={expected_model!r}"
            )
        raw_records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        requested = manifest.get("requested_size")
        if requested is not None and requested != len(raw_records):
            raise CarrierAnalysisError(f"arm {label} batch is incomplete: {directory}")
        current_audit_hash = _run_current_audit(directory, raw_records, manifest)
        audit_hash = _validate_audit(directory, expected_records=len(raw_records))
        batch_positions: list[int] = []
        for record_index, record in enumerate(raw_records):
            if not isinstance(record, Mapping):
                raise CarrierAnalysisError(f"arm {label} record is not an object")
            record_path = f"arm_{label}[{len(records)}]"
            _validate_record_manifest_binding(
                record,
                manifest,
                path=record_path,
            )
            metric = _episode_metrics(
                record,
                carrier=current_carrier,
                path=record_path,
            )
            if metric.task_position <= prior_position:
                raise CarrierAnalysisError(
                    f"arm {label} task positions must be unique and strictly increasing"
                )
            prior_position = metric.task_position
            batch_positions.append(metric.task_position)
            records.append(metric)
        hashes.append(
            {
                "manifest_sha256": _file_sha256(manifest_path),
                "records_sha256": _file_sha256(records_path),
                "audit_sha256": audit_hash,
                "current_audit_sha256": current_audit_hash,
            }
        )
        started_at = manifest.get("run_started_at_utc")
        if started_at is not None and not isinstance(started_at, str):
            raise CarrierAnalysisError(
                f"arm {label} run_started_at_utc must be a string or null"
            )
        batches.append(
            {
                "input_index": batch_index,
                "result_dir": str(directory.resolve()),
                "task_positions": batch_positions,
                "run_started_at_utc": started_at,
            }
        )
    assert signature is not None and carrier is not None
    return ArmData(
        label=label,
        carrier=carrier,
        records=tuple(records),
        manifest_signature=signature,
        input_hashes=tuple(hashes),
        batches=tuple(batches),
    )


def _mean(values: Sequence[float | int]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _p90(values: Sequence[float | int]) -> float | int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.90 * len(ordered)) - 1)]


def _numeric_summary(values: Iterable[float | int | None]) -> dict[str, Any]:
    present = [value for value in values if value is not None]
    return {
        "available": len(present),
        "total": sum(present) if present else None,
        "mean": _mean(present),
        "median": _median(present),
        "p90": _p90(present),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _arm_summary(arm: ArmData) -> dict[str, Any]:
    records = arm.records
    code_counts: Counter[str] = Counter()
    for record in records:
        code_counts.update(record.protocol_error_codes)
    attempt_categories = {
        category: sum(
            record.provider_attempt_categories.get(category, 0) for record in records
        )
        for category in PROVIDER_ATTEMPT_CATEGORIES
    }
    single = sum(record.exact_single_action_turns for record in records)
    authored = sum(record.authored_responses for record in records)
    carrier_valid = sum(record.carrier_valid_turns for record in records)
    return {
        "carrier": arm.carrier,
        "episodes": len(records),
        "correct": sum(record.correct for record in records),
        "legal": sum(record.legal for record in records),
        "strict_artifact_correct": sum(record.strict_correct is True for record in records),
        "strict_artifact_available": sum(record.strict_correct is not None for record in records),
        "schema_match": sum(record.schema_match is True for record in records),
        "schema_match_available": sum(record.schema_match is not None for record in records),
        "model_turns": sum(record.model_turns for record in records),
        "authored_responses": authored,
        "provider_attempts": sum(record.provider_attempts for record in records),
        "provider_attempt_categories": attempt_categories,
        "primitive_calls": sum(record.primitive_calls for record in records),
        "successful_actions": sum(record.successful_actions for record in records),
        "error_actions": sum(record.error_actions for record in records),
        "exact_single_action_turns": single,
        "single_action_adherence": _rate(single, authored),
        "carrier_valid_turns": carrier_valid,
        "carrier_valid_rate": _rate(carrier_valid, authored),
        "multi_action_turns": sum(record.multi_action_turns for record in records),
        "multi_action_episodes": sum(record.multi_action_turns > 0 for record in records),
        "zero_action_turns": sum(record.zero_action_turns for record in records),
        "carrier_rejections": sum(record.carrier_rejections for record in records),
        "carrier_rejection_episodes": sum(not record.carrier_clean for record in records),
        "protocol_errors": sum(record.protocol_errors for record in records),
        "semantic_protocol_errors": sum(
            record.semantic_protocol_errors for record in records
        ),
        "argument_validation_errors": sum(
            record.argument_validation_errors for record in records
        ),
        "state_validation_errors": sum(
            record.state_validation_errors for record in records
        ),
        "execution_errors": sum(record.execution_errors for record in records),
        "protocol_error_codes": dict(sorted(code_counts.items())),
        "recovered_after_carrier_rejection": sum(
            record.recovered_after_carrier_rejection for record in records
        ),
        "carrier_rejection_then_legal": sum(
            record.carrier_rejection_then_legal for record in records
        ),
        "carrier_rejection_then_correct": sum(
            record.carrier_rejection_then_correct for record in records
        ),
        "restore_count": sum(record.restore_count for record in records),
        "restore_episodes": sum(record.restore_count > 0 for record in records),
        "provider_failures": sum(record.provider_failure for record in records),
        "context_overflows": sum(record.context_overflow for record in records),
        "tokens": {
            field: _numeric_summary(record.usage[field] for record in records)
            for field in TOKEN_FIELDS
        },
        "provider_elapsed_seconds": _numeric_summary(
            record.provider_elapsed_seconds for record in records
        ),
        "elapsed_seconds": _numeric_summary(record.elapsed_seconds for record in records),
    }


def _mcnemar(a_values: Sequence[bool], b_values: Sequence[bool]) -> dict[str, Any]:
    both = sum(a and b for a, b in zip(a_values, b_values))
    a_only = sum(a and not b for a, b in zip(a_values, b_values))
    b_only = sum(not a and b for a, b in zip(a_values, b_values))
    neither = len(a_values) - both - a_only - b_only
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(a_only, b_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "both": both,
        "a_only": a_only,
        "b_only": b_only,
        "neither": neither,
        "discordant": discordant,
        "net_b_minus_a": b_only - a_only,
        "exact_two_sided_p": p_value,
    }


def _paired_numeric(
    a_records: Sequence[EpisodeMetrics],
    b_records: Sequence[EpisodeMetrics],
    getter,
) -> dict[str, Any]:
    deltas: list[float] = []
    for a_record, b_record in zip(a_records, b_records):
        a_value, b_value = getter(a_record), getter(b_record)
        if a_value is not None and b_value is not None:
            deltas.append(float(b_value) - float(a_value))
    return _numeric_summary(deltas)


def analyze_arms(arm_a: ArmData, arm_b: ArmData) -> dict[str, Any]:
    if arm_a.carrier == arm_b.carrier:
        raise CarrierAnalysisError("A and B must use distinct carrier identities")
    if _canonical(arm_a.manifest_signature) != _canonical(arm_b.manifest_signature):
        raise CarrierAnalysisError("A/B common manifests, tools, or budgets differ")
    a_keys = [(record.task_position, record.task_id) for record in arm_a.records]
    b_keys = [(record.task_position, record.task_id) for record in arm_b.records]
    if not a_keys or a_keys != b_keys:
        raise CarrierAnalysisError("A/B cohorts are incomplete or task order differs")

    a_summary, b_summary = _arm_summary(arm_a), _arm_summary(arm_b)
    pair_rows: list[dict[str, Any]] = []
    for a_record, b_record in zip(arm_a.records, arm_b.records):
        pair_rows.append(
            {
                "task_position": a_record.task_position,
                "task_id": a_record.task_id,
                "correct": {"A": a_record.correct, "B": b_record.correct},
                "legal": {"A": a_record.legal, "B": b_record.legal},
                "carrier_clean": {
                    "A": a_record.carrier_clean,
                    "B": b_record.carrier_clean,
                },
                "multi_action_turns": {
                    "A": a_record.multi_action_turns,
                    "B": b_record.multi_action_turns,
                },
                "protocol_errors": {
                    "A": a_record.protocol_errors,
                    "B": b_record.protocol_errors,
                },
                "semantic_protocol_errors": {
                    "A": a_record.semantic_protocol_errors,
                    "B": b_record.semantic_protocol_errors,
                },
                "argument_validation_errors": {
                    "A": a_record.argument_validation_errors,
                    "B": b_record.argument_validation_errors,
                },
                "state_validation_errors": {
                    "A": a_record.state_validation_errors,
                    "B": b_record.state_validation_errors,
                },
                "execution_errors": {
                    "A": a_record.execution_errors,
                    "B": b_record.execution_errors,
                },
                "recovered_after_carrier_rejection": {
                    "A": a_record.recovered_after_carrier_rejection,
                    "B": b_record.recovered_after_carrier_rejection,
                },
                "model_turns": {"A": a_record.model_turns, "B": b_record.model_turns},
                "provider_attempts": {
                    "A": a_record.provider_attempts,
                    "B": b_record.provider_attempts,
                },
                "primitive_calls": {
                    "A": a_record.primitive_calls,
                    "B": b_record.primitive_calls,
                },
                "tokens": {
                    field: {"A": a_record.usage[field], "B": b_record.usage[field]}
                    for field in TOKEN_FIELDS
                },
                "elapsed_seconds": {
                    "A": a_record.elapsed_seconds,
                    "B": b_record.elapsed_seconds,
                },
                "provider_elapsed_seconds": {
                    "A": a_record.provider_elapsed_seconds,
                    "B": b_record.provider_elapsed_seconds,
                },
                "provider_attempt_categories": {
                    category: {
                        "A": a_record.provider_attempt_categories.get(category, 0),
                        "B": b_record.provider_attempt_categories.get(category, 0),
                    }
                    for category in PROVIDER_ATTEMPT_CATEGORIES
                },
            }
        )

    paired = {
        "correct": _mcnemar(
            [record.correct for record in arm_a.records],
            [record.correct for record in arm_b.records],
        ),
        "legal": _mcnemar(
            [record.legal for record in arm_a.records],
            [record.legal for record in arm_b.records],
        ),
        "carrier_clean": _mcnemar(
            [record.carrier_clean for record in arm_a.records],
            [record.carrier_clean for record in arm_b.records],
        ),
        "b_minus_a": {
            "model_turns": _paired_numeric(
                arm_a.records, arm_b.records, lambda record: record.model_turns
            ),
            "primitive_calls": _paired_numeric(
                arm_a.records, arm_b.records, lambda record: record.primitive_calls
            ),
            "elapsed_seconds": _paired_numeric(
                arm_a.records, arm_b.records, lambda record: record.elapsed_seconds
            ),
            "provider_elapsed_seconds": _paired_numeric(
                arm_a.records,
                arm_b.records,
                lambda record: record.provider_elapsed_seconds,
            ),
            "provider_attempt_categories": {
                category: _paired_numeric(
                    arm_a.records,
                    arm_b.records,
                    lambda record, name=category: record.provider_attempt_categories.get(
                        name, 0
                    ),
                )
                for category in PROVIDER_ATTEMPT_CATEGORIES
            },
            "tokens": {
                field: _paired_numeric(
                    arm_a.records,
                    arm_b.records,
                    lambda record, token_field=field: record.usage[token_field],
                )
                for field in TOKEN_FIELDS
            },
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "cohort": {
            "size": len(a_keys),
            "task_positions": [position for position, _ in a_keys],
            "task_ids": [task_id for _, task_id in a_keys],
        },
        "input_hashes": {"A": list(arm_a.input_hashes), "B": list(arm_b.input_hashes)},
        "execution_order_evidence": {
            "schedule_verified": False,
            "counterbalanced": None,
            "reason": "result artifacts contain no scheduler evidence",
            "batches_in_cli_order": {
                "A": list(arm_a.batches),
                "B": list(arm_b.batches),
            },
        },
        "arms": {"A": a_summary, "B": b_summary},
        "paired": paired,
        "pairs": pair_rows,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    arms = report["arms"]
    paired = report["paired"]
    rows = [
        "# Checkpoint carrier A/B aggregate",
        "",
        f"Cohort size: {report['cohort']['size']}",
        "",
        "| Metric | A | B | Paired B-A / p |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("correct", "legal"):
        rows.append(
            f"| {metric} | {arms['A'][metric]} | {arms['B'][metric]} | "
            f"{paired[metric]['net_b_minus_a']} / {paired[metric]['exact_two_sided_p']:.6g} |"
        )
    for metric in (
        "exact_single_action_turns",
        "multi_action_turns",
        "carrier_rejections",
        "protocol_errors",
        "semantic_protocol_errors",
        "argument_validation_errors",
        "state_validation_errors",
        "execution_errors",
        "recovered_after_carrier_rejection",
        "restore_count",
        "model_turns",
        "provider_attempts",
        "primitive_calls",
    ):
        rows.append(f"| {metric} | {arms['A'][metric]} | {arms['B'][metric]} |  |")
    for category in PROVIDER_ATTEMPT_CATEGORIES:
        rows.append(
            f"| provider_attempt:{category} | "
            f"{arms['A']['provider_attempt_categories'][category]} | "
            f"{arms['B']['provider_attempt_categories'][category]} | "
            f"{_fmt(paired['b_minus_a']['provider_attempt_categories'][category]['mean'])} |"
        )
    rows.extend(
        [
            "",
            "| Cost | A total | B total | Mean paired B-A |",
            "|---|---:|---:|---:|",
        ]
    )
    for field in TOKEN_FIELDS:
        rows.append(
            f"| {field} | {_fmt(arms['A']['tokens'][field]['total'])} | "
            f"{_fmt(arms['B']['tokens'][field]['total'])} | "
            f"{_fmt(paired['b_minus_a']['tokens'][field]['mean'])} |"
        )
    rows.append(
        f"| elapsed_seconds | {_fmt(arms['A']['elapsed_seconds']['total'])} | "
        f"{_fmt(arms['B']['elapsed_seconds']['total'])} | "
        f"{_fmt(paired['b_minus_a']['elapsed_seconds']['mean'])} |"
    )
    rows.append(
        f"| provider_elapsed_seconds | "
        f"{_fmt(arms['A']['provider_elapsed_seconds']['total'])} | "
        f"{_fmt(arms['B']['provider_elapsed_seconds']['total'])} | "
        f"{_fmt(paired['b_minus_a']['provider_elapsed_seconds']['mean'])} |"
    )
    rows.append("")
    return "\n".join(rows)


def analyze_result_dirs(
    arm_a_dirs: Sequence[Path],
    arm_b_dirs: Sequence[Path],
    *,
    expected_mode: str = "hybrid",
    expected_model: str = "deepseek-v4-flash",
) -> dict[str, Any]:
    arm_a = load_arm(
        "A", arm_a_dirs, expected_mode=expected_mode, expected_model=expected_model
    )
    arm_b = load_arm(
        "B", arm_b_dirs, expected_mode=expected_mode, expected_model=expected_model
    )
    return analyze_arms(arm_a, arm_b)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", type=Path, nargs="+", required=True)
    parser.add_argument("--arm-b", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--expected-mode", default="hybrid")
    parser.add_argument("--expected-model", default="deepseek-v4-flash")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze_result_dirs(
        args.arm_a,
        args.arm_b,
        expected_mode=args.expected_mode,
        expected_model=args.expected_model,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
