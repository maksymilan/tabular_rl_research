#!/usr/bin/env python3
"""Structural, provider-history, and no-leak gates for checkpoint-relalg artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote

from .checkpoint_store import CHECKPOINT_COMMIT_ELIGIBILITY_NONE
from .protocol import (
    ADMISSION_STATUS,
    BACKEND,
    CARRIERS,
    CARRIER_ABLATION_PROTOCOL_VERSION,
    CARRIER_NATIVE_TOOL_CALLS,
    CARRIER_POLICY_VERSION,
    DEFAULT_CHECKPOINT_GUIDANCE_PROFILE,
    DEFAULT_ATOMIC_OPERATOR_PROFILE,
    DEFAULT_CARRIER,
    CHECKPOINT_POLICY_VERSION,
    DIALECT,
    ENVIRONMENT_RENDERER_VERSION,
    EXECUTOR_VERSION,
    PROTOCOL_VERSION,
    SCHEME,
    capability_manifest,
    carrier_experiment_arm,
    checkpoint_commit_eligibility_for_guidance_profile,
    get_system_prompt,
    prompt_hash,
    normalize_carrier,
    normalize_atomic_operator_profile,
    normalize_checkpoint_guidance_profile,
    trim_provider_phase_history,
    tool_schema_hash,
)
from .provider_tools import (
    NativeToolCallError,
    attempted_action_from_native_message,
    native_authored_action_count,
    native_carrier_envelope_valid,
    validate_native_assistant_message,
)
from .provider import (
    ProviderEmptyText,
    ProviderShapeError,
    _normalize_assistant_message,
    _validated_provider_usage,
    provider_error_counts_as_batch_failure,
    provider_response_envelope_version,
    tool_result_message,
)
from .text_json_carrier import (
    TextJSONActionError,
    attempted_action_from_text_json_message,
    decode_text_json_result_message,
    text_json_authored_action_count,
    text_json_carrier_envelope_valid,
    text_json_exact_single_action,
    validate_text_json_assistant_message,
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

PREFIX_BATCH_RUNNER_VERSION = (
    "checkpoint-relalg-causal-official-deepseek-carrier-ab-v2"
)
LEGACY_BATCH_CONTROL_VERSION = "checkpoint-relalg-batch-control-v1"
BATCH_CONTROL_VERSION = "checkpoint-relalg-batch-control-v2-task-local-output-failures"
SELECTION_IDENTITY_VERSION = "checkpoint-relalg-selection-identity-v1"
PUBLIC_IDENTITY_FIELDS = [
    "position",
    "example_id",
    "example_index",
    "db_id",
    "question_sha256",
    "external_knowledge_sha256",
]
INFRASTRUCTURE_FAILURE_TYPES = frozenset(
    {
        "provider_error",
        "context_length_exceeded",
        "batch_limit_reached",
        "hidden_verifier_error",
    }
)


def _is_strict_batch_manifest(manifest: Mapping[str, Any]) -> bool:
    return manifest.get("runner") == PREFIX_BATCH_RUNNER_VERSION or any(
        field in manifest
        for field in (
            "dataset_manifest",
            "dataset_manifest_sha256",
            "dataset_source_identity",
            "selection_identity",
            "batch_control_version",
            "batch_limits",
        )
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


def provider_history_issues(
    messages: Any,
    *,
    root: str,
    carrier: str = DEFAULT_CARRIER,
) -> list[str]:
    """Check native assistant-call/tool-result order in one actual request prefix."""

    if not isinstance(messages, list):
        return [f"{root} is not a message list"]
    try:
        active_carrier = normalize_carrier(carrier)
    except ValueError:
        return [f"{root} has invalid carrier {carrier!r}"]
    if active_carrier != CARRIER_NATIVE_TOOL_CALLS:
        issues: list[str] = []
        awaiting_feedback = False
        for index, message in enumerate(messages):
            path = f"{root}[{index}]"
            if not isinstance(message, Mapping):
                issues.append(f"{path} is not an object")
                continue
            role = message.get("role")
            if role == "assistant":
                if awaiting_feedback:
                    issues.append(f"{path} starts before text-json Harness feedback")
                awaiting_feedback = True
            elif role == "tool":
                issues.append(f"{path} uses role=tool in the text-json carrier")
            elif role == "user":
                valid_feedback = False
                try:
                    decode_text_json_result_message(message)
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
                else:
                    valid_feedback = True
                if awaiting_feedback:
                    if not valid_feedback:
                        issues.append(f"{path} does not contain text-json Harness feedback")
                    awaiting_feedback = False
                elif valid_feedback:
                    issues.append(f"{path} contains unexpected text-json Harness feedback")
            elif awaiting_feedback:
                issues.append(f"{path} interrupts text-json Harness feedback")
                awaiting_feedback = False
        if awaiting_feedback:
            issues.append(f"{root} ends before text-json Harness feedback")
        return issues
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
            if set(message) != {"role", "tool_call_id", "content"}:
                issues.append(f"{path} native tool result fields differ from policy")
            content = message.get("content")
            if not isinstance(content, str):
                issues.append(f"{path}.content is not encoded JSON")
            else:
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    issues.append(f"{path}.content is not valid JSON")
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
    carrier = record.get("carrier", DEFAULT_CARRIER)
    try:
        atomic_operator_profile = normalize_atomic_operator_profile(
            record.get("atomic_operator_profile", DEFAULT_ATOMIC_OPERATOR_PROFILE)
        )
    except ValueError:
        atomic_operator_profile = DEFAULT_ATOMIC_OPERATOR_PROFILE
        issues.append(f"{root}.atomic_operator_profile is invalid")
    try:
        checkpoint_guidance_profile = normalize_checkpoint_guidance_profile(
            record.get(
                "checkpoint_guidance_profile",
                DEFAULT_CHECKPOINT_GUIDANCE_PROFILE,
            )
        )
    except ValueError:
        checkpoint_guidance_profile = DEFAULT_CHECKPOINT_GUIDANCE_PROFILE
        issues.append(f"{root}.checkpoint_guidance_profile is invalid")
    expected_commit_eligibility = checkpoint_commit_eligibility_for_guidance_profile(
        checkpoint_guidance_profile
    )
    actual_commit_eligibility = record.get(
        "checkpoint_commit_eligibility_policy",
        CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
    )
    if actual_commit_eligibility != expected_commit_eligibility:
        issues.append(
            f"{root}.checkpoint_commit_eligibility_policy does not match guidance profile"
        )
    runtime_payload = record.get("runtime_config")
    runtime_commit_eligibility = (
        runtime_payload.get(
            "checkpoint_commit_eligibility_policy",
            CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
        )
        if isinstance(runtime_payload, Mapping)
        else CHECKPOINT_COMMIT_ELIGIBILITY_NONE
    )
    if runtime_commit_eligibility != expected_commit_eligibility:
        issues.append(
            f"{root}.runtime_config checkpoint commit eligibility does not match guidance profile"
        )
    if mode in {"direct", "atomic", "hybrid"} and carrier in CARRIERS:
        if _canonical(record.get("capability_manifest")) != _canonical(
            capability_manifest(
                mode,
                carrier,
                atomic_operator_profile,
                expected_commit_eligibility,
            )
        ):
            issues.append(f"{root}.capability_manifest does not match the current mode")
        if "carrier" in record:
            from tool_modules.registry import build_checkpoint_relalg_tool_scheme

            expected_response_envelope_version = (
                provider_response_envelope_version(carrier)
            )
            if (
                record.get("provider_response_envelope_version")
                != expected_response_envelope_version
            ):
                issues.append(
                    f"{root}.provider_response_envelope_version does not match "
                    "the current carrier response envelope"
                )
            expected_scheme = build_checkpoint_relalg_tool_scheme(
                mode=mode,
                carrier=carrier,
                atomic_operator_profile=atomic_operator_profile,
                checkpoint_commit_eligibility_policy=expected_commit_eligibility,
            ).manifest_fields()
            for field, expected in expected_scheme.items():
                if field in {
                    "provider_response_envelope_version",
                    "prompt_hash",
                    "teacher_prompt_sha256",
                }:
                    continue
                if _canonical(record.get(field)) != _canonical(expected):
                    issues.append(f"{root}.{field} does not match the current tool scheme")
            if record.get("tool_schema_hash") != tool_schema_hash(
                mode, atomic_operator_profile
            ):
                issues.append(f"{root}.tool_schema_hash does not match current protocol")
            if record.get("prompt_hash") != prompt_hash(
                mode,
                teacher=True,
                carrier=carrier,
                checkpoint_guidance_profile=checkpoint_guidance_profile,
                atomic_operator_profile=atomic_operator_profile,
            ):
                issues.append(f"{root}.prompt_hash does not match current protocol")
    return issues


def audit_record(record: Mapping[str, Any], *, record_index: int = 0) -> dict[str, Any]:
    root = f"record[{record_index}]"
    issues: list[str] = []
    if record.get("tool_scheme") != SCHEME:
        issues.append(f"{root}.tool_scheme is not {SCHEME!r}")
    if record.get("protocol_version") != PROTOCOL_VERSION:
        issues.append(f"{root}.protocol_version is not {PROTOCOL_VERSION!r}")
    mode = record.get("mode")
    carrier_declared = "carrier" in record
    raw_carrier = record.get("carrier", DEFAULT_CARRIER)
    try:
        carrier = normalize_carrier(raw_carrier)
    except ValueError:
        issues.append(f"{root}.carrier is invalid: {raw_carrier!r}")
        carrier = DEFAULT_CARRIER
    try:
        checkpoint_guidance_profile = normalize_checkpoint_guidance_profile(
            record.get(
                "checkpoint_guidance_profile",
                DEFAULT_CHECKPOINT_GUIDANCE_PROFILE,
            )
        )
    except ValueError:
        issues.append(f"{root}.checkpoint_guidance_profile is invalid")
        checkpoint_guidance_profile = DEFAULT_CHECKPOINT_GUIDANCE_PROFILE
    try:
        atomic_operator_profile = normalize_atomic_operator_profile(
            record.get("atomic_operator_profile", DEFAULT_ATOMIC_OPERATOR_PROFILE)
        )
    except ValueError:
        issues.append(f"{root}.atomic_operator_profile is invalid")
        atomic_operator_profile = DEFAULT_ATOMIC_OPERATOR_PROFILE
    if carrier_declared:
        if record.get("carrier_ablation_protocol_version") != CARRIER_ABLATION_PROTOCOL_VERSION:
            issues.append(f"{root}.carrier_ablation_protocol_version is invalid")
        if record.get("carrier_policy_version") != CARRIER_POLICY_VERSION:
            issues.append(f"{root}.carrier_policy_version is invalid")
        if record.get("experiment_arm") != carrier_experiment_arm(carrier):
            issues.append(f"{root}.experiment_arm does not match carrier")
        within_batch_order = record.get("within_batch_order")
        if within_batch_order not in {None, "A_then_B", "B_then_A"}:
            issues.append(f"{root}.within_batch_order is invalid")
    if mode not in {"direct", "atomic", "hybrid"}:
        issues.append(f"{root}.mode is invalid: {mode!r}")
    else:
        if record.get("tool_schema_sha256") != tool_schema_hash(
            mode, atomic_operator_profile
        ):
            issues.append(f"{root}.tool_schema_sha256 does not match current protocol")
        if record.get("teacher_prompt_sha256") != prompt_hash(
            mode,
            teacher=True,
            carrier=carrier,
            checkpoint_guidance_profile=checkpoint_guidance_profile,
            atomic_operator_profile=atomic_operator_profile,
        ):
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
        if carrier_declared and request_options.get("carrier") != carrier:
            issues.append(f"{root}.provider_request_options.carrier differs from record")
        if carrier == CARRIER_NATIVE_TOOL_CALLS:
            if request_options.get("tool_choice") != "auto":
                issues.append(f"{root}.provider_request_options.tool_choice is not auto")
            if "response_format" in request_options:
                issues.append(f"{root}.native request options contain response_format")
        else:
            if "tool_choice" in request_options or "tools" in request_options:
                issues.append(f"{root}.text-json request options contain native tool controls")
            if request_options.get("response_format") != {"type": "json_object"}:
                issues.append(f"{root}.text-json response_format is not JSON Output")
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
    audited_provider_usage: Counter[str] = Counter()
    terminal_result_index: int | None = None
    terminal_result: Mapping[str, Any] | None = None
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
        if terminal_result_index is not None:
            issues.append(
                f"{path} appears after terminal result in turn {terminal_result_index}"
            )
        if not isinstance(turn, Mapping):
            issues.append(f"{path} is not an object")
            continue
        model_input = turn.get("model_input")
        issues.extend(no_leak_issues(model_input, root=f"{path}.model_input"))
        issues.extend(
            provider_history_issues(
                model_input,
                root=f"{path}.model_input",
                carrier=carrier,
            )
        )
        if carrier_declared and turn.get("carrier") != carrier:
            issues.append(f"{path}.carrier differs from record")
        if isinstance(model_input, list) and mode in {"direct", "atomic", "hybrid"}:
            if len(model_input) < 2:
                issues.append(f"{path}.model_input must contain system and current user context")
            else:
                system = model_input[0]
                if not isinstance(system, Mapping) or system.get("role") != "system":
                    issues.append(f"{path}.model_input does not start with system")
                elif system.get("content") != get_system_prompt(
                    mode,
                    teacher=True,
                    carrier=carrier,
                    checkpoint_guidance_profile=checkpoint_guidance_profile,
                    atomic_operator_profile=atomic_operator_profile,
                ):
                    issues.append(f"{path}.model_input system prompt differs from current teacher prompt")
                if (
                    not isinstance(model_input[-1], Mapping)
                    or model_input[-1].get("role") != "user"
                ):
                    issues.append(f"{path}.model_input does not end with current user context")
                if _canonical(model_input[1:-1]) != _canonical(expected_phase_history):
                    issues.append(f"{path}.model_input phase history differs from causal transcript")
        assistant = turn.get("assistant_message")
        if carrier_declared and assistant is None:
            provider_error = turn.get("provider_error")
            if not isinstance(provider_error, Mapping):
                issues.append(f"{path} has no assistant or provider_error")
            else:
                if (
                    record.get("runner") == PREFIX_BATCH_RUNNER_VERSION
                    or "batch_control_version" in record
                ):
                    expected_provider_error_fields = {
                        "type",
                        "message",
                        "finish_reason",
                        "response_envelope_sha256",
                        "response_model",
                        "http_status",
                        "retryable",
                    }
                    if set(provider_error) != expected_provider_error_fields:
                        issues.append(
                            f"{path}.provider_error fields differ from v2 policy"
                        )
                    http_status = provider_error.get("http_status")
                    retryable = provider_error.get("retryable")
                    if http_status is not None and (
                        isinstance(http_status, bool)
                        or not isinstance(http_status, int)
                        or not 400 <= http_status <= 599
                    ):
                        issues.append(f"{path}.provider_error.http_status is invalid")
                    if not isinstance(retryable, bool):
                        issues.append(f"{path}.provider_error.retryable is invalid")
                    if isinstance(http_status, int) and not isinstance(
                        http_status, bool
                    ):
                        expected_retryable = http_status in {
                            408,
                            409,
                            425,
                            429,
                        } or 500 <= http_status <= 599
                        if retryable is not expected_retryable:
                            issues.append(
                                f"{path}.provider_error.retryable differs from HTTP policy"
                            )
                if turn_index != len(turns) - 1:
                    issues.append(f"{path} provider_error turn is not the final turn")
                expected_failure_type = (
                    "context_length_exceeded"
                    if provider_error.get("type") == "context_length_exceeded"
                    else "provider_error"
                )
                if record.get("failure_type") != expected_failure_type:
                    issues.append(
                        f"{path}.provider_error does not match record.failure_type"
                    )
        carrier_metrics = turn.get("carrier_metrics")
        if carrier_declared and not isinstance(carrier_metrics, Mapping):
            issues.append(f"{path}.carrier_metrics is missing")
            carrier_metrics = {}
        elif not isinstance(carrier_metrics, Mapping):
            carrier_metrics = {}
        metric_keys = {
            "carrier",
            "provider_response_present",
            "authored_action_count",
            "exact_single_action",
            "carrier_envelope_valid",
            "carrier_error_code",
            "action_validation_error_code",
            "provider_attempt_count",
            "provider_elapsed_seconds",
        }
        if carrier_declared and set(carrier_metrics) != metric_keys:
            issues.append(f"{path}.carrier_metrics fields differ from policy")
        if carrier_declared and carrier_metrics.get("carrier") != carrier:
            issues.append(f"{path}.carrier_metrics.carrier differs from record")
        turn_usage = turn.get("provider_usage")
        if carrier_declared and not isinstance(turn_usage, Mapping):
            issues.append(f"{path}.provider_usage is missing")
            turn_usage = {}
        elif not isinstance(turn_usage, Mapping):
            turn_usage = {}
        for usage_key, usage_value in turn_usage.items():
            if (
                not isinstance(usage_key, str)
                or isinstance(usage_value, bool)
                or not isinstance(usage_value, int)
                or usage_value < 0
            ):
                issues.append(f"{path}.provider_usage contains an invalid counter")
                continue
            audited_provider_usage[usage_key] += usage_value
        if "provider_failed_usage" in turn and _canonical(
            turn.get("provider_failed_usage")
        ) != _canonical(turn_usage):
            issues.append(f"{path}.provider_failed_usage differs from provider_usage")
        attempt_events = turn.get("provider_attempt_events")
        if carrier_declared and not isinstance(attempt_events, list):
            issues.append(f"{path}.provider_attempt_events is missing")
            attempt_events = []
        elif not isinstance(attempt_events, list):
            attempt_events = []
        attempt_usage: Counter[str] = Counter()
        event_keys = {
            "attempt_index",
            "max_tokens",
            "finish_reason",
            "usage",
            "shape_category",
            "response_envelope_sha256",
            "response_model",
            "raw_assistant_message",
            "elapsed_seconds_from_request_start",
        }
        previous_attempt_elapsed = -1.0
        for event_index, event in enumerate(attempt_events, start=1):
            event_path = f"{path}.provider_attempt_events[{event_index - 1}]"
            if not isinstance(event, Mapping) or set(event) != event_keys:
                issues.append(f"{event_path} fields differ from policy")
                continue
            if event.get("attempt_index") != event_index:
                issues.append(f"{event_path}.attempt_index is not contiguous")
            if (
                isinstance(event.get("max_tokens"), bool)
                or not isinstance(event.get("max_tokens"), int)
                or event["max_tokens"] < 1
            ):
                issues.append(f"{event_path}.max_tokens is invalid")
            if event.get("finish_reason") is not None and not isinstance(
                event.get("finish_reason"), str
            ):
                issues.append(f"{event_path}.finish_reason is invalid")
            if event.get("shape_category") not in {
                "accepted_response",
                "completion_length",
                "missing_tool_calls",
                "provider_shape",
                "context_overflow",
                "transport",
                "model_mismatch",
                "empty_text",
                "insufficient_system_resource",
                "content_filter",
            }:
                issues.append(f"{event_path}.shape_category is invalid")
            envelope_hash = event.get("response_envelope_sha256")
            if envelope_hash is not None and (
                not isinstance(envelope_hash, str)
                or len(envelope_hash) != 64
                or any(character not in "0123456789abcdef" for character in envelope_hash)
            ):
                issues.append(f"{event_path}.response_envelope_sha256 is invalid")
            raw_assistant_message = event.get("raw_assistant_message")
            if raw_assistant_message is not None and not isinstance(
                raw_assistant_message, Mapping
            ):
                issues.append(f"{event_path}.raw_assistant_message is invalid")
            else:
                if envelope_hash != _response_envelope_sha256(raw_assistant_message):
                    issues.append(
                        f"{event_path}.response_envelope_sha256 differs from raw assistant"
                    )
                if raw_assistant_message is not None:
                    issues.extend(
                        no_leak_issues(
                            raw_assistant_message,
                            root=f"{event_path}.raw_assistant_message",
                        )
                    )
            response_model = event.get("response_model")
            if response_model is not None and (
                not isinstance(response_model, str) or not response_model
            ):
                issues.append(f"{event_path}.response_model is invalid")
            requested_model = (
                request_options.get("model")
                if isinstance(request_options, Mapping)
                else None
            )
            shape_category = event.get("shape_category")
            if (
                isinstance(response_model, str)
                and isinstance(requested_model, str)
                and response_model != requested_model
            ):
                if shape_category != "model_mismatch":
                    issues.append(
                        f"{event_path} mismatched response_model is not model_mismatch"
                    )
            elif shape_category == "model_mismatch":
                issues.append(
                    f"{event_path} model_mismatch lacks a distinct response_model"
                )
            event_finish_reason = event.get("finish_reason")
            category_finish_reason = {
                "completion_length": "length",
                "insufficient_system_resource": "insufficient_system_resource",
                "content_filter": "content_filter",
            }.get(shape_category)
            if (
                category_finish_reason is not None
                and event_finish_reason != category_finish_reason
            ):
                issues.append(
                    f"{event_path}.finish_reason differs from shape_category"
                )
            if shape_category == "accepted_response" and event_finish_reason != (
                "tool_calls" if carrier == CARRIER_NATIVE_TOOL_CALLS else "stop"
            ):
                issues.append(
                    f"{event_path}.finish_reason cannot be accepted by carrier"
                )
            if shape_category == "missing_tool_calls" and (
                carrier != CARRIER_NATIVE_TOOL_CALLS
                or event_finish_reason != "tool_calls"
            ):
                issues.append(
                    f"{event_path}.missing_tool_calls is invalid for carrier/finish_reason"
                )
            if shape_category == "empty_text" and (
                carrier == CARRIER_NATIVE_TOOL_CALLS or event_finish_reason != "stop"
            ):
                issues.append(
                    f"{event_path}.empty_text is invalid for carrier/finish_reason"
                )
            attempt_elapsed = event.get("elapsed_seconds_from_request_start")
            if (
                isinstance(attempt_elapsed, bool)
                or not isinstance(attempt_elapsed, (int, float))
                or not math.isfinite(float(attempt_elapsed))
                or attempt_elapsed < previous_attempt_elapsed
            ):
                issues.append(f"{event_path}.elapsed offset is invalid or non-monotonic")
            else:
                previous_attempt_elapsed = float(attempt_elapsed)
            event_usage = event.get("usage")
            if not isinstance(event_usage, Mapping):
                issues.append(f"{event_path}.usage is invalid")
                continue
            for usage_key, usage_value in event_usage.items():
                if (
                    not isinstance(usage_key, str)
                    or isinstance(usage_value, bool)
                    or not isinstance(usage_value, int)
                    or usage_value < 0
                ):
                    issues.append(f"{event_path}.usage contains an invalid counter")
                    continue
                attempt_usage[usage_key] += usage_value
            if event.get("shape_category") == "accepted_response":
                for usage_key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ):
                    usage_value = event_usage.get(usage_key)
                    if (
                        isinstance(usage_value, bool)
                        or not isinstance(usage_value, int)
                        or usage_value < 0
                    ):
                        issues.append(
                            f"{event_path}.usage.{usage_key} is required for an accepted response"
                        )
                prompt_tokens = event_usage.get("prompt_tokens")
                completion_tokens = event_usage.get("completion_tokens")
                total_tokens = event_usage.get("total_tokens")
                if (
                    isinstance(prompt_tokens, int)
                    and not isinstance(prompt_tokens, bool)
                    and isinstance(completion_tokens, int)
                    and not isinstance(completion_tokens, bool)
                    and isinstance(total_tokens, int)
                    and not isinstance(total_tokens, bool)
                    and total_tokens != prompt_tokens + completion_tokens
                ):
                    issues.append(
                        f"{event_path}.usage.total_tokens differs from prompt + completion"
                    )
                cache_hit = event_usage.get("prompt_cache_hit_tokens")
                cache_miss = event_usage.get("prompt_cache_miss_tokens")
                if (
                    isinstance(cache_hit, int)
                    and not isinstance(cache_hit, bool)
                    and isinstance(cache_miss, int)
                    and not isinstance(cache_miss, bool)
                    and isinstance(prompt_tokens, int)
                    and not isinstance(prompt_tokens, bool)
                    and cache_hit + cache_miss != prompt_tokens
                ):
                    issues.append(
                        f"{event_path}.usage prompt cache counters differ from prompt_tokens"
                    )
                reasoning_tokens = event_usage.get(
                    "completion_tokens_details.reasoning_tokens"
                )
                if (
                    reasoning_tokens is not None
                    and (
                        isinstance(reasoning_tokens, bool)
                        or not isinstance(reasoning_tokens, int)
                        or reasoning_tokens < 0
                        or (
                            isinstance(completion_tokens, int)
                            and not isinstance(completion_tokens, bool)
                            and reasoning_tokens > completion_tokens
                        )
                    )
                ):
                    issues.append(
                        f"{event_path}.usage reasoning_tokens is invalid"
                    )
            recomputed_category: str | None = None
            if (
                isinstance(response_model, str)
                and isinstance(requested_model, str)
                and response_model != requested_model
            ):
                recomputed_category = "model_mismatch"
            elif shape_category in {"transport", "context_overflow"}:
                if raw_assistant_message is not None:
                    issues.append(
                        f"{event_path}.{shape_category} unexpectedly stores an assistant envelope"
                    )
                if response_model is not None:
                    issues.append(
                        f"{event_path}.{shape_category} unexpectedly stores response_model"
                    )
            elif not isinstance(response_model, str) or not response_model:
                recomputed_category = "provider_shape"
            else:
                try:
                    _validated_provider_usage(dict(event_usage))
                except ProviderShapeError:
                    recomputed_category = "provider_shape"
                else:
                    if event_finish_reason == "length":
                        recomputed_category = "completion_length"
                    elif event_finish_reason == "insufficient_system_resource":
                        recomputed_category = "insufficient_system_resource"
                    elif event_finish_reason == "content_filter":
                        recomputed_category = "content_filter"
                    elif raw_assistant_message is None:
                        recomputed_category = "provider_shape"
                    else:
                        expected_finish_reason = (
                            "tool_calls"
                            if carrier == CARRIER_NATIVE_TOOL_CALLS
                            else "stop"
                        )
                        if event_finish_reason != expected_finish_reason:
                            recomputed_category = "provider_shape"
                        else:
                            try:
                                _, normalized_calls = _normalize_assistant_message(
                                    dict(raw_assistant_message),
                                    carrier=carrier,
                                )
                            except ProviderEmptyText:
                                recomputed_category = "empty_text"
                            except ProviderShapeError:
                                recomputed_category = "provider_shape"
                            else:
                                recomputed_category = (
                                    "missing_tool_calls"
                                    if carrier == CARRIER_NATIVE_TOOL_CALLS
                                    and not normalized_calls
                                    else "accepted_response"
                                )
            if (
                recomputed_category is not None
                and shape_category != recomputed_category
            ):
                issues.append(
                    f"{event_path}.shape_category differs from raw provider response"
                )
        if carrier_declared and len(attempt_events) != carrier_metrics.get(
            "provider_attempt_count"
        ):
            issues.append(f"{path}.provider attempt event count differs from metrics")
        if (
            isinstance(turn.get("provider_error"), Mapping)
            and turn["provider_error"].get("retryable") is False
            and len(attempt_events) != 1
        ):
            issues.append(
                f"{path} non-retryable provider error did not stop after one attempt"
            )
        if carrier_declared and _canonical(dict(sorted(attempt_usage.items()))) != _canonical(
            turn_usage
        ):
            issues.append(f"{path}.provider attempt usage differs from turn usage")
        retry_events = turn.get("provider_retry_events")
        if carrier_declared and not isinstance(retry_events, list):
            issues.append(f"{path}.provider_retry_events is missing")
            retry_events = []
        elif not isinstance(retry_events, list):
            retry_events = []
        for retry_index, retry_event in enumerate(retry_events, start=1):
            retry_path = f"{path}.provider_retry_events[{retry_index - 1}]"
            if not isinstance(retry_event, Mapping):
                issues.append(f"{retry_path} is not an object")
                continue
            if retry_event.get("attempt") != retry_index:
                issues.append(f"{retry_path}.attempt is not contiguous")
            if retry_event.get("type") not in {
                "completion_length",
                "missing_tool_calls",
                "provider_shape",
                "transport",
                "insufficient_system_resource",
            }:
                issues.append(f"{retry_path}.type is invalid")
            expected_retry_fields = {
                "completion_length": {
                    "type",
                    "attempt",
                    "max_tokens",
                    "next_max_tokens",
                },
                "missing_tool_calls": {"type", "attempt"},
                "provider_shape": {"type", "attempt", "error"},
                "transport": {"type", "attempt", "error"},
                "insufficient_system_resource": {"type", "attempt"},
            }.get(retry_event.get("type"))
            if expected_retry_fields is not None and set(retry_event) != expected_retry_fields:
                issues.append(f"{retry_path} fields differ from retry policy")
            if retry_event.get("type") in {"provider_shape", "transport"} and (
                not isinstance(retry_event.get("error"), str)
                or not retry_event.get("error")
            ):
                issues.append(f"{retry_path}.error is invalid")
            if any(
                key in retry_event
                for key in ("content", "reasoning_content", "message", "tool_calls")
            ):
                issues.append(f"{retry_path} stores provider-authored content")
            if retry_index <= len(attempt_events):
                source_attempt = attempt_events[retry_index - 1]
                source_category = (
                    source_attempt.get("shape_category")
                    if isinstance(source_attempt, Mapping)
                    else None
                )
                expected_categories = {
                    "completion_length": {"completion_length"},
                    "missing_tool_calls": {"missing_tool_calls"},
                    "provider_shape": {"provider_shape", "empty_text"},
                    "transport": {"transport"},
                    "insufficient_system_resource": {
                        "insufficient_system_resource"
                    },
                }.get(retry_event.get("type"), set())
                if source_category not in expected_categories:
                    issues.append(f"{retry_path}.type differs from source attempt category")
                if retry_index < len(attempt_events):
                    next_attempt = attempt_events[retry_index]
                    source_budget = source_attempt.get("max_tokens")
                    next_budget = (
                        next_attempt.get("max_tokens")
                        if isinstance(next_attempt, Mapping)
                        else None
                    )
                    if retry_event.get("type") == "completion_length":
                        if retry_event.get("max_tokens") != source_budget:
                            issues.append(f"{retry_path}.max_tokens differs from source attempt")
                        if retry_event.get("next_max_tokens") != next_budget:
                            issues.append(f"{retry_path}.next_max_tokens differs from next attempt")
                        if (
                            not isinstance(next_budget, int)
                            or not isinstance(source_budget, int)
                            or next_budget <= source_budget
                        ):
                            issues.append(f"{retry_path} did not increase completion budget")
                    elif next_budget != source_budget:
                        issues.append(f"{retry_path} unexpectedly changed completion budget")
        attempt_count_metric = carrier_metrics.get("provider_attempt_count")
        if carrier_declared and attempt_count_metric != len(retry_events) + 1:
            issues.append(f"{path}.provider_attempt_count differs from retry history")
        elapsed_metric = carrier_metrics.get("provider_elapsed_seconds")
        if carrier_declared and (
            isinstance(elapsed_metric, bool)
            or not isinstance(elapsed_metric, (int, float))
            or not math.isfinite(float(elapsed_metric))
            or elapsed_metric < 0
        ):
            issues.append(f"{path}.provider_elapsed_seconds is not finite non-negative")
        if (
            carrier_declared
            and attempt_events
            and isinstance(elapsed_metric, (int, float))
            and not isinstance(elapsed_metric, bool)
        ):
            final_offset = attempt_events[-1].get("elapsed_seconds_from_request_start")
            if (
                isinstance(final_offset, (int, float))
                and not isinstance(final_offset, bool)
                and math.isfinite(float(final_offset))
                and float(elapsed_metric) < float(final_offset)
            ):
                issues.append(
                    f"{path}.provider_elapsed_seconds precedes the final attempt event"
                )
        if carrier_declared and attempt_events:
            last_attempt = attempt_events[-1]
            last_category = (
                last_attempt.get("shape_category")
                if isinstance(last_attempt, Mapping)
                else None
            )
            if assistant is not None:
                if turn.get("provider_error") is not None:
                    issues.append(f"{path} stores both assistant response and provider_error")
                if last_category != "accepted_response":
                    issues.append(f"{path} final attempt category is not accepted_response")
                if last_attempt.get("finish_reason") != turn.get("finish_reason"):
                    issues.append(f"{path} final attempt finish_reason differs from turn")
                if last_attempt.get("response_envelope_sha256") != _response_envelope_sha256(
                    assistant
                ):
                    issues.append(f"{path} final attempt envelope hash differs from assistant")
                if _canonical(last_attempt.get("raw_assistant_message")) != _canonical(
                    assistant
                ):
                    issues.append(f"{path} final attempt raw assistant differs from turn")
                requested_model = (
                    request_options.get("model")
                    if isinstance(request_options, Mapping)
                    else None
                )
                if last_attempt.get("response_model") != requested_model:
                    issues.append(f"{path} accepted attempt model differs from requested model")
                expected_finish_reason = (
                    "tool_calls"
                    if carrier == CARRIER_NATIVE_TOOL_CALLS
                    else "stop"
                )
                if last_attempt.get("finish_reason") != expected_finish_reason:
                    issues.append(
                        f"{path} accepted attempt finish_reason is invalid for carrier"
                    )
            else:
                provider_error = turn.get("provider_error")
                if not isinstance(provider_error, Mapping):
                    pass
                else:
                    legacy_provider_error_fields = {
                        "type",
                        "message",
                        "finish_reason",
                        "response_envelope_sha256",
                        "response_model",
                    }
                    current_provider_error_fields = {
                        *legacy_provider_error_fields,
                        "http_status",
                        "retryable",
                    }
                    provider_error_keys = frozenset(provider_error)
                    if provider_error_keys not in {
                        frozenset(legacy_provider_error_fields),
                        frozenset(current_provider_error_fields),
                    }:
                        issues.append(
                            f"{path}.provider_error fields differ from policy"
                        )
                    if provider_error_keys == frozenset(current_provider_error_fields):
                        http_status = provider_error.get("http_status")
                        if http_status is not None and (
                            isinstance(http_status, bool)
                            or not isinstance(http_status, int)
                            or not 400 <= http_status <= 599
                        ):
                            issues.append(f"{path}.provider_error.http_status is invalid")
                        if not isinstance(provider_error.get("retryable"), bool):
                            issues.append(f"{path}.provider_error.retryable is invalid")
                    if (
                        not isinstance(provider_error.get("message"), str)
                        or not provider_error.get("message")
                    ):
                        issues.append(f"{path}.provider_error.message is invalid")
                    if last_category == "accepted_response":
                        issues.append(f"{path} provider_error follows an accepted final attempt")
                    if last_attempt.get("finish_reason") != provider_error.get(
                        "finish_reason"
                    ):
                        issues.append(f"{path} final attempt finish_reason differs from provider_error")
                    if last_attempt.get("response_envelope_sha256") != provider_error.get(
                        "response_envelope_sha256"
                    ):
                        issues.append(f"{path} final attempt hash differs from provider_error")
                    if last_attempt.get("response_model") != provider_error.get(
                        "response_model"
                    ):
                        issues.append(f"{path} final attempt model differs from provider_error")
                    expected_error_types = {
                        "model_mismatch": {"ProviderModelMismatch"},
                        "context_overflow": {"context_length_exceeded"},
                        "completion_length": {"ProviderCompletionTruncated"},
                        "missing_tool_calls": {"ProviderShapeError"},
                        "empty_text": {"ProviderEmptyText"},
                        "provider_shape": {"ProviderShapeError"},
                        "transport": {"ProviderError"},
                        "insufficient_system_resource": {
                            "ProviderInsufficientSystemResource"
                        },
                        "content_filter": {"ProviderContentFiltered"},
                    }
                    allowed_error_types = expected_error_types.get(last_category)
                    if (
                        allowed_error_types is None
                        or provider_error.get("type") not in allowed_error_types
                    ):
                        issues.append(f"{path}.provider_error type differs from final attempt")
                    provider_error_type = provider_error.get("type")
                    required_categories = {
                        "ProviderModelMismatch": {"model_mismatch"},
                        "context_length_exceeded": {"context_overflow"},
                        "ProviderCompletionTruncated": {"completion_length"},
                        "ProviderEmptyText": {"empty_text"},
                        "ProviderShapeError": {"provider_shape", "missing_tool_calls"},
                        "ProviderError": {"transport"},
                        "ProviderInsufficientSystemResource": {
                            "insufficient_system_resource"
                        },
                        "ProviderContentFiltered": {"content_filter"},
                    }.get(provider_error_type)
                    if required_categories is None or last_category not in required_categories:
                        issues.append(f"{path} final attempt category differs from provider_error")
                    requested_model = (
                        request_options.get("model")
                        if isinstance(request_options, Mapping)
                        else None
                    )
                    response_model = last_attempt.get("response_model")
                    if (
                        isinstance(response_model, str)
                        and isinstance(requested_model, str)
                        and response_model != requested_model
                    ):
                        if (
                            last_category != "model_mismatch"
                            or provider_error_type != "ProviderModelMismatch"
                        ):
                            issues.append(
                                f"{path} mismatched response_model was downgraded"
                            )
                    elif last_category == "model_mismatch":
                        issues.append(
                            f"{path} model_mismatch lacks a distinct response_model"
                        )
                    expected_failure_finish = {
                        "completion_length": "length",
                        "insufficient_system_resource": "insufficient_system_resource",
                        "content_filter": "content_filter",
                    }.get(last_category)
                    if (
                        expected_failure_finish is not None
                        and last_attempt.get("finish_reason") != expected_failure_finish
                    ):
                        issues.append(
                            f"{path} final attempt finish_reason differs from category"
                        )
        if assistant is not None:
            primitive_turns += 1
            if carrier == CARRIER_NATIVE_TOOL_CALLS:
                calls = assistant.get("tool_calls") if isinstance(assistant, Mapping) else None
                if not isinstance(calls, list) or not calls:
                    issues.append(f"{path}.assistant_message has no native calls")
                    calls = []
                result_messages = turn.get("tool_result_messages")
                result_messages_field = "tool_result_messages"
                validator = lambda active_mode, active_message: validate_native_assistant_message(  # noqa: E731
                    active_mode,
                    active_message,
                    atomic_operator_profile=atomic_operator_profile,
                )
                attempted_from_message = attempted_action_from_native_message
                error_type = NativeToolCallError
                rejection_field = "native_rejection"
                rejection_details = lambda exc: {  # noqa: E731
                    "argument_path": exc.path,
                    "call_count": len(calls),
                }
                authored_count = native_authored_action_count(assistant)
                exact_single_action = authored_count == 1
                envelope_valid = native_carrier_envelope_valid(assistant)
            else:
                calls = []
                result_messages = turn.get("text_result_messages")
                result_messages_field = "text_result_messages"
                validator = lambda active_mode, active_message: validate_text_json_assistant_message(  # noqa: E731
                    active_mode,
                    active_message,
                    atomic_operator_profile=atomic_operator_profile,
                )
                attempted_from_message = attempted_action_from_text_json_message
                error_type = TextJSONActionError
                rejection_field = "text_json_rejection"
                rejection_details = lambda exc: {  # noqa: E731
                    "argument_path": exc.path,
                    "carrier": carrier,
                }
                authored_count = text_json_authored_action_count(assistant)
                exact_single_action = text_json_exact_single_action(assistant)
                envelope_valid = text_json_carrier_envelope_valid(assistant)
            if not isinstance(result_messages, list):
                issues.append(f"{path}.{result_messages_field} is not a list")
                result_messages = []
            elif carrier != CARRIER_NATIVE_TOOL_CALLS and len(result_messages) != 1:
                issues.append(f"{path}.text_result_messages must contain exactly one message")
            issues.extend(
                provider_history_issues(
                    [assistant, *result_messages],
                    root=f"{path}.authored_turn",
                    carrier=carrier,
                )
            )
            result = turn.get("result")
            result_code = (
                (result.get("error") or {}).get("code")
                if isinstance(result, Mapping)
                else None
            )
            carrier_error_code: str | None = None
            action_validation_error_code: str | None = None
            try:
                lowered = validator(str(mode), assistant)
            except error_type as exc:
                if envelope_valid:
                    action_validation_error_code = exc.code
                else:
                    carrier_error_code = exc.code
                if turn.get("action") is not None:
                    issues.append(f"{path} executes an action from a rejected carrier message")
                rejection = turn.get(rejection_field)
                if not isinstance(rejection, Mapping):
                    issues.append(f"{path} rejected carrier message has no rejection record")
                else:
                    expected_rejection = {
                        "code": exc.code,
                        "message": exc.message,
                        **rejection_details(exc),
                    }
                    if _canonical(rejection) != _canonical(expected_rejection):
                        issues.append(f"{path}.{rejection_field} differs from carrier validator")
                attempted = attempted_from_message(assistant)
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
                        "details": rejection_details(exc),
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
                    "tool_call_id": lowered.get("tool_call_id"),
                }
                if _canonical(turn.get("action")) != _canonical(expected_action):
                    issues.append(f"{path}.action differs from the authored carrier action")
                if turn.get(rejection_field) is not None:
                    issues.append(f"{path} records a rejection for a valid carrier action")
            if carrier_declared:
                expected_carrier_metrics = {
                    "carrier": carrier,
                    "provider_response_present": True,
                    "authored_action_count": authored_count,
                    "exact_single_action": exact_single_action,
                    "carrier_envelope_valid": envelope_valid,
                    "carrier_error_code": carrier_error_code,
                    "action_validation_error_code": action_validation_error_code,
                    "provider_attempt_count": carrier_metrics.get("provider_attempt_count"),
                    "provider_elapsed_seconds": carrier_metrics.get("provider_elapsed_seconds"),
                }
                if _canonical(carrier_metrics) != _canonical(expected_carrier_metrics):
                    issues.append(f"{path}.carrier_metrics differs from authored response")
                attempt_count = carrier_metrics.get("provider_attempt_count")
                elapsed = carrier_metrics.get("provider_elapsed_seconds")
                if isinstance(attempt_count, bool) or not isinstance(attempt_count, int) or attempt_count < 1:
                    issues.append(f"{path}.carrier_metrics.provider_attempt_count is invalid")
                if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or elapsed < 0:
                    issues.append(f"{path}.carrier_metrics.provider_elapsed_seconds is invalid")
            response_metadata = turn.get("provider_response_metadata")
            if not isinstance(response_metadata, Mapping) or not isinstance(
                response_metadata.get("model"), str
            ):
                issues.append(f"{path}.provider_response_metadata.model is missing")
            else:
                if set(response_metadata) - {
                    "id",
                    "model",
                    "system_fingerprint",
                    "object",
                }:
                    issues.append(
                        f"{path}.provider_response_metadata fields differ from policy"
                    )
                for metadata_key, metadata_value in response_metadata.items():
                    if not isinstance(metadata_value, str) or not metadata_value:
                        issues.append(
                            f"{path}.provider_response_metadata.{metadata_key} is invalid"
                        )
            if isinstance(response_metadata, Mapping) and isinstance(request_options, Mapping):
                response_model = response_metadata.get("model")
                requested_model = request_options.get("model")
                if response_model is not None and requested_model is not None and response_model != requested_model:
                    issues.append(f"{path}.provider response model differs from requested model")
            for result_index, result_message in enumerate(result_messages):
                result_path = f"{path}.{result_messages_field}[{result_index}]"
                try:
                    if carrier == CARRIER_NATIVE_TOOL_CALLS:
                        call_id = None
                        if result_index < len(calls):
                            raw_call = calls[result_index]
                            if isinstance(raw_call, Mapping):
                                call_id = raw_call.get("id")
                        if not isinstance(call_id, str) or not isinstance(result, dict):
                            raise ValueError("native result cannot be deterministically rendered")
                        expected_result_message = tool_result_message(call_id, result)
                        if _canonical(result_message) != _canonical(expected_result_message):
                            issues.append(
                                f"{result_path} differs from deterministic native tool result"
                            )
                        content = (
                            result_message.get("content")
                            if isinstance(result_message, Mapping)
                            else None
                        )
                        if not isinstance(content, str):
                            raise ValueError("content is not encoded JSON")
                        decoded = json.loads(content)
                    else:
                        decoded = decode_text_json_result_message(result_message)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    issues.append(f"{result_path} is invalid: {type(exc).__name__}")
                    continue
                if _canonical(decoded) != _canonical(result):
                    issues.append(f"{result_path}.content differs from turn.result")
            rejection = turn.get(rejection_field)
            if (
                isinstance(rejection, Mapping)
                and rejection.get("code") != result_code
                and not (
                    primitive_turns > max_primitive_calls
                    and result_code == "primitive_call_limit_reached"
                )
            ):
                issues.append(f"{path}.{rejection_field} code does not match result error")
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
                    expected_phase_history = trim_provider_phase_history(
                        expected_phase_history,
                        atomic_operator_profile,
                    )
        elif carrier_declared:
            expected_failed_metrics = {
                "carrier": carrier,
                "provider_response_present": False,
                "authored_action_count": 0,
                "exact_single_action": False,
                "carrier_envelope_valid": False,
                "carrier_error_code": None,
                "action_validation_error_code": None,
                "provider_attempt_count": carrier_metrics.get("provider_attempt_count"),
                "provider_elapsed_seconds": carrier_metrics.get("provider_elapsed_seconds"),
            }
            if _canonical(carrier_metrics) != _canonical(expected_failed_metrics):
                issues.append(f"{path}.carrier_metrics differs from provider failure")
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
            if (
                result.get("status") == "success"
                and result.get("episode_ended") is True
            ):
                if terminal_result_index is None:
                    terminal_result_index = turn_index
                    terminal_result = result
                else:
                    issues.append(f"{path} contains a second terminal result")
    if any(key in record for key in ("gold_sql", "gold_rows", "gold_result")):
        issues.append(f"{root} stores hidden gold content")
    if carrier_declared and _canonical(dict(sorted(audited_provider_usage.items()))) != _canonical(
        record.get("provider_usage")
    ):
        issues.append(f"{root}.provider_usage differs from the sum of turn usage")
    if carrier_declared:
        final_runtime = record.get("final_runtime")
        if not isinstance(final_runtime, Mapping):
            issues.append(f"{root}.final_runtime is missing")
            final_runtime = {}
        has_terminal_answer = terminal_result_index is not None
        terminal_table = final_runtime.get("terminal_table")
        runtime_done = final_runtime.get("done")
        runtime_failure_type = final_runtime.get("failure_type")
        if record.get("legal") is not has_terminal_answer:
            issues.append(f"{root}.legal differs from terminal outcome")
        correct = record.get("correct")
        if not isinstance(correct, bool):
            issues.append(f"{root}.correct is not boolean")
            correct = False
        if record.get("official_execution_accuracy") is not correct:
            issues.append(
                f"{root}.official_execution_accuracy differs from correct"
            )
        if correct and not has_terminal_answer:
            issues.append(f"{root}.correct is true without a terminal answer")
        if has_terminal_answer:
            if terminal_result_index != len(turns) - 1:
                issues.append(f"{root} terminal answer is not the final turn")
            if runtime_done is not True:
                issues.append(f"{root}.final_runtime.done is false after terminal answer")
            expected_table = (
                terminal_result.get("table")
                if isinstance(terminal_result, Mapping)
                else None
            )
            if terminal_table != expected_table or not isinstance(terminal_table, str):
                issues.append(
                    f"{root}.final_runtime.terminal_table differs from terminal result"
                )
            if runtime_failure_type is not None:
                issues.append(
                    f"{root}.final_runtime.failure_type is set after terminal answer"
                )
            scorer_error_type = record.get("scorer_error_type")
            expected_record_failure = (
                None
                if correct
                else (
                    "hidden_verifier_error"
                    if isinstance(scorer_error_type, str) and scorer_error_type
                    else "wrong_answer"
                )
            )
            if record.get("failure_type") != expected_record_failure:
                issues.append(
                    f"{root}.failure_type differs from terminal scoring outcome"
                )
        else:
            if terminal_table is not None:
                issues.append(
                    f"{root}.final_runtime has terminal_table without terminal result"
                )
            if runtime_done is True:
                if not isinstance(runtime_failure_type, str) or not runtime_failure_type:
                    issues.append(
                        f"{root}.final_runtime terminal failure_type is missing"
                    )
                elif record.get("failure_type") != runtime_failure_type:
                    issues.append(
                        f"{root}.failure_type differs from terminal runtime failure"
                    )
            elif not isinstance(record.get("failure_type"), str) or not record.get(
                "failure_type"
            ):
                issues.append(f"{root}.failure_type is missing for nonterminal record")
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


def _response_envelope_sha256(message: Any) -> str | None:
    if not isinstance(message, Mapping):
        return None
    return hashlib.sha256(_canonical(message).encode("utf-8")).hexdigest()


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
    try:
        carrier = normalize_carrier(record.get("carrier", DEFAULT_CARRIER))
    except ValueError:
        carrier = DEFAULT_CARRIER
        issues.append(f"record[{record_index}].carrier is invalid under fresh replay")
    try:
        checkpoint_guidance_profile = normalize_checkpoint_guidance_profile(
            record.get(
                "checkpoint_guidance_profile",
                DEFAULT_CHECKPOINT_GUIDANCE_PROFILE,
            )
        )
    except ValueError:
        checkpoint_guidance_profile = DEFAULT_CHECKPOINT_GUIDANCE_PROFILE
        issues.append(
            f"record[{record_index}].checkpoint_guidance_profile is invalid under fresh replay"
        )
    expected_commit_eligibility = checkpoint_commit_eligibility_for_guidance_profile(
        checkpoint_guidance_profile
    )
    try:
        atomic_operator_profile = normalize_atomic_operator_profile(
            record.get("atomic_operator_profile", DEFAULT_ATOMIC_OPERATOR_PROFILE)
        )
    except ValueError:
        atomic_operator_profile = DEFAULT_ATOMIC_OPERATOR_PROFILE
        issues.append(
            f"record[{record_index}].atomic_operator_profile is invalid under fresh replay"
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
            checkpoint_commit_eligibility_policy=runtime_payload.get(
                "checkpoint_commit_eligibility_policy",
                CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
            ),
        )
        if (
            config.checkpoint_commit_eligibility_policy
            != expected_commit_eligibility
        ):
            issues.append(
                f"record[{record_index}].runtime_config checkpoint commit eligibility "
                "differs from guidance profile under fresh replay"
            )
        runtime = CheckpointRelalgRuntime(
            connection,
            mode=str(record.get("mode")),
            atomic_operator_profile=atomic_operator_profile,
            config=config,
        )
        for turn_index, turn in enumerate(record.get("turns") or []):
            path = f"record[{record_index}].turns[{turn_index}]"
            if runtime.done:
                issues.append(f"{path} appears after replay runtime reached terminal state")
                continue
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
            if isinstance(model_input, list) and model_input:
                expected_system = {
                    "role": "system",
                    "content": get_system_prompt(
                        str(record.get("mode")),
                        teacher=True,
                        carrier=carrier,
                        checkpoint_guidance_profile=checkpoint_guidance_profile,
                        atomic_operator_profile=atomic_operator_profile,
                    ),
                }
                if _canonical(model_input[0]) != _canonical(expected_system):
                    issues.append(f"{path}.model_input system differs under fresh replay")
                issues.extend(
                    provider_history_issues(
                        model_input,
                        root=f"{path}.model_input",
                        carrier=carrier,
                    )
                )
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
                if carrier == CARRIER_NATIVE_TOOL_CALLS:
                    validator = lambda active_mode, active_message: validate_native_assistant_message(  # noqa: E731
                        active_mode,
                        active_message,
                        atomic_operator_profile=atomic_operator_profile,
                    )
                    error_type = NativeToolCallError
                    attempted_from_message = attempted_action_from_native_message
                else:
                    validator = lambda active_mode, active_message: validate_text_json_assistant_message(  # noqa: E731
                        active_mode,
                        active_message,
                        atomic_operator_profile=atomic_operator_profile,
                    )
                    error_type = TextJSONActionError
                    attempted_from_message = attempted_action_from_text_json_message
                try:
                    authored = validator(str(record.get("mode")), assistant)
                except error_type as exc:
                    attempted = attempted_from_message(assistant)
                    details = {"argument_path": exc.path}
                    if carrier == CARRIER_NATIVE_TOOL_CALLS:
                        details["call_count"] = len(assistant.get("tool_calls") or [])
                    else:
                        details["carrier"] = carrier
                    actual = runtime.reject_native_turn(
                        code=exc.code,
                        message=exc.message,
                        details=details,
                        attempted_tool=attempted["tool"],
                        attempted_arguments=attempted["arguments"],
                    )
                else:
                    actual = runtime.apply(authored["tool"], authored["arguments"])
            elif (
                isinstance(
                    turn.get(
                        "native_rejection"
                        if carrier == CARRIER_NATIVE_TOOL_CALLS
                        else "text_json_rejection"
                    ),
                    Mapping,
                )
                and isinstance(expected, Mapping)
            ):
                # A carrier rejection without its raw authored assistant message
                # cannot establish what was actually attempted.
                issues.append(f"{path} carrier rejection has no authored assistant message")
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
        replay_has_answer = runtime.terminal_table is not None
        if record.get("legal") is not replay_has_answer:
            issues.append(f"record[{record_index}].legal differs under fresh replay")
        correct = record.get("correct")
        if correct is True and not replay_has_answer:
            issues.append(
                f"record[{record_index}].correct is true without replay answer"
            )
        if replay_has_answer:
            scorer_error_type = record.get("scorer_error_type")
            expected_failure_type = (
                None
                if correct is True
                else (
                    "hidden_verifier_error"
                    if isinstance(scorer_error_type, str) and scorer_error_type
                    else "wrong_answer"
                )
            )
            if record.get("failure_type") != expected_failure_type:
                issues.append(
                    f"record[{record_index}].failure_type differs from replay terminal outcome"
                )
        elif runtime.done and record.get("failure_type") != runtime.failure_type:
            issues.append(
                f"record[{record_index}].failure_type differs from replay runtime failure"
            )
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
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _manifest_config_sha256(manifest: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"config_sha256", "created_at_utc"}
    }
    return _canonical_sha256(payload)


def _task_example_index(task: Mapping[str, Any], position: int) -> int:
    value = task.get("example_index", task.get("index", position))
    if isinstance(value, bool) or not isinstance(value, int):
        return position
    return value


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
        position_ids.append({"position": position, "example_id": example_id})
        public_tasks.append({
            "position": position,
            "example_id": example_id,
            "example_index": _task_example_index(task, position),
            "db_id": task.get("db_id") if isinstance(task.get("db_id"), str) else None,
            "question_sha256": _canonical_sha256(task.get("question")),
            "external_knowledge_sha256": _canonical_sha256(
                task.get("external_knowledge")
            ),
        })
    identity: dict[str, Any] = {
        "identity_version": SELECTION_IDENTITY_VERSION,
        "start": start,
        "requested_size": requested_size,
        "selected_count": len(selected),
        "position_example_id_sequence_sha256": _canonical_sha256(position_ids),
        "public_task_identity_sha256": _canonical_sha256(public_tasks),
        "public_identity_fields": list(PUBLIC_IDENTITY_FIELDS),
    }
    identity["selection_identity_sha256"] = _canonical_sha256(identity)
    return identity


def _resolve_source_path(value: str, *, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _dataset_source_identity(
    source_manifest: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
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
            str(_resolve_source_path(source_path, project_root=project_root))
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


def _manifest_binding_report(
    manifest: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    issues: list[str] = []
    strict_batch = _is_strict_batch_manifest(manifest)
    stored_config_sha256 = manifest.get("config_sha256")
    if strict_batch or "config_sha256" in manifest:
        if not _valid_sha256(stored_config_sha256):
            issues.append("manifest.config_sha256 is missing or invalid")
        elif stored_config_sha256 != _manifest_config_sha256(manifest):
            issues.append("manifest.config_sha256 differs from canonical manifest content")
    run_started_at = manifest.get("run_started_at_utc")
    if "carrier" in manifest and not isinstance(run_started_at, str):
        issues.append("manifest.run_started_at_utc is missing")
    elif isinstance(run_started_at, str):
        try:
            parsed_run_started_at = datetime.fromisoformat(run_started_at)
        except ValueError:
            issues.append("manifest.run_started_at_utc is invalid")
        else:
            if parsed_run_started_at.tzinfo is None:
                issues.append("manifest.run_started_at_utc is not timezone-aware")
    manifest_carrier = manifest.get("carrier")
    if manifest_carrier in CARRIERS:
        expected_response_envelope_version = provider_response_envelope_version(
            manifest_carrier
        )
        if (
            manifest.get("provider_response_envelope_version")
            != expected_response_envelope_version
        ):
            issues.append(
                "manifest.provider_response_envelope_version does not match "
                "the current carrier response envelope"
            )
    bound_fields = (
        "tool_scheme_registry_version",
        "tool_scheme",
        "protocol_version",
        "protocol_hash",
        "assistant_carrier",
        "top_level_tools",
        "atomic_tools",
        "max_batch_calls",
        "mode",
        "atomic_operator_profile",
        "tool_schema_sha256",
        "student_prompt_sha256",
        "teacher_prompt_sha256",
        "admission_status",
        "capability_manifest",
        "backend",
        "dialect",
        "environment_renderer_version",
        "checkpoint_policy_version",
        "checkpoint_guidance_profile",
        "checkpoint_commit_eligibility_policy",
        "executor_version",
        "tool_schema_hash",
        "carrier_ablation_protocol_version",
        "carrier_policy_version",
        "provider_response_envelope_version",
        "experiment_arm",
        "within_batch_order",
        "carrier",
        "prompt_hash",
        "provider_request_options",
        "runtime_config",
        "denotation_comparison",
        "strict_artifact_audit_version",
        "sft_export_eligible",
        "rl_admission_eligible",
    )
    strict_batch_fields = (
        "runner",
        "dataset_manifest_sha256",
        "dataset_source_identity_sha256",
        "selection_identity_sha256",
        "selection_start",
        "selection_requested_size",
        "batch_control_version",
        "batch_limits",
    )
    if strict_batch:
        for field in (
            "dataset_manifest",
            "dataset_manifest_sha256",
            "dataset_source_identity",
            "selection_identity",
            "batch_control_version",
            "batch_limits",
        ):
            if field not in manifest:
                issues.append(f"manifest.{field} is missing for the v2 batch runner")
    strict_record_expected = {
        "runner": manifest.get("runner"),
        "dataset_manifest_sha256": manifest.get("dataset_manifest_sha256"),
        "dataset_source_identity_sha256": (
            manifest.get("dataset_source_identity", {}).get(
                "dataset_source_identity_sha256"
            )
            if isinstance(manifest.get("dataset_source_identity"), Mapping)
            else None
        ),
        "selection_identity_sha256": (
            manifest.get("selection_identity", {}).get("selection_identity_sha256")
            if isinstance(manifest.get("selection_identity"), Mapping)
            else None
        ),
        "selection_start": manifest.get("start"),
        "selection_requested_size": manifest.get("requested_size"),
        "batch_control_version": manifest.get("batch_control_version"),
        "batch_limits": manifest.get("batch_limits"),
    }
    for record_index, record in enumerate(records):
        for field in bound_fields:
            if _canonical(record.get(field)) != _canonical(manifest.get(field)):
                issues.append(
                    f"record[{record_index}].{field} differs from manifest"
                )
        if strict_batch:
            for field in strict_batch_fields:
                if _canonical(record.get(field)) != _canonical(
                    strict_record_expected[field]
                ):
                    issues.append(
                        f"record[{record_index}].{field} differs from manifest identity"
                    )
    return {
        "passed": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "bound_fields": list(bound_fields) + (
            list(strict_batch_fields) if strict_batch else []
        ),
    }


def _cohort_identity_report(
    manifest: Mapping[str, Any],
    records: list[Mapping[str, Any]],
    *,
    tasks_json: Path | None,
    tasks: list[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    strict_batch = _is_strict_batch_manifest(manifest)
    issues: list[str] = []
    if not strict_batch:
        return {
            "required": False,
            "complete": True,
            "passed": True,
            "issue_count": 0,
            "issues": [],
        }
    if tasks_json is None or not tasks_json.exists() or tasks is None:
        issues.append("dataset is unavailable for v2 cohort identity verification")
        return {
            "required": True,
            "complete": False,
            "passed": False,
            "issue_count": len(issues),
            "issues": issues,
        }

    actual_dataset_sha256 = _file_sha256(tasks_json)
    if manifest.get("dataset_sha256") != actual_dataset_sha256:
        issues.append("manifest.dataset_sha256 differs from the actual dataset file")

    source_manifest_value = manifest.get("dataset_manifest")
    source_manifest_path = (
        Path(source_manifest_value).expanduser().resolve()
        if isinstance(source_manifest_value, str) and source_manifest_value
        else None
    )
    expected_source_identity: dict[str, Any] | None = None
    if source_manifest_path is None or not source_manifest_path.exists():
        issues.append("manifest.dataset_manifest is unavailable")
    else:
        actual_source_manifest_sha256 = _file_sha256(source_manifest_path)
        if manifest.get("dataset_manifest_sha256") != actual_source_manifest_sha256:
            issues.append(
                "manifest.dataset_manifest_sha256 differs from the actual source manifest"
            )
        try:
            source_manifest = json.loads(
                source_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(
                "manifest.dataset_manifest cannot be decoded: "
                f"{type(exc).__name__}"
            )
        else:
            if not isinstance(source_manifest, Mapping):
                issues.append("manifest.dataset_manifest root is not an object")
            else:
                expected_source_identity = _dataset_source_identity(
                    source_manifest,
                    project_root=Path(__file__).resolve().parents[3],
                )
                if _canonical(manifest.get("dataset_source_identity")) != _canonical(
                    expected_source_identity
                ):
                    issues.append(
                        "manifest.dataset_source_identity differs from source manifest"
                    )
                if (
                    expected_source_identity.get("harness_source_path")
                    != str(tasks_json.expanduser().resolve())
                ):
                    issues.append(
                        "dataset source identity does not name the audited dataset"
                    )
                if (
                    expected_source_identity.get("harness_source_sha256")
                    != actual_dataset_sha256
                ):
                    issues.append(
                        "dataset source identity hash differs from the audited dataset"
                    )
                if expected_source_identity.get("harness_source_records") != len(tasks):
                    issues.append(
                        "dataset source identity record count differs from the dataset"
                    )
                if expected_source_identity.get("order_preserved") is not True:
                    issues.append("dataset source manifest does not certify preserved order")

    start = manifest.get("start")
    requested_size = manifest.get("requested_size")
    expected_selection: dict[str, Any] | None = None
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or start < 0
        or isinstance(requested_size, bool)
        or not isinstance(requested_size, int)
        or requested_size < 1
    ):
        issues.append("manifest start/requested_size is invalid")
    else:
        expected_selection = _selection_identity(
            tasks,
            start=start,
            requested_size=requested_size,
        )
        if expected_selection["selected_count"] != requested_size:
            issues.append("requested task slice extends beyond the dataset")
        if _canonical(manifest.get("selection_identity")) != _canonical(
            expected_selection
        ):
            issues.append("manifest.selection_identity differs from the dataset slice")

        positions = [record.get("task_position") for record in records]
        expected_positions = list(range(start, start + requested_size))
        if len(records) != requested_size:
            issues.append(
                "result directory is incomplete: record count differs from requested_size"
            )
        if positions != expected_positions:
            issues.append(
                "record task positions do not exactly match the requested ordered slice"
            )
        if len(set(positions)) != len(positions):
            issues.append("record task positions are not unique")
        example_ids = [record.get("example_id") for record in records]
        if any(not isinstance(value, str) or not value for value in example_ids):
            issues.append("record example IDs must be nonempty strings")
        elif len(set(example_ids)) != len(example_ids):
            issues.append("record example IDs are not unique")
        selected_tasks = list(enumerate(tasks))[start : start + requested_size]
        expected_example_ids = [task.get("example_id") for _, task in selected_tasks]
        if example_ids != expected_example_ids:
            issues.append("record example IDs differ from the requested dataset slice")

    complete = (
        isinstance(requested_size, int)
        and not isinstance(requested_size, bool)
        and len(records) == requested_size
    )
    return {
        "required": True,
        "complete": complete,
        "passed": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "actual_dataset_sha256": actual_dataset_sha256,
        "expected_dataset_source_identity_sha256": (
            expected_source_identity.get("dataset_source_identity_sha256")
            if expected_source_identity is not None
            else None
        ),
        "expected_selection_identity_sha256": (
            expected_selection.get("selection_identity_sha256")
            if expected_selection is not None
            else None
        ),
    }


def _provider_budget_report(
    manifest: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    strict_batch = _is_strict_batch_manifest(manifest)
    issues: list[str] = []
    if not strict_batch:
        return {
            "required": False,
            "passed": True,
            "issue_count": 0,
            "issues": [],
            "provider_attempts": sum(
                len(turn.get("provider_attempt_events") or [])
                for record in records
                for turn in (record.get("turns") or [])
                if isinstance(turn, Mapping)
            ),
            "provider_tokens": sum(
                int((record.get("provider_usage") or {}).get("total_tokens", 0))
                for record in records
                if isinstance(record.get("provider_usage"), Mapping)
            ),
        }

    max_tokens = manifest.get("max_tokens")
    max_completion_tokens = manifest.get("max_completion_tokens")
    api_retries = manifest.get("api_retries")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (max_tokens, max_completion_tokens, api_retries)
    ):
        issues.append("manifest provider token/retry bounds are invalid")
        max_tokens = 1
        max_completion_tokens = 1
        api_retries = 1
    elif max_completion_tokens < max_tokens:
        issues.append("manifest.max_completion_tokens is below max_tokens")

    provider_attempts = 0
    provider_tokens = 0
    for record_index, record in enumerate(records):
        usage = record.get("provider_usage")
        total_tokens = usage.get("total_tokens") if isinstance(usage, Mapping) else None
        if isinstance(total_tokens, bool) or not isinstance(total_tokens, int) or total_tokens < 0:
            issues.append(f"record[{record_index}].provider_usage.total_tokens is invalid")
        else:
            provider_tokens += total_tokens
        for turn_index, turn in enumerate(record.get("turns") or []):
            if not isinstance(turn, Mapping):
                continue
            events = turn.get("provider_attempt_events")
            if not isinstance(events, list):
                continue
            provider_attempts += len(events)
            if len(events) > api_retries:
                issues.append(
                    f"record[{record_index}].turns[{turn_index}] exceeds api_retries"
                )
            expected_budget = max_tokens
            for event_index, event in enumerate(events):
                if not isinstance(event, Mapping):
                    continue
                path = (
                    f"record[{record_index}].turns[{turn_index}]"
                    f".provider_attempt_events[{event_index}]"
                )
                if event.get("max_tokens") != expected_budget:
                    issues.append(f"{path}.max_tokens differs from bounded retry progression")
                if (
                    isinstance(event.get("max_tokens"), int)
                    and not isinstance(event.get("max_tokens"), bool)
                    and event["max_tokens"] > max_completion_tokens
                ):
                    issues.append(f"{path}.max_tokens exceeds max_completion_tokens")
                if event.get("shape_category") == "completion_length":
                    if event_index + 1 < len(events) and expected_budget >= max_completion_tokens:
                        issues.append(
                            f"{path} retries a length completion at the completion cap"
                        )
                    expected_budget = min(
                        max_completion_tokens,
                        max(expected_budget + 1, expected_budget * 2),
                    )
                elif event_index + 1 < len(events) and event.get("shape_category") in {
                    "accepted_response",
                    "model_mismatch",
                    "context_overflow",
                    "content_filter",
                }:
                    issues.append(f"{path} is terminal but has a later provider attempt")

    limits = manifest.get("batch_limits")
    max_provider_attempts = (
        limits.get("max_provider_attempts") if isinstance(limits, Mapping) else None
    )
    max_provider_tokens = (
        limits.get("max_provider_tokens") if isinstance(limits, Mapping) else None
    )
    for field, value in (
        ("max_provider_attempts", max_provider_attempts),
        ("max_provider_tokens", max_provider_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            issues.append(f"manifest.batch_limits.{field} is invalid")
    if isinstance(max_provider_attempts, int) and provider_attempts > max_provider_attempts:
        issues.append("provider attempts exceed the batch limit")
    if isinstance(max_provider_tokens, int) and provider_tokens > max_provider_tokens:
        issues.append("provider tokens exceed the batch limit")
    return {
        "required": True,
        "passed": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "provider_attempts": provider_attempts,
        "provider_tokens": provider_tokens,
    }


def _batch_control_report(
    result_dir: Path,
    manifest: Mapping[str, Any],
    records: list[Mapping[str, Any]],
    *,
    manifest_config_sha256: Any,
    all_jsonl_sha256: str,
) -> dict[str, Any]:
    strict_batch = _is_strict_batch_manifest(manifest)
    if not strict_batch:
        return {
            "required": False,
            "complete": True,
            "passed": True,
            "issue_count": 0,
            "issues": [],
        }
    issues: list[str] = []
    status_path = result_dir / "batch_status.json"
    if not status_path.exists():
        issues.append("batch_status.json is missing")
        return {
            "required": True,
            "complete": False,
            "passed": False,
            "issue_count": len(issues),
            "issues": issues,
        }
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"batch_status.json cannot be decoded: {type(exc).__name__}")
        return {
            "required": True,
            "complete": False,
            "passed": False,
            "issue_count": len(issues),
            "issues": issues,
        }
    expected_fields = {
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
    if not isinstance(status, Mapping):
        issues.append("batch_status.json root is not an object")
        status = {}
    elif set(status) != expected_fields:
        issues.append("batch_status.json fields differ from batch-control policy")
    batch_control_version = status.get("batch_control_version")
    if batch_control_version not in {
        LEGACY_BATCH_CONTROL_VERSION,
        BATCH_CONTROL_VERSION,
    }:
        issues.append("batch status control version is invalid")
    if status.get("batch_control_version") != manifest.get("batch_control_version"):
        issues.append("batch status control version differs from manifest")
    if _canonical(status.get("limits")) != _canonical(manifest.get("batch_limits")):
        issues.append("batch status limits differ from manifest")
    if status.get("manifest_config_sha256") != manifest_config_sha256:
        issues.append("batch status manifest hash differs from manifest")
    if status.get("all_jsonl_sha256") != all_jsonl_sha256:
        issues.append("batch status all.jsonl hash differs from current artifact")
    if status.get("requested_size") != manifest.get("requested_size"):
        issues.append("batch status requested size differs from manifest")
    if status.get("completed_records") != len(records):
        issues.append("batch status completed record count differs from all.jsonl")
    recorded_at = status.get("recorded_at_utc")
    if not isinstance(recorded_at, str):
        issues.append("batch status recorded_at_utc is missing")
    else:
        try:
            parsed_recorded_at = datetime.fromisoformat(recorded_at)
        except ValueError:
            issues.append("batch status recorded_at_utc is invalid")
        else:
            if parsed_recorded_at.tzinfo is None:
                issues.append("batch status recorded_at_utc is not timezone-aware")

    provider_attempts = 0
    provider_tokens = 0
    total_provider_failures = 0
    consecutive_provider_failures = 0
    semantic_failures = 0
    consecutive_semantic_failures = 0
    record_wall_seconds = 0.0
    for record_index, record in enumerate(records):
        provider_attempts += sum(
            len(turn.get("provider_attempt_events") or [])
            for turn in (record.get("turns") or [])
            if isinstance(turn, Mapping)
        )
        usage = record.get("provider_usage")
        if isinstance(usage, Mapping):
            value = usage.get("total_tokens")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                provider_tokens += value
        failure_type = record.get("failure_type")
        provider_failure = failure_type in {
            "provider_error",
            "context_length_exceeded",
        }
        if (
            provider_failure
            and batch_control_version == BATCH_CONTROL_VERSION
        ):
            turns = record.get("turns") or []
            provider_error = None
            for turn in reversed(turns):
                if isinstance(turn, Mapping) and isinstance(
                    turn.get("provider_error"), Mapping
                ):
                    provider_error = turn["provider_error"]
                    break
            provider_failure = provider_error_counts_as_batch_failure(
                provider_error
            )
        semantic_failure = (
            record.get("correct") is not True
            and failure_type not in INFRASTRUCTURE_FAILURE_TYPES
        )
        if provider_failure:
            total_provider_failures += 1
            consecutive_provider_failures += 1
        else:
            consecutive_provider_failures = 0
        if semantic_failure:
            semantic_failures += 1
            consecutive_semantic_failures += 1
        else:
            consecutive_semantic_failures = 0
        elapsed = record.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            issues.append(f"record[{record_index}].elapsed_seconds is invalid")
        else:
            record_wall_seconds += float(elapsed)

    counters = status.get("counters")
    expected_counter_fields = {
        "provider_attempts",
        "provider_tokens",
        "total_provider_failures",
        "consecutive_provider_failures",
        "consecutive_semantic_failures",
        "semantic_failures",
        "total_wall_seconds",
    }
    if not isinstance(counters, Mapping) or set(counters) != expected_counter_fields:
        issues.append("batch status counters differ from batch-control policy")
        counters = {}
    expected_counters = {
        "provider_attempts": provider_attempts,
        "provider_tokens": provider_tokens,
        "total_provider_failures": total_provider_failures,
        "consecutive_provider_failures": consecutive_provider_failures,
        "consecutive_semantic_failures": consecutive_semantic_failures,
        "semantic_failures": semantic_failures,
    }
    for field, expected in expected_counters.items():
        if counters.get(field) != expected:
            issues.append(f"batch status counter {field} differs from all.jsonl")
    total_wall_seconds = counters.get("total_wall_seconds")
    if (
        isinstance(total_wall_seconds, bool)
        or not isinstance(total_wall_seconds, (int, float))
        or not math.isfinite(float(total_wall_seconds))
        or float(total_wall_seconds) < record_wall_seconds
    ):
        issues.append("batch status total wall seconds is invalid or below record time")

    limits = manifest.get("batch_limits")
    expected_limit_fields = {
        "max_provider_attempts",
        "max_provider_tokens",
        "max_wall_seconds",
        "max_consecutive_provider_failures",
        "max_total_provider_failures",
        "max_consecutive_semantic_failures",
    }
    if not isinstance(limits, Mapping) or set(limits) != expected_limit_fields:
        issues.append("manifest.batch_limits fields differ from batch-control policy")
        limits = {}
    for field in expected_limit_fields:
        value = limits.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            issues.append(f"manifest.batch_limits.{field} is invalid")
    comparisons = {
        "max_provider_attempts": provider_attempts,
        "max_provider_tokens": provider_tokens,
        "max_wall_seconds": total_wall_seconds,
        "max_consecutive_provider_failures": consecutive_provider_failures,
        "max_total_provider_failures": total_provider_failures,
        "max_consecutive_semantic_failures": consecutive_semantic_failures,
    }
    for field, actual in comparisons.items():
        limit = limits.get(field)
        if (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and isinstance(limit, (int, float))
            and not isinstance(limit, bool)
            and actual > limit
        ):
            issues.append(f"batch counter exceeds {field}")

    state = status.get("state")
    if state not in {"running", "stopped", "completed", "incomplete"}:
        issues.append("batch status state is invalid")
    complete = (
        state == "completed"
        and len(records) == manifest.get("requested_size")
    )
    if state == "completed":
        if status.get("stop_code") is not None:
            issues.append("completed batch has a stop code")
        if status.get("stop_detail") not in ({}, None):
            issues.append("completed batch has stop detail")
        if not complete:
            issues.append("completed batch does not contain the requested cohort")
        if any(record.get("batch_stop_code") is not None for record in records):
            issues.append("completed batch contains record-level stop metadata")
    else:
        if state == "stopped":
            stop_code = status.get("stop_code")
            allowed_stop_codes = {
                "max_provider_attempts",
                "max_provider_tokens",
                "max_wall_seconds",
                "max_consecutive_provider_failures",
                "max_total_provider_failures",
                "max_consecutive_semantic_failures",
                "non_retryable_provider_http",
            }
            if stop_code not in allowed_stop_codes:
                issues.append("stopped batch has an invalid stop code")
            stop_detail = status.get("stop_detail")
            if stop_detail is not None and not isinstance(stop_detail, Mapping):
                issues.append("stopped batch stop detail is not an object")
            else:
                issues.extend(
                    no_leak_issues(stop_detail, root="batch_status.stop_detail")
                )
            if stop_code in comparisons:
                actual = comparisons[stop_code]
                limit = limits.get(stop_code)
                if (
                    not isinstance(stop_detail, Mapping)
                    or set(stop_detail) != {"counter", "limit"}
                    or stop_detail.get("counter") != actual
                    or stop_detail.get("limit") != limit
                ):
                    issues.append("stopped batch cap detail differs from counter/limit")
                if (
                    not isinstance(actual, (int, float))
                    or isinstance(actual, bool)
                    or not isinstance(limit, (int, float))
                    or isinstance(limit, bool)
                    or actual < limit
                ):
                    issues.append("stopped batch trigger is not supported by its counter")
            elif stop_code == "non_retryable_provider_http" and total_provider_failures < 1:
                issues.append("non-retryable provider stop has no provider failure record")
            if stop_code == "non_retryable_provider_http" and records:
                final_turns = records[-1].get("turns") or []
                final_turn = final_turns[-1] if final_turns else None
                final_error = (
                    final_turn.get("provider_error")
                    if isinstance(final_turn, Mapping)
                    else None
                )
                final_http_status = (
                    final_error.get("http_status")
                    if isinstance(final_error, Mapping)
                    else None
                )
                if (
                    not isinstance(final_error, Mapping)
                    or final_error.get("retryable") is not False
                    or isinstance(final_http_status, bool)
                    or not isinstance(final_http_status, int)
                    or final_http_status in {408, 409, 425, 429}
                    or 500 <= final_http_status <= 599
                ):
                    issues.append(
                        "non-retryable provider stop does not match the final record"
                    )
                if _canonical(stop_detail) != _canonical({
                    "http_status": final_http_status,
                    "retryable": False,
                }):
                    issues.append(
                        "non-retryable provider stop detail differs from final record"
                    )
            for record_index, record in enumerate(records[:-1]):
                if record.get("batch_stop_code") is not None:
                    issues.append(
                        f"record[{record_index}] has batch stop metadata before the final record"
                    )
            if records and records[-1].get("batch_stop_code") is not None:
                if records[-1].get("batch_stop_code") != stop_code:
                    issues.append("final record batch stop code differs from batch status")
                if _canonical(records[-1].get("batch_stop_detail")) != _canonical(
                    stop_detail
                ):
                    issues.append("final record batch stop detail differs from batch status")
        elif state == "incomplete":
            if status.get("stop_code") != "runner_interrupted":
                issues.append("incomplete batch has an invalid stop code")
            stop_detail = status.get("stop_detail")
            if (
                not isinstance(stop_detail, Mapping)
                or set(stop_detail) != {"exception_type"}
                or not isinstance(stop_detail.get("exception_type"), str)
                or not stop_detail.get("exception_type")
            ):
                issues.append("incomplete batch has invalid interruption detail")
        elif state == "running":
            if status.get("stop_code") is not None or status.get("stop_detail") is not None:
                issues.append("running batch has stop metadata")
        issues.append(f"batch is incomplete: state={state!r}")

    return {
        "required": True,
        "complete": complete,
        "state": state,
        "stop_code": status.get("stop_code"),
        "passed": not issues and complete,
        "issue_count": len(issues),
        "issues": issues,
        "artifact_sha256": _file_sha256(status_path),
        "recomputed_counters": {
            **expected_counters,
            "minimum_total_wall_seconds": record_wall_seconds,
        },
    }


def audit_result_dir(
    result_dir: Path,
    *,
    tasks_json: Path | None = None,
) -> dict[str, Any]:
    records_path = result_dir / "all.jsonl"
    manifest_path = result_dir / "manifest.json"
    records = load_jsonl(records_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = audit_records(records)
    manifest_binding = _manifest_binding_report(manifest, records)
    report["manifest_binding"] = manifest_binding
    report["artifact_sha256"] = {
        "manifest.json": _file_sha256(manifest_path),
        "all.jsonl": _file_sha256(records_path),
    }
    if tasks_json is None:
        dataset = manifest.get("dataset")
        tasks_json = Path(dataset) if isinstance(dataset, str) and dataset else None
    tasks: list[Mapping[str, Any]] | None = None
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
        if not isinstance(tasks, list) or not all(
            isinstance(task, Mapping) for task in tasks
        ):
            replay = {
                "passed": False,
                "records": len(records),
                "passed_records": 0,
                "failed_records": len(records),
                "issue_count": 1,
                "records_detail": [],
                "issues": ["tasks JSON root is not a list of objects"],
            }
            tasks = None
        else:
            replay = fresh_replay_records(records, tasks)
    report["fresh_replay"] = replay
    cohort_identity = _cohort_identity_report(
        manifest,
        records,
        tasks_json=tasks_json,
        tasks=tasks,
    )
    report["cohort_identity"] = cohort_identity
    provider_budget = _provider_budget_report(manifest, records)
    report["provider_budget"] = provider_budget
    batch_control = _batch_control_report(
        result_dir,
        manifest,
        records,
        manifest_config_sha256=manifest.get("config_sha256"),
        all_jsonl_sha256=report["artifact_sha256"]["all.jsonl"],
    )
    report["batch_control"] = batch_control
    if batch_control.get("required") and batch_control.get("artifact_sha256"):
        report["artifact_sha256"]["batch_status.json"] = batch_control[
            "artifact_sha256"
        ]
    report["passed"] = bool(
        report["passed"]
        and manifest_binding["passed"]
        and replay["passed"]
        and cohort_identity["passed"]
        and provider_budget["passed"]
        and batch_control["passed"]
    )
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
