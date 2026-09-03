#!/usr/bin/env python3
"""Generate causal SFT trajectories through a closed external-model↔harness loop.

Each model turn sees only prefix-visible state. The classic atomic protocols emit one call;
version51+ native-bundle protocols may emit a bounded provider-native bundle whose calls are all validated against the same
pre-state. The harness returns structured observations or actionable errors. A trajectory is
eligible for SFT only when its terminal answer is correct under the hidden denotation verifier;
successful traces may include genuine error-feedback recovery.
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
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from copy import deepcopy
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "harness"))
sys.path.insert(0, str(ROOT / "src" / "sft"))

from denotation import add_denotation_comparison_argument  # noqa: E402
from atomic_database_context import (  # noqa: E402
    CATALOG_BIRD_INSPECT_SEMANTICS_PROFILE,
    CATALOG_CONTEXT_PROFILE,
    CATALOG_CONTEXT_PROFILES,
    DATABASE_CONTEXT_PROFILES,
    FULL_BIRD_CONTEXT_PROFILES,
    build_full_bird_database_context,
    build_full_context_student_prompt,
    build_full_context_teacher_prompt,
    disabled_tools_for_profile,
    enrich_catalog_perception_output,
    model_visible_tool_schema_hash,
    validate_profile_tool,
)
from provider_client import load_api_config  # noqa: E402
from atomic_version40 import (  # noqa: E402
    PROTOCOL_VERSION as VERSION40_PROTOCOL_VERSION,
    STUDENT_SYSTEM_PROMPT as VERSION40_STUDENT_SYSTEM_PROMPT,
    parse_assistant_strict as parse_assistant_strict_version40,
    protocol_hash as protocol_hash_version40,
    provider_system_prompt as provider_system_prompt_version40,
    tool_schema_hash as tool_schema_hash_version40,
)
from atomic_version41 import (  # noqa: E402
    PROTOCOL_VERSION as VERSION41_PROTOCOL_VERSION,
    STUDENT_SYSTEM_PROMPT as VERSION41_STUDENT_SYSTEM_PROMPT,
    parse_assistant_strict as parse_assistant_strict_version41,
    protocol_hash as protocol_hash_version41,
    provider_system_prompt as provider_system_prompt_version41,
    tool_schema_hash as tool_schema_hash_version41,
)
from atomic_version42 import (  # noqa: E402
    PROTOCOL_VERSION as VERSION42_PROTOCOL_VERSION,
    STUDENT_SYSTEM_PROMPT as VERSION42_STUDENT_SYSTEM_PROMPT,
    lower_terminal_evidence as lower_terminal_evidence_version42,
    parse_assistant_strict as parse_assistant_strict_version42,
    protocol_hash as protocol_hash_version42,
    provider_system_prompt as provider_system_prompt_version42,
    tool_schema_hash as tool_schema_hash_version42,
)
from atomic_version43 import (  # noqa: E402
    PROTOCOL_VERSION as VERSION43_PROTOCOL_VERSION,
    STUDENT_SYSTEM_PROMPT as VERSION43_STUDENT_SYSTEM_PROMPT,
    lower_terminal_evidence as lower_terminal_evidence_version43,
    parse_assistant_strict as parse_assistant_strict_version43,
    protocol_hash as protocol_hash_version43,
    provider_system_prompt as provider_system_prompt_version43,
    tool_schema_hash as tool_schema_hash_version43,
)
from atomic_version44 import (  # noqa: E402
    PROTOCOL_VERSION as VERSION44_PROTOCOL_VERSION,
    STUDENT_SYSTEM_PROMPT as VERSION44_STUDENT_SYSTEM_PROMPT,
    parse_assistant_strict as parse_assistant_strict_version44,
    protocol_hash as protocol_hash_version44,
    provider_system_prompt as provider_system_prompt_version44,
    teacher_system_prompt as teacher_system_prompt_version44,
    tool_schema_hash as tool_schema_hash_version44,
)
from atomic_version45 import (  # noqa: E402
    PROTOCOL_VERSION as VERSION45_PROTOCOL_VERSION,
    STUDENT_SYSTEM_PROMPT as VERSION45_STUDENT_SYSTEM_PROMPT,
    parse_assistant_strict as parse_assistant_strict_version45,
    protocol_hash as protocol_hash_version45,
    provider_system_prompt as provider_system_prompt_version45,
    teacher_system_prompt as teacher_system_prompt_version45,
    tool_schema_hash as tool_schema_hash_version45,
)
from atomic_version46 import (  # noqa: E402
    LATEST_OBSERVATION_FULL as VERSION46_LATEST_OBSERVATION_FULL,
    PROTOCOL_VERSION as VERSION46_PROTOCOL_VERSION,
    RESIDENT_STATE_PROFILE as VERSION46_RESIDENT_STATE_PROFILE,
    protocol_hash as protocol_hash_version46,
)
from atomic_version47 import (  # noqa: E402
    LATEST_OBSERVATION_FULL as VERSION47_LATEST_OBSERVATION_FULL,
    PROTOCOL_VERSION as VERSION47_PROTOCOL_VERSION,
    RESIDENT_STATE_PROFILE as VERSION47_RESIDENT_STATE_PROFILE,
    protocol_hash as protocol_hash_version47,
)
from atomic_version48 import (  # noqa: E402
    INTERPRET_BEFORE_ACT_SUFFIX,
    LATEST_OBSERVATION_FULL as VERSION48_LATEST_OBSERVATION_FULL,
    PROTOCOL_VERSION as VERSION48_PROTOCOL_VERSION,
    RESIDENT_STATE_PROFILE as VERSION48_RESIDENT_STATE_PROFILE,
    protocol_hash as protocol_hash_version48,
)
from atomic_version49 import (  # noqa: E402
    LATEST_OBSERVATION_FULL as VERSION49_LATEST_OBSERVATION_FULL,
    PROTOCOL_VERSION as VERSION49_PROTOCOL_VERSION,
    RESIDENT_STATE_PROFILE as VERSION49_RESIDENT_STATE_PROFILE,
    protocol_hash as protocol_hash_version49,
)
from atomic_version50 import (  # noqa: E402
    PROTOCOL_VERSION as VERSION50_PROTOCOL_VERSION,
    PROVIDER_ASSISTANT_CARRIER as VERSION50_PROVIDER_ASSISTANT_CARRIER,
    protocol_hash as protocol_hash_version50,
)
from tool_modules.native_tool_bundle.protocol import (  # noqa: E402
    PROTOCOL_VERSION as VERSION51_PROTOCOL_VERSION,
    PROVIDER_ASSISTANT_CARRIER as VERSION51_PROVIDER_ASSISTANT_CARRIER,
    protocol_hash as protocol_hash_version51,
)
from tool_modules.native_tool_bundle.compact_protocol import (  # noqa: E402
    PROTOCOL_VERSION as VERSION52_PROTOCOL_VERSION,
    STUDENT_SYSTEM_PROMPT as VERSION52_STUDENT_SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT as VERSION52_TEACHER_SYSTEM_PROMPT,
    protocol_hash as protocol_hash_version52,
)
from tool_modules.native_tool_bundle.compact_prompt import (  # noqa: E402
    PROMPT_PROFILE as VERSION52_PROMPT_PROFILE,
    static_request_size_audit as native_compact_prompt_size_audit,
)
from tool_modules.native_tool_bundle.reviewed_protocol import (  # noqa: E402
    PROTOCOL_VERSION as VERSION53_PROTOCOL_VERSION,
    STUDENT_SYSTEM_PROMPT as VERSION53_STUDENT_SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT as VERSION53_TEACHER_SYSTEM_PROMPT,
    protocol_hash as protocol_hash_version53,
)
from tool_modules.native_tool_bundle.reviewed_prompt import (  # noqa: E402
    PROMPT_PROFILE as VERSION53_PROMPT_PROFILE,
    static_request_size_audit as native_reviewed_prompt_size_audit,
)
from tool_modules.native_tool_bundle.no_plan_protocol import (  # noqa: E402
    MODEL_ARG_SCHEMA as VERSION54_MODEL_ARG_SCHEMA,
    PROTOCOL_VERSION as VERSION54_PROTOCOL_VERSION,
    STUDENT_SYSTEM_PROMPT as VERSION54_STUDENT_SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT as VERSION54_TEACHER_SYSTEM_PROMPT,
    protocol_hash as protocol_hash_version54,
    tool_schema_hash as tool_schema_hash_version54,
    validate_model_action as validate_model_action_version54,
)
from tool_modules.native_tool_bundle.no_plan_prompt import (  # noqa: E402
    PROMPT_PROFILE as VERSION54_PROMPT_PROFILE,
    static_request_size_audit as native_no_plan_prompt_size_audit,
)
from tool_modules.native_tool_bundle.bundle_credit import (  # noqa: E402
    aggregate_summaries as aggregate_native_bundle_credit,
    analyze_record as analyze_native_bundle_credit,
)
from tool_modules.native_tool_bundle.provider_tools import (  # noqa: E402
    MAX_NATIVE_BUNDLE_CALLS,
    native_bundle_history_messages,
    native_atomic_tools,
)
from prompt_contract import TEACHER_ONE_ACTION_RULE  # noqa: E402
from provider_adapter import (  # noqa: E402
    DEEPSEEK_CARRIER_CHOICES,
    DEEPSEEK_CARRIER_JSON_OUTPUT,
    DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
    DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
    adapt_provider_response,
    is_deepseek_split_model,
    provider_default_max_tokens,
    provider_request_messages,
    provider_request_audit_options,
    provider_request_options,
    provider_rejection_message,
    provider_system_prompt,
)
from rollout import (  # noqa: E402
    ChatAPIError,
    ContextOverflowError,
    db_path,
    error_limit_reached,
    execute_tool,
    format_tool_error,
    is_context_overflow,
    new_ctx,
    overview,
    score,
    task_db_path,
    task_gold_sql,
    validate_tool_arguments_against_state,
)
from executor import Harness  # noqa: E402
from protocol import (  # noqa: E402
    AdjacentActionGuard,
    MODEL_ARG_SCHEMA,
    POLICY_PROMPT_CANONICAL,
    POLICY_PROMPT_VARIANTS,
    PROTOCOL_VERSION as DEFAULT_PROTOCOL_VERSION,
    ProtocolError,
    RESIDENT_STATE_PROFILE_VERSION39,
    assistant_message,
    first_user_message,
    get_system_prompt,
    model_context_messages,
    parse_assistant_strict,
    policy_system_prompt,
    protocol_hash,
    rolling_legal_history_messages,
    rolling_system_prompt,
    state_context_message,
    teacher_system_prompt,
    tool_error_message,
    tool_schema_hash,
    tool_output_message,
    validate_model_action,
)
from tool_modules.registry import (  # noqa: E402
    ATOMIC_ASSISTANT_CARRIER,
    ATOMIC_TOOL_SCHEME,
    NATIVE_TOOL_BUNDLE_SCHEME,
    TOOL_SCHEME_REGISTRY_VERSION,
)
from training_result_quality import (  # noqa: E402
    EMPTY_RESULT_POLICY_VERSION,
    apply_empty_result_target_annotation,
    training_quality_summary,
)

SPIDER = ROOT / "data" / "spider_data"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_ERRORS_PER_TYPE = 3
DEFAULT_MAX_STEPS = 30
DEFAULT_MAX_TOKENS = 1024
MIN_CONTEXT_RETRY_TOKENS = 256
MAX_COMPLETION_RETRY_TOKENS = 8192
DIAGNOSTIC_TRAINING_ADMISSION = "diagnostic_only_pending_protocol_scale_gate"
ELIGIBLE_TRAINING_ADMISSION = "eligible_by_current_context_contract"
ATOMIC_PROTOCOL_VERSIONS = (
    DEFAULT_PROTOCOL_VERSION,
    VERSION40_PROTOCOL_VERSION,
    VERSION41_PROTOCOL_VERSION,
    VERSION42_PROTOCOL_VERSION,
    VERSION43_PROTOCOL_VERSION,
    VERSION44_PROTOCOL_VERSION,
    VERSION45_PROTOCOL_VERSION,
    VERSION46_PROTOCOL_VERSION,
    VERSION47_PROTOCOL_VERSION,
    VERSION48_PROTOCOL_VERSION,
    VERSION49_PROTOCOL_VERSION,
    VERSION50_PROTOCOL_VERSION,
    VERSION51_PROTOCOL_VERSION,
    VERSION52_PROTOCOL_VERSION,
    VERSION53_PROTOCOL_VERSION,
    VERSION54_PROTOCOL_VERSION,
)
FULL_REASONING_NO_PLAN_PROTOCOL_VERSIONS = frozenset({
    VERSION40_PROTOCOL_VERSION,
    VERSION41_PROTOCOL_VERSION,
    VERSION42_PROTOCOL_VERSION,
    VERSION43_PROTOCOL_VERSION,
})
NO_PLAN_PROTOCOL_VERSIONS = frozenset({
    *FULL_REASONING_NO_PLAN_PROTOCOL_VERSIONS,
    VERSION44_PROTOCOL_VERSION,
    VERSION45_PROTOCOL_VERSION,
    VERSION54_PROTOCOL_VERSION,
})
CONTEXT_ABLATION_PROTOCOL_VERSIONS = frozenset({
    VERSION46_PROTOCOL_VERSION,
    VERSION47_PROTOCOL_VERSION,
    VERSION48_PROTOCOL_VERSION,
    VERSION49_PROTOCOL_VERSION,
})


def selected_resident_context(protocol_version: str) -> tuple[str, bool]:
    if protocol_version == VERSION49_PROTOCOL_VERSION:
        return VERSION49_RESIDENT_STATE_PROFILE, VERSION49_LATEST_OBSERVATION_FULL
    if protocol_version == VERSION48_PROTOCOL_VERSION:
        return VERSION48_RESIDENT_STATE_PROFILE, VERSION48_LATEST_OBSERVATION_FULL
    if protocol_version == VERSION47_PROTOCOL_VERSION:
        return VERSION47_RESIDENT_STATE_PROFILE, VERSION47_LATEST_OBSERVATION_FULL
    if protocol_version == VERSION46_PROTOCOL_VERSION:
        return VERSION46_RESIDENT_STATE_PROFILE, VERSION46_LATEST_OBSERVATION_FULL
    return RESIDENT_STATE_PROFILE_VERSION39, False


def selected_history_observation_policy(
    protocol_version: str,
    *,
    full_reasoning_no_plan: bool,
) -> str:
    if full_reasoning_no_plan:
        return "unabridged-recent-4"
    if protocol_version == VERSION49_PROTOCOL_VERSION:
        return "compact-recent-4-active-exact-rows-inactive-read-cards"
    if protocol_version in {VERSION47_PROTOCOL_VERSION, VERSION48_PROTOCOL_VERSION}:
        return "latest-unabridged-older-compact-resident-read-cards"
    if protocol_version == VERSION46_PROTOCOL_VERSION:
        return "compact-recent-4-resident-handle-cards"
    return "compact-resident"
DATA_GENERATION_SUFFIX = (
    "\n\nDATA GENERATION STRICTNESS\n"
    + TEACHER_ONE_ACTION_RULE
    + " The reason should be specific to the current question, "
    "visible schema/observations, and the next tool arguments. After the first turn, do not restate "
    "the original user question; continue from the current environment state or error feedback. "
    "Do not repeat the exact same read_subtable arguments when that read is already present in "
    "CURRENT ENVIRONMENT STATE; change conditions, order_by, offset, columns, or limit if different "
    "rows are needed. If a plan item has no evidence yet, omit the evidence "
    "field or set it to null; never use an empty string for evidence."
)
VERSION44_DATA_GENERATION_SUFFIX = (
    "\n\nDATA GENERATION STRICTNESS\n"
    + TEACHER_ONE_ACTION_RULE
    + " The reason should be specific to the current question, visible schema/observations, and "
    "the next tool arguments. After the first turn, do not restate the original user question; "
    "continue from the current environment state or error feedback. Do not repeat the exact same "
    "inspect_rows or search_values arguments when that observation is already present in CURRENT "
    "ENVIRONMENT STATE; change conditions, order_by, offset, columns/column, limit, or query when "
    "different evidence is needed."
)


def selected_protocol_hash(protocol_version: str, system_prompt: str) -> str:
    if protocol_version == VERSION54_PROTOCOL_VERSION:
        return protocol_hash_version54(system_prompt)
    if protocol_version == VERSION53_PROTOCOL_VERSION:
        return protocol_hash_version53(system_prompt)
    if protocol_version == VERSION52_PROTOCOL_VERSION:
        return protocol_hash_version52(system_prompt)
    if protocol_version == VERSION51_PROTOCOL_VERSION:
        return protocol_hash_version51(system_prompt)
    if protocol_version == VERSION50_PROTOCOL_VERSION:
        return protocol_hash_version50(system_prompt)
    if protocol_version == VERSION49_PROTOCOL_VERSION:
        return protocol_hash_version49(system_prompt)
    if protocol_version == VERSION48_PROTOCOL_VERSION:
        return protocol_hash_version48(system_prompt)
    if protocol_version == VERSION47_PROTOCOL_VERSION:
        return protocol_hash_version47(system_prompt)
    if protocol_version == VERSION46_PROTOCOL_VERSION:
        return protocol_hash_version46(system_prompt)
    if protocol_version == VERSION45_PROTOCOL_VERSION:
        return protocol_hash_version45(system_prompt)
    if protocol_version == VERSION44_PROTOCOL_VERSION:
        return protocol_hash_version44(system_prompt)
    if protocol_version == VERSION43_PROTOCOL_VERSION:
        return protocol_hash_version43(system_prompt)
    if protocol_version == VERSION42_PROTOCOL_VERSION:
        return protocol_hash_version42(system_prompt)
    if protocol_version == VERSION41_PROTOCOL_VERSION:
        return protocol_hash_version41(system_prompt)
    if protocol_version == VERSION40_PROTOCOL_VERSION:
        return protocol_hash_version40(system_prompt)
    if protocol_version == DEFAULT_PROTOCOL_VERSION:
        return protocol_hash(system_prompt)
    raise ValueError(f"unknown atomic protocol version {protocol_version!r}")


def selected_tool_schema_hash(protocol_version: str) -> str:
    if protocol_version == VERSION54_PROTOCOL_VERSION:
        return tool_schema_hash_version54()
    if protocol_version in {
        VERSION51_PROTOCOL_VERSION,
        VERSION52_PROTOCOL_VERSION,
        VERSION53_PROTOCOL_VERSION,
    }:
        return tool_schema_hash()
    if protocol_version == VERSION50_PROTOCOL_VERSION:
        return tool_schema_hash()
    if protocol_version in CONTEXT_ABLATION_PROTOCOL_VERSIONS:
        return tool_schema_hash()
    if protocol_version == VERSION45_PROTOCOL_VERSION:
        return tool_schema_hash_version45()
    if protocol_version == VERSION44_PROTOCOL_VERSION:
        return tool_schema_hash_version44()
    if protocol_version == VERSION43_PROTOCOL_VERSION:
        return tool_schema_hash_version43()
    if protocol_version == VERSION42_PROTOCOL_VERSION:
        return tool_schema_hash_version42()
    if protocol_version == VERSION41_PROTOCOL_VERSION:
        return tool_schema_hash_version41()
    if protocol_version == VERSION40_PROTOCOL_VERSION:
        return tool_schema_hash_version40()
    if protocol_version == DEFAULT_PROTOCOL_VERSION:
        return tool_schema_hash()
    raise ValueError(f"unknown atomic protocol version {protocol_version!r}")


def sft_export_eligible(
    *,
    context_mode: str,
    history_turns: int,
    rolling_prompt_variant: str,
    denotation_comparison: str,
    diagnostic_only: bool = False,
) -> bool:
    """Return whether a rollout matches the current SFT/RL causal context contract."""
    return bool(
        not diagnostic_only
        and context_mode == "rolling-legal-history"
        and history_turns == 4
        and rolling_prompt_variant == "full"
        and denotation_comparison == "bird-set"
    )


def training_admission(*, diagnostic_only: bool) -> str:
    return (
        DIAGNOSTIC_TRAINING_ADMISSION
        if diagnostic_only
        else ELIGIBLE_TRAINING_ADMISSION
    )


PLAN_POLICY_OPTIONAL = "optional"
PLAN_POLICY_REQUIRED_RESIDENT = "required-resident"
PLAN_POLICY_CHOICES = (PLAN_POLICY_OPTIONAL, PLAN_POLICY_REQUIRED_RESIDENT)
REQUIRED_RESIDENT_PLAN_SUFFIX = (
    "\n\nREQUIRED RESIDENT PLAN EXPERIMENT\n"
    "Your FIRST valid action must be one plan tool call that creates 2 to 4 concrete "
    "subgoals. Do not execute a data tool before that plan is accepted. The harness stores this "
    "plan in CURRENT ENVIRONMENT STATE, where it remains visible on every later request; plan "
    "actions are deliberately not repeated in the bounded rolling transcript. After obtaining "
    "decisive intermediate evidence, make a batched plan update before answer_from_context: put "
    "all currently changed items into one ops list containing update ops. Update again only after "
    "a later non-plan tool makes new material progress; never repeat a plan update when resident "
    "goal/status/evidence already shows the same state. Each plan update is exactly one action: do "
    "not emit any second tool call or a complete solution in the same response.\n"
    "Required plan argument examples (these are arguments, not extra actions):\n"
    'Initial: {"ops":[{"op":"create","id":"inspect","goal":"Inspect needed schemas",'
    '"status":"pending"},{"op":"create","id":"solve","goal":"Build the exact result table",'
    '"status":"pending"}]}\n'
    'Later batched update: {"ops":[{"op":"update","id":"inspect","status":"done",'
    '"evidence":"step_2"},{"op":"update","id":"solve","status":"in_progress"}]}\n'
    'Every item must include the "op" field. Omit evidence until a real prior step supports it.'
)


class ProviderCarrierError(ChatAPIError):
    """A provider returned no model-visible action after bounded same-request retries."""

    def __init__(self, message: str, *, usage: dict):
        super().__init__(message)
        self.usage = usage


class ResidentPlanPolicyTracker:
    """Enforce the opt-in forced-plan experiment without changing the default protocol."""

    def __init__(self, policy: str = PLAN_POLICY_OPTIONAL):
        if policy not in PLAN_POLICY_CHOICES:
            raise ValueError(f"unknown plan policy {policy!r}")
        self.policy = policy
        self.initial_plan_created = False
        self.non_plan_success_seen = False
        self.non_plan_success_since_update = False
        self.updated_after_work = False

    @property
    def required(self) -> bool:
        return self.policy == PLAN_POLICY_REQUIRED_RESIDENT

    def validate_before_execution(self, tool: str, args: dict) -> None:
        if not self.required:
            return
        if not self.initial_plan_created:
            if tool != "plan":
                raise ProtocolError(
                    "required resident-plan policy: the first valid action must be plan"
                )
            ops = args.get("ops") if isinstance(args, dict) else None
            if (
                not isinstance(ops, list)
                or not 2 <= len(ops) <= 4
                or any(not isinstance(op, dict) or op.get("op") not in {"create", "add"}
                       for op in ops)
            ):
                raise ProtocolError(
                    "required resident-plan policy: the initial plan must contain 2 to 4 "
                    "create/add ops"
                )
            return
        if tool == "plan":
            if not self.non_plan_success_since_update:
                raise ProtocolError(
                    "required resident-plan policy: new non-plan progress is required before "
                    "another batched plan update; do not repeat unchanged plan state"
                )
            ops = args.get("ops") if isinstance(args, dict) else None
            if (
                not isinstance(ops, list)
                or not ops
                or any(not isinstance(op, dict) or op.get("op") != "update" for op in ops)
            ):
                raise ProtocolError(
                    "required resident-plan policy: a later plan call must batch one or more "
                    "update ops"
                )
            return
        if tool == "answer_from_context" and not self.updated_after_work:
            raise ProtocolError(
                "required resident-plan policy: after a non-plan tool succeeds, plan must be "
                "updated with an update op before answer_from_context"
            )

    def record_success(self, tool: str, args: dict) -> None:
        if not self.required:
            return
        if tool == "plan":
            if not self.initial_plan_created:
                self.initial_plan_created = True
                return
            ops = args.get("ops") if isinstance(args, dict) else None
            if self.non_plan_success_seen and isinstance(ops, list) and any(
                isinstance(op, dict) and op.get("op") == "update" for op in ops
            ):
                self.updated_after_work = True
                self.non_plan_success_since_update = False
            return
        self.non_plan_success_seen = True
        self.non_plan_success_since_update = True


def retain_in_rolling_history(tool: str, plan_policy: str) -> bool:
    """Resident-only plan actions should not be duplicated in bounded assistant history."""
    return not (
        plan_policy == PLAN_POLICY_REQUIRED_RESIDENT
        and tool == "plan"
    )


def compact_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def protocol_failure_type(exc: ProtocolError) -> str:
    if exc.failure_type:
        return exc.failure_type
    text = str(exc).lower()
    # Provider carrier violations are response-format failures.  Their actionable retry message
    # includes a canonical JSON example containing the word ``arguments``; keyword classification
    # must not turn that incidental text into an argument-validation error.
    if (
        "split-response transport error" in text
        or "native function-call transport error" in text
    ):
        return "protocol_error"
    if any(marker in text for marker in (
        "arguments", "unexpected", "requires", "missing required", "must be", "must contain",
    )):
        return "argument_validation_error"
    return "protocol_error"


def error_event(action_index: int, error_type: str, message: str,
                state_before: dict, state_after: dict | None = None,
                attempted_tool: str | None = None,
                attempted_arguments: dict | None = None,
                error_code: str | None = None,
                details: dict | None = None) -> dict:
    """Audit a rejected model action without making it an SFT supervision target."""
    state_after = state_before if state_after is None else state_after
    event = {
        "action_index": action_index,
        "step_id": f"step_{action_index}",
        "error_type": error_type,
        "message": message,
        "state_before_hash": __import__("hashlib").sha256(compact_json(state_before).encode()).hexdigest(),
        "state_after_hash": __import__("hashlib").sha256(compact_json(state_after).encode()).hexdigest(),
    }
    if error_code:
        event["error_code"] = error_code
    if details:
        event["details"] = deepcopy(details)
    if attempted_tool:
        event["attempted_tool"] = attempted_tool
        event["attempted_arguments"] = attempted_arguments or {}
    return event


def add_usage(target: collections.Counter, usage: dict | None, prefix: str = "") -> None:
    """Accumulate numeric usage fields; flatten provider-specific nested detail objects."""
    if not isinstance(usage, dict):
        return
    for key, value in usage.items():
        if not prefix and key in {
            "api_finish_reason",
            "api_retry_events",
            "provider_request_options",
            "provider_response_metadata",
            "provider_native_assistant_message",
            "provider_native_rejection_reason",
            "provider_native_bundle_calls",
        }:
            continue
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, (int, float)):
            target[name] += value
        elif isinstance(value, dict):
            add_usage(target, value, name)


def extract_native_function_call(message: dict) -> tuple[str, dict | None, str | None]:
    """Lower one untouched provider tool call to the canonical action JSON payload."""
    content = message.get("content")
    audit_message = {
        "role": "assistant",
        "content": content,
        "reasoning_content": message.get("reasoning_content"),
        "tool_calls": deepcopy(message.get("tool_calls")),
    }
    if content is not None and (not isinstance(content, str) or content.strip()):
        return "", audit_message, "nonempty_assistant_content"
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return "", audit_message, "missing_tool_calls"
    if len(tool_calls) != 1:
        return "", audit_message, f"tool_call_count_{len(tool_calls)}"
    call = tool_calls[0]
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict):
        return "", audit_message, "missing_function_payload"
    tool = function.get("name")
    if tool not in MODEL_ARG_SCHEMA:
        return "", audit_message, "unknown_function"
    raw_arguments = function.get("arguments")
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return "", audit_message, "invalid_arguments_json"
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
        raw_arguments = json.dumps(
            arguments, ensure_ascii=False, separators=(",", ":")
        )
    else:
        return "", audit_message, "arguments_not_object_json"
    if not isinstance(arguments, dict):
        return "", audit_message, "arguments_not_object_json"
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        return "", audit_message, "missing_tool_call_id"
    normalized_message = {
        "role": "assistant",
        "content": content,
        "reasoning_content": message.get("reasoning_content"),
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": tool, "arguments": raw_arguments},
        }],
    }
    action = json.dumps(
        {"tool": tool, "arguments": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return action, normalized_message, None


def extract_native_function_bundle(
    message: dict,
    *,
    model_arg_schema: dict[str, tuple[set[str], set[str]]] | None = None,
) -> tuple[str, dict | None, str | None]:
    """Preserve and lower one bounded provider-native assistant tool-call bundle."""
    active_schema = MODEL_ARG_SCHEMA if model_arg_schema is None else model_arg_schema
    content = message.get("content")
    audit_message = {
        "role": "assistant",
        "content": content,
        "reasoning_content": message.get("reasoning_content"),
        "tool_calls": deepcopy(message.get("tool_calls")),
    }
    if content is not None and not isinstance(content, str):
        return "", audit_message, "invalid_assistant_content_type"
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return "", audit_message, "missing_tool_calls"
    if not 1 <= len(tool_calls) <= MAX_NATIVE_BUNDLE_CALLS:
        return "", audit_message, f"tool_call_count_{len(tool_calls)}"

    calls: list[dict] = []
    normalized_calls: list[dict] = []
    seen_ids: set[str] = set()
    for call in tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            return "", audit_message, "missing_function_payload"
        tool = function.get("name")
        if tool not in active_schema:
            return "", audit_message, "unknown_function"
        raw_arguments = function.get("arguments")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                return "", audit_message, "invalid_arguments_json"
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
            raw_arguments = json.dumps(
                arguments, ensure_ascii=False, separators=(",", ":")
            )
        else:
            return "", audit_message, "arguments_not_object_json"
        if not isinstance(arguments, dict):
            return "", audit_message, "arguments_not_object_json"
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            return "", audit_message, "missing_tool_call_id"
        if call_id in seen_ids:
            return "", audit_message, "duplicate_tool_call_id"
        seen_ids.add(call_id)
        calls.append({"id": call_id, "tool": tool, "arguments": arguments})
        normalized_calls.append({
            "id": call_id,
            "type": "function",
            "function": {"name": tool, "arguments": raw_arguments},
        })
    normalized_message = {
        "role": "assistant",
        "content": content,
        "reasoning_content": message.get("reasoning_content"),
        "tool_calls": normalized_calls,
    }
    payload = json.dumps(
        {"calls": calls}, ensure_ascii=False, separators=(",", ":")
    )
    return payload, normalized_message, None


def native_rejection_is_semantic(reason: str | None) -> bool:
    """Classify authored invalid native calls that should receive harness feedback."""
    if not reason:
        return False
    if reason == "nonempty_assistant_content":
        return True
    if reason.startswith("tool_call_count_"):
        return reason != "tool_call_count_0"
    return reason in {
        "missing_function_payload",
        "unknown_function",
        "invalid_arguments_json",
        "arguments_not_object_json",
        "duplicate_tool_call_id",
        "invalid_assistant_content_type",
    }


def request_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: int,
    deepseek_carrier: str = DEEPSEEK_CARRIER_JSON_OUTPUT,
    native_model_arg_schema: dict[str, tuple[set[str], set[str]]] | None = None,
    chat_template_kwargs: dict[str, object] | None = None,
) -> tuple[str, dict, str]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if not is_deepseek_split_model(model):
        payload["temperature"] = 0
    request_options = provider_request_options(
        model,
        carrier=deepseek_carrier,
        native_model_arg_schema=native_model_arg_schema,
    )
    payload.update(request_options)
    if chat_template_kwargs is not None:
        payload["chat_template_kwargs"] = dict(chat_template_kwargs)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        error_cls = ContextOverflowError if is_context_overflow(detail) else ChatAPIError
        raise error_cls(f"HTTP {exc.code}: {detail[:1200]}", status=exc.code, body=detail) from exc
    except urllib.error.URLError as exc:
        raise ChatAPIError(f"URL error: {exc}") from exc
    choice = data["choices"][0]
    message = choice["message"]
    usage = dict(data.get("usage") or {})
    # A successful HTTP response can still be length-truncated. The bounded caller decides whether
    # to retry the same semantic turn with a larger completion budget.
    usage["api_finish_reason"] = choice.get("finish_reason")
    usage["provider_request_options"] = provider_request_audit_options(
        model,
        carrier=deepseek_carrier,
        native_model_arg_schema=native_model_arg_schema,
    )
    usage["provider_response_metadata"] = {
        key: data.get(key)
        for key in ("id", "model", "system_fingerprint", "object")
        if data.get(key) is not None
    }
    if deepseek_carrier in {
        DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
        DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
    }:
        extractor = (
            extract_native_function_bundle
            if deepseek_carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE
            else extract_native_function_call
        )
        if extractor is extract_native_function_bundle:
            content, native_message, rejection = extractor(
                message,
                model_arg_schema=native_model_arg_schema,
            )
        else:
            content, native_message, rejection = extractor(message)
        usage["provider_native_assistant_message"] = native_message
        usage["provider_native_rejection_reason"] = rejection
        if deepseek_carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE and content:
            usage["provider_native_bundle_calls"] = json.loads(content)["calls"]
        return content, usage, message.get("reasoning_content") or ""
    return message.get("content") or "", usage, message.get("reasoning_content") or ""


def chat_with_retries(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: int,
    retries: int,
    deepseek_carrier: str = DEEPSEEK_CARRIER_JSON_OUTPUT,
    native_model_arg_schema: dict[str, tuple[set[str], set[str]]] | None = None,
    chat_template_kwargs: dict[str, object] | None = None,
) -> tuple[str, dict, str]:
    last: Exception | None = None
    budget = max_tokens
    transport_retries = 0
    context_retries = 0
    completion_retries = 0
    carrier_retries = 0
    retry_events: list[dict] = []
    accumulated_usage: collections.Counter = collections.Counter()
    for attempt in range(max(1, retries)):
        try:
            request_kwargs = {
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
                "messages": messages,
                "max_tokens": budget,
                "timeout": timeout,
                "deepseek_carrier": deepseek_carrier,
                "native_model_arg_schema": native_model_arg_schema,
            }
            if chat_template_kwargs is not None:
                request_kwargs["chat_template_kwargs"] = chat_template_kwargs
            text, response_usage, reasoning = request_chat(**request_kwargs)
            response_usage = dict(response_usage or {})
            add_usage(accumulated_usage, response_usage)
            finish_reason = response_usage.get("api_finish_reason")
            if (
                finish_reason == "length"
                and attempt + 1 < max(1, retries)
                and budget < MAX_COMPLETION_RETRY_TOKENS
            ):
                retry_events.append({
                    "type": "completion_length",
                    "request_attempt": attempt + 1,
                    "max_tokens": budget,
                    "finish_reason": finish_reason,
                    "visible_content_present": bool(text.strip()),
                    "reasoning_content_present": bool(reasoning.strip()),
                    "provider_response_metadata": deepcopy(
                        response_usage.get("provider_response_metadata") or {}
                    ),
                })
                budget = min(MAX_COMPLETION_RETRY_TOKENS, budget * 2)
                completion_retries += 1
                continue
            native_rejection = response_usage.get("provider_native_rejection_reason")
            native_semantic_rejection = (
                deepseek_carrier in {
                    DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
                    DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
                }
                and native_rejection_is_semantic(native_rejection)
            )
            carrier_empty = (
                is_deepseek_split_model(model)
                and finish_reason != "length"
                and not native_semantic_rejection
                and (
                    not text.strip()
                    or (
                        deepseek_carrier in {
                            DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
                            DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
                        }
                        and (
                            native_rejection
                            or (
                                deepseek_carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS
                                and not reasoning.strip()
                            )
                        )
                    )
                )
            )
            if carrier_empty:
                carrier_event = {
                    "type": (
                        "provider_native_carrier_invalid"
                        if deepseek_carrier in {
                            DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
                            DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
                        }
                        else "provider_carrier_empty"
                    ),
                    "request_attempt": attempt + 1,
                    "max_tokens": budget,
                    "finish_reason": finish_reason,
                    "visible_content_present": False,
                    "reasoning_content_present": bool(reasoning.strip()),
                    "provider_response_metadata": deepcopy(
                        response_usage.get("provider_response_metadata") or {}
                    ),
                }
                if deepseek_carrier in {
                    DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS,
                    DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE,
                }:
                    carrier_event["native_rejection_reason"] = native_rejection
                retry_events.append(carrier_event)
                if attempt + 1 < max(1, retries):
                    carrier_retries += 1
                    time.sleep(min(2 ** attempt, 8))
                    continue

            usage = dict(accumulated_usage)
            usage["api_finish_reason"] = finish_reason
            usage["provider_request_options"] = deepcopy(
                response_usage.get("provider_request_options") or {}
            )
            usage["provider_response_metadata"] = deepcopy(
                response_usage.get("provider_response_metadata") or {}
            )
            for key in (
                "provider_native_assistant_message",
                "provider_native_rejection_reason",
                "provider_native_bundle_calls",
            ):
                if key in response_usage:
                    usage[key] = deepcopy(response_usage.get(key))
            # These are client request retries, deliberately separate from environment error events.
            usage["api_request_attempts"] = attempt + 1
            usage["api_transport_retries"] = transport_retries
            usage["api_context_retries"] = context_retries
            usage["api_completion_retries"] = completion_retries
            usage["api_carrier_retries"] = carrier_retries
            if retry_events:
                usage["api_retry_events"] = retry_events
            if carrier_empty:
                raise ProviderCarrierError(
                    "DeepSeek provider carrier returned no executable single action after "
                    f"{attempt + 1} request attempts",
                    usage=usage,
                )
            return text, usage, reasoning
        except ContextOverflowError:
            if budget <= MIN_CONTEXT_RETRY_TOKENS:
                raise
            budget = max(MIN_CONTEXT_RETRY_TOKENS, budget // 2)
            context_retries += 1
        except ChatAPIError as exc:
            last = exc
            status = exc.status
            retryable = (
                status is None
                or status in {408, 409, 425, 429}
                or status >= 500
            )
            if not retryable:
                raise
            if attempt + 1 < retries:
                transport_retries += 1
                time.sleep(min(2 ** attempt, 8))
        except Exception as exc:  # noqa: BLE001 - API surfaces many transient transport errors
            last = exc
            if attempt + 1 < retries:
                transport_retries += 1
                time.sleep(min(2 ** attempt, 8))
    if last:
        raise last
    raise ChatAPIError("chat request failed without an exception")


def split_path(split: str) -> Path:
    if split == "train":
        return SPIDER / "train_spider.json"
    if split == "dev":
        return SPIDER / "dev.json"
    raise ValueError(f"unsupported split: {split}")


def load_examples(args: argparse.Namespace) -> list[tuple[int, dict]]:
    if args.examples_file:
        path = Path(args.examples_file)
        if path.suffix == ".jsonl":
            examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            examples = payload.get("examples", payload) if isinstance(payload, dict) else payload
        if not isinstance(examples, list):
            raise ValueError(f"{args.examples_file}: expected JSONL records, a JSON list, or {{'examples': [...]}}")
        indexed = [(int(ex.get("example_index", i)), ex) for i, ex in enumerate(examples)]
    else:
        examples = json.loads(split_path(args.split).read_text(encoding="utf-8"))
        indexed = list(enumerate(examples))

    if args.indices_file:
        payload = json.loads(Path(args.indices_file).read_text(encoding="utf-8"))
        requested = payload.get("indices") if isinstance(payload, dict) else payload
        if not isinstance(requested, list) or any(
            not isinstance(index, int) for index in requested
        ):
            raise ValueError("--indices-file must contain an integer list or {'indices': [...]}")
        if len(set(requested)) != len(requested):
            raise ValueError("--indices-file contains duplicate indices")
        by_index = {index: ex for index, ex in indexed}
        missing = [index for index in requested if index not in by_index]
        if missing:
            raise ValueError(f"--indices-file has indices absent from the source: {missing}")
        indexed = [(index, by_index[index]) for index in requested]
    if args.start:
        indexed = indexed[args.start:]
    if args.limit:
        indexed = indexed[: args.limit]
    return [(i, ex) for i, ex in indexed if Path(task_db_path(ex)).exists()]


def trajectory_id(split: str, example_index: int, ex: dict) -> str:
    return str(ex.get("trajectory_id") or ex.get("example_id") or f"rollout_{split}_{example_index}")


def observation_for_error(step_id: str, error_type: str, message: str) -> str:
    return tool_error_message(step_id, error_type, message)


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that treats punctuation followed by a closing quote as a boundary."""
    parts: list[str] = []
    start = 0
    i = 0
    while i < len(text):
        if text[i] in ".!?":
            end = i + 1
            while end < len(text) and text[end] in "\"'”’)]}":
                end += 1
            if end >= len(text) or text[end].isspace():
                piece = text[start:end].strip()
                if piece:
                    parts.append(piece)
                while end < len(text) and text[end].isspace():
                    end += 1
                start = end
                i = end
                continue
        i += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def normalize_reasoning_text(text: str) -> str:
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
    return " ".join(text.split())


def brief_reasoning(text: str, limit: int = 360) -> str:
    text = normalize_reasoning_text(text)
    if len(text) <= limit:
        return text
    parts = split_sentences(text)
    kept = []
    total = 0
    for part in parts:
        if total + len(part) + 1 > limit:
            break
        kept.append(part)
        total += len(part) + 1
    return " ".join(kept).strip() or text[:limit].rstrip() + "..."


QUESTION_RESTATEMENT = re.compile(
    r"^\s*(?:"
    r"the\s+(?:question|user)\s+(?:asks|wants|is\s+asking)|"
    r"we\s+need\s+to\s+(?:answer|find|list|show|compute|determine)|"
    r"i\s+need\s+to\s+(?:answer|find|list|show|compute|determine)\s+(?:the\s+)?(?:question|user)"
    r")\b",
    re.I,
)


def strip_question_restatements(text: str, *, min_keep: int = 40) -> str:
    """Remove leading sentence(s) that only restate the prompt, preserving action-specific logic."""
    sentences = split_sentences(normalize_reasoning_text(text))
    while len(sentences) > 1 and QUESTION_RESTATEMENT.search(sentences[0]):
        candidate = " ".join(sentences[1:]).strip()
        if len(candidate) < min_keep:
            break
        sentences = sentences[1:]
    return " ".join(sentences).strip() or text


def prepare_think(text: str, *, source: str, limit: int = 360) -> str:
    text = brief_reasoning(text, limit=limit * 2)
    if source in {"tagged", "pre_tool_text", "reasoning_content"}:
        text = strip_question_restatements(text)
    return brief_reasoning(text, limit=limit)


def recover_think(
    raw_text: str,
    parsed_think: str,
    tool: str,
    args: dict,
    reasoning_content: str = "",
) -> tuple[str, str]:
    """Use the model's own pre-tool prose when it omitted <think> tags."""
    if parsed_think and parsed_think.strip():
        return prepare_think(parsed_think, source="tagged"), "tagged"
    prefix = raw_text.split("<tool_call>", 1)[0]
    prefix = re.sub(r"</?think>", "", prefix, flags=re.I).strip()
    prefix = re.sub(r"^```(?:json)?|```$", "", prefix, flags=re.I | re.M).strip()
    if prefix:
        return prepare_think(prefix, source="pre_tool_text"), "pre_tool_text"
    if tool == "answer_from_context" and isinstance(args, dict) and str(args.get("reason", "")).strip():
        return prepare_think(str(args["reason"]), source="answer_reason"), "answer_reason"
    if reasoning_content and reasoning_content.strip():
        return prepare_think(reasoning_content, source="reasoning_content"), "reasoning_content"
    return (
        f"I will call {tool} because it is the next necessary operation for this question.",
        "fallback_template",
    )


def execute_native_tool_bundle(
    *,
    h: Harness,
    ctx: dict,
    calls: list[dict],
    reasoning: str,
    model_turn_index: int,
    primitive_count: int,
    created: set[str],
    steps: list[dict],
    error_events: list[dict],
    error_counts: collections.Counter,
    plan_tracker: ResidentPlanPolicyTracker,
    table_output_rows: int,
    database_context_profile: str,
    ex: dict,
    denotation_comparison: str,
    gold_sql: str,
    model_arg_schema: dict[str, tuple[set[str], set[str]]] | None = None,
    model_action_validator: Callable[[str, dict], None] = validate_model_action,
) -> dict:
    """Validate against one pre-state, then execute a native call bundle in order."""
    active_schema = MODEL_ARG_SCHEMA if model_arg_schema is None else model_arg_schema
    if not 1 <= len(calls) <= MAX_NATIVE_BUNDLE_CALLS:
        raise ProtocolError(
            f"native bundle must contain 1..{MAX_NATIVE_BUNDLE_CALLS} calls"
        )
    terminal_calls = [
        call for call in calls if call.get("tool") == "answer_from_context"
    ]
    terminal_mixed = bool(terminal_calls) and len(calls) != 1
    prevalidation_errors: dict[str, Exception] = {}
    seen_signatures: set[str] = set()
    for call in calls:
        call_id = call["id"]
        tool = call["tool"]
        arguments = call["arguments"]
        try:
            if terminal_mixed:
                raise ProtocolError(
                    "answer_from_context must be the sole call in a native bundle"
                )
            if tool not in active_schema:
                raise ProtocolError(
                    f"unknown tool {tool!r}; legal tools: {sorted(active_schema)}",
                    code="unknown_tool",
                    details={"legal_tools": sorted(active_schema)},
                    attempted_tool=tool,
                    attempted_arguments=deepcopy(arguments),
                )
            signature = compact_json({"tool": tool, "arguments": arguments})
            if signature in seen_signatures:
                raise ProtocolError(
                    "a native bundle cannot repeat an identical primitive call"
                )
            seen_signatures.add(signature)
            model_action_validator(tool, arguments)
            validate_profile_tool(database_context_profile, tool, arguments)
            plan_tracker.validate_before_execution(tool, arguments)
            # This check is deliberately completed for every call before any call executes.
            validate_tool_arguments_against_state(h, tool, arguments)
        except Exception as exc:  # noqa: BLE001 - returned as per-call feedback
            prevalidation_errors[call_id] = exc

    results: list[dict] = []
    tool_messages: list[dict] = []
    last_error: dict | None = None
    added_errors = 0
    successful_tools = 0
    terminal: dict | None = None
    nonrecoverable = False
    for call in calls:
        call_id = call["id"]
        tool = call["tool"]
        arguments = deepcopy(call["arguments"])
        if nonrecoverable:
            state = ctx["environment"].snapshot()
            blocked_content = compact_json({
                "error": {
                    "type": "blocked",
                    "message": "not executed after a prior nonrecoverable bundle error",
                }
            })
            results.append({
                "call_id": call_id,
                "step_id": None,
                "tool": tool,
                "arguments": arguments,
                "status": "blocked",
                "environment_state_before": state,
                "environment_state": state,
            })
            tool_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": blocked_content,
            })
            continue

        primitive_count += 1
        step_id = f"step_{primitive_count}"
        state_before = ctx["environment"].snapshot()
        try:
            if call_id in prevalidation_errors:
                raise prevalidation_errors[call_id]
            if tool == "answer_from_context":
                correct, pred_sample, gold_sample = score(
                    h,
                    gold_sql,
                    arguments,
                    created,
                    denotation_comparison=denotation_comparison,
                )
                output = {
                    "correct": correct,
                    "pred_sample": pred_sample,
                    "gold_sample": gold_sample,
                }
                terminal = output
                step_record = {
                    "step_id": step_id,
                    "model_turn_index": model_turn_index,
                    "native_tool_call_id": call_id,
                    "think": reasoning,
                    "think_source": "provider_bundle_shared",
                    "tool_call": {"tool": tool, "arguments": arguments},
                    "tool_output": {"terminal_scoring": output},
                    "environment_state_before": state_before,
                    "environment_state": ctx["environment"].snapshot(),
                    "last_tool_error_before": last_error,
                }
                apply_empty_result_target_annotation(step_record)
                steps.append(step_record)
                result = {
                    "call_id": call_id,
                    "step_id": step_id,
                    "tool": tool,
                    "arguments": arguments,
                    "status": "success",
                    "output": output,
                    "environment_state_before": state_before,
                    "environment_state": ctx["environment"].snapshot(),
                }
                tool_content = compact_json({
                    "step_id": step_id,
                    "terminal_ready": True,
                })
            else:
                output, table_name = execute_tool(
                    h,
                    tool,
                    arguments,
                    ctx,
                    step_id,
                    table_output_rows=table_output_rows,
                )
                output, enrichment_audit = enrich_catalog_perception_output(
                    database_context_profile,
                    ex,
                    tool,
                    arguments,
                    output,
                )
                if enrichment_audit is not None:
                    raise RuntimeError(
                        "native-tool-bundle catalog-v1 unexpectedly produced perception enrichment"
                    )
                if table_name:
                    created.add(table_name)
                plan_tracker.record_success(tool, arguments)
                successful_tools += 1
                step_record = {
                    "step_id": step_id,
                    "model_turn_index": model_turn_index,
                    "native_tool_call_id": call_id,
                    "think": reasoning,
                    "think_source": "provider_bundle_shared",
                    "tool_call": {"tool": tool, "arguments": arguments},
                    "tool_output": output,
                    "environment_state_before": state_before,
                    "environment_state": ctx["environment"].snapshot(),
                    "last_tool_error_before": last_error,
                }
                apply_empty_result_target_annotation(step_record)
                steps.append(step_record)
                result = {
                    "call_id": call_id,
                    "step_id": step_id,
                    "tool": tool,
                    "arguments": arguments,
                    "status": "success",
                    "output": output,
                    "table": table_name,
                    "environment_state_before": state_before,
                    "environment_state": ctx["environment"].snapshot(),
                }
                tool_content = tool_output_message(step_id, output)
                last_error = None
        except Exception as exc:  # noqa: BLE001
            added_errors += 1
            state_after = ctx["environment"].snapshot()
            error_type = (
                protocol_failure_type(exc)
                if isinstance(exc, ProtocolError)
                else "execution_error"
            )
            if compact_json(state_after) != compact_json(state_before):
                error_type = "nonrecoverable_execution_error"
                nonrecoverable = True
            message = format_tool_error(exc, h, tool, arguments)
            error_counts[error_type] += 1
            event = error_event(
                primitive_count,
                error_type,
                message,
                state_before,
                state_after,
                tool,
                arguments,
                getattr(exc, "code", type(exc).__name__),
                getattr(exc, "details", None),
            )
            event["model_turn_index"] = model_turn_index
            event["native_tool_call_id"] = call_id
            error_events.append(event)
            tool_content = tool_error_message(
                step_id,
                error_type,
                message,
                error_code=getattr(exc, "code", type(exc).__name__),
                details=getattr(exc, "details", None),
                attempted_tool=tool,
                attempted_arguments=arguments,
            )
            last_error = json.loads(tool_content)
            result = {
                "call_id": call_id,
                "step_id": step_id,
                "tool": tool,
                "arguments": arguments,
                "status": "error",
                "error": last_error["error"],
                "environment_state_before": state_before,
                "environment_state": state_after,
            }
        results.append(result)
        tool_messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": tool_content,
        })

    return {
        "primitive_count": primitive_count,
        "added_errors": added_errors,
        "successful_tools": successful_tools,
        "results": results,
        "tool_messages": tool_messages,
        "last_error": last_error,
        "terminal": terminal,
        "nonrecoverable": nonrecoverable,
    }


def run_rollout(
    *,
    example_index: int,
    ex: dict,
    split: str,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    max_steps: int,
    max_tokens: int,
    api_retries: int,
    api_timeout: int,
    max_errors_per_type: int,
    table_output_rows: int,
    context_mode: str,
    history_turns: int,
    rolling_prompt_variant: str,
    prompt_audit: dict[str, str] | None = None,
    policy_prompt_variant: str = POLICY_PROMPT_CANONICAL,
    plan_policy: str = PLAN_POLICY_OPTIONAL,
    deepseek_carrier: str = DEEPSEEK_CARRIER_JSON_OUTPUT,
    denotation_comparison: str = "bird-set",
    diagnostic_only: bool = False,
    database_context_profile: str = CATALOG_CONTEXT_PROFILE,
    schema_value_count: int = 2,
    schema_metadata_json: str | None = None,
    atomic_protocol_version: str = DEFAULT_PROTOCOL_VERSION,
) -> dict:
    if atomic_protocol_version not in ATOMIC_PROTOCOL_VERSIONS:
        raise ValueError(f"unknown atomic protocol version {atomic_protocol_version!r}")
    version50 = atomic_protocol_version == VERSION50_PROTOCOL_VERSION
    version51 = atomic_protocol_version == VERSION51_PROTOCOL_VERSION
    version52 = atomic_protocol_version == VERSION52_PROTOCOL_VERSION
    version53 = atomic_protocol_version == VERSION53_PROTOCOL_VERSION
    version54 = atomic_protocol_version == VERSION54_PROTOCOL_VERSION
    native_bundle_protocol = version51 or version52 or version53 or version54
    native_model_arg_schema = (
        VERSION54_MODEL_ARG_SCHEMA if version54 else MODEL_ARG_SCHEMA
    )
    native_model_action_validator = (
        validate_model_action_version54 if version54 else validate_model_action
    )
    native_tool_calls = deepseek_carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS
    native_tool_bundle = deepseek_carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE
    if native_bundle_protocol:
        version_label = atomic_protocol_version
        if not diagnostic_only:
            raise ValueError(f"{version_label} is diagnostic-only pending a paired scale gate")
        if not native_tool_bundle:
            raise ValueError(f"{version_label} requires DeepSeek native tool bundles")
        if context_mode != "rolling-legal-history" or history_turns != 4:
            raise ValueError(f"{version_label} requires four provider-native history turns")
        if rolling_prompt_variant != "full":
            raise ValueError(f"{version_label} requires --rolling-prompt-variant full")
        if policy_prompt_variant != POLICY_PROMPT_CANONICAL:
            raise ValueError(f"{version_label} does not combine with policy prompt ablations")
        if plan_policy != PLAN_POLICY_OPTIONAL:
            raise ValueError(f"{version_label} requires the optional plan policy")
        if database_context_profile != CATALOG_CONTEXT_PROFILE:
            raise ValueError(f"{version_label} supports only catalog-v1")
    if version50:
        if not diagnostic_only:
            raise ValueError("version50 is diagnostic-only")
        if not native_tool_calls:
            raise ValueError("version50 requires DeepSeek native tool calls")
        if context_mode != "rolling-legal-history" or history_turns != 4:
            raise ValueError("version50 requires rolling legal history with exactly four turns")
        if rolling_prompt_variant != "full":
            raise ValueError("version50 requires the full rolling prompt")
        if policy_prompt_variant != POLICY_PROMPT_CANONICAL:
            raise ValueError("version50 does not combine with policy prompt ablations")
        if plan_policy != PLAN_POLICY_OPTIONAL:
            raise ValueError("version50 requires the optional plan policy")
        if database_context_profile != CATALOG_CONTEXT_PROFILE:
            raise ValueError("version50 supports only catalog-v1")
    elif native_tool_calls:
        raise ValueError("DeepSeek single native tool calls are isolated to version50")
    if native_tool_bundle and not native_bundle_protocol:
        raise ValueError(
            "DeepSeek native tool bundles are isolated to version51-version54"
        )
    version40 = atomic_protocol_version == VERSION40_PROTOCOL_VERSION
    version41 = atomic_protocol_version == VERSION41_PROTOCOL_VERSION
    version42 = atomic_protocol_version == VERSION42_PROTOCOL_VERSION
    version43 = atomic_protocol_version == VERSION43_PROTOCOL_VERSION
    version44 = atomic_protocol_version == VERSION44_PROTOCOL_VERSION
    version45 = atomic_protocol_version == VERSION45_PROTOCOL_VERSION
    context_ablation = atomic_protocol_version in CONTEXT_ABLATION_PROTOCOL_VERSIONS
    full_reasoning_no_plan = (
        atomic_protocol_version in FULL_REASONING_NO_PLAN_PROTOCOL_VERSIONS
    )
    if full_reasoning_no_plan:
        version_label = atomic_protocol_version
        if not diagnostic_only:
            raise ValueError(f"{version_label} is diagnostic-only")
        if context_mode != "rolling-legal-history" or history_turns != 4:
            raise ValueError(
                f"{version_label} requires rolling legal history with exactly four turns"
            )
        if rolling_prompt_variant != "full":
            raise ValueError(f"{version_label} uses one fixed full prompt")
        if policy_prompt_variant != POLICY_PROMPT_CANONICAL:
            raise ValueError(
                f"{version_label} does not combine with policy prompt ablations"
            )
        if plan_policy != PLAN_POLICY_OPTIONAL:
            raise ValueError(
                f"{version_label} removes plan and cannot use a required plan policy"
            )
        if database_context_profile != CATALOG_CONTEXT_PROFILE:
            raise ValueError(
                f"{version_label} currently supports only the catalog-v1 context profile"
            )
    if version44 or version45:
        version_label = atomic_protocol_version
        if not diagnostic_only:
            raise ValueError(f"{version_label} is diagnostic-only")
        if context_mode != "rolling-legal-history" or history_turns != 4:
            raise ValueError(
                f"{version_label} requires rolling legal history with exactly four turns"
            )
        if rolling_prompt_variant != "full":
            raise ValueError(f"{version_label} uses the full version39-style prompt")
        if policy_prompt_variant != POLICY_PROMPT_CANONICAL:
            raise ValueError(f"{version_label} does not combine with policy prompt ablations")
        if plan_policy != PLAN_POLICY_OPTIONAL:
            raise ValueError(
                f"{version_label} removes plan and cannot use a required plan policy"
            )
        if database_context_profile != CATALOG_BIRD_INSPECT_SEMANTICS_PROFILE:
            raise ValueError(
                f"{version_label} requires catalog-bird-semantics-inspect-only-v1"
            )
    if context_ablation:
        version_label = atomic_protocol_version
        if not diagnostic_only:
            raise ValueError(f"{version_label} is diagnostic-only")
        if context_mode != "rolling-legal-history" or history_turns != 4:
            raise ValueError(
                f"{version_label} requires rolling legal history with exactly four turns"
            )
        if rolling_prompt_variant != "full":
            raise ValueError(f"{version_label} uses the full version39 prompt")
        if policy_prompt_variant != POLICY_PROMPT_CANONICAL:
            raise ValueError(
                f"{version_label} does not combine with policy prompt ablations"
            )
        if plan_policy != PLAN_POLICY_OPTIONAL:
            raise ValueError(f"{version_label} requires the optional plan policy")
        if database_context_profile != CATALOG_CONTEXT_PROFILE:
            raise ValueError(f"{version_label} supports only catalog-v1")
    effective_plan_policy = (
        "disabled-by-protocol"
        if atomic_protocol_version in NO_PLAN_PROTOCOL_VERSIONS
        else plan_policy
    )
    task_path = task_db_path(ex)
    gold_sql = task_gold_sql(ex)
    if not gold_sql:
        raise ValueError(f"task has no gold SQL: {ex.get('db_id')} / {ex.get('question')}")
    h = Harness(task_path)
    execution_catalog = overview(h)
    if database_context_profile in FULL_BIRD_CONTEXT_PROFILES:
        dataset_overview, database_context_audit = build_full_bird_database_context(
            h,
            ex,
            execution_catalog,
            value_count=schema_value_count,
            schema_metadata_json=schema_metadata_json,
            profile=database_context_profile,
        )
    elif database_context_profile in CATALOG_CONTEXT_PROFILES:
        dataset_overview = execution_catalog
        database_context_audit = {
            "context_profile": database_context_profile,
            "schema_value_count": None,
            "disabled_model_tools": [],
            "model_visible_tool_schema_sha256": model_visible_tool_schema_hash(
                database_context_profile
            ),
        }
        if database_context_profile == CATALOG_BIRD_INSPECT_SEMANTICS_PROFILE:
            database_context_audit.update({
                "perception_enrichment": {
                    "describe_table": "ordinary atomic output; no BIRD annotations",
                    "inspect_column": (
                        "BIRD column_name as semantic_name plus column_description; "
                        "live value-domain fields unchanged"
                    ),
                    "canonical_state_mutated": False,
                },
                "perception_enrichment_events": [],
            })
    else:
        raise ValueError(
            f"unknown database context profile: {database_context_profile!r}"
        )
    disabled_model_tools = set(disabled_tools_for_profile(database_context_profile))
    if version54:
        disabled_model_tools.add("plan")
    if atomic_protocol_version in NO_PLAN_PROTOCOL_VERSIONS:
        database_context_audit["model_visible_tool_schema_sha256"] = (
            selected_tool_schema_hash(atomic_protocol_version)
        )
    external_knowledge = ex.get("external_knowledge") or None
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": first_user_message(dataset_overview, ex["question"])},
    ]
    created: set[str] = set()
    # Model-visible full schema is an input ablation, not a mutation of canonical resident state.
    # Execution and state tracking therefore retain the ordinary lazy catalog.
    ctx = new_ctx(execution_catalog)
    if version45:
        ctx["search_values_mode"] = "bounded-v1"
    last_error: dict | None = None
    turns: list[dict] = []
    steps: list[dict] = []
    errors = action_count = primitive_count = 0
    error_counts = collections.Counter()
    error_events: list[dict] = []
    legal_history: list[dict] = []
    native_assistant_history: list[tuple[str, dict]] = []
    native_bundle_history: list[dict] = []
    reasoning_history: list[dict] = []
    adjacent_guard = AdjacentActionGuard()
    plan_tracker = ResidentPlanPolicyTracker(
        PLAN_POLICY_OPTIONAL
        if atomic_protocol_version in NO_PLAN_PROTOCOL_VERSIONS
        else plan_policy
    )
    successful_tool_steps = 0
    usage = collections.Counter()
    started = time.time()
    resident_state_profile, latest_observation_full = selected_resident_context(
        atomic_protocol_version
    )
    rec = {
        "tool_scheme": (
            NATIVE_TOOL_BUNDLE_SCHEME if native_bundle_protocol else ATOMIC_TOOL_SCHEME
        ),
        "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
        "assistant_carrier": (
            VERSION51_PROVIDER_ASSISTANT_CARRIER
            if native_bundle_protocol
            else ATOMIC_ASSISTANT_CARRIER
        ),
        "provider_assistant_carrier": (
            VERSION51_PROVIDER_ASSISTANT_CARRIER
            if native_bundle_protocol
            else (VERSION50_PROVIDER_ASSISTANT_CARRIER if version50 else None)
        ),
        "protocol_version": atomic_protocol_version,
        "protocol_hash": selected_protocol_hash(
            atomic_protocol_version,
            system_prompt,
        ),
        "resident_state_profile": resident_state_profile,
        "latest_observation_full": latest_observation_full,
        "active_relation_policy": (
            "recent-4-references-plus-dependency-closure-v1"
            if atomic_protocol_version == VERSION49_PROTOCOL_VERSION
            else None
        ),
        "value_search_policy": (
            "bounded-sql-candidate-v1"
            if version45
            else ("exhaustive-distinct-v1" if version44 else None)
        ),
        "example_index": example_index,
        "trajectory_id": trajectory_id(split, example_index, ex),
        "db_id": ex["db_id"],
        "question": ex["question"],
        "gold_sql": gold_sql,
        "difficulty": (ex.get("metadata") or {}).get("difficulty_proxy"),
        "correct": False,
        "legal": False,
        "steps": 0,
        "model_turns": 0,
        "primitive_actions": 0,
        "errors": 0,
        "failure_type": None,
        "fail": None,
        "turns": turns,
        "error_events": error_events,
        "outcome": None,
        "plan_policy": effective_plan_policy,
        "history_reasoning": (
            "provider-native-per-model-turn"
            if native_bundle_protocol
            else (
                "complete-all-successful-and-rejected"
                if full_reasoning_no_plan
                else "omitted-from-provider-history"
            )
        ),
        "reasoning_history_events": reasoning_history,
        "history_observations": (
            "one-tool-result-per-call-recent-4-provider-turns"
            if native_bundle_protocol
            else selected_history_observation_policy(
                atomic_protocol_version,
                full_reasoning_no_plan=full_reasoning_no_plan,
            )
        ),
        "terminal_evidence_policy": (
            "explicit-columns-unique-bare-deterministic-project-v2"
            if version43
            else (
                "explicit-columns-deterministic-project-v1"
                if version42
                else "exact-table"
            )
        ),
        "policy_prompt_variant": policy_prompt_variant,
        "denotation_comparison": denotation_comparison,
        "training_admission": training_admission(diagnostic_only=diagnostic_only),
        "database_context_profile": database_context_profile,
        "database_context_audit": database_context_audit,
        "disabled_model_tools": sorted(disabled_model_tools),
        "sft_export_eligible": sft_export_eligible(
            context_mode=context_mode,
            history_turns=history_turns,
            rolling_prompt_variant=rolling_prompt_variant,
            denotation_comparison=denotation_comparison,
            diagnostic_only=diagnostic_only,
        ),
    }

    while action_count < max_steps:
        action_count += 1
        state_before = ctx["environment"].snapshot()
        if native_bundle_protocol:
            model_input = native_bundle_history_messages(
                system_prompt=system_prompt,
                overview=dataset_overview,
                question=ex["question"],
                state=state_before,
                last_error=last_error,
                external_knowledge=external_knowledge,
                history=native_bundle_history,
                history_turns=history_turns,
            )
        elif context_mode == "rolling-legal-history":
            model_input = rolling_legal_history_messages(
                system_prompt,
                dataset_overview,
                ex["question"],
                state_before,
                last_error,
                external_knowledge,
                legal_history,
                history_turns,
                compact_observations=not full_reasoning_no_plan,
                preserve_all_reasoning=full_reasoning_no_plan,
                reasoning_history=(
                    reasoning_history if full_reasoning_no_plan else None
                ),
                resident_state_profile=resident_state_profile,
                latest_observation_full=latest_observation_full,
            )
        else:
            model_input = model_context_messages(
                system_prompt,
                dataset_overview,
                ex["question"],
                state_before,
                last_error,
                external_knowledge,
            )
        if not native_bundle_protocol:
            model_input = provider_request_messages(
                model,
                model_input,
                carrier=deepseek_carrier,
                preserve_reasoning=full_reasoning_no_plan,
                native_assistant_history=native_assistant_history,
            )
        turn = {
            "turn_index": len(turns),
            "model_input": deepcopy(model_input),
            "provider_request_options": provider_request_audit_options(
                model,
                carrier=deepseek_carrier,
                native_model_arg_schema=(
                    native_model_arg_schema if native_bundle_protocol else None
                ),
            ),
            "deepseek_carrier": deepseek_carrier,
        }
        try:
            text, call_usage, reasoning_content = chat_with_retries(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=model_input,
                max_tokens=max_tokens,
                timeout=api_timeout,
                retries=api_retries,
                deepseek_carrier=deepseek_carrier,
                native_model_arg_schema=(
                    native_model_arg_schema if native_bundle_protocol else None
                ),
            )
            add_usage(usage, call_usage)
            turn["api_finish_reason"] = call_usage.get("api_finish_reason")
            if call_usage.get("provider_response_metadata"):
                turn["provider_response_metadata"] = deepcopy(
                    call_usage["provider_response_metadata"]
                )
            if call_usage.get("api_retry_events"):
                turn["provider_retry_events"] = deepcopy(
                    call_usage["api_retry_events"]
                )
        except ProviderCarrierError as exc:
            add_usage(usage, exc.usage)
            rec.update({
                "failure_type": "provider_carrier_error",
                "fail": f"api: {type(exc).__name__}: {exc}",
                "steps": action_count - 1,
                "errors": errors,
            })
            turn["api_error"] = rec["fail"]
            turn["api_error_type"] = "provider_carrier_error"
            if exc.usage.get("api_retry_events"):
                turn["provider_retry_events"] = deepcopy(
                    exc.usage["api_retry_events"]
                )
            turns.append(turn)
            break
        except ContextOverflowError as exc:
            rec.update({
                "failure_type": "context_overflow",
                "fail": f"api: {type(exc).__name__}: {exc}",
                "steps": action_count - 1,
                "errors": errors,
            })
            turn["api_error"] = rec["fail"]
            turn["api_error_type"] = "context_overflow"
            turns.append(turn)
            break
        except Exception as exc:  # noqa: BLE001
            rec.update({
                "failure_type": "api_error",
                "fail": f"api: {type(exc).__name__}: {exc}",
                "steps": action_count - 1,
                "errors": errors,
            })
            turn["api_error"] = rec["fail"]
            turn["api_error_type"] = "api_error"
            turns.append(turn)
            break

        if native_bundle_protocol:
            native_assistant_message = call_usage.get(
                "provider_native_assistant_message"
            )
            native_rejection = call_usage.get("provider_native_rejection_reason")
            _, adapter_record = adapt_provider_response(
                model,
                text,
                reasoning_content,
                carrier=deepseek_carrier,
            )
            turn["raw_model_output"] = (
                native_assistant_message.get("content")
                if isinstance(native_assistant_message, dict)
                else None
            )
            turn["provider_native_tool_call"] = deepcopy(
                (native_assistant_message or {}).get("tool_calls")
            )
            turn["provider_reasoning_content"] = reasoning_content
            turn["response_adapter"] = adapter_record
            turn["model_output"] = text
            if native_rejection or not adapter_record.get("eligible"):
                errors += 1
                error_type = "protocol_error"
                error_counts[error_type] += 1
                reason = native_rejection or adapter_record.get("rejection_reason")
                message = (
                    "DeepSeek native tool-bundle transport error: "
                    f"{reason or 'invalid_native_bundle'}"
                )
                state_hash = __import__("hashlib").sha256(
                    compact_json(state_before).encode()
                ).hexdigest()
                event = {
                    "action_index": None,
                    "model_turn_index": action_count,
                    "step_id": None,
                    "error_type": error_type,
                    "message": message,
                    "state_before_hash": state_hash,
                    "state_after_hash": state_hash,
                    "error_code": str(reason or "invalid_native_bundle"),
                }
                error_events.append(event)
                turn["execution_error"] = message
                turn["execution_error_type"] = error_type
                turn["error_event"] = event
                turns.append(turn)
                last_error = {
                    "error": {
                        "type": error_type,
                        "message": message,
                        "code": str(reason or "invalid_native_bundle"),
                    }
                }
                if error_limit_reached(
                    error_type,
                    error_counts[error_type],
                    max_errors_per_type,
                ):
                    rec["failure_type"] = error_type
                    rec["fail"] = (
                        f"aborted after {error_counts[error_type]} "
                        f"{error_type} events: {message}"
                    )
                    break
                continue

            calls = deepcopy(call_usage.get("provider_native_bundle_calls") or [])
            turn["parsed"] = {
                "reasoning": reasoning_content,
                "calls": deepcopy(calls),
            }
            turn["think_source"] = "provider_reasoning_content"
            turn["feedback_recovery"] = bool(last_error)
            turn["recovered_from_error_type"] = (
                (last_error or {}).get("error", {}).get("type")
            )
            bundle = execute_native_tool_bundle(
                h=h,
                ctx=ctx,
                calls=calls,
                reasoning=reasoning_content.strip(),
                model_turn_index=action_count,
                primitive_count=primitive_count,
                created=created,
                steps=steps,
                error_events=error_events,
                error_counts=error_counts,
                plan_tracker=plan_tracker,
                table_output_rows=table_output_rows,
                database_context_profile=database_context_profile,
                ex=ex,
                denotation_comparison=denotation_comparison,
                gold_sql=gold_sql,
                model_arg_schema=native_model_arg_schema,
                model_action_validator=native_model_action_validator,
            )
            primitive_count = bundle["primitive_count"]
            errors += bundle["added_errors"]
            successful_tool_steps += bundle["successful_tools"]
            last_error = bundle["last_error"]
            turn["native_bundle_results"] = deepcopy(bundle["results"])
            turn["tool_output"] = deepcopy(bundle["results"])
            turns.append(turn)
            if isinstance(native_assistant_message, dict):
                native_bundle_history.append({
                    "model_turn_index": action_count,
                    "assistant": deepcopy(native_assistant_message),
                    "tool_messages": deepcopy(bundle["tool_messages"]),
                })
            if bundle["terminal"] is not None:
                terminal = bundle["terminal"]
                rec["legal"] = True
                rec["correct"] = bool(terminal["correct"])
                rec["pred_sample"] = terminal["pred_sample"]
                rec["gold_sample"] = terminal["gold_sample"]
                rec["failure_type"] = None if rec["correct"] else "wrong_answer"
                if rec["correct"]:
                    rec["outcome"] = (
                        "recovered_success" if error_events else "clean_success"
                    )
                break
            if bundle["nonrecoverable"]:
                rec["failure_type"] = "nonrecoverable_execution_error"
                rec["fail"] = "a native bundle call changed state before failing"
                break
            exhausted = [
                error_type
                for error_type, count in error_counts.items()
                if error_limit_reached(error_type, count, max_errors_per_type)
            ]
            if exhausted:
                rec["failure_type"] = sorted(exhausted)[0]
                rec["fail"] = (
                    "aborted after native bundle error budget was exhausted: "
                    + ", ".join(
                        f"{name}={error_counts[name]}" for name in sorted(exhausted)
                    )
                )
                break
            continue

        raw_model_output = text
        text, adapter_record = adapt_provider_response(
            model,
            raw_model_output,
            reasoning_content,
            carrier=deepseek_carrier,
        )
        native_rejection = call_usage.get("provider_native_rejection_reason")
        if native_tool_calls and native_rejection_is_semantic(native_rejection):
            adapter_record.update({
                "applied": False,
                "eligible": False,
                "carrier": deepseek_carrier,
                "name": VERSION50_PROVIDER_ASSISTANT_CARRIER,
                "rejection_reason": native_rejection,
            })
        native_assistant_message = call_usage.get("provider_native_assistant_message")
        if native_tool_calls:
            turn["raw_model_output"] = (
                native_assistant_message.get("content")
                if isinstance(native_assistant_message, dict)
                else None
            )
            turn["provider_native_tool_call"] = deepcopy(
                (native_assistant_message or {}).get("tool_calls")
            )
            if isinstance(native_assistant_message, dict):
                native_assistant_history.append((text, deepcopy(native_assistant_message)))
        else:
            turn["raw_model_output"] = raw_model_output
        turn["response_adapter"] = adapter_record
        turn["model_output"] = text
        if reasoning_content:
            turn["provider_reasoning_content"] = reasoning_content
        messages.append({"role": "assistant", "content": text})
        try:
            rejection = provider_rejection_message(adapter_record)
            if rejection:
                adjacent_guard.clear()
                raise ProtocolError(rejection)
            step_id = f"step_{action_count}"
            if version43:
                parser = parse_assistant_strict_version43
            elif version42:
                parser = parse_assistant_strict_version42
            elif version41:
                parser = parse_assistant_strict_version41
            elif version40:
                parser = parse_assistant_strict_version40
            elif version45:
                parser = parse_assistant_strict_version45
            elif version44:
                parser = parse_assistant_strict_version44
            else:
                parser = parse_assistant_strict
            think, tool, args = parser(
                text,
                adjacent_guard=adjacent_guard,
                step_id=step_id,
            )
            validate_profile_tool(database_context_profile, tool, args)
            think_source = "model"
            turn["parsed"] = {"think": think, "tool": tool, "arguments": args}
            turn["think_source"] = think_source
            turn["feedback_recovery"] = bool(last_error)
            turn["recovered_from_error_type"] = (last_error or {}).get("error", {}).get("type")
            plan_tracker.validate_before_execution(tool, args)
            if tool == "answer_from_context":
                validate_tool_arguments_against_state(h, tool, args)
                score_args = args
                terminal_projection = None
                if version43:
                    score_args, terminal_projection = (
                        lower_terminal_evidence_version43(h, args)
                    )
                    turn["terminal_projection"] = terminal_projection
                elif version42:
                    score_args, terminal_projection = (
                        lower_terminal_evidence_version42(h, args)
                    )
                    turn["terminal_projection"] = terminal_projection
                if full_reasoning_no_plan:
                    reasoning_history.append({
                        "step_id": step_id,
                        "status": "success",
                        "reasoning": think,
                    })
                rec["legal"] = True
                rec["steps"] = action_count
                rec["errors"] = errors
                correct, pred_sample, gold_sample = score(
                    h,
                    gold_sql,
                    score_args,
                    created,
                    denotation_comparison=denotation_comparison,
                )
                rec["correct"] = correct
                rec["pred_sample"] = pred_sample
                rec["gold_sample"] = gold_sample
                if not correct:
                    rec["failure_type"] = "wrong_answer"
                else:
                    rec["outcome"] = "recovered_success" if error_events else "clean_success"
                turns.append(turn)
                steps.append({
                    "step_id": f"step_{action_count}",
                    "think": think,
                    "think_source": think_source,
                    "tool_call": {"tool": tool, "arguments": args},
                    "tool_output": {
                        "final_answer": args.get("answer"),
                        "terminal_projection": terminal_projection,
                    },
                    "environment_state_before": state_before,
                    "environment_state": ctx["environment"].snapshot(),
                    "last_tool_error_before": last_error,
                    "feedback_recovery": bool(last_error),
                    "recovered_from_error_type": (last_error or {}).get("error", {}).get("type"),
                })
                break

            out, table_name = execute_tool(
                h,
                tool,
                args,
                ctx,
                step_id,
                table_output_rows=table_output_rows,
            )
            out, enrichment_audit = enrich_catalog_perception_output(
                database_context_profile,
                ex,
                tool,
                args,
                out,
            )
            if enrichment_audit is not None:
                database_context_audit.setdefault(
                    "perception_enrichment_events",
                    [],
                ).append({
                    "step_id": step_id,
                    **enrichment_audit,
                })
            turn["tool_output"] = out
            step_record = {
                "step_id": step_id,
                "think": think,
                "think_source": think_source,
                "tool_call": {"tool": tool, "arguments": args},
                "tool_output": out,
                "environment_state_before": state_before,
                "environment_state": ctx["environment"].snapshot(),
                "last_tool_error_before": last_error,
                "feedback_recovery": bool(last_error),
                "recovered_from_error_type": (last_error or {}).get("error", {}).get("type"),
            }
            apply_empty_result_target_annotation(step_record)
            steps.append(step_record)
            turns.append(turn)
            adjacent_guard.mark_last("success")
            last_error = None
            successful_tool_steps += 1
            plan_tracker.record_success(tool, args)
            if full_reasoning_no_plan:
                reasoning_history.append({
                    "step_id": step_id,
                    "status": "success",
                    "reasoning": think,
                })
            if retain_in_rolling_history(tool, plan_policy):
                legal_history.append({
                    "step_id": step_id,
                    "assistant": text,
                    "observation": tool_output_message(step_id, out),
                })
            if table_name:
                created.add(table_name)
            messages.append({"role": "user", "content": tool_output_message(step_id, out)})
        except (ProtocolError, Exception) as exc:  # noqa: BLE001
            errors += 1
            parsed = turn.get("parsed") or {}
            attempted_tool = parsed.get("tool") or getattr(exc, "attempted_tool", None)
            attempted_arguments = (
                parsed.get("arguments")
                if parsed.get("tool")
                else getattr(exc, "attempted_arguments", None)
            )
            state_after = ctx["environment"].snapshot()
            error_type = protocol_failure_type(exc) if isinstance(exc, ProtocolError) else "execution_error"
            adjacent_guard.mark_last("rejected")
            if error_type == "execution_error" and compact_json(state_after) != compact_json(state_before):
                error_type = "nonrecoverable_execution_error"
            message = format_tool_error(exc, h, attempted_tool, attempted_arguments)
            turn["execution_error"] = message
            turn["execution_error_type"] = error_type
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
            turn["error_event"] = event
            error_events.append(event)
            turns.append(turn)

            step_id = f"step_{action_count}"
            rejected_reasoning = (
                parsed.get("think")
                or turn.get("provider_reasoning_content")
            )
            if (
                full_reasoning_no_plan
                and isinstance(rejected_reasoning, str)
                and rejected_reasoning.strip()
            ):
                reasoning_history.append({
                    "step_id": step_id,
                    "status": "rejected",
                    "reasoning": rejected_reasoning.strip(),
                    "error_type": error_type,
                    "error_code": str(getattr(exc, "code", type(exc).__name__)),
                })
            if error_type == "nonrecoverable_execution_error":
                rec.update({
                    "failure_type": error_type,
                    "fail": message,
                    "steps": action_count,
                    "errors": errors,
                })
                break
            error_counts[error_type] += 1
            if error_limit_reached(
                error_type,
                error_counts[error_type],
                max_errors_per_type,
            ):
                rec.update({
                    "failure_type": error_type,
                    "fail": f"aborted after {error_counts[error_type]} {error_type} events: {message}",
                    "steps": action_count,
                    "errors": errors,
                })
                break
            error_observation = tool_error_message(
                step_id,
                error_type,
                message,
                error_code=getattr(exc, "code", type(exc).__name__),
                details=getattr(exc, "details", None),
                attempted_tool=attempted_tool,
                attempted_arguments=attempted_arguments,
            )
            last_error = json.loads(error_observation)
            adjacent_guard.mark_last("rejected", last_error["error"])
            messages.append({"role": "user", "content": error_observation})
            continue
    else:
        rec.update({
            "failure_type": "max_steps",
            "fail": "max_steps",
            "steps": primitive_count if native_bundle_protocol else action_count,
            "errors": errors,
        })

    rec["steps"] = primitive_count if native_bundle_protocol else rec.get("steps", action_count)
    rec["model_turns"] = action_count
    rec["primitive_actions"] = primitive_count if native_bundle_protocol else action_count
    rec["errors"] = errors
    rec["training_result_quality"] = training_quality_summary(steps)
    if rec["training_result_quality"]["empty_terminal_evidence"]:
        rec["sft_export_eligible"] = False
    if native_bundle_protocol:
        rec["provider_native_history"] = native_bundle_history
        rec["native_bundle_rl_statistics"] = analyze_native_bundle_credit(rec)
    rec["final_messages"] = messages
    rec["elapsed_seconds"] = round(time.time() - started, 3)
    rec["usage"] = dict(usage)
    if rec["correct"]:
        adapter_turns = [turn.get("response_adapter") or {} for turn in turns]
        rec["trajectory"] = {
            "tool_scheme": (
                NATIVE_TOOL_BUNDLE_SCHEME if native_bundle_protocol else ATOMIC_TOOL_SCHEME
            ),
            "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
            "assistant_carrier": (
                VERSION51_PROVIDER_ASSISTANT_CARRIER
                if native_bundle_protocol
                else ATOMIC_ASSISTANT_CARRIER
            ),
            "provider_assistant_carrier": (
                VERSION51_PROVIDER_ASSISTANT_CARRIER
                if native_bundle_protocol
                else (VERSION50_PROVIDER_ASSISTANT_CARRIER if version50 else None)
            ),
            "protocol_version": atomic_protocol_version,
            "protocol_hash": selected_protocol_hash(
                atomic_protocol_version,
                system_prompt,
            ),
            "trajectory_id": rec["trajectory_id"],
            "schema_version": (
                (
                    "v8-native-tool-bundle-no-plan"
                    if version54
                    else (
                        "v7-native-tool-bundle-reviewed-prompt"
                        if version53
                        else (
                            "v6-native-tool-bundle-compact-prompt"
                            if version52
                            else "v5-native-tool-bundle"
                        )
                    )
                )
                if native_bundle_protocol
                else "v4-external-rollout"
            ),
            "source": {
                "dataset": ex.get("dataset", "spider"),
                "split": ex.get("split", split),
                "example_id": ex.get("example_id"),
                "db_id": ex["db_id"],
                "db_path": task_path,
                "external_knowledge": external_knowledge,
                "gold_sql": gold_sql,
            },
            "question": ex["question"],
            "difficulty": rec["difficulty"],
            "label_status": "verified",
            "training_admission": training_admission(
                diagnostic_only=diagnostic_only,
            ),
            "sft_export_eligible": sft_export_eligible(
                context_mode=context_mode,
                history_turns=history_turns,
                rolling_prompt_variant=rolling_prompt_variant,
                denotation_comparison=denotation_comparison,
                diagnostic_only=diagnostic_only,
            ) and not rec["training_result_quality"]["empty_terminal_evidence"],
            "training_result_quality": deepcopy(rec["training_result_quality"]),
            # Canonical replay starts from the ordinary execution catalog. The distinct
            # model-visible full-context payload is independently hashed in database_context_audit.
            "initial_state": {"dataset_overview": execution_catalog},
            "steps": steps,
            "provider_native_history": (
                deepcopy(native_bundle_history) if native_bundle_protocol else None
            ),
            "native_bundle_rl_statistics": (
                deepcopy(rec.get("native_bundle_rl_statistics"))
                if native_bundle_protocol
                else None
            ),
            "rollout_generation": {
                "tool_scheme": (
                    NATIVE_TOOL_BUNDLE_SCHEME if native_bundle_protocol else ATOMIC_TOOL_SCHEME
                ),
                "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
                "assistant_carrier": (
                    VERSION51_PROVIDER_ASSISTANT_CARRIER
                    if native_bundle_protocol
                    else ATOMIC_ASSISTANT_CARRIER
                ),
                "provider_assistant_carrier": (
                    VERSION51_PROVIDER_ASSISTANT_CARRIER
                    if native_bundle_protocol
                    else (VERSION50_PROVIDER_ASSISTANT_CARRIER if version50 else None)
                ),
                "protocol_version": atomic_protocol_version,
                "method": "external_llm_closed_loop",
                "model": model,
                "protocol_hash": selected_protocol_hash(
                    atomic_protocol_version,
                    system_prompt,
                ),
                "prompt_contract": dict(prompt_audit or {}),
                "context_mode": context_mode,
                "history_turns": history_turns,
                "rolling_prompt_variant": rolling_prompt_variant,
                "policy_prompt_variant": policy_prompt_variant,
                "plan_policy": effective_plan_policy,
                "history_reasoning": (
                    "provider-native-per-model-turn"
                    if native_bundle_protocol
                    else (
                        "complete-all-successful-and-rejected"
                        if full_reasoning_no_plan
                        else "omitted-from-provider-history"
                    )
                ),
                "history_observations": (
                    "one-tool-result-per-call-recent-4-provider-turns"
                    if native_bundle_protocol
                    else selected_history_observation_policy(
                        atomic_protocol_version,
                        full_reasoning_no_plan=full_reasoning_no_plan,
                    )
                ),
                "resident_state_profile": resident_state_profile,
                "latest_observation_full": latest_observation_full,
                "active_relation_policy": (
                    "recent-4-references-plus-dependency-closure-v1"
                    if atomic_protocol_version == VERSION49_PROTOCOL_VERSION
                    else None
                ),
                "terminal_evidence_policy": (
                    "explicit-columns-unique-bare-deterministic-project-v2"
                    if version43
                    else (
                        "explicit-columns-deterministic-project-v1"
                        if version42
                        else "exact-table"
                    )
                ),
                "deepseek_carrier": deepseek_carrier,
                "denotation_comparison": denotation_comparison,
                "database_context_profile": database_context_profile,
                "database_context_audit": database_context_audit,
                "disabled_model_tools": sorted(disabled_model_tools),
                "sft_export_eligible": sft_export_eligible(
                    context_mode=context_mode,
                    history_turns=history_turns,
                    rolling_prompt_variant=rolling_prompt_variant,
                    denotation_comparison=denotation_comparison,
                    diagnostic_only=diagnostic_only,
                ) and not rec["training_result_quality"]["empty_terminal_evidence"],
                "empty_result_policy": EMPTY_RESULT_POLICY_VERSION,
                "training_admission": training_admission(
                    diagnostic_only=diagnostic_only,
                ),
                "error_actions_are_sft_targets": False,
                "errors": errors,
                "successful_tool_steps": successful_tool_steps,
                "action_count": action_count,
                "model_turn_count": action_count,
                "primitive_action_count": (
                    primitive_count if native_bundle_protocol else action_count
                ),
                "native_bundle_max_calls": (
                    MAX_NATIVE_BUNDLE_CALLS if native_bundle_protocol else None
                ),
                "native_bundle_prevalidation_state": (
                    "bundle-pre-state" if native_bundle_protocol else None
                ),
                "error_feedback_training_policy": (
                    "preserve-error-turn-and-tool-feedback; later-corrected-bundle-is-target"
                    if native_bundle_protocol
                    else None
                ),
                "error_events": error_events,
                "error_counts": dict(error_counts),
                "provider_adapter": {
                    "applied_actions": sum(bool(adapter.get("applied")) for adapter in adapter_turns),
                    "unusable_actions": sum(
                        adapter.get("name") != "none" and not adapter.get("applied")
                        for adapter in adapter_turns
                    ),
                    "names": sorted({adapter.get("name") for adapter in adapter_turns if adapter.get("name")}),
                },
                "outcome": rec["outcome"],
                "elapsed_seconds": rec["elapsed_seconds"],
                "usage": dict(usage),
            },
        }
    return rec


def read_completed(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    # New pass@k-style generation marks an example complete only after either a
                    # correct attempt or after exhausting attempts_per_example. Older all.jsonl
                    # files did not carry this field; treat those records as completed so --resume
                    # remains backward-compatible.
                    if record.get("example_complete", "legacy") in (True, "legacy"):
                        done.add(record.get("trajectory_id"))
    return done


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def audit_summary(path: Path) -> dict:
    """Aggregate all persisted attempts so a resumed run has an honest manifest."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    final_by_trajectory: dict[str, dict] = {}
    for record in records:
        trajectory_id = str(record.get("trajectory_id"))
        previous = final_by_trajectory.get(trajectory_id)
        # A higher attempt index is a declared whole-episode retry and is eligible for success@k.
        # A duplicate write of the same attempt must never overwrite its first observed outcome.
        if previous is None or int(record.get("attempt_index", 1)) > int(previous.get("attempt_index", 1)):
            final_by_trajectory[trajectory_id] = record
    finals = list(final_by_trajectory.values())
    raw_counts = collections.Counter()
    final_counts = collections.Counter()
    usage_total = collections.Counter()
    for record in records:
        raw_counts["total"] += 1
        raw_counts["correct" if record.get("correct") else "failed"] += 1
        if record.get("legal"):
            raw_counts["legal"] += 1
        if record.get("failure_type"):
            raw_counts[f"failure:{record['failure_type']}"] += 1
        add_usage(usage_total, record.get("usage") or {})
    for record in finals:
        final_counts["examples"] += 1
        final_counts["examples_correct" if record.get("correct") else "examples_failed"] += 1
        if record.get("failure_type"):
            final_counts[f"final_failure:{record['failure_type']}"] += 1
    return {
        "raw_attempt_records": len(records),
        "unique_examples": len(finals),
        "duplicate_attempt_records": len(records) - len(finals),
        "counts": dict(raw_counts + final_counts),
        "usage_total": dict(usage_total),
    }


_NON_SEMANTIC_PROVIDER_FAILURES = {
    "api_error",
    "context_overflow",
    "provider_carrier_error",
}


def is_semantic_failure_for_ordered_stop(record: dict) -> bool:
    """Return whether one final example outcome advances the ordered failure streak.

    Provider/transport failures are audited by their separate acceptance gate. Every other
    incorrect final outcome is a model-policy semantic failure for the ordered pilot stop.
    """
    return bool(
        not record.get("correct")
        and record.get("failure_type") not in _NON_SEMANTIC_PROVIDER_FAILURES
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--examples-file", default="",
                        help="optional JSON from build_rollout_examples.py; overrides --split source")
    parser.add_argument(
        "--indices-file",
        default="",
        help="optional ordered JSON index list applied after loading the source cohort",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", required=True, help="verified trajectory JSONL output")
    parser.add_argument("--failures-out", default="", help="default: OUT.failures.jsonl")
    parser.add_argument("--all-out", default="", help="default: OUT.all.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-consecutive-semantic-failures",
        type=int,
        default=0,
        help=(
            "source-order pilot stop; 0 disables it. With parallel workers, at most the "
            "already-issued bounded window may finish after the stop point and is marked "
            "outside the acceptance gate"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-errors-per-type", type=int,
                        default=DEFAULT_MAX_ERRORS_PER_TYPE,
                        help="terminate only after this many recoverable errors of one class")
    parser.add_argument("--attempts-per-example", type=int, default=1,
                        help="whole-trajectory attempts per example; stop early once verifier-correct")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="provider default when omitted; explicit value always wins")
    parser.add_argument("--api-timeout", type=int, default=300)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--table-output-rows", type=int, default=0)
    parser.add_argument("--context-mode", choices=["state-only", "rolling-legal-history"],
                        default="rolling-legal-history")
    parser.add_argument("--history-turns", type=int, default=4,
                        help=(
                            "number of atomic legal pairs or native provider assistant turns to "
                            "retain; 0 keeps all only for compatible atomic protocols"
                        ))
    parser.add_argument("--rolling-prompt-variant", choices=["full", "compact"], default="full",
                        help="rolling-only prompt ablation; full preserves existing runs")
    parser.add_argument(
        "--atomic-protocol-version",
        choices=ATOMIC_PROTOCOL_VERSIONS,
        default=DEFAULT_PROTOCOL_VERSION,
        help=(
            "version39 preserves the current default; version40 enables the isolated no-plan, "
            "inspect_rows, full-reasoning recent-4 diagnostic; version41 changes only its prompt "
            "by adding one consolidated output contract and Gate50 error corrections; version42 "
            "requires explicit terminal evidence columns and deterministically projects them; "
            "version43 additionally resolves only unique bare terminal-column suffixes; version44 "
            "starts a version39-style checkpoint candidate with no plan, inspect_rows, fuzzy "
            "search_values, and inspect-only BIRD column semantics; version45 changes only that "
            "search to bounded SQL candidate recall before fuzzy ranking; version46 replaces "
            "verbose derivations with Harness handle cards; version47 additionally archives "
            "resident row values while keeping the latest observation full; version48 adds only "
            "brief interpret-before-act teacher guidance; version49 keeps exact rows for the "
            "Harness-derived active dependency closure and archives only inactive relations; "
            "version50 keeps version39 semantics and changes only the DeepSeek provider carrier "
            "to single native function calls; version51 moves the same primitive tools to a "
            "bounded provider-native multi-call turn with bundle-pre-state validation; version52 "
            "keeps that execution surface and replaces duplicated text/schema instructions with "
            "one compact native-schema-authoritative prompt plus soft bundle scheduling rules; "
            "version53 keeps every tool and execution semantic unchanged and applies the reviewed "
            "role-separated authority, evidence, representation, and scheduling clarifications; "
            "version54 removes only the public plan function and its teacher-only prompt sentence"
        ),
    )
    parser.add_argument(
        "--policy-prompt-variant",
        choices=POLICY_PROMPT_VARIANTS,
        default=POLICY_PROMPT_CANONICAL,
        help="auditable policy ablation; canonical preserves the version20 prompt",
    )
    parser.add_argument(
        "--plan-policy",
        choices=PLAN_POLICY_CHOICES,
        default=PLAN_POLICY_OPTIONAL,
        help="optional (default) or force a resident-only initial plan plus later plan update",
    )
    parser.add_argument(
        "--deepseek-carrier",
        choices=DEEPSEEK_CARRIER_CHOICES,
        default=DEEPSEEK_CARRIER_JSON_OUTPUT,
        help=(
            "auditable provider carrier; native-tool-calls is version50 and "
            "native-tool-bundle is version51-version54"
        ),
    )
    add_denotation_comparison_argument(parser)
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help=(
            "force every output and the manifest to remain ineligible for SFT export "
            "while preserving the same causal generation protocol"
        ),
    )
    parser.add_argument(
        "--database-context-profile",
        choices=DATABASE_CONTEXT_PROFILES,
        default=CATALOG_CONTEXT_PROFILE,
        help=(
            "model-visible database context; catalog-bird-semantics-inspect-only-v1 keeps the "
            "lazy catalog and enriches only inspect_column; full-bird-schema-samples-v1 reveals "
            "complete BIRD schema/column semantics/live examples"
        ),
    )
    parser.add_argument(
        "--schema-value-count",
        type=int,
        default=2,
        help="representative distinct live values per source column for full-context diagnostics",
    )
    parser.add_argument(
        "--schema-metadata-json",
        default="",
        help="optional BIRD tables metadata JSON; otherwise resolved from each task database path",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_consecutive_semantic_failures < 0:
        parser.error("--max-consecutive-semantic-failures must be non-negative")
    if args.schema_value_count < 0:
        parser.error("--schema-value-count must be non-negative")
    if (
        args.database_context_profile in FULL_BIRD_CONTEXT_PROFILES
        and not args.diagnostic_only
    ):
        parser.error(
            "full-bird-schema-samples-v1 is an unpromoted diagnostic profile; "
            "pass --diagnostic-only"
        )
    if (
        args.database_context_profile
        == CATALOG_BIRD_INSPECT_SEMANTICS_PROFILE
        and not args.diagnostic_only
    ):
        parser.error(
            "catalog-bird-semantics-inspect-only-v1 is diagnostic-only; "
            "pass --diagnostic-only"
        )
    if (
        args.database_context_profile in FULL_BIRD_CONTEXT_PROFILES
        and args.rolling_prompt_variant != "full"
    ):
        parser.error(
            "full-bird-schema-samples-v1 requires --rolling-prompt-variant full"
        )
    if args.atomic_protocol_version in FULL_REASONING_NO_PLAN_PROTOCOL_VERSIONS:
        version_label = args.atomic_protocol_version
        if not args.diagnostic_only:
            parser.error(f"{version_label} is diagnostic-only; pass --diagnostic-only")
        if args.context_mode != "rolling-legal-history":
            parser.error(
                f"{version_label} requires --context-mode rolling-legal-history"
            )
        if args.history_turns != 4:
            parser.error(f"{version_label} requires --history-turns 4")
        if args.rolling_prompt_variant != "full":
            parser.error(
                f"{version_label} has one fixed layered prompt; "
                "use --rolling-prompt-variant full"
            )
        if args.policy_prompt_variant != POLICY_PROMPT_CANONICAL:
            parser.error(
                f"{version_label} does not combine with policy prompt ablations"
            )
        if args.plan_policy != PLAN_POLICY_OPTIONAL:
            parser.error(
                f"{version_label} removes plan; do not select a required plan policy"
            )
        if args.database_context_profile != CATALOG_CONTEXT_PROFILE:
            parser.error(
                f"{version_label} currently supports only "
                "--database-context-profile catalog-v1"
            )
    if args.atomic_protocol_version == VERSION44_PROTOCOL_VERSION:
        if not args.diagnostic_only:
            parser.error("version44 is diagnostic-only; pass --diagnostic-only")
        if args.context_mode != "rolling-legal-history":
            parser.error("version44 requires --context-mode rolling-legal-history")
        if args.history_turns != 4:
            parser.error("version44 requires --history-turns 4")
        if args.rolling_prompt_variant != "full":
            parser.error(
                "version44 uses the full version39-style prompt; "
                "use --rolling-prompt-variant full"
            )
        if args.policy_prompt_variant != POLICY_PROMPT_CANONICAL:
            parser.error("version44 does not combine with policy prompt ablations")
        if args.plan_policy != PLAN_POLICY_OPTIONAL:
            parser.error("version44 removes plan; do not select a required plan policy")
        if (
            args.database_context_profile
            != CATALOG_BIRD_INSPECT_SEMANTICS_PROFILE
        ):
            parser.error(
                "version44 requires --database-context-profile "
                "catalog-bird-semantics-inspect-only-v1"
            )
    if args.atomic_protocol_version == VERSION45_PROTOCOL_VERSION:
        if not args.diagnostic_only:
            parser.error("version45 is diagnostic-only; pass --diagnostic-only")
        if args.context_mode != "rolling-legal-history":
            parser.error("version45 requires --context-mode rolling-legal-history")
        if args.history_turns != 4:
            parser.error("version45 requires --history-turns 4")
        if args.rolling_prompt_variant != "full":
            parser.error(
                "version45 uses the full version39-style prompt; "
                "use --rolling-prompt-variant full"
            )
        if args.policy_prompt_variant != POLICY_PROMPT_CANONICAL:
            parser.error("version45 does not combine with policy prompt ablations")
        if args.plan_policy != PLAN_POLICY_OPTIONAL:
            parser.error("version45 removes plan; do not select a required plan policy")
        if (
            args.database_context_profile
            != CATALOG_BIRD_INSPECT_SEMANTICS_PROFILE
        ):
            parser.error(
                "version45 requires --database-context-profile "
                "catalog-bird-semantics-inspect-only-v1"
            )
    if args.atomic_protocol_version in CONTEXT_ABLATION_PROTOCOL_VERSIONS:
        version_label = args.atomic_protocol_version
        if not args.diagnostic_only:
            parser.error(f"{version_label} is diagnostic-only; pass --diagnostic-only")
        if args.context_mode != "rolling-legal-history":
            parser.error(f"{version_label} requires --context-mode rolling-legal-history")
        if args.history_turns != 4:
            parser.error(f"{version_label} requires --history-turns 4")
        if args.rolling_prompt_variant != "full":
            parser.error(f"{version_label} uses the full version39 prompt")
        if args.policy_prompt_variant != POLICY_PROMPT_CANONICAL:
            parser.error(f"{version_label} does not combine with policy prompt ablations")
        if args.plan_policy != PLAN_POLICY_OPTIONAL:
            parser.error(f"{version_label} requires --plan-policy optional")
        if args.database_context_profile != CATALOG_CONTEXT_PROFILE:
            parser.error(f"{version_label} supports only --database-context-profile catalog-v1")
    if args.atomic_protocol_version == VERSION50_PROTOCOL_VERSION:
        if not args.diagnostic_only:
            parser.error("version50 is diagnostic-only; pass --diagnostic-only")
        if args.deepseek_carrier != DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS:
            parser.error("version50 requires --deepseek-carrier native-tool-calls")
        if args.context_mode != "rolling-legal-history" or args.history_turns != 4:
            parser.error("version50 requires rolling-legal-history with --history-turns 4")
        if args.rolling_prompt_variant != "full":
            parser.error("version50 requires --rolling-prompt-variant full")
        if args.policy_prompt_variant != POLICY_PROMPT_CANONICAL:
            parser.error("version50 does not combine with policy prompt ablations")
        if args.plan_policy != PLAN_POLICY_OPTIONAL:
            parser.error("version50 requires --plan-policy optional")
        if args.database_context_profile != CATALOG_CONTEXT_PROFILE:
            parser.error("version50 supports only --database-context-profile catalog-v1")
    elif args.deepseek_carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_CALLS:
        parser.error("native-tool-calls is isolated to --atomic-protocol-version version50")
    if args.atomic_protocol_version in {
        VERSION51_PROTOCOL_VERSION,
        VERSION52_PROTOCOL_VERSION,
        VERSION53_PROTOCOL_VERSION,
        VERSION54_PROTOCOL_VERSION,
    }:
        version_label = args.atomic_protocol_version
        if not args.diagnostic_only:
            parser.error(f"{version_label} is diagnostic-only; pass --diagnostic-only")
        if args.deepseek_carrier != DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE:
            parser.error(f"{version_label} requires --deepseek-carrier native-tool-bundle")
        if args.context_mode != "rolling-legal-history" or args.history_turns != 4:
            parser.error(
                f"{version_label} requires rolling-legal-history with --history-turns 4"
            )
        if args.rolling_prompt_variant != "full":
            parser.error(f"{version_label} requires --rolling-prompt-variant full")
        if args.policy_prompt_variant != POLICY_PROMPT_CANONICAL:
            parser.error(f"{version_label} does not combine with policy prompt ablations")
        if args.plan_policy != PLAN_POLICY_OPTIONAL:
            parser.error(f"{version_label} requires --plan-policy optional")
        if args.database_context_profile != CATALOG_CONTEXT_PROFILE:
            parser.error(f"{version_label} supports only --database-context-profile catalog-v1")
    elif args.deepseek_carrier == DEEPSEEK_CARRIER_NATIVE_TOOL_BUNDLE:
        parser.error(
            "native-tool-bundle is isolated to --atomic-protocol-version "
            "version51-version54"
        )
    if args.max_tokens is None:
        args.max_tokens = provider_default_max_tokens(args.model, DEFAULT_MAX_TOKENS)

    api_key, base_url = load_api_config()
    if not api_key or not base_url:
        parser.error("api.md must define API_KEY and BASE_URL")
    if is_deepseek_split_model(args.model) and base_url.rstrip("/") != "https://api.deepseek.com":
        parser.error("new DeepSeek runs must use BASE_URL=https://api.deepseek.com")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path = Path(args.failures_out) if args.failures_out else out_path.with_suffix(".failures.jsonl")
    all_path = Path(args.all_out) if args.all_out else out_path.with_suffix(".all.jsonl")
    manifest_path = out_path.with_suffix(".manifest.json")

    examples = load_examples(args)
    completed = read_completed(all_path) if args.resume else set()
    work = [
        (i, ex) for i, ex in examples
        if trajectory_id(args.split, i, ex) not in completed
    ]
    version40 = args.atomic_protocol_version == VERSION40_PROTOCOL_VERSION
    version41 = args.atomic_protocol_version == VERSION41_PROTOCOL_VERSION
    version42 = args.atomic_protocol_version == VERSION42_PROTOCOL_VERSION
    version43 = args.atomic_protocol_version == VERSION43_PROTOCOL_VERSION
    version44 = args.atomic_protocol_version == VERSION44_PROTOCOL_VERSION
    version45 = args.atomic_protocol_version == VERSION45_PROTOCOL_VERSION
    version50 = args.atomic_protocol_version == VERSION50_PROTOCOL_VERSION
    version51 = args.atomic_protocol_version == VERSION51_PROTOCOL_VERSION
    version52 = args.atomic_protocol_version == VERSION52_PROTOCOL_VERSION
    version53 = args.atomic_protocol_version == VERSION53_PROTOCOL_VERSION
    version54 = args.atomic_protocol_version == VERSION54_PROTOCOL_VERSION
    native_bundle_protocol = version51 or version52 or version53 or version54
    full_reasoning_no_plan = (
        args.atomic_protocol_version in FULL_REASONING_NO_PLAN_PROTOCOL_VERSIONS
    )
    if version54:
        student_prompt = VERSION54_STUDENT_SYSTEM_PROMPT
        canonical_teacher_prompt = VERSION54_TEACHER_SYSTEM_PROMPT
        system_prompt = VERSION54_TEACHER_SYSTEM_PROMPT
    elif version53:
        student_prompt = VERSION53_STUDENT_SYSTEM_PROMPT
        canonical_teacher_prompt = VERSION53_TEACHER_SYSTEM_PROMPT
        system_prompt = VERSION53_TEACHER_SYSTEM_PROMPT
    elif version52:
        student_prompt = VERSION52_STUDENT_SYSTEM_PROMPT
        canonical_teacher_prompt = VERSION52_TEACHER_SYSTEM_PROMPT
        system_prompt = VERSION52_TEACHER_SYSTEM_PROMPT
    elif version45:
        student_prompt = rolling_system_prompt(
            VERSION45_STUDENT_SYSTEM_PROMPT,
            compact=False,
        )
        canonical_teacher_prompt = (
            teacher_system_prompt_version45(student_prompt)
            + VERSION44_DATA_GENERATION_SUFFIX
        )
        system_prompt = provider_system_prompt_version45(
            args.model,
            canonical_teacher_prompt,
            carrier=args.deepseek_carrier,
        )
    elif version44:
        student_prompt = rolling_system_prompt(
            VERSION44_STUDENT_SYSTEM_PROMPT,
            compact=False,
        )
        canonical_teacher_prompt = (
            teacher_system_prompt_version44(student_prompt)
            + VERSION44_DATA_GENERATION_SUFFIX
        )
        system_prompt = provider_system_prompt_version44(
            args.model,
            canonical_teacher_prompt,
            carrier=args.deepseek_carrier,
        )
    elif version43:
        student_prompt = VERSION43_STUDENT_SYSTEM_PROMPT
        canonical_teacher_prompt = VERSION43_STUDENT_SYSTEM_PROMPT
        system_prompt = provider_system_prompt_version43(
            args.model,
            carrier=args.deepseek_carrier,
        )
    elif version42:
        student_prompt = VERSION42_STUDENT_SYSTEM_PROMPT
        canonical_teacher_prompt = VERSION42_STUDENT_SYSTEM_PROMPT
        system_prompt = provider_system_prompt_version42(
            args.model,
            carrier=args.deepseek_carrier,
        )
    elif version41:
        student_prompt = VERSION41_STUDENT_SYSTEM_PROMPT
        canonical_teacher_prompt = VERSION41_STUDENT_SYSTEM_PROMPT
        system_prompt = provider_system_prompt_version41(
            args.model,
            carrier=args.deepseek_carrier,
        )
    elif version40:
        student_prompt = VERSION40_STUDENT_SYSTEM_PROMPT
        canonical_teacher_prompt = VERSION40_STUDENT_SYSTEM_PROMPT
        system_prompt = provider_system_prompt_version40(
            args.model,
            carrier=args.deepseek_carrier,
        )
    else:
        student_prompt = (
            build_full_context_student_prompt(args.database_context_profile)
            if args.database_context_profile in FULL_BIRD_CONTEXT_PROFILES
            else get_system_prompt()
        )
        if args.context_mode == "rolling-legal-history":
            student_prompt = rolling_system_prompt(
                student_prompt,
                compact=args.rolling_prompt_variant == "compact",
            )
        base_system_prompt = (
            build_full_context_teacher_prompt(
                student_prompt,
                args.database_context_profile,
            )
            if args.database_context_profile in FULL_BIRD_CONTEXT_PROFILES
            else teacher_system_prompt(student_prompt)
        )
        base_system_prompt = policy_system_prompt(
            base_system_prompt,
            args.policy_prompt_variant,
        )
        prompt_suffix = DATA_GENERATION_SUFFIX
        if args.atomic_protocol_version == VERSION48_PROTOCOL_VERSION:
            prompt_suffix += INTERPRET_BEFORE_ACT_SUFFIX
        if args.plan_policy == PLAN_POLICY_REQUIRED_RESIDENT:
            prompt_suffix += REQUIRED_RESIDENT_PLAN_SUFFIX
        canonical_teacher_prompt = base_system_prompt + prompt_suffix
        system_prompt = provider_system_prompt(
            args.model,
            canonical_teacher_prompt,
            carrier=args.deepseek_carrier,
        )
    prompt_audit = {
        "teacher_prompt_role": "teacher-generation",
        "teacher_canonical_prompt_sha256": hashlib.sha256(
            canonical_teacher_prompt.encode("utf-8")
        ).hexdigest(),
        "teacher_provider_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "student_runtime_prompt_sha256": hashlib.sha256(
            student_prompt.encode("utf-8")
        ).hexdigest(),
        "tool_schema_sha256": selected_tool_schema_hash(
            args.atomic_protocol_version
        ),
        "model_visible_tool_schema_sha256": (
            selected_tool_schema_hash(args.atomic_protocol_version)
            if (
                args.atomic_protocol_version in NO_PLAN_PROTOCOL_VERSIONS
                or native_bundle_protocol
            )
            else model_visible_tool_schema_hash(args.database_context_profile)
        ),
        "database_context_profile": args.database_context_profile,
        "resident_state_profile": selected_resident_context(
            args.atomic_protocol_version
        )[0],
        "latest_observation_full": selected_resident_context(
            args.atomic_protocol_version
        )[1],
        "active_relation_policy": (
            "recent-4-references-plus-dependency-closure-v1"
            if args.atomic_protocol_version == VERSION49_PROTOCOL_VERSION
            else None
        ),
        "disabled_model_tools": sorted(
            disabled_tools_for_profile(args.database_context_profile)
            | ({"plan"} if version54 else set())
        ),
        "provider_assistant_carrier": (
            VERSION51_PROVIDER_ASSISTANT_CARRIER
            if native_bundle_protocol
            else (VERSION50_PROVIDER_ASSISTANT_CARRIER if version50 else None)
        ),
        "provider_prompt_profile": (
            VERSION54_PROMPT_PROFILE
            if version54
            else (
                VERSION53_PROMPT_PROFILE
                if version53
                else (VERSION52_PROMPT_PROFILE if version52 else None)
            )
        ),
        "static_request_size_audit": (
            native_no_plan_prompt_size_audit(
                native_atomic_tools(VERSION54_MODEL_ARG_SCHEMA)
            )
            if version54
            else (
                native_reviewed_prompt_size_audit(native_atomic_tools())
                if version53
                else (
                    native_compact_prompt_size_audit(native_atomic_tools())
                    if version52
                    else None
                )
            )
        ),
    }
    started = time.time()
    counts = collections.Counter()
    tool_hist = collections.Counter()
    error_turn_hist = collections.Counter()
    usage_total = collections.Counter()
    native_bundle_credit_summaries: list[dict] = []
    training_result_quality_counts = collections.Counter()

    def process_once(item: tuple[int, dict], attempt_index: int) -> dict:
        i, ex = item
        rec = run_rollout(
            example_index=i,
            ex=ex,
            split=args.split,
            base_url=base_url,
            api_key=api_key,
            model=args.model,
            system_prompt=system_prompt,
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            api_retries=args.api_retries,
            api_timeout=args.api_timeout,
            max_errors_per_type=args.max_errors_per_type,
            table_output_rows=args.table_output_rows,
            context_mode=args.context_mode,
            history_turns=args.history_turns,
            rolling_prompt_variant=args.rolling_prompt_variant,
            prompt_audit=prompt_audit,
            policy_prompt_variant=args.policy_prompt_variant,
            plan_policy=args.plan_policy,
            deepseek_carrier=args.deepseek_carrier,
            denotation_comparison=args.denotation_comparison,
            diagnostic_only=args.diagnostic_only,
            database_context_profile=args.database_context_profile,
            schema_value_count=args.schema_value_count,
            schema_metadata_json=args.schema_metadata_json or None,
            atomic_protocol_version=args.atomic_protocol_version,
        )
        rec["attempt_index"] = attempt_index
        rec["attempts_per_example"] = max(1, args.attempts_per_example)
        return rec

    def process(item: tuple[int, dict]) -> list[dict]:
        attempts = []
        max_attempts = max(1, args.attempts_per_example)
        for attempt_index in range(1, max_attempts + 1):
            rec = process_once(item, attempt_index)
            rec["example_complete"] = bool(rec.get("correct")) or attempt_index == max_attempts
            attempts.append(rec)
            if rec.get("correct"):
                break
        return attempts

    persisted_this_run = 0
    issued_this_run = 0
    acceptance_gate_examples = 0
    semantic_failure_streak = 0
    maximum_semantic_failure_streak = 0
    semantic_stop_triggered = False
    semantic_stop_after_work_position: int | None = None
    semantic_stop_after_example_index: int | None = None
    semantic_stop_inflight_overrun = 0

    def persist_attempts(
        attempts: list[dict],
        *,
        work_position: int,
        acceptance_gate_in_scope: bool,
    ) -> None:
        nonlocal persisted_this_run
        final = attempts[-1]
        counts["examples"] += 1
        counts["examples_correct" if final.get("correct") else "examples_failed"] += 1
        counts["attempts_total"] += len(attempts)
        for rec in attempts:
            rec["ordered_pilot_gate"] = {
                "work_position": work_position,
                "source_example_index": work[work_position][0],
                "acceptance_gate_in_scope": acceptance_gate_in_scope,
                "inflight_after_semantic_stop": not acceptance_gate_in_scope,
            }
            if native_bundle_protocol and rec.get("native_bundle_rl_statistics"):
                native_bundle_credit_summaries.append(
                    rec["native_bundle_rl_statistics"]
                )
            result_quality = rec.get("training_result_quality") or {}
            training_result_quality_counts["empty_result_context_only_steps"] += int(
                result_quality.get("empty_result_context_only_steps") or 0
            )
            if result_quality.get("empty_terminal_evidence"):
                training_result_quality_counts["empty_terminal_trajectories"] += 1
            elif result_quality.get("has_terminal_evidence"):
                training_result_quality_counts["nonempty_terminal_trajectories"] += 1
            else:
                training_result_quality_counts["no_terminal_trajectories"] += 1
            counts["total"] += 1
            counts["correct" if rec.get("correct") else "failed"] += 1
            if rec.get("legal"):
                counts["legal"] += 1
            if rec.get("failure_type"):
                counts[f"failure:{rec['failure_type']}"] += 1
            add_usage(usage_total, rec.get("usage") or {})
            if rec.get("correct"):
                traj = rec["trajectory"]
                traj["rollout_generation"]["attempt_index"] = rec.get("attempt_index")
                traj["rollout_generation"]["attempts_per_example"] = rec.get("attempts_per_example")
                traj["rollout_generation"]["ordered_pilot_gate"] = rec[
                    "ordered_pilot_gate"
                ]
                append_jsonl(out_path, traj)
                for step in traj.get("steps", []):
                    tool_hist[step["tool_call"]["tool"]] += 1
                    if step.get("think_source"):
                        counts[f"think_source:{step['think_source']}"] += 1
                    if step.get("tool_status") == "error":
                        error_turn_hist[step["tool_call"]["tool"]] += 1
            else:
                append_jsonl(failure_path, rec)
            append_jsonl(all_path, {
                k: v for k, v in rec.items()
                if k != "trajectory"
            })
        persisted_this_run += 1
        status = "OK " if final.get("correct") else "ERR"
        scope = "gate" if acceptance_gate_in_scope else "inflight-after-stop"
        print(
            f"[{persisted_this_run}/{len(work)}] {status} scope={scope} "
            f"attempts={len(attempts)} steps={final.get('steps')} "
            f"errs={final.get('errors')} type={final.get('failure_type')} "
            f"{final.get('trajectory_id')}"
        )

    worker_count = max(1, args.workers)
    if args.max_consecutive_semantic_failures == 0:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_positions = {
                pool.submit(process, item): work_position
                for work_position, item in enumerate(work)
            }
            issued_this_run = len(future_positions)
            for future in as_completed(future_positions):
                persist_attempts(
                    future.result(),
                    work_position=future_positions[future],
                    acceptance_gate_in_scope=True,
                )
        acceptance_gate_examples = persisted_this_run
    else:
        # Keep at most one worker-sized source-order window in flight. New work is submitted only
        # after the contiguous prefix has been observed, so detecting the stop never launches a
        # later window. Calls already issued before the fifth failure may finish for audit, but do
        # not count toward the acceptance gate.
        next_submit = 0
        next_persist = 0
        pending: dict = {}
        ready: dict[int, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            while pending or ready or (
                not semantic_stop_triggered and next_submit < len(work)
            ):
                while (
                    not semantic_stop_triggered
                    and next_submit < len(work)
                    and next_submit < next_persist + worker_count
                ):
                    future = pool.submit(process, work[next_submit])
                    pending[future] = next_submit
                    next_submit += 1
                    issued_this_run += 1

                if pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        work_position = pending.pop(future)
                        ready[work_position] = future.result()

                while next_persist in ready:
                    attempts = ready.pop(next_persist)
                    gate_in_scope = not semantic_stop_triggered
                    persist_attempts(
                        attempts,
                        work_position=next_persist,
                        acceptance_gate_in_scope=gate_in_scope,
                    )
                    if gate_in_scope:
                        acceptance_gate_examples += 1
                        final = attempts[-1]
                        if is_semantic_failure_for_ordered_stop(final):
                            semantic_failure_streak += 1
                        else:
                            semantic_failure_streak = 0
                        maximum_semantic_failure_streak = max(
                            maximum_semantic_failure_streak,
                            semantic_failure_streak,
                        )
                        if (
                            semantic_failure_streak
                            >= args.max_consecutive_semantic_failures
                        ):
                            semantic_stop_triggered = True
                            semantic_stop_after_work_position = next_persist
                            semantic_stop_after_example_index = work[next_persist][0]
                    else:
                        semantic_stop_inflight_overrun += 1
                    next_persist += 1

                if semantic_stop_triggered and not pending and not ready:
                    break

    persisted = audit_summary(all_path)
    manifest = {
        "generator": "src/sft/generate_teacher_rollouts.py",
        "tool_scheme": (
            NATIVE_TOOL_BUNDLE_SCHEME if native_bundle_protocol else ATOMIC_TOOL_SCHEME
        ),
        "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
        "assistant_carrier": (
            VERSION51_PROVIDER_ASSISTANT_CARRIER
            if native_bundle_protocol
            else ATOMIC_ASSISTANT_CARRIER
        ),
        "provider_assistant_carrier": (
            VERSION51_PROVIDER_ASSISTANT_CARRIER
            if native_bundle_protocol
            else (VERSION50_PROVIDER_ASSISTANT_CARRIER if version50 else None)
        ),
        "protocol_version": args.atomic_protocol_version,
        "method": "external_llm_closed_loop",
        "model": args.model,
        "split": args.split,
        "examples_file": args.examples_file or None,
        "indices_file": args.indices_file or None,
        "source_count": len(examples),
        "planned_this_run": len(work),
        "attempted_this_run": issued_this_run,
        "persisted_this_run": persisted_this_run,
        "resume": args.resume,
        "output": str(out_path),
        "failures_output": str(failure_path),
        "all_output": str(all_path),
        "protocol_hash": selected_protocol_hash(
            args.atomic_protocol_version,
            system_prompt,
        ),
        "value_search_policy": (
            "bounded-sql-candidate-v1"
            if version45
            else ("exhaustive-distinct-v1" if version44 else None)
        ),
        **prompt_audit,
        "max_steps": args.max_steps,
        "max_native_bundle_calls": (
            MAX_NATIVE_BUNDLE_CALLS if native_bundle_protocol else None
        ),
        "model_turn_boundary": (
            "provider-native-assistant-turn" if native_bundle_protocol else "single-action"
        ),
        "workers": max(1, args.workers),
        "ordered_semantic_failure_stop": {
            "threshold": args.max_consecutive_semantic_failures,
            "definition": (
                "incorrect final example excluding api_error, context_overflow, "
                "and provider_carrier_error"
            ),
            "source_ordered": bool(args.max_consecutive_semantic_failures),
            "acceptance_gate_examples": acceptance_gate_examples,
            "maximum_streak": maximum_semantic_failure_streak,
            "triggered": semantic_stop_triggered,
            "stop_after_work_position": semantic_stop_after_work_position,
            "stop_after_example_index": semantic_stop_after_example_index,
            "inflight_overrun_examples": semantic_stop_inflight_overrun,
        },
        "max_errors_per_type": args.max_errors_per_type,
        "attempts_per_example": max(1, args.attempts_per_example),
        "max_tokens": args.max_tokens,
        "table_output_rows": args.table_output_rows,
        "context_mode": args.context_mode,
        "history_turns": args.history_turns,
        "rolling_prompt_variant": args.rolling_prompt_variant,
        "policy_prompt_variant": args.policy_prompt_variant,
        "plan_policy": (
            "disabled-by-protocol"
            if args.atomic_protocol_version in NO_PLAN_PROTOCOL_VERSIONS
            else args.plan_policy
        ),
        "history_reasoning": (
            "provider-native-per-model-turn"
            if native_bundle_protocol
            else (
                "complete-all-successful-and-rejected"
                if full_reasoning_no_plan
                else "omitted-from-provider-history"
            )
        ),
        "history_observations": (
            "one-tool-result-per-call-recent-4-provider-turns"
            if native_bundle_protocol
            else selected_history_observation_policy(
                args.atomic_protocol_version,
                full_reasoning_no_plan=full_reasoning_no_plan,
            )
        ),
        "terminal_evidence_policy": (
            "explicit-columns-unique-bare-deterministic-project-v2"
            if version43
            else (
                "explicit-columns-deterministic-project-v1"
                if version42
                else "exact-table"
            )
        ),
        "deepseek_carrier": args.deepseek_carrier,
        "denotation_comparison": args.denotation_comparison,
        "database_context_profile": args.database_context_profile,
        "schema_value_count": (
            args.schema_value_count
            if args.database_context_profile in FULL_BIRD_CONTEXT_PROFILES
            else None
        ),
        "schema_metadata_json": args.schema_metadata_json or None,
        "disabled_model_tools": sorted(
            disabled_tools_for_profile(args.database_context_profile)
            | ({"plan"} if version54 else set())
        ),
        "sft_export_eligible": sft_export_eligible(
            context_mode=args.context_mode,
            history_turns=args.history_turns,
            rolling_prompt_variant=args.rolling_prompt_variant,
            denotation_comparison=args.denotation_comparison,
            diagnostic_only=args.diagnostic_only,
        ),
        "training_admission": training_admission(
            diagnostic_only=args.diagnostic_only,
        ),
        "teacher_parser": "strict_no_repair",
        "error_actions_are_sft_targets": False,
        "error_feedback_training_policy": (
            "preserve-error-turn-and-tool-feedback; later-corrected-bundle-is-target"
            if native_bundle_protocol
            else None
        ),
        "native_bundle_rl_statistics": (
            aggregate_native_bundle_credit(native_bundle_credit_summaries)
            if native_bundle_protocol
            else None
        ),
        "training_result_quality": {
            "empty_result_policy": EMPTY_RESULT_POLICY_VERSION,
            "empty_intermediate_call_target": "context-only-no-loss",
            "empty_terminal_trajectory": "exclude-whole-trajectory",
            "counts": dict(sorted(training_result_quality_counts.items())),
        },
        "api_transport_retries_per_request": args.api_retries,
        "provider_request_options": provider_request_audit_options(
            args.model,
            carrier=args.deepseek_carrier,
            native_model_arg_schema=(
                VERSION54_MODEL_ARG_SCHEMA if version54 else None
            ),
        ),
        "counts": persisted["counts"],
        "raw_attempt_records": persisted["raw_attempt_records"],
        "unique_examples": persisted["unique_examples"],
        "duplicate_attempt_records": persisted["duplicate_attempt_records"],
        "tool_hist": dict(tool_hist.most_common()),
        "error_turn_tool_hist": dict(error_turn_hist.most_common()),
        "usage_total": persisted["usage_total"],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if persisted["counts"].get("total"):
        manifest["accuracy"] = persisted["counts"].get("correct", 0) / persisted["counts"]["total"]
        manifest["legal_rate"] = persisted["counts"].get("legal", 0) / persisted["counts"]["total"]
        manifest["example_success_rate"] = (
            persisted["counts"].get("examples_correct", 0) /
            max(1, persisted["counts"].get("examples", 0))
        )
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
