#!/usr/bin/env python3
"""Unified variable-width action-block protocol.

Work turns contain one top-level ``action_block`` with one to five nonterminal primitive calls.
Termination is one separate top-level ``answer_from_context`` action. A single-call work block is
the safe atomic case; wider blocks reduce model round trips when arguments are already grounded.
The harness owns ordered execution, handles, resident state, and blocked propagation. The model
only reasons, calls tools, and interprets factual feedback.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

from action_carrier import (
    ACTIVE_ACTION_CARRIER,
    ActionCarrierError,
    parse_action_carrier,
    render_action_carrier,
)
from protocol import (
    ProtocolError,
    TOOL_SPECS,
    first_user_message,
    state_context_message,
    tool_output_message,
    validate_action_block_arguments,
)
from prompt_contract import ACTION_BLOCK_TOOL_SPECS


BATCH_PLAN_PROTOCOL_VERSION = "action-block-v4"
STRUCTURED_ERROR_FEEDBACK_PROTOCOL_VERSION = "action-block-v9"
LOW_FRICTION_INTERFACE_PROTOCOL_VERSION = "action-block-v10"
SAFE_LOW_FRICTION_INTERFACE_PROTOCOL_VERSION = "action-block-v11"
UNIFIED_ACTION_BLOCK_PROTOCOL_VERSION = "action-block-v32"
BATCH_PLAN_TOOL = "action_block"
TERMINAL_TOOL = "answer_from_context"
MAX_ACTION_BLOCK_CALLS = 5
BATCH_CARRIER_PROVIDER_NATIVE = "provider-native-reasoning-raw-json"
BATCH_CARRIER_INLINE_THINK = ACTIVE_ACTION_CARRIER
BATCH_CARRIERS = (
    BATCH_CARRIER_PROVIDER_NATIVE,
    BATCH_CARRIER_INLINE_THINK,
)
EXECUTABLE_TOOLS = tuple(
    tool for tool in TOOL_SPECS
    if tool not in {"plan", TERMINAL_TOOL}
)
OBSERVATION_ONLY_TOOLS = frozenset({
    "describe_table",
    "inspect_column",
    "read_subtable",
})
CALL_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
LOCAL_REF_RE = re.compile(r"^\$([A-Za-z][A-Za-z0-9_]{0,31})$")
LOCAL_COLUMN_REF_RE = re.compile(
    r"^\$([A-Za-z][A-Za-z0-9_]{0,31})\.(.+)$"
)


class BatchPlanProtocolError(ProtocolError):
    """The top-level batch-plan action is malformed."""


class LocalReferenceError(ProtocolError):
    """A local reference is malformed, missing, forward, or cannot resolve structurally."""


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _observation_reference_error(
    value: str,
    binding: dict,
) -> str | None:
    tool = binding.get("tool")
    if tool not in OBSERVATION_ONLY_TOOLS:
        return None
    message = (
        f"local reference {value!r} targets observation-only {tool}; observation-only calls "
        "do not produce reusable tables or scalar values"
    )
    source_reference = binding.get("source_reference")
    if isinstance(source_reference, str) and source_reference:
        message += (
            f". The observed input was {source_reference!r}; if that relation is the intended "
            "result, cite the producer relation instead of the observation call"
        )
    return message


def _atomic_specs() -> str:
    return "\n".join(ACTION_BLOCK_TOOL_SPECS[tool] for tool in EXECUTABLE_TOOLS)


def build_batch_plan_system_prompt(
    max_batch_calls: int = MAX_ACTION_BLOCK_CALLS,
    *,
    assistant_carrier: str = BATCH_CARRIER_PROVIDER_NATIVE,
) -> str:
    """Build the active unified action-block provider contract."""
    if not 1 <= max_batch_calls <= MAX_ACTION_BLOCK_CALLS:
        raise ValueError(
            f"active action blocks require max_batch_calls in 1..{MAX_ACTION_BLOCK_CALLS}"
        )
    if assistant_carrier not in BATCH_CARRIERS:
        raise ValueError(
            f"unknown action-block assistant carrier {assistant_carrier!r}; "
            f"expected one of {BATCH_CARRIERS}"
        )
    tools = _atomic_specs()
    if assistant_carrier == BATCH_CARRIER_PROVIDER_NATIVE:
        format_rule = (
            "RESPONSE FORMAT: Follow the provider-specific split-response contract below. Its visible action is "
            "one complete JSON action object."
        )
    else:
        format_rule = (
            "RESPONSE FORMAT: Output exactly one non-empty <think>brief reason</think> block followed directly by "
            "one complete raw JSON object with only tool and arguments. No <tool_call> tag, "
            "Markdown, prose, or second action."
        )
    prompt = f"""You are a relational table-tool agent using one unified action-block interface.

WORKSPACE
The opening overview is a catalog, not table schemas. AVAILABLE TOOL CONTEXT is read-only factual
state maintained by the harness: observed schemas and values, resident result handles, row reads,
scalar-producing step ids, and fact-only derivations. You reason, call tools, and interpret their
feedback; you never author plans, dependencies, statuses, handles, or environment state.

ATOMIC TOOLS AVAILABLE INSIDE action_block.calls
{tools}

TOP-LEVEL ACTION CONTRACT
On each turn emit exactly one of these two actions:

1. WORK ACTION — one action_block:
{{"tool":"action_block","arguments":{{"calls":[
  {{"id":"local_id","tool":"atomic_tool","arguments":{{...}}}}
]}}}}

2. TERMINAL ACTION — one standalone answer_from_context:
{{"tool":"answer_from_context","arguments":
  {{"evidence":{{"table":"exact_result_handle"}},"reason":"brief optional reason"}}}}

answer_from_context is never nested in action_block.calls. Its evidence must already be resident
before this turn. If any observation, filter, join, aggregate, rank, scalar, project, or read is
still needed, emit a work action now and wait for its feedback before answering on the next turn.

1. calls contains 1 to {max_batch_calls} calls. Every call has exactly id, tool, arguments; ids are
   unique and begin with a letter. Calls contain only the observation and relational tools listed
   above. The list order is both execution order and feedback order.
2. Use each atomic tool's public arguments exactly as documented above. The harness performs no
   spelling repair, schema repair, column repair, predicate rewrite, or alternative-shape rewrite.
   Copy resident table handles and step ids exactly from returned feedback.
3. "$id" is block-local and may refer only to an earlier call in the same list. In a table position
   it means that call's result table; in value_ref it means that call's producing step.
   "$id.column" means that earlier call's exact output column. Forward and cross-block "$id"
   references are invalid; use the returned resident handle or step id in a later block.
4. One block is one semantic action. Put calls together only when every tool, column, predicate,
   literal, grain, and output choice is already determined from feedback visible before the block.
   An execution-dependent later call may consume an earlier result through "$id", but if seeing an
   intermediate schema, value, row, or error could change the next choice, end the block and decide
   again after feedback.
5. Calls run in list order. A failed call is reported as error. A later call that references that
   failed result is not run and is reported as blocked. Independent later calls still run. Every
   submitted call receives one result entry with its full atomic output, error, or blocked cause.
6. The standalone terminal action's cited resident table must already have exactly the answer rows,
   columns, column order, grain, ordering, and duplicates. Use an explicit project call in an
   earlier work action when helper columns remain or column order is wrong. The harness never
   writes values or reshapes terminal evidence. A table from which the answer is merely inferable
   is not exact: before terminating, derive any requested winner row and project only the question's
   answer fields in their requested order. The reason cannot drop rows/columns, combine fields, or
   replace an identifier with a label.
7. Unknown schemas and uncertain text literals require an earlier observation block. Preserve the
   question's population and grain through joins, filters, ranking, aggregation, and projection.
   read_subtable observes rows but does not reshape them. scalar_compute operand order is semantic.
8. Every join item uses exactly
   {{"table":"new_table","on":[{{"left":"base.column","right":"new_column"}}]}}.
   on is always a list of pair objects. Copy on.left from the exact logical columns currently
   exposed by its base. A derived base without base_role uses its resident handle namespace, for
   example "filter_001.column". With base_role "orders", use "orders.column". on.right remains
   the new table's bare column.

{format_rule}

ORDERED MULTI-CALL EXAMPLE AFTER ALL CHOICES ARE GROUNDED
{{"tool":"action_block","arguments":{{"calls":[
  {{"id":"filtered","tool":"condition_filter","arguments":{{"table":"orders",
    "conditions":{{"column":"amount","op":">","value":100}}}}}},
  {{"id":"exact","tool":"project","arguments":{{"table":"$filtered",
    "expressions":["order_id"],"distinct":true}}}},
  {{"id":"rows","tool":"read_subtable","arguments":{{"table":"$exact","limit":20}}}}
]}}}}
"""
    if assistant_carrier == BATCH_CARRIER_PROVIDER_NATIVE:
        prompt += (
            "\n\nPROVIDER-NATIVE SPLIT RESPONSE\n"
            "Put one non-empty action reason in the separate native reasoning field. Keep it "
            "under 120 words and reserve output budget for the action; reasoning is incomplete "
            "until the action is emitted. If uncertain, emit the next legal observation instead "
            "of extending the analysis. After reasoning, the entire visible response is exactly "
            "one JSON object with only \"tool\" and \"arguments\". JSON Output is enabled.\n"
            "Copy this visible shape, replacing only JSON values:\n"
            '{"tool":"action_block","arguments":{"calls":['
            '{"id":"schema","tool":"describe_table","arguments":'
            '{"tables":["Document"]}}]}}\n'
            "When the exact result table is resident, copy this visible terminal shape:\n"
            '{"tool":"answer_from_context","arguments":{"evidence":'
            '{"table":"project_001"}}}\n'
            "Even when terminating, native reasoning is not the visible response: emit the "
            "complete terminal JSON after reasoning. answer_from_context stays at the top level, "
            "never inside calls. Put every parameter inside arguments. Do not add "
            "prose, Markdown, XML tags, a second action, or any text before or after the JSON."
        )
    return prompt


def batch_plan_protocol_hash(
    system_prompt: str,
    max_batch_calls: int,
    *,
    protocol_version: str = UNIFIED_ACTION_BLOCK_PROTOCOL_VERSION,
) -> str:
    payload = {
        "version": protocol_version,
        "system_prompt": system_prompt,
        "max_batch_calls": max_batch_calls,
        "atomic_tools": {
            tool: ACTION_BLOCK_TOOL_SPECS[tool] for tool in EXECUTABLE_TOOLS
        },
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def parse_batch_plan_action(
    raw_content: str,
    *,
    max_batch_calls: int = MAX_ACTION_BLOCK_CALLS,
) -> tuple[str, dict]:
    """Strictly parse one active provider-visible work or terminal action without repair."""
    return _parse_batch_plan_action(
        raw_content,
        max_batch_calls=max_batch_calls,
        allow_legacy_nested_terminal=False,
    )


def parse_legacy_batch_plan_action(
    raw_content: str,
    *,
    max_batch_calls: int,
) -> tuple[str, dict]:
    """Parse retired action-block shapes for named offline replay/reproduction only."""
    return _parse_batch_plan_action(
        raw_content,
        max_batch_calls=max_batch_calls,
        allow_legacy_nested_terminal=True,
    )


def _parse_batch_plan_action(
    raw_content: str,
    *,
    max_batch_calls: int,
    allow_legacy_nested_terminal: bool,
) -> tuple[str, dict]:
    if not allow_legacy_nested_terminal and not (
        1 <= max_batch_calls <= MAX_ACTION_BLOCK_CALLS
    ):
        raise BatchPlanProtocolError(
            f"active action blocks require max_batch_calls in 1..{MAX_ACTION_BLOCK_CALLS}"
        )
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise BatchPlanProtocolError("visible response is empty")
    try:
        action = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise BatchPlanProtocolError(f"visible response is not one complete JSON object: {exc}") from exc
    if not isinstance(action, dict) or set(action) != {"tool", "arguments"}:
        raise BatchPlanProtocolError(
            'action must be an object containing exactly "tool" and "arguments"'
        )
    tool = action.get("tool")
    arguments = action.get("arguments")
    allowed_top_level = {BATCH_PLAN_TOOL, TERMINAL_TOOL}
    if tool not in allowed_top_level:
        raise BatchPlanProtocolError(
            "top-level tool must be action_block or answer_from_context"
        )
    if not isinstance(arguments, dict):
        raise BatchPlanProtocolError("action.arguments must be an object")

    if tool == TERMINAL_TOOL:
        _validate_terminal_arguments(arguments)
        return tool, arguments

    if set(arguments) != {"calls"}:
        raise BatchPlanProtocolError(
            'action_block.arguments must contain exactly "calls"'
        )
    calls = arguments.get("calls")
    if not isinstance(calls, list) or not 1 <= len(calls) <= max_batch_calls:
        raise BatchPlanProtocolError(
            f"action_block.calls must contain 1 to {max_batch_calls} calls"
        )
    seen: set[str] = set()
    for index, call in enumerate(calls):
        where = f"action_block.calls[{index}]"
        if not isinstance(call, dict) or set(call) != {"id", "tool", "arguments"}:
            raise BatchPlanProtocolError(
                f"{where} must contain exactly id, tool, and arguments"
            )
        call_id = call.get("id")
        if not isinstance(call_id, str) or not CALL_ID_RE.fullmatch(call_id):
            raise BatchPlanProtocolError(
                f"{where}.id must begin with a letter and contain at most 32 letters, digits, "
                "or underscores"
            )
        if call_id in seen:
            raise BatchPlanProtocolError(f"{where}.id duplicates {call_id!r}")
        seen.add(call_id)
        if not isinstance(call.get("tool"), str) or not call["tool"].strip():
            raise BatchPlanProtocolError(f"{where}.tool must be a non-empty string")
        if not isinstance(call.get("arguments"), dict):
            raise BatchPlanProtocolError(f"{where}.arguments must be an object")
    terminal_calls = [call for call in calls if call.get("tool") == TERMINAL_TOOL]
    if terminal_calls:
        if not allow_legacy_nested_terminal:
            raise BatchPlanProtocolError(
                "answer_from_context is a standalone top-level terminal action and cannot "
                "appear in action_block.calls"
            )
        if len(terminal_calls) != 1 or len(calls) != 1:
            raise BatchPlanProtocolError(
                "answer_from_context must be the one and only call in its action_block"
            )
        terminal_call = terminal_calls[0]
        _validate_terminal_arguments(terminal_call["arguments"])
    for index, call in enumerate(calls):
        if call.get("tool") in {"plan", "update_plan"}:
            raise BatchPlanProtocolError(
                f"action_block.calls[{index}].tool cannot manage plans or environment state"
            )
    return tool, arguments


def render_batch_plan_assistant(
    reasoning: str,
    tool: str,
    arguments: dict,
) -> str:
    """Render the strict inline carrier used by local student models and SFT."""
    return render_action_carrier(reasoning, tool, arguments)


def parse_batch_plan_assistant(
    text: str,
    *,
    max_batch_calls: int = MAX_ACTION_BLOCK_CALLS,
) -> tuple[str, str, dict]:
    """Parse one inline-think action-block turn without provider-specific fields."""
    try:
        reasoning, action = parse_action_carrier(text)
    except ActionCarrierError as exc:
        raise BatchPlanProtocolError(str(exc)) from exc
    tool, arguments = parse_batch_plan_action(
        _compact(action),
        max_batch_calls=max_batch_calls,
    )
    return reasoning, tool, arguments


def parse_legacy_batch_plan_assistant(
    text: str,
    *,
    max_batch_calls: int,
) -> tuple[str, str, dict]:
    """Parse a retired inline v4-v11 action for named replay only."""
    try:
        reasoning, action = parse_action_carrier(text)
    except ActionCarrierError as exc:
        raise BatchPlanProtocolError(str(exc)) from exc
    tool, arguments = parse_legacy_batch_plan_action(
        _compact(action),
        max_batch_calls=max_batch_calls,
    )
    return reasoning, tool, arguments


def _validate_terminal_arguments(arguments: dict) -> None:
    try:
        validate_action_block_arguments(TERMINAL_TOOL, arguments)
    except ProtocolError as exc:
        raise BatchPlanProtocolError(str(exc)) from exc


def validate_atomic_call(tool: str, arguments: dict) -> None:
    if tool not in EXECUTABLE_TOOLS:
        raise ProtocolError(
            f"unknown or non-nestable atomic tool {tool!r}; legal tools: "
            f"{list(EXECUTABLE_TOOLS)}"
        )
    validate_action_block_arguments(tool, arguments)


def local_reference_ids(value: Any) -> list[str]:
    """Return syntactically valid local ids referenced anywhere in an argument tree."""
    found: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
            return
        if not isinstance(item, str) or not item.startswith("$"):
            return
        match = LOCAL_COLUMN_REF_RE.fullmatch(item) or LOCAL_REF_RE.fullmatch(item)
        if match and match.group(1) not in found:
            found.append(match.group(1))

    visit(value)
    return found


def resolve_local_references(
    value: Any,
    bindings: dict[str, dict],
    declared_ids: set[str],
    *,
    parent_key: str | None = None,
) -> Any:
    """Resolve exact ``$id`` strings against earlier calls in the current block."""
    if isinstance(value, list):
        return [
            resolve_local_references(
                item, bindings, declared_ids, parent_key=parent_key
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: resolve_local_references(
                item, bindings, declared_ids, parent_key=key
            )
            for key, item in value.items()
        }
    if not isinstance(value, str) or not value.startswith("$"):
        return value
    column_match = LOCAL_COLUMN_REF_RE.fullmatch(value)
    if column_match:
        call_id, requested = column_match.groups()
        if call_id not in declared_ids:
            raise LocalReferenceError(
                f"local column reference {value!r} names no call in this block; resident handles "
                'must be written without "$"'
            )
        binding = bindings.get(call_id)
        if binding is None:
            raise LocalReferenceError(
                f"local column reference {value!r} is forward; only earlier calls may be "
                "referenced"
            )
        if binding.get("status") != "success":
            raise LocalReferenceError(
                f"local column reference {value!r} depends on a "
                f"{binding.get('status')} call"
            )
        columns = binding.get("columns")
        if not isinstance(columns, list):
            observation_error = _observation_reference_error(value, binding)
            if observation_error:
                raise LocalReferenceError(observation_error)
            raise LocalReferenceError(
                f"local column reference {value!r} targets a call with no column output"
            )
        matches = [
            column for column in columns
            if isinstance(column, str) and column == requested
        ]
        if len(matches) != 1:
            raise LocalReferenceError(
                f"local column reference {value!r} must exactly match one output column; "
                f"available columns: {columns}"
            )
        resolved = matches[0]
        if parent_key == "value_ref":
            step_id = binding.get("step_id")
            if not step_id:
                raise LocalReferenceError(
                    f"local scalar reference {value!r} has no producing step id"
                )
            return step_id
        if parent_key == "column_value":
            raise LocalReferenceError(
                f"local result reference {value!r} cannot be used as column_value because "
                "column_value compares two columns in the same input table; cite a verified "
                "one-cell result with value_ref"
            )
        # join_tables requires on.left in its introduced logical namespace. Some unary derived
        # tools expose bare output columns in their result metadata even though the join resolver
        # introduces them as dynamic_handle.column.
        if parent_key == "left" and "." not in resolved and binding.get("table"):
            return f"{binding['table']}.{resolved}"
        return resolved
    match = LOCAL_REF_RE.fullmatch(value)
    if not match:
        raise LocalReferenceError(
            f"local reference {value!r} must be exactly $ followed by a valid earlier call id"
        )
    call_id = match.group(1)
    if call_id not in declared_ids:
        raise LocalReferenceError(
            f"local reference {value!r} names no call in this block; resident handles must be "
            'written without "$"'
        )
    binding = bindings.get(call_id)
    if binding is None:
        raise LocalReferenceError(
            f"local reference {value!r} is forward; only earlier calls may be referenced"
        )
    if binding.get("status") != "success":
        raise LocalReferenceError(
            f"local reference {value!r} depends on a {binding.get('status')} call"
        )
    if parent_key == "value_ref":
        step_id = binding.get("step_id")
        if not step_id:
            raise LocalReferenceError(
                f"local reference {value!r} has no producing step id"
            )
        return step_id
    table = binding.get("table")
    if not table:
        observation_error = _observation_reference_error(value, binding)
        if observation_error:
            raise LocalReferenceError(observation_error)
        raise LocalReferenceError(
            f"local reference {value!r} is not a table-producing call"
        )
    return table


def atomic_output_for_model(step_id: str, output: dict) -> dict:
    """Return the same immediate output payload exposed by one atomic tool observation."""
    try:
        envelope = json.loads(tool_output_message(step_id, output))
    except json.JSONDecodeError:
        return deepcopy(output)
    visible = envelope.get("output")
    return deepcopy(visible) if isinstance(visible, dict) else deepcopy(output)


def render_batch_observation(
    batch_index: int,
    results: list[dict],
    *,
    structured_error_feedback: bool = False,
) -> str:
    visible_results = []
    for result in results:
        item = {
            key: deepcopy(result.get(key))
            for key in ("call_id", "step_id", "tool", "status")
            if result.get(key) is not None
        }
        if result.get("status") == "success":
            item["output"] = atomic_output_for_model(
                result["step_id"], result.get("output") or {}
            )
        elif result.get("status") == "error":
            item["error"] = deepcopy(result.get("error") or {})
        else:
            item["blocked_by"] = deepcopy(result.get("blocked_by") or [])
            item["root_causes"] = deepcopy(result.get("root_causes") or [])
            item["reason"] = str(
                result.get("reason") or "a required earlier call did not succeed"
            )
        if result.get("interface_resolutions"):
            item["interface_resolutions"] = deepcopy(
                result["interface_resolutions"]
            )
        visible_results.append(item)
    counts: dict[str, int] = {}
    for item in visible_results:
        status = str(item.get("status"))
        counts[status] = counts.get(status, 0) + 1
    partial_failure = (
        counts.get("error", 0) > 0 or counts.get("blocked", 0) > 0
    )
    return "ACTION BLOCK RESULTS\n" + _compact({
        "block_index": batch_index,
        "block_status": (
            "partial_failure" if partial_failure else "success"
        ),
        "counts": counts,
        "root_error_calls": [
            item.get("call_id")
            for item in visible_results
            if item.get("status") == "error"
        ],
        "results": visible_results,
    })


def action_block_context_message(
    state: dict | None,
    last_error: dict | None = None,
) -> str:
    """Render harness-owned facts without exposing mutable plan control state."""
    visible = deepcopy(state or {})
    visible.pop("plan", None)
    message = state_context_message(visible, last_error=last_error)
    return (
        message.replace(
            "CURRENT ENVIRONMENT STATE",
            "AVAILABLE TOOL CONTEXT (read-only; maintained by harness)",
            1,
        )
        .replace("LAST TOOL ERROR", "LAST ACTION ERROR", 1)
    )


def build_batch_plan_messages(
    *,
    system_prompt: str,
    overview: dict,
    question: str,
    external_knowledge: str | None,
    state: dict,
    last_error: dict | None,
    legal_history: list[dict],
    history_turns: int,
) -> list[dict]:
    """Render bounded legal batch history plus the latest resident state."""
    if history_turns < 0:
        raise ValueError("history_turns must be non-negative")
    initial = first_user_message(overview, question, external_knowledge)
    if not legal_history:
        if state.get("tables") or state.get("values") or last_error:
            initial += "\n\n" + action_block_context_message(
                state,
                last_error=last_error,
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial},
        ]

    retained = legal_history if history_turns == 0 else legal_history[-history_turns:]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial},
    ]
    for index, item in enumerate(retained):
        assistant = item.get("assistant")
        observation = item.get("observation")
        if not isinstance(assistant, str) or not assistant.strip():
            raise ValueError("legal batch history has an empty assistant action")
        if not isinstance(observation, str) or not observation.strip():
            raise ValueError("legal batch history has an empty observation")
        messages.append({"role": "assistant", "content": assistant})
        if index == len(retained) - 1:
            observation += "\n\n" + action_block_context_message(
                state, last_error=last_error
            )
        messages.append({"role": "user", "content": observation})
    return messages
