#!/usr/bin/env python3
"""Experimental hybrid action-block protocol.

The model emits one ordered block of existing atomic tools. The harness derives dependencies from
local result references, executes independent branches despite recoverable failures, and reports
root failures separately from descendants that were never attempted. ``answer_from_context``
remains a separate grounded terminal action and selects the exact answer columns from one resident
evidence table.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

from protocol import (
    ProtocolError,
    TOOL_SPECS,
    compact_resident_observation,
    first_user_message,
    state_context_message,
    tool_output_message,
    validate_model_arguments,
)


BATCH_PLAN_PROTOCOL_VERSION = "action-block-v4"
STRUCTURED_ERROR_FEEDBACK_PROTOCOL_VERSION = "action-block-v9"
LOW_FRICTION_INTERFACE_PROTOCOL_VERSION = "action-block-v10"
SAFE_LOW_FRICTION_INTERFACE_PROTOCOL_VERSION = "action-block-v11"
BATCH_PLAN_TOOL = "action_block"
TERMINAL_TOOL = "answer_from_context"
BATCH_CARRIER_PROVIDER_NATIVE = "provider-native-reasoning-raw-json"
BATCH_CARRIER_INLINE_THINK = "inline-think-raw-json"
BATCH_CARRIERS = (
    BATCH_CARRIER_PROVIDER_NATIVE,
    BATCH_CARRIER_INLINE_THINK,
)
EXECUTABLE_TOOLS = tuple(
    tool for tool in TOOL_SPECS
    if tool not in {"plan", TERMINAL_TOOL}
)
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


def _atomic_specs() -> str:
    return "\n".join(TOOL_SPECS[tool] for tool in EXECUTABLE_TOOLS)


def build_batch_plan_system_prompt(
    max_batch_calls: int = 8,
    *,
    assistant_carrier: str = BATCH_CARRIER_PROVIDER_NATIVE,
) -> str:
    """Build the concise action-block-v4 provider contract."""
    if max_batch_calls < 1:
        raise ValueError("max_batch_calls must be positive")
    if assistant_carrier not in BATCH_CARRIERS:
        raise ValueError(
            f"unknown action-block assistant carrier {assistant_carrier!r}; "
            f"expected one of {BATCH_CARRIERS}"
        )
    tools = _atomic_specs()
    if assistant_carrier == BATCH_CARRIER_PROVIDER_NATIVE:
        format_rule = (
            "1. Visible output is exactly one JSON object with only tool and arguments. JSON Output "
            "is enabled.\n"
            "   Put one non-empty brief reason in the provider's native reasoning field. No "
            "Markdown/XML/tags."
        )
    else:
        format_rule = (
            "1. Output exactly one non-empty <think>brief reason</think> block followed directly by "
            "one complete raw JSON object with only tool and arguments. No <tool_call> tag, "
            "Markdown, prose, or second action."
        )
    return f"""You are a relational table-tool agent using a hybrid action-block interface.

CONTEXT
The opening overview is a catalog, not table schemas. CURRENT ENVIRONMENT STATE is authoritative:
it contains observed schemas/values, resident derived-table handles, row reads, scalar-producing
step ids, and fact-only relation derivations. A handle exposes columns and row_count but not row
values until read_subtable runs.

ATOMIC TOOLS AVAILABLE INSIDE action_block.calls
{tools}

TOP-LEVEL ACTIONS
Emit only action_block or answer_from_context.

action_block arguments are exactly:
{{"calls":[{{"id":"local_id","tool":"atomic_tool","arguments":{{...}}}}, ...]}}
- calls contains 1 to {max_batch_calls} ordered calls. Every call has exactly id, tool, arguments.
  ids are unique, begin with a letter, and exist only inside this block. plan and
  answer_from_context cannot be nested.
- A block may mix schema/value inspection and relational execution. Include every call whose
  arguments are already grounded. End the block when choosing a later argument requires reading
  and semantically interpreting a new schema, value, row, or error observation.
- The harness executes calls in order. A root error is an attempted call that failed. A blocked
  call is not attempted because an earlier referenced call failed or was blocked; it is not another
  error. Independent later calls still execute after a recoverable root error.
- "$local_id" refers to an earlier successful call in this SAME block. In table/base/in_table
  positions it resolves to that call's produced table; in value_ref it resolves to its step id.
  "$local_id.column" resolves exactly one actual output column, including a dotted suffix such as
  "$joined.orders.customer_id". Forward references are invalid.
- Local references expire after the block. Copy an existing resident handle such as filter_001
  exactly, without "$". If join base is "$filtered", use "$filtered.column" for its on.left because
  the runtime handle namespace is not known before execution.
- After partial failure, use root error facts, blocked_by/root_causes, and reusable successful
  handles to produce a corrected block. Repair or replace a root cause before retrying descendants.

answer_from_context arguments are exactly:
{{"evidence":{{"table":"result_handle","columns":["answer_column",...]}},"reason":"brief optional reason"}}
It is terminal. The table already has final rows/grain/order/duplicates; columns selects one
existing column per requested answer slot. The environment grounds this projection but never
changes rows. Never write answer values.

FINAL CHECK
- QUESTION defines answer slots; use an explicit EXTERNAL KNOWLEDGE phrase-to-column mapping.
  For a person's name, prefer available human-readable name components over a surrogate id unless
  id is requested.
- Omit helper columns used only to locate/join/filter/rank/verify. A scalar selects one column.
- Terminal only drops columns; relational tools establish rows, grain, ordering, and distinctness.

EXAMPLE AFTER SCHEMAS ARE VISIBLE
{{"tool":"action_block","arguments":{{"calls":[
  {{"id":"joined","tool":"join_tables","arguments":{{"base":"orders","joins":[
    {{"table":"customers","on":[{{"left":"orders.customer_id","right":"id"}}]}}
  ]}}}},
  {{"id":"filtered","tool":"condition_filter","arguments":{{"table":"$joined",
    "conditions":{{"column":"customers.country","op":"=","value":"France"}}}}}},
  {{"id":"exact","tool":"project","arguments":{{"table":"$filtered",
    "expressions":["orders.order_id"],"distinct":true}}}},
  {{"id":"rows","tool":"read_subtable","arguments":{{"table":"$exact","limit":20}}}}
]}}}}

RULES
{format_rule}
2. Resolve unknown schemas before using columns. Inspect text domains before uncertain literals.
3. Join on.left is an exact introduced relation.column; on.right is the new table's bare column.
   condition_filter.column_value compares two columns in its SAME input table; for another table
   use join_tables, in_table, or a grounded value_ref.
4. Establish the intended joined population before ranking or aggregating when requested outputs
   come from related tables. Preserve population and grain. Use aggregation where for metrics sharing a
   population and output_layout=columns only for category row-to-column reshape.
5. Keep the exact final population and grain. At terminal, select only the question's answer slots
   in evidence.columns. read_subtable observes but does not reshape. Do not normalize,
   concatenate, round, or deduplicate unless the question requires it.
6. Errors are recoverable factual feedback. Continue from resident successes, repair root errors,
   and do not count or independently retry blocked descendants.
"""


def batch_plan_protocol_hash(
    system_prompt: str,
    max_batch_calls: int,
    *,
    protocol_version: str = BATCH_PLAN_PROTOCOL_VERSION,
) -> str:
    payload = {
        "version": protocol_version,
        "system_prompt": system_prompt,
        "max_batch_calls": max_batch_calls,
        "atomic_tools": {tool: TOOL_SPECS[tool] for tool in EXECUTABLE_TOOLS},
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def parse_batch_plan_action(
    raw_content: str,
    *,
    max_batch_calls: int,
) -> tuple[str, dict]:
    """Strictly parse one provider-visible JSON action without repair."""
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
    if tool not in {BATCH_PLAN_TOOL, TERMINAL_TOOL}:
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
    return tool, arguments


def render_batch_plan_assistant(
    reasoning: str,
    tool: str,
    arguments: dict,
) -> str:
    """Render the strict inline carrier used by local student models and SFT."""
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise BatchPlanProtocolError("inline action-block reasoning must be non-empty")
    raw_action = _compact({"tool": tool, "arguments": arguments})
    return f"<think>{reasoning.strip()}</think>\n{raw_action}"


def parse_batch_plan_assistant(
    text: str,
    *,
    max_batch_calls: int,
) -> tuple[str, str, dict]:
    """Parse one inline-think action-block turn without provider-specific fields."""
    if not isinstance(text, str) or not text.strip():
        raise BatchPlanProtocolError("inline action-block response is empty")
    match = re.fullmatch(
        r"\s*<think>(?P<reason>.*?)</think>\s*(?P<action>\{.*\})\s*",
        text,
        re.S,
    )
    if match is None or not match.group("reason").strip():
        raise BatchPlanProtocolError(
            "inline action-block response must contain one non-empty <think> block "
            "followed directly by one raw JSON action"
        )
    tool, arguments = parse_batch_plan_action(
        match.group("action"),
        max_batch_calls=max_batch_calls,
    )
    return match.group("reason").strip(), tool, arguments


def _validate_terminal_arguments(arguments: dict) -> None:
    if set(arguments) not in ({"evidence"}, {"evidence", "reason"}):
        raise BatchPlanProtocolError(
            "answer_from_context.arguments must contain evidence and optional reason"
        )
    evidence = arguments.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {"table", "columns"}:
        raise BatchPlanProtocolError(
            'answer_from_context.evidence must be exactly '
            '{"table":"result_handle","columns":["answer_column",...]}'
        )
    table = evidence.get("table")
    if not isinstance(table, str) or not table.strip():
        raise BatchPlanProtocolError(
            "answer_from_context.evidence.table must be a non-empty resident handle"
        )
    columns = evidence.get("columns")
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) or not column.strip() for column in columns)
    ):
        raise BatchPlanProtocolError(
            "answer_from_context.evidence.columns must be a non-empty list of column names"
        )
    normalized = [column.casefold() for column in columns]
    if len(normalized) != len(set(normalized)):
        raise BatchPlanProtocolError(
            "answer_from_context.evidence.columns must not contain duplicates"
        )
    if "reason" in arguments and not isinstance(arguments["reason"], str):
        raise BatchPlanProtocolError(
            "answer_from_context.reason must be a string when present"
        )


def validate_atomic_call(tool: str, arguments: dict) -> None:
    if tool not in EXECUTABLE_TOOLS:
        raise ProtocolError(
            f"unknown or non-nestable atomic tool {tool!r}; legal tools: "
            f"{list(EXECUTABLE_TOOLS)}"
        )
    validate_model_arguments(tool, arguments)


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
            raise LocalReferenceError(
                f"local column reference {value!r} targets a call with no column output"
            )
        exact = [
            column for column in columns
            if isinstance(column, str) and column.casefold() == requested.casefold()
        ]
        suffix = [
            column for column in columns
            if (
                isinstance(column, str)
                and column.rsplit(".", 1)[-1].casefold() == requested.casefold()
            )
        ]
        matches = exact or suffix
        if len(matches) != 1:
            raise LocalReferenceError(
                f"local column reference {value!r} must resolve to exactly one output column; "
                f"available columns: {columns}"
            )
        resolved = matches[0]
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
        raise LocalReferenceError(
            f"local reference {value!r} is not a table-producing call"
        )
    return table


def summarize_atomic_output(step_id: str, output: dict) -> dict:
    """Use the production resident-history compactor for one successful atomic result."""
    compacted = compact_resident_observation(tool_output_message(step_id, output))
    try:
        envelope = json.loads(compacted)
    except json.JSONDecodeError:
        return {"output": deepcopy(output)}
    return {
        key: deepcopy(value)
        for key, value in envelope.items()
        if key not in {"step_id", "status"}
    }


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
            for key in ("call_id", "step_id", "tool", "status", "dependencies")
            if result.get(key) is not None
        }
        if result.get("status") == "success":
            item.update(summarize_atomic_output(result["step_id"], result.get("output") or {}))
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
    if partial_failure and structured_error_feedback:
        reusable_outputs = {}
        for result in results:
            if result.get("status") != "success" or not result.get("table"):
                continue
            summary = summarize_atomic_output(
                result["step_id"], result.get("output") or {}
            ).get("output_summary") or {}
            reusable = {
                "step_id": result.get("step_id"),
                "table": result.get("table"),
            }
            for key in ("columns", "column_namespaces", "row_count"):
                if key in summary:
                    reusable[key] = deepcopy(summary[key])
            reusable_outputs[str(result.get("call_id"))] = reusable
    else:
        reusable_outputs = [
            {
                "call_id": result.get("call_id"),
                "step_id": result.get("step_id"),
                "table": result.get("table"),
            }
            for result in results
            if result.get("status") == "success" and result.get("table")
        ]
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
        "reusable_outputs": reusable_outputs,
        "results": visible_results,
    })


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
        if state.get("plan") or state.get("tables") or state.get("values") or last_error:
            initial += "\n\n" + state_context_message(state, last_error=last_error)
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
            observation += "\n\n" + state_context_message(
                state, last_error=last_error
            )
        messages.append({"role": "user", "content": observation})
    return messages
