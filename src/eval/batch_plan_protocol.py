#!/usr/bin/env python3
"""Sequential variable-width action-block protocol.

Work turns contain one top-level ``action_block`` with one to eight nonterminal primitive calls.
Termination is one separate top-level ``answer_from_context`` action. A single-call work block is
the safe atomic case; wider blocks are only short consecutive operations whose arguments are
already determined. They are not plans or programs. The harness owns ordered execution, handles,
resident state, dependency discovery, and blocked propagation. The model only reasons, calls
tools, and interprets the complete per-call factual feedback.
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
SEQUENTIAL_ACTION_BLOCK_PROTOCOL_VERSION = "action-block-v33"
SIMPLE_SEQUENTIAL_ACTION_BLOCK_PROTOCOL_VERSION = "action-block-v34"
SIMPLE_SCALAR_CELL_PROTOCOL_VERSION = "action-block-v35"
BATCH_PLAN_TOOL = "action_block"
TERMINAL_TOOL = "answer_from_context"
SIMPLE_JOIN_TOOL = "join"
MAX_ACTION_BLOCK_CALLS = 8
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
SEQUENTIAL_EXECUTABLE_TOOLS = tuple(
    SIMPLE_JOIN_TOOL if tool == "join_tables" else tool
    for tool in EXECUTABLE_TOOLS
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
RESIDENT_SCALAR_CELL_REF_RE = re.compile(r"^(step_[1-9][0-9]*)\.(.+)$")


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


def _public_tool_spec(
    tool: str,
    protocol_version: str,
) -> str:
    if tool == SIMPLE_JOIN_TOOL:
        return (
            "join(left, right, left_on, right_on, how?) -> attach exactly one right table "
            "to one left table/result. left_on is one exact column of left; right_on is one "
            "bare column of right. how is inner (default) or left. For a cross join, set "
            'how="cross" and omit left_on/right_on.'
        )
    if (
        tool == "scalar_compute"
        and protocol_version == SIMPLE_SCALAR_CELL_PROTOCOL_VERSION
    ):
        return (
            "scalar_compute(operation, operands, result_name?) -> derive a grounded 1x1 "
            "table. operation is add|subtract|multiply|divide|percent|percent_change|"
            "date_diff_days. operands is ordered; each operand is exactly "
            '{"value":literal}, "$earlier_id.exact_column" for an earlier one-row result '
            'in this block, or "step_id.exact_column" for a resident one-row result from '
            "an earlier block. A referenced result must have exactly one row and the named "
            "column must match exactly. Operand order is semantic."
        )
    return ACTION_BLOCK_TOOL_SPECS[tool]


def _atomic_specs(
    protocol_version: str = SIMPLE_SCALAR_CELL_PROTOCOL_VERSION,
) -> str:
    return "\n".join(
        _public_tool_spec(tool, protocol_version)
        for tool in SEQUENTIAL_EXECUTABLE_TOOLS
    )


def build_batch_plan_system_prompt(
    max_batch_calls: int = MAX_ACTION_BLOCK_CALLS,
    *,
    assistant_carrier: str = BATCH_CARRIER_PROVIDER_NATIVE,
    protocol_version: str = SIMPLE_SCALAR_CELL_PROTOCOL_VERSION,
) -> str:
    """Build the active sequential action-block provider contract."""
    if not 1 <= max_batch_calls <= MAX_ACTION_BLOCK_CALLS:
        raise ValueError(
            f"active action blocks require max_batch_calls in 1..{MAX_ACTION_BLOCK_CALLS}"
        )
    if assistant_carrier not in BATCH_CARRIERS:
        raise ValueError(
            f"unknown action-block assistant carrier {assistant_carrier!r}; "
            f"expected one of {BATCH_CARRIERS}"
        )
    tools = _atomic_specs(protocol_version)
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
    prompt = f"""You are a relational table-tool agent using short sequential action blocks.

WORKSPACE
The opening overview is a catalog, not table schemas. AVAILABLE TOOL CONTEXT is read-only factual
state maintained by the harness: observed schemas and values, resident result handles, row reads,
scalar-producing step ids, and fact-only derivations. You choose operations; the harness owns
execution, result handles, state, and dependencies.

TOOL REFERENCE — EXACT CALL SIGNATURES AND SEMANTICS
{tools}

TWO TOP-LEVEL ACTIONS
Choose exactly one action per turn.

WORK — execute a short consecutive sequence:
{{"tool":"action_block","arguments":{{"calls":[
  {{"id":"local_id","tool":"atomic_tool","arguments":{{...}}}}
]}}}}

TERMINAL — cite an exact resident result:
{{"tool":"answer_from_context","arguments":
  {{"evidence":{{"table":"exact_result_handle"}},"reason":"brief optional reason"}}}}

SEQUENTIAL RULES
1. calls contains 1 to {max_batch_calls} entries. Each entry has exactly id, tool, arguments. ids
   are unique, begin with a letter, and exist only inside this action block. Calls execute from top
   to bottom. This is a short operation sequence, not a program: do not declare a DAG, dependencies,
   exports, a result root, plans, statuses, handles, or environment state.
2. Combine calls only while the next call is fully determined now. A missing result handle is fine:
   refer to an earlier call with "$id". But if you must inspect a new schema, value, row, or error
   before choosing the next tool or any of its arguments, end the block and wait for feedback.
   A one-call block is always valid.
3. "$id" refers to an earlier call's result table in a table argument. "$id.column" refers to one
   exact output column. In scalar_compute operands, "$id.column" reads that cell only when the
   earlier result has exactly one row; in a later turn use "step_id.column".
   References must point backward in the same block. For table arguments in a later turn, use the
   exact resident handle.
4. The harness returns one ordered results[] entry for every submitted call. Each entry contains
   that atomic call's complete output, exact error, or blocked cause. A failed call is attempted;
   only later calls that depend on it are blocked. Independent later calls still execute.
5. Use only the exact signatures above; the harness does not repair an argument. answer_from_context
   is never inside calls. Cite it only on a later turn after its resident table
   already has exactly the requested rows, columns, column order, grain, ordering, and duplicates.
   The reason cannot alter the cited data. If helper columns remain, derive a final project first.

{format_rule}

ONE BEST-PRACTICE EXAMPLE
Assume both schemas and the literal "EU" are already grounded, so one join edge, a filter,
projection, and row inspection are consecutive and need no unseen choice:
{{"tool":"action_block","arguments":{{"calls":[
  {{"id":"linked","tool":"join","arguments":{{"left":"orders","right":"customers",
    "left_on":"customer_id","right_on":"id"}}}},
  {{"id":"filtered","tool":"condition_filter","arguments":{{"table":"$linked",
    "conditions":{{"column":"customers.region","op":"=","value":"EU"}}}}}},
  {{"id":"exact","tool":"project","arguments":{{"table":"$filtered",
    "expressions":["order_id"],"distinct":true}}}},
  {{"id":"rows","tool":"read_subtable","arguments":{{"table":"$exact","limit":20}}}}
]}}}}

The environment returns four results[] entries in this same order, including the complete output
of linked, filtered, exact, and rows. After reading that feedback, cite the returned project handle in a
separate terminal turn. If the filter literal were unknown, the best practice would instead be a
one-call inspect_column block, followed by a new decision after its output.
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
            "complete terminal JSON after reasoning. Put every parameter inside arguments. Do not "
            "add prose, Markdown, XML tags, a second action, or text before or after the JSON."
        )
    return prompt


def batch_plan_protocol_hash(
    system_prompt: str,
    max_batch_calls: int,
    *,
    protocol_version: str = UNIFIED_ACTION_BLOCK_PROTOCOL_VERSION,
) -> str:
    public_tools = (
        SEQUENTIAL_EXECUTABLE_TOOLS
        if protocol_version in {
            SIMPLE_SEQUENTIAL_ACTION_BLOCK_PROTOCOL_VERSION,
            SIMPLE_SCALAR_CELL_PROTOCOL_VERSION,
        }
        else EXECUTABLE_TOOLS
    )
    payload = {
        "version": protocol_version,
        "system_prompt": system_prompt,
        "max_batch_calls": max_batch_calls,
        "atomic_tools": {
            tool: _public_tool_spec(tool, protocol_version)
            for tool in public_tools
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


def validate_sequential_atomic_call(tool: str, arguments: dict) -> None:
    """Validate the v34 primitive surface without changing frozen atomic semantics."""
    if tool != SIMPLE_JOIN_TOOL:
        validate_atomic_call(tool, arguments)
        return
    if not isinstance(arguments, dict):
        raise ProtocolError("join.arguments must be an object")
    required = {"left", "right"}
    allowed = required | {"left_on", "right_on", "how"}
    missing = required - set(arguments)
    unexpected = set(arguments) - allowed
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected {sorted(unexpected)}")
        raise ProtocolError("join arguments: " + "; ".join(details))
    for key in ("left", "right"):
        if not isinstance(arguments.get(key), str) or not arguments[key].strip():
            raise ProtocolError(f"join.{key} must be a non-empty table or result reference")
    how = arguments.get("how", "inner")
    if how not in {"inner", "left", "cross"}:
        raise ProtocolError("join.how must be inner, left, or cross")
    has_left_on = "left_on" in arguments
    has_right_on = "right_on" in arguments
    if how == "cross":
        if has_left_on or has_right_on:
            raise ProtocolError(
                "join with how=cross must omit left_on and right_on"
            )
        return
    if not has_left_on or not has_right_on:
        raise ProtocolError(
            "join requires left_on and right_on unless how=cross"
        )
    for key in ("left_on", "right_on"):
        if not isinstance(arguments.get(key), str) or not arguments[key].strip():
            raise ProtocolError(f"join.{key} must be a non-empty column name")
    if "." in arguments["right_on"]:
        raise ProtocolError("join.right_on must be a bare column of the right table")


def lower_sequential_atomic_call(tool: str, arguments: dict) -> tuple[str, dict]:
    """Deterministically lower one simple public join edge to the frozen executor."""
    if tool != SIMPLE_JOIN_TOOL:
        return tool, deepcopy(arguments)
    validate_sequential_atomic_call(tool, arguments)
    left = arguments["left"]
    left_on = arguments.get("left_on")
    how = arguments.get("how", "inner")
    if how == "cross":
        join_item = {
            "table": arguments["right"],
            "on": [],
            "type": "cross",
        }
    else:
        if left.startswith("$"):
            public_left = f"{left}.{left_on}"
        elif "." in str(left_on):
            public_left = left_on
        else:
            public_left = f"{left}.{left_on}"
        join_item = {
            "table": arguments["right"],
            "on": [{
                "left": public_left,
                "right": arguments["right_on"],
            }],
        }
        if how != "inner":
            join_item["type"] = how
    return "join_tables", {
        "base": left,
        "joins": [join_item],
    }


def publicize_sequential_error(message: str) -> str:
    """Map private join executor paths back to the v34 one-edge public surface."""
    return (
        str(message)
        .replace("join_tables.joins[0].on[0].left", "join.left_on")
        .replace("join_tables.joins[0].on[0].right", "join.right_on")
        .replace("join_tables.joins[0].on", "join keys")
        .replace("join_tables", "join")
    )


def publicize_simple_scalar_error(message: str) -> str:
    """Keep private scalar executor argument carriers out of v35 feedback."""
    return (
        publicize_sequential_error(message)
        .replace(
            "value_ref must cite a scalar-producing step",
            "a scalar cell operand must cite an earlier successful one-row result and exact "
            "column",
        )
        .replace(
            "value_ref",
            "scalar cell reference",
        )
    )


def prepare_simple_scalar_cell_arguments(
    tool: str,
    arguments: dict,
    *,
    bindings: dict[str, dict],
    declared_ids: set[str],
    ctx: dict,
    resolutions: list[dict] | None = None,
) -> dict:
    """Lower v35 public scalar cell strings to the frozen internal operand carrier.

    This adapter is deliberately strict: it verifies successful production, exact one-row
    cardinality, and an exact column name. It never chooses a row, resolves a suffix, computes an
    aggregate, or guesses a producer.
    """
    if tool != "scalar_compute":
        return deepcopy(arguments)
    if not isinstance(arguments, dict):
        raise ProtocolError("scalar_compute.arguments must be an object")
    operands = arguments.get("operands")
    if not isinstance(operands, list) or not operands:
        # The frozen validator owns the exact arity rules; this message only guards the public
        # operand carrier before local-reference resolution can reinterpret it.
        raise ProtocolError("scalar_compute.operands must be a non-empty ordered list")

    lowered = deepcopy(arguments)
    lowered_operands = []
    history = ctx.get("history") or {}
    for index, operand in enumerate(operands):
        where = f"scalar_compute.operands[{index}]"
        if isinstance(operand, dict):
            if set(operand) != {"value"}:
                raise ProtocolError(
                    f'{where} must be exactly {{"value":literal}}, '
                    '"$earlier_id.exact_column", or "step_id.exact_column"'
                )
            lowered_operands.append(deepcopy(operand))
            continue
        if not isinstance(operand, str):
            raise ProtocolError(
                f'{where} must be exactly {{"value":literal}}, '
                '"$earlier_id.exact_column", or "step_id.exact_column"'
            )

        local_match = LOCAL_COLUMN_REF_RE.fullmatch(operand)
        resident_match = RESIDENT_SCALAR_CELL_REF_RE.fullmatch(operand)
        if local_match:
            call_id, column = local_match.groups()
            if call_id not in declared_ids:
                raise LocalReferenceError(
                    f"scalar cell reference {operand!r} names no call in this block"
                )
            binding = bindings.get(call_id)
            if binding is None:
                raise LocalReferenceError(
                    f"scalar cell reference {operand!r} is forward; only earlier calls may be "
                    "referenced"
                )
            if binding.get("status") != "success":
                raise LocalReferenceError(
                    f"scalar cell reference {operand!r} depends on a "
                    f"{binding.get('status')} call"
                )
            step_id = binding.get("step_id")
            row_count = binding.get("row_count")
            columns = binding.get("columns")
            internal_reference = f"${call_id}"
            source_kind = "same_block_one_cell"
        elif resident_match:
            step_id, column = resident_match.groups()
            record = history.get(step_id)
            if not isinstance(record, dict):
                raise LocalReferenceError(
                    f"scalar cell reference {operand!r} names no resident producing step"
                )
            output = record.get("output")
            if not isinstance(output, dict):
                raise LocalReferenceError(
                    f"scalar cell reference {operand!r} targets a step with no table output"
                )
            row_count = output.get("row_count")
            columns = output.get("columns")
            internal_reference = step_id
            source_kind = "resident_one_cell"
        else:
            raise ProtocolError(
                f'{where} must be exactly {{"value":literal}}, '
                '"$earlier_id.exact_column", or "step_id.exact_column"'
            )

        if row_count != 1:
            raise LocalReferenceError(
                f"scalar cell reference {operand!r} requires exactly one source row; "
                f"observed row_count={row_count!r}"
            )
        if not isinstance(columns, list) or column not in columns:
            raise LocalReferenceError(
                f"scalar cell reference {operand!r} must exactly match one output column; "
                f"available columns: {columns}"
            )
        if not isinstance(step_id, str) or not step_id:
            raise LocalReferenceError(
                f"scalar cell reference {operand!r} has no producing step id"
            )
        lowered_operand = {
            "value_ref": internal_reference,
            "column": column,
        }
        lowered_operands.append(lowered_operand)
        if resolutions is not None:
            resolutions.append({
                "path": f"operands.{index}",
                "provided": operand,
                "resolved": {
                    "producing_step": step_id,
                    "column": column,
                },
                "rule": source_kind,
            })
    lowered["operands"] = lowered_operands
    return lowered


def _publicize_sequential_derivations(value: Any) -> Any:
    visible = deepcopy(value)

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if (
                item.get("schema") == "relation-derivation-v1"
                and item.get("operator") == "join_tables"
            ):
                item["operator"] = SIMPLE_JOIN_TOOL
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(visible)
    return visible


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


def render_sequential_observation(
    batch_index: int,
    results: list[dict],
    *,
    structured_error_feedback: bool = False,
) -> str:
    """Render v34 feedback using only its public one-edge join vocabulary."""
    visible = _publicize_sequential_derivations(results)
    for result in visible:
        error = result.get("error")
        if isinstance(error, dict) and error.get("message"):
            error["message"] = publicize_sequential_error(error["message"])
    return render_batch_observation(
        batch_index,
        visible,
        structured_error_feedback=structured_error_feedback,
    )


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


def build_sequential_messages(**kwargs) -> list[dict]:
    """Build v34 context without leaking the private multi-edge join executor name."""
    updated = dict(kwargs)
    updated["state"] = _publicize_sequential_derivations(
        kwargs.get("state") or {}
    )
    if isinstance(kwargs.get("last_error"), dict):
        last_error = _publicize_sequential_derivations(kwargs["last_error"])
        error = last_error.get("error")
        if isinstance(error, dict) and error.get("message"):
            error["message"] = publicize_sequential_error(error["message"])
        updated["last_error"] = last_error
    return build_batch_plan_messages(**updated)


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
