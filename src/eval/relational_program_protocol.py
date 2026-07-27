#!/usr/bin/env python3
"""Relational-program tool scheme with harness-derived dependency graphs.

Perception remains interactive at the top level. Deterministic relational primitives may be
submitted as one declarative program whose dependencies are inferred exclusively from exact
``$node`` and ``$node.column`` parameter references. The harness validates and topologically
orders the graph; the model never authors edges, statuses, handles, or an execution schedule.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from action_carrier import (
    ACTIVE_ACTION_CARRIER,
    ActionCarrierError,
    parse_action_carrier,
    render_action_carrier,
)
from batch_plan_protocol import (
    BATCH_CARRIERS,
    BATCH_CARRIER_INLINE_THINK,
    BATCH_CARRIER_PROVIDER_NATIVE,
    CALL_ID_RE,
    LOCAL_COLUMN_REF_RE,
    LOCAL_REF_RE,
    TERMINAL_TOOL,
    BatchPlanProtocolError,
    atomic_output_for_model,
    build_batch_plan_messages,
)
from protocol import ProtocolError, validate_model_arguments


RELATIONAL_PROGRAM_PROTOCOL_VERSION = "relational-program-v3"
OBSERVE_TOOL = "observe"
RELATIONAL_PROGRAM_TOOL = "relational_program"
MAX_RELATIONAL_PROGRAM_CALLS = 8

OBSERVE_OPERATIONS = {
    "schema": "describe_table",
    "column": "inspect_column",
    "rows": "read_subtable",
}
PROGRAM_OPERATIONS = {
    "filter": "condition_filter",
    "select": "project",
    "scalar": "scalar_compute",
    "join": "join_tables",
    "aggregate": "group_aggregate",
    "rank": "extreme_value_select",
    "combine": "set_op",
}
UNDERLYING_TO_PUBLIC_OPERATION = {
    **{tool: f"{OBSERVE_TOOL}.{operation}" for operation, tool in OBSERVE_OPERATIONS.items()},
    **{tool: operation for operation, tool in PROGRAM_OPERATIONS.items()},
}
PERCEPTION_TOOLS = tuple(OBSERVE_OPERATIONS.values())
RELATIONAL_PRIMITIVE_TOOLS = (
    *PROGRAM_OPERATIONS.values(),
)
TOP_LEVEL_TOOLS = (
    OBSERVE_TOOL,
    RELATIONAL_PROGRAM_TOOL,
    TERMINAL_TOOL,
)


class RelationalProgramProtocolError(BatchPlanProtocolError):
    """The relational-program carrier or static graph is invalid."""


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def build_relational_program_system_prompt(
    max_program_calls: int = MAX_RELATIONAL_PROGRAM_CALLS,
    *,
    assistant_carrier: str = BATCH_CARRIER_PROVIDER_NATIVE,
) -> str:
    """Build the exclusive model-facing contract for relational-program v3."""
    if not 1 <= max_program_calls <= MAX_RELATIONAL_PROGRAM_CALLS:
        raise ValueError(
            "active relational programs require max_program_calls in "
            f"1..{MAX_RELATIONAL_PROGRAM_CALLS}"
        )
    if assistant_carrier not in BATCH_CARRIERS:
        raise ValueError(
            f"unknown relational-program assistant carrier {assistant_carrier!r}; "
            f"expected one of {BATCH_CARRIERS}"
        )
    if assistant_carrier == BATCH_CARRIER_PROVIDER_NATIVE:
        response_rule = (
            "Put one non-empty reason in the provider-native reasoning field. The entire visible "
            'response is one JSON object with only "tool" and "arguments"; JSON Output is enabled.'
        )
    else:
        response_rule = (
            "Output exactly one non-empty <think>brief reason</think> block followed directly by "
            'one raw JSON object with only "tool" and "arguments".'
        )
    prompt = f"""You are a relational table agent. This protocol has exactly three top-level tools:
observe, relational_program, and answer_from_context. No other tool can be called directly.

AVAILABLE FACTS
The opening overview is a catalog, not a schema. AVAILABLE TOOL CONTEXT is harness-maintained
factual state: observed schemas/values, resident table handles, row reads, producing step ids, and
fact-only derivations. You never author SQL, handles, step ids, graph edges, execution status, or
environment state.

TOP-LEVEL TOOL 1: observe
Use observe when the next transformation depends on information not yet visible.
- Schema: {{"tool":"observe","arguments":{{"operation":"schema","tables":["table"]}}}}
- Column domain: {{"tool":"observe","arguments":
  {{"operation":"column","table":"table","column":"column","top_k":10}}}}
- Rows: {{"tool":"observe","arguments":
  {{"operation":"rows","table":"table_or_handle","columns":["column"],"limit":10}}}}
Schema requires a non-empty source-table list. Column inspects one column. Rows reads 1..20 rows;
omit columns to read all columns. Observation never reshapes or creates a result table.

TOP-LEVEL TOOL 2: relational_program
Submit one declarative program:
{{"tool":"relational_program","arguments":{{
  "calls":[{{"id":"node_id","operation":"operation_name","arguments":{{...}}}}],
  "result":"primary_result_id",
  "exports":["optional_additional_result_id"]
}}}}

Each node operation is one of the following variants. These are node operations, not top-level
tools, and they can appear only inside relational_program.calls:
- filter(table, conditions, return_columns?): retain matching rows. A predicate uses column plus
  op =|!=|>|>=|<|<= and value or column_value; op in uses values or in_table; between uses low/high;
  like or contains uses value; is_null needs no value. Compose with and/or/not.
- select(table, expressions, distinct?): derive exactly the listed output expressions and order;
  expressions may use "expr AS alias". distinct defaults false.
- scalar(operation, operands, result_name?): derive one grounded 1x1 table. operation is
  add|subtract|multiply|divide|percent|percent_change|date_diff_days. Each operand is exactly value,
  value_ref, or value_ref+column; operand order is semantic.
- join(base, joins, base_role?): derive one connected join component. joins is an ordered non-empty
  list. Each item has table, on, optional type inner|left|cross, and optional role. Each on pair has
  left as an exact logical relation.column already present and right as a bare column of the newly
  attached table; cross uses on=[].
- aggregate(table, group_by, aggregations, passthrough?, output_layout?, category_values?,
  output_columns?): derive grouped results. group_by=[] is one global group. Each aggregation has
  op sum|count|count_distinct|mean|min|max, column, as, and optional where predicate.
- rank(table, order_by, top_k?, return_columns?): order rows by a list of columns optionally ending
  in DESC and retain top_k when supplied.
- combine(left, right, op): derive union|union_all|intersect|except over compatible tables.

TOP-LEVEL TOOL 3: answer_from_context
Terminate with one already resident exact result table:
{{"tool":"answer_from_context","arguments":
  {{"evidence":{{"table":"exact_resident_handle"}},"reason":"brief optional reason"}}}}

PROGRAM RULES
1. calls contains 1 to {max_program_calls} deterministic relational calls. Every call has exactly
   id, operation, arguments. ids are unique and begin with a letter.
2. Express parameter dependency only with exact "$id" or "$id.column" values. "$id" denotes that
   node's produced table, or its producing step when used as value_ref. "$id.column" denotes one
   exact output column. List order is not execution order: the harness derives the DAG from these
   references, rejects undefined references/cycles, and runs a stable topological order.
   Local references exist only inside the current program. Never reuse "$id" in a later turn;
   later programs copy the exact resident handle or producing step id returned by the harness.
3. result names the program's primary result node. exports optionally names other required output
   roots. Every submitted node must contribute to result or an export; disconnected work is
   rejected. Omit exports when there are no additional roots; if independent branches are
   intentionally retained, list each branch root in exports. Do not provide dependency lists,
   statuses, handles, or a schedule.
4. A failed primitive is a root error. Its transitive descendants are blocked and not executed;
   independent nodes continue. Every node returns its own full output, error, or blocked record.
5. Observation is intentionally outside the program. If an intermediate schema, value, row, or
   error could change the next relational choice, end the program, use observe on the returned
   resident handle, then submit another program. Programs may be used multiple times in one
   episode and may consume exact resident handles from earlier turns.
6. Use each operation's arguments exactly. Preserve population, grain, duplicates, and
   column order. The harness performs no spelling, schema, predicate, column, or argument repair.
   In join, joins[].on is always a list of pair objects. If base is "$filtered", write
   on.left as "$filtered.exact_column"; never write "filtered.exact_column" or the old source-table
   namespace. on.right is always the bare column of the newly attached table. If base is a source
   table or a resident handle from an earlier turn, copy its exact visible logical namespace.
7. answer_from_context is never nested. Its cited table must already have exactly the requested
   rows and columns. Observing a table does not reshape it; use select for the exact result before
   terminating.

EXAMPLE AFTER SCHEMA AND LITERALS ARE GROUNDED
{{"tool":"relational_program","arguments":{{
  "calls":[
    {{"id":"exact","operation":"select","arguments":
      {{"table":"$filtered","expressions":["order_id"],"distinct":true}}}},
    {{"id":"filtered","operation":"filter","arguments":
      {{"table":"orders","conditions":{{"column":"amount","op":">","value":100}}}}}}
  ],
  "result":"exact"
}}}}

LOCAL-RESULT JOIN EXAMPLE
{{"tool":"relational_program","arguments":{{
  "calls":[
    {{"id":"joined","operation":"join","arguments":{{
      "base":"$filtered",
      "joins":[{{"table":"customers","on":[
        {{"left":"$filtered.customer_id","right":"id"}}
      ]}}]
    }}}},
    {{"id":"filtered","operation":"filter","arguments":{{
      "table":"orders","conditions":{{"column":"status","op":"=","value":"open"}}
    }}}},
    {{"id":"exact","operation":"select","arguments":{{
      "table":"$joined","expressions":["customers.name"],"distinct":true
    }}}}
  ],
  "result":"exact"
}}}}

RESPONSE FORMAT
{response_rule}
Do not add prose, Markdown, XML, a second action, or text after the action object.
"""
    if assistant_carrier == BATCH_CARRIER_PROVIDER_NATIVE:
        prompt += (
            "\nKeep native reasoning under 120 words and reserve output budget for the action. "
            "If uncertain, emit one legal perception action rather than extending analysis. "
            "Native reasoning is not the visible response; always emit the complete JSON action."
        )
    return prompt


def relational_program_protocol_hash(
    system_prompt: str,
    max_program_calls: int,
    *,
    protocol_version: str = RELATIONAL_PROGRAM_PROTOCOL_VERSION,
) -> str:
    payload = {
        "version": protocol_version,
        "system_prompt": system_prompt,
        "max_program_calls": max_program_calls,
        "top_level_tools": TOP_LEVEL_TOOLS,
        "observe_operations": OBSERVE_OPERATIONS,
        "program_operations": PROGRAM_OPERATIONS,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _all_local_references(value: Any) -> list[str]:
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
        if not match:
            raise RelationalProgramProtocolError(
                f"local reference {item!r} must be exactly $id or $id.column"
            )
        if match.group(1) not in found:
            found.append(match.group(1))

    visit(value)
    return found


def compile_relational_program(
    arguments: dict,
    *,
    max_program_calls: int = MAX_RELATIONAL_PROGRAM_CALLS,
) -> tuple[dict, dict]:
    """Validate a program and return action-block-shaped calls in topological order."""
    if not isinstance(arguments, dict):
        raise RelationalProgramProtocolError(
            "relational_program.arguments must be an object"
        )
    if not 1 <= max_program_calls <= MAX_RELATIONAL_PROGRAM_CALLS:
        raise RelationalProgramProtocolError(
            "active relational programs require max_program_calls in "
            f"1..{MAX_RELATIONAL_PROGRAM_CALLS}"
        )
    allowed = {"calls", "result", "exports"}
    if "calls" not in arguments or "result" not in arguments:
        raise RelationalProgramProtocolError(
            'relational_program.arguments requires "calls" and "result"'
        )
    unexpected = sorted(set(arguments) - allowed)
    if unexpected:
        raise RelationalProgramProtocolError(
            f"relational_program.arguments has unexpected keys: {unexpected}"
        )
    calls = arguments.get("calls")
    if not isinstance(calls, list) or not 1 <= len(calls) <= max_program_calls:
        raise RelationalProgramProtocolError(
            f"relational_program.calls must contain 1 to {max_program_calls} calls"
        )
    result = arguments.get("result")
    if not isinstance(result, str) or not CALL_ID_RE.fullmatch(result):
        raise RelationalProgramProtocolError(
            "relational_program.result must be one valid call id"
        )
    exports = arguments.get("exports", [])
    if not isinstance(exports, list) or any(
        not isinstance(item, str) or not CALL_ID_RE.fullmatch(item)
        for item in exports
    ):
        raise RelationalProgramProtocolError(
            "relational_program.exports must be a list of valid call ids"
        )
    if len(set(exports)) != len(exports):
        raise RelationalProgramProtocolError(
            "relational_program.exports must not contain duplicates"
        )

    by_id: dict[str, dict] = {}
    original_index: dict[str, int] = {}
    for index, call in enumerate(calls):
        where = f"relational_program.calls[{index}]"
        if not isinstance(call, dict) or set(call) != {
            "id",
            "operation",
            "arguments",
        }:
            raise RelationalProgramProtocolError(
                f"{where} must contain exactly id, operation, and arguments"
            )
        call_id = call.get("id")
        if not isinstance(call_id, str) or not CALL_ID_RE.fullmatch(call_id):
            raise RelationalProgramProtocolError(
                f"{where}.id must begin with a letter and contain at most 32 letters, "
                "digits, or underscores"
            )
        if call_id in by_id:
            raise RelationalProgramProtocolError(
                f"{where}.id duplicates {call_id!r}"
            )
        operation = call.get("operation")
        if operation not in PROGRAM_OPERATIONS:
            raise RelationalProgramProtocolError(
                f"{where}.operation must be one of "
                f"{list(PROGRAM_OPERATIONS)}"
            )
        if not isinstance(call.get("arguments"), dict):
            raise RelationalProgramProtocolError(
                f"{where}.arguments must be an object"
            )
        by_id[call_id] = {
            "id": call_id,
            "tool": PROGRAM_OPERATIONS[operation],
            "arguments": deepcopy(call["arguments"]),
        }
        original_index[call_id] = index

    roots = [result, *exports]
    missing_roots = [root for root in roots if root not in by_id]
    if missing_roots:
        raise RelationalProgramProtocolError(
            f"program result/exports name undefined calls: {missing_roots}"
        )

    dependencies: dict[str, list[str]] = {}
    for call_id, call in by_id.items():
        refs = _all_local_references(call["arguments"])
        missing = [ref for ref in refs if ref not in by_id]
        if missing:
            raise RelationalProgramProtocolError(
                f"call {call_id!r} references undefined calls: {missing}"
            )
        dependencies[call_id] = refs

    pending = set(by_id)
    scheduled: list[str] = []
    while pending:
        ready = sorted(
            (
                call_id
                for call_id in pending
                if all(dep in scheduled for dep in dependencies[call_id])
            ),
            key=original_index.__getitem__,
        )
        if not ready:
            cycle_nodes = sorted(pending, key=original_index.__getitem__)
            raise RelationalProgramProtocolError(
                f"relational program contains a dependency cycle: {cycle_nodes}"
            )
        for call_id in ready:
            scheduled.append(call_id)
            pending.remove(call_id)

    reachable: set[str] = set()

    def mark(call_id: str) -> None:
        if call_id in reachable:
            return
        reachable.add(call_id)
        for dependency in dependencies[call_id]:
            mark(dependency)

    for root in roots:
        mark(root)
    orphaned = [
        call_id for call_id in scheduled if call_id not in reachable
    ]
    if orphaned:
        raise RelationalProgramProtocolError(
            "every program call must contribute to result or exports; "
            f"disconnected calls: {orphaned}"
        )

    graph = {
        "schema": "relational-program-graph-v1",
        "dependencies": {
            call_id: dependencies[call_id] for call_id in scheduled
        },
        "topological_order": scheduled,
        "result": result,
        "exports": list(exports),
        "authored_operations": {
            call["id"]: call["operation"] for call in calls
        },
    }
    return {"calls": [by_id[call_id] for call_id in scheduled]}, graph


def validate_relational_program_call(tool: str, arguments: dict) -> None:
    if tool not in (*PERCEPTION_TOOLS, *RELATIONAL_PRIMITIVE_TOOLS):
        raise ProtocolError(
            f"unknown relational-program primitive {tool!r}"
        )
    try:
        validate_model_arguments(tool, arguments)
    except ProtocolError as exc:
        public_name = UNDERLYING_TO_PUBLIC_OPERATION.get(tool, tool)
        raise ProtocolError(str(exc).replace(tool, public_name)) from exc


def _prepare_observe(arguments: dict) -> tuple[str, dict]:
    if not isinstance(arguments, dict):
        raise RelationalProgramProtocolError("observe.arguments must be an object")
    operation = arguments.get("operation")
    if operation not in OBSERVE_OPERATIONS:
        raise RelationalProgramProtocolError(
            f"observe.operation must be one of {list(OBSERVE_OPERATIONS)}"
        )
    forwarded = {
        key: deepcopy(value)
        for key, value in arguments.items()
        if key != "operation"
    }
    tool = OBSERVE_OPERATIONS[operation]
    try:
        validate_relational_program_call(tool, forwarded)
    except ProtocolError as exc:
        raise RelationalProgramProtocolError(str(exc)) from exc
    return tool, forwarded


def prepare_relational_work_action(
    tool: str,
    arguments: dict,
    *,
    max_program_calls: int = MAX_RELATIONAL_PROGRAM_CALLS,
) -> tuple[dict, dict]:
    """Convert one direct perception or declarative program into executable calls."""
    if tool == OBSERVE_TOOL:
        perception_tool, perception_arguments = _prepare_observe(arguments)
        return {
            "calls": [{
                "id": "observation",
                "tool": perception_tool,
                "arguments": perception_arguments,
            }]
        }, {
            "schema": "relational-program-graph-v1",
            "kind": "interactive_perception",
            "dependencies": {"observation": []},
            "topological_order": ["observation"],
            "result": None,
            "exports": [],
            "authored_operations": {
                "observation": f"{OBSERVE_TOOL}.{arguments['operation']}"
            },
        }
    if tool != RELATIONAL_PROGRAM_TOOL:
        raise RelationalProgramProtocolError(
            "nonterminal top-level tool must be observe or relational_program"
        )
    return compile_relational_program(
        arguments,
        max_program_calls=max_program_calls,
    )


def parse_relational_program_action(
    raw_content: str,
    *,
    max_batch_calls: int = MAX_RELATIONAL_PROGRAM_CALLS,
) -> tuple[str, dict]:
    """Strictly parse one provider-visible action without repair."""
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise RelationalProgramProtocolError("visible response is empty")
    try:
        action = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise RelationalProgramProtocolError(
            f"visible response is not one complete JSON object: {exc}"
        ) from exc
    if not isinstance(action, dict) or set(action) != {"tool", "arguments"}:
        raise RelationalProgramProtocolError(
            'action must contain exactly "tool" and "arguments"'
        )
    tool = action.get("tool")
    arguments = action.get("arguments")
    if tool not in TOP_LEVEL_TOOLS:
        raise RelationalProgramProtocolError(
            f"top-level tool must be one of {list(TOP_LEVEL_TOOLS)}"
        )
    if not isinstance(arguments, dict):
        raise RelationalProgramProtocolError("action.arguments must be an object")
    if tool == RELATIONAL_PROGRAM_TOOL:
        compile_relational_program(
            arguments,
            max_program_calls=max_batch_calls,
        )
    elif tool == OBSERVE_TOOL:
        _prepare_observe(arguments)
    else:
        try:
            validate_model_arguments(tool, arguments)
        except ProtocolError as exc:
            raise RelationalProgramProtocolError(str(exc)) from exc
    return tool, arguments


def render_relational_program_assistant(
    reasoning: str,
    tool: str,
    arguments: dict,
) -> str:
    return render_action_carrier(reasoning, tool, arguments)


def parse_relational_program_assistant(
    text: str,
    *,
    max_batch_calls: int = MAX_RELATIONAL_PROGRAM_CALLS,
) -> tuple[str, str, dict]:
    try:
        reasoning, action = parse_action_carrier(text)
    except ActionCarrierError as exc:
        raise RelationalProgramProtocolError(str(exc)) from exc
    tool, arguments = parse_relational_program_action(
        _compact(action),
        max_batch_calls=max_batch_calls,
    )
    return reasoning, tool, arguments


def _publicize_exact_operation_names(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _publicize_exact_operation_names(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _publicize_exact_operation_names(child)
            for child in value
        ]
    if isinstance(value, str):
        return UNDERLYING_TO_PUBLIC_OPERATION.get(value, value)
    return deepcopy(value)


def build_relational_program_messages(**kwargs: Any) -> list[dict]:
    public_kwargs = deepcopy(kwargs)
    for key in ("state", "last_error", "legal_history"):
        if key in public_kwargs:
            public_kwargs[key] = _publicize_exact_operation_names(
                public_kwargs[key]
            )
    return build_batch_plan_messages(**public_kwargs)


def render_relational_program_observation(
    program_index: int,
    results: list[dict],
    *,
    structured_error_feedback: bool = False,
) -> str:
    del structured_error_feedback
    visible_results = []
    for result in results:
        item = {
            key: deepcopy(result.get(key))
            for key in ("call_id", "step_id", "status")
            if result.get(key) is not None
        }
        underlying = result.get("tool")
        if underlying in UNDERLYING_TO_PUBLIC_OPERATION:
            item["operation"] = UNDERLYING_TO_PUBLIC_OPERATION[underlying]
        if result.get("status") == "success":
            item["output"] = _publicize_exact_operation_names(
                atomic_output_for_model(
                    result["step_id"], result.get("output") or {}
                )
            )
        elif result.get("status") == "error":
            item["error"] = deepcopy(result.get("error") or {})
        else:
            item["blocked_by"] = deepcopy(result.get("blocked_by") or [])
            item["root_causes"] = deepcopy(result.get("root_causes") or [])
            item["reason"] = str(
                result.get("reason")
                or "a required earlier node did not succeed"
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
    partial_failure = bool(
        counts.get("error", 0) or counts.get("blocked", 0)
    )
    return "RELATIONAL PROGRAM RESULTS\n" + _compact({
        "program_index": program_index,
        "program_status": (
            "partial_failure" if partial_failure else "success"
        ),
        "counts": counts,
        "root_error_nodes": [
            item.get("call_id")
            for item in visible_results
            if item.get("status") == "error"
        ],
        "results": visible_results,
    })


ASSISTANT_CARRIERS = BATCH_CARRIERS
PROVIDER_NATIVE_CARRIER = BATCH_CARRIER_PROVIDER_NATIVE
INLINE_THINK_CARRIER = BATCH_CARRIER_INLINE_THINK
ACTION_CARRIER = ACTIVE_ACTION_CARRIER
