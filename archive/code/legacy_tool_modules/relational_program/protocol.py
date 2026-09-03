#!/usr/bin/env python3
"""Relational-program tool scheme with harness-derived dependency graphs.

Perception remains interactive at the top level. Deterministic relational primitives may be
submitted as one declarative program whose dependencies are inferred exclusively from typed
node-reference objects. Source relations, prior resident outputs, and current-program nodes have
disjoint public shapes. The harness validates and topologically orders the graph; the model never
authors edges, statuses, handles, or an execution schedule.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

from tool_modules._bootstrap import activate_legacy_paths

activate_legacy_paths()

from action_carrier import (
    ACTIVE_ACTION_CARRIER,
    ActionCarrierError,
    parse_action_carrier,
    render_action_carrier,
)
from tool_modules.action_block.protocol import (
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


RELATIONAL_PROGRAM_PROTOCOL_VERSION = "relational-program-v6"
OBSERVE_TOOL = "observe"
RELATIONAL_PROGRAM_TOOL = "relational_program"
MAX_RELATIONAL_PROGRAM_CALLS = 8
TYPED_REFERENCE_SCHEMA = "typed-relational-reference-v2"

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
    """Build the exclusive model-facing contract for relational-program v6."""
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
  op =|!=|>|>=|<|<= and exactly one of value, column_value, or value_from; op in uses values or
  in_table; between uses low/high; like or contains uses value; is_null needs no value. Compose with
  and/or/not. value_from is a typed node or resident_step reference to a grounded 1x1 result.
- select(table, expressions, distinct?): derive exactly the listed output expressions and order;
  expressions may use "expr AS alias". distinct defaults false.
- scalar(operation, operands, result_name?): derive one grounded 1x1 table. operation is
  add|subtract|multiply|divide|percent|percent_change|date_diff_days. Each operand is exactly
  {{"value":scalar}}, one typed {{"node":"..."}} / {{"resident_step":"..."}} reference, or that
  typed reference plus "column" for a named cell; operand order is semantic.
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
2. Every relation or step reference is one typed JSON object. Never use a bare string in table,
   base, joins[].table, combine left/right, in_table, value_from, or a scalar operand:
   - {{"source_table":"orders"}} means one immutable catalog table.
   - {{"resident_table":"filter_002"}} means one exact table handle returned before this program.
   - {{"node":"filtered"}} means the table produced by one node in this program.
   - {{"resident_step":"step_7"}} means one producing step id returned before this program.
   - In scalar operands, {{"node":"metrics","column":"metric"}} and
     {{"resident_step":"step_7","column":"metric"}} select one named cell from a one-row result.
   - In a filter/aggregate predicate, use "value_from":{{"node":"scalar_node"}} or
     "value_from":{{"resident_step":"step_7"}} for one grounded 1x1 comparison value.
   - {{"node":"filtered","column":"customer_id"}} is valid only as join on.left and means one exact
     output column of that current-program node.
   List order is not execution order: the harness derives the DAG only from node references,
   rejects undefined references/cycles, and runs a stable topological order. Node references exist
   only inside the current program. A later program must use the returned resident_table or
   resident_step identity, never an earlier node id.
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
   In join, joins[].on is always a list of pair objects. If base is {{"node":"filtered"}}, write
   on.left as {{"node":"filtered","column":"exact_column"}}. on.right is always the bare column of
   the newly attached table. If base is a source table or a resident table from an earlier turn,
   on.left remains the exact visible logical namespace string such as "orders.customer_id" or
   "filter_002.customer_id".
   If a current-node base also supplies base_role, that role is the namespace for the base node's
   bare columns; the harness derives that namespace from base_role before execution.
   Scalar operands use typed references directly. For example:
   "operands":[{{"node":"metrics","column":"part"}},{{"node":"metrics","column":"total"}}].
7. answer_from_context is never nested. Its cited table must already have exactly the requested
   rows and columns. Observing a table does not reshape it; use select for the exact result before
   terminating.

EXAMPLE AFTER SCHEMA AND LITERALS ARE GROUNDED
{{"tool":"relational_program","arguments":{{
  "calls":[
    {{"id":"exact","operation":"select","arguments":
      {{"table":{{"node":"filtered"}},"expressions":["order_id"],"distinct":true}}}},
    {{"id":"filtered","operation":"filter","arguments":
      {{"table":{{"source_table":"orders"}},
       "conditions":{{"column":"amount","op":">","value":100}}}}}}
  ],
  "result":"exact"
}}}}

LOCAL-RESULT JOIN EXAMPLE
{{"tool":"relational_program","arguments":{{
  "calls":[
    {{"id":"joined","operation":"join","arguments":{{
      "base":{{"node":"filtered"}},
      "joins":[{{"table":{{"source_table":"customers"}},"on":[
        {{"left":{{"node":"filtered","column":"customer_id"}},"right":"id"}}
      ]}}]
    }}}},
    {{"id":"filtered","operation":"filter","arguments":{{
      "table":{{"source_table":"orders"}},
      "conditions":{{"column":"status","op":"=","value":"open"}}
    }}}},
    {{"id":"exact","operation":"select","arguments":{{
      "table":{{"node":"joined"}},"expressions":["customers.name"],"distinct":true
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
        "typed_reference_schema": TYPED_REFERENCE_SCHEMA,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


_REFERENCE_KEYS = frozenset({
    "source_table",
    "resident_table",
    "resident_step",
    "node",
})
_REFERENCE_KEYSETS = frozenset({
    frozenset({"source_table"}),
    frozenset({"resident_table"}),
    frozenset({"resident_step"}),
    frozenset({"resident_step", "column"}),
    frozenset({"node"}),
    frozenset({"node", "column"}),
})


def _path_text(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "arguments"


def _is_table_reference_path(
    operation: str,
    path: tuple[str, ...],
) -> bool:
    if not path:
        return False
    key = path[-1]
    if key in {"table", "base", "in_table"}:
        return True
    return operation == "combine" and key in {"left", "right"}


def _compile_typed_references(
    value: Any,
    *,
    operation: str,
    path: tuple[str, ...] = (),
) -> tuple[Any, list[dict]]:
    """Validate and lower public typed references to the private executor carrier."""
    if isinstance(value, list):
        lowered = []
        references: list[dict] = []
        for index, child in enumerate(value):
            compiled, child_references = _compile_typed_references(
                child,
                operation=operation,
                path=(*path, str(index)),
            )
            lowered.append(compiled)
            references.extend(child_references)
        return lowered, references

    if isinstance(value, dict):
        keys = frozenset(value)
        if keys & _REFERENCE_KEYS:
            if keys not in _REFERENCE_KEYSETS:
                raise RelationalProgramProtocolError(
                    f"typed reference at {_path_text(path)} must be exactly one of "
                    '{"source_table":...}, {"resident_table":...}, '
                    '{"resident_step":...}, {"resident_step":...,"column":...}, '
                    '{"node":...}, or {"node":...,"column":...}'
                )
            reference_key = next(key for key in _REFERENCE_KEYS if key in value)
            target = value.get(reference_key)
            if not isinstance(target, str) or not target.strip():
                raise RelationalProgramProtocolError(
                    f"typed reference {reference_key} at {_path_text(path)} "
                    "must be a non-empty string"
                )
            if reference_key == "node" and not CALL_ID_RE.fullmatch(target):
                raise RelationalProgramProtocolError(
                    f"typed node reference at {_path_text(path)} must name one valid node id"
                )

            table_position = _is_table_reference_path(operation, path)
            value_from_position = bool(path and path[-1] == "value_from")
            scalar_operand_position = (
                operation == "scalar"
                and len(path) >= 2
                and path[-2] == "operands"
                and path[-1].isdigit()
            )
            join_left_position = (
                operation == "join"
                and len(path) >= 3
                and path[-1] == "left"
                and "on" in path
            )
            column = value.get("column")
            if column is not None and (
                not isinstance(column, str) or not column.strip()
            ):
                raise RelationalProgramProtocolError(
                    f"typed node column at {_path_text(path)} must be a non-empty string"
                )

            if reference_key in {"source_table", "resident_table"}:
                if not table_position:
                    raise RelationalProgramProtocolError(
                        f"{reference_key} at {_path_text(path)} is valid only in a "
                        "table/base/join-table/in_table or combine input position"
                    )
                lowered: Any = target
            elif reference_key == "resident_step":
                if column is not None and not scalar_operand_position:
                    raise RelationalProgramProtocolError(
                        f"resident_step+column at {_path_text(path)} is valid only as one "
                        "scalar operand"
                    )
                if not (value_from_position or scalar_operand_position):
                    raise RelationalProgramProtocolError(
                        f"resident_step at {_path_text(path)} is valid only as value_from "
                        "or one scalar operand"
                    )
                lowered = (
                    {
                        "value_ref": target,
                        **({"column": column} if column is not None else {}),
                    }
                    if scalar_operand_position
                    else target
                )
            elif column is not None:
                if not (join_left_position or scalar_operand_position):
                    raise RelationalProgramProtocolError(
                        f"node+column at {_path_text(path)} is valid only as join on.left "
                        "or one scalar operand"
                    )
                lowered = (
                    {"value_ref": f"${target}", "column": column}
                    if scalar_operand_position
                    else f"${target}.{column}"
                )
            else:
                if not (
                    table_position
                    or value_from_position
                    or scalar_operand_position
                ):
                    raise RelationalProgramProtocolError(
                        f"node reference at {_path_text(path)} is valid only in a table "
                        "position, value_from, or one scalar operand"
                    )
                lowered = (
                    {"value_ref": f"${target}"}
                    if scalar_operand_position
                    else f"${target}"
                )

            return lowered, [{
                "path": _path_text(path),
                "kind": reference_key,
                "target": target,
                **({"column": column} if column is not None else {}),
            }]

        if "value_ref" in value:
            raise RelationalProgramProtocolError(
                f"value_ref at {_path_text((*path, 'value_ref'))} is not public in "
                f"{RELATIONAL_PROGRAM_PROTOCOL_VERSION}; use a direct typed scalar operand "
                "or predicate value_from"
            )

        lowered_dict = {}
        references = []
        for key, child in value.items():
            compiled, child_references = _compile_typed_references(
                child,
                operation=operation,
                path=(*path, str(key)),
            )
            lowered_key = "value_ref" if key == "value_from" else key
            lowered_dict[lowered_key] = compiled
            references.extend(child_references)
        return lowered_dict, references

    if isinstance(value, str):
        if (
            LOCAL_COLUMN_REF_RE.fullmatch(value)
            or LOCAL_REF_RE.fullmatch(value)
        ):
            raise RelationalProgramProtocolError(
                f"legacy local reference {value!r} at {_path_text(path)} is not valid in "
                f"{RELATIONAL_PROGRAM_PROTOCOL_VERSION}; use a typed node reference object"
            )
        if (
            _is_table_reference_path(operation, path)
            or (path and path[-1] in {"value_ref", "value_from"})
        ):
            raise RelationalProgramProtocolError(
                f"reference at {_path_text(path)} must be a typed object distinguishing "
                "source_table, resident_table, resident_step, or node"
            )
    return deepcopy(value), []


def _lower_join_base_role_columns(
    authored_arguments: dict,
    lowered_arguments: dict,
) -> list[dict]:
    """Honor an authored base_role when a join consumes a current-program node."""
    base = authored_arguments.get("base")
    base_role = authored_arguments.get("base_role")
    if (
        not isinstance(base, dict)
        or set(base) != {"node"}
        or not isinstance(base.get("node"), str)
        or not isinstance(base_role, str)
        or not base_role.strip()
    ):
        return []
    authored_joins = authored_arguments.get("joins")
    lowered_joins = lowered_arguments.get("joins")
    if not isinstance(authored_joins, list) or not isinstance(lowered_joins, list):
        return []

    events: list[dict] = []
    for join_index, (authored_join, lowered_join) in enumerate(
        zip(authored_joins, lowered_joins, strict=False)
    ):
        if not isinstance(authored_join, dict) or not isinstance(lowered_join, dict):
            continue
        authored_edges = authored_join.get("on")
        lowered_edges = lowered_join.get("on")
        if not isinstance(authored_edges, list) or not isinstance(lowered_edges, list):
            continue
        for edge_index, (authored_edge, lowered_edge) in enumerate(
            zip(authored_edges, lowered_edges, strict=False)
        ):
            if not isinstance(authored_edge, dict) or not isinstance(lowered_edge, dict):
                continue
            authored_left = authored_edge.get("left")
            if (
                not isinstance(authored_left, dict)
                or set(authored_left) != {"node", "column"}
                or authored_left.get("node") != base["node"]
                or not isinstance(authored_left.get("column"), str)
                or "." in authored_left["column"]
            ):
                continue
            lowered = f"{base_role}.{authored_left['column']}"
            lowered_edge["left"] = lowered
            events.append({
                "path": f"joins.{join_index}.on.{edge_index}.left",
                "rule": "current_node_base_role_namespace",
                "node": base["node"],
                "base_role": base_role,
                "column": authored_left["column"],
                "lowered": lowered,
            })
    return events


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
    authored_references: dict[str, list[dict]] = {}
    compiler_lowerings: dict[str, list[dict]] = {}
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
        lowered_arguments, references = _compile_typed_references(
            call["arguments"],
            operation=operation,
        )
        lowerings = (
            _lower_join_base_role_columns(
                call["arguments"],
                lowered_arguments,
            )
            if operation == "join"
            else []
        )
        by_id[call_id] = {
            "id": call_id,
            "tool": PROGRAM_OPERATIONS[operation],
            "arguments": lowered_arguments,
        }
        authored_references[call_id] = references
        compiler_lowerings[call_id] = lowerings
        original_index[call_id] = index

    roots = [result, *exports]
    missing_roots = [root for root in roots if root not in by_id]
    if missing_roots:
        raise RelationalProgramProtocolError(
            f"program result/exports name undefined calls: {missing_roots}"
        )

    dependencies: dict[str, list[str]] = {}
    for call_id, call in by_id.items():
        refs = [
            reference["target"]
            for reference in authored_references[call_id]
            if reference["kind"] == "node"
        ]
        refs = list(dict.fromkeys(refs))
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
        "schema": "relational-program-graph-v2",
        "reference_schema": TYPED_REFERENCE_SCHEMA,
        "dependencies": {
            call_id: dependencies[call_id] for call_id in scheduled
        },
        "topological_order": scheduled,
        "result": result,
        "exports": list(exports),
        "authored_operations": {
            call["id"]: call["operation"] for call in calls
        },
        "authored_references": {
            call_id: deepcopy(authored_references[call_id])
            for call_id in scheduled
        },
        "compiler_lowerings": {
            call_id: deepcopy(compiler_lowerings[call_id])
            for call_id in scheduled
            if compiler_lowerings[call_id]
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
    allowed_by_operation = {
        "schema": {"operation", "tables"},
        "column": {"operation", "table", "column", "top_k"},
        "rows": {"operation", "table", "columns", "limit"},
    }
    unexpected = sorted(set(arguments) - allowed_by_operation[operation])
    if unexpected:
        if operation == "rows" and any(
            key in {"condition", "conditions", "filter", "where"}
            for key in unexpected
        ):
            raise RelationalProgramProtocolError(
                "observe.rows has no condition, conditions, filter, or where argument. It reads rows "
                "from exactly one supplied table without filtering. Legal keys are operation, "
                "table, optional columns, and optional limit; filter is the program operation "
                "that applies row conditions."
            )
        raise RelationalProgramProtocolError(
            f"observe.{operation} has unexpected keys {unexpected}; legal keys are "
            f"{sorted(allowed_by_operation[operation])}"
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
            "schema": "relational-program-graph-v2",
            "reference_schema": TYPED_REFERENCE_SCHEMA,
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


def _publicize_relational_error(value: Any) -> Any:
    """Render private executor names/references back into the public relational scheme."""
    if isinstance(value, dict):
        return {
            key: _publicize_relational_error(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _publicize_relational_error(child)
            for child in value
        ]
    if not isinstance(value, str):
        return deepcopy(value)

    column_match = LOCAL_COLUMN_REF_RE.fullmatch(value)
    if column_match:
        return {
            "node": column_match.group(1),
            "column": column_match.group(2),
        }
    local_match = LOCAL_REF_RE.fullmatch(value)
    if local_match:
        return {"node": local_match.group(1)}

    rendered = value
    for private_name, public_name in UNDERLYING_TO_PUBLIC_OPERATION.items():
        rendered = re.sub(
            rf"\b{re.escape(private_name)}\b",
            public_name,
            rendered,
        )
    rendered = re.sub(
        r"'\$([A-Za-z][A-Za-z0-9_]{0,31})\.([^']+)'",
        lambda match: "'" + _compact({
            "node": match.group(1),
            "column": match.group(2),
        }) + "'",
        rendered,
    )
    rendered = re.sub(
        r"'\$([A-Za-z][A-Za-z0-9_]{0,31})'",
        lambda match: "'" + _compact({"node": match.group(1)}) + "'",
        rendered,
    )
    return rendered


def build_relational_program_messages(**kwargs: Any) -> list[dict]:
    public_kwargs = deepcopy(kwargs)
    for key in ("state", "legal_history"):
        if key in public_kwargs:
            public_kwargs[key] = _publicize_exact_operation_names(
                public_kwargs[key]
            )
    if "last_error" in public_kwargs:
        public_kwargs["last_error"] = _publicize_relational_error(
            public_kwargs["last_error"]
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
            item["error"] = _publicize_relational_error(
                result.get("error") or {}
            )
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
