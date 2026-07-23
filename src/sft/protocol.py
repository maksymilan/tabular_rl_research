#!/usr/bin/env python3
"""Model<->harness interaction protocol — the single source of truth shared by
export_sft_dataset.py (renders training messages) and src/eval/rollout.py (renders live
messages and parses model output). One module for both guarantees the SFT data format and
the rollout format can never drift apart.

Message protocol (chat roles):
  system   : agent role + tool specs + interaction rules            (SYSTEM_PROMPT)
  user #1  : dataset overview JSON + the question                   (first_user_message)
  assistant: "<think>...</think>\n<tool_call>{...}</tool_call>"     (assistant_message)
  user     : CURRENT ENVIRONMENT STATE rendered from harness state  (state_context_message)
  ... repeats; the dialogue ends with the assistant's answer_from_context call.

Tool outputs are kept in the harness/debug event log, but the model-visible context is rebuilt
from resident environment state before each turn. This keeps SFT, eval and RL from accumulating a
duplicated transcript of old observations.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy

# v0 action space = exactly the tools present in the compiled Spider data. Perception / fuzzy tools
# (inspect_column, semantic_match, ...) enter with the v1 data; exposing unlearned tools at eval
# time only invites illegal calls.
TOOL_SPECS: dict[str, str] = {
    "plan":
        'plan(ops) -> update the task plan managed by the harness. For a multi-step task, call it '
        'early to split the question into subgoals, and later add/update/delete subgoals as '
        'observations change. A simple direct task may omit it. '
        'ops: [{"op": create|add|update|delete, "id": "...", "goal": "...", '
        '"status": pending|in_progress|done|blocked, "evidence": "step_k"}]. '
        'A plan item has only goal/status/evidence: goal is the intended subtask, status is progress, '
        'and evidence is a prior step id whose actual tool output the harness will attach to the '
        'plan state. Do NOT write result/conclusion/notes/value fields or final answer values in the '
        'plan. The plan is control state only: it cannot be used as value_ref or final-answer support.',
    "condition_filter":
        'condition_filter(table, conditions, return_columns=None) -> new table with the rows that '
        'satisfy `conditions`; return_columns optionally keeps only named output columns.\n'
        '  conditions: a predicate {"column": c, "op": o, "value": v} with op in '
        '=,!=,>,>=,<,<= | {"op":"in","values":[..]} | {"op":"between","low":x,"high":y} | '
        '{"op":"like","value":pat} | {"op":"contains","value":s} | {"op":"is_null"} | '
        'column-vs-column via {"column":a,"op":o,"column_value":b}; '
        'a scalar computed by an earlier step via {"column":a,"op":o,"value_ref":step_id}; '
        'set membership against a computed table via {"column":a,"op":"in","in_table":table}; '
        'combine with {"and":[..]}, {"or":[..]}, {"not": ..}.',
    "project":
        'project(table, expressions, distinct=false) -> new table with exactly the given columns. '
        '`expressions` is a list of column names or SQL scalar expressions, optionally with '
        '"expr AS alias"; distinct=true removes duplicate projected rows. Exact '
        'namespace.column identifiers remain valid inside expressions; a bare downstream column '
        'name is accepted only when it identifies exactly one available column.',
    "scalar_compute":
        'scalar_compute(operation, operands, result_name="value") -> a grounded one-row, one-column '
        'table. operation: add|subtract|multiply|divide|percent|percent_change|date_diff_days. '
        'Each operand is exactly {"value_ref":"step_k"} for a prior scalar-producing step or '
        '{"value":constant} for a constant stated by the task. Operand order matters for subtract, '
        'divide, percent (part/whole*100), percent_change ((new-old)/old*100), and date_diff_days '
        '(start,end). Cite the resulting table directly or reuse its producing step as value_ref.',
    "join_tables":
        'join_tables(base, joins, base_role=None) -> ONE new table for a connected join component. '
        '`base` is a source table or earlier handle. `joins` is an ordered list of '
        '{"table": T, "on": [{"left": "known_relation.column", "right": "new_column"}], '
        '"type": "inner|left|cross"?, "role": "semantic_role"?}. `left` must be an exact logical '
        'column already introduced; `right` is a bare column of the newly attached table. `type` '
        'defaults to inner; cross uses on=[]. Output columns use a flat relation.column namespace, '
        'never recursively nest a join handle. Derived-handle state may group exact names as '
        'column_namespaces={relation:[column,..]}; reconstruct each as relation.column and never '
        'prefix it with the derived handle. Omit roles normally; use `base_role`/`role` only when '
        'the same relation occurs more than once (self-join). Put a whole consecutive join chain in '
        'ONE call; use project separately if the result must be narrowed.',
    "group_aggregate":
        'group_aggregate(table, group_by, aggregations, passthrough=None) -> new table grouped by '
        '`group_by` (list of columns; [] = whole table as one group, used for scalar count/sum/'
        'avg/min/max answers too). aggregations: '
        '[{"op": sum|count|count_distinct|mean|min|max, "column": col or "*", "as": name}, ..] '
        '([] with group_by = DISTINCT). passthrough: extra non-grouped columns to carry through.',
    "extreme_value_select":
        'extreme_value_select(table, order_by, top_k=None, return_columns=None) -> new table with '
        'the rows ordered by `order_by` (list of "col" or "col DESC") keeping the top `top_k` '
        '(None = all rows, just ordered). return_columns optionally projects.',
    "set_op":
        'set_op(left, right, op) -> new table combining two tables with op: '
        'union|union_all|intersect|except (their columns must align).',
    "describe_table":
        'describe_table(tables) -> the columns, types, primary keys and foreign keys of one or more '
        'tables (tables is a list; pass several at once). The opening overview lists only table names '
        'and relations, so read the schema of the tables you need before operating on them.',
    "inspect_column":
        'inspect_column(table, column) -> the distinct count, most frequent values and NULL flag of a '
        'column. Use it to ground a filter literal (does "France" exist? what is the exact spelling?) '
        'before condition_filter.',
    "read_subtable":
        'read_subtable(table, limit=20, columns=None) -> up to 20 actual rows of a table '
        '(limit must be 1..20); columns optionally limits which columns are observed. Tool results otherwise '
        'show only a table handle (name, columns, row_count); read_subtable is how you SEE rows, e.g. '
        'the evidence rows before answering. Reading columns does not project or change the table.',
    "answer_from_context":
        'answer_from_context(evidence, reason="") -> TERMINAL. evidence must be {"table": name} '
        'for a grounded table holding the exact answer rows, columns, and column order. This same '
        'rule covers scalar answers: cite the 1x1 table produced by group_aggregate or '
        'scalar_compute. The model never writes answer data in the terminal call. Use project first '
        'when helper columns remain or column order is wrong; think/reason text cannot repair data.',
}

TOOLS = set(TOOL_SPECS)
# ``aggregate`` and the parser repairs below exist only to read historical trajectory artifacts.
# New model turns must use ``TOOLS`` through ``parse_assistant_strict``.  Keeping this distinction
# explicit prevents old data compatibility from quietly widening the live agent interface.
LEGACY_TOOLS = {"aggregate"}
REPLAY_COMPAT_TOOLS = TOOLS | LEGACY_TOOLS
ACCEPTED_TOOLS = REPLAY_COMPAT_TOOLS

PROTOCOL_VERSION = "version13"  # public tool versions now increment numerically: version1, version2, ...
ROLLING_CONTEXT_VERSION = "v2-bounded-legal-history-resident-observations"
ROLLING_COMPACT_PROMPT_VERSION = "v1-safe-compact"

# Strict per-tool argument schema (required, optional). Unlisted keys are rejected so the SFT data
# and the live rollout can never silently drift. V2b: a predicate's `value_ref` cites the producing
# step_id directly; there is no add_to_memory tool and no model-authored value.
_ARG_SCHEMA: dict[str, tuple[set, set]] = {
    "plan": ({"ops"}, set()),
    "condition_filter": ({"table", "conditions"}, {"return_columns", "preview_k"}),
    "project": ({"table", "expressions"}, {"distinct"}),
    "scalar_compute": ({"operation", "operands"}, {"result_name"}),
    "join_tables": (set(), {"base", "joins", "base_role",
                            "tables", "on", "join_types", "prefixes",
                            "left", "right", "join_type", "left_prefix", "right_prefix",
                            "return_columns"}),
    "group_aggregate": ({"table", "group_by", "aggregations"}, {"passthrough"}),
    "aggregate": ({"table", "column", "op"}, set()),
    "extreme_value_select": ({"table", "order_by"}, {"top_k", "return_columns"}),
    "set_op": ({"left", "right", "op"}, set()),
    "describe_table": ({"tables"}, set()),
    "inspect_column": ({"table", "column"}, {"top_k"}),
    "read_subtable": ({"table"}, {"limit", "columns"}),
    "answer_from_context": (set(), {"answer", "evidence", "reason"}),
}

CANONICAL_CALL_COOKBOOK = (
    "CANONICAL CALLS (copy these argument shapes; replace names and values only)\n"
    'Filter: {"tool":"condition_filter","arguments":{"table":"people","conditions":'
    '{"and":[{"column":"city","op":"=","value":"Paris"},{"column":"age","op":">=","value":18}]}}}\n'
    'Project exact final columns: {"tool":"project","arguments":{"table":"filter_001",'
    '"expressions":["name","email"]}}\n'
    'Project unique exact columns: {"tool":"project","arguments":{"table":"filter_001",'
    '"expressions":["first_name","middle_name","last_name"],"distinct":true}}\n'
    'Compute a column with project: {"tool":"project","arguments":{"table":"sales",'
    '"expressions":["product","price * quantity AS revenue"]}}\n'
    'Join a three-table path: {"tool":"join_tables","arguments":'
    '{"base":"orders","joins":['
    '{"table":"customers","on":[{"left":"orders.customer_id","right":"id"}]},'
    '{"table":"regions","on":[{"left":"customers.region_id","right":"id"}]}]}}\n'
    'Self-join with roles: {"tool":"join_tables","arguments":{"base":"employees",'
    '"base_role":"employee","joins":[{"table":"employees","role":"manager",'
    '"on":[{"left":"employee.manager_id","right":"id"}]}]}}\n'
    'Aggregate: {"tool":"group_aggregate","arguments":{"table":"filter_001","group_by":["department"],'
    '"aggregations":[{"op":"sum","column":"salary","as":"total_salary"}]}}\n'
    'Scalar aggregate: {"tool":"group_aggregate","arguments":{"table":"filter_001","group_by":[],'
    '"aggregations":[{"op":"count","column":"*","as":"count"}]}}\n'
    'Compute a grounded percentage: {"tool":"scalar_compute","arguments":{"operation":"percent",'
    '"operands":[{"value_ref":"step_5"},{"value_ref":"step_3"}],"result_name":"percentage"}}\n'
    'Top 3 with exact output: {"tool":"extreme_value_select","arguments":{"table":"employees",'
    '"order_by":["sick_leave_hours DESC"],"top_k":3,"return_columns":["job_title"]}}\n'
    'Set operation after aligning both inputs with project: {"tool":"set_op","arguments":'
    '{"left":"project_001","right":"project_002","op":"union"}}\n'
    'Any final answer, including a scalar: {"tool":"answer_from_context","arguments":'
    '{"evidence":{"table":"project_003"},'
    '"reason":"The evidence table has exactly the requested rows and columns."}}\n'
)

# Historical version1-version4 join fields remain valid only when replaying old artifacts.
_MODEL_FORBIDDEN_ARGUMENTS: dict[str, set[str]] = {
    "condition_filter": {"preview_k"},
    "answer_from_context": {"answer"},
    "join_tables": {
        "tables", "on", "join_types", "prefixes", "return_columns",
        "left", "right", "join_type", "left_prefix", "right_prefix",
    },
}


def validate_arguments(tool: str, args: dict) -> None:
    """Strict per-tool argument schema; raises ProtocolError on any missing/unexpected key."""
    schema = _ARG_SCHEMA.get(tool)
    if schema is None:
        return
    required, optional = schema
    keys = set(args)
    missing = required - keys
    if missing:
        raise ProtocolError(f"{tool}: missing arguments {sorted(missing)}")
    extra = keys - required - optional
    if extra:
        raise ProtocolError(f"{tool}: unexpected arguments {sorted(extra)}")
    if tool == "answer_from_context" and "answer" not in keys and "evidence" not in keys:
        raise ProtocolError('answer_from_context requires at least "evidence" or "answer"')
    if tool == "join_tables":
        new_form = "base" in keys or "joins" in keys or "base_role" in keys
        nway_v4 = "tables" in keys
        binary_legacy = "left" in keys or "right" in keys
        if sum((new_form, nway_v4, binary_legacy)) != 1:
            raise ProtocolError(
                "join_tables requires exactly one compatible shape: base+joins, tables+on, "
                "or left+right+on"
            )
        required_shape = (
            {"base", "joins"} if new_form
            else {"tables", "on"} if nway_v4
            else {"left", "right", "on"}
        )
        missing_shape = sorted(required_shape - keys)
        if missing_shape:
            raise ProtocolError(f"join_tables: missing arguments {missing_shape}")


def _validate_model_join(args: dict) -> None:
    base = args.get("base")
    joins = args.get("joins")
    if not isinstance(base, str) or not base.strip():
        raise ProtocolError("join_tables.base must be a non-empty table or handle")
    if not isinstance(joins, list) or not joins:
        raise ProtocolError("join_tables.joins must be a non-empty list")

    role_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def namespace(table: str, role, where: str) -> str:
        if role is not None:
            if not isinstance(role, str) or not role_pattern.fullmatch(role):
                raise ProtocolError(
                    f"{where} must be an identifier containing only letters, digits, and underscores"
                )
            return role
        return table

    namespaces = [namespace(base, args.get("base_role"), "join_tables.base_role")]
    for index, item in enumerate(joins):
        where = f"join_tables.joins[{index}]"
        if not isinstance(item, dict):
            raise ProtocolError(f"{where} must be an object")
        extra = sorted(set(item) - {"table", "on", "type", "role"})
        missing = sorted({"table", "on"} - set(item))
        if missing:
            raise ProtocolError(f"{where}: missing fields {missing}")
        if extra:
            raise ProtocolError(f"{where}: unexpected fields {extra}")
        table = item.get("table")
        if not isinstance(table, str) or not table.strip():
            raise ProtocolError(f"{where}.table must be a non-empty table or handle")
        join_type = item.get("type", "inner")
        if join_type not in {"inner", "left", "cross"}:
            raise ProtocolError(f"{where}.type must be inner, left, or cross")
        edges = item.get("on")
        if not isinstance(edges, list):
            raise ProtocolError(f"{where}.on must be a list")
        if join_type != "cross" and not edges:
            raise ProtocolError(f"{where}.on must contain at least one equality edge")
        if join_type == "cross" and edges:
            raise ProtocolError(f"{where}.on must be [] for a cross join")
        for edge_index, edge in enumerate(edges):
            edge_where = f"{where}.on[{edge_index}]"
            if not isinstance(edge, dict) or set(edge) != {"left", "right"}:
                raise ProtocolError(f"{edge_where} must contain exactly left and right")
            left, right = edge.get("left"), edge.get("right")
            if not isinstance(left, str) or "." not in left:
                raise ProtocolError(
                    f"{edge_where}.left must be an exact known_relation.column reference"
                )
            if not isinstance(right, str) or not right or "." in right:
                raise ProtocolError(f"{edge_where}.right must be a bare column of the new table")
        namespaces.append(namespace(table, item.get("role"), f"{where}.role"))
    folded = [item.casefold() for item in namespaces]
    duplicates = sorted({name for name in folded if folded.count(name) > 1})
    if duplicates:
        raise ProtocolError(
            "join_tables relation namespaces must be unique; add semantic base_role/role for "
            f"repeated relations: {duplicates}"
        )


def validate_model_arguments(tool: str, args: dict) -> None:
    """Validate the current public action API, excluding replay-only compatibility forms."""
    validate_arguments(tool, args)
    forbidden = sorted(set(args) & _MODEL_FORBIDDEN_ARGUMENTS.get(tool, set()))
    if forbidden:
        raise ProtocolError(f"{tool}: legacy arguments are not valid in new episodes: {forbidden}")
    if tool == "join_tables":
        _validate_model_join(args)
    if tool == "read_subtable":
        limit = args.get("limit", 20)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ProtocolError("read_subtable: limit must be an integer from 1 to 20")
    if tool == "project" and not isinstance(args.get("distinct", False), bool):
        raise ProtocolError("project: distinct must be true or false")
    if tool == "scalar_compute":
        operation = args.get("operation")
        allowed = {
            "add", "subtract", "multiply", "divide", "percent",
            "percent_change", "date_diff_days",
        }
        if operation not in allowed:
            raise ProtocolError(
                f"scalar_compute: operation must be one of {sorted(allowed)}"
            )
        operands = args.get("operands")
        if not isinstance(operands, list) or len(operands) < 2:
            raise ProtocolError("scalar_compute: operands must contain at least two items")
        if operation in {"divide", "percent", "percent_change", "date_diff_days"} and len(operands) != 2:
            raise ProtocolError(f"scalar_compute: {operation} requires exactly two operands")
        for index, operand in enumerate(operands):
            if not isinstance(operand, dict) or len(operand) != 1:
                raise ProtocolError(
                    f"scalar_compute: operands[{index}] must contain exactly value_ref or value"
                )
            if set(operand) == {"value_ref"}:
                if not isinstance(operand["value_ref"], str) or not operand["value_ref"].strip():
                    raise ProtocolError(
                        f"scalar_compute: operands[{index}].value_ref must be a step id"
                    )
            elif set(operand) != {"value"}:
                raise ProtocolError(
                    f"scalar_compute: operands[{index}] must contain exactly value_ref or value"
                )
    if tool == "answer_from_context":
        evidence = args.get("evidence")
        if (
            not isinstance(evidence, dict)
            or set(evidence) != {"table"}
            or not isinstance(evidence.get("table"), str)
            or not evidence["table"].strip()
        ):
            raise ProtocolError(
                'answer_from_context: evidence must be exactly {"table":"result_handle"}; '
                "scalar answers also cite a grounded 1x1 table"
            )


def protocol_hash(system_prompt: str | None = None) -> str:
    """Stable hash of the model<->harness contract (version + system prompt + tool specs). SFT
    manifests and rollout runs record it so a train/eval protocol mismatch is detectable."""
    payload = json.dumps({"version": PROTOCOL_VERSION, "system": system_prompt or SYSTEM_PROMPT, "tools": TOOL_SPECS},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

SYSTEM_PROMPT = (
    "You are a table-reasoning agent. You answer questions over a relational dataset by calling "
    "tools, one call per turn. The opening overview is a CATALOG: table names + row counts + "
    "foreign-key relations only (no columns) — so it stays small on large databases. When the task "
    "includes EXTERNAL KNOWLEDGE, treat it as part of the user-provided task context. Read the "
    "columns of the tables you need with describe_table before operating. After each tool call, "
    "the harness updates a CURRENT ENVIRONMENT STATE message. Treat that state as the authoritative "
    "workspace: it contains the resident plan, known schemas, inspected values, table handles, row "
    "reads, scalar values, and the step ids you may cite later (e.g. as a predicate's value_ref). "
    "You will not receive a full transcript of old tool observations; use the state instead of "
    "repeating previous reads. A table-producing tool's state entry is only a HANDLE (table name, "
    "columns, row_count) until you call read_subtable to see rows.\n\n"
    "TOOLS\n" + "\n".join(TOOL_SPECS.values()) + "\n\n" + CANONICAL_CALL_COOKBOOK + "\n"
    "RULES\n"
    "1. Each turn, output exactly: <think>brief reasoning</think> then "
    '<tool_call>{"tool": "<name>", "arguments": {...}}</tool_call>. Nothing else.\n'
    "2. Use plan for multi-step tasks to break the question into subgoals; update it when a subgoal "
    "starts, completes, or changes. Simple direct tasks may proceed without plan.\n"
    "3. describe_table the needed tables first; inspect_column before filtering by a text value.\n"
    "4. To use a computed scalar as a threshold, set the predicate's "
    '{"value_ref": step_id} to the step that produced that scalar. To answer with a scalar, cite '
    "that producing 1x1 table as terminal evidence; never copy its value into the final call.\n"
    "5. For every answer, make the evidence table's rows, columns, and column order exactly match "
    "the requested output. read_subtable only observes rows; it does not change table shape. If "
    "extra/helper columns remain, project first, then cite that table. Never write answer data "
    "inside answer_from_context.\n"
    "6. If a derived handle exposes column_namespaces, form exact references as namespace.column. "
    "The handle is the table argument, never a replacement column namespace.\n"
    "7. In downstream filters, projections, grouping, and ordering, prefer namespace.column; a "
    "bare column is valid only when exactly one available logical column has that suffix.\n"
    "8. Preserve the database output slots exactly. Keep separate fields in separate columns; do "
    "not concatenate names, replace an ID/code with a label, translate/case-normalize stored text, "
    "or round a numeric result unless the question explicitly requests that transformation. Apply "
    "distinct=true only when unique/distinct rows are requested or required by the task wording.\n"
    "9. For top-k, order and limit before answering and use return_columns (or project) so helper "
    "ranking columns are absent. For arithmetic over aggregate results, call scalar_compute and "
    "cite its 1x1 result table instead of manually writing a terminal number.\n"
)

SYSTEM_PROMPT_COMPACT = (
    "You are a table-tool agent. Answer by calling one tool per turn.\n\n"
    "STRICT FORMAT\n"
    "Each turn output exactly:\n"
    "<think>brief reason for this action</think>\n"
    '<tool_call>{"tool":"...","arguments":{...}}</tool_call>\n'
    "No text outside these tags.\n\n"
    "CONTEXT\n"
    "The opening overview is only a catalog: table names, row counts, and relations. It has no "
    "columns. Use describe_table only for relevant unresolved tables. Tool-created tables return "
    "handles (table, columns, row_count), not rows. Use existing handles instead of restarting from "
    "source tables. Use plan for multi-step tasks, then update it as work completes or changes. "
    "A separate CURRENT ENVIRONMENT STATE message may summarize the current plan and known table "
    "handles before your turn.\n\n"
    "POLICY\n"
    "Inspect a text column before filtering by a literal unless that column was already inspected. "
    "Avoid repeating the same observation. Use read_subtable only when row values are needed. Every "
    "final answer, including a scalar aggregate, cites its exact result table as evidence; never "
    "write answer values in the terminal call. The cited table must contain exactly the requested "
    "rows and columns; project first if it does not.\n\n"
    "TOOLS\n"
    "plan(ops), describe_table(tables), inspect_column(table,column,top_k?), condition_filter(table,conditions,return_columns?), "
    "project(table,expressions,distinct?), scalar_compute(operation,operands,result_name?), "
    "join_tables(base,joins,base_role?), "
    "group_aggregate(table,group_by,aggregations,passthrough?), "
    "extreme_value_select(table,order_by,top_k?,return_columns?), set_op(left,right,op), "
    "read_subtable(table,limit?,columns?), answer_from_context(evidence,reason?).\n"
)

ROLLING_HISTORY_SYSTEM_SUFFIX = (
    "\n\nROLLING LEGAL HISTORY\n"
    "In this mode, the user may include a bounded transcript of earlier assistant actions that the "
    "harness executed successfully, paired with their tool-result messages. Continue from that "
    "legal history instead of restarting the task. The transcript is intentionally bounded, so do "
    "not assume it contains every old observation. CURRENT ENVIRONMENT STATE remains the "
    "authoritative factual workspace; LAST TOOL ERROR is the authoritative record of a rejected "
    "action. Never treat a plan item or unexecuted text as factual evidence."
)

# This is a rolling-only ablation. It deliberately retains the public action contract and the
# current join/value-reference rules that the generic compact prompt predates.
ROLLING_SYSTEM_PROMPT_COMPACT = (
    "You are a relational table-tool agent. Solve the user question with exactly one tool action "
    "per turn.\n\n"
    "FORMAT\n"
    "Output only <think>specific reason for the next action</think> followed by one complete "
    '<tool_call>{"tool":"name","arguments":{...}}</tool_call>. No prose outside the tags, no '
    "second action, no shorthand JSON, and no legacy tool fields.\n\n"
    "CONTEXT\n"
    "The first user message is a catalog of table names, row counts, and relations, not schemas. "
    "Call describe_table before using unresolved columns. CURRENT ENVIRONMENT STATE is the factual "
    "workspace: use its handles, schemas, inspected values, reads, scalar-producing step ids, and "
    "plan. A bounded transcript may contain only earlier harness-successful actions/results; it is "
    "continuity context, not complete evidence. LAST TOOL ERROR describes a rejected action. Do not "
    "repeat a read already present in state. Table handles expose metadata only until read_subtable "
    "returns rows.\n\n"
    "TOOLS\n"
    "plan(ops); describe_table(tables); inspect_column(table,column,top_k?); "
    "condition_filter(table,conditions,return_columns?); project(table,expressions,distinct?); "
    "scalar_compute(operation,operands,result_name?); "
    "join_tables(base,joins,base_role?); "
    "group_aggregate(table,group_by,aggregations,passthrough?); "
    "extreme_value_select(table,order_by,top_k?,return_columns?); set_op(left,right,op); "
    "read_subtable(table,limit?,columns?); answer_from_context(evidence,reason?).\n\n"
    "RULES\n"
    "plan is control only: goals/status/evidence may cite prior step ids, never results or answer "
    "values. Inspect text domains before literal filters unless already inspected. conditions support "
    "comparisons, like, in, between, null, and/or/not; cite a scalar as value_ref:step_id or a "
    "computed table as in_table. Use existing handles rather than restarting from sources. For an "
    "n-way join, base starts the component and each joins item attaches one new table. Each on.left "
    "is an exact already-visible relation.column and on.right is a bare column of the new table. "
    "When base is a derived handle, copy on.left from its column_namespaces as namespace.column; "
    "never invent handle.column. Use semantic roles only for repeated relations; never emit SQL "
    "aliases such as L. or R. Downstream scalar expressions may use these exact logical columns; "
    "the harness quotes them as single identifiers. "
    "For every answer, read the evidence handle then call answer_from_context with that evidence. "
    "Scalar answers also cite their grounded 1x1 result table. Every evidence table is scored "
    "exactly: project away helper columns and fix column order before citing it. Never put answer "
    "values in the terminal call; think/reason cannot repair incorrect answer data. Keep separate "
    "database fields separate; do not replace "
    "IDs/codes with labels, normalize stored text, or round computed values unless explicitly "
    "requested. Use project distinct=true for unique rows and scalar_compute for arithmetic over "
    "grounded scalar steps. The harness strictly validates and executes the action."
)


def get_system_prompt() -> str:
    """Return the default train/eval prompt, or a compact eval-only variant.

    The default remains SYSTEM_PROMPT so SFT data and existing protocol hashes stay stable.
    Set EVAL_SYSTEM_PROMPT_VARIANT=compact for prompt-ablation evaluations.
    """
    variant = os.environ.get("EVAL_SYSTEM_PROMPT_VARIANT", "").strip().lower()
    if variant in {"compact", "short"}:
        return SYSTEM_PROMPT_COMPACT
    return SYSTEM_PROMPT


def rolling_system_prompt(system_prompt: str, *, compact: bool = False) -> str:
    """Return the full or safe-compact bounded-history contract.

    ``compact`` is intentionally rolling-only and opt-in. The state-only protocol and existing
    full-prompt rolling artifacts remain byte-for-byte stable.
    """
    return ROLLING_SYSTEM_PROMPT_COMPACT if compact else system_prompt + ROLLING_HISTORY_SYSTEM_SUFFIX


class ProtocolError(Exception):
    """Model output does not parse into a legal tool call."""


def _compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def first_user_message(
    overview: dict,
    question: str,
    external_knowledge: str | None = None,
) -> str:
    message = f"DATASET OVERVIEW\n{_compact(overview)}\n\nQUESTION\n{question}"
    if external_knowledge and external_knowledge.strip():
        message += f"\n\nEXTERNAL KNOWLEDGE\n{external_knowledge.strip()}"
    return message


def assistant_message(think: str, tool: str, arguments: dict) -> str:
    return (f"<think>{think}</think>\n"
            f"<tool_call>{_compact({'tool': tool, 'arguments': arguments})}</tool_call>")


def _compact_output_columns(output: dict) -> dict:
    """Model-visible compact form for contiguous flat logical namespaces."""
    visible = dict(output)
    columns = visible.get("columns")
    if not isinstance(columns, list) or not columns:
        return visible
    groups: dict[str, list[str]] = {}
    closed: set[str] = set()
    current: str | None = None
    for column in columns:
        if not isinstance(column, str) or "." not in column:
            return visible
        namespace, name = column.split(".", 1)
        if not namespace or not name:
            return visible
        if namespace != current:
            if namespace in closed:
                return visible
            if current is not None:
                closed.add(current)
            current = namespace
            groups[namespace] = []
        groups[namespace].append(name)
    visible.pop("columns", None)
    visible["column_namespaces"] = groups
    return visible


def tool_output_message(step_id: str, output: dict, status: str = "success",
                        state: dict | None = None) -> str:
    """Observation envelope: a stable `step_id` (so the model can cite it as `source_step_id`),
    a status, and the tool output. Identical offline (SFT) and online (rollout)."""
    msg = {"step_id": step_id, "status": status, "output": _compact_output_columns(output)}
    if state is not None:
        msg["state"] = state
    return _compact(msg)


def compact_resident_observation(observation: str) -> str:
    """Replace duplicated historical payloads with a structured resident-state pointer.

    Rolling context still carries the successful action/result pair, but schemas, inspected
    values, row samples, and scalar samples already live in CURRENT ENVIRONMENT STATE. Keeping
    those large facts in both places caused avoidable context overflow. Harness-authored metadata
    remains visible here; the complete factual payload remains visible once in resident state.
    Non-standard legacy strings are left untouched rather than guessed at.
    """
    try:
        envelope = json.loads(observation)
    except (TypeError, json.JSONDecodeError):
        return observation
    if not isinstance(envelope, dict) or envelope.get("status") != "success":
        return observation
    output = envelope.get("output")
    if not isinstance(output, dict):
        return observation
    output = _compact_output_columns(output)

    summary: dict = {}
    for key in (
        "table", "kind", "columns", "column_namespaces", "row_count",
        "column", "distinct_count", "has_null",
    ):
        if output.get(key) is not None:
            summary[key] = output[key]

    if isinstance(output.get("tables"), list):
        summary["tables"] = [
            {
                key: table[key]
                for key in ("table_name", "row_count")
                if isinstance(table, dict) and table.get(key) is not None
            }
            | ({"column_count": len(table.get("columns", []))} if isinstance(table, dict) else {})
            for table in output["tables"]
        ]
    if isinstance(output.get("rows"), list):
        summary["returned_row_count"] = len(output["rows"])
    if isinstance(output.get("result_sample"), list):
        summary["result_sample_row_count"] = len(output["result_sample"])
    if isinstance(output.get("frequent_values"), list):
        summary["frequent_value_count"] = len(output["frequent_values"])
    if isinstance(output.get("changes"), list):
        summary["changes"] = output["changes"]

    # Small outputs that contain no resident factual payload remain useful verbatim. Large factual
    # fields are represented by metadata above and resolved from the appended current state.
    resident_fields = {"tables", "rows", "result_sample", "frequent_values", "plan", "final_answer"}
    if not (set(output) & resident_fields):
        summary = output
    envelope = {
        "step_id": envelope.get("step_id"),
        "status": "success",
        "output_summary": summary,
        "full_output": "resident_in_current_environment_state",
    }
    return _compact(envelope)


def tool_error_message(step_id: str, error_type: str, message: str) -> str:
    return _compact({
        "step_id": step_id,
        "status": "error",
        "error": {"type": error_type, "message": message},
    })


def _compact_state_columns(state: dict | None) -> dict:
    """Compact only the model-visible rendering; canonical harness snapshots stay unchanged."""
    visible = deepcopy(state or {"plan": [], "tables": {}, "values": {}})
    tables = visible.get("tables")
    if isinstance(tables, dict):
        for name, entry in list(tables.items()):
            if isinstance(entry, dict):
                tables[name] = _compact_output_columns(entry)
    for item in visible.get("plan") or []:
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence")
        if not isinstance(evidence, dict) or not isinstance(evidence.get("output"), dict):
            continue
        evidence["output"] = _compact_output_columns(evidence["output"])
    return visible


def environment_state_message(state: dict | None, last_error: dict | None = None) -> str:
    text = "CURRENT ENVIRONMENT STATE\n" + _compact(_compact_state_columns(state))
    if last_error:
        text += "\n\nLAST TOOL ERROR\n" + _compact(last_error)
    return text


def state_context_message(state: dict | None, last_error: dict | None = None) -> str:
    """Model-visible mutable context between assistant turns."""
    return environment_state_message(state, last_error=last_error)


def _state_is_empty(state: dict | None) -> bool:
    if not isinstance(state, dict):
        return True
    return not state.get("plan") and not state.get("tables") and not state.get("values")


def task_context_message(
    overview: dict,
    question: str,
    state: dict | None,
    last_error: dict | None = None,
    external_knowledge: str | None = None,
) -> str:
    """One-turn user context shared by online rollout and step-level SFT samples."""
    text = first_user_message(overview, question, external_knowledge)
    if not _state_is_empty(state) or last_error:
        text += "\n\n" + state_context_message(state, last_error)
    return text


def model_context_messages(system: str, overview: dict, question: str, state: dict | None,
                           last_error: dict | None = None,
                           external_knowledge: str | None = None) -> list[dict]:
    """Build the complete model-visible context for one turn from resident state.

    This is the state-only path used by online eval/RL. It deliberately ignores any accumulated
    debug transcript so old observations cannot re-enter the prompt.
    """
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": task_context_message(
            overview, question, state, last_error, external_knowledge,
        )},
    ]


def rolling_legal_history_messages(
    system: str,
    overview: dict,
    question: str,
    state: dict | None,
    last_error: dict | None,
    external_knowledge: str | None,
    legal_history: list[dict],
    history_turns: int,
    *,
    compact_observations: bool = True,
) -> list[dict]:
    """Render a bounded transcript of harness-successful assistant/tool pairs.

    Rejected assistant text never becomes context. Its structured error is carried only in the
    current user message, so training and online inference cannot teach the model to imitate an
    invalid call. ``history_turns=0`` retains every legal pair for experiments; production callers
    should use an explicit positive bound.
    """
    if history_turns < 0:
        raise ValueError("history_turns must be non-negative")
    initial = first_user_message(overview, question, external_knowledge)
    if not legal_history:
        content = initial
        if state or last_error:
            content += "\n\n" + state_context_message(state, last_error)
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]

    retained = legal_history if history_turns == 0 else legal_history[-history_turns:]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": initial}]
    for index, item in enumerate(retained):
        assistant = item.get("assistant")
        observation = item.get("observation")
        if not isinstance(assistant, str) or not assistant.strip():
            raise ValueError("rolling legal history has an empty assistant action")
        if not isinstance(observation, str) or not observation.strip():
            raise ValueError("rolling legal history has an empty tool observation")
        messages.append({"role": "assistant", "content": assistant})
        if compact_observations:
            observation = compact_resident_observation(observation)
        if index == len(retained) - 1:
            observation += "\n\n" + state_context_message(state, last_error)
        messages.append({"role": "user", "content": observation})
    return messages


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


def _extract_balanced_json(text: str, start: int) -> str | None:
    """Extract one balanced JSON object from text[start:], respecting strings.

    This only helps when the model emits a complete JSON object but forgets the closing
    </tool_call> tag. Truncated JSON remains a protocol error.
    """
    begin = text.find("{", start)
    if begin < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(begin, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[begin:i + 1]
    return None


def _tool_call_payloads(text: str) -> list[str]:
    payloads = _TOOL_CALL_RE.findall(text)
    if payloads:
        return payloads
    tag = text.rfind("<tool_call>")
    if tag < 0:
        return []
    balanced = _extract_balanced_json(text, tag + len("<tool_call>"))
    return [balanced] if balanced else []


def _loads_tool_call(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Common unambiguous shorthand:
        #   {"answer_from_context","arguments":{...}}
        # Treat it as {"tool":"answer_from_context","arguments":{...}}.
        m = re.match(r'\s*\{\s*"(?P<tool>[A-Za-z_][\w]*)"\s*,\s*"arguments"\s*:\s*(?P<args>\{.*\})\s*\}\s*$', raw, re.S)
        if m and m.group("tool") in ACCEPTED_TOOLS:
            return {"tool": m.group("tool"), "arguments": json.loads(m.group("args"))}
        raise


_HANDLE_RE = re.compile(r"\b(?:project|filter|join|group|top|setop|derive)_\d{3}\b")


def _repair_truncated_answer_call(text: str) -> dict | None:
    """Recover an answer_from_context call when only the long answer JSON was truncated.

    This is intentionally narrow: it only synthesizes the terminal call when the model clearly
    attempted answer_from_context and cited a concrete evidence table handle. The harness still
    scores the cited table; no answer values are guessed from free text.
    """
    if "answer_from_context" not in text:
        return None
    table = None
    m = re.search(r'"evidence"\s*:\s*\{\s*"table"\s*:\s*"([^"]+)"', text)
    if m:
        table = m.group(1)
    if table is None:
        m = re.search(r"evidence table\s+`?((?:project|filter|join|group|top|setop|derive)_\d{3})`?", text, re.I)
        if m:
            table = m.group(1)
    if table is None:
        handles = _HANDLE_RE.findall(text)
        if handles:
            table = handles[-1]
    if table is None:
        return None
    return {
        "tool": "answer_from_context",
        "arguments": {
            "answer": [],
            "evidence": {"table": table},
            "reason": "Derived by the cited evidence table.",
        },
    }


def _normalize_answer_args(args: dict) -> dict:
    args = dict(args)
    evidence = args.get("evidence")
    if isinstance(evidence, str):
        args["evidence"] = {"table": evidence}
    elif evidence is None and "evidence" not in args:
        args["evidence"] = None
    if "answer" not in args:
        args["answer"] = []
    return args


def parse_assistant(text: str) -> tuple[str, str, dict]:
    """Parse a model turn into (think, tool, arguments). Raises ProtocolError."""
    m = _tool_call_payloads(text)
    if not m:
        call = _repair_truncated_answer_call(text)
        if call is None:
            raise ProtocolError("no <tool_call>{...}</tool_call> block found")
    else:
        try:
            call = _loads_tool_call(m[-1])  # last block wins if the model quoted an example
        except json.JSONDecodeError as e:
            call = _repair_truncated_answer_call(text)
            if call is None:
                raise ProtocolError(f"tool_call is not valid JSON: {e}") from e
    tool = call.get("tool")
    args = call.get("arguments")
    if tool is None:
        # Another common answer shorthand:
        #   {"answer_from_context": {"answer": ..., "evidence": ...}}
        shorthand = [(key, value) for key, value in call.items() if key in ACCEPTED_TOOLS]
        if len(shorthand) == 1:
            tool, value = shorthand[0]
            args = value.get("arguments") if isinstance(value, dict) and isinstance(value.get("arguments"), dict) else value
    if tool not in ACCEPTED_TOOLS:
        raise ProtocolError(f"unknown tool {tool!r}; legal tools: {sorted(TOOLS)}")
    if not isinstance(args, dict):
        raise ProtocolError('tool_call must have an "arguments" object')
    if tool == "answer_from_context":
        args = _normalize_answer_args(args)
    validate_arguments(tool, args)
    tm = _THINK_RE.search(text)
    return (tm.group(1).strip() if tm else ""), tool, args


def parse_assistant_strict(text: str) -> tuple[str, str, dict]:
    """Parse one fully-formed teacher action without repair or argument normalization.

    External teacher generation may use this mode when protocol mistakes should become explicit
    environment feedback. It intentionally rejects the legacy convenience repairs in
    :func:`parse_assistant`: balanced JSON without a closing tag, shorthand tool-call objects,
    truncated terminal answers, and omitted answer fields.
    """
    payloads = _TOOL_CALL_RE.findall(text)
    if len(payloads) != 1:
        raise ProtocolError(
            "expected exactly one complete <tool_call>{...}</tool_call> block; "
            f"received {len(payloads)}"
        )
    try:
        call = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"tool_call is not valid JSON: {exc}") from exc
    if not isinstance(call, dict):
        raise ProtocolError("tool_call JSON must be an object")
    if set(call) != {"tool", "arguments"}:
        raise ProtocolError('tool_call must contain exactly "tool" and "arguments" keys')
    tool = call.get("tool")
    args = call.get("arguments")
    if tool not in TOOLS:
        raise ProtocolError(f"unknown tool {tool!r}; legal tools: {sorted(TOOLS)}")
    if not isinstance(args, dict):
        raise ProtocolError('tool_call must have an "arguments" object')
    think_blocks = _THINK_RE.findall(text)
    if len(think_blocks) != 1 or not think_blocks[0].strip():
        raise ProtocolError("expected exactly one non-empty <think>...</think> block")
    validate_model_arguments(tool, args)
    return think_blocks[0].strip(), tool, args
