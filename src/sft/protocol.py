#!/usr/bin/env python3
"""Model<->harness interaction protocol shared by SFT export, evaluation, and RL.

``prompt_contract.py`` owns role-separated prompt text; this module binds its shared student
semantics to public argument validation, message rendering, parsing, and replay compatibility.
Using these same functions in every runtime prevents train/eval/RL carrier drift.

Message protocol (chat roles):
  system   : agent role + tool specs + interaction rules            (SYSTEM_PROMPT)
  user #1  : dataset overview JSON + the question                   (first_user_message)
  assistant: "<think>...</think>\n{...}"                            (assistant_message)
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

from action_carrier import (
    ACTIVE_ACTION_CARRIER,
    ActionCarrierError,
    parse_action_carrier,
    parse_legacy_tagged_action_carrier,
    render_action_carrier,
)
from prompt_contract import (
    PUBLIC_TOOL_ARGUMENTS,
    ROLLING_HISTORY_CONTRACT,
    SHARED_TOOL_SPECS,
    add_teacher_guidance,
    build_student_system_prompt,
)
from public_tool_contract import (
    ACTION_BLOCK_PUBLIC_TOOL_ARGUMENTS,
    AGGREGATION_LAYOUTS,
    AGGREGATION_OPERATIONS,
    AGGREGATION_OPTIONAL,
    AGGREGATION_REQUIRED,
    JOIN_EDGE_KEYS,
    JOIN_ITEM_OPTIONAL,
    JOIN_ITEM_REQUIRED,
    JOIN_TYPES,
    ROW_DATE_EXPRESSION_OPERAND_KEYSETS,
    ROW_DATE_EXPRESSION_OPERATIONS,
    SCALAR_OPERAND_KEYSETS,
    SCALAR_OPERATIONS,
    VERSION40_PUBLIC_TOOL_ARGUMENTS,
)

# Teacher-only elaborations for the same public tools. These descriptions may explain edge cases
# but cannot widen the tool names or argument schemas declared below.
TEACHER_TOOL_GUIDANCE: dict[str, str] = {
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
        '{"op":"like","value":pat} | {"op":"contains","value":s} | '
        '{"op":"on_date","value":"YYYY-MM-DD"} | {"op":"is_null"} | '
        'column-vs-column via {"column":a,"op":o,"column_value":b}; '
        'a scalar computed by an earlier step via {"column":a,"op":o,"value_ref":step_id}; '
        'set membership against a computed table via {"column":a,"op":"in","in_table":table}; '
        'combine with {"and":[..]}, {"or":[..]}, {"not": ..}.',
    "project":
        'project(table, expressions, distinct=false) -> new table with exactly the given columns. '
        '`expressions` is a list of column names, SQL scalar expressions optionally with '
        '"expr AS alias", or typed date expressions '
        '{"op":"date_diff_days","operands":[{"column":"start"},{"column":"end"}],"as":"days"} '
        'and {"op":"extract_year","operands":[{"column":"date"}],"as":"year"}. '
        'date_diff_days operand order is start,end. distinct=true removes duplicate rows. Exact '
        'namespace.column identifiers remain valid inside expressions; a bare downstream column '
        'name is accepted only when it identifies exactly one available column. Project preserves '
        'the input row orientation: it cannot turn category rows into separate columns; use '
        'group_aggregate(output_layout="columns") for that reshape. Arguments are only table, '
        'expressions, and optional distinct; project has no limit argument.',
    "scalar_compute":
        'scalar_compute(operation, operands, result_name="value") -> a grounded one-row, one-column '
        'table. operation: add|subtract|multiply|divide|percent|percent_change|date_diff_days. '
        'Each operand is exactly {"value_ref":"step_k"} for a prior 1x1 step, '
        '{"value_ref":"step_k","column":"metric"} for one named column of a prior one-row table, or '
        '{"value":constant} for a constant stated by the task. Operand order matters for subtract, '
        'divide, percent (part/whole*100), percent_change ((new-old)/old*100), and date_diff_days '
        '(start,end). A value_ref must cite the step that PRODUCED the resident result table, never '
        'a plan, describe_table, inspect_column, or read_subtable observation step. If you read a '
        'one-row table at step_5 that was produced at step_4, cite step_4. Cite the resulting '
        'scalar table directly or reuse its producing step as value_ref.',
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
        'ONE call; use project separately if the result must be narrowed. Per-edge keys are only '
        'table, on, optional type, and optional role; type never belongs at the top level. In every '
        'on pair, left is the exact introduced logical column and right has NO dot: use '
        '{"left":"orders.customer_id","right":"id"}, never right="customers.id".',
    "group_aggregate":
        'group_aggregate(table, group_by, aggregations, passthrough=None, output_layout="rows", '
        'category_values=None, output_columns=None) -> new table grouped by '
        '`group_by` (list of columns; [] = whole table as one group, used for scalar count/sum/'
        'avg/min/max answers too). aggregations: '
        '[{"op": sum|count|count_distinct|mean|min|max, "column": col or "*", "as": name, '
        '"where": condition?}, ..]. Optional `where` uses the same predicate shape as '
        'condition_filter and applies only to that aggregate. Use several conditional aggregates '
        'in one call when the requested metrics must share one fixed input row set and grain. '
        'Default output_layout="rows" returns one row per group. To return one row with one ordered '
        'column per category, use output_layout="columns" with exactly one group_by column, one '
        'aggregation, ordered category_values, and equally ordered output_columns. '
        '([] with group_by = DISTINCT). passthrough is a LIST of extra non-grouped columns. '
        'Top-level arguments are table, group_by, aggregations, and optional passthrough/'
        'output_layout/category_values/output_columns. Put where only inside the aggregation it '
        'conditions; top-level where, conditions, and result_name are invalid.',
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
        'read_subtable(table, limit=20, columns=None, conditions=None, order_by=None, offset=0) -> '
        'up to 20 matching actual rows without deriving a table. conditions uses the same typed '
        'predicate tree as condition_filter, including on_date for calendar-date matching. '
        'columns limits which columns are observed; order_by is a list of exact columns optionally '
        'ending in ASC/DESC. A positive offset requires order_by and reads a later deterministic '
        'page. Tool results otherwise '
        'show only a table handle (name, columns, row_count); read_subtable is how you SEE rows, e.g. '
        'the evidence rows before answering. Reading does not project, filter into a new handle, or '
        'change the table.',
    "answer_from_context":
        'answer_from_context(evidence, reason="") -> TERMINAL. evidence must be {"table": name} '
        'for a grounded table holding the exact answer rows, columns, and column order. This same '
        'rule covers scalar answers: cite the 1x1 table produced by group_aggregate or '
        'scalar_compute. The model never writes answer data in the terminal call. Use project first '
        'when helper columns remain or column order is wrong; think/reason text cannot repair data.',
}

TOOL_SPECS = SHARED_TOOL_SPECS
TOOLS = set(TOOL_SPECS)
# ``aggregate`` and the parser repairs below exist only to read historical trajectory artifacts.
# New model turns must use ``TOOLS`` through ``parse_assistant_strict``.  Keeping this distinction
# explicit prevents old data compatibility from quietly widening the live agent interface.
LEGACY_TOOLS = {"aggregate", "pivot"}
REPLAY_COMPAT_TOOLS = TOOLS | LEGACY_TOOLS
# inspect_rows is the version40 public alias for the same read-only executor operation. It is
# replay/execution-compatible here but remains absent from the default version39 model surface.
ACCEPTED_TOOLS = REPLAY_COMPAT_TOOLS | {"inspect_rows"}

PROTOCOL_VERSION = "version39"  # model-visible resident-state compaction
ROLLING_CONTEXT_VERSION = "v2-bounded-legal-history-resident-observations"
ROLLING_COMPACT_PROMPT_VERSION = "v1-safe-compact"
POLICY_PROMPT_CANONICAL = "canonical"
POLICY_PROMPT_RELATIONAL_INVARIANTS = "relational-invariants"
POLICY_PROMPT_VARIANTS = (
    POLICY_PROMPT_CANONICAL,
    POLICY_PROMPT_RELATIONAL_INVARIANTS,
)
HISTORY_POLICY_RECENT = "recent"
HISTORY_POLICY_HEAD_TAIL = "head-tail"
HISTORY_POLICIES = (HISTORY_POLICY_RECENT, HISTORY_POLICY_HEAD_TAIL)
STUDENT_PROMPT_CANONICAL = "canonical"
STUDENT_PROMPT_FORMAL = "formal"
STUDENT_PROMPT_VARIANTS = (
    STUDENT_PROMPT_CANONICAL,
    STUDENT_PROMPT_FORMAL,
)
RELATIONAL_INVARIANTS_SUFFIX = (
    "\n\nRELATIONAL DECISION INVARIANTS — EXPERIMENTAL PROMPT VARIANT\n"
    "Before aggregation or ranking, identify the current input population and what one row "
    "represents; do not change that grain or collapse matching rows unless the task asks for it. "
    "If answer eligibility or output columns depend on another relation, form the required join "
    "before ranking or aggregation, because operations before and after a join may use different "
    "populations. Use inner join unless the task explicitly requires retaining base rows without "
    "matches. Project exactly the entity and output slots defined by the question or EXTERNAL "
    "KNOWLEDGE; do not replace an identifier/code with a label or vice versa. Do not sum, average, "
    "count, or otherwise combine repeated observed values unless the requested answer calls for "
    "that aggregation."
)

# Replay-compatible argument schema for historical artifacts. New model actions use the public
# schema derived from ``PUBLIC_TOOL_ARGUMENTS`` below; compatibility fields never enter its prompt
# or hash.
_ARG_SCHEMA: dict[str, tuple[set, set]] = {
    "plan": ({"ops"}, set()),
    "condition_filter": ({"table", "conditions"}, {"return_columns", "preview_k"}),
    "project": ({"table", "expressions"}, {"distinct"}),
    "scalar_compute": ({"operation", "operands"}, {"result_name"}),
    "join_tables": (set(), {"base", "joins", "base_role",
                            "tables", "on", "join_types", "prefixes",
                            "left", "right", "join_type", "left_prefix", "right_prefix",
                            "return_columns"}),
    "group_aggregate": (
        {"table", "group_by", "aggregations"},
        {"passthrough", "output_layout", "category_values", "output_columns"},
    ),
    "pivot": ({"table", "key_column", "value_column", "key_values"}, {"output_columns"}),
    "aggregate": ({"table", "column", "op"}, set()),
    "extreme_value_select": (
        {"table", "order_by"},
        {"top_k", "return_columns"},
    ),
    "set_op": ({"left", "right", "op"}, set()),
    "describe_table": ({"tables"}, set()),
    "inspect_column": ({"table", "column"}, {"top_k"}),
    "read_subtable": (
        {"table"},
        {"limit", "columns", "conditions", "order_by", "offset"},
    ),
    "inspect_rows": (
        {"table"},
        {"limit", "columns", "conditions", "order_by", "offset"},
    ),
    "answer_from_context": (set(), {"answer", "evidence", "reason"}),
}

# Public atomic action schema. Replay accepts older shapes through ``_ARG_SCHEMA`` above, while
# every newly authored model action is checked only against this schema. Keeping the two maps
# separate prevents historical compatibility fields from leaking into prompts, hashes, or RL.
MODEL_ARG_SCHEMA: dict[str, tuple[set[str], set[str]]] = {
    tool: (set(required), set(optional))
    for tool, (required, optional) in PUBLIC_TOOL_ARGUMENTS.items()
}
VERSION40_MODEL_ARG_SCHEMA: dict[str, tuple[set[str], set[str]]] = {
    tool: (set(required), set(optional))
    for tool, (required, optional) in VERSION40_PUBLIC_TOOL_ARGUMENTS.items()
}
ACTION_BLOCK_MODEL_ARG_SCHEMA: dict[str, tuple[set[str], set[str]]] = {
    tool: (set(required), set(optional))
    for tool, (required, optional) in ACTION_BLOCK_PUBLIC_TOOL_ARGUMENTS.items()
}
if set(MODEL_ARG_SCHEMA) != TOOLS:
    raise RuntimeError("public argument schema and model-visible tool contract have drifted")

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
    'Compute a date difference for every row: {"tool":"project","arguments":{"table":"courses",'
    '"expressions":["person",{"op":"date_diff_days","operands":'
    '[{"column":"start_date"},{"column":"end_date"}],"as":"duration_days"}]}}\n'
    'Read one matching row without deriving a table: {"tool":"read_subtable","arguments":'
    '{"table":"transactions","columns":["id","amount"],"conditions":'
    '{"column":"posted_at","op":"on_date","value":"2024-01-31"},"limit":1}}\n'
    'Read the next deterministic page: {"tool":"read_subtable","arguments":'
    '{"table":"transactions","columns":["id","amount"],"order_by":["id"],'
    '"offset":20,"limit":20}}\n'
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
    'Conditional aggregates on one fixed input: {"tool":"group_aggregate","arguments":'
    '{"table":"patients","group_by":[],"aggregations":['
    '{"op":"count","column":"*","as":"female_count","where":'
    '{"column":"gender","op":"=","value":"F"}},'
    '{"op":"count","column":"*","as":"male_count","where":'
    '{"column":"gender","op":"=","value":"M"}}]}}\n'
    'Aggregate categories directly into exact output columns: {"tool":"group_aggregate","arguments":'
    '{"table":"hypertension_patients","group_by":["gender"],"aggregations":'
    '[{"op":"count_distinct","column":"patient","as":"patient_count"}],'
    '"output_layout":"columns","category_values":["M","F"]}}\n'
    'Compute a grounded percentage: {"tool":"scalar_compute","arguments":{"operation":"percent",'
    '"operands":[{"value_ref":"step_5"},{"value_ref":"step_3"}],"result_name":"percentage"}}\n'
    'Compute from two named metrics in one prior row: {"tool":"scalar_compute","arguments":'
    '{"operation":"percent","operands":['
    '{"value_ref":"step_7","column":"usa_nominees"},'
    '{"value_ref":"step_7","column":"total_nominees"}],"result_name":"percentage"}}\n'
    'Top 3 with exact output: {"tool":"extreme_value_select","arguments":{"table":"employees",'
    '"order_by":["sick_leave_hours DESC"],"top_k":3,"return_columns":["job_title"]}}\n'
    'Set operation after aligning both inputs with project: {"tool":"set_op","arguments":'
    '{"left":"project_001","right":"project_002","op":"union"}}\n'
    'Any final answer, including a scalar: {"tool":"answer_from_context","arguments":'
    '{"evidence":{"table":"project_003"},'
    '"reason":"The evidence table has exactly the requested rows and columns."}}\n'
)

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


def _validate_argument_keys(
    tool: str,
    args: dict,
    schema: dict[str, tuple[set[str], set[str]]],
) -> None:
    """Validate required/optional keys for one explicitly selected schema."""
    required, optional = schema[tool]
    keys = set(args)
    missing = sorted(required - keys)
    if missing:
        raise ProtocolError(f"{tool}: missing arguments {missing}")
    extra = sorted(keys - required - optional)
    if extra:
        raise ProtocolError(f"{tool}: unexpected arguments {extra}")


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
        extra = sorted(set(item) - JOIN_ITEM_REQUIRED - JOIN_ITEM_OPTIONAL)
        missing = sorted(JOIN_ITEM_REQUIRED - set(item))
        if missing:
            raise ProtocolError(f"{where}: missing fields {missing}")
        if extra:
            raise ProtocolError(f"{where}: unexpected fields {extra}")
        table = item.get("table")
        if not isinstance(table, str) or not table.strip():
            raise ProtocolError(f"{where}.table must be a non-empty table or handle")
        join_type = item.get("type", "inner")
        if join_type not in JOIN_TYPES:
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
            if not isinstance(edge, dict) or set(edge) != JOIN_EDGE_KEYS:
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


def _validate_model_group_aggregate(args: dict) -> None:
    group_by = args.get("group_by")
    if not isinstance(group_by, list) or not all(isinstance(item, str) for item in group_by):
        raise ProtocolError("group_aggregate.group_by must be a list of columns")
    passthrough = args.get("passthrough", [])
    if not isinstance(passthrough, list) or not all(isinstance(item, str) for item in passthrough):
        raise ProtocolError("group_aggregate.passthrough must be a list of columns")
    aggregations = args.get("aggregations")
    if not isinstance(aggregations, list):
        raise ProtocolError("group_aggregate.aggregations must be a list")
    allowed_ops = set(AGGREGATION_OPERATIONS)
    for index, aggregation in enumerate(aggregations):
        where = f"group_aggregate.aggregations[{index}]"
        if not isinstance(aggregation, dict):
            raise ProtocolError(f"{where} must be an object")
        missing = sorted(AGGREGATION_REQUIRED - set(aggregation))
        extra = sorted(
            set(aggregation) - AGGREGATION_REQUIRED - AGGREGATION_OPTIONAL
        )
        if missing:
            raise ProtocolError(f"{where}: missing fields {missing}")
        if extra:
            raise ProtocolError(f"{where}: unexpected fields {extra}")
        op = aggregation.get("op")
        if op not in allowed_ops:
            raise ProtocolError(f"{where}.op must be one of {sorted(allowed_ops)}")
        column = aggregation.get("column")
        if not isinstance(column, str) or not column.strip():
            raise ProtocolError(f"{where}.column must be a non-empty column name or *")
        alias = aggregation.get("as")
        if not isinstance(alias, str) or not alias.strip():
            raise ProtocolError(f"{where}.as must be a non-empty output column name")
        condition = aggregation.get("where")
        if "where" in aggregation and (
            not isinstance(condition, (dict, list)) or not condition
        ):
            raise ProtocolError(f"{where}.where must be a non-empty condition predicate")
        if condition and op == "count_distinct" and column == "*":
            raise ProtocolError(
                f"{where}: conditional count_distinct requires a named column, not *"
            )
        if column == "*" and op not in {"count", "count_distinct"}:
            raise ProtocolError(f"{where}: only count may aggregate column *")
    output_layout = args.get("output_layout", "rows")
    if output_layout not in AGGREGATION_LAYOUTS:
        raise ProtocolError("group_aggregate.output_layout must be rows or columns")
    category_values = args.get("category_values")
    output_columns = args.get("output_columns")
    if output_layout == "rows":
        if category_values is not None or output_columns is not None:
            raise ProtocolError(
                "group_aggregate.category_values/output_columns require output_layout=columns"
            )
        return
    if len(group_by) != 1 or len(aggregations) != 1 or passthrough:
        raise ProtocolError(
            "group_aggregate output_layout=columns requires exactly one group_by column, "
            "one aggregation, and no passthrough"
        )
    if not isinstance(category_values, list) or not category_values:
        raise ProtocolError(
            "group_aggregate.category_values must be a non-empty ordered list "
            "for output_layout=columns"
        )
    if not all(
        item is not None and isinstance(item, (str, int, float, bool))
        for item in category_values
    ):
        raise ProtocolError(
            "group_aggregate.category_values must contain non-null scalar values"
        )
    category_keys = [(type(item).__name__, repr(item)) for item in category_values]
    if len(set(category_keys)) != len(category_keys):
        raise ProtocolError("group_aggregate.category_values must be unique")
    if output_columns is not None:
        if (
            not isinstance(output_columns, list)
            or not all(isinstance(item, str) and item.strip() for item in output_columns)
        ):
            raise ProtocolError(
                "group_aggregate.output_columns must be a list of non-empty column names"
            )
        if len(output_columns) != len(category_values):
            raise ProtocolError(
                "group_aggregate.output_columns must have the same length as category_values"
            )
        folded = [item.casefold() for item in output_columns]
        if len(set(folded)) != len(folded):
            raise ProtocolError("group_aggregate.output_columns must be unique")


def _validate_row_inspection(args: dict, *, tool: str) -> None:
    limit = args.get("limit", 20)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ProtocolError(f"{tool}: limit must be an integer from 1 to 20")
    columns = args.get("columns")
    if columns is not None and (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(item, str) and item.strip() for item in columns)
    ):
        raise ProtocolError(
            f"{tool}: columns must be a non-empty list of non-empty column names"
        )
    conditions = args.get("conditions")
    if conditions is not None and (
        not isinstance(conditions, (dict, list)) or not conditions
    ):
        raise ProtocolError(
            f"{tool}: conditions must be a non-empty condition predicate"
        )
    order_by = args.get("order_by")
    if order_by is not None and (
        not isinstance(order_by, list)
        or not order_by
        or not all(isinstance(item, str) and item.strip() for item in order_by)
    ):
        raise ProtocolError(
            f"{tool}: order_by must be a non-empty list of column strings"
        )
    offset = args.get("offset", 0)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ProtocolError(f"{tool}: offset must be a non-negative integer")
    if offset and not order_by:
        raise ProtocolError(
            f"{tool}: positive offset requires order_by for deterministic pagination"
        )


def _validate_project(args: dict, *, allow_typed_date_rows: bool) -> None:
    if not isinstance(args.get("distinct", False), bool):
        raise ProtocolError("project: distinct must be true or false")
    expressions = args.get("expressions")
    if not isinstance(expressions, list) or not expressions:
        raise ProtocolError("project: expressions must be a non-empty list")
    arity = {
        "date_diff_days": 2,
        "extract_year": 1,
    }
    for index, expression in enumerate(expressions):
        if isinstance(expression, str) and expression.strip():
            continue
        if not allow_typed_date_rows or not isinstance(expression, dict):
            expected = (
                "non-empty strings"
                if not allow_typed_date_rows
                else "non-empty strings or typed date-expression objects"
            )
            raise ProtocolError(f"project: expressions[{index}] must contain {expected}")
        if set(expression) != {"op", "operands", "as"}:
            raise ProtocolError(
                f"project: expressions[{index}] typed object must contain exactly op, operands, as"
            )
        operation = expression.get("op")
        if operation not in ROW_DATE_EXPRESSION_OPERATIONS:
            raise ProtocolError(
                f"project: expressions[{index}].op must be one of "
                f"{sorted(ROW_DATE_EXPRESSION_OPERATIONS)}"
            )
        alias = expression.get("as")
        if not isinstance(alias, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", alias
        ):
            raise ProtocolError(
                f"project: expressions[{index}].as must be an identifier"
            )
        operands = expression.get("operands")
        if not isinstance(operands, list) or len(operands) != arity[operation]:
            raise ProtocolError(
                f"project: expressions[{index}] {operation} requires "
                f"{arity[operation]} operands"
            )
        for operand_index, operand in enumerate(operands):
            keys = set(operand) if isinstance(operand, dict) else set()
            if frozenset(keys) not in ROW_DATE_EXPRESSION_OPERAND_KEYSETS:
                raise ProtocolError(
                    f"project: expressions[{index}].operands[{operand_index}] must be "
                    "exactly column or value"
                )
            if "column" in keys and (
                not isinstance(operand["column"], str) or not operand["column"].strip()
            ):
                raise ProtocolError(
                    f"project: expressions[{index}].operands[{operand_index}].column "
                    "must be a non-empty column name"
                )
            if "value" in keys and (
                operand["value"] is None
                or isinstance(operand["value"], (dict, list))
            ):
                raise ProtocolError(
                    f"project: expressions[{index}].operands[{operand_index}].value "
                    "must be a non-null scalar"
                )


def _validate_extreme_value_select(args: dict) -> None:
    order_by = args.get("order_by")
    if (
        not isinstance(order_by, list)
        or not order_by
        or not all(isinstance(item, str) and item.strip() for item in order_by)
    ):
        raise ProtocolError(
            "extreme_value_select: order_by must be a non-empty list of column strings"
        )
    top_k = args.get("top_k")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
    ):
        raise ProtocolError("extreme_value_select: top_k must be a positive integer")
    return_columns = args.get("return_columns")
    if return_columns is not None and (
        not isinstance(return_columns, list)
        or not return_columns
        or not all(isinstance(item, str) and item.strip() for item in return_columns)
    ):
        raise ProtocolError(
            "extreme_value_select: return_columns must be a non-empty list of columns"
        )


def _validate_common_model_arguments(
    tool: str,
    args: dict,
    *,
    allow_typed_date_rows: bool,
) -> None:
    if tool == "join_tables":
        _validate_model_join(args)
    if tool == "group_aggregate":
        _validate_model_group_aggregate(args)
    if tool in {"read_subtable", "inspect_rows"}:
        _validate_row_inspection(args, tool=tool)
    if tool == "project":
        _validate_project(args, allow_typed_date_rows=allow_typed_date_rows)
    if tool == "extreme_value_select":
        _validate_extreme_value_select(args)
    if tool == "scalar_compute":
        operation = args.get("operation")
        allowed = set(SCALAR_OPERATIONS)
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
            keys = set(operand) if isinstance(operand, dict) else set()
            if frozenset(keys) not in SCALAR_OPERAND_KEYSETS:
                raise ProtocolError(
                    f"scalar_compute: operands[{index}] must be exactly value, value_ref, "
                    "or value_ref+column"
                )
            if "value_ref" in keys:
                if not isinstance(operand["value_ref"], str) or not operand["value_ref"].strip():
                    raise ProtocolError(
                        f"scalar_compute: operands[{index}].value_ref must be a step id"
                    )
                if "column" in keys and (
                    not isinstance(operand["column"], str) or not operand["column"].strip()
                ):
                    raise ProtocolError(
                        f"scalar_compute: operands[{index}].column must be a non-empty column name"
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


def validate_model_arguments(tool: str, args: dict) -> None:
    """Validate the current atomic action API, excluding replay compatibility."""
    _validate_argument_keys(tool, args, MODEL_ARG_SCHEMA)
    _validate_common_model_arguments(tool, args, allow_typed_date_rows=True)


def validate_version40_model_arguments(tool: str, args: dict) -> None:
    """Validate the isolated version40 public action API."""
    _validate_argument_keys(tool, args, VERSION40_MODEL_ARG_SCHEMA)
    _validate_common_model_arguments(tool, args, allow_typed_date_rows=True)


def validate_action_block_arguments(tool: str, args: dict) -> None:
    """Validate the frozen action-block-v32 primitive API."""
    _validate_argument_keys(tool, args, ACTION_BLOCK_MODEL_ARG_SCHEMA)
    _validate_common_model_arguments(tool, args, allow_typed_date_rows=False)


def tool_schema_hash() -> str:
    """Full hash of public tool names, argument keys, and concise semantics."""
    arguments = {
        tool: {
            "required": sorted(required),
            "optional": sorted(optional),
        }
        for tool, (required, optional) in sorted(MODEL_ARG_SCHEMA.items())
    }
    payload = json.dumps(
        {"tools": TOOL_SPECS, "arguments": arguments},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protocol_hash(system_prompt: str | None = None) -> str:
    """Stable hash of protocol version, selected prompt, and public tool schema."""
    payload = json.dumps(
        {
            "version": PROTOCOL_VERSION,
            "system": system_prompt or SYSTEM_PROMPT,
            "tool_schema_sha256": tool_schema_hash(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

# The concise student runtime prompt is shared by SFT export, evaluation, and RL. The detailed
# descriptions and cookbook above are retained as teacher-only guidance.
SYSTEM_PROMPT = build_student_system_prompt()
STUDENT_SYSTEM_PROMPT = SYSTEM_PROMPT
FORMAL_STUDENT_SYSTEM_PROMPT = build_student_system_prompt(
    include_action_grammar=True
)
TEACHER_SYSTEM_PROMPT = add_teacher_guidance(
    STUDENT_SYSTEM_PROMPT,
    tool_guidance=TEACHER_TOOL_GUIDANCE,
    call_cookbook=CANONICAL_CALL_COOKBOOK,
)

SYSTEM_PROMPT_COMPACT = (
    "You are a table-tool agent. Answer by calling one tool per turn.\n\n"
    "STRICT FORMAT\n"
    "Each turn output exactly:\n"
    "<think>brief reason for this action</think>\n"
    '{"tool":"...","arguments":{...}}\n'
    "No other text and no tool_call tags.\n\n"
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
    "rows and columns; project first if it does not. Compute several conditional metrics that share "
    "one population in one group_aggregate call using per-aggregation where predicates. Use "
    "group_aggregate output_layout=columns, not project, to turn grouped category rows into one row "
    "of separate metric columns. scalar_compute may cite a named metric from a one-row table with "
    "a value_ref+column operand.\n\n"
    "TOOLS\n"
    "plan(ops), describe_table(tables), inspect_column(table,column,top_k?), condition_filter(table,conditions,return_columns?), "
    "project(table,expressions,distinct?), scalar_compute(operation,operands,result_name?), "
    "join_tables(base,joins,base_role?), "
    "group_aggregate(table,group_by,aggregations,passthrough?,output_layout?,category_values?,output_columns?), "
    "extreme_value_select(table,order_by,top_k?,return_columns?), set_op(left,right,op), "
    "read_subtable(table,limit?,columns?,conditions?,order_by?,offset?), "
    "answer_from_context(evidence,reason?).\n"
)

ROLLING_HISTORY_SYSTEM_SUFFIX = ROLLING_HISTORY_CONTRACT

# This is a rolling-only ablation. It deliberately retains the public action contract and the
# current join/value-reference rules that the generic compact prompt predates.
ROLLING_SYSTEM_PROMPT_COMPACT = (
    "You are a relational table-tool agent. Solve the user question with exactly one tool action "
    "per turn.\n\n"
    "FORMAT\n"
    "Output only <think>specific reason for the next action</think> followed by one complete "
    '{"tool":"name","arguments":{...}} JSON object. No other prose, no tool_call tags, no '
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
    "group_aggregate(table,group_by,aggregations,passthrough?,output_layout?,category_values?,output_columns?); "
    "extreme_value_select(table,order_by,top_k?,return_columns?); set_op(left,right,op); "
    "read_subtable(table,limit?,columns?,conditions?,order_by?,offset?); "
    "answer_from_context(evidence,reason?).\n\n"
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
    "grounded scalar steps; when one prior row contains several metrics, cite each as "
    "value_ref+column without recomputing it. Use aggregation-level where predicates when several conditional metrics "
    "must retain one input population and grain. Use group_aggregate output_layout=columns rather "
    "than project when grouped categories must become separate columns in one row. The harness "
    "strictly validates and executes the action."
)


def get_system_prompt() -> str:
    """Return the selected student runtime prompt."""
    variant = os.environ.get("EVAL_SYSTEM_PROMPT_VARIANT", "").strip().lower()
    if variant in {"compact", "short"}:
        return SYSTEM_PROMPT_COMPACT
    if variant in {"formal", "structured"}:
        return FORMAL_STUDENT_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def teacher_system_prompt(student_prompt: str) -> str:
    """Add teacher-only quality guidance to an already selected student context contract."""
    return add_teacher_guidance(
        student_prompt,
        tool_guidance=TEACHER_TOOL_GUIDANCE,
        call_cookbook=CANONICAL_CALL_COOKBOOK,
    )


def get_teacher_system_prompt() -> str:
    """Return teacher guidance over the same selected public semantics as the student."""
    return teacher_system_prompt(get_system_prompt())


def rolling_system_prompt(system_prompt: str, *, compact: bool = False) -> str:
    """Return the full or safe-compact bounded-history contract.

    ``compact`` is intentionally rolling-only and opt-in. Historical artifacts retain their
    stored prompts; new version26 episodes use the role-selected prompt passed here.
    """
    return ROLLING_SYSTEM_PROMPT_COMPACT if compact else system_prompt + ROLLING_HISTORY_SYSTEM_SUFFIX


def student_runtime_system_prompt(
    *,
    context_mode: str = "rolling-legal-history",
    compact: bool = False,
    student_prompt_variant: str = STUDENT_PROMPT_CANONICAL,
) -> str:
    """One prompt selector shared by SFT export, evaluation defaults, and RL."""
    if student_prompt_variant not in STUDENT_PROMPT_VARIANTS:
        raise ValueError(
            f"unsupported student_prompt_variant: {student_prompt_variant!r}; "
            f"expected one of {STUDENT_PROMPT_VARIANTS}"
        )
    if compact and student_prompt_variant != STUDENT_PROMPT_CANONICAL:
        raise ValueError("compact and formal student prompt variants cannot be combined")
    base = (
        SYSTEM_PROMPT_COMPACT
        if compact
        else FORMAL_STUDENT_SYSTEM_PROMPT
        if student_prompt_variant == STUDENT_PROMPT_FORMAL
        else SYSTEM_PROMPT
    )
    if context_mode == "rolling-legal-history":
        return rolling_system_prompt(base, compact=compact)
    if context_mode == "state-only":
        return base
    raise ValueError(f"unsupported context_mode: {context_mode}")


def policy_system_prompt(system_prompt: str, variant: str = POLICY_PROMPT_CANONICAL) -> str:
    """Apply an auditable policy-prompt ablation without changing shared tool semantics."""
    if variant == POLICY_PROMPT_CANONICAL:
        return system_prompt
    if variant == POLICY_PROMPT_RELATIONAL_INVARIANTS:
        return system_prompt + RELATIONAL_INVARIANTS_SUFFIX
    raise ValueError(
        f"unknown policy prompt variant {variant!r}; expected one of {POLICY_PROMPT_VARIANTS}"
    )


class ProtocolError(Exception):
    """Model output does not parse into a legal tool call."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "protocol_error",
        details: dict | None = None,
        failure_type: str | None = None,
        attempted_tool: str | None = None,
        attempted_arguments: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = deepcopy(details or {})
        self.failure_type = failure_type
        self.attempted_tool = attempted_tool
        self.attempted_arguments = deepcopy(attempted_arguments)


class AdjacentDuplicateActionError(ProtocolError):
    """Two consecutive parsed model turns contain the same tool and arguments."""

    def __init__(
        self,
        *,
        step_id: str,
        previous_step_id: str,
        previous_status: str,
        previous_error: dict | None,
        tool: str,
        arguments: dict,
    ) -> None:
        previous_rejection = deepcopy(previous_error or {})
        retained_error = ""
        if previous_rejection:
            retained_error = (
                " The preceding rejection remains authoritative: "
                f"{previous_rejection.get('type', 'error')}: "
                f"{previous_rejection.get('message', '')}."
            )
        tool_semantics = ""
        if tool == "read_subtable":
            tool_semantics = (
                " an identical read_subtable call requests the same rows; change conditions, "
                "order_by, offset, columns, or limit to inspect different rows."
            )
        super().__init__(
            (
                "the parsed tool and arguments exactly match the immediately preceding "
                f"action at {previous_step_id}; this call was not executed and the current "
                f"environment state is unchanged.{retained_error}{tool_semantics} "
                "The next action must change the tool or at least one argument"
            ),
            code="adjacent_identical_action",
            details={
                "comparison": "canonical_tool_and_arguments",
                "current_step_id": step_id,
                "previous_step_id": previous_step_id,
                "previous_status": previous_status,
                "state_changed_by_rejected_call": False,
                **(
                    {"previous_rejection": previous_rejection}
                    if previous_rejection
                    else {}
                ),
            },
            failure_type="no_progress_error",
            attempted_tool=tool,
            attempted_arguments=arguments,
        )


class AdjacentActionGuard:
    """Reject only exactly equal parsed actions on consecutive model turns."""

    def __init__(self) -> None:
        self._previous: dict | None = None

    @staticmethod
    def canonical_key(tool: str, arguments: dict) -> str:
        return json.dumps(
            {"tool": tool, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def clear(self) -> None:
        """Break adjacency after a turn that did not contain one parseable action."""
        self._previous = None

    def prime(
        self,
        tool: str,
        arguments: dict,
        *,
        step_id: str,
        status: str,
        error_context: dict | None = None,
    ) -> None:
        """Seed the immediately preceding parsed action when resuming a real episode prefix."""
        self._previous = {
            "key": self.canonical_key(tool, arguments),
            "step_id": step_id,
            "status": status,
            "error_context": deepcopy(error_context),
        }

    def observe(self, tool: str, arguments: dict, *, step_id: str) -> None:
        key = self.canonical_key(tool, arguments)
        previous = self._previous
        self._previous = {
            "key": key,
            "step_id": step_id,
            "status": "parsed",
            "error_context": None,
        }
        if previous is not None and previous["key"] == key:
            self._previous["status"] = "rejected"
            raise AdjacentDuplicateActionError(
                step_id=step_id,
                previous_step_id=previous["step_id"],
                previous_status=previous["status"],
                previous_error=previous.get("error_context"),
                tool=tool,
                arguments=arguments,
            )

    def mark_last(self, status: str, error_context: dict | None = None) -> None:
        if self._previous is not None:
            self._previous["status"] = status
            retained = deepcopy(error_context)
            if (
                isinstance(retained, dict)
                and retained.get("code") == "adjacent_identical_action"
            ):
                previous = (retained.get("details") or {}).get("previous_rejection")
                if isinstance(previous, dict):
                    retained = deepcopy(previous)
            self._previous["error_context"] = retained


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
    return render_action_carrier(think, tool, arguments)


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
    those large facts in both places caused avoidable context overflow. The complete factual
    payload, including table-bound derivation metadata, remains visible once in resident state.
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
    resident_fields = {
        "table", "tables", "rows", "result_sample", "frequent_values", "plan", "final_answer",
    }
    if not (set(output) & resident_fields):
        summary = output
    envelope = {
        "step_id": envelope.get("step_id"),
        "status": "success",
        "output_summary": summary,
        "full_output": "resident_in_current_environment_state",
    }
    return _compact(envelope)


def tool_error_message(
    step_id: str,
    error_type: str,
    message: str,
    *,
    error_code: str | None = None,
    details: dict | None = None,
    attempted_tool: str | None = None,
    attempted_arguments: dict | None = None,
) -> str:
    error = {"type": error_type, "message": message}
    if error_code:
        error["code"] = error_code
    if details:
        error["details"] = deepcopy(details)
    envelope = {
        "step_id": step_id,
        "status": "error",
        "error": error,
    }
    if attempted_tool is not None:
        envelope["attempted_action"] = {
            "tool": attempted_tool,
            "arguments": deepcopy(attempted_arguments or {}),
        }
    return _compact(envelope)


def _visible_read_key(read: dict) -> str:
    """Return a semantic+payload identity for one model-visible read.

    ``read_subtable`` defaults to ``limit=20``. Canonical state deliberately preserves whether
    that default was authored explicitly, but the two forms expose the same observation and need
    not duplicate all rows in every later prompt. Inline previews retain their own limit semantics
    because their ``note`` distinguishes them from an actual read call.
    """
    comparable = deepcopy(read)
    comparable.pop("from_step", None)
    comparable.pop("equivalent_from_steps", None)
    comparable["columns"] = list(comparable.get("columns") or [])
    comparable["conditions"] = deepcopy(comparable.get("conditions"))
    comparable["order_by"] = list(comparable.get("order_by") or [])
    comparable["offset"] = comparable.get("offset") or 0
    if comparable.get("note") is None and comparable.get("limit") is None:
        comparable["limit"] = 20
    return json.dumps(
        comparable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _deduplicate_visible_reads(entry: dict) -> None:
    """Keep the latest copy of completely equivalent reads and retain their source steps."""
    reads = entry.get("reads")
    if not isinstance(reads, list) or len(reads) < 2:
        return
    latest_by_key: dict[str, tuple[dict, list[str]]] = {}
    order: list[str] = []
    for raw_read in reads:
        if not isinstance(raw_read, dict):
            # Malformed historical state is rendered unchanged rather than guessed into a group.
            key = f"non_object:{len(order)}"
            order.append(key)
            latest_by_key[key] = (deepcopy(raw_read), [])
            continue
        read = deepcopy(raw_read)
        key = _visible_read_key(read)
        observed_steps = [
            step
            for step in read.pop("equivalent_from_steps", [])
            if isinstance(step, str) and step
        ]
        if isinstance(read.get("from_step"), str) and read["from_step"]:
            observed_steps.append(read["from_step"])
        if key not in latest_by_key:
            order.append(key)
            latest_by_key[key] = (read, observed_steps)
            continue
        _previous, previous_steps = latest_by_key[key]
        latest_by_key[key] = (read, previous_steps + observed_steps)

    compacted = []
    for key in order:
        read, steps = latest_by_key[key]
        unique_steps = list(dict.fromkeys(steps))
        if len(unique_steps) > 1:
            read["equivalent_from_steps"] = unique_steps
        compacted.append(read)
    entry["reads"] = compacted


def _contains_exact_string(value: object, target: str) -> bool:
    """Conservatively detect a resident reference without interpreting model-authored text."""
    if isinstance(value, str):
        return value == target
    if isinstance(value, dict):
        return any(_contains_exact_string(item, target) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_exact_string(item, target) for item in value)
    return False


def _zero_filter_summary_candidate(
    name: str,
    entry: object,
    visible: dict,
) -> tuple[str, str] | None:
    """Return a grouping key for an unobserved, unreferenced zero-row filter handle."""
    if not isinstance(entry, dict):
        return None
    derivation = entry.get("derivation")
    if (
        entry.get("row_count") != 0
        or entry.get("reads")
        or entry.get("inspected_columns")
        or not isinstance(derivation, dict)
        or derivation.get("operator") != "condition_filter"
    ):
        return None
    inputs = derivation.get("inputs")
    if not isinstance(inputs, list):
        return None
    source = next(
        (
            item.get("ref")
            for item in inputs
            if isinstance(item, dict)
            and item.get("kind") == "table"
            and item.get("role") == "input"
            and isinstance(item.get("ref"), str)
        ),
        None,
    )
    if not source:
        return None

    # A handle remains fully rendered whenever any other resident fact depends on it.
    if _contains_exact_string(visible.get("plan"), name) or _contains_exact_string(
        visible.get("values"), name
    ):
        return None
    for other_name, other_entry in (visible.get("tables") or {}).items():
        if other_name != name and _contains_exact_string(other_entry, name):
            return None

    output_shape = {
        key: deepcopy(entry[key])
        for key in ("columns", "column_namespaces")
        if entry.get(key) is not None
    }
    return source, json.dumps(
        output_shape,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _fold_unreferenced_zero_row_filters(visible: dict) -> None:
    """Fold repeated empty filter branches into fact-only model-visible summaries.

    A singleton stays expanded, as do handles that were read, inspected, or referenced by another
    resident fact. Canonical state is not touched, so every handle remains available to execution,
    replay, and audit.
    """
    tables = visible.get("tables")
    if not isinstance(tables, dict):
        return
    groups: dict[tuple[str, str], list[str]] = {}
    for name, entry in tables.items():
        candidate = _zero_filter_summary_candidate(name, entry, visible)
        if candidate is not None:
            groups.setdefault(candidate, []).append(name)

    runs = []
    folded_names = set()
    for (source, _shape_key), names in groups.items():
        if len(names) < 2:
            continue
        first = tables[names[0]]
        shape = {
            key: deepcopy(first[key])
            for key in ("columns", "column_namespaces")
            if first.get(key) is not None
        }
        attempts = []
        for name in names:
            entry = tables[name]
            semantics = (entry.get("derivation") or {}).get("semantics") or {}
            attempts.append({
                "table": name,
                "from_step": entry.get("created_by"),
                "predicate": deepcopy(semantics.get("predicate")),
            })
        runs.append({
            "input": source,
            **shape,
            "attempts": attempts,
        })
        folded_names.update(names)

    if not runs:
        return
    for name in folded_names:
        tables.pop(name, None)
    summaries = visible.setdefault("fact_summaries", {})
    summaries["unreferenced_zero_row_filter_runs"] = runs


def _compact_state_columns(state: dict | None) -> dict:
    """Compact only the model-visible rendering; canonical harness snapshots stay unchanged."""
    visible = deepcopy(state or {"plan": [], "tables": {}, "values": {}})
    tables = visible.get("tables")
    if isinstance(tables, dict):
        for name, entry in list(tables.items()):
            if isinstance(entry, dict):
                _deduplicate_visible_reads(entry)
                tables[name] = _compact_output_columns(entry)
        _fold_unreferenced_zero_row_filters(visible)
    for item in visible.get("plan") or []:
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence")
        if not isinstance(evidence, dict):
            continue
        # Canonical harness snapshots retain the grounded evidence output for replay/audit, but
        # repeating that full schema/table payload inside every resident plan item duplicates the
        # authoritative table state and can dominate every later prompt. The model needs only the
        # grounded step identity to understand plan progress; factual reuse still comes from tables
        # and values in CURRENT ENVIRONMENT STATE.
        item["evidence"] = {
            key: deepcopy(evidence[key])
            for key in ("step_id", "tool")
            if evidence.get(key) is not None
        }
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
    preserve_all_reasoning: bool = False,
    history_policy: str = HISTORY_POLICY_RECENT,
    history_head_turns: int = 0,
) -> list[dict]:
    """Render a bounded transcript of harness-successful assistant/tool pairs.

    Rejected assistant text never becomes context. Its structured error is carried only in the
    current user message, so training and online inference cannot teach the model to imitate an
    invalid call. ``history_turns=0`` retains every legal pair for experiments; production callers
    should use an explicit positive bound.
    """
    if history_turns < 0:
        raise ValueError("history_turns must be non-negative")
    if history_policy not in HISTORY_POLICIES:
        raise ValueError(f"unknown history_policy {history_policy!r}")
    if history_head_turns < 0:
        raise ValueError("history_head_turns must be non-negative")
    if history_policy == HISTORY_POLICY_RECENT and history_head_turns:
        raise ValueError("recent history_policy does not accept history_head_turns")
    if preserve_all_reasoning and history_policy != HISTORY_POLICY_RECENT:
        raise ValueError("complete reasoning retention currently requires recent history_policy")
    initial = first_user_message(overview, question, external_knowledge)
    if not legal_history:
        content = initial
        if state or last_error:
            content += "\n\n" + state_context_message(state, last_error)
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]

    if history_turns == 0:
        retained = legal_history
        older = []
    elif history_policy == HISTORY_POLICY_RECENT:
        retained = legal_history[-history_turns:]
        older = legal_history[:-history_turns] if preserve_all_reasoning else []
    else:
        head = legal_history[:history_head_turns]
        tail_start = max(history_head_turns, len(legal_history) - history_turns)
        retained = [*head, *legal_history[tail_start:]]
        older = []
    if older:
        older_reasons = []
        for item in older:
            assistant = item.get("assistant")
            if not isinstance(assistant, str) or not assistant.strip():
                raise ValueError("rolling legal history has an empty assistant action")
            match = re.fullmatch(
                r"\s*<think>(?P<reasoning>.*?)</think>\s*\{.*\}\s*",
                assistant,
                re.DOTALL,
            )
            if match is None or not match.group("reasoning").strip():
                raise ValueError(
                    "complete reasoning retention requires canonical assistant history"
                )
            older_reasons.append(match.group("reasoning").strip())
        initial += (
            "\n\nOLDER SUCCESSFUL MODEL REASONING\n"
            "The following complete model-authored reasoning is continuity context, not factual "
            "evidence. Its older tool calls and observations are outside the four-pair window. "
            "Revise it whenever current harness state or feedback disagrees.\n"
            + json.dumps(
                {"reasoning": older_reasons},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
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
    """Replay-compatible parser for active and historical assistant artifacts."""
    m = _tool_call_payloads(text)
    if not m:
        try:
            _, call = parse_action_carrier(text)
        except ActionCarrierError:
            start = text.rfind("</think>")
            raw = _extract_balanced_json(
                text,
                start + len("</think>") if start >= 0 else 0,
            )
            if raw is not None:
                try:
                    call = _loads_tool_call(raw)
                except json.JSONDecodeError:
                    call = None
            else:
                call = None
            if call is None:
                call = _repair_truncated_answer_call(text)
            if call is None:
                raise ProtocolError("no complete JSON tool action found")
    else:
        try:
            call = _loads_tool_call(m[-1])  # last block wins for historical replay
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


def parse_assistant_structure_strict(text: str) -> tuple[str, str, dict]:
    """Parse the carrier and action object before tool-specific argument validation."""
    try:
        think, call = parse_action_carrier(text)
    except ActionCarrierError as exc:
        raise ProtocolError(
            str(exc),
            code=exc.code,
            details=exc.details,
        ) from exc
    if not isinstance(call, dict):
        raise ProtocolError(
            "action JSON must be an object",
            code="action_json_not_object",
            details={"received_type": type(call).__name__},
        )
    if set(call) != {"tool", "arguments"}:
        raise ProtocolError(
            'action must contain exactly "tool" and "arguments" keys',
            code="invalid_action_keys",
            details={
                "required_keys": ["arguments", "tool"],
                "received_keys": sorted(str(key) for key in call),
            },
        )
    tool = call.get("tool")
    args = call.get("arguments")
    if not isinstance(tool, str) or not tool:
        raise ProtocolError(
            "action.tool must be one non-empty tool name",
            code="invalid_tool_name",
            details={"received_type": type(tool).__name__},
        )
    if not isinstance(args, dict):
        raise ProtocolError(
            'action must have an "arguments" object',
            code="arguments_not_object",
            details={"received_type": type(args).__name__},
            attempted_tool=tool,
        )
    return think, tool, args


def validate_model_action(tool: str, args: dict) -> None:
    """Validate one structurally parsed action and attach its public argument contract."""
    if tool not in TOOLS:
        raise ProtocolError(
            f"unknown tool {tool!r}; legal tools: {sorted(TOOLS)}",
            code="unknown_tool",
            details={"legal_tools": sorted(TOOLS)},
            attempted_tool=tool,
            attempted_arguments=args,
        )
    try:
        validate_model_arguments(tool, args)
    except ProtocolError as exc:
        if exc.code == "protocol_error":
            exc.code = "argument_validation_error"
        exc.failure_type = exc.failure_type or "argument_validation_error"
        exc.attempted_tool = tool
        exc.attempted_arguments = deepcopy(args)
        required, optional = MODEL_ARG_SCHEMA[tool]
        exc.details = {
            **exc.details,
            "expected_arguments": {
                "required": sorted(required),
                "optional": sorted(optional),
            },
        }
        raise


def parse_assistant_strict(
    text: str,
    *,
    adjacent_guard: AdjacentActionGuard | None = None,
    step_id: str | None = None,
) -> tuple[str, str, dict]:
    """Parse one active action and optionally enforce exact adjacent-action progress.

    The guard compares only the parsed ``tool`` and ``arguments`` of two consecutive model
    turns. It ignores reasoning text and JSON key order, and it never searches farther back.
    """
    try:
        think, tool, args = parse_assistant_structure_strict(text)
    except ProtocolError:
        if adjacent_guard is not None:
            adjacent_guard.clear()
        raise
    if adjacent_guard is not None:
        if not step_id:
            raise ValueError("step_id is required when adjacent_guard is provided")
        adjacent_guard.observe(tool, args, step_id=step_id)
    try:
        validate_model_action(tool, args)
    except ProtocolError:
        if adjacent_guard is not None:
            adjacent_guard.mark_last("rejected")
        raise
    return think, tool, args


def parse_legacy_assistant_strict(text: str) -> tuple[str, str, dict]:
    """Parse the retired tagged carrier only for deterministic artifact migration."""
    try:
        think, call = parse_legacy_tagged_action_carrier(text)
    except ActionCarrierError as exc:
        raise ProtocolError(str(exc)) from exc
    if set(call) != {"tool", "arguments"}:
        raise ProtocolError('legacy action must contain exactly "tool" and "arguments" keys')
    tool = call.get("tool")
    args = call.get("arguments")
    if tool not in TOOLS:
        raise ProtocolError(f"unknown tool {tool!r}; legal tools: {sorted(TOOLS)}")
    if not isinstance(args, dict):
        raise ProtocolError('legacy action must have an "arguments" object')
    validate_model_arguments(tool, args)
    return think, tool, args
