"""Public structural contract for model-authored table-tool actions.

This module contains only callable syntax and operator-local semantics.  It is shared by the
student-prompt renderer and the strict protocol validator so that nested field names and enum
values cannot drift between what the model sees and what the harness accepts.

Concrete task examples and policy advice do not belong here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolContract:
    required: tuple[str, ...]
    optional: tuple[str, ...]
    semantics: str


PLAN_OPERATIONS = ("create", "add", "update", "delete")
PLAN_STATUSES = ("pending", "in_progress", "done", "blocked")
CONDITION_COMPARISON_OPERATORS = ("=", "!=", ">", ">=", "<", "<=")
CONDITION_VALUE_OPERATORS = ("like", "contains")
CONDITION_DATE_OPERATORS = ("on_date",)
SCALAR_OPERATIONS = (
    "add",
    "subtract",
    "multiply",
    "divide",
    "percent",
    "percent_change",
    "date_diff_days",
)
SCALAR_OPERAND_KEYSETS = (
    frozenset({"value"}),
    frozenset({"value_ref"}),
    frozenset({"value_ref", "column"}),
)
ROW_DATE_EXPRESSION_OPERATIONS = (
    "date_diff_days",
    "extract_year",
)
ROW_DATE_EXPRESSION_OPERAND_KEYSETS = (
    frozenset({"column"}),
    frozenset({"value"}),
)
JOIN_ITEM_REQUIRED = frozenset({"table", "on"})
JOIN_ITEM_OPTIONAL = frozenset({"type", "role"})
JOIN_EDGE_KEYS = frozenset({"left", "right"})
JOIN_TYPES = ("inner", "left", "cross")
AGGREGATION_REQUIRED = frozenset({"op", "column", "as"})
AGGREGATION_OPTIONAL = frozenset({"where"})
AGGREGATION_OPERATIONS = ("sum", "count", "count_distinct", "mean", "min", "max")
AGGREGATION_LAYOUTS = ("rows", "columns")
SET_OPERATIONS = ("union", "union_all", "intersect", "except")


PUBLIC_TOOL_CONTRACTS: dict[str, ToolContract] = {
    "plan": ToolContract(
        ("ops",),
        (),
        "update harness-owned control state. ops is a list of create/add/update/delete "
        "items with id, goal, status (pending|in_progress|done|blocked), and optional evidence "
        "(a prior step id). Plans contain no answer values and are not factual evidence.",
    ),
    "describe_table": ToolContract(
        ("tables",),
        (),
        "reveal columns, types, primary keys, and foreign keys for a "
        "non-empty list of catalog tables.",
    ),
    "inspect_column": ToolContract(
        ("table", "column"),
        ("top_k",),
        "reveal distinct-count, frequent values, and NULL "
        "presence for one column.",
    ),
    "read_subtable": ToolContract(
        ("table",),
        ("limit", "columns"),
        "observe up to 20 rows (limit 1..20). This does "
        "not derive or reshape a table; omitted columns means all columns.",
    ),
    "condition_filter": ToolContract(
        ("table", "conditions"),
        ("return_columns",),
        "derive matching rows. A predicate "
        "uses column with op =|!=|>|>=|<|<= and value or column_value; op in uses values or "
        "in_table; between uses low/high; like or contains uses value; is_null needs no value. "
        "A computed scalar uses value_ref to its producing step. Compose predicates with and/or/not.",
    ),
    "project": ToolContract(
        ("table", "expressions"),
        ("distinct",),
        "derive exactly the listed output expressions "
        "and order; expressions may use 'expr AS alias'. distinct defaults false. Projection "
        "selects columns/expressions but does not turn category rows into columns.",
    ),
    "scalar_compute": ToolContract(
        ("operation", "operands"),
        ("result_name",),
        "derive a grounded 1x1 table. operation "
        "is add|subtract|multiply|divide|percent|percent_change|date_diff_days. Each operand is "
        "exactly value, value_ref, or value_ref+column; value_ref cites the step that produced the "
        "resident one-row table. Operand order is semantic.",
    ),
    "join_tables": ToolContract(
        ("base", "joins"),
        ("base_role",),
        "derive one connected join component. joins is an "
        "ordered non-empty list; each item has table, on, optional type (inner|left|cross), and "
        "optional role. Each on pair has left as an exact logical relation.column already present "
        "and right as a bare column of the newly attached table; cross uses on=[]. Output logical "
        "columns remain flat relation.column names. Use roles only to disambiguate repeated relations.",
    ),
    "group_aggregate": ToolContract(
        ("table", "group_by", "aggregations"),
        ("passthrough", "output_layout", "category_values", "output_columns"),
        "derive grouped results. group_by=[] is one global "
        "group. Each aggregation has op sum|count|count_distinct|mean|min|max, column, as, and "
        "optional where predicate. output_layout defaults rows. columns layout requires one group "
        "column, one aggregation, ordered category_values, and optional equally sized output_columns.",
    ),
    "extreme_value_select": ToolContract(
        ("table", "order_by"),
        ("top_k", "return_columns"),
        "derive rows ordered by a "
        "list of columns optionally ending in DESC, retaining top_k when supplied.",
    ),
    "set_op": ToolContract(
        ("left", "right", "op"),
        (),
        "derive union|union_all|intersect|except over two "
        "column-compatible tables.",
    ),
    "answer_from_context": ToolContract(
        ("evidence",),
        ("reason",),
        'terminal. evidence is exactly {"table": handle} '
        "and must cite the grounded table whose rows, columns, and column order are the answer. "
        "Scalar answers cite their grounded 1x1 table; the call contains no authored answer values.",
    ),
}

# Action-block v32 and the v33 sequential-segment diagnostic share this frozen
# primitive surface even when the independently selectable atomic scheme evolves.
ACTION_BLOCK_PUBLIC_TOOL_CONTRACTS = dict(PUBLIC_TOOL_CONTRACTS)
ACTION_BLOCK_PUBLIC_TOOL_ARGUMENTS: dict[
    str, tuple[tuple[str, ...], tuple[str, ...]]
] = {
    tool: (contract.required, contract.optional)
    for tool, contract in ACTION_BLOCK_PUBLIC_TOOL_CONTRACTS.items()
}

# Atomic version37 adds only two bounded capabilities that were missing from the observation
# surface: row-addressed reads and typed row-wise date calculations.  The frozen action-block
# schemes above deliberately retain their previous primitive contract.
PUBLIC_TOOL_CONTRACTS = dict(PUBLIC_TOOL_CONTRACTS)
PUBLIC_TOOL_CONTRACTS["read_subtable"] = ToolContract(
    ("table",),
    ("limit", "columns", "conditions", "order_by", "offset"),
    "observe up to 20 matching rows without deriving a table. conditions uses the same typed "
    "predicate shape as condition_filter, including on_date for calendar-date matching. order_by "
    "is a list of exact columns optionally ending in ASC or DESC. offset is a non-negative row "
    "offset and requires order_by so pagination is deterministic. Omitted columns means all columns.",
)
PUBLIC_TOOL_CONTRACTS["project"] = ToolContract(
    ("table", "expressions"),
    ("distinct",),
    "derive exactly the listed output expressions and order. Each expression is either an exact "
    "column/'expr AS alias' string or a typed date expression {op,operands,as}; op is "
    "date_diff_days (start,end) or extract_year (date), and each operand is exactly {column} or "
    "{value}. distinct defaults false. Projection does not turn category rows into columns.",
)

PUBLIC_TOOL_ARGUMENTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    tool: (contract.required, contract.optional)
    for tool, contract in PUBLIC_TOOL_CONTRACTS.items()
}

# Atomic version40 is an isolated prompt/tool-surface diagnostic. It keeps every version39
# execution semantic, including the join contract, but removes the model-visible plan tool and
# renames the read-only row observer to align with inspect_column. The executor retains a
# replay-only read_subtable alias; it is not exposed through this public contract.
VERSION40_PUBLIC_TOOL_CONTRACTS: dict[str, ToolContract] = {}
for _tool, _contract in PUBLIC_TOOL_CONTRACTS.items():
    if _tool == "plan":
        continue
    if _tool == "read_subtable":
        VERSION40_PUBLIC_TOOL_CONTRACTS["inspect_rows"] = ToolContract(
            _contract.required,
            _contract.optional,
            "observe up to 20 matching rows without deriving a table. conditions uses the same "
            "typed predicate shape as condition_filter, including on_date for calendar-date "
            "matching. order_by is a list of exact columns optionally ending in ASC or DESC. "
            "offset is a non-negative row offset and requires order_by so pagination is "
            "deterministic. Omitted columns means all columns.",
        )
        continue
    VERSION40_PUBLIC_TOOL_CONTRACTS[_tool] = _contract

VERSION40_PUBLIC_TOOL_ARGUMENTS: dict[
    str, tuple[tuple[str, ...], tuple[str, ...]]
] = {
    tool: (contract.required, contract.optional)
    for tool, contract in VERSION40_PUBLIC_TOOL_CONTRACTS.items()
}

# Atomic version44 starts a new isolated checkpoint candidate from version39. It deliberately
# keeps the version39 relational and terminal semantics while changing only the public perception
# surface approved for the next SFT generation round:
# - plan is absent;
# - read_subtable is exposed only as inspect_rows;
# - search_values performs one-table, paginated fuzzy value discovery.
VERSION44_PUBLIC_TOOL_CONTRACTS: dict[str, ToolContract] = {}
for _tool, _contract in PUBLIC_TOOL_CONTRACTS.items():
    if _tool == "plan":
        continue
    if _tool == "read_subtable":
        VERSION44_PUBLIC_TOOL_CONTRACTS["inspect_rows"] = ToolContract(
            _contract.required,
            _contract.optional,
            "observe up to 20 matching rows without deriving a table. conditions uses the same "
            "typed predicate shape as condition_filter, including on_date for calendar-date "
            "matching. order_by is a list of exact columns optionally ending in ASC or DESC. "
            "offset is a non-negative row offset and requires order_by so pagination is "
            "deterministic. Omitted columns means all columns.",
        )
        continue
    VERSION44_PUBLIC_TOOL_CONTRACTS[_tool] = _contract

_version44_items = list(VERSION44_PUBLIC_TOOL_CONTRACTS.items())
_version44_inspect_index = next(
    index for index, (tool, _) in enumerate(_version44_items)
    if tool == "inspect_column"
)
_version44_items.insert(
    _version44_inspect_index + 1,
    (
        "search_values",
        ToolContract(
            ("table", "query"),
            ("column", "limit", "offset"),
            "fuzzily search actual stored values in exactly one table without deriving a table. "
            "column optionally restricts the search; when omitted every column is searched. "
            "limit defaults to 20 and cannot exceed 20. offset reads the next page in one stable "
            "relevance ordering. Each match identifies its exact table, column, stored value, "
            "frequency, and deterministic lexical match kind.",
        ),
    ),
)
VERSION44_PUBLIC_TOOL_CONTRACTS = dict(_version44_items)

VERSION44_PUBLIC_TOOL_ARGUMENTS: dict[
    str, tuple[tuple[str, ...], tuple[str, ...]]
] = {
    tool: (contract.required, contract.optional)
    for tool, contract in VERSION44_PUBLIC_TOOL_CONTRACTS.items()
}

# Version45 keeps version44's public calls and changes only search execution/observation semantics:
# SQL recalls a deterministic bounded candidate pool before lexical fuzzy ranking. Saturation is
# explicit so the model can narrow table/column/query instead of mistaking bounded recall for an
# exhaustive domain scan.
VERSION45_PUBLIC_TOOL_CONTRACTS = dict(VERSION44_PUBLIC_TOOL_CONTRACTS)
_version45_search = VERSION45_PUBLIC_TOOL_CONTRACTS["search_values"]
VERSION45_PUBLIC_TOOL_CONTRACTS["search_values"] = ToolContract(
    _version45_search.required,
    _version45_search.optional,
    "search a deterministic bounded pool of actual stored values in exactly one table without "
    "deriving a table. column optionally restricts the search; when omitted every column is "
    "searched. limit defaults to 20 and cannot exceed 20. offset reads the next page in one stable "
    "candidate-pool relevance ordering. Each match identifies its exact table, column, stored "
    "value, frequency, and lexical match kind. candidate_truncated=true means candidate recall "
    "saturated and the query should be narrowed rather than treated as exhaustive. Exact or "
    "case-insensitive exact hits suppress broader fuzzy alternatives.",
)

VERSION45_PUBLIC_TOOL_ARGUMENTS: dict[
    str, tuple[tuple[str, ...], tuple[str, ...]]
] = {
    tool: (contract.required, contract.optional)
    for tool, contract in VERSION45_PUBLIC_TOOL_CONTRACTS.items()
}


def _choices(values: tuple[str, ...]) -> str:
    return "|".join(values)


def render_action_grammar() -> str:
    """Render only high-entropy nested shapes, not task examples or solution recipes.

    Simple top-level tools remain fully specified by ``PUBLIC_TOOL_CONTRACTS``. Repeating their
    low-entropy call shapes here would make a syntax reference look like a tool-selection policy.
    """
    return "\n".join(
        (
            "HIGH-ENTROPY ACTION GRAMMAR",
            "Notation: str is a JSON string and scalar is string|number|boolean|null. The shapes "
            "below expand only nested arguments that are easy to confuse; they do not imply tool "
            "priority. Emit ordinary valid JSON values, never the type words.",
            'condition: {"column":"str","op":"' + _choices(CONDITION_COMPARISON_OPERATORS)
            + '","value":scalar} | same with "column_value":"str" or "value_ref":"step_k" '
            '| {"column":"str","op":"in","values":[scalar,...]} '
            '| {"column":"str","op":"in","in_table":"handle"} '
            '| {"column":"str","op":"between","low":scalar,"high":scalar} '
            '| {"column":"str","op":"' + _choices(CONDITION_VALUE_OPERATORS)
            + '","value":scalar} | {"column":"str","op":"'
            + _choices(CONDITION_DATE_OPERATORS)
            + '","value":"YYYY-MM-DD"} | {"column":"str","op":"is_null"} '
            '| {"and":[condition,...]} | {"or":[condition,...]} | {"not":condition}',
            'project.expressions: ["exact_column", "scalar expression AS output_name", ...]; '
            "an exact dotted logical column, including spaces, is one whole JSON string; a typed "
            'date expression is {"op":"' + _choices(ROW_DATE_EXPRESSION_OPERATIONS)
            + '","operands":[{"column":"str"}|{"value":scalar},...],"as":"output_name"}',
            'scalar_compute.operands: [{"value":scalar} | {"value_ref":"step_k"} | '
            '{"value_ref":"step_k","column":"metric"}, ...]; operation: '
            + _choices(SCALAR_OPERATIONS),
            'join_tables arguments: {"base":"table_or_handle","joins":['
            '{"table":"new_table","on":[{"left":"known_relation.column",'
            '"right":"new_bare_column"}]},...]}; omit roles for ordinary joins. Each item may add '
            '"type":"' + _choices(JOIN_TYPES)
            + '"; cross uses on=[]. Only a repeated relation may add base_role/role identifier '
            "strings. For left: copy a dotted name exactly when the base state exposes "
            "column_namespaces; otherwise qualify its bare base column as base_or_role.column. "
            "Right is always a bare column of the newly attached table",
            'group_aggregate.aggregations: [{"op":"' + _choices(AGGREGATION_OPERATIONS)
            + '","column":"column_or_*","as":"output_name"},...]; an aggregation may add '
            '"where":condition. '
            'output_layout is "' + _choices(AGGREGATION_LAYOUTS)
            + '"; columns layout requires one group_by column, one aggregation, ordered '
            "category_values, and equally ordered optional output_columns",
            'answer_from_context.evidence: {"table":"exact_result_handle"}; scalar answers cite '
            "their grounded 1x1 table",
        )
    )
