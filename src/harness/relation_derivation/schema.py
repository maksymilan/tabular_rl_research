"""Canonical schema and validation for relation derivation metadata."""
from __future__ import annotations

from typing import Any, Protocol


DERIVATION_SCHEMA = "relation-derivation-v1"
SUPPORTED_TABLE_OPERATORS = frozenset({
    "condition_filter",
    "project",
    "scalar_compute",
    "join_tables",
    "group_aggregate",
    "extreme_value_select",
    "set_op",
    # Historical trajectories may still replay this table-producing action.
    "pivot",
})


class RelationInspector(Protocol):
    """Minimal executor inspection surface needed for factual derivation metadata."""

    def table_columns(self, table: str) -> list[str]:
        ...

    def count_null_rows(self, table: str, column: str) -> int:
        ...


_SEMANTIC_CONTRACTS = {
    "condition_filter": {
        "row_operation", "column_operation", "predicate",
        "predicate_columns", "projected_columns",
    },
    "project": {
        "row_operation", "column_operation", "column_lineage",
    },
    "scalar_compute": {
        "row_operation", "column_operation", "operation", "result_column",
    },
    "join_tables": {
        "row_operation", "column_operation", "edges",
    },
    "group_aggregate": {
        "row_operation", "column_operation", "row_grain", "layout", "aggregations",
    },
    "extreme_value_select": {
        "row_operation", "column_operation", "order_by", "top_k", "projected_columns",
    },
    "set_op": {
        "row_operation", "column_operation", "operation", "duplicate_semantics",
    },
    "pivot": {
        "row_operation", "column_operation", "key_column", "value_column",
        "key_values", "output_columns", "historical_operator",
    },
}

_OPERATOR_OPERATIONS = {
    "condition_filter": ({"filter"}, {"preserve", "project"}),
    "project": ({"preserve", "deduplicate"}, {"project"}),
    "scalar_compute": ({"scalar"}, {"create"}),
    "join_tables": ({"join"}, {"concatenate_namespaced", "project", "merge_legacy"}),
    "group_aggregate": (
        {"aggregate", "deduplicate", "preserve"},
        {"group_and_compute", "project", "preserve"},
    ),
    "extreme_value_select": ({"order", "ordered_prefix"}, {"preserve", "project"}),
    "set_op": ({"set"}, {"align_by_position"}),
    "pivot": ({"reshape"}, {"pivot"}),
}

if (
    frozenset(_SEMANTIC_CONTRACTS) != SUPPORTED_TABLE_OPERATORS
    or frozenset(_OPERATOR_OPERATIONS) != SUPPORTED_TABLE_OPERATORS
):
    raise RuntimeError(
        "relation derivation semantic contracts and supported operators must match"
    )


def _contains_policy_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).casefold() in {"advice", "recommendation", "recommended_action"}
            or _contains_policy_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_policy_key(item) for item in value)
    return False


def validate_relation_derivation(
    derivation: dict,
    output_columns: list[str] | None = None,
) -> None:
    """Reject schema drift before derivation metadata enters a trajectory or resident state."""
    if not isinstance(derivation, dict):
        raise ValueError("relation derivation must be an object")
    if set(derivation) != {"schema", "operator", "inputs", "semantics"}:
        raise ValueError(
            "relation derivation requires exactly schema/operator/inputs/semantics"
        )
    if derivation.get("schema") != DERIVATION_SCHEMA:
        raise ValueError(f"unsupported relation derivation schema: {derivation.get('schema')!r}")
    if derivation.get("operator") not in SUPPORTED_TABLE_OPERATORS:
        raise ValueError(f"unsupported derivation operator: {derivation.get('operator')!r}")
    inputs = derivation.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("relation derivation inputs must be a list")
    for item in inputs:
        if (
            not isinstance(item, dict)
            or item.get("kind") not in {"table", "value", "constant"}
            or not isinstance(item.get("role"), str)
        ):
            raise ValueError(f"invalid relation derivation input: {item!r}")
        if item["kind"] in {"table", "value"} and not isinstance(item.get("ref"), str):
            raise ValueError(f"relation derivation {item['kind']} input requires ref: {item!r}")
        if item["kind"] == "constant" and "value" not in item:
            raise ValueError(f"relation derivation constant input requires value: {item!r}")
    semantics = derivation.get("semantics")
    if not isinstance(semantics, dict):
        raise ValueError("relation derivation semantics must be an object")
    operator = derivation["operator"]
    missing = _SEMANTIC_CONTRACTS[operator] - set(semantics)
    if missing:
        raise ValueError(
            f"{operator} relation derivation is missing semantic fields: {sorted(missing)}"
        )
    if not isinstance(semantics.get("row_operation"), str):
        raise ValueError("relation derivation semantics require string row_operation")
    if not isinstance(semantics.get("column_operation"), str):
        raise ValueError("relation derivation semantics require string column_operation")
    allowed_rows, allowed_columns = _OPERATOR_OPERATIONS[operator]
    if semantics["row_operation"] not in allowed_rows:
        raise ValueError(
            f"{operator} has invalid row_operation {semantics['row_operation']!r}"
        )
    if semantics["column_operation"] not in allowed_columns:
        raise ValueError(
            f"{operator} has invalid column_operation {semantics['column_operation']!r}"
        )
    if operator == "project":
        lineage = semantics.get("column_lineage")
        if not isinstance(lineage, list) or any(
            not isinstance(item, dict)
            or set(item) != {"output", "sources", "kind", "expression"}
            or not isinstance(item.get("output"), str)
            or not isinstance(item.get("sources"), list)
            for item in lineage
        ):
            raise ValueError("project derivation requires complete column_lineage entries")
        if output_columns is not None and [item["output"] for item in lineage] != output_columns:
            raise ValueError(
                "project derivation lineage must cover the output schema in order"
            )
    if operator == "join_tables":
        edges = semantics.get("edges")
        if not isinstance(edges, list) or any(
            not isinstance(edge, dict)
            or edge.get("index") != index
            or not isinstance(edge.get("on"), list)
            or edge.get("join_type") not in {"inner", "left", "cross"}
            for index, edge in enumerate(edges)
        ):
            raise ValueError("join derivation requires ordered, typed edge semantics")
        joined_inputs = [item for item in inputs if item.get("role") == "joined"]
        if len(edges) != len(joined_inputs):
            raise ValueError("join derivation requires one ordered edge per joined input")
    if _contains_policy_key(derivation):
        raise ValueError("relation derivation must contain facts only, never advice")
