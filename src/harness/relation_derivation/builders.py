"""Operator-specific builders for fact-only relation derivation metadata."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .lineage import (
    condition_columns,
    condition_inputs,
    expression_core,
    expression_kind,
    expression_source_columns,
)
from .schema import (
    DERIVATION_SCHEMA,
    SUPPORTED_TABLE_OPERATORS,
    RelationInspector,
    validate_relation_derivation,
)


def _table_input(role: str, ref: Any, **metadata: Any) -> dict:
    item = {"kind": "table", "role": role, "ref": ref}
    item.update({key: value for key, value in metadata.items() if value is not None})
    return item


def _condition_filter(
    args: dict,
    output_columns: list[str],
    _inspector: RelationInspector,
    _output_table: str,
) -> dict:
    return_columns = list(args.get("return_columns") or [])
    return {
        "inputs": (
            [_table_input("input", args.get("table"))]
            + condition_inputs(args.get("conditions"))
        ),
        "semantics": {
            "row_operation": "filter",
            "predicate": deepcopy(args.get("conditions")),
            "predicate_columns": condition_columns(args.get("conditions")),
            "column_operation": "project" if return_columns else "preserve",
            "projected_columns": return_columns or list(output_columns),
        },
    }


def _project(
    args: dict,
    output_columns: list[str],
    inspector: RelationInspector,
    _output_table: str,
) -> dict:
    input_table = args.get("table")
    input_columns = inspector.table_columns(input_table)
    expressions = list(args.get("expressions") or [])
    lineage = []
    output_index = 0
    authored_expressions = expressions or ["*"]
    for expression in authored_expressions:
        core = expression_core(str(expression))
        if core == "*":
            for source in input_columns:
                if output_index >= len(output_columns):
                    break
                lineage.append({
                    "output": output_columns[output_index],
                    "sources": [source],
                    "kind": "column",
                    "expression": "*",
                })
                output_index += 1
            continue
        sources = expression_source_columns(str(expression), input_columns)
        item = {
            "output": (
                output_columns[output_index]
                if output_index < len(output_columns)
                else f"column_{output_index + 1}"
            ),
            "sources": sources,
            "kind": expression_kind(str(expression), input_columns, sources),
            "expression": str(expression),
        }
        lineage.append(item)
        output_index += 1
    if output_index != len(output_columns):
        raise ValueError(
            "project derivation could not align authored expressions with output columns: "
            f"expressions={expressions!r}, output_columns={output_columns!r}"
        )
    return {
        "inputs": [_table_input("input", input_table)],
        "semantics": {
            "row_operation": "deduplicate" if args.get("distinct", False) else "preserve",
            "column_operation": "project",
            "column_lineage": lineage,
        },
    }


def _scalar_compute(
    args: dict,
    _columns: list[str],
    _inspector: RelationInspector,
    _output_table: str,
) -> dict:
    inputs = []
    for index, operand in enumerate(args.get("operands") or []):
        if isinstance(operand, dict) and isinstance(operand.get("value_ref"), str):
            item = {
                "kind": "value",
                "role": "operand",
                "ref": operand["value_ref"],
                "operand_index": index,
            }
            if isinstance(operand.get("column"), str):
                item["column"] = operand["column"]
        else:
            item = {
                "kind": "constant",
                "role": "operand",
                "operand_index": index,
                "value": deepcopy(operand.get("value") if isinstance(operand, dict) else operand),
            }
        inputs.append(item)
    return {
        "inputs": inputs,
        "semantics": {
            "row_operation": "scalar",
            "column_operation": "create",
            "operation": args.get("operation"),
            "result_column": args.get("result_name", "value"),
        },
    }


def _public_join(args: dict, output_table: str, inspector: RelationInspector) -> dict:
    base = args.get("base")
    inputs = [
        _table_input(
            "base",
            base,
            namespace=args.get("base_role") or base,
        )
    ]
    edges = []
    for index, edge in enumerate(args.get("joins") or []):
        if not isinstance(edge, dict):
            continue
        table = edge.get("table")
        namespace = edge.get("role") or table
        join_type = edge.get("type", "inner")
        on = deepcopy(edge.get("on") or [])
        inputs.append(_table_input("joined", table, namespace=namespace))
        item = {
            "index": index,
            "input": table,
            "namespace": namespace,
            "join_type": join_type,
            "on": on,
        }
        if join_type == "left" and on and isinstance(on[0], dict):
            right_key = on[0].get("right")
            if isinstance(namespace, str) and isinstance(right_key, str):
                logical_key = f"{namespace}.{right_key}"
                try:
                    item["null_extended_output_rows"] = inspector.count_null_rows(
                        output_table,
                        logical_key,
                    )
                except (KeyError, ValueError):
                    # Historical projections may omit the join key. The edge semantics remain
                    # complete; only this optional measured output statistic is unavailable.
                    pass
        edges.append(item)
    return {
        "inputs": inputs,
        "semantics": {
            "row_operation": "join",
            "column_operation": "concatenate_namespaced",
            "edges": edges,
        },
    }


def _symmetric_join(
    args: dict,
    output_table: str,
    inspector: RelationInspector,
) -> dict:
    left = args.get("left")
    right = args.get("right")
    left_namespace = args.get("left_alias") or left
    right_namespace = args.get("right_alias") or right
    how = args.get("how", "inner")
    on = deepcopy(args.get("on") or [])
    edge = {
        "index": 0,
        "left_input": left,
        "right_input": right,
        "left_namespace": left_namespace,
        "right_namespace": right_namespace,
        "join_type": how,
        "on": on,
    }
    if how == "left" and on and isinstance(on[0], dict):
        right_key = on[0].get("right")
        if isinstance(right_key, str):
            base = right_key.rsplit(".", 1)[-1]
            candidates = [
                column
                for column in inspector.table_columns(output_table)
                if column.casefold() == right_key.casefold()
                or column.casefold() == f"{right_namespace}.{base}".casefold()
                or column.casefold().endswith("." + base.casefold())
            ]
            if len(candidates) == 1:
                try:
                    edge["null_extended_output_rows"] = inspector.count_null_rows(
                        output_table,
                        candidates[0],
                    )
                except (KeyError, ValueError):
                    pass
    return {
        "inputs": [
            _table_input("left", left, namespace=left_namespace),
            _table_input("right", right, namespace=right_namespace),
        ],
        "semantics": {
            "row_operation": "join",
            "column_operation": "concatenate_namespaced",
            "edges": [edge],
        },
    }


def _historical_join(args: dict) -> dict:
    tables = args.get("tables")
    if not isinstance(tables, list):
        tables = [args.get("left"), args.get("right")]
    tables = [table for table in tables if table is not None]
    join_types = args.get("join_types")
    if isinstance(join_types, str):
        join_types = [join_types] * max(0, len(tables) - 1)
    if not isinstance(join_types, list):
        join_types = [args.get("join_type", "inner")] * max(0, len(tables) - 1)
    on = args.get("on") or []
    if on and isinstance(on, list) and isinstance(on[0], dict):
        on = [on]
    inputs = [
        _table_input("base" if index == 0 else "joined", table)
        for index, table in enumerate(tables)
    ]
    prefixes = args.get("prefixes")
    if isinstance(prefixes, list):
        for index, prefix in enumerate(prefixes[:len(inputs)]):
            if prefix is not None:
                inputs[index]["namespace"] = prefix
    edges = [
        {
            "index": index,
            "input": table,
            "join_type": (
                join_types[index]
                if index < len(join_types)
                else "inner"
            ),
            "on": deepcopy(on[index] if index < len(on) else []),
        }
        for index, table in enumerate(tables[1:])
    ]
    return {
        "inputs": inputs,
        "semantics": {
            "row_operation": "join",
            "column_operation": (
                "project" if args.get("return_columns") else "merge_legacy"
            ),
            "edges": edges,
            "historical_call_shape": True,
        },
    }


def _join_tables(
    args: dict,
    _columns: list[str],
    inspector: RelationInspector,
    output_table: str,
) -> dict:
    if args.get("base") is not None or args.get("joins") is not None:
        return _public_join(args, output_table, inspector)
    return _historical_join(args)


def _join(
    args: dict,
    _columns: list[str],
    inspector: RelationInspector,
    output_table: str,
) -> dict:
    return _symmetric_join(args, output_table, inspector)


def _group_aggregate(
    args: dict,
    _columns: list[str],
    _inspector: RelationInspector,
    _output_table: str,
) -> dict:
    group_by = list(args.get("group_by") or [])
    aggregations = []
    for aggregation in args.get("aggregations") or []:
        if not isinstance(aggregation, dict):
            continue
        item = {
            "output": aggregation.get("as"),
            "op": aggregation.get("op"),
            "source": aggregation.get("column", "*"),
        }
        if aggregation.get("where") is not None:
            item["predicate"] = deepcopy(aggregation.get("where"))
            item["condition_columns"] = condition_columns(aggregation.get("where"))
        aggregations.append(item)
    if aggregations:
        row_operation = "aggregate"
    elif group_by:
        row_operation = "deduplicate"
    else:
        # Historical direct executor calls permitted the empty/empty shape, which renders SELECT *.
        row_operation = "preserve"
    semantics = {
        "row_operation": row_operation,
        "column_operation": "group_and_compute" if aggregations else (
            "project" if group_by else "preserve"
        ),
        "row_grain": group_by,
        "layout": args.get("output_layout", "rows"),
        "aggregations": aggregations,
    }
    passthrough = args.get("passthrough")
    if passthrough:
        semantics["passthrough"] = list(passthrough)
    if args.get("output_layout", "rows") == "columns":
        semantics["category_axis"] = {
            "column": group_by[0] if group_by else None,
            "values": deepcopy(args.get("category_values") or []),
            "outputs": deepcopy(args.get("output_columns") or []),
        }
    predicate_inputs: list[dict] = []
    seen_predicate_inputs: set[tuple[str, str]] = set()
    for aggregation in args.get("aggregations") or []:
        if not isinstance(aggregation, dict):
            continue
        for item in condition_inputs(aggregation.get("where")):
            identity = (item["kind"], item["ref"])
            if identity not in seen_predicate_inputs:
                seen_predicate_inputs.add(identity)
                predicate_inputs.append(item)
    return {
        "inputs": [_table_input("input", args.get("table"))] + predicate_inputs,
        "semantics": semantics,
    }


def _extreme_value_select(
    args: dict,
    _columns: list[str],
    _inspector: RelationInspector,
    _output_table: str,
) -> dict:
    return_columns = list(args.get("return_columns") or [])
    return {
        "inputs": [_table_input("input", args.get("table"))],
        "semantics": {
            "row_operation": "ordered_prefix" if args.get("top_k") is not None else "order",
            "order_by": list(args.get("order_by") or []),
            "top_k": args.get("top_k"),
            "column_operation": "project" if return_columns else "preserve",
            "projected_columns": return_columns or list(_columns),
        },
    }


def _set_op(
    args: dict,
    _columns: list[str],
    _inspector: RelationInspector,
    _output_table: str,
) -> dict:
    operation = args.get("op")
    return {
        "inputs": [
            _table_input("left", args.get("left")),
            _table_input("right", args.get("right")),
        ],
        "semantics": {
            "row_operation": "set",
            "column_operation": "align_by_position",
            "operation": operation,
            "duplicate_semantics": (
                "preserve" if operation == "union_all" else "deduplicate"
            ),
        },
    }


def _pivot(
    args: dict,
    _columns: list[str],
    _inspector: RelationInspector,
    _output_table: str,
) -> dict:
    return {
        "inputs": [_table_input("input", args.get("table"))],
        "semantics": {
            "row_operation": "reshape",
            "column_operation": "pivot",
            "key_column": args.get("key_column"),
            "value_column": args.get("value_column"),
            "key_values": deepcopy(args.get("key_values") or []),
            "output_columns": list(_columns),
            "historical_operator": True,
        },
    }


_BUILDERS = {
    "condition_filter": _condition_filter,
    "project": _project,
    "scalar_compute": _scalar_compute,
    "join": _join,
    "join_tables": _join_tables,
    "group_aggregate": _group_aggregate,
    "extreme_value_select": _extreme_value_select,
    "set_op": _set_op,
    "pivot": _pivot,
}

if frozenset(_BUILDERS) != SUPPORTED_TABLE_OPERATORS:
    raise RuntimeError(
        "relation derivation builders and supported operators must match"
    )


def build_relation_derivation(
    inspector: RelationInspector,
    tool: str,
    args: dict,
    output_table: str,
    output_columns: list[str],
) -> dict:
    """Build the canonical derivation record for one successful table-producing action."""
    if tool not in SUPPORTED_TABLE_OPERATORS:
        raise ValueError(f"unsupported table-producing operator for derivation: {tool}")
    body = _BUILDERS[tool](args, output_columns, inspector, output_table)
    derivation = {
        "schema": DERIVATION_SCHEMA,
        "operator": tool,
        "inputs": body["inputs"],
        "semantics": body["semantics"],
    }
    validate_relation_derivation(derivation, output_columns=output_columns)
    return derivation
