"""Tests for the fact-only, table-bound relation derivation schema."""
from __future__ import annotations

from copy import deepcopy
import json

from common import T, employees_db
from relation_derivation import (
    DERIVATION_SCHEMA,
    SUPPORTED_TABLE_OPERATORS,
    build_relation_derivation,
    expression_source_columns,
    validate_relation_derivation,
)


def _derive(harness, tool: str, args: dict, output: dict) -> dict:
    return build_relation_derivation(
        harness,
        tool,
        args,
        output["table_name"],
        output["columns"],
    )


def run():
    t = T("relation_derivation")
    h = employees_db()

    t.check(
        "complete active table operator coverage",
        SUPPORTED_TABLE_OPERATORS == {
            "condition_filter",
            "project",
            "scalar_compute",
            "join",
            "join_tables",
            "group_aggregate",
            "extreme_value_select",
            "set_op",
            "pivot",
        },
        str(SUPPORTED_TABLE_OPERATORS),
    )

    sources = expression_source_columns(
        "name || ':' || dept AS label",
        ["id", "name", "dept"],
    )
    t.check("expression lineage resolves every source", sources == ["name", "dept"], str(sources))

    project_args = {
        "table": "employees",
        "expressions": ["name || ':' || dept AS label", "salary"],
    }
    projected = h.project(**project_args)
    project_derivation = _derive(h, "project", project_args, projected)
    t.check(
        "project derivation records full column lineage",
        project_derivation["semantics"]["column_lineage"] == [
            {
                "output": "label",
                "sources": ["name", "dept"],
                "kind": "expression",
                "expression": "name || ':' || dept AS label",
            },
            {
                "output": "salary",
                "sources": ["salary"],
                "kind": "column",
                "expression": "salary",
            },
        ],
        str(project_derivation),
    )

    wildcard_args = {"table": "employees", "expressions": ["*"]}
    wildcard = h.project(**wildcard_args)
    wildcard_derivation = _derive(h, "project", wildcard_args, wildcard)
    t.check(
        "wildcard projection expands to complete ordered lineage",
        [item["output"] for item in wildcard_derivation["semantics"]["column_lineage"]]
        == wildcard["columns"]
        and all(
            item["sources"] == [item["output"]]
            for item in wildcard_derivation["semantics"]["column_lineage"]
        ),
        str(wildcard_derivation),
    )

    aggregate_args = {
        "table": "employees",
        "group_by": ["dept"],
        "aggregations": [
            {"op": "count_distinct", "column": "id", "as": "people"},
            {
                "op": "sum",
                "column": "salary",
                "as": "senior_salary",
                "where": {"column": "age", "op": ">=", "value": 40},
            },
        ],
    }
    aggregated = h.group_aggregate(**aggregate_args)
    aggregate_derivation = _derive(h, "group_aggregate", aggregate_args, aggregated)
    t.check(
        "aggregate derivation records grain and metric semantics",
        aggregate_derivation["semantics"]["row_grain"] == ["dept"]
        and aggregate_derivation["semantics"]["aggregations"][0]["op"] == "count_distinct"
        and aggregate_derivation["semantics"]["aggregations"][1]["condition_columns"] == ["age"],
        str(aggregate_derivation),
    )

    left_args = {
        "base": "employees",
        "joins": [{
            "table": "depts",
            "type": "left",
            "on": [{"left": "employees.dept", "right": "dept"}],
        }],
    }
    joined = h.join_tables(**left_args)
    join_derivation = _derive(h, "join_tables", left_args, joined)
    t.check(
        "join derivation records ordered edge semantics",
        join_derivation["semantics"]["edges"][0]["on"]
        == [{"left": "employees.dept", "right": "dept"}],
        str(join_derivation),
    )
    t.check(
        "left join measured fact is tied to its edge",
        join_derivation["semantics"]["edges"][0]["null_extended_output_rows"] == 0,
        str(join_derivation),
    )

    symmetric_args = {
        "left": "employees",
        "right": "depts",
        "on": [{"left": "employees.dept", "right": "depts.dept"}],
        "how": "left",
    }
    symmetric = h.join(**symmetric_args)
    symmetric_derivation = _derive(h, "join", symmetric_args, symmetric)
    t.check(
        "symmetric join derivation records both inputs and one edge",
        symmetric_derivation["inputs"][0]["role"] == "left"
        and symmetric_derivation["inputs"][1]["role"] == "right"
        and symmetric_derivation["semantics"]["edges"][0]["on"]
        == [{"left": "employees.dept", "right": "depts.dept"}],
        str(symmetric_derivation),
    )

    nway_args = {
        "base": "employees",
        "joins": [
            {
                "table": "depts",
                "type": "inner",
                "on": [{"left": "employees.dept", "right": "dept"}],
            },
            {
                "table": "sales",
                "type": "left",
                "on": [{"left": "employees.dept", "right": "dept"}],
            },
        ],
    }
    nway = h.join_tables(**nway_args)
    nway_derivation = _derive(h, "join_tables", nway_args, nway)
    t.check(
        "n-way join keeps every edge in execution order",
        [edge["index"] for edge in nway_derivation["semantics"]["edges"]] == [0, 1]
        and [edge["input"] for edge in nway_derivation["semantics"]["edges"]]
        == ["depts", "sales"]
        and nway_derivation["semantics"]["edges"][1]["null_extended_output_rows"] == 1,
        str(nway_derivation),
    )

    grouped_for_pivot_args = {
        "table": "employees",
        "group_by": ["dept"],
        "aggregations": [{"op": "count", "column": "*", "as": "people"}],
    }
    grouped_for_pivot = h.group_aggregate(**grouped_for_pivot_args)
    pivot_args = {
        "table": grouped_for_pivot["table_name"],
        "key_column": "dept",
        "value_column": "people",
        "key_values": ["eng", "sales", "hr"],
        "output_columns": ["engineering", "sales", "human_resources"],
    }
    pivoted = h.pivot(**pivot_args)
    pivot_derivation = _derive(h, "pivot", pivot_args, pivoted)
    t.check(
        "historical pivot replay has the same validated envelope",
        pivot_derivation["schema"] == DERIVATION_SCHEMA
        and pivot_derivation["semantics"]["row_operation"] == "reshape"
        and pivot_derivation["semantics"]["output_columns"] == pivoted["columns"],
        str(pivot_derivation),
    )

    serialized = json.dumps(
        [
            project_derivation,
            wildcard_derivation,
            aggregate_derivation,
            join_derivation,
            nway_derivation,
            pivot_derivation,
        ],
        sort_keys=True,
    )
    t.check("all derivations use one schema", serialized.count(DERIVATION_SCHEMA) == 6, serialized)
    t.check("derivations contain no policy advice", "advice" not in serialized, serialized)

    invalid = dict(project_derivation)
    invalid["semantics"] = {**project_derivation["semantics"], "advice": "pick a column"}
    try:
        validate_relation_derivation(invalid)
    except ValueError:
        rejected_policy = True
    else:
        rejected_policy = False
    t.check("schema validation rejects policy fields", rejected_policy, str(invalid))

    invalid_operation = deepcopy(project_derivation)
    invalid_operation["semantics"]["row_operation"] = "aggregate"
    try:
        validate_relation_derivation(invalid_operation)
    except ValueError:
        rejected_invalid_operation = True
    else:
        rejected_invalid_operation = False
    t.check(
        "schema validation rejects operator-semantic mismatch",
        rejected_invalid_operation,
        str(invalid_operation),
    )

    return t.result()
