#!/usr/bin/env python3
"""Validate trajectory JSON files against the local tool-design contract."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_ROOT = ROOT / "tool_design" / "trajectory"

TOP_LEVEL_FIELDS = {
    "trajectory_id",
    "question_type",
    "source_sample",
    "question",
    "dataset_gold_answer",
    "dataset_annotations",
    "initial_state",
    "steps",
}
STEP_FIELDS = {"step_id", "tool_name", "tool_call", "tool_output", "state_delta"}
INITIAL_STATE_FIELDS = {
    "dataset_overview",
    "static_task_memory",
    "dynamic_table_context",
    "context_budget",
}
OVERVIEW_REQUIRED_FIELDS = {"tables"}
OVERVIEW_OPTIONAL_FIELDS = {"relations"}
OVERVIEW_TABLE_FIELDS = {"table_name", "num_rows", "columns"}
RELATION_FIELDS = {"from", "to", "type"}
RELATION_CARDINALITIES = {
    "one_to_one",
    "one_to_many",
    "many_to_one",
    "many_to_many",
}
OVERVIEW_COLUMN_REQUIRED_FIELDS = {"name", "semantic_type"}
OVERVIEW_COLUMN_OPTIONAL_FIELDS = {"statistics", "unit", "format"}
OVERVIEW_STATISTIC_FIELDS = {
    "count",
    "non_null_count",
    "parse_coverage",
    "sum",
    "mean",
    "min",
    "max",
    "variance",
    "stddev",
}
CONTEXT_BUDGET_FIELDS = {
    "max_rows_visible",
    "max_columns_visible",
    "current_rows_visible",
    "current_columns_visible",
    "estimated_tokens",
}
CONTEXT_FIELDS = {"context_id", "tables"}
CONTEXT_TABLE_FIELDS = {
    "table_name",
    "columns",
    "rows",
    "data",
    "source_step_ids",
}
MEMORY_ITEM_FIELDS = {
    "type",
    "content",
    "supporting_step_ids",
    "supporting_rows_or_columns",
    "confidence_or_check",
    "invalidating_condition",
}
MEMORY_SUPPORT_FIELDS = {"tables", "rows", "columns", "cells"}

ROW_COMMON_ARGUMENTS = {
    "mode",
    "table_name",
    "search_scope",
    "return_columns",
    "top_k",
}
ROW_MODE_ARGUMENTS = {
    "condition_filter": {"conditions"},
    "entity_match": {"query", "entity_columns"},
    "extreme_value_select": {"target_column", "order"},
    "semantic_match": {"query", "semantic_columns"},
}
ROW_COMMON_OUTPUT = {
    "matched_rows",
    "matched_row_count",
    "matched_set_statistics",
    "visible_row_policy",
    "current_table_context",
}
ROW_MODE_OUTPUT = {
    "condition_filter": set(),
    "entity_match": set(),
    "extreme_value_select": {"extreme_value_result"},
    "semantic_match": {"semantic_match_result"},
}
ROW_STATE_DELTA = {
    "dynamic_table_context_changed",
    "context_id_before",
    "context_id_after",
    "added_rows",
    "added_columns",
    "static_task_memory_changed",
}
MATCHED_SET_STATISTIC_FIELDS = {
    "table_name",
    "column",
    "scope",
    "count",
    "non_null_count",
    "sum",
    "mean",
    "min",
    "max",
    "variance",
    "stddev",
    "parse_coverage",
}
CREATED_TABLE_FIELDS = {
    "table_name",
    "kind",
    "intent",
    "columns",
    "num_rows",
    "definition_in_harness",
}
GROUP_AGGREGATE_ARGUMENTS = {"table_name", "search_scope", "group_by", "aggregations"}
GROUP_AGGREGATE_OUTPUT = {"created_table", "group_sample_policy", "group_sample"}
JOIN_TABLES_ARGUMENTS = {"left_table", "right_table", "on", "join_type", "return_columns"}
JOIN_TABLES_OUTPUT = {"created_table", "join_diagnostics", "row_sample_policy", "row_sample"}
DERIVED_STATE_DELTA = {
    "data_view_changed",
    "added_table",
    "dynamic_table_context_changed",
    "static_task_memory_changed",
}
AGGREGATION_OPS = {"sum", "count", "mean", "min", "max"}
JOIN_TYPES = {"inner", "left"}
DERIVED_TABLE_KINDS = {"group", "join"}


def require_fields(value: dict, expected: set[str], location: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{location}: fields {sorted(actual)} do not match {sorted(expected)}"
        )


def validate_context(value: dict, location: str) -> None:
    require_fields(value, CONTEXT_FIELDS, location)
    for index, table in enumerate(value["tables"]):
        table_location = f"{location}.tables[{index}]"
        require_fields(table, CONTEXT_TABLE_FIELDS, table_location)
        if len(table["rows"]) != len(table["data"]):
            raise ValueError(f"{table_location}: rows and data lengths differ")
        for row_id, row in zip(table["rows"], table["data"]):
            if row.get("_row_id") != row_id:
                raise ValueError(f"{table_location}: row id mismatch for {row_id}")


def validate_overview(value: dict, location: str) -> dict[str, set[str]]:
    actual_fields = set(value)
    if not OVERVIEW_REQUIRED_FIELDS.issubset(actual_fields):
        raise ValueError(f"{location}: missing required field 'tables'")
    unsupported = actual_fields - (OVERVIEW_REQUIRED_FIELDS | OVERVIEW_OPTIONAL_FIELDS)
    if unsupported:
        raise ValueError(f"{location}: unsupported fields {sorted(unsupported)}")
    table_columns: dict[str, set[str]] = {}
    for table_index, table in enumerate(value["tables"]):
        table_location = f"{location}.tables[{table_index}]"
        require_fields(table, OVERVIEW_TABLE_FIELDS, table_location)
        if table["table_name"] in table_columns:
            raise ValueError(f"{table_location}: duplicate table name")
        columns: set[str] = set()
        for column_index, column in enumerate(table["columns"]):
            column_location = f"{table_location}.columns[{column_index}]"
            actual_fields = set(column)
            if not OVERVIEW_COLUMN_REQUIRED_FIELDS.issubset(actual_fields):
                raise ValueError(f"{column_location}: missing required column fields")
            unsupported = actual_fields - (
                OVERVIEW_COLUMN_REQUIRED_FIELDS | OVERVIEW_COLUMN_OPTIONAL_FIELDS
            )
            if unsupported:
                raise ValueError(
                    f"{column_location}: unsupported fields {sorted(unsupported)}"
                )
            if column["name"] in columns:
                raise ValueError(f"{column_location}: duplicate column name")
            columns.add(column["name"])
            if "statistics" in column:
                unsupported_statistics = (
                    set(column["statistics"]) - OVERVIEW_STATISTIC_FIELDS
                )
                if unsupported_statistics:
                    raise ValueError(
                        f"{column_location}: unsupported statistics "
                        f"{sorted(unsupported_statistics)}"
                    )
        table_columns[table["table_name"]] = columns
    for relation_index, relation in enumerate(value.get("relations", [])):
        relation_location = f"{location}.relations[{relation_index}]"
        require_fields(relation, RELATION_FIELDS, relation_location)
        if relation["type"] not in RELATION_CARDINALITIES:
            raise ValueError(
                f"{relation_location}: unsupported cardinality {relation['type']}"
            )
        validate_qualified_column(relation["from"], table_columns, relation_location)
        validate_qualified_column(relation["to"], table_columns, relation_location)
    return table_columns


def validate_memory_item(value: dict, stored: bool, location: str) -> None:
    expected = MEMORY_ITEM_FIELDS | ({"id"} if stored else set())
    require_fields(value, expected, location)
    unsupported = set(value["supporting_rows_or_columns"]) - MEMORY_SUPPORT_FIELDS
    if unsupported:
        raise ValueError(f"{location}: unsupported support fields {sorted(unsupported)}")


def validate_step_references(value: object, step_ids: set[str], location: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"supporting_step_ids", "source_step_ids"}:
                for step_id in item:
                    if step_id not in step_ids:
                        raise ValueError(f"{location}: unknown step reference {step_id}")
            else:
                validate_step_references(item, step_ids, location)
    elif isinstance(value, list):
        for item in value:
            validate_step_references(item, step_ids, location)


def validate_qualified_column(
    qualified_column: str, table_columns: dict[str, set[str]], location: str
) -> None:
    if "." not in qualified_column:
        raise ValueError(f"{location}: column is not qualified: {qualified_column}")
    table_name, column_name = qualified_column.split(".", 1)
    if column_name not in table_columns.get(table_name, set()):
        raise ValueError(f"{location}: unknown column {qualified_column}")


def register_created_table(created: dict, table_columns: dict[str, set[str]], location: str) -> None:
    require_fields(created, CREATED_TABLE_FIELDS, f"{location}.created_table")
    if created["kind"] not in DERIVED_TABLE_KINDS:
        raise ValueError(f"{location}.created_table: unsupported kind {created['kind']}")
    columns: set[str] = set()
    for index, column in enumerate(created["columns"]):
        if not OVERVIEW_COLUMN_REQUIRED_FIELDS.issubset(set(column)):
            raise ValueError(
                f"{location}.created_table.columns[{index}]: missing required fields"
            )
        columns.add(column["name"])
    table_columns[created["table_name"]] = columns


def validate_step(step: dict, table_columns: dict[str, set[str]], location: str) -> None:
    require_fields(step, STEP_FIELDS, location)
    require_fields(step["tool_call"], {"tool", "arguments"}, f"{location}.tool_call")
    if step["tool_call"]["tool"] != step["tool_name"]:
        raise ValueError(f"{location}: tool_name and tool_call.tool differ")

    tool_name = step["tool_name"]
    arguments = step["tool_call"]["arguments"]
    output = step["tool_output"]
    state_delta = step["state_delta"]

    if tool_name == "retrieve_column_context":
        require_fields(arguments, {"requested_columns"}, f"{location}.arguments")
        require_fields(
            output,
            {"retrieved_columns", "sample_row_policy", "current_table_context"},
            f"{location}.tool_output",
        )
        require_fields(state_delta, ROW_STATE_DELTA, f"{location}.state_delta")
        for request in arguments["requested_columns"]:
            require_fields(request, {"table_name", "column"}, f"{location}.request")
            if request["column"] not in table_columns.get(request["table_name"], set()):
                raise ValueError(f"{location}: unknown requested column {request}")
        for qualified_column in output["retrieved_columns"]:
            validate_qualified_column(qualified_column, table_columns, location)
        validate_context(output["current_table_context"], f"{location}.current_table_context")
        return

    if tool_name == "retrieve_row_context":
        mode = arguments.get("mode")
        if mode not in ROW_MODE_ARGUMENTS:
            raise ValueError(f"{location}: unsupported retrieve_row_context mode {mode}")
        require_fields(
            arguments,
            ROW_COMMON_ARGUMENTS | ROW_MODE_ARGUMENTS[mode],
            f"{location}.arguments",
        )
        require_fields(
            output,
            ROW_COMMON_OUTPUT | ROW_MODE_OUTPUT[mode],
            f"{location}.tool_output",
        )
        require_fields(state_delta, ROW_STATE_DELTA, f"{location}.state_delta")
        if arguments["table_name"] not in table_columns:
            raise ValueError(f"{location}: unknown table {arguments['table_name']}")
        if arguments["search_scope"] not in {"full_table", "dynamic_table_context"}:
            raise ValueError(f"{location}: unsupported search_scope")
        for qualified_column in arguments["return_columns"]:
            validate_qualified_column(qualified_column, table_columns, location)
        if mode == "condition_filter":
            for condition in arguments["conditions"]:
                require_fields(
                    condition, {"column", "operator", "value"}, f"{location}.condition"
                )
                validate_qualified_column(condition["column"], table_columns, location)
        if mode == "entity_match":
            for qualified_column in arguments["entity_columns"]:
                validate_qualified_column(qualified_column, table_columns, location)
        if mode == "semantic_match":
            for qualified_column in arguments["semantic_columns"]:
                validate_qualified_column(qualified_column, table_columns, location)
        if mode == "extreme_value_select":
            validate_qualified_column(arguments["target_column"], table_columns, location)
        for statistic in output["matched_set_statistics"]:
            require_fields(
                statistic,
                MATCHED_SET_STATISTIC_FIELDS,
                f"{location}.matched_set_statistics",
            )
            validate_qualified_column(
                f"{statistic['table_name']}.{statistic['column']}",
                table_columns,
                location,
            )
        validate_context(output["current_table_context"], f"{location}.current_table_context")
        return

    if tool_name == "add_to_memory":
        require_fields(arguments, {"items"}, f"{location}.arguments")
        require_fields(output, {"task_memory"}, f"{location}.tool_output")
        require_fields(
            state_delta,
            {
                "static_task_memory_changed",
                "added_memory_ids",
                "dynamic_table_context_changed",
            },
            f"{location}.state_delta",
        )
        for item in arguments["items"]:
            validate_memory_item(item, False, f"{location}.items")
        for item in output["task_memory"]:
            validate_memory_item(item, True, f"{location}.task_memory")
        return

    if tool_name == "answer_from_context":
        require_fields(
            arguments,
            {
                "answer",
                "evidence_rows",
                "evidence_columns",
                "supporting_memory_ids",
                "reason",
            },
            f"{location}.arguments",
        )
        require_fields(output, {"final_answer", "evidence"}, f"{location}.tool_output")
        require_fields(
            output["evidence"], {"rows", "columns", "memory_ids"}, f"{location}.evidence"
        )
        require_fields(
            state_delta,
            {
                "final_answer_submitted",
                "dynamic_table_context_changed",
                "static_task_memory_changed",
            },
            f"{location}.state_delta",
        )
        return

    if tool_name == "group_aggregate":
        require_fields(arguments, GROUP_AGGREGATE_ARGUMENTS, f"{location}.arguments")
        require_fields(output, GROUP_AGGREGATE_OUTPUT, f"{location}.tool_output")
        require_fields(state_delta, DERIVED_STATE_DELTA, f"{location}.state_delta")
        if arguments["table_name"] not in table_columns:
            raise ValueError(f"{location}: unknown table {arguments['table_name']}")
        if arguments["search_scope"] not in {"full_table", "dynamic_table_context"}:
            raise ValueError(f"{location}: unsupported search_scope")
        for qualified_column in arguments["group_by"]:
            validate_qualified_column(qualified_column, table_columns, location)
        for aggregation in arguments["aggregations"]:
            if "op" not in aggregation or "as" not in aggregation:
                raise ValueError(f"{location}: aggregation needs 'op' and 'as'")
            if aggregation["op"] not in AGGREGATION_OPS:
                raise ValueError(f"{location}: unsupported aggregation op {aggregation['op']}")
            if "column" in aggregation:
                validate_qualified_column(aggregation["column"], table_columns, location)
            elif aggregation["op"] != "count":
                raise ValueError(f"{location}: aggregation {aggregation['op']} requires a column")
        register_created_table(output["created_table"], table_columns, location)
        return

    if tool_name == "join_tables":
        require_fields(arguments, JOIN_TABLES_ARGUMENTS, f"{location}.arguments")
        require_fields(output, JOIN_TABLES_OUTPUT, f"{location}.tool_output")
        require_fields(state_delta, DERIVED_STATE_DELTA, f"{location}.state_delta")
        for side in ("left_table", "right_table"):
            if arguments[side] not in table_columns:
                raise ValueError(f"{location}: unknown table {arguments[side]}")
        if arguments["join_type"] not in JOIN_TYPES:
            raise ValueError(f"{location}: unsupported join_type {arguments['join_type']}")
        for condition in arguments["on"]:
            require_fields(condition, {"left", "right"}, f"{location}.on")
            validate_qualified_column(condition["left"], table_columns, location)
            validate_qualified_column(condition["right"], table_columns, location)
        for qualified_column in arguments["return_columns"]:
            validate_qualified_column(qualified_column, table_columns, location)
        register_created_table(output["created_table"], table_columns, location)
        return

    raise ValueError(f"{location}: unsupported tool {tool_name}")


def validate_trajectory(path: Path) -> None:
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    require_fields(trajectory, TOP_LEVEL_FIELDS, str(path))
    require_fields(
        trajectory["source_sample"],
        {"path", "record_index", "dataset", "record_id"},
        f"{path}.source_sample",
    )
    require_fields(
        trajectory["dataset_annotations"],
        {"evidence", "label_assessment"},
        f"{path}.dataset_annotations",
    )
    require_fields(
        trajectory["dataset_annotations"]["label_assessment"],
        {"status", "reason"},
        f"{path}.label_assessment",
    )
    label_status = trajectory["dataset_annotations"]["label_assessment"]["status"]
    if label_status not in {"accepted", "conflict"}:
        raise ValueError(f"{path}: unsupported label assessment status {label_status}")

    initial_state = trajectory["initial_state"]
    require_fields(initial_state, INITIAL_STATE_FIELDS, f"{path}.initial_state")
    require_fields(
        initial_state["context_budget"], CONTEXT_BUDGET_FIELDS, f"{path}.context_budget"
    )
    validate_context(initial_state["dynamic_table_context"], f"{path}.dynamic_table_context")
    table_columns = validate_overview(
        initial_state["dataset_overview"], f"{path}.dataset_overview"
    )

    step_ids = {step["step_id"] for step in trajectory["steps"]}
    if len(step_ids) != len(trajectory["steps"]):
        raise ValueError(f"{path}: duplicate step ids")

    for step in trajectory["steps"]:
        validate_step(step, table_columns, f"{path}:{step['step_id']}")
        validate_step_references(step, step_ids, f"{path}:{step['step_id']}")

    if not trajectory["steps"] or trajectory["steps"][-1]["tool_name"] != "answer_from_context":
        raise ValueError(f"{path}: final step must be answer_from_context")
    final_answer = trajectory["steps"][-1]["tool_output"]["final_answer"]
    if label_status == "accepted" and final_answer != trajectory["dataset_gold_answer"]:
        raise ValueError(f"{path}: accepted label does not match final answer")
    if label_status == "conflict" and final_answer == trajectory["dataset_gold_answer"]:
        raise ValueError(f"{path}: conflict label unexpectedly matches final answer")


def main() -> None:
    paths = sorted(TRAJECTORY_ROOT.glob("**/trajectory.json"))
    for path in paths:
        validate_trajectory(path)
        print(f"ok {path.relative_to(ROOT)}")
    print(f"validated {len(paths)} trajectories")


if __name__ == "__main__":
    main()
