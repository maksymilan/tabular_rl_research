#!/usr/bin/env python3
"""Validate trajectory JSON files against the local tool-design contract.

Structural field sets and enums are NOT hardcoded here: they are loaded from
``tool_design/tool_io_spec.json`` (the single source of truth). To change a
format, edit that spec first, then conform the trajectories. This module only
adds the cross-field / stateful checks that a flat field spec cannot express
(qualified-column existence, derived-table column tracking, relation integrity,
step-reference integrity, and final-answer/label consistency).
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY_ROOT = ROOT / "tool_design" / "trajectory"
SPEC_PATH = ROOT / "tool_design" / "tool_io_spec.json"

SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
ENUMS = SPEC["enums"]
TRAJ = SPEC["trajectory"]
STATE = SPEC["state"]
TOOLS = SPEC["tools"]


def enum(name: str) -> set[str]:
    return set(ENUMS[name])


def require_fields(value: dict, expected: set[str], location: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{location}: fields {sorted(actual)} do not match {sorted(expected)}"
        )


def validate_context(value: dict, location: str) -> None:
    require_fields(value, set(STATE["dynamic_table_context"]), location)
    for index, table in enumerate(value["tables"]):
        table_location = f"{location}.tables[{index}]"
        require_fields(table, set(STATE["context_table"]), table_location)
        if len(table["rows"]) != len(table["data"]):
            raise ValueError(f"{table_location}: rows and data lengths differ")
        for row_id, row in zip(table["rows"], table["data"]):
            if row.get("_row_id") != row_id:
                raise ValueError(f"{table_location}: row id mismatch for {row_id}")


def validate_overview(value: dict, location: str) -> dict[str, set[str]]:
    required = set(STATE["dataset_overview"]["required"])
    optional = set(STATE["dataset_overview"]["optional"])
    actual = set(value)
    if not required.issubset(actual):
        raise ValueError(f"{location}: missing required fields {sorted(required - actual)}")
    unsupported = actual - (required | optional)
    if unsupported:
        raise ValueError(f"{location}: unsupported fields {sorted(unsupported)}")

    column_required = set(STATE["overview_column"]["required"])
    column_optional = set(STATE["overview_column"]["optional"])
    statistic_fields = set(STATE["overview_statistic"])

    table_columns: dict[str, set[str]] = {}
    for table_index, table in enumerate(value["tables"]):
        table_location = f"{location}.tables[{table_index}]"
        require_fields(table, set(STATE["overview_table"]), table_location)
        if table["table_name"] in table_columns:
            raise ValueError(f"{table_location}: duplicate table name")
        columns: set[str] = set()
        for column_index, column in enumerate(table["columns"]):
            column_location = f"{table_location}.columns[{column_index}]"
            actual_fields = set(column)
            if not column_required.issubset(actual_fields):
                raise ValueError(f"{column_location}: missing required column fields")
            unsupported = actual_fields - (column_required | column_optional)
            if unsupported:
                raise ValueError(
                    f"{column_location}: unsupported fields {sorted(unsupported)}"
                )
            if column["name"] in columns:
                raise ValueError(f"{column_location}: duplicate column name")
            columns.add(column["name"])
            if "statistics" in column:
                unsupported_statistics = set(column["statistics"]) - statistic_fields
                if unsupported_statistics:
                    raise ValueError(
                        f"{column_location}: unsupported statistics "
                        f"{sorted(unsupported_statistics)}"
                    )
        table_columns[table["table_name"]] = columns

    for relation_index, relation in enumerate(value.get("relations", [])):
        relation_location = f"{location}.relations[{relation_index}]"
        require_fields(relation, set(STATE["relation"]), relation_location)
        if relation["type"] not in enum("relation_cardinality"):
            raise ValueError(
                f"{relation_location}: unsupported cardinality {relation['type']}"
            )
        validate_qualified_column(relation["from"], table_columns, relation_location)
        validate_qualified_column(relation["to"], table_columns, relation_location)
    return table_columns


def validate_memory_item(value: dict, stored: bool, location: str) -> None:
    expected = set(STATE["memory_item"]["required"])
    if stored:
        expected = expected | set(STATE["memory_item"]["stored_only"])
    require_fields(value, expected, location)
    if value["type"] not in enum("memory_type"):
        raise ValueError(f"{location}: unsupported memory type {value['type']}")
    unsupported = set(value["supporting_rows_or_columns"]) - set(STATE["memory_support"])
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


def register_created_table(
    created: dict, table_columns: dict[str, set[str]], location: str
) -> None:
    require_fields(created, set(STATE["created_table"]), f"{location}.created_table")
    if created["kind"] not in enum("derived_table_kind"):
        raise ValueError(f"{location}.created_table: unsupported kind {created['kind']}")
    column_required = set(STATE["overview_column"]["required"])
    columns: set[str] = set()
    for index, column in enumerate(created["columns"]):
        if not column_required.issubset(set(column)):
            raise ValueError(
                f"{location}.created_table.columns[{index}]: missing required fields"
            )
        columns.add(column["name"])
    table_columns[created["table_name"]] = columns


def validate_step(step: dict, table_columns: dict[str, set[str]], location: str) -> None:
    require_fields(step, set(TRAJ["step"]), location)
    require_fields(step["tool_call"], set(TRAJ["tool_call"]), f"{location}.tool_call")
    if step["tool_call"]["tool"] != step["tool_name"]:
        raise ValueError(f"{location}: tool_name and tool_call.tool differ")

    tool_name = step["tool_name"]
    if tool_name not in TOOLS:
        raise ValueError(f"{location}: unsupported tool {tool_name}")
    spec = TOOLS[tool_name]
    arguments = step["tool_call"]["arguments"]
    output = step["tool_output"]
    state_delta = step["state_delta"]

    # ---- structural field-set checks, driven entirely by the spec ----
    mode = None
    if "mode_arguments" in spec:
        mode = arguments.get(spec["mode_field"])
        if mode not in spec["mode_arguments"]:
            raise ValueError(f"{location}: unsupported {tool_name} mode {mode}")
        require_fields(
            arguments,
            set(spec["common_arguments"]) | set(spec["mode_arguments"][mode]),
            f"{location}.arguments",
        )
        require_fields(
            output,
            set(spec["common_output"]) | set(spec["mode_output"][mode]),
            f"{location}.tool_output",
        )
    else:
        require_fields(arguments, set(spec["arguments"]), f"{location}.arguments")
        require_fields(output, set(spec["output"]), f"{location}.tool_output")
    require_fields(state_delta, set(spec["state_delta"]), f"{location}.state_delta")

    # ---- per-tool semantic / cross-field checks ----
    if tool_name == "retrieve_column_context":
        for request in arguments["requested_columns"]:
            require_fields(request, set(STATE["requested_column"]), f"{location}.request")
            if request["column"] not in table_columns.get(request["table_name"], set()):
                raise ValueError(f"{location}: unknown requested column {request}")
        for qualified_column in output["retrieved_columns"]:
            validate_qualified_column(qualified_column, table_columns, location)
        validate_context(output["current_table_context"], f"{location}.current_table_context")
        return

    if tool_name == "retrieve_row_context":
        if arguments["table_name"] not in table_columns:
            raise ValueError(f"{location}: unknown table {arguments['table_name']}")
        if arguments["search_scope"] not in enum("search_scope"):
            raise ValueError(f"{location}: unsupported search_scope")
        for qualified_column in arguments["return_columns"]:
            validate_qualified_column(qualified_column, table_columns, location)
        if mode == "condition_filter":
            for condition in arguments["conditions"]:
                require_fields(condition, set(STATE["condition"]), f"{location}.condition")
                validate_qualified_column(condition["column"], table_columns, location)
                if condition["operator"] not in enum("condition_operator"):
                    raise ValueError(f"{location}: unsupported operator {condition['operator']}")
        if mode == "entity_match":
            for qualified_column in arguments["entity_columns"]:
                validate_qualified_column(qualified_column, table_columns, location)
        if mode == "semantic_match":
            for qualified_column in arguments["semantic_columns"]:
                validate_qualified_column(qualified_column, table_columns, location)
        if mode == "extreme_value_select":
            validate_qualified_column(arguments["target_column"], table_columns, location)
            if arguments["order"] not in enum("extreme_order"):
                raise ValueError(f"{location}: unsupported order {arguments['order']}")
        for statistic in output["matched_set_statistics"]:
            require_fields(
                statistic, set(STATE["matched_set_statistic"]), f"{location}.matched_set_statistics"
            )
            validate_qualified_column(
                f"{statistic['table_name']}.{statistic['column']}", table_columns, location
            )
        validate_context(output["current_table_context"], f"{location}.current_table_context")
        return

    if tool_name == "group_aggregate":
        if arguments["table_name"] not in table_columns:
            raise ValueError(f"{location}: unknown table {arguments['table_name']}")
        if arguments["search_scope"] not in enum("search_scope"):
            raise ValueError(f"{location}: unsupported search_scope")
        for qualified_column in arguments["group_by"]:
            validate_qualified_column(qualified_column, table_columns, location)
        for aggregation in arguments["aggregations"]:
            if "op" not in aggregation or "as" not in aggregation:
                raise ValueError(f"{location}: aggregation needs 'op' and 'as'")
            if aggregation["op"] not in enum("aggregation_op"):
                raise ValueError(f"{location}: unsupported aggregation op {aggregation['op']}")
            if "column" in aggregation:
                validate_qualified_column(aggregation["column"], table_columns, location)
            elif aggregation["op"] != "count":
                raise ValueError(f"{location}: aggregation {aggregation['op']} requires a column")
        register_created_table(output["created_table"], table_columns, location)
        return

    if tool_name == "join_tables":
        for side in ("left_table", "right_table"):
            if arguments[side] not in table_columns:
                raise ValueError(f"{location}: unknown table {arguments[side]}")
        if arguments["join_type"] not in enum("join_type"):
            raise ValueError(f"{location}: unsupported join_type {arguments['join_type']}")
        for condition in arguments["on"]:
            require_fields(condition, set(STATE["join_condition"]), f"{location}.on")
            validate_qualified_column(condition["left"], table_columns, location)
            validate_qualified_column(condition["right"], table_columns, location)
        for qualified_column in arguments["return_columns"]:
            validate_qualified_column(qualified_column, table_columns, location)
        register_created_table(output["created_table"], table_columns, location)
        return

    if tool_name == "answer_from_context":
        require_fields(output["evidence"], set(STATE["answer_evidence"]), f"{location}.evidence")
        return

    # drop_context, refine_memory: structural validation above is sufficient.
    return


def validate_trajectory(path: Path) -> None:
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    require_fields(trajectory, set(TRAJ["top_level"]), str(path))
    require_fields(
        trajectory["source_sample"], set(TRAJ["source_sample"]), f"{path}.source_sample"
    )
    require_fields(
        trajectory["dataset_annotations"],
        set(TRAJ["dataset_annotations"]),
        f"{path}.dataset_annotations",
    )
    require_fields(
        trajectory["dataset_annotations"]["label_assessment"],
        set(TRAJ["label_assessment"]),
        f"{path}.label_assessment",
    )
    label_status = trajectory["dataset_annotations"]["label_assessment"]["status"]
    if label_status not in enum("label_status"):
        raise ValueError(f"{path}: unsupported label assessment status {label_status}")

    initial_state = trajectory["initial_state"]
    require_fields(initial_state, set(STATE["initial_state"]), f"{path}.initial_state")
    require_fields(
        initial_state["context_budget"], set(STATE["context_budget"]), f"{path}.context_budget"
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
    print(f"validated {len(paths)} trajectories against {SPEC_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
