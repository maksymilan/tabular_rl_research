"""Unit tests for canonical visible-cell binding semantics."""
from common import T
from observation_binding import (
    ArgumentPathError,
    VisibleCell,
    action_literal_slots,
    replace_argument_slot,
    same_scalar,
    singleton_visible_cell,
)


def run():
    t = T("observation_binding")

    slots = action_literal_slots(
        "condition_filter",
        {
            "table": "items",
            "conditions": {
                "or": [
                    {"column": "item_id", "op": "=", "value": 7},
                    {"column": "status", "op": "in", "values": ["a", "b"]},
                    {
                        "column": "score",
                        "op": ">",
                        "value": {"value_ref": "step_1"},
                    },
                ]
            },
        },
    )
    t.check(
        "condition literals retain exact argument paths",
        [
            (slot.column, slot.value, slot.argument_path)
            for slot in slots
        ] == [
            ("item_id", 7, ("conditions", "or", 0, "value")),
            ("status", "a", ("conditions", "or", 1, "values", 0)),
            ("status", "b", ("conditions", "or", 1, "values", 1)),
        ],
    )

    aggregate_slots = action_literal_slots(
        "group_aggregate",
        {
            "group_by": ["category"],
            "output_layout": "columns",
            "category_values": ["x", "y"],
            "aggregations": [{
                "op": "count",
                "column": "*",
                "as": "active_count",
                "where": {"column": "active", "op": "=", "value": 1},
            }],
        },
    )
    t.check(
        "aggregate predicates and categories share one slot contract",
        [slot.argument_path for slot in aggregate_slots] == [
            ("aggregations", 0, "where", "value"),
            ("category_values", 0),
            ("category_values", 1),
        ],
    )

    history = {
        "step_1": {
            "output": {"columns": ["id", "score"], "rows": [[1, 8], [2, 7]]},
        },
        "step_2": {
            "output": {"columns": ["left_id", "right_id"], "rows": [[42, 42]]},
        },
        "step_3": {
            "output": {"rows": [[42, "entity"]]},
            "observed_columns": ["entity_id", "name"],
        },
    }
    source = singleton_visible_cell("step_3", history["step_3"], 42)
    t.check(
        "unique singleton cell is replay-bindable",
        source is not None
        and source.step_id == "step_3"
        and source.column == "entity_id"
        and source.column_index == 0,
    )
    restored = None
    if source is not None:
        restored = VisibleCell.from_replay_target(
            source.replay_target(),
            value=source.value,
        )
    t.check("visible-cell replay metadata round-trips", restored == source)
    t.check(
        "negative replay locators are rejected",
        VisibleCell.from_replay_target(
            {
                "step": "step_3",
                "row_index": -1,
                "column_index": 0,
                "column": "entity_id",
            },
            value=42,
        )
        is None,
    )
    t.check(
        "multi-row observations do not create replay bindings",
        singleton_visible_cell("step_1", history["step_1"], 1) is None,
    )
    t.check(
        "ambiguous singleton cells do not create replay bindings",
        singleton_visible_cell("step_2", history["step_2"], 42) is None,
    )
    t.check("boolean and integer equality remains typed", not same_scalar(True, 1))

    arguments = {
        "conditions": {
            "or": [
                {"column": "item_id", "op": "=", "value": 7},
            ],
        },
    }
    replace_argument_slot(
        arguments,
        ("conditions", "or", 0, "value"),
        42,
    )
    t.check(
        "canonical argument paths replace their parsed literal slot",
        arguments["conditions"]["or"][0]["value"] == 42,
    )
    invalid_path_rejected = False
    try:
        replace_argument_slot(arguments, ("conditions", "and", 0, "value"), 9)
    except ArgumentPathError:
        invalid_path_rejected = True
    t.check("invalid argument paths are rejected", invalid_path_rejected)

    return t.result()
