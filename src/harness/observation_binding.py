"""Harness-owned semantics for literals copied from model-visible table cells.

This module is deliberately independent of reward and counterfactual policy.  It provides one
canonical parser for literal argument slots and one conservative rule for stable replay bindings:
the value must occur in exactly one cell of a singleton visible row.  Provenance records the
result; downstream replay consumes it without re-inferring causality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


ArgumentPath = tuple[str | int, ...]


class ArgumentPathError(ValueError):
    """An argument path cannot be resolved against the supplied action arguments."""


@dataclass(frozen=True)
class LiteralSlot:
    """One scalar literal consumed by a relational tool argument."""

    column: str | None
    value: Any
    argument_path: ArgumentPath


@dataclass(frozen=True)
class VisibleCell:
    """One unambiguous scalar cell in a model-visible singleton row."""

    step_id: str
    row_index: int
    column_index: int
    column: str
    value: Any

    @classmethod
    def from_replay_target(
        cls,
        target: Any,
        *,
        value: Any,
    ) -> VisibleCell | None:
        if not isinstance(target, dict):
            return None
        step_id = target.get("step")
        row_index = target.get("row_index")
        column_index = target.get("column_index")
        column = target.get("column")
        if (
            not isinstance(step_id, str)
            or not isinstance(row_index, int)
            or isinstance(row_index, bool)
            or row_index < 0
            or not isinstance(column_index, int)
            or isinstance(column_index, bool)
            or column_index < 0
            or not isinstance(column, str)
        ):
            return None
        return cls(
            step_id=step_id,
            row_index=row_index,
            column_index=column_index,
            column=column,
            value=value,
        )

    def replay_target(self) -> dict[str, Any]:
        return {
            "step": self.step_id,
            "row_index": self.row_index,
            "column_index": self.column_index,
            "column": self.column,
        }


def replace_argument_slot(
    arguments: dict[str, Any],
    path: ArgumentPath,
    replacement: Any,
) -> None:
    """Replace one parsed literal slot without interpreting the surrounding tool contract."""
    if not path:
        raise ArgumentPathError("argument path cannot be empty")

    cursor: Any = arguments
    for component in path[:-1]:
        if isinstance(component, str) and isinstance(cursor, dict) and component in cursor:
            cursor = cursor[component]
        elif (
            isinstance(component, int)
            and not isinstance(component, bool)
            and isinstance(cursor, list)
            and 0 <= component < len(cursor)
        ):
            cursor = cursor[component]
        else:
            raise ArgumentPathError(f"argument path does not exist: {list(path)!r}")

    final = path[-1]
    if isinstance(final, str) and isinstance(cursor, dict) and final in cursor:
        cursor[final] = replacement
    elif (
        isinstance(final, int)
        and not isinstance(final, bool)
        and isinstance(cursor, list)
        and 0 <= final < len(cursor)
    ):
        cursor[final] = replacement
    else:
        raise ArgumentPathError(f"argument path does not exist: {list(path)!r}")


def same_scalar(left: Any, right: Any) -> bool:
    """Typed scalar equality used consistently by grounding and replay binding."""
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    try:
        return left == right
    except Exception:
        return False


def base_column(column: Any) -> str | None:
    if not isinstance(column, str):
        return None
    if "__" in column:
        return column.split("__", 1)[-1]
    if "." in column:
        return column.rsplit(".", 1)[-1]
    return column


def _scalar_value_slots(value: Any, path: ArgumentPath) -> Iterable[tuple[Any, ArgumentPath]]:
    if isinstance(value, dict):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _scalar_value_slots(item, path + (index,))
        return
    yield value, path


def condition_literal_slots(
    condition: Any,
    path: ArgumentPath = (),
) -> list[LiteralSlot]:
    """Parse condition-tree constants while retaining their exact mutable argument paths."""
    if isinstance(condition, list):
        return [
            slot
            for index, item in enumerate(condition)
            for slot in condition_literal_slots(item, path + (index,))
        ]
    if not isinstance(condition, dict):
        return []

    slots: list[LiteralSlot] = []
    for key in ("and", "or"):
        for index, item in enumerate(condition.get(key, []) or []):
            slots.extend(condition_literal_slots(item, path + (key, index)))
    if "not" in condition:
        slots.extend(condition_literal_slots(condition["not"], path + ("not",)))

    column = base_column(condition.get("column"))
    if "value" in condition:
        slots.extend(
            LiteralSlot(column, value, value_path)
            for value, value_path in _scalar_value_slots(
                condition["value"],
                path + ("value",),
            )
        )
    if "values" in condition:
        slots.extend(
            LiteralSlot(column, value, value_path)
            for value, value_path in _scalar_value_slots(
                condition["values"],
                path + ("values",),
            )
        )
    return slots


def action_literal_slots(tool: str, arguments: dict[str, Any]) -> list[LiteralSlot]:
    """Return every literal slot supported by the active relational argument contracts."""
    if tool == "condition_filter":
        return [
            LiteralSlot(slot.column, slot.value, ("conditions",) + slot.argument_path)
            for slot in condition_literal_slots(arguments.get("conditions"))
        ]
    if tool == "group_aggregate":
        slots = [
            LiteralSlot(
                slot.column,
                slot.value,
                ("aggregations", aggregation_index, "where") + slot.argument_path,
            )
            for aggregation_index, aggregation in enumerate(arguments.get("aggregations") or [])
            if isinstance(aggregation, dict)
            for slot in condition_literal_slots(aggregation.get("where"))
        ]
        group_by = arguments.get("group_by") or []
        if arguments.get("output_layout") == "columns" and len(group_by) == 1:
            slots.extend(
                LiteralSlot(
                    base_column(group_by[0]),
                    value,
                    ("category_values", index),
                )
                for index, value in enumerate(arguments.get("category_values") or [])
            )
        return slots
    if tool == "pivot":
        return [
            LiteralSlot(
                base_column(arguments.get("key_column")),
                value,
                ("key_values", index),
            )
            for index, value in enumerate(arguments.get("key_values") or [])
        ]
    return []


def singleton_visible_cell(
    step_id: str,
    record: dict[str, Any],
    value: Any,
) -> VisibleCell | None:
    """Return the unique singleton cell equal to ``value`` in one visible record.

    Multi-row observations are excluded because selecting one row can encode an unexecuted
    relation such as argmax.  Repeated equal cells are excluded because their source is ambiguous.
    """
    output = record.get("output") or {}
    rows = output.get("rows")
    columns = record.get("observed_columns") or output.get("columns")
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], (list, tuple))
        or not isinstance(columns, list)
    ):
        return None
    matches = [
        index
        for index, cell in enumerate(rows[0])
        if index < len(columns) and same_scalar(cell, value)
    ]
    if len(matches) != 1:
        return None
    column_index = matches[0]
    column = columns[column_index]
    if not isinstance(column, str):
        return None
    return VisibleCell(
        step_id=step_id,
        row_index=0,
        column_index=column_index,
        column=column,
        value=rows[0][column_index],
    )
