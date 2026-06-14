#!/usr/bin/env python3
"""Plan: the intermediate representation between the SQL compiler and the harness executor.

A `Plan` is an ordered list of `Step`s. Each step names one abstract tool call. A step refers to
its input table(s) either by a **source table name** or by the **id of an earlier step** (e.g.
"s1"). `run_plan` executes the steps in order against a `Harness`, resolving step-ids to the real
view names the harness assigns, and returns the final result rows.

This decoupling lets the compiler emit a pure, inspectable plan (no execution), and lets the
executor / verifier run it. The model-facing trajectory is rendered from the same plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Which argument keys of each tool are table references (source name or step id).
TABLE_REF_ARGS: dict[str, list[str]] = {
    "condition_filter": ["table"],
    "group_aggregate": ["table"],
    "derive_column": ["table"],
    "project": ["table"],
    "extreme_value_select": ["table"],
    "aggregate": ["table"],
    "read_subtable": ["table"],
    "window": ["table"],
    "join_tables": ["left", "right"],
    "set_op": ["left", "right"],
}

# Tools whose result is a value/rows (terminal) rather than a registered table.
# (`extreme_value_select` is now table-producing — merged with the old `order_limit`.)
TERMINAL_TOOLS = {"aggregate", "read_subtable"}


@dataclass
class Step:
    id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # compact, for plan printing / reports
        a = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.id}: {self.tool}({a})"


Plan = list[Step]


def resolve_cond(cond, id_to_table: dict, values: dict | None = None):
    """Resolve subquery references inside a condition tree.
    - `in_table` (an IN-subquery's set table, by step id) -> its real view name. Always resolved.
    - `value_ref` (a scalar subquery's value, parked in memory) -> the literal. Resolved only when
      `values` is given (i.e. at execution; the trajectory keeps `value_ref` so the model's action
      references the memory entry, not a magic constant)."""
    if isinstance(cond, list):
        return [resolve_cond(c, id_to_table, values) for c in cond]
    if not isinstance(cond, dict):
        return cond
    for key in ("and", "or"):
        if key in cond:
            return {key: [resolve_cond(x, id_to_table, values) for x in cond[key]]}
    if "not" in cond:
        return {"not": resolve_cond(cond["not"], id_to_table, values)}
    out = dict(cond)
    if "in_table" in out:
        out["in_table"] = id_to_table.get(out["in_table"], out["in_table"])
    if values is not None and "value_ref" in out:
        out["value"] = values[out.pop("value_ref")]
    return out


def run_plan(harness, plan: Plan) -> list[tuple]:
    """Execute a plan against a Harness; return the final result as a list of rows.

    Intermediate (table-producing) steps register a view; their id maps to the new table name.
    Scalar steps (`aggregate`) record their value under the step id; `add_to_memory` parks a
    referenced scalar under a memory key so later predicates can reference it via `value_ref`.
    The final step's output is returned as rows (table read out, scalar wrapped as one row).
    """
    id_to_table: dict[str, str] = {}
    values: dict[str, Any] = {}
    final: Any = None

    for step in plan:
        args = dict(step.args)
        for key in TABLE_REF_ARGS.get(step.tool, []):
            ref = args.get(key)
            if ref in id_to_table:
                args[key] = id_to_table[ref]
        if step.tool == "add_to_memory":
            # V2a: a pure memory op. Extract the scalar from the cited source and park it under THIS
            # step's id, which is exactly what the predicate's `value_ref` points at. No SQL view.
            src = step.args["source"]
            if src in id_to_table:                       # single-row subquery: take its one cell
                rows = harness.rows(id_to_table[src])
                val = rows[0][0] if rows else None
            else:
                val = values.get(src)                    # aggregate scalar
            values[step.id] = val
            continue
        if step.tool == "condition_filter":
            args["conditions"] = resolve_cond(args.get("conditions"), id_to_table, values)
        method = getattr(harness, step.tool, None)
        if method is None:
            raise ValueError(f"harness has no tool {step.tool!r}")
        result = method(**args)
        if isinstance(result, dict) and "table_name" in result:
            id_to_table[step.id] = result["table_name"]
            final = ("table", result["table_name"])
        else:
            if step.tool == "aggregate":
                values[step.id] = result
            final = ("value", result)

    if final is None:
        return []
    kind, payload = final
    if kind == "table":
        return harness.rows(payload)
    if isinstance(payload, list):  # rows from a row-returning tool
        return payload
    return [(payload,)]  # scalar
