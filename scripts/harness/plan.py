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


def resolve_value_refs(cond, values: dict):
    """Substitute every `{value_ref: K}` leaf in a condition tree with `{value: values[K]}`.
    `value_ref` threads a scalar produced earlier (an aggregate's result, parked in memory) into a
    predicate — this is how a scalar subquery's value reaches the filter that consumes it."""
    if isinstance(cond, list):
        return [resolve_value_refs(c, values) for c in cond]
    if not isinstance(cond, dict):
        return cond
    if "value_ref" in cond:
        out = {k: v for k, v in cond.items() if k != "value_ref"}
        out["value"] = values[cond["value_ref"]]
        return out
    out = {}
    for k, v in cond.items():
        if k in ("and", "or"):
            out[k] = [resolve_value_refs(x, values) for x in v]
        elif k == "not":
            out[k] = resolve_value_refs(v, values)
        else:
            out[k] = v
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
        if step.tool == "condition_filter":
            args["conditions"] = resolve_value_refs(args.get("conditions"), values)
        elif step.tool == "add_to_memory":
            args = {"key": step.args["key"], "value": values.get(step.args["source"]),
                    "content": step.args.get("content", "")}
        method = getattr(harness, step.tool, None)
        if method is None:
            raise ValueError(f"harness has no tool {step.tool!r}")
        result = method(**args)
        if isinstance(result, dict) and "table_name" in result:
            id_to_table[step.id] = result["table_name"]
            final = ("table", result["table_name"])
        elif step.tool == "add_to_memory":
            values[step.args["key"]] = args["value"]
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
