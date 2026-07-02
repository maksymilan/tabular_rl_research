#!/usr/bin/env python3
"""Task-local environment state shown to the model.

This module is intentionally independent of SQLite execution. The harness owns the
facts; this layer only organizes what has already been exposed to the model:

- task plan items authored through the model-visible plan tool;
- source and derived table handles;
- schemas observed through describe_table;
- column domains observed through inspect_column;
- bounded row reads observed through read_subtable.

The same updater is used by offline trajectory emission and online rollout so the
SFT text and the RL/eval context do not drift.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


class EnvironmentStateError(ValueError):
    """Invalid model-authored environment-state operation."""


def _table_key(name: str) -> str:
    return str(name)


def _status(value: Any) -> str:
    s = str(value or "pending")
    if s not in {"pending", "in_progress", "done", "blocked", "deleted"}:
        raise EnvironmentStateError(f"invalid plan status {s!r}")
    return s


class EnvironmentState:
    """Resident task state rendered alongside transient observations."""

    def __init__(self, catalog: dict | None = None):
        self.catalog = deepcopy(catalog or {})
        self.plan: dict[str, dict] = {}
        self.plan_order: list[str] = []
        self.tables: dict[str, dict] = {}
        for table in self.catalog.get("tables", []):
            name = table.get("table_name")
            if not name:
                continue
            self.tables[_table_key(name)] = {
                "kind": "source",
                "row_count": table.get("num_rows", table.get("row_count")),
                "schema": None,
                "inspected_columns": {},
                "reads": [],
                "created_by": None,
                "columns": None,
            }

    def snapshot(self) -> dict:
        """Return a compact, JSON-serializable resident-state snapshot."""
        plan = [deepcopy(self.plan[item_id]) for item_id in self.plan_order
                if item_id in self.plan and self.plan[item_id].get("status") != "deleted"]
        tables = {}
        for name, table in self.tables.items():
            if (
                table.get("kind") == "source"
                and table.get("schema") is None
                and not table.get("inspected_columns")
                and not table.get("reads")
                and table.get("columns") is None
                and table.get("created_by") is None
            ):
                continue
            out = {k: deepcopy(v) for k, v in table.items()
                   if v not in (None, {}, [])}
            tables[name] = out
        return {"plan": plan, "tables": tables}

    # ---- plan tool ----
    def apply_plan_ops(self, ops: list[dict], step_id: str) -> dict:
        if not isinstance(ops, list) or not ops:
            raise EnvironmentStateError("plan.ops must be a non-empty list")
        changes = []
        for op in ops:
            if not isinstance(op, dict):
                raise EnvironmentStateError("each plan op must be an object")
            action = op.get("op")
            item_id = op.get("id")
            if action not in {"create", "add", "update", "delete"}:
                raise EnvironmentStateError(f"invalid plan op {action!r}")
            if not isinstance(item_id, str) or not item_id.strip():
                raise EnvironmentStateError("plan op requires a non-empty string id")
            item_id = item_id.strip()
            if action in {"create", "add"}:
                if item_id in self.plan and self.plan[item_id].get("status") != "deleted":
                    raise EnvironmentStateError(f"plan item {item_id!r} already exists")
                goal = op.get("goal")
                if not isinstance(goal, str) or not goal.strip():
                    raise EnvironmentStateError(f"{action} requires a non-empty goal")
                item = {
                    "id": item_id,
                    "goal": goal.strip(),
                    "status": _status(op.get("status")),
                    "created_by": step_id,
                    "updated_by": step_id,
                }
                for key in ("depends_on", "notes", "evidence_step_id"):
                    if key in op:
                        item[key] = deepcopy(op[key])
                self.plan[item_id] = item
                if item_id not in self.plan_order:
                    self.plan_order.append(item_id)
                changes.append({"op": action, "id": item_id})
                continue

            if item_id not in self.plan or self.plan[item_id].get("status") == "deleted":
                raise EnvironmentStateError(f"unknown plan item {item_id!r}")
            if action == "delete":
                self.plan[item_id]["status"] = "deleted"
                self.plan[item_id]["updated_by"] = step_id
                if "reason" in op:
                    self.plan[item_id]["delete_reason"] = op["reason"]
                changes.append({"op": "delete", "id": item_id})
                continue

            allowed = {"goal", "status", "depends_on", "notes", "evidence_step_id"}
            updated = False
            for key in allowed:
                if key not in op:
                    continue
                self.plan[item_id][key] = _status(op[key]) if key == "status" else deepcopy(op[key])
                updated = True
            if not updated:
                raise EnvironmentStateError(f"update for {item_id!r} changes no fields")
            self.plan[item_id]["updated_by"] = step_id
            changes.append({"op": "update", "id": item_id})
        return {"changes": changes, "plan": self.snapshot()["plan"]}

    # ---- table context updates ----
    def apply_tool_result(self, tool: str, args: dict, output: dict, step_id: str) -> None:
        if tool == "describe_table":
            for table in output.get("tables", []):
                name = table.get("table_name")
                if not name:
                    continue
                entry = self._ensure_table(name)
                entry["row_count"] = table.get("row_count", entry.get("row_count"))
                entry["schema"] = {
                    "from_step": step_id,
                    "columns": deepcopy(table.get("columns", [])),
                    "foreign_keys": deepcopy(table.get("foreign_keys", [])),
                }
            return

        if tool == "inspect_column":
            table = args.get("table")
            column = output.get("column") or args.get("column")
            if table and column:
                entry = self._ensure_table(table)
                entry.setdefault("inspected_columns", {})[column] = {
                    "from_step": step_id,
                    "distinct_count": output.get("distinct_count"),
                    "has_null": output.get("has_null"),
                    "frequent_values": deepcopy(output.get("frequent_values", [])),
                    "truncated": output.get("truncated"),
                    **({"queried_value": output.get("queried_value"),
                        "value_present": output.get("value_present")}
                       if "queried_value" in output else {}),
                }
            return

        if tool == "read_subtable":
            table = args.get("table") or output.get("table")
            if table:
                entry = self._ensure_table(table)
                read = {
                    "from_step": step_id,
                    "columns": deepcopy(args.get("columns")),
                    "limit": args.get("limit"),
                    "row_count": output.get("row_count"),
                    "rows": deepcopy(output.get("rows", [])),
                }
                entry.setdefault("reads", []).append(read)
            return

        table_name = output.get("table")
        if table_name:
            entry = self._ensure_table(table_name)
            entry["kind"] = output.get("kind", entry.get("kind", "derived"))
            entry["created_by"] = step_id
            entry["columns"] = deepcopy(output.get("columns"))
            entry["row_count"] = output.get("row_count", entry.get("row_count"))
            if output.get("rows"):
                entry.setdefault("reads", []).append({
                    "from_step": step_id,
                    "row_count": output.get("row_count"),
                    "rows": deepcopy(output.get("rows", [])),
                    "note": "scalar-shaped table output",
                })

    def _ensure_table(self, name: str) -> dict:
        key = _table_key(name)
        if key not in self.tables:
            self.tables[key] = {
                "kind": "derived",
                "row_count": None,
                "schema": None,
                "inspected_columns": {},
                "reads": [],
                "created_by": None,
                "columns": None,
            }
        return self.tables[key]
