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


def _compact_evidence_output(output: Any) -> Any:
    """Keep plan evidence grounded in tool output without turning it into a large transcript."""
    if not isinstance(output, dict):
        return deepcopy(output)
    if "table" in output:
        out = {key: deepcopy(output.get(key)) for key in ("table", "kind", "columns", "row_count")
               if output.get(key) is not None}
        if output.get("rows"):
            out["rows"] = deepcopy(output.get("rows", [])[:5])
        return out
    if "result_sample" in output:
        return {
            "row_count": output.get("row_count"),
            "result_sample": deepcopy(output.get("result_sample", [])[:5]),
        }
    if "rows" in output:
        out = {
            "row_count": output.get("row_count"),
            "rows": deepcopy(output.get("rows", [])[:5]),
        }
        if output.get("columns"):
            out["columns"] = deepcopy(output.get("columns"))
        if output.get("table"):
            out["table"] = output.get("table")
        return out
    if "tables" in output:
        tables = []
        for table in output.get("tables", [])[:8]:
            if not isinstance(table, dict):
                tables.append(deepcopy(table))
                continue
            item = dict(table)
            if isinstance(item.get("columns"), list):
                item["columns"] = deepcopy(item["columns"][:40])
            tables.append(item)
        return {"tables": tables}
    if "final_answer" in output:
        return {"final_answer": deepcopy(output.get("final_answer", [])[:50])}
    return deepcopy(output)


def _ground_evidence(value: Any, history: dict[str, dict] | None) -> dict | None:
    """Resolve a model-supplied evidence step id into harness-authored evidence.

    The model only names the step. The environment copies that step's actual tool output into the
    resident plan state, so plan evidence is grounded and cannot contain model-authored conclusions.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        step_id = value.get("step_id") or value.get("id")
    else:
        step_id = value
    if not isinstance(step_id, str) or not step_id.strip():
        raise EnvironmentStateError("plan evidence must be a step_id string")
    step_id = step_id.strip()
    if history is None:
        return {"step_id": step_id}
    record = history.get(step_id)
    if not isinstance(record, dict):
        raise EnvironmentStateError(f"plan evidence references unknown step {step_id!r}")
    return {
        "step_id": step_id,
        "tool": record.get("tool"),
        "output": _compact_evidence_output(record.get("output")),
    }


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
        plan = []
        for item_id in self.plan_order:
            if item_id not in self.plan or self.plan[item_id].get("status") == "deleted":
                continue
            item = self.plan[item_id]
            out = {
                "id": item.get("id"),
                "goal": item.get("goal"),
                "status": item.get("status"),
            }
            if item.get("evidence") is not None:
                out["evidence"] = deepcopy(item["evidence"])
            plan.append(out)
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
    def apply_plan_ops(self, ops: list[dict], step_id: str,
                       history: dict[str, dict] | None = None) -> dict:
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
                evidence_value = op.get("evidence", op.get("evidence_step_id"))
                evidence = _ground_evidence(evidence_value, history)
                if evidence is not None:
                    item["evidence"] = evidence
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

            allowed = {"goal", "status", "evidence", "evidence_step_id"}
            updated = False
            for key in allowed:
                if key not in op:
                    continue
                if key == "status":
                    self.plan[item_id][key] = _status(op[key])
                elif key in {"evidence", "evidence_step_id"}:
                    evidence = _ground_evidence(op[key], history)
                    if evidence is None:
                        self.plan[item_id].pop("evidence", None)
                    else:
                        self.plan[item_id]["evidence"] = evidence
                else:
                    self.plan[item_id][key] = deepcopy(op[key])
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
