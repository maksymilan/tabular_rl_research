#!/usr/bin/env python3
"""Sampling-only SQL structure profiles for representative atomic-tool cohorts.

This module deliberately does *not* compile SQL into executable tool calls or trajectories.  It
maps gold SQL syntax to aggregate counts for the relational atomic-tool families that SQL can
identify.  The profile is suitable only for hidden cohort balancing; it must never be rendered in
a model/teacher prompt or treated as an action label.

Planning and database-exploration calls (``plan``, ``describe_table``, ``inspect_column``, and
``read_subtable``) are policy decisions and therefore cannot be inferred from SQL.  Their actual
frequencies must be measured after a causal model<->harness rollout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import sqlglot
from sqlglot import exp


PROFILE_VERSION = "sql-atomic-tool-sampling-profile-v1"
SQL_IDENTIFIABLE_ATOMIC_TOOLS = (
    "condition_filter",
    "project",
    "scalar_compute",
    "join_tables",
    "group_aggregate",
    "extreme_value_select",
    "set_op",
    "answer_from_context",
)
NON_SQL_IDENTIFIABLE_ATOMIC_TOOLS = (
    "plan",
    "describe_table",
    "inspect_column",
    "read_subtable",
)

_ARITHMETIC_TYPES = (
    exp.Add,
    exp.Sub,
    exp.Mul,
    exp.Div,
    exp.Mod,
    exp.Pow,
)
_SET_OPERATION_TYPES = (exp.Union, exp.Intersect, exp.Except)


@dataclass(frozen=True)
class SqlAtomicToolProfile:
    """A non-executable, sampling-only structural profile."""

    counts: dict[str, int]
    select_scopes: int
    parse_dialect: str = "sqlite"
    version: str = PROFILE_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parse_dialect": self.parse_dialect,
            "select_scopes": self.select_scopes,
            "counts": dict(self.counts),
            "non_sql_identifiable_tools": list(NON_SQL_IDENTIFIABLE_ATOMIC_TOOLS),
            "sampling_only": True,
            "executable_actions": False,
        }


def _nearest_select(node: exp.Expression) -> exp.Select | None:
    current = node.parent
    while current is not None:
        if isinstance(current, exp.Select):
            return current
        current = current.parent
    return None


def _direct_nodes(
    select: exp.Select,
    node_types: type[exp.Expression] | tuple[type[exp.Expression], ...],
) -> Iterable[exp.Expression]:
    """Yield descendants owned by ``select``, excluding nested SELECT scopes."""
    for node in select.walk():
        if node is select:
            continue
        if isinstance(node, node_types) and _nearest_select(node) is select:
            yield node


def _has_direct_aggregate(select: exp.Select) -> bool:
    return any(True for _ in _direct_nodes(select, exp.AggFunc))


def _has_scalar_arithmetic(select: exp.Select) -> bool:
    """Detect arithmetic over an aggregate/subquery rather than ordinary row expressions."""
    for arithmetic in _direct_nodes(select, _ARITHMETIC_TYPES):
        has_aggregate = any(
            isinstance(node, exp.AggFunc) and _nearest_select(node) is select
            for node in arithmetic.walk()
        )
        has_subquery = any(isinstance(node, exp.Subquery) for node in arithmetic.walk())
        if has_aggregate or has_subquery:
            return True
    return False


def profile_sql(sql: str, *, dialect: str = "sqlite") -> SqlAtomicToolProfile:
    """Return SQL-identifiable atomic-tool demand counts without creating a plan.

    Counts represent minimum structural operation families, not a privileged execution path.  A
    single typed predicate tree or multi-edge join is counted once per SELECT scope because the
    public atomic interface can express each in one call.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL profile input must be a non-empty string")
    tree = sqlglot.parse_one(sql, read=dialect)
    selects = list(tree.find_all(exp.Select))
    counts = {tool: 0 for tool in SQL_IDENTIFIABLE_ATOMIC_TOOLS}

    for select in selects:
        # SELECT-list construction is an output-shaping requirement even when another relational
        # operator may happen to return the same physical columns.
        counts["project"] += 1

        joins = list(select.args.get("joins") or [])
        from_clause = select.args.get("from_")
        extra_from_relations = 0
        if from_clause is not None:
            expressions = list(from_clause.args.get("expressions") or [])
            extra_from_relations = max(0, len(expressions) - 1)
        if joins or extra_from_relations:
            counts["join_tables"] += 1

        if select.args.get("where") is not None:
            counts["condition_filter"] += 1
        if select.args.get("having") is not None:
            # HAVING is a second filter over aggregate-grain rows.
            counts["condition_filter"] += 1

        if select.args.get("group") is not None or _has_direct_aggregate(select):
            counts["group_aggregate"] += 1

        if (
            select.args.get("order") is not None
            or select.args.get("limit") is not None
            or select.args.get("offset") is not None
        ):
            counts["extreme_value_select"] += 1

        if _has_scalar_arithmetic(select):
            counts["scalar_compute"] += 1

    counts["set_op"] = sum(
        1 for node in tree.walk() if isinstance(node, _SET_OPERATION_TYPES)
    )
    counts["answer_from_context"] = 1
    return SqlAtomicToolProfile(counts=counts, select_scopes=len(selects), parse_dialect=dialect)


def sql_from_task(row: dict[str, Any]) -> str:
    """Resolve a harness-only gold SQL field from a normalized task record."""
    value = row.get("gold_sql") or row.get("query") or row.get("SQL")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{row.get('example_id') or '<unknown>'}: missing gold SQL")
    return value


def profile_task(row: dict[str, Any]) -> SqlAtomicToolProfile:
    return profile_sql(sql_from_task(row))

