"""Task-agnostic expression lineage and condition-dependency analysis."""
from __future__ import annotations

import re
from typing import Any


def _unique_source_aliases(
    columns: list[str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    exact = {column.casefold(): column for column in columns}
    by_suffix: dict[str, list[str]] = {}
    for column in columns:
        suffix = column.rsplit(".", 1)[-1].rsplit("__", 1)[-1].casefold()
        by_suffix.setdefault(suffix, []).append(column)
    return exact, by_suffix


def _resolve_source_column(
    token: str,
    exact: dict[str, str],
    by_suffix: dict[str, list[str]],
) -> str | None:
    resolved = exact.get(token.casefold())
    if resolved is not None:
        return resolved
    matches = by_suffix.get(token.casefold(), [])
    return matches[0] if len(matches) == 1 else None


def expression_core(expression: str) -> str:
    """Remove a simple trailing SQL alias from one projection expression."""
    alias = re.match(
        r"^\s*(.+?)\s+AS\s+[A-Za-z_][A-Za-z0-9_]*\s*$",
        str(expression),
        re.I,
    )
    return (alias.group(1) if alias else str(expression)).strip()


def expression_source_columns(
    expression: str,
    available_columns: list[str],
) -> list[str]:
    """Return exact/unique source columns referenced by a scalar project expression."""
    core = expression_core(str(expression))
    exact, by_suffix = _unique_source_aliases(available_columns)
    direct = _resolve_source_column(core, exact, by_suffix)
    if direct is not None:
        return [direct]
    without_literals = re.sub(r"'(?:''|[^'])*'", " ", str(expression))
    tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?",
        without_literals,
    )
    resolved: list[str] = []
    for token in tokens:
        column = _resolve_source_column(token, exact, by_suffix)
        if column is not None and column not in resolved:
            resolved.append(column)
    return resolved


def expression_kind(
    expression: str,
    available_columns: list[str],
    sources: list[str],
) -> str:
    """Classify a projection item without interpreting task intent."""
    core = expression_core(expression)
    exact, by_suffix = _unique_source_aliases(available_columns)
    if _resolve_source_column(core, exact, by_suffix) is not None:
        return "column"
    if not sources:
        return "constant"
    return "expression"


def condition_columns(condition: Any) -> list[str]:
    """Return condition columns in first-occurrence order."""
    columns: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return
        for key in ("and", "or"):
            if key in node:
                visit(node[key])
        if "not" in node:
            visit(node["not"])
        for key in ("column", "column_value"):
            value = node.get(key)
            if isinstance(value, str) and value not in columns:
                columns.append(value)

    visit(condition)
    return columns


def condition_inputs(condition: Any) -> list[dict]:
    """Return unique table/value dependencies referenced inside a condition tree."""
    inputs: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, role: str, ref: str) -> None:
        key = (kind, ref)
        if key in seen:
            return
        seen.add(key)
        inputs.append({"kind": kind, "role": role, "ref": ref})

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return
        value_ref = node.get("value_ref")
        if isinstance(value_ref, str):
            add("value", "predicate_value", value_ref)
        in_table = node.get("in_table")
        if isinstance(in_table, str):
            add("table", "predicate_membership", in_table)
        for value in node.values():
            visit(value)

    visit(condition)
    return inputs
