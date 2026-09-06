#!/usr/bin/env python3
"""Harness-owned target support for process-reward potential shaping.

Gold SQL is hidden from the actor.  On training tasks only, the reward side may execute and parse
it to construct the bounded T/C/R support sets described by the process-reward contract:

* T: source relations referenced by the gold query;
* C: source columns referenced by the gold query;
* R: normalized gold-result rows (or one explicit empty-result unit).

The module never compiles the gold query into model actions and never changes environment state.
It is deliberately independent from rollout policy, provenance, and relation execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlglot import exp, parse_one
from sqlglot.optimizer.scope import Scope, traverse_scope


EMPTY_RESULT_UNIT = "__verified_empty_result__"
MAX_TARGET_SUPPORT_ROWS = 10_000


def canonical_table(value: str) -> str:
    return value.strip().casefold()


def base_column(value: str) -> str:
    return value.rsplit(".", 1)[-1].strip().strip('"`[]').casefold()


def qualified_column(table: str, column: str) -> str:
    return f"{canonical_table(table)}.{base_column(column)}"


def normalized_row_unit(row: Any) -> tuple[Any, ...]:
    """Use the same raw SQLite cell equality as the BIRD reference set scorer."""
    return tuple(row)


@dataclass(frozen=True)
class TargetSupport:
    tables: frozenset[str]
    columns: frozenset[str]
    row_units: frozenset[tuple[Any, ...] | str]
    output_columns: tuple[str, ...]
    sql_parse_complete: bool
    rows_complete: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


def build_target_support(harness, gold_sql: str) -> TargetSupport:
    """Build bounded hidden target support without exposing it to the actor."""
    cursor = harness.conn.execute(gold_sql)
    output_columns = tuple(
        str(item[0]) if item and item[0] is not None else f"column_{index + 1}"
        for index, item in enumerate(cursor.description or ())
    )
    gold_rows = cursor.fetchmany(MAX_TARGET_SUPPORT_ROWS + 1)
    gold_result_exhaustive = len(gold_rows) <= MAX_TARGET_SUPPORT_ROWS
    bounded_rows = gold_rows[:MAX_TARGET_SUPPORT_ROWS]
    row_units: set[tuple[Any, ...] | str]
    if not bounded_rows:
        row_units = {EMPTY_RESULT_UNIT}
    else:
        row_units = {normalized_row_unit(row) for row in bounded_rows}

    tables: set[str] = set()
    columns: set[str] = set()
    ambiguous_columns: list[str] = []
    parse_error: str | None = None
    try:
        tree = parse_one(gold_sql, read="sqlite")
        source_table_names = set(harness.schema())
        table_nodes = [
            node
            for node in tree.find_all(exp.Table)
            if canonical_table(node.name) in source_table_names
        ]
        table_columns: dict[str, set[str]] = {}
        for node in table_nodes:
            name = canonical_table(node.name)
            tables.add(name)
            try:
                table_columns[name] = {base_column(column) for column in harness._cols(node.name)}
            except Exception:
                table_columns[name] = set()

        for scope in traverse_scope(tree):
            physical_sources: dict[str, str] = {}
            has_derived_source = False
            for alias, (_, selected) in scope.selected_sources.items():
                if isinstance(selected, exp.Table):
                    name = canonical_table(selected.name)
                    if name in source_table_names:
                        physical_sources[canonical_table(alias)] = name
                elif isinstance(selected, Scope):
                    has_derived_source = True
            for node in scope.columns:
                column = base_column(node.name)
                if column == "*":
                    continue
                if node.table:
                    owner = physical_sources.get(canonical_table(node.table))
                    if owner:
                        columns.add(qualified_column(owner, column))
                    else:
                        # A column projected from a CTE/subquery is grounded by the nested scope.
                        # Keep a suffix target as a conservative bridge for SELECT * projections.
                        if not any(base_column(item) == column for item in columns):
                            columns.add(f"*.{column}")
                            ambiguous_columns.append(column)
                    continue
                candidates = [
                    table
                    for table in physical_sources.values()
                    if not table_columns.get(table) or column in table_columns[table]
                ]
                if len(candidates) == 1:
                    columns.add(qualified_column(candidates[0], column))
                elif candidates or (
                    has_derived_source
                    and not any(base_column(item) == column for item in columns)
                ):
                    columns.add(f"*.{column}")
                    ambiguous_columns.append(column)
        sql_parse_complete = True
    except Exception as exc:  # malformed benchmark SQL must disable T/C shaping, not the episode
        sql_parse_complete = False
        parse_error = f"{type(exc).__name__}: {exc}"

    return TargetSupport(
        tables=frozenset(tables),
        columns=frozenset(columns),
        row_units=frozenset(row_units),
        output_columns=output_columns,
        sql_parse_complete=sql_parse_complete,
        # R is deliberately a bounded set of verifiable target evidence units.  The selected
        # support set is complete even when the full answer denotation contains more rows; the
        # reward contract does not require enumerating every raw/result row.
        rows_complete=True,
        diagnostics={
            "ambiguous_unqualified_columns": sorted(set(ambiguous_columns)),
            "parse_error": parse_error,
            "target_row_units": len(row_units),
            "target_rows_capped": not gold_result_exhaustive,
            "gold_result_exhaustive": gold_result_exhaustive,
        },
    )


def matching_target_columns(
    column: str,
    root_tables: set[str],
    target: TargetSupport,
) -> set[str]:
    """Return exact target-column units supported by one observed logical column."""
    suffix = base_column(column)
    namespace = None
    if "." in column:
        namespace = canonical_table(column.rsplit(".", 1)[0])
    matches: set[str] = set()
    wildcard = f"*.{suffix}"
    if wildcard in target.columns:
        matches.add(wildcard)
    candidate_tables = {canonical_table(item) for item in root_tables}
    if namespace and namespace in target.tables:
        candidate_tables.add(namespace)
    for table in candidate_tables:
        unit = qualified_column(table, suffix)
        if unit in target.columns:
            matches.add(unit)
    return matches


def target_projection_indices(
    columns: list[str],
    target: TargetSupport,
) -> tuple[list[int], str]:
    """Resolve a relation/observation to the target output orientation."""
    target_width = len(target.output_columns)
    if target_width and len(columns) == target_width:
        return list(range(target_width)), "position"
    if not target_width or len(columns) <= target_width:
        return [], "none"
    selected_indices: list[int] = []
    used: set[int] = set()
    for wanted in target.output_columns:
        suffix = base_column(wanted)
        candidates = [
            index
            for index, candidate in enumerate(columns)
            if index not in used and base_column(candidate) == suffix
        ]
        if len(candidates) != 1:
            return [], "none"
        selected_indices.append(candidates[0])
        used.add(candidates[0])
    return selected_indices, "named_projection"


def observed_target_row_units(
    columns: list[str],
    rows: list[list[Any]] | list[tuple[Any, ...]],
    target: TargetSupport,
) -> set[tuple[Any, ...] | str]:
    """Return target evidence units actually exposed by one bounded row observation."""
    selected_indices, _ = target_projection_indices(columns, target)
    if not selected_indices:
        return set()
    if not rows and target.row_units == frozenset({EMPTY_RESULT_UNIT}):
        return {EMPTY_RESULT_UNIT}
    if EMPTY_RESULT_UNIT in target.row_units:
        return set()
    return {
        normalized_row_unit([row[index] for index in selected_indices])
        for row in rows
    } & set(target.row_units)


def table_target_row_units(
    harness,
    table: str,
    target: TargetSupport,
) -> tuple[set[tuple[Any, ...] | str], dict[str, Any]]:
    """Find gold-result support present in a resident table.

    Equal-width tables are compared positionally because BIRD denotation ignores output column
    labels.  Wider intermediate tables are compared only when every gold output name resolves
    uniquely to a physical column, avoiding arbitrary column-subset search.
    """
    columns = list(harness._cols(table))
    selected_indices, matching_mode = target_projection_indices(columns, target)

    row_count = int(
        harness.conn.execute(f"SELECT COUNT(*) FROM {harness._src(table)}").fetchone()[0]
    )
    if row_count == 0 and target.row_units == frozenset({EMPTY_RESULT_UNIT}) and selected_indices:
        return {EMPTY_RESULT_UNIT}, {
            "compatible": True,
            "matching_mode": matching_mode,
            "rows_complete": True,
        }
    if not selected_indices or not target.row_units or EMPTY_RESULT_UNIT in target.row_units:
        return set(), {
            "compatible": bool(selected_indices),
            "matching_mode": matching_mode,
            "rows_complete": row_count <= MAX_TARGET_SUPPORT_ROWS,
        }

    quoted = ", ".join(f'"{columns[index].replace(chr(34), chr(34) * 2)}"' for index in selected_indices)
    rows = harness.conn.execute(
        f"SELECT {quoted} FROM {harness._src(table)} LIMIT {MAX_TARGET_SUPPORT_ROWS + 1}"
    ).fetchall()
    rows_complete = len(rows) <= MAX_TARGET_SUPPORT_ROWS
    units = {
        normalized_row_unit(row)
        for row in rows[:MAX_TARGET_SUPPORT_ROWS]
    } & set(target.row_units)
    return units, {
        "compatible": True,
        "matching_mode": matching_mode,
        "rows_complete": rows_complete,
    }
