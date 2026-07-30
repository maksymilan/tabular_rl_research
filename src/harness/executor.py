#!/usr/bin/env python3
"""Minimal runnable harness executor (core of final_tool_design.md).

Dispatches abstract tool calls by translating them into *composed SQL* over a SQLite backend
(Spider/BIRD DBs are already SQLite; JSON datasets are loaded into SQLite by an adapter). Each
table-producing tool registers a new named view (a SQL SELECT); reading/scalar tools run a
SELECT. This proves the round-trip: a compiled tool chain yields the same result as the
original gold SQL.

Out of scope here (layered above the SQL executor): semantic_match (embedding backend), memory
tools, answer_from_context, provenance/row-id bookkeeping. This module is the relational core.
"""
from __future__ import annotations

import bisect
from datetime import date, datetime
from difflib import SequenceMatcher
import math
import re
import sqlite3
from typing import Any
import unicodedata

_AGG = {
    "sum": "SUM", "count": "COUNT", "count_distinct": "COUNT", "mean": "AVG",
    "avg": "AVG", "min": "MIN", "max": "MAX", "total": "TOTAL",
}
_BOUNDED_VALUE_SEARCH_CANDIDATES_PER_COLUMN = 4_096


def _normalize_search_text(value: Any) -> str:
    """Deterministic lexical normalization used only for value discovery ranking."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.sub(r"[\W_]+", " ", text, flags=re.UNICODE).split())


def _character_ngrams(text: str, size: int = 3) -> set[str]:
    compact = text.replace(" ", "")
    if len(compact) < size:
        return {compact} if compact else set()
    return {
        compact[index:index + size]
        for index in range(len(compact) - size + 1)
    }


def _value_search_candidate_anchors(query: str) -> tuple[str, ...]:
    """Return a small deterministic set of SQL-recall anchors for bounded fuzzy search."""
    normalized = _normalize_search_text(query)
    tokens = sorted(
        {token for token in normalized.split() if len(token) >= 3},
        key=lambda token: (-len(token), token),
    )
    anchors = list(tokens[:4])
    compact = normalized.replace(" ", "")
    trigrams = [
        compact[index:index + 3]
        for index in range(max(0, len(compact) - 2))
    ]
    if trigrams:
        for index in (0, len(trigrams) // 2, len(trigrams) - 1):
            trigram = trigrams[index]
            if trigram and trigram not in anchors:
                anchors.append(trigram)
    if len(compact) >= 3:
        for trigram in (compact[:3], compact[-3:]):
            if trigram not in anchors:
                anchors.append(trigram)
    elif normalized and not anchors:
        anchors.append(normalized)
    return tuple(anchors[:8])


def _escaped_like_contains(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _value_search_match(query: str, value: Any) -> tuple[int, float, str] | None:
    """Return an ascending match tier, descending score, and auditable match kind."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    value_text = str(value)
    query_normalized = _normalize_search_text(query)
    value_normalized = _normalize_search_text(value_text)
    if not query_normalized or not value_normalized:
        return None
    if query == value_text:
        return 0, 1.0, "exact"
    if query_normalized == value_normalized:
        return 1, 1.0, "normalized_exact"
    if value_normalized.startswith(query_normalized):
        score = len(query_normalized) / max(1, len(value_normalized))
        return 2, score, "prefix"
    query_tokens = query_normalized.split()
    value_tokens = value_normalized.split()
    if query_tokens and all(token in value_tokens for token in query_tokens):
        score = len(query_tokens) / max(1, len(value_tokens))
        return 3, score, "token"
    if query_normalized in value_normalized:
        score = len(query_normalized) / max(1, len(value_normalized))
        return 4, score, "substring"
    if len(query_normalized) < 3 or len(value_normalized) > 256:
        return None
    ratio = SequenceMatcher(
        None,
        query_normalized,
        value_normalized,
        autojunk=False,
    ).ratio()
    query_ngrams = _character_ngrams(query_normalized)
    value_ngrams = _character_ngrams(value_normalized)
    union = query_ngrams | value_ngrams
    trigram = (
        len(query_ngrams & value_ngrams) / len(union)
        if union
        else 0.0
    )
    score = max(ratio, trigram)
    threshold = 0.72 if len(query_normalized) >= 5 else 0.80
    if score < threshold:
        return None
    return 5, score, "fuzzy"


def _qid(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _lit(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


class Harness:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.text_factory = lambda b: b.decode("utf-8", "replace")
        self.views: dict[str, str] = {}
        self._lc: dict[str, str] = {}  # lowercased name -> canonical (SQL identifiers are case-insensitive)
        self._n = 0
        self.register_sources()

    def register_sources(self) -> None:
        """(Re)scan sqlite_master and register every base table as a source view."""
        for (name,) in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall():
            self.views.setdefault(name, f'SELECT * FROM "{name}"')
            self._lc.setdefault(name.lower(), name)

    def schema(self) -> dict[str, list[str]]:
        """{lowercased table name -> [column names]} for the compiler's qualified-column mode."""
        out: dict[str, list[str]] = {}
        for (name,) in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            out[name.lower()] = [r[1] for r in self.conn.execute(f'PRAGMA table_info("{name}")')]
        return out

    # ---- helpers ----
    def _sql(self, table: str) -> str:
        if table in self.views:
            return self.views[table]
        canon = self._lc.get(table.lower())  # case-insensitive fallback (SQL identifiers)
        if canon is not None:
            return self.views[canon]
        alias = self._prefix_alias(table)
        if alias is not None:
            return self.views[alias]
        raise KeyError(f"unknown table: {table}; valid tables/handles: {self.available_tables()}")

    def _src(self, table: str) -> str:
        return f"({self._sql(table)})"

    def _cols(self, table: str) -> list[str]:
        cur = self.conn.execute(f"SELECT * FROM {self._src(table)} LIMIT 0")
        return [d[0] for d in cur.description]

    def _cols_of_sql(self, sql: str) -> list[str]:
        cur = self.conn.execute(f"SELECT * FROM ({sql}) LIMIT 0")
        return [d[0] for d in cur.description]

    def _new(self, kind: str, sql: str) -> dict:
        self._n += 1
        name = f"{kind}_{self._n:03d}"
        self.views[name] = sql
        n = self.conn.execute(f"SELECT COUNT(*) FROM ({sql})").fetchone()[0]
        return {"table_name": name, "kind": kind, "row_count": n, "columns": self._cols(name)}

    def rows(self, table: str) -> list[tuple]:
        return self.conn.execute(self._sql(table)).fetchall()

    def table_columns(self, table: str) -> list[str]:
        """Return exact logical columns for a source or derived relation handle."""
        return self._cols(table)

    def count_null_rows(self, table: str, column: str) -> int:
        """Count output rows where one exact/uniquely resolvable logical column is NULL."""
        columns = self._cols(table)
        resolved = self._resolve_col(columns, column)
        if resolved not in columns:
            raise ValueError(
                f"unknown column {column!r} for {table!r}; available columns: {columns}"
            )
        return self.conn.execute(
            f"SELECT COUNT(*) FROM {self._src(table)} WHERE {_qid(resolved)} IS NULL"
        ).fetchone()[0]

    def available_tables(self) -> list[str]:
        return sorted(self.views)

    def _prefix_alias(self, table: str) -> str | None:
        """Map a unique column prefix like T1 back to the one handle that contains T1__* columns.

        This is a compatibility shim for models that confuse join column prefixes with table
        handles. Ambiguous prefixes are left unresolved so the error remains explicit.
        """
        marker = f"{table}__"
        matches = []
        for name, sql in self.views.items():
            try:
                cols = self._cols_of_sql(sql)
            except sqlite3.Error:
                continue
            if any(c.startswith(marker) for c in cols):
                matches.append(name)
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _resolve_col(cols: list[str], requested: str) -> str:
        if not isinstance(requested, str):
            return requested
        lowered = {c.lower(): c for c in cols}
        # version5 joins materialize flat logical names such as ``orders.id``. Prefer an exact
        # logical-column match before the historical table/SQL-qualifier fallback strips the dot.
        if requested.lower() in lowered:
            return lowered[requested.lower()]
        base = requested.split(".")[-1]
        if base.lower() in lowered:
            return lowered[base.lower()]
        suffix_matches = [c for c in cols if c.lower().endswith("__" + base.lower())]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        dotted_matches = [c for c in cols if c.lower().endswith("." + base.lower())]
        return dotted_matches[0] if len(dotted_matches) == 1 else base

    def _col_sql(self, cols: list[str], requested: str) -> str:
        resolved = self._resolve_col(cols, requested)
        return _qid(resolved) if resolved in cols else str(requested)

    def _first_prefixed_join_col_sql(self, cols: list[str], requested: str, prefix: str | None) -> str:
        """Resolve the documented ``P__column`` form on a prefixed join's first edge.

        Prefixes are materialized *after* the first join predicate is evaluated, so this is the
        one point where the model-facing name and the source query's physical column name differ.
        Accept only the declared first-table prefix; arbitrary aliases remain invalid and surface
        as normal execution errors.
        """
        if isinstance(requested, str) and "." in requested:
            return requested
        resolved = self._resolve_col(cols, requested)
        if resolved in cols:
            return _qid(resolved)
        if isinstance(prefix, str) and prefix:
            base = requested.split(".")[-1] if isinstance(requested, str) else requested
            marker = f"{prefix}__"
            if isinstance(base, str) and base.lower().startswith(marker.lower()):
                source_col = self._resolve_col(cols, base[len(marker):])
                if source_col in cols:
                    return _qid(source_col)
        return str(requested)

    def _join_col_sql(self, cols: list[str], requested: str) -> str:
        """Resolve a join argument without accepting SQL implementation aliases."""
        if isinstance(requested, str) and "." in requested:
            return requested
        return self._col_sql(cols, requested)

    def _single_col_select(self, table: str, target_column: str) -> str:
        cols = self._cols(table)
        if len(cols) == 1:
            return cols[0]
        resolved = self._resolve_col(cols, target_column)
        if resolved in cols:
            return resolved
        raise ValueError(
            f"in_table {table!r} must resolve to one column for membership against {target_column!r}; "
            f"available columns: {cols}"
        )

    def _resolve_cond_columns(self, cols: list[str], cond):
        cond = self._normalize_condition(cond)
        if isinstance(cond, list):
            return [self._resolve_cond_columns(cols, item) for item in cond]
        if not isinstance(cond, dict):
            return cond
        out = {}
        for key, value in cond.items():
            if key in {"and", "or"}:
                out[key] = [self._resolve_cond_columns(cols, item) for item in value]
            elif key == "not":
                out[key] = self._resolve_cond_columns(cols, value)
            elif key == "column" and value != "*":
                out[key] = self._resolve_col(cols, value)
            elif key == "column_value":
                out[key] = self._resolve_col(cols, value)
            else:
                out[key] = value
        return out

    def _normalize_condition(self, cond):
        if isinstance(cond, list):
            return [self._normalize_condition(item) for item in cond]
        if not isinstance(cond, dict):
            return cond
        if len(cond) == 1:
            key, value = next(iter(cond.items()))
            if key in {"contains", "like", "in"} and isinstance(value, dict):
                out = dict(value)
                out.setdefault("op", key)
                return self._normalize_condition(out)
        out = {}
        for key, value in cond.items():
            if key in {"and", "or"}:
                out[key] = [self._normalize_condition(item) for item in value]
            elif key == "not":
                out[key] = self._normalize_condition(value)
            elif key == "op" and isinstance(value, str):
                out[key] = value.strip().lower().replace("_", " ")
            else:
                out[key] = value
        return out

    # ---- table-producing tools ----
    def _render_leaf(self, c: dict, cols: list[str] | None = None) -> str:
        col, op = c["column"], str(c.get("op", "=")).strip().lower().replace("_", " ")
        qcol = self._col_sql(cols, col) if cols and col != "*" else (_qid(col) if col != "*" else "*")
        if col == "*" and c.get("value") is None:
            return "1=1"
        if op == "contains":
            return f"{qcol} LIKE '%' || {_lit(c['value'])} || '%'"
        if op == "not contains":
            return f"{qcol} NOT LIKE '%' || {_lit(c['value'])} || '%'"
        if op == "like":
            value = str(c["value"]).replace("*", "%")
            return f"{qcol} LIKE {_lit(value)}"
        if op == "not like":
            value = str(c["value"]).replace("*", "%")
            return f"{qcol} NOT LIKE {_lit(value)}"
        if op in {"in", "not in"} and "in_table" in c:  # membership against an IN-subquery's (single-column) table
            member_col = self._single_col_select(c["in_table"], col)
            neg = "NOT " if op == "not in" else ""
            return f"{qcol} {neg}IN (SELECT {_qid(member_col)} FROM ({self._sql(c['in_table'])}))"
        if op in {"in", "not in"}:
            neg = "NOT " if op == "not in" else ""
            values = c.get("values", c.get("value", []))
            if not isinstance(values, (list, tuple)):
                values = [values]
            return f"{qcol} {neg}IN ({', '.join(_lit(v) for v in values)})"
        if op == "between":
            return f"{qcol} BETWEEN {_lit(c['low'])} AND {_lit(c['high'])}"
        if op == "is null":
            return f"{qcol} IS NULL"
        if op == "is not null":
            return f"{qcol} IS NOT NULL"
        if "column_value" in c:  # column-vs-column predicate
            rhs = self._col_sql(cols, c["column_value"]) if cols else _qid(c["column_value"])
            return f"{qcol} {op} {rhs}"
        return f"{qcol} {op} {_lit(c['value'])}"

    def _render_cond(self, cond, cols: list[str] | None = None) -> str:
        """Render a boolean condition (tree dict with and/or/not, or a list = implicit AND)."""
        if isinstance(cond, list):
            cond = {"and": cond}
        if "and" in cond:
            return "(" + " AND ".join(self._render_cond(x, cols) for x in cond["and"]) + ")"
        if "or" in cond:
            return "(" + " OR ".join(self._render_cond(x, cols) for x in cond["or"]) + ")"
        if "not" in cond:
            return "NOT (" + self._render_cond(cond["not"], cols) + ")"
        return self._render_leaf(cond, cols)

    def condition_filter(self, table: str, conditions, return_columns: list[str] | None = None,
                         preview_k: int | None = None) -> dict:
        source_cols = self._cols(table)
        conditions = self._resolve_cond_columns(source_cols, conditions)
        where = self._render_cond(conditions, source_cols) if conditions else "1=1"
        sql = f"SELECT * FROM {self._src(table)} WHERE {where}"
        if return_columns:
            cols = self._cols_of_sql(sql)
            sel = ", ".join(
                f"{_qid(self._resolve_col(cols, col))} AS {_qid(self._resolve_col(cols, col))}"
                for col in return_columns
            )
            sql = f"SELECT {sel} FROM ({sql})"
        return self._new("filter", sql)

    def derive_column(self, table: str, new_column: str, expression: str) -> dict:
        return self._new(
            "derive", f"SELECT *, ({expression}) AS {new_column} FROM {self._src(table)}"
        )

    def group_aggregate(self, table: str, group_by: list[str], aggregations: list[dict],
                        passthrough: list[str] | None = None, output_layout: str = "rows",
                        category_values: list | None = None,
                        output_columns: list[str] | None = None) -> dict:
        # passthrough: columns selected but not grouped/aggregated (SQLite's lenient bare-column
        # extension; they are functionally dependent on the group key in practice).
        cols = self._cols(table)
        group_by = [self._resolve_col(cols, col) for col in group_by]
        passthrough = [self._resolve_col(cols, col) for col in (passthrough or [])]
        if output_layout not in {"rows", "columns"}:
            raise ValueError("group_aggregate.output_layout must be rows or columns")
        if output_layout == "rows" and (
            category_values is not None or output_columns is not None
        ):
            raise ValueError(
                "group_aggregate.category_values/output_columns require output_layout=columns"
            )
        gb = ", ".join(self._col_sql(cols, col) for col in group_by)
        extra = ", ".join(self._col_sql(cols, col) for col in passthrough)
        if (
            output_layout == "rows"
            and aggregations
            and all(str(a.get("op", "")).lower() == "distinct" for a in aggregations)
        ):
            sel = ", ".join(
                f"{self._col_sql(cols, a.get('column', '*'))} AS {_qid(a.get('as', a.get('column', 'value')))}"
                for a in aggregations
                if a.get("column", "*") != "*"
            )
            return self._new("group", f"SELECT DISTINCT {sel or '*'} FROM {self._src(table)}")
        if (
            len(aggregations) == 1
            and str(aggregations[0].get("op", "")).lower() == "count_distinct"
            and aggregations[0].get("column", "*") == "*"
            and not aggregations[0].get("where")
            and not group_by
            and output_layout == "rows"
        ):
            alias = aggregations[0].get("as", "count")
            return self._new(
                "group",
                f"SELECT COUNT(*) AS {_qid(alias)} FROM (SELECT DISTINCT * FROM {self._src(table)})",
            )

        def render_aggregation(aggregation: dict) -> str:
            op = str(aggregation["op"]).lower()
            column = aggregation.get("column", "*")
            condition = aggregation.get("where")
            if condition:
                if op == "count_distinct" and column == "*":
                    raise ValueError(
                        "group_aggregate count_distinct over * does not support where; "
                        "name the distinct column"
                    )
                resolved = self._resolve_cond_columns(cols, condition)
                predicate = self._render_cond(resolved, cols)
                value_sql = "1" if column == "*" else self._col_sql(cols, column)
                expression = f"CASE WHEN {predicate} THEN {value_sql} END"
            else:
                expression = self._col_sql(cols, column) if column != "*" else "*"
            distinct = "DISTINCT " if op == "count_distinct" else ""
            return f"{_AGG[op]}({distinct}{expression}) AS {_qid(aggregation['as'])}"

        if output_layout == "columns":
            if len(group_by) != 1 or len(aggregations) != 1 or passthrough:
                raise ValueError(
                    "group_aggregate output_layout=columns requires exactly one group_by column, "
                    "one aggregation, and no passthrough"
                )
            if not category_values:
                raise ValueError(
                    "group_aggregate.category_values must be non-empty for output_layout=columns"
                )
            aliases = output_columns or [str(item) for item in category_values]
            if len(aliases) != len(category_values):
                raise ValueError(
                    "group_aggregate.output_columns must have the same length as category_values"
                )
            if len(set(aliases)) != len(aliases):
                raise ValueError("group_aggregate wide output column names must be unique")
            wide_aggregations = []
            for category, alias in zip(category_values, aliases):
                aggregation = dict(aggregations[0])
                category_condition = {
                    "column": group_by[0],
                    "op": "=",
                    "value": category,
                }
                if aggregation.get("where"):
                    aggregation["where"] = {
                        "and": [aggregation["where"], category_condition],
                    }
                else:
                    aggregation["where"] = category_condition
                aggregation["as"] = alias
                wide_aggregations.append(render_aggregation(aggregation))
            return self._new(
                "group",
                f"SELECT {', '.join(wide_aggregations)} FROM {self._src(table)}",
            )

        aggs = ", ".join(render_aggregation(a) for a in aggregations)
        parts = [p for p in (gb, extra, aggs) if p]
        sel = ", ".join(parts) if parts else "*"
        tail = f" GROUP BY {gb}" if gb else ""
        return self._new("group", f"SELECT {sel} FROM {self._src(table)}{tail}")

    def pivot(self, table: str, key_column: str, value_column: str, key_values: list,
              output_columns: list[str] | None = None) -> dict:
        """Turn one grouped key/value row per requested key into one row with one column per key."""
        cols = self._cols(table)
        key = self._resolve_col(cols, key_column)
        value = self._resolve_col(cols, value_column)
        if key not in cols or value not in cols:
            raise ValueError(
                f"pivot requires existing key/value columns; available columns: {cols}"
            )
        if not key_values:
            raise ValueError("pivot.key_values must contain at least one category")
        aliases = output_columns or [str(item) for item in key_values]
        if len(aliases) != len(key_values):
            raise ValueError("pivot.output_columns must have the same length as key_values")
        duplicate = self.conn.execute(
            f"SELECT {_qid(key)}, COUNT(*) FROM {self._src(table)} "
            f"WHERE {_qid(key)} IN ({', '.join(_lit(item) for item in key_values)}) "
            f"GROUP BY {_qid(key)} HAVING COUNT(*) > 1 LIMIT 1"
        ).fetchone()
        if duplicate:
            raise ValueError(
                f"pivot input has {duplicate[1]} rows for key {duplicate[0]!r}; "
                "group to exactly one row per key before pivoting"
            )
        select = ", ".join(
            f"MAX(CASE WHEN {_qid(key)} = {_lit(item)} THEN {_qid(value)} END) AS {_qid(alias)}"
            for item, alias in zip(key_values, aliases)
        )
        return self._new("pivot", f"SELECT {select} FROM {self._src(table)}")

    def _join_component(self, base, joins, base_role=None) -> dict:
        """Execute the version5 public join shape with a flat, stable logical namespace."""
        if not isinstance(base, str) or not base:
            raise ValueError("join_tables.base must be a non-empty table or handle")
        if not isinstance(joins, list) or not joins:
            raise ValueError("join_tables.joins must be a non-empty list")

        def logical_pairs(table: str, role: str | None) -> tuple[str, list[tuple[str, str]]]:
            source_cols = self._cols(table)
            namespace = role or table
            pairs = [
                (column if "." in column else f"{namespace}.{column}", column)
                for column in source_cols
            ]
            lowered = [logical.casefold() for logical, _ in pairs]
            if len(lowered) != len(set(lowered)):
                raise ValueError(
                    f"join_tables namespace {namespace!r} creates duplicate logical columns; "
                    "use a distinct semantic role"
                )
            return namespace, pairs

        _, cur_pairs = logical_pairs(base, base_role)
        cur_src = self._src(base)
        last_sql = None
        join_sql = {"inner": "JOIN", "left": "LEFT JOIN", "cross": "CROSS JOIN"}

        for index, item in enumerate(joins):
            where = f"join_tables.joins[{index}]"
            if not isinstance(item, dict):
                raise ValueError(f"{where} must be an object")
            table = item.get("table")
            edges = item.get("on")
            join_type = item.get("type", "inner")
            if not isinstance(table, str) or not table:
                raise ValueError(f"{where}.table must be a non-empty table or handle")
            if join_type not in join_sql:
                raise ValueError(f"{where}.type must be inner, left, or cross")
            if not isinstance(edges, list):
                raise ValueError(f"{where}.on must be a list")
            if join_type != "cross" and not edges:
                raise ValueError(f"{where}.on must contain at least one equality edge")
            if join_type == "cross" and edges:
                raise ValueError(f"{where}.on must be [] for a cross join")

            namespace, right_pairs = logical_pairs(table, item.get("role"))
            current = {logical.casefold(): (logical, physical) for logical, physical in cur_pairs}
            right_source = {physical.casefold(): physical for _, physical in right_pairs}
            predicates = []
            for edge_index, edge in enumerate(edges):
                edge_where = f"{where}.on[{edge_index}]"
                if not isinstance(edge, dict) or set(edge) != {"left", "right"}:
                    raise ValueError(f"{edge_where} must contain exactly left and right")
                left_ref, right_ref = edge.get("left"), edge.get("right")
                left_match = current.get(left_ref.casefold()) if isinstance(left_ref, str) else None
                right_match = right_source.get(right_ref.casefold()) if isinstance(right_ref, str) else None
                if left_match is None:
                    raise ValueError(
                        f"{edge_where}.left {left_ref!r} is not an introduced logical column; "
                        f"available columns: {[logical for logical, _ in cur_pairs]}"
                    )
                if right_match is None:
                    raise ValueError(
                        f"{edge_where}.right {right_ref!r} is not a column of {table!r}; "
                        f"available columns: {list(right_source.values())}"
                    )
                predicates.append(f"L.{_qid(left_match[1])} = R.{_qid(right_match)}")

            existing = {logical.casefold() for logical, _ in cur_pairs}
            collisions = [
                logical for logical, _ in right_pairs if logical.casefold() in existing
            ]
            if collisions:
                raise ValueError(
                    f"join_tables relation namespace {namespace!r} collides with existing columns "
                    f"{collisions}; add a distinct semantic role for the repeated relation"
                )

            left_select = [
                f"L.{_qid(physical)} AS {_qid(logical)}"
                for logical, physical in cur_pairs
            ]
            right_select = [
                f"R.{_qid(physical)} AS {_qid(logical)}"
                for logical, physical in right_pairs
            ]
            on_clause = f" ON {' AND '.join(predicates)}" if predicates else ""
            last_sql = (
                f"SELECT {', '.join(left_select + right_select)} "
                f"FROM {cur_src} AS L {join_sql[join_type]} {self._src(table)} AS R{on_clause}"
            )
            cur_pairs = [(logical, logical) for logical, _ in cur_pairs + right_pairs]
            cur_src = f"({last_sql})"

        return self._new("join", last_sql)

    def join(self, left, right, on, how="inner", left_alias=None, right_alias=None) -> dict:
        """Execute one symmetric SQL-style equality join edge.

        This is the active public join surface. ``join_tables`` below is retained only for
        deterministic replay of historical trajectories. Each authored join column is resolved
        independently against its own input, so neither side has a special "bare new table"
        convention.
        """
        if not isinstance(left, str) or not left:
            raise ValueError("join.left must be a non-empty table or handle")
        if not isinstance(right, str) or not right:
            raise ValueError("join.right must be a non-empty table or handle")
        if how not in {"inner", "left", "cross"}:
            raise ValueError("join.how must be inner, left, or cross")
        if not isinstance(on, list):
            raise ValueError("join.on must be a list")
        if how == "cross" and on:
            raise ValueError("join.on must be [] when how=cross")
        if how != "cross" and not on:
            raise ValueError("join.on must contain at least one equality pair")

        alias_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        for key, alias in (("left_alias", left_alias), ("right_alias", right_alias)):
            if alias is not None and (
                not isinstance(alias, str) or alias_pattern.fullmatch(alias) is None
            ):
                raise ValueError(f"join.{key} must be a SQL-style identifier")

        left_cols = self._cols(left)
        right_cols = self._cols(right)

        def resolve_input_column(
            columns: list[str],
            requested: str,
            table: str,
            alias: str | None,
            where: str,
        ) -> str:
            if not isinstance(requested, str) or not requested.strip():
                raise ValueError(f"{where} must be a non-empty column reference")
            exact = [item for item in columns if item.casefold() == requested.casefold()]
            if len(exact) == 1:
                return exact[0]

            if alias and requested.casefold().startswith(alias.casefold() + "."):
                # SQL aliases qualify the complete column name of a derived relation. A derived
                # logical column may itself contain dots, e.g. ``filter_001.order_id``; therefore
                # ``g.filter_001.order_id`` strips only the leading alias, not every qualifier.
                aliased_column = requested[len(alias) + 1:]
                alias_matches = [
                    item for item in columns
                    if item.casefold() == aliased_column.casefold()
                ]
                if len(alias_matches) == 1:
                    return alias_matches[0]

            qualifier = None
            base = requested
            if "." in requested:
                qualifier, base = requested.rsplit(".", 1)
                allowed_qualifiers = {table.casefold()}
                if alias:
                    allowed_qualifiers.add(alias.casefold())
                physical_namespaces = {
                    item.rsplit(".", 1)[0].casefold()
                    for item in columns
                    if "." in item
                }
                allowed_qualifiers.update(physical_namespaces)
                if qualifier.casefold() not in allowed_qualifiers:
                    raise ValueError(
                        f"{where} qualifier {qualifier!r} does not name input {table!r}; "
                        f"available columns: {columns}"
                    )
            matches = [
                item for item in columns
                if item.casefold() == base.casefold()
                or item.casefold().endswith("." + base.casefold())
                or item.casefold().endswith("__" + base.casefold())
            ]
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise ValueError(
                    f"{where} {requested!r} is not a column of input {table!r}; "
                    f"available columns: {columns}"
                )
            raise ValueError(
                f"{where} {requested!r} is ambiguous within input {table!r}; "
                f"matching columns: {matches}"
            )

        def logical_pairs(
            table: str,
            columns: list[str],
            alias: str | None,
        ) -> list[tuple[str, str]]:
            if alias:
                pairs = [(f"{alias}.{column}", column) for column in columns]
            else:
                pairs = [
                    (column if "." in column else f"{table}.{column}", column)
                    for column in columns
                ]
            lowered = [logical.casefold() for logical, _ in pairs]
            if len(lowered) != len(set(lowered)):
                raise ValueError(
                    f"join alias {alias!r} collapses distinct columns of {table!r}; "
                    "use the existing exact logical column names"
                )
            return pairs

        predicates = []
        resolved_edges: list[tuple[str, str]] = []
        for index, edge in enumerate(on):
            where = f"join.on[{index}]"
            if not isinstance(edge, dict) or set(edge) != {"left", "right"}:
                raise ValueError(f"{where} must contain exactly left and right")
            left_column = resolve_input_column(
                left_cols, edge.get("left"), left, left_alias, f"{where}.left"
            )
            right_column = resolve_input_column(
                right_cols, edge.get("right"), right, right_alias, f"{where}.right"
            )
            resolved_edges.append((left_column, right_column))
            predicates.append(f"L.{_qid(left_column)} = R.{_qid(right_column)}")

        left_pairs = logical_pairs(left, left_cols, left_alias)
        right_pairs = logical_pairs(right, right_cols, right_alias)
        left_by_physical = {physical: logical for logical, physical in left_pairs}
        right_by_physical = {physical: logical for logical, physical in right_pairs}
        joined_equal_names = {
            left_by_physical[left_column].casefold()
            for left_column, right_column in resolved_edges
            if left_by_physical[left_column].casefold()
            == right_by_physical[right_column].casefold()
        }
        left_names = {logical.casefold() for logical, _ in left_pairs}
        kept_right_pairs = []
        collisions = []
        for logical, physical in right_pairs:
            folded = logical.casefold()
            if folded not in left_names:
                kept_right_pairs.append((logical, physical))
            elif folded not in joined_equal_names:
                collisions.append(logical)
        if collisions:
            raise ValueError(
                f"join outputs have colliding logical columns {collisions}; "
                "use left_alias/right_alias to distinguish repeated relations"
            )

        select_items = [
            f"L.{_qid(physical)} AS {_qid(logical)}"
            for logical, physical in left_pairs
        ] + [
            f"R.{_qid(physical)} AS {_qid(logical)}"
            for logical, physical in kept_right_pairs
        ]
        join_sql = {"inner": "JOIN", "left": "LEFT JOIN", "cross": "CROSS JOIN"}[how]
        on_clause = f" ON {' AND '.join(predicates)}" if predicates else ""
        sql = (
            f"SELECT {', '.join(select_items)} "
            f"FROM {self._src(left)} AS L {join_sql} {self._src(right)} AS R{on_clause}"
        )
        return self._new("join", sql)

    def join_tables(self, left=None, right=None, on=None, join_type="inner",
                    return_columns=None, left_prefix=None, right_prefix=None,
                    tables=None, join_types=None, prefixes=None,
                    base=None, joins=None, base_role=None) -> dict:
        """N-way join folded left-to-right in ONE step (one result handle).

        The version5 public form is ``base`` + ordered ``joins``. Historical version1-version4
        forms remain available here for deterministic replay, but are rejected by the live parser.

        - `tables`: ordered source names / step handles, len >= 2.
        - `on[k]`: the join conditions (list of {left,right}) attaching `tables[k+1]` to the
          accumulated left; `on` has len(tables)-1 entries.
        - `prefixes[i]`: the `<prefix>__<col>` qualifier for `tables[i]` (qualified mode, resolves
          shared / self-join column names INTERNALLY). Omit / None = bare mode with shared-column
          dedup. The accumulated intermediate is prefixed once (on the first fold) and thereafter
          passes through unchanged.
        - `join_types[k]` (or a single `join_type` for all folds): inner | left | cross.

        The legacy 2-table call — `join_tables(left, right, on=[{...}], left_prefix, right_prefix)`
        with a FLAT `on` list — is still accepted and mapped onto the N=2 fold."""
        if base is not None or joins is not None or base_role is not None:
            if any(value is not None for value in (
                left, right, tables, join_types, prefixes, return_columns, left_prefix, right_prefix
            )) or on is not None or join_type != "inner":
                raise ValueError(
                    "join_tables version5 base+joins cannot be mixed with historical join arguments"
                )
            return self._join_component(base, joins, base_role)
        if tables is None:                       # legacy 2-table form -> N=2 fold
            tables = [left, right]
            on = [on or []]
            join_types = [join_type]
            prefixes = None if (left_prefix is None and right_prefix is None) else [left_prefix, right_prefix]
        else:
            on = on or []
            if isinstance(on, dict):
                on = [[on]]
            elif isinstance(on, list) and on and isinstance(on[0], dict):
                # Model-facing n-way joins use nested on-chain syntax:
                #   on=[[{left,right}], ...]
                # In 2-table cases, models often emit the older flat syntax:
                #   on=[{left,right}, ...]
                # Accept that form as a compatibility shim.
                if len(tables) == 2:
                    on = [on]
                elif len(on) == len(tables) - 1:
                    on = [[edge] for edge in on]
                else:
                    raise ValueError(
                        "join_tables.on must be a list of per-join edge lists; "
                        "flat on is only unambiguous for two-table joins"
                    )
            if join_types is None:
                join_types = [join_type] * (len(tables) - 1)
            elif isinstance(join_types, str):
                join_types = [join_types] * (len(tables) - 1)
            if prefixes is not None and len(prefixes) < len(tables):
                prefixes = list(prefixes) + [None] * (len(tables) - len(prefixes))
        jt_map = {"inner": "JOIN", "left": "LEFT JOIN", "cross": "CROSS JOIN"}

        cur_src = self._src(tables[0])           # parenthesized SQL usable in FROM
        cur_cols = self._cols(tables[0])
        cur_prefixed = False
        last_sql = None
        for k in range(1, len(tables)):
            rt = tables[k]
            rc = self._cols(rt)
            jt = jt_map.get(join_types[k - 1] if k - 1 < len(join_types) else "inner", "JOIN")
            edges = on[k - 1] if k - 1 < len(on) else []
            first_prefix = prefixes[0] if prefixes is not None and k == 1 and not cur_prefixed else None
            cond = " AND ".join(
                f"L.{self._first_prefixed_join_col_sql(cur_cols, e['left'], first_prefix)} = "
                f"R.{self._join_col_sql(rc, e['right'])}" for e in edges
            )
            oncl = f" ON {cond}" if edges and jt != "CROSS JOIN" else ""
            if prefixes is not None:
                lp = prefixes[0] if not cur_prefixed else None       # prefix the base only on fold 1
                rp = prefixes[k]
                left_sel = ([f"L.{_qid(c)} AS {_qid(f'{lp}__{c}')}" for c in cur_cols] if lp
                            else [f"L.{_qid(c)}" for c in cur_cols])
                right_sel = [f"R.{_qid(c)} AS {_qid(f'{rp}__{c}')}" for c in rc] if rp else [f"R.{_qid(c)}" for c in rc]
                sel = ", ".join(left_sel + right_sel)
            else:
                # bare mode: dedupe shared column names (case-insensitive; shared cols are equal
                # across the join so keeping the left side is value-correct).
                seen = {c.lower() for c in cur_cols}
                sel = ", ".join([f"L.{_qid(c)}" for c in cur_cols] +
                                [f"R.{_qid(c)}" for c in rc if c.lower() not in seen])
            last_sql = f"SELECT {sel} FROM {cur_src} AS L {jt} {self._src(rt)} AS R{oncl}"
            cur_cols = self._cols_of_sql(last_sql)
            cur_src = f"({last_sql})"
            cur_prefixed = True
        if return_columns:                       # optional explicit projection over the final result
            def qual(name: str) -> str:
                if not isinstance(name, str) or "." in name:
                    raise ValueError(
                        f"join_tables.return_columns uses exact model-facing output columns, never "
                        f"table.column; got {name!r}; available columns: {cur_cols}"
                    )
                resolved = self._resolve_col(cur_cols, name)
                if resolved not in cur_cols:
                    raise ValueError(
                        f"join_tables.return_columns requires an exact output column; got {name!r}; "
                        f"available columns: {cur_cols}"
                    )
                return resolved
            resolved_columns = [qual(c) for c in return_columns]
            sel = ", ".join(f"{_qid(c)} AS {_qid(c)}" for c in resolved_columns)
            last_sql = f"SELECT {sel} FROM ({last_sql})"
        return self._new("join", last_sql)

    def set_op(self, left: str, right: str, op: str) -> dict:
        m = {"union": "UNION", "union_all": "UNION ALL",
             "intersect": "INTERSECT", "except": "EXCEPT"}
        left = self._unwrap_table_ref(left)
        right = self._unwrap_table_ref(right)
        left_sql, left_cols = self._set_side_sql(left)
        right_sql, right_cols = self._set_side_sql(right)
        if len(left_cols) != len(right_cols):
            projected = []
            for col in left_cols:
                resolved = self._resolve_col(right_cols, col)
                if resolved not in right_cols:
                    projected = []
                    break
                projected.append(resolved)
            if projected:
                right_sql = "SELECT " + ", ".join(
                    f"{_qid(src)} AS {_qid(dst)}" for src, dst in zip(projected, left_cols)
                ) + f" FROM ({right_sql})"
            else:
                raise ValueError(
                    f"set_op {op} requires aligned columns; left {left} columns={left_cols}, "
                    f"right {right} columns={right_cols}. Project both sides to the same columns first."
                )
        return self._new("setop", f"{left_sql} {m[op]} {right_sql}")

    @staticmethod
    def _unwrap_table_ref(ref):
        if isinstance(ref, dict) and set(ref) == {"table"}:
            return ref["table"]
        return ref

    def _set_side_sql(self, ref) -> tuple[str, list[str]]:
        if isinstance(ref, str) and "." in ref and ref not in self.views and ref.lower() not in self._lc:
            table, column = ref.rsplit(".", 1)
            cols = self._cols(table)
            resolved = self._resolve_col(cols, column)
            return f"SELECT {self._col_sql(cols, column)} AS {_qid(column)} FROM {self._src(table)}", [column]
        return self._sql(ref), self._cols(ref)

    def window(self, table, partition_by, order_by, fn, as_) -> dict:
        part = f"PARTITION BY {', '.join(partition_by)} " if partition_by else ""
        order = f"ORDER BY {', '.join(order_by)}" if order_by else ""
        return self._new(
            "window", f"SELECT *, {fn}() OVER ({part}{order}) AS {as_} FROM {self._src(table)}"
        )

    # ---- reading / scalar tools ----
    def aggregate(self, table: str, column: str, op: str):
        d = "DISTINCT " if op == "count_distinct" else ""
        cols = self._cols(table)
        if column != "*":
            column = self._col_sql(cols, column)
        return self.conn.execute(
            f"SELECT {_AGG[op]}({d}{column if column != '*' else '*'}) FROM {self._src(table)}"
        ).fetchone()[0]

    def extreme_value_select(self, table, order_by, top_k=None, return_columns=None) -> dict:
        """Table-producing ORDER BY [... LIMIT k]: keep the extreme rows under an ordering.
        Merged from the old `order_limit` — one tool now covers a plain ORDER BY/LIMIT, a top-k
        pick, and multi-column ordering. `order_by`: list of 'col' or 'col DESC'.
        `return_columns`: optional projection. The result is small and inlined by `preview`."""
        cols = self._cols(table)

        def render_order(item: str) -> str:
            parts = item.rsplit(" ", 1)
            if len(parts) == 2 and parts[1].upper() in {"ASC", "DESC"}:
                return f"{self._col_sql(cols, parts[0])} {parts[1].upper()}"
            return self._col_sql(cols, item)

        order = f" ORDER BY {', '.join(render_order(item) for item in order_by)}" if order_by else ""
        lim = f" LIMIT {int(top_k)}" if top_k is not None else ""
        sel = ", ".join(self._col_sql(cols, col) for col in return_columns) if return_columns else "*"
        return self._new("top", f"SELECT {sel} FROM {self._src(table)}{order}{lim}")

    def project(self, table, expressions, distinct: bool = False) -> dict:
        """Realize a SELECT projection: SELECT <expressions> FROM (src). Table-producing.
        `expressions` are SQL column/expression strings, optionally `expr AS alias`.
        `distinct=True` removes duplicate projected rows."""
        cols = self._cols(table)

        def quote_expression_columns(expression: str) -> str:
            """Quote exact/unique logical columns inside a SQL scalar expression.

            SQLite parses ``relation.column`` as a table qualifier, but version6 join outputs use
            that whole string as one physical column name. Rewrite only harness-known identifiers,
            outside single-quoted literals; ambiguous bare suffixes remain unresolved errors.
            """
            aliases: dict[str, str] = {column.casefold(): column for column in cols}
            by_base: dict[str, list[str]] = {}
            for column in cols:
                base = column.rsplit(".", 1)[-1].rsplit("__", 1)[-1]
                by_base.setdefault(base.casefold(), []).append(column)
            for base, matches in by_base.items():
                if len(matches) == 1:
                    aliases.setdefault(base, matches[0])
            ordered = sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True)
            segments = re.split(r"('(?:''|[^'])*')", expression)
            for index in range(0, len(segments), 2):
                segment = segments[index]
                for identifier, canonical in ordered:
                    segment = re.sub(
                        rf'(?<![A-Za-z0-9_."]){re.escape(identifier)}(?![A-Za-z0-9_."])',
                        _qid(canonical),
                        segment,
                        flags=re.I,
                    )
                segments[index] = segment
            return "".join(segments)

        def render(expr: str) -> str:
            raw = str(expr).strip()
            resolved_raw = self._resolve_col(cols, raw)
            if resolved_raw in cols:
                return self._col_sql(cols, resolved_raw)
            alias = re.match(r"^(.+?)\s+AS\s+([A-Za-z_][\w]*)$", raw, re.I)
            if alias and alias.group(1).strip() in cols:
                source = alias.group(1).strip()
                return f"{self._col_sql(cols, source)} AS {_qid(alias.group(2))}"
            cast = re.match(
                r"^\s*([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)?)\s*::\s*([A-Za-z_][\w]*)"
                r"(?:\s+AS\s+([A-Za-z_][\w]*))?\s*$",
                expr,
                re.I,
            )
            if cast:
                col = self._resolve_col(cols, cast.group(1))
                alias = cast.group(3) or col
                return f"CAST({self._col_sql(cols, col)} AS {cast.group(2).upper()}) AS {_qid(alias)}"
            m = re.match(r"^\s*([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)?)\s+AS\s+([A-Za-z_][\w]*)\s*$", expr, re.I)
            if m:
                return f"{self._col_sql(cols, m.group(1))} AS {_qid(m.group(2))}"
            if re.match(r"^\s*[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)?\s*$", expr):
                return self._col_sql(cols, expr)
            if alias:
                return (
                    f"{quote_expression_columns(alias.group(1).strip())} "
                    f"AS {_qid(alias.group(2))}"
                )
            return quote_expression_columns(raw)

        sel = ", ".join(render(expr) for expr in expressions) if expressions else "*"
        select = "SELECT DISTINCT" if distinct else "SELECT"
        return self._new("project", f"{select} {sel} FROM {self._src(table)}")

    def scalar_compute(
        self,
        operation: str,
        operands: list,
        result_name: str = "value",
    ) -> dict:
        """Compute one scalar and return a 1x1 table.

        The online layer resolves model-visible ``value_ref`` operands before calling this method.
        A table-shaped result follows the same scalar-grounding and evidence path as an aggregate.
        """
        if not isinstance(operands, list):
            raise ValueError("scalar_compute.operands must be a list")
        operation = str(operation).strip().lower()
        arity = {
            "add": (2, None),
            "subtract": (2, None),
            "multiply": (2, None),
            "divide": (2, 2),
            "percent": (2, 2),
            "percent_change": (2, 2),
            "date_diff_days": (2, 2),
        }
        if operation not in arity:
            raise ValueError(
                "scalar_compute.operation must be add, subtract, multiply, divide, percent, "
                "percent_change, or date_diff_days"
            )
        minimum, maximum = arity[operation]
        if len(operands) < minimum or maximum is not None and len(operands) > maximum:
            expected = str(minimum) if maximum == minimum else f"at least {minimum}"
            raise ValueError(
                f"scalar_compute {operation} requires {expected} operands; got {len(operands)}"
            )
        if not isinstance(result_name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", result_name
        ):
            raise ValueError("scalar_compute.result_name must be an identifier")
        if any(value is None for value in operands):
            raise ValueError("scalar_compute operands cannot be NULL")

        def numeric(value):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"scalar_compute {operation} requires numeric operands; got {value!r}"
                )
            if not math.isfinite(float(value)):
                raise ValueError("scalar_compute operands must be finite")
            return value

        if operation == "add":
            result = sum(numeric(value) for value in operands)
        elif operation == "subtract":
            result = numeric(operands[0])
            for value in operands[1:]:
                result -= numeric(value)
        elif operation == "multiply":
            result = 1
            for value in operands:
                result *= numeric(value)
        elif operation in {"divide", "percent", "percent_change"}:
            left, right = (numeric(value) for value in operands)
            if right == 0:
                raise ValueError(f"scalar_compute {operation} cannot divide by zero")
            if operation == "divide":
                result = left / right
            elif operation == "percent":
                # Match the common BIRD/SQLite expression
                # CAST(part AS REAL) * 100 / whole exactly. Reordering these
                # operations can differ by one ULP under raw set equality.
                result = left * 100 / right
            else:
                result = (left - right) / right * 100
        else:
            def parse_temporal(value):
                if isinstance(value, datetime):
                    return value
                if isinstance(value, date):
                    return datetime.combine(value, datetime.min.time())
                if not isinstance(value, str):
                    raise ValueError(
                        "scalar_compute date_diff_days requires ISO date/time strings"
                    )
                normalized = value.strip().replace("Z", "+00:00")
                try:
                    return datetime.fromisoformat(normalized)
                except ValueError:
                    try:
                        return datetime.combine(
                            date.fromisoformat(normalized), datetime.min.time()
                        )
                    except ValueError as exc:
                        raise ValueError(
                            "scalar_compute date_diff_days requires ISO date/time strings"
                        ) from exc

            start, end = (parse_temporal(value) for value in operands)
            result = (end - start).total_seconds() / 86400

        return self._new(
            "scalar",
            f"SELECT {_lit(result)} AS {_qid(result_name)}",
        )

    def preview(self, table: str, cell_limit: int = 100) -> dict:
        """Inline a small table's rows (rows*cols <= cell_limit), else a truncated head flagged with
        `is_truncated`. NOTE: superseded as the perception path by V2-ctx — the emitter now emits
        metadata-only handles and the model reads rows explicitly via `read_subtable`. Kept for the
        live rollout's table observation and ad-hoc inspection."""
        cols = self._cols(table)
        total = self.conn.execute(f"SELECT COUNT(*) FROM {self._src(table)}").fetchone()[0]
        max_rows = max(1, cell_limit // max(1, len(cols)))
        rows = self.conn.execute(f"SELECT * FROM {self._src(table)} LIMIT {max_rows}").fetchall()
        out = {"columns": cols, "row_count": total, "rows": [list(r) for r in rows]}
        if total > len(rows):
            out["is_truncated"] = True
            out["note"] = f"showing first {len(rows)} of {total} rows (exceeds {cell_limit}-cell preview budget)"
        return out

    def read_subtable(
        self,
        table,
        columns=None,
        limit=20,
        order_by=None,
        offset=0,
    ):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ValueError("read_subtable.limit must be an integer from 1 to 20")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("read_subtable.offset must be a non-negative integer")
        if offset > 0 and not order_by:
            raise ValueError(
                "read_subtable.offset>0 requires order_by for deterministic pagination"
            )

        available = self._cols(table)
        if columns is not None:
            if (
                not isinstance(columns, list)
                or not columns
                or not all(isinstance(item, str) and item for item in columns)
            ):
                raise ValueError(
                    "read_subtable.columns must be a non-empty list of column names"
                )
            selected = []
            for requested in columns:
                resolved = self._resolve_col(available, requested)
                if resolved not in available:
                    raise ValueError(
                        f"read_subtable.columns contains unknown column {requested!r}; "
                        f"available columns: {available}"
                    )
                selected.append(resolved)
            if len({item.casefold() for item in selected}) != len(selected):
                raise ValueError("read_subtable.columns must not contain duplicates")
        else:
            selected = list(available)

        authored_order = []
        if order_by is not None:
            if (
                not isinstance(order_by, list)
                or not order_by
                or not all(isinstance(item, str) and item.strip() for item in order_by)
            ):
                raise ValueError(
                    'read_subtable.order_by must be a non-empty list of "column" or "column DESC"'
                )
            for item in order_by:
                match = re.fullmatch(r"\s*(.+?)(?:\s+(ASC|DESC))?\s*", item, re.I)
                requested = match.group(1)
                direction = (match.group(2) or "ASC").upper()
                resolved = self._resolve_col(available, requested)
                if resolved not in available:
                    raise ValueError(
                        f"read_subtable.order_by contains unknown column {requested!r}; "
                        f"available columns: {available}"
                    )
                authored_order.append((resolved, direction))

        # Always make the page order total and reproducible. Authored ordering has priority; every
        # remaining column is an ascending tie-breaker. Truly duplicate rows are interchangeable.
        ordered_names = {column.casefold() for column, _ in authored_order}
        total_order = authored_order + [
            (column, "ASC")
            for column in available
            if column.casefold() not in ordered_names
        ]
        select_sql = ", ".join(_qid(column) for column in selected)
        order_sql = ", ".join(
            f"{_qid(column)} {direction}" for column, direction in total_order
        )
        fetched = self.conn.execute(
            f"SELECT {select_sql} FROM {self._src(table)} "
            f"ORDER BY {order_sql} LIMIT ? OFFSET ?",
            (limit + 1, offset),
        ).fetchall()
        page = fetched[:limit]
        has_more = len(fetched) > limit
        return {
            "table": table,
            "columns": selected,
            "rows": [list(row) for row in page],
            "row_count": len(page),
            "limit": limit,
            "offset": offset,
            "order_by": [
                f"{column} {direction}" if direction == "DESC" else column
                for column, direction in authored_order
            ],
            "has_more": has_more,
            "next_offset": offset + len(page) if has_more else None,
        }

    # ---- resident perception (context-management layer: structure + value-domain, no row dump) ----
    def describe_table(self, tables) -> dict:
        """Schema of one or more tables (multi-table in one call to avoid long describe chains):
        columns + types + PK flags + foreign keys. NO row values (that is inspect_column /
        read_subtable). Joins the model's RESIDENT world-model. Works on source tables and on
        derived views (views carry columns but no PK/FK)."""
        if isinstance(tables, str):
            tables = [tables]
        out = []
        for t in tables:
            canon = self._lc.get(t.lower(), t)
            info = list(self.conn.execute(f'PRAGMA table_info("{canon}")'))
            if info:                                            # a real base table
                columns = [{"name": r[1], "type": (r[2] or "text").lower(), "pk": bool(r[5])} for r in info]
                fks = [{"column": r[3], "references": f"{r[2]}.{r[4]}"}
                       for r in self.conn.execute(f'PRAGMA foreign_key_list("{canon}")')]
            else:                                               # a derived view: columns only
                columns = [{"name": c} for c in self._cols(t)]
                fks = []
            n = self.conn.execute(f"SELECT COUNT(*) FROM {self._src(t)}").fetchone()[0]
            out.append({"table_name": t, "row_count": n, "columns": columns, "foreign_keys": fks})
        return {"tables": out}

    def inspect_column(self, table: str, column: str, top_k: int = 10, full_below: int = 50) -> dict:
        """Value-domain of one column for grounding a filter literal (does 'France' exist? spelling?):
        distinct count + most frequent values + NULL flag. NO numeric aggregates. Joins the RESIDENT
        world-model. A small enum-like column (distinct-count <= `full_below`) is shown in FULL, so a
        filter literal on it is genuinely visible instead of lost to truncation; a larger column shows
        the `top_k` most-frequent values and is marked `truncated` (there the literal cannot be
        confirmed here — the condition_filter result validates it)."""
        src = self._src(table)
        qcol = self._col_sql(self._cols(table), column)
        n_distinct = self.conn.execute(f"SELECT COUNT(DISTINCT {qcol}) FROM {src}").fetchone()[0]
        n_null = self.conn.execute(f"SELECT COUNT(*) FROM {src} WHERE {qcol} IS NULL").fetchone()[0]
        limit = n_distinct if n_distinct <= full_below else int(top_k)
        freq = self.conn.execute(
            f"SELECT {qcol}, COUNT(*) c FROM {src} GROUP BY {qcol} ORDER BY c DESC LIMIT {limit}"
        ).fetchall()
        return {"column": column, "distinct_count": n_distinct, "has_null": bool(n_null),
                "frequent_values": [v for v, _ in freq], "truncated": n_distinct > limit}

    def search_values(
        self,
        table: str,
        query: str,
        column: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        """Search a deterministic bounded candidate pool of stored values in one table."""
        if not isinstance(table, str) or not table.strip():
            raise ValueError("search_values.table must be a non-empty table name")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search_values.query must be a non-empty string")
        if len(query) > 256:
            raise ValueError("search_values.query cannot exceed 256 characters")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > 20
        ):
            raise ValueError("search_values.limit must be an integer from 1 to 20")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
        ):
            raise ValueError("search_values.offset must be a non-negative integer")

        available = self._cols(table)
        if column is None:
            searched_columns = list(available)
        else:
            resolved = self._resolve_col(available, column)
            if resolved not in available:
                raise ValueError(
                    f"unknown column {column!r} for {table!r}; "
                    f"available columns: {available}"
                )
            searched_columns = [resolved]

        source = self._src(table)
        anchors = _value_search_candidate_anchors(query)
        candidate_limit = _BOUNDED_VALUE_SEARCH_CANDIDATES_PER_COLUMN
        candidates: dict[tuple[str, str, str], tuple[Any, int, str]] = {}
        truncated_columns: list[str] = []

        def add_candidate(value: Any, frequency: Any, searched_column: str) -> None:
            key = (searched_column, type(value).__name__, repr(value))
            candidates[key] = (value, int(frequency), searched_column)

        for searched_column in searched_columns:
            quoted = _qid(searched_column)
            exact_rows = self.conn.execute(
                f"SELECT {quoted}, COUNT(*) AS frequency "
                f"FROM {source} "
                f"WHERE {quoted} IS NOT NULL "
                f"AND CAST({quoted} AS TEXT) = ? COLLATE NOCASE "
                f"GROUP BY {quoted}",
                (query,),
            ).fetchall()
            for value, frequency in exact_rows:
                add_candidate(value, frequency, searched_column)

        has_exact_candidate = any(
            (
                (matched := _value_search_match(query, value)) is not None
                and matched[0] <= 1
            )
            for value, _, _ in candidates.values()
        )

        if not has_exact_candidate and anchors:
            for searched_column in searched_columns:
                quoted = _qid(searched_column)
                conditions = " OR ".join(
                    f"LOWER(CAST({quoted} AS TEXT)) LIKE ? ESCAPE '\\'"
                    for _ in anchors
                )
                parameters = [
                    *(_escaped_like_contains(anchor.lower()) for anchor in anchors),
                    len(query),
                    candidate_limit + 1,
                ]
                recalled = self.conn.execute(
                    f"SELECT {quoted}, COUNT(*) AS frequency "
                    f"FROM {source} "
                    f"WHERE {quoted} IS NOT NULL AND ({conditions}) "
                    f"GROUP BY {quoted} "
                    f"ORDER BY ABS(LENGTH(CAST({quoted} AS TEXT)) - ?) ASC, "
                    f"COUNT(*) DESC, LOWER(CAST({quoted} AS TEXT)) ASC, "
                    f"TYPEOF({quoted}) ASC, CAST({quoted} AS TEXT) COLLATE BINARY ASC "
                    f"LIMIT ?",
                    parameters,
                ).fetchall()
                if len(recalled) > candidate_limit:
                    truncated_columns.append(searched_column)
                    recalled = recalled[:candidate_limit]
                for value, frequency in recalled:
                    add_candidate(value, frequency, searched_column)

        retain = offset + limit + 1
        ranked: list[tuple[tuple, dict]] = []
        total_matches = 0
        for value, frequency, searched_column in candidates.values():
            matched = _value_search_match(query, value)
            if matched is None:
                continue
            tier, score, match_type = matched
            total_matches += 1
            item = {
                "table": table,
                "column": searched_column,
                "value": value,
                "frequency": frequency,
                "match_type": match_type,
                "score": round(float(score), 6),
            }
            sort_key = (
                tier,
                -float(score),
                -frequency,
                searched_column.casefold(),
                _normalize_search_text(value),
                type(value).__name__,
                str(value),
            )
            bisect.insort(ranked, (sort_key, item))
            if len(ranked) > retain:
                ranked.pop()

        matches = [item for _, item in ranked[offset:offset + limit]]
        consumed = offset + len(matches)
        has_more = consumed < total_matches
        return {
            "table": table,
            "query": query,
            "column": column,
            "searched_columns": searched_columns,
            "matches": matches,
            "total_matches": total_matches,
            "total_matches_scope": "bounded_candidate_pool",
            "candidate_count": len(candidates),
            "candidate_limit_per_column": candidate_limit,
            "candidate_truncated": bool(truncated_columns),
            "truncated_columns": truncated_columns,
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
            "next_offset": consumed if has_more else None,
        }

    # ---- verification ----
    def gold(self, sql: str) -> list[tuple]:
        return self.conn.execute(sql).fetchall()


def _selftest() -> None:
    h = Harness(":memory:")
    h.conn.executescript(
        """
        CREATE TABLE employees(id INT, name TEXT, dept TEXT, salary INT);
        INSERT INTO employees VALUES
          (1,'A','eng',1200),(2,'B','eng',900),(3,'C','sales',1500),
          (4,'D','sales',1100),(5,'E','eng',2000),(6,'F','hr',800);
        CREATE TABLE depts(dept TEXT, location TEXT);
        INSERT INTO depts VALUES ('eng','SF'),('sales','NY'),('hr','LA');
        """
    )
    h.register_sources()

    def eq(a, b):
        return sorted(map(repr, a)) == sorted(map(repr, b))

    # T1: filter -> group_aggregate  ==  WHERE ... GROUP BY ...
    f = h.condition_filter("employees", [{"column": "salary", "op": ">", "value": 1000}])
    g = h.group_aggregate(f["table_name"], ["dept"], [{"op": "mean", "column": "salary", "as": "avg_sal"}])
    gold = h.gold("SELECT dept, AVG(salary) FROM employees WHERE salary>1000 GROUP BY dept")
    assert eq(h.rows(g["table_name"]), gold), ("T1", h.rows(g["table_name"]), gold)

    # T2: filter -> join (column from each side)
    j = h.join_tables(
        f["table_name"], "depts",
        on=[{"left": "dept", "right": "dept"}], return_columns=["name", "location"],
    )
    gold2 = h.gold("SELECT e.name, d.location FROM employees e JOIN depts d ON e.dept=d.dept WHERE e.salary>1000")
    assert eq(h.rows(j["table_name"]), gold2), ("T2", h.rows(j["table_name"]), gold2)

    # T3: derive_column -> aggregate scalar  ==  SUM(expr)
    d = h.derive_column("employees", "bonus", "salary * 0.1")
    total = h.aggregate(d["table_name"], "bonus", "sum")
    assert abs(total - h.gold("SELECT SUM(salary*0.1) FROM employees")[0][0]) < 1e-9, ("T3", total)

    # T4: extreme_value_select (table-producing ORDER BY ... LIMIT)
    top = h.extreme_value_select("employees", ["salary DESC"], 2, ["name", "salary"])
    assert h.rows(top["table_name"]) == h.gold(
        "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 2"), ("T4", top)

    print("all harness self-tests passed (T1 filter+group, T2 join, T3 derive+aggregate, T4 extreme)")


if __name__ == "__main__":
    _selftest()
