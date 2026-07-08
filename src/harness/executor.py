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

import re
import sqlite3
from typing import Any

_AGG = {
    "sum": "SUM", "count": "COUNT", "count_distinct": "COUNT", "mean": "AVG",
    "avg": "AVG", "min": "MIN", "max": "MAX", "total": "TOTAL",
}


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
        base = requested.split(".")[-1]
        lowered = {c.lower(): c for c in cols}
        if base.lower() in lowered:
            return lowered[base.lower()]
        suffix_matches = [c for c in cols if c.lower().endswith("__" + base.lower())]
        return suffix_matches[0] if len(suffix_matches) == 1 else base

    def _col_sql(self, cols: list[str], requested: str) -> str:
        resolved = self._resolve_col(cols, requested)
        return _qid(resolved) if resolved in cols else str(requested)

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
        if op == "is_null":
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
                f"{_qid(self._resolve_col(cols, col))} AS {_qid(col.split('.')[-1])}"
                for col in return_columns
            )
            sql = f"SELECT {sel} FROM ({sql})"
        return self._new("filter", sql)

    def derive_column(self, table: str, new_column: str, expression: str) -> dict:
        return self._new(
            "derive", f"SELECT *, ({expression}) AS {new_column} FROM {self._src(table)}"
        )

    def group_aggregate(self, table: str, group_by: list[str], aggregations: list[dict],
                        passthrough: list[str] | None = None) -> dict:
        # passthrough: columns selected but not grouped/aggregated (SQLite's lenient bare-column
        # extension; they are functionally dependent on the group key in practice).
        cols = self._cols(table)
        group_by = [self._resolve_col(cols, col) for col in group_by]
        passthrough = [self._resolve_col(cols, col) for col in (passthrough or [])]
        gb = ", ".join(self._col_sql(cols, col) for col in group_by)
        extra = ", ".join(self._col_sql(cols, col) for col in passthrough)
        if aggregations and all(str(a.get("op", "")).lower() == "distinct" for a in aggregations):
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
            and not group_by
        ):
            alias = aggregations[0].get("as", "count")
            return self._new(
                "group",
                f"SELECT COUNT(*) AS {_qid(alias)} FROM (SELECT DISTINCT * FROM {self._src(table)})",
            )
        aggs = ", ".join(
            f"{_AGG[a['op']]}({'DISTINCT ' if a['op'] == 'count_distinct' else ''}"
            f"{self._col_sql(cols, a.get('column', '*')) if a.get('column', '*') != '*' else '*'}) AS {_qid(a['as'])}"
            for a in aggregations
        )
        parts = [p for p in (gb, extra, aggs) if p]
        sel = ", ".join(parts) if parts else "*"
        tail = f" GROUP BY {gb}" if gb else ""
        return self._new("group", f"SELECT {sel} FROM {self._src(table)}{tail}")

    def join_tables(self, left=None, right=None, on=None, join_type="inner",
                    return_columns=None, left_prefix=None, right_prefix=None,
                    tables=None, join_types=None, prefixes=None) -> dict:
        """N-way join folded left-to-right in ONE step (one result handle).

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
            cond = " AND ".join(
                f"L.{self._col_sql(cur_cols, e['left'])} = R.{self._col_sql(rc, e['right'])}" for e in edges
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
                return self._resolve_col(cur_cols, name)
            sel = ", ".join(f"{_qid(qual(c))} AS {_qid(c.split('.')[-1])}" for c in return_columns)
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

    def project(self, table, expressions) -> dict:
        """Realize a SELECT projection: SELECT <expressions> FROM (src). Table-producing.
        `expressions` are SQL column/expression strings, optionally `expr AS alias`."""
        cols = self._cols(table)

        def render(expr: str) -> str:
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
            return expr

        sel = ", ".join(render(expr) for expr in expressions) if expressions else "*"
        return self._new("project", f"SELECT {sel} FROM {self._src(table)}")

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

    def read_subtable(self, table, columns=None, limit=20):
        if columns:
            available = self._cols(table)
            cols = ", ".join(self._col_sql(available, col) for col in columns)
        else:
            cols = "*"
        return self.conn.execute(
            f"SELECT {cols} FROM {self._src(table)} LIMIT {int(limit)}"
        ).fetchall()

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
