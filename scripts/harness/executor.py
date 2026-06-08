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

import sqlite3
from typing import Any

_AGG = {
    "sum": "SUM", "count": "COUNT", "count_distinct": "COUNT", "mean": "AVG",
    "avg": "AVG", "min": "MIN", "max": "MAX", "total": "TOTAL",
}


def _lit(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


class Harness:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
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
        raise KeyError(f"unknown table: {table}")

    def _src(self, table: str) -> str:
        return f"({self._sql(table)})"

    def _cols(self, table: str) -> list[str]:
        cur = self.conn.execute(f"SELECT * FROM {self._src(table)} LIMIT 0")
        return [d[0] for d in cur.description]

    def _new(self, kind: str, sql: str) -> dict:
        self._n += 1
        name = f"{kind}_{self._n:03d}"
        self.views[name] = sql
        n = self.conn.execute(f"SELECT COUNT(*) FROM ({sql})").fetchone()[0]
        return {"table_name": name, "kind": kind, "row_count": n, "columns": self._cols(name)}

    def rows(self, table: str) -> list[tuple]:
        return self.conn.execute(self._sql(table)).fetchall()

    # ---- table-producing tools ----
    def _render_leaf(self, c: dict) -> str:
        col, op = c["column"], c.get("op", "=")
        if op == "contains":
            return f"{col} LIKE '%' || {_lit(c['value'])} || '%'"
        if op == "like":
            return f"{col} LIKE {_lit(c['value'])}"
        if op == "in":
            return f"{col} IN ({', '.join(_lit(v) for v in c['values'])})"
        if op == "between":
            return f"{col} BETWEEN {_lit(c['low'])} AND {_lit(c['high'])}"
        if op == "is_null":
            return f"{col} IS NULL"
        if "column_value" in c:  # column-vs-column predicate
            return f"{col} {op} {c['column_value']}"
        return f"{col} {op} {_lit(c['value'])}"

    def _render_cond(self, cond) -> str:
        """Render a boolean condition (tree dict with and/or/not, or a list = implicit AND)."""
        if isinstance(cond, list):
            cond = {"and": cond}
        if "and" in cond:
            return "(" + " AND ".join(self._render_cond(x) for x in cond["and"]) + ")"
        if "or" in cond:
            return "(" + " OR ".join(self._render_cond(x) for x in cond["or"]) + ")"
        if "not" in cond:
            return "NOT (" + self._render_cond(cond["not"]) + ")"
        return self._render_leaf(cond)

    def condition_filter(self, table: str, conditions) -> dict:
        where = self._render_cond(conditions) if conditions else "1=1"
        return self._new("filter", f"SELECT * FROM {self._src(table)} WHERE {where}")

    def derive_column(self, table: str, new_column: str, expression: str) -> dict:
        return self._new(
            "derive", f"SELECT *, ({expression}) AS {new_column} FROM {self._src(table)}"
        )

    def group_aggregate(self, table: str, group_by: list[str], aggregations: list[dict],
                        passthrough: list[str] | None = None) -> dict:
        # passthrough: columns selected but not grouped/aggregated (SQLite's lenient bare-column
        # extension; they are functionally dependent on the group key in practice).
        gb = ", ".join(group_by)
        extra = ", ".join(passthrough or [])
        aggs = ", ".join(
            f"{_AGG[a['op']]}({'DISTINCT ' if a['op'] == 'count_distinct' else ''}"
            f"{a.get('column', '*')}) AS {a['as']}"
            for a in aggregations
        )
        parts = [p for p in (gb, extra, aggs) if p]
        sel = ", ".join(parts) if parts else "*"
        tail = f" GROUP BY {gb}" if gb else ""
        return self._new("group", f"SELECT {sel} FROM {self._src(table)}{tail}")

    def join_tables(self, left, right, on, join_type="inner", return_columns=None,
                    left_prefix=None, right_prefix=None) -> dict:
        # `left_prefix`/`right_prefix`: when set, rename that side's columns to `<prefix>__<col>`
        # so a multi-table join disambiguates shared / self-join column names INTERNALLY — the
        # compiler no longer emits separate rename steps for join disambiguation. A side with
        # prefix=None passes through unchanged: bare single-table mode, or an already-prefixed
        # accumulated intermediate (left side of every join after the first).
        lc = self._cols(left)
        rc = self._cols(right)
        jt = {"inner": "JOIN", "left": "LEFT JOIN", "cross": "CROSS JOIN"}.get(join_type, "JOIN")

        def qual(name: str) -> str:
            b = name.split(".")[-1]
            return f"L.{b}" if b in lc else f"R.{b}"

        cond = " AND ".join(
            f"L.{o['left'].split('.')[-1]} = R.{o['right'].split('.')[-1]}" for o in (on or [])
        )
        oncl = f" ON {cond}" if on and join_type != "cross" else ""
        if return_columns:
            sel = ", ".join(f"{qual(c)} AS {c.split('.')[-1]}" for c in return_columns)
        elif left_prefix is not None or right_prefix is not None:
            # qualified mode: prefix each base side's columns; pass an already-prefixed side through.
            left_sel = [f"L.{c} AS {left_prefix}__{c}" for c in lc] if left_prefix else [f"L.{c}" for c in lc]
            right_sel = [f"R.{c} AS {right_prefix}__{c}" for c in rc] if right_prefix else [f"R.{c}" for c in rc]
            sel = ", ".join(left_sel + right_sel)
        else:
            # No projection / no prefixing: emit all columns but DEDUPE shared names (mostly the
            # join key) so downstream bare references are unambiguous. Case-insensitive (SQL ids),
            # else SQLite auto-renames collisions to `col:1` (invalid). Shared columns are equal
            # across the join, so keeping the left side is value-correct.
            seen = {c.lower() for c in lc}
            sel = ", ".join([f"L.{c}" for c in lc] +
                            [f"R.{c}" for c in rc if c.lower() not in seen])
        return self._new(
            "join", f"SELECT {sel} FROM {self._src(left)} AS L {jt} {self._src(right)} AS R{oncl}"
        )

    def set_op(self, left: str, right: str, op: str) -> dict:
        m = {"union": "UNION", "union_all": "UNION ALL",
             "intersect": "INTERSECT", "except": "EXCEPT"}
        return self._new("setop", f"{self._sql(left)} {m[op]} {self._sql(right)}")

    def window(self, table, partition_by, order_by, fn, as_) -> dict:
        part = f"PARTITION BY {', '.join(partition_by)} " if partition_by else ""
        order = f"ORDER BY {', '.join(order_by)}" if order_by else ""
        return self._new(
            "window", f"SELECT *, {fn}() OVER ({part}{order}) AS {as_} FROM {self._src(table)}"
        )

    # ---- reading / scalar tools ----
    def aggregate(self, table: str, column: str, op: str):
        d = "DISTINCT " if op == "count_distinct" else ""
        return self.conn.execute(
            f"SELECT {_AGG[op]}({d}{column}) FROM {self._src(table)}"
        ).fetchone()[0]

    def extreme_value_select(self, table, order_by, top_k=None, return_columns=None) -> dict:
        """Table-producing ORDER BY [... LIMIT k]: keep the extreme rows under an ordering.
        Merged from the old `order_limit` — one tool now covers a plain ORDER BY/LIMIT, a top-k
        pick, and multi-column ordering. `order_by`: list of 'col' or 'col DESC'.
        `return_columns`: optional projection. The result is small and inlined by `preview`."""
        order = f" ORDER BY {', '.join(order_by)}" if order_by else ""
        lim = f" LIMIT {int(top_k)}" if top_k is not None else ""
        sel = ", ".join(return_columns) if return_columns else "*"
        return self._new("top", f"SELECT {sel} FROM {self._src(table)}{order}{lim}")

    def project(self, table, expressions) -> dict:
        """Realize a SELECT projection: SELECT <expressions> FROM (src). Table-producing.
        `expressions` are SQL column/expression strings, optionally `expr AS alias`."""
        sel = ", ".join(expressions) if expressions else "*"
        return self._new("project", f"SELECT {sel} FROM {self._src(table)}")

    def preview(self, table: str, cell_limit: int = 100) -> dict:
        """Model-perceivable view of a table after a table-producing tool: inline ALL rows when the
        table is small (rows*cols <= cell_limit) else a truncated head flagged with `is_truncated`,
        so the model never silently reasons over data it cannot see. Used by the emitter (and RL
        rollouts) to CLOSE THE LOOP: the model perceives intermediate content, not just a handle."""
        cols = self._cols(table)
        total = self.conn.execute(f"SELECT COUNT(*) FROM {self._src(table)}").fetchone()[0]
        max_rows = max(1, cell_limit // max(1, len(cols)))
        rows = self.conn.execute(f"SELECT * FROM {self._src(table)} LIMIT {max_rows}").fetchall()
        out = {"columns": cols, "row_count": total, "rows": [list(r) for r in rows]}
        if total > len(rows):
            out["is_truncated"] = True
            out["note"] = f"showing first {len(rows)} of {total} rows (exceeds {cell_limit}-cell preview budget)"
        return out

    def read_subtable(self, table, columns=None, limit=10):
        cols = ", ".join(columns) if columns else "*"
        return self.conn.execute(
            f"SELECT {cols} FROM {self._src(table)} LIMIT {int(limit)}"
        ).fetchall()

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
