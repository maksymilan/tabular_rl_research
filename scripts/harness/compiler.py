#!/usr/bin/env python3
"""SQL -> tool-call Plan compiler (implements tool_design/final_tool_design.md).

Parses a SQL string with sqlglot and walks the **logical** relational pipeline
    FROM/JOIN -> WHERE -> GROUP BY -> HAVING -> ORDER BY/LIMIT -> SELECT(project) -> DISTINCT
emitting one `Step` (abstract tool call) per stage and threading named intermediate tables.

Design choices:
- Columns are rendered **bare** (table qualifiers stripped); each pipeline stage is a
  single-source view whose columns are bare. This also neutralizes most single-table aliases.
- Aggregates referenced by HAVING / ORDER BY but absent from SELECT are still materialized as
  group columns, then projected away by a final projection (so the result matches the gold SELECT).
- Unsupported constructs raise `CompileError` (surfaced in the coverage report, never silently
  mis-compiled). v1 scope-outs: subqueries in WHERE, correlated subqueries, UNION/INTERSECT,
  window functions, multi-table column-ambiguous joins.
"""
from __future__ import annotations

import itertools

import sqlglot
from sqlglot import expressions as E

from plan import Step


class CompileError(Exception):
    """Raised when the SQL uses a construct outside the compiler's current support."""


_CMP = {E.EQ: "=", E.NEQ: "!=", E.GT: ">", E.GTE: ">=", E.LT: "<", E.LTE: "<="}
_AGG = {E.Sum: "sum", E.Count: "count", E.Avg: "mean", E.Min: "min", E.Max: "max"}
_AGG_TYPES = tuple(_AGG.keys())


class Compiler:
    def __init__(self) -> None:
        self._ids = itertools.count(1)

    def _id(self) -> str:
        return f"s{next(self._ids)}"

    # ---------- public ----------
    def compile(self, sql: str) -> list[Step]:
        try:
            ast = sqlglot.parse_one(sql, read="sqlite")
        except Exception as exc:  # sqlglot parse error
            raise CompileError(f"parse error: {exc}") from exc
        if not isinstance(ast, E.Select):
            raise CompileError(f"top-level {type(ast).__name__} unsupported (only SELECT)")
        return self._select(ast)

    # ---------- rendering ----------
    def _bare(self, node: E.Expression) -> str:
        """Render an expression with table qualifiers removed (bare column names)."""
        n = node.copy()
        for col in n.find_all(E.Column):
            col.set("table", None)
        return n.sql(dialect="sqlite")

    def _literal(self, lit: E.Literal):
        if lit.is_string:
            return lit.this
        s = lit.this
        try:
            return int(s)
        except ValueError:
            return float(s)

    # ---------- FROM + JOIN ----------
    def _from(self, sel: E.Select, steps: list[Step]) -> str:
        frm = sel.args.get("from_") or sel.args.get("from")  # sqlglot>=30 uses "from_"
        if frm is None:
            raise CompileError("missing FROM")
        cur = self._table_ref(frm.this, steps)
        for join in sel.args.get("joins", []) or []:
            right = self._table_ref(join.this, steps)
            on = join.args.get("on")
            if on is None:
                raise CompileError("only ON-condition joins supported")
            jt = "left" if (join.side or "").lower() == "left" else "inner"
            sid = self._id()
            steps.append(Step(sid, "join_tables", {
                "left": cur, "right": right, "on": self._join_on(on), "join_type": jt,
            }))
            cur = sid
        return cur

    def _table_ref(self, node: E.Expression, steps: list[Step]) -> str:
        if isinstance(node, E.Table):
            return node.name
        if isinstance(node, E.Subquery):
            sub = self._select(node.this)
            steps.extend(sub)
            return sub[-1].id
        raise CompileError(f"FROM/JOIN source {type(node).__name__} unsupported")

    def _join_on(self, cond: E.Expression) -> list[dict]:
        if isinstance(cond, E.And):
            return self._join_on(cond.this) + self._join_on(cond.expression)
        if isinstance(cond, E.EQ):
            return [{"left": self._bare(cond.this), "right": self._bare(cond.expression)}]
        raise CompileError("join ON supports equality / AND of equalities only")

    # ---------- WHERE / HAVING predicates ----------
    def _predicates(self, cond: E.Expression, colmap: dict[str, str] | None = None) -> list[dict]:
        colmap = colmap or {}
        if isinstance(cond, E.And):
            return self._predicates(cond.this, colmap) + self._predicates(cond.expression, colmap)
        for cls, op in _CMP.items():
            if isinstance(cond, cls):
                col = self._resolve(cond.this, colmap)
                rhs = cond.expression
                if isinstance(rhs, E.Literal):
                    return [{"column": col, "op": op, "value": self._literal(rhs)}]
                if isinstance(rhs, E.Column):
                    return [{"column": col, "op": op, "column_value": self._bare(rhs)}]
                raise CompileError(f"predicate RHS {type(rhs).__name__} unsupported")
        if isinstance(cond, E.Like):
            pat = cond.expression
            if isinstance(pat, E.Literal) and pat.is_string:
                return [{"column": self._resolve(cond.this, colmap), "op": "contains",
                         "value": pat.this.strip("%")}]
        raise CompileError(f"predicate {type(cond).__name__} unsupported")

    def _resolve(self, node: E.Expression, colmap: dict[str, str]) -> str:
        key = self._bare(node)
        return colmap.get(key, key)

    # ---------- aggregates ----------
    def _agg_of(self, node: E.Expression):
        for cls, op in _AGG.items():
            if isinstance(node, cls):
                inner = node.this
                if isinstance(inner, E.Distinct):
                    op = "count_distinct" if op == "count" else op
                    inner = inner.expressions[0] if inner.expressions else None
                col = "*" if (inner is None or isinstance(inner, E.Star)) else self._bare(inner)
                return op, col
        return None

    def _select_items(self, sel: E.Select):
        """Return (keys, aggs, agg_map): keys=non-agg select exprs, aggs=specs, agg_map=expr->alias."""
        keys, aggs, agg_map = [], [], {}
        for proj in sel.expressions:
            inner = proj.this if isinstance(proj, E.Alias) else proj
            alias = proj.alias if isinstance(proj, E.Alias) else None
            af = self._agg_of(inner)
            if af:
                op, col = af
                name = alias or f"{op}_{len(aggs) + 1}"
                aggs.append({"op": op, "column": col, "as": name})
                agg_map[self._bare(inner)] = name
            else:
                keys.append((self._bare(inner), alias))
        return keys, aggs, agg_map

    def _extend_aggs(self, node: E.Expression, aggs: list, agg_map: dict) -> None:
        """Materialize aggregates referenced (e.g. in HAVING/ORDER) but not already in SELECT."""
        for agg_node in node.find_all(*_AGG_TYPES):
            key = self._bare(agg_node)
            if key not in agg_map:
                op, col = self._agg_of(agg_node)
                name = f"{op}_{len(aggs) + 1}"
                aggs.append({"op": op, "column": col, "as": name})
                agg_map[key] = name

    def _projection(self, sel: E.Select, agg_map: dict) -> list[str]:
        out = []
        for proj in sel.expressions:
            inner = proj.this if isinstance(proj, E.Alias) else proj
            alias = proj.alias if isinstance(proj, E.Alias) else None
            col = agg_map[self._bare(inner)] if self._agg_of(inner) else self._bare(inner)
            out.append(f"{col} AS {alias}" if alias else col)
        return out

    # ---------- the SELECT pipeline ----------
    def _select(self, sel: E.Select) -> list[Step]:
        steps: list[Step] = []
        cur = self._from(sel, steps)

        where = sel.args.get("where")
        if where is not None:
            sid = self._id()
            steps.append(Step(sid, "condition_filter",
                              {"table": cur, "conditions": self._predicates(where.this)}))
            cur = sid

        keys, aggs, agg_map = self._select_items(sel)
        group = sel.args.get("group")
        order = sel.args.get("order")
        limit = sel.args.get("limit")

        # SELECT agg(...) with no GROUP BY and no plain columns -> single scalar (terminal)
        if group is None and aggs and not keys:
            if len(aggs) != 1:
                raise CompileError("multiple scalar aggregates without GROUP BY unsupported in v1")
            a = aggs[0]
            sid = self._id()
            steps.append(Step(sid, "aggregate", {"table": cur, "column": a["column"], "op": a["op"]}))
            return steps
        if group is None and aggs and keys:
            raise CompileError("mixed aggregate + non-aggregate without GROUP BY unsupported")

        if group is not None:
            having = sel.args.get("having")
            if having is not None:
                self._extend_aggs(having.this, aggs, agg_map)
            if order is not None:
                for o in order.expressions:
                    self._extend_aggs(o.this, aggs, agg_map)
            group_by = [self._bare(g) for g in group.expressions]
            sid = self._id()
            steps.append(Step(sid, "group_aggregate",
                              {"table": cur, "group_by": group_by, "aggregations": aggs}))
            cur = sid
            if having is not None:
                sid = self._id()
                steps.append(Step(sid, "condition_filter",
                                  {"table": cur, "conditions": self._predicates(having.this, agg_map)}))
                cur = sid

        # ORDER BY [+ LIMIT] before final projection (may reference non-projected columns)
        if order is not None or limit is not None:
            ob = []
            if order is not None:
                for o in order.expressions:
                    col = self._resolve(o.this, agg_map)
                    ob.append(f"{col} DESC" if o.args.get("desc") else col)
            lim = self._literal(limit.expression) if limit is not None else None
            sid = self._id()
            steps.append(Step(sid, "order_limit", {"table": cur, "order_by": ob, "limit": lim}))
            cur = sid

        # final projection to SELECT columns (drops extra having/order aggregates)
        proj = self._projection(sel, agg_map)
        if proj != ["*"]:
            sid = self._id()
            steps.append(Step(sid, "project", {"table": cur, "expressions": proj}))
            cur = sid

        if sel.args.get("distinct") is not None:
            cols = [p.split(" AS ")[0] for p in proj] if proj != ["*"] else ["*"]
            sid = self._id()
            steps.append(Step(sid, "group_aggregate", {"table": cur, "group_by": cols, "aggregations": []}))
            cur = sid

        return steps
