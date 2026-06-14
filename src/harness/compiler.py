#!/usr/bin/env python3
"""SQL -> tool-call Plan compiler (implements tool_design/final_tool_design.md).

Parses a SQL string with sqlglot and walks the **logical** relational pipeline
    FROM/JOIN -> WHERE -> GROUP BY -> HAVING -> ORDER BY/LIMIT -> SELECT(project) -> DISTINCT
emitting one `Step` (abstract tool call) per stage and threading named intermediate tables.

Column rendering has two modes:
- **bare** (default / single-table / no schema): strip table qualifiers; columns are bare. Works
  whenever there is no column-name ambiguity.
- **qualified** (joins + a known schema): at FROM time each base table's columns are renamed to
  `<alias>__<col>`, and every column reference renders to `<alias>__<col>`. This resolves shared
  column names, self-joins, and ambiguity in multi-table queries. Pass `Compiler(schema)`.

Unsupported constructs raise `CompileError` (surfaced in coverage reports, never mis-compiled).
v1 scope-outs: subqueries in WHERE, correlated subqueries.
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
    def __init__(self, schema: dict[str, list[str]] | None = None) -> None:
        # schema: {lowercased table name -> [column names]}; enables qualified-column mode for joins
        self.schema = schema
        self._ids = itertools.count(1)
        self._qual: dict[str, str] | None = None  # current SELECT's alias -> table (or None = bare)

    def _id(self) -> str:
        return f"s{next(self._ids)}"

    # ---------- public ----------
    def compile(self, sql: str) -> list[Step]:
        try:
            ast = sqlglot.parse_one(sql, read="sqlite")
        except Exception as exc:
            raise CompileError(f"parse error: {exc}") from exc
        return self._node(ast)

    _SETOP = {E.Union: "union", E.Intersect: "intersect", E.Except: "except"}

    def _node(self, ast: E.Expression) -> list[Step]:
        if isinstance(ast, E.Select):
            return self._select(ast)
        for cls, op in self._SETOP.items():
            if isinstance(ast, cls):
                if op == "union" and ast.args.get("distinct") is False:
                    op = "union_all"
                left = self._node(ast.this)
                right = self._node(ast.expression)
                steps = left + right
                sid = self._id()
                steps.append(Step(sid, "set_op",
                                  {"left": left[-1].id, "right": right[-1].id, "op": op}))
                return steps
        raise CompileError(f"top-level {type(ast).__name__} unsupported (only SELECT / set-op)")

    # ---------- column rendering (bare vs qualified) ----------
    def _bare(self, node: E.Expression) -> str:
        if self._qual is not None:
            return self._qualify(node)
        n = node.copy()
        for col in n.find_all(E.Column):
            col.set("table", None)
        return n.sql(dialect="sqlite")

    def _qualify(self, node: E.Expression) -> str:
        """Render `node` with each column rewritten to `<alias>__<col>` (qualified mode)."""
        n = node.copy()
        cols = [n] if isinstance(n, E.Column) else list(n.find_all(E.Column))
        for c in cols:
            alias = c.table or self._find_alias(c.name)
            combined = f"{alias}__{c.name}"
            c.set("table", None)
            c.this.set("this", combined)
        return n.sql(dialect="sqlite")

    def _find_alias(self, name: str) -> str:
        nl = name.lower()
        for alias, table in self._qual.items():
            if any(c.lower() == nl for c in self.schema.get(table, [])):
                return alias
        return next(iter(self._qual))  # ambiguous/unknown -> first alias (best effort)

    def _aliases(self, sel: E.Select):
        """alias -> lowercased table name for FROM + JOIN base tables; None if any source is a
        subquery or an unknown table (then fall back to bare rendering)."""
        frm = sel.args.get("from_") or sel.args.get("from")
        if frm is None:
            return None
        nodes = [frm.this] + [j.this for j in (sel.args.get("joins") or [])]
        out: dict[str, str] = {}
        for nd in nodes:
            if not isinstance(nd, E.Table) or nd.name.lower() not in (self.schema or {}):
                return None
            out[nd.alias or nd.name] = nd.name.lower()
        return out

    def _literal(self, lit: E.Literal):
        if lit.is_string:
            return lit.this
        s = lit.this
        try:
            return int(s)
        except ValueError:
            return float(s)

    def _str_value(self, node: E.Expression):
        """Spider writes string values with double quotes -> quoted identifiers (Columns)."""
        if isinstance(node, E.Column) and node.this.quoted and not node.table:
            return node.name
        return None

    # ---------- FROM + JOIN ----------
    def _from(self, sel: E.Select, steps: list[Step]) -> str:
        if self._qual is not None:
            return self._from_qualified(sel, steps)
        frm = sel.args.get("from_") or sel.args.get("from")
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
            steps.append(Step(sid, "join_tables",
                              {"left": cur, "right": right, "on": self._join_on(on), "join_type": jt}))
            cur = sid
        return cur

    def _from_qualified(self, sel: E.Select, steps: list[Step]) -> str:
        # Qualified mode WITHOUT separate rename steps: `join_tables` prefixes each base table's
        # columns to `<alias>__<col>` internally (left_prefix / right_prefix). The first base table
        # is prefixed by the first join (left_prefix); each newly-joined right table by right_prefix;
        # the accumulated intermediate is already prefixed and passes through (left_prefix=None).
        frm = sel.args.get("from_") or sel.args.get("from")
        first = frm.this
        cur = self._qual[first.alias or first.name]   # base table name (resolved case-insensitively)
        left_prefix = first.alias or first.name        # the first base table still needs prefixing
        for join in sel.args.get("joins", []) or []:
            jn = join.this
            r_alias = jn.alias or jn.name
            on = join.args.get("on")
            if on is None:
                raise CompileError("only ON-condition joins supported")
            jt = "left" if (join.side or "").lower() == "left" else "inner"
            sid = self._id()
            steps.append(Step(sid, "join_tables", {
                "left": cur, "right": self._qual[r_alias],
                "on": self._join_on_internal(on, r_alias, left_prefix),
                "join_type": jt, "left_prefix": left_prefix, "right_prefix": r_alias,
            }))
            cur = sid
            left_prefix = None                          # accumulated intermediate already prefixed
        return cur

    def _join_on_internal(self, cond: E.Expression, right_alias: str, left_prefix):
        """ON keys in SOURCE terms for internalized prefixing: the right (newly-joined base) key is
        bare; the left key is bare while the left is still an un-prefixed base table (the first
        join, `left_prefix` set) and `<owner>__<col>` once the left is a prefixed intermediate.
        Routes each equality by which operand belongs to the newly-joined right table."""
        if isinstance(cond, E.And):
            return (self._join_on_internal(cond.this, right_alias, left_prefix)
                    + self._join_on_internal(cond.expression, right_alias, left_prefix))
        if isinstance(cond, E.EQ):
            a, b = cond.this, cond.expression
            a_alias = a.table or (self._find_alias(a.name) if isinstance(a, E.Column) else None)
            b_alias = b.table or (self._find_alias(b.name) if isinstance(b, E.Column) else None)
            if b_alias == right_alias:                  # a = left side, b = right (newly joined)
                l_node, l_alias, r_node = a, a_alias, b
            else:                                       # written right-table-first: swap
                l_node, l_alias, r_node = b, b_alias, a
            l_key = l_node.name if left_prefix is not None else f"{l_alias}__{l_node.name}"
            return [{"left": l_key, "right": r_node.name}]
        raise CompileError("join ON supports equality / AND of equalities only")

    def _table_ref(self, node: E.Expression, steps: list[Step]) -> str:
        if isinstance(node, E.Table):
            return node.name
        if isinstance(node, E.Subquery):
            sub = self._node(node.this)
            steps.extend(sub)
            return sub[-1].id
        raise CompileError(f"FROM/JOIN source {type(node).__name__} unsupported")

    def _join_on(self, cond: E.Expression) -> list[dict]:
        if isinstance(cond, E.And):
            return self._join_on(cond.this) + self._join_on(cond.expression)
        if isinstance(cond, E.EQ):
            return [{"left": self._bare(cond.this), "right": self._bare(cond.expression)}]
        raise CompileError("join ON supports equality / AND of equalities only")

    # ---------- WHERE / HAVING conditions (boolean tree) ----------
    def _condition(self, cond: E.Expression, steps: list[Step], colmap: dict[str, str] | None = None):
        colmap = colmap or {}
        if isinstance(cond, E.Paren):
            return self._condition(cond.this, steps, colmap)
        if isinstance(cond, E.And):
            return {"and": [self._condition(cond.this, steps, colmap),
                            self._condition(cond.expression, steps, colmap)]}
        if isinstance(cond, E.Or):
            return {"or": [self._condition(cond.this, steps, colmap),
                           self._condition(cond.expression, steps, colmap)]}
        if isinstance(cond, E.Not):
            return {"not": self._condition(cond.this, steps, colmap)}
        for cls, op in _CMP.items():
            if isinstance(cond, cls):
                col = self._resolve(cond.this, colmap)
                rhs = cond.expression
                if isinstance(rhs, E.Literal):
                    return {"column": col, "op": op, "value": self._literal(rhs)}
                sv = self._str_value(rhs)
                if sv is not None:
                    return {"column": col, "op": op, "value": sv}
                if isinstance(rhs, E.Column):
                    return {"column": col, "op": op, "column_value": self._bare(rhs)}
                ref = self._scalar_subquery(rhs, steps)
                if ref is not None:
                    return {"column": col, "op": op, "value_ref": ref}
                raise CompileError(f"predicate RHS {type(rhs).__name__} unsupported")
        if isinstance(cond, E.Like):
            pat = cond.expression
            val = pat.this if (isinstance(pat, E.Literal) and pat.is_string) else self._str_value(pat)
            if val is not None:
                return {"column": self._resolve(cond.this, colmap), "op": "like", "value": val}
            raise CompileError("LIKE pattern must be a string literal")
        if isinstance(cond, E.In):
            sub = cond.args.get("query")
            if sub is not None:
                return self._in_subquery(cond.this, sub, steps, colmap)
            out = []
            for e in cond.expressions:
                if isinstance(e, E.Literal):
                    out.append(self._literal(e))
                elif self._str_value(e) is not None:
                    out.append(self._str_value(e))
                else:
                    raise CompileError("IN list must be literals")
            return {"column": self._resolve(cond.this, colmap), "op": "in", "values": out}
        if isinstance(cond, E.Between):
            return {"column": self._resolve(cond.this, colmap), "op": "between",
                    "low": self._literal(cond.args["low"]), "high": self._literal(cond.args["high"])}
        if isinstance(cond, E.Is) and isinstance(cond.expression, E.Null):
            return {"column": self._resolve(cond.this, colmap), "op": "is_null"}
        raise CompileError(f"predicate {type(cond).__name__} unsupported")

    def _in_subquery(self, this_node: E.Expression, sub: E.Expression, steps: list[Step], colmap):
        """`col IN (subquery)` -> compile the subquery to a single-column table and test membership
        against it (`in_table`); a set-valued subquery stays a table, NOT memory. If the subquery is
        scalar (ends in `aggregate`), `IN` degenerates to equality, routed through memory like any
        scalar subquery. `NOT IN` is the parser's `Not(In(...))`, rendered as `NOT (col IN ...)`."""
        inner = sub.this if isinstance(sub, (E.Subquery, E.Paren)) else sub
        sub_steps = self._node(inner)   # SELECT or a set-op (UNION/INTERSECT/EXCEPT) -> a table
        steps.extend(sub_steps)
        last = sub_steps[-1]
        col = self._resolve(this_node, colmap)
        if last.tool == "aggregate":          # IN (scalar subquery) == equality to that scalar
            mem = self._id()
            # model cites the source step only; the harness grounds value/key/derivation (V2a).
            steps.append(Step(mem, "add_to_memory", {"type": "derived_value", "source": last.id}))
            return {"column": col, "op": "=", "value_ref": mem}
        return {"column": col, "op": "in", "in_table": last.id}

    def _scalar_subquery(self, node: E.Expression, steps: list[Step]):
        """Uncorrelated scalar subquery in a predicate -> compile it to an `aggregate` step, park the
        scalar with `add_to_memory` (the memory entry cites the aggregate, the predicate cites the
        memory entry -> an explicit provenance chain for the process reward), and return the memory
        key for the predicate's `value_ref`. Returns None if `node` is not a subquery, so the caller
        falls through to its normal 'unsupported RHS' error. Correlated / non-scalar subqueries
        either raise here or fail round-trip verification and are dropped (never mis-compiled)."""
        inner = node.this if isinstance(node, (E.Subquery, E.Paren)) else node
        if not isinstance(inner, E.Select):
            return None
        sub = self._node(inner)
        if not sub:
            return None
        steps.extend(sub)
        last = sub[-1]
        mem = self._id()
        # model cites the source step only; the harness extracts the scalar and builds the
        # derivation/key/content (V2a). Single-row subqueries are grounded as a 1x1 table.
        steps.append(Step(mem, "add_to_memory", {"type": "derived_value", "source": last.id}))
        return mem

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
        saved = self._qual
        self._qual = self._aliases(sel) if (sel.args.get("joins") and self.schema is not None) else None
        try:
            return self._select_body(sel)
        finally:
            self._qual = saved

    def _select_body(self, sel: E.Select) -> list[Step]:
        steps: list[Step] = []
        cur = self._from(sel, steps)

        where = sel.args.get("where")
        if where is not None:
            conds = self._condition(where.this, steps)   # may append subquery + add_to_memory steps
            sid = self._id()
            steps.append(Step(sid, "condition_filter", {"table": cur, "conditions": conds}))
            cur = sid

        keys, aggs, agg_map = self._select_items(sel)
        group = sel.args.get("group")
        order = sel.args.get("order")
        limit = sel.args.get("limit")

        if group is None and aggs and not keys:
            if len(aggs) == 1:
                a = aggs[0]
                sid = self._id()
                steps.append(Step(sid, "aggregate", {"table": cur, "column": a["column"], "op": a["op"]}))
                return steps
            sid = self._id()
            steps.append(Step(sid, "group_aggregate", {"table": cur, "group_by": [], "aggregations": aggs}))
            return steps
        if group is None and aggs and keys:
            # SQLite bare-column extension: non-aggregated columns selected alongside aggregates with
            # no GROUP BY -> a single row, the bare columns from an arbitrary row. Model it as a
            # whole-table group (group_by=[]) carrying the bare columns through as passthrough.
            sid = self._id()
            steps.append(Step(sid, "group_aggregate",
                              {"table": cur, "group_by": [], "aggregations": aggs,
                               "passthrough": [k for k, _ in keys]}))
            cur = sid
            proj = self._projection(sel, agg_map)
            if proj != ["*"]:
                sid = self._id()
                steps.append(Step(sid, "project", {"table": cur, "expressions": proj}))
                cur = sid
            return steps

        if group is not None:
            having = sel.args.get("having")
            if having is not None:
                self._extend_aggs(having.this, aggs, agg_map)
            if order is not None:
                for o in order.expressions:
                    self._extend_aggs(o.this, aggs, agg_map)
            group_by = [self._bare(g) for g in group.expressions]
            # SQLite allows selecting non-grouped, non-aggregated columns (arbitrary per group);
            # carry them through so the final projection can reference them.
            extras = [k for k, _ in keys if k not in group_by]
            g_args = {"table": cur, "group_by": group_by, "aggregations": aggs}
            if extras:
                g_args["passthrough"] = extras
            sid = self._id()
            steps.append(Step(sid, "group_aggregate", g_args))
            cur = sid
            if having is not None:
                conds = self._condition(having.this, steps, agg_map)
                sid = self._id()
                steps.append(Step(sid, "condition_filter", {"table": cur, "conditions": conds}))
                cur = sid

        if order is not None or limit is not None:
            ob = []
            if order is not None:
                for o in order.expressions:
                    col = self._resolve(o.this, agg_map)
                    ob.append(f"{col} DESC" if o.args.get("desc") else col)
            lim = self._literal(limit.expression) if limit is not None else None
            sid = self._id()
            steps.append(Step(sid, "extreme_value_select", {"table": cur, "order_by": ob, "top_k": lim}))
            cur = sid

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

        if not steps:               # bare `SELECT * FROM t`: materialize the source so there is a result
            sid = self._id()
            steps.append(Step(sid, "project", {"table": cur, "expressions": ["*"]}))

        return steps
