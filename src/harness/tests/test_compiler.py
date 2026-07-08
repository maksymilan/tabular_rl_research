"""Unit tests for the SQL->Plan compiler (structure + error handling)."""
from common import T
from compiler import Compiler, CompileError


def run():
    t = T("compiler")

    # full pipeline order: WHERE -> GROUP -> HAVING -> ORDER/LIMIT
    p = Compiler().compile(
        "SELECT dept, AVG(salary) AS a FROM employees WHERE salary>1000 "
        "GROUP BY dept HAVING AVG(salary)>1200 ORDER BY a DESC LIMIT 2"
    )
    t.check("pipeline order",
            [s.tool for s in p] ==
            ["condition_filter", "group_aggregate", "condition_filter", "extreme_value_select", "project"],
            str([s.tool for s in p]))
    t.check("HAVING aggregate resolved to alias",
            p[2].args["conditions"]["column"] == "a", str(p[2].args))
    t.check("ORDER aggregate resolved to alias",
            p[3].args["order_by"] == ["a DESC"], str(p[3].args))

    # join emitted first (N-way form: tables list + per-join on-chain)
    pj = Compiler().compile("SELECT name FROM employees JOIN depts ON employees.dept=depts.dept")
    t.check("join emitted",
            pj[0].tool == "join_tables"
            and pj[0].args["tables"] == ["employees", "depts"]
            and pj[0].args["on"] == [[{"left": "dept", "right": "dept"}]],
            str(pj[0]))

    # qualified join: prefixing is internalized into join_tables (no separate rename steps)
    sch = {"employees": ["id", "name", "dept", "salary"], "depts": ["dept", "location"]}
    pq2 = Compiler(sch).compile("SELECT e.name, d.location FROM employees e JOIN depts d ON e.dept=d.dept")
    t.check("join internalizes prefixing (no rename step before join)",
            pq2[0].tool == "join_tables" and pq2[0].args.get("prefixes") == ["e", "d"]
            and pq2[0].args.get("tables") == ["employees", "depts"]
            and [s.tool for s in pq2].count("join_tables") == 1
            and "project" not in [s.tool for s in pq2[:1]],
            str([s.tool for s in pq2]))

    # a consecutive multi-table chain compresses into ONE N-way join step
    sch3 = {"a": ["id", "bid"], "b": ["id", "cid"], "c": ["id", "v"]}
    pn = Compiler(sch3).compile("SELECT c.v FROM a JOIN b ON a.bid=b.id JOIN c ON b.cid=c.id")
    t.check("N-way join chain -> one step",
            [s.tool for s in pn].count("join_tables") == 1
            and pn[0].args["tables"] == ["a", "b", "c"]
            and len(pn[0].args["on"]) == 2 and pn[0].args.get("prefixes") == ["a", "b", "c"],
            str(pn[0]))

    # scalar aggregate (no GROUP BY) -> single-row group_aggregate
    ps = Compiler().compile("SELECT COUNT(*) FROM employees")
    t.check("scalar aggregate as group_aggregate",
            ps[-1].tool == "group_aggregate"
            and ps[-1].args["group_by"] == []
            and ps[-1].args["aggregations"][0]["op"] == "count", str(ps))

    # DISTINCT -> group_aggregate with no aggregations
    pd = Compiler().compile("SELECT DISTINCT dept FROM employees")
    t.check("distinct as group-no-agg",
            pd[-1].tool == "group_aggregate" and pd[-1].args["aggregations"] == [], str(pd))

    # bare-column rendering strips table qualifiers
    pq = Compiler().compile("SELECT e.name FROM employees AS e WHERE e.salary > 1000")
    t.check("strips table qualifier",
            pq[0].args["conditions"]["column"] == "salary", str(pq[0].args))

    # boolean tree: OR
    po = Compiler().compile("SELECT name FROM employees WHERE dept='eng' OR dept='hr'")
    t.check("OR -> boolean tree", "or" in po[0].args["conditions"], str(po[0].args))

    # set operation -> set_op terminal
    pi = Compiler().compile("SELECT dept FROM employees INTERSECT SELECT dept FROM depts")
    t.check("set op emitted", pi[-1].tool == "set_op" and pi[-1].args["op"] == "intersect", str(pi[-1]))

    # multiple scalar aggregates -> single-row group_aggregate
    pm = Compiler().compile("SELECT MAX(salary), MIN(salary) FROM employees")
    t.check("multi scalar agg", pm[-1].tool == "group_aggregate" and pm[-1].args["group_by"] == [], str(pm[-1]))

    # scalar subquery -> scalar-shaped group_aggregate + filter referencing it directly via value_ref
    psub = Compiler().compile("SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)")
    tools = [s.tool for s in psub]
    agg = next(s for s in psub if s.tool == "group_aggregate" and s.args.get("group_by") == [])
    filt = next(s for s in psub if s.tool == "condition_filter")
    t.check("scalar subquery -> group_aggregate + value_ref(step)",
            tools[:2] == ["group_aggregate", "condition_filter"]
            and filt.args["conditions"].get("value_ref") == agg.id, str(tools))

    # IN (subquery) -> compile the subquery to a table, test membership via in_table
    pin = Compiler().compile("SELECT a FROM t WHERE a IN (SELECT b FROM s)")
    filt = next(s for s in pin if s.tool == "condition_filter")
    t.check("IN (subquery) -> in_table membership",
            filt.args["conditions"].get("op") == "in" and "in_table" in filt.args["conditions"], str(pin))

    # unsupported constructs raise CompileError (honest coverage gaps)
    for bad, label in [
        ("INSERT INTO t VALUES (1)", "non-SELECT"),
        ("SELECT a FROM t JOIN s ON t.x > s.y", "non-equality join ON"),
    ]:
        try:
            Compiler().compile(bad)
            t.check(f"reject {label}", False, "no CompileError raised")
        except CompileError:
            t.check(f"reject {label}", True)

    return t.result()
