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
            ["condition_filter", "group_aggregate", "condition_filter", "order_limit", "project"],
            str([s.tool for s in p]))
    t.check("HAVING aggregate resolved to alias",
            p[2].args["conditions"]["column"] == "a", str(p[2].args))
    t.check("ORDER aggregate resolved to alias",
            p[3].args["order_by"] == ["a DESC"], str(p[3].args))

    # join emitted first
    pj = Compiler().compile("SELECT name FROM employees JOIN depts ON employees.dept=depts.dept")
    t.check("join emitted", pj[0].tool == "join_tables" and pj[0].args["on"] == [{"left": "dept", "right": "dept"}],
            str(pj[0]))

    # scalar aggregate (no GROUP BY) -> terminal aggregate
    ps = Compiler().compile("SELECT COUNT(*) FROM employees")
    t.check("scalar aggregate terminal", ps[-1].tool == "aggregate" and ps[-1].args["op"] == "count", str(ps))

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

    # unsupported constructs raise CompileError (honest coverage gaps)
    for bad, label in [
        ("INSERT INTO t VALUES (1)", "non-SELECT"),
        ("SELECT a FROM t WHERE a > (SELECT MAX(b) FROM s)", "scalar subquery in WHERE"),
    ]:
        try:
            Compiler().compile(bad)
            t.check(f"reject {label}", False, "no CompileError raised")
        except CompileError:
            t.check(f"reject {label}", True)

    return t.result()
