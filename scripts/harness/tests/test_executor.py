"""Unit tests for the harness executor (each tool vs equivalent gold SQL)."""
from common import T, employees_db, norm


def run():
    t = T("executor")
    h = employees_db()

    f = h.condition_filter("employees", [{"column": "salary", "op": ">", "value": 1000}])
    t.check("condition_filter count", f["row_count"] == 4, str(f))  # 1200,1500,1100,2000

    fc = h.condition_filter("employees", [{"column": "name", "op": "contains", "value": "A"}])
    t.check("condition_filter contains", norm(h.rows(fc["table_name"])) ==
            norm(h.gold("SELECT * FROM employees WHERE name LIKE '%A%'")))

    g = h.group_aggregate(f["table_name"], ["dept"], [{"op": "mean", "column": "salary", "as": "a"}])
    t.check("group_aggregate", norm(h.rows(g["table_name"])) ==
            norm(h.gold("SELECT dept,AVG(salary) FROM employees WHERE salary>1000 GROUP BY dept")))

    j = h.join_tables("employees", "depts", [{"left": "dept", "right": "dept"}], "inner", ["name", "location"])
    t.check("join_tables", norm(h.rows(j["table_name"])) ==
            norm(h.gold("SELECT e.name,d.location FROM employees e JOIN depts d ON e.dept=d.dept")))

    d = h.derive_column("sales", "total", "price*qty")
    t.check("derive_column + aggregate(sum expr)",
            abs(h.aggregate(d["table_name"], "total", "sum") -
                h.gold("SELECT SUM(price*qty) FROM sales")[0][0]) < 1e-9)

    t.check("aggregate count_distinct",
            h.aggregate("employees", "dept", "count_distinct") ==
            h.gold("SELECT COUNT(DISTINCT dept) FROM employees")[0][0])

    t.check("extreme_value_select",
            h.extreme_value_select("employees", "salary", "max", 2, ["name"]) ==
            h.gold("SELECT name FROM employees ORDER BY salary DESC LIMIT 2"))

    p = h.project("employees", ["name", "salary"])
    t.check("project", norm(h.rows(p["table_name"])) == norm(h.gold("SELECT name,salary FROM employees")))

    o = h.order_limit("employees", ["salary DESC"], 2)
    t.check("order_limit", h.rows(o["table_name"]) == h.gold("SELECT * FROM employees ORDER BY salary DESC LIMIT 2"))

    s = h.set_op("employees", "employees", "union")
    t.check("set_op union (dedup)", norm(h.rows(s["table_name"])) ==
            norm(h.gold("SELECT * FROM employees UNION SELECT * FROM employees")))

    return t.result()
