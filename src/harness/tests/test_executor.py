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

    not_in = h.condition_filter("employees", {"column": "dept", "op": "not_in", "values": ["eng", "hr"]})
    t.check("condition_filter not_in values", norm(h.rows(not_in["table_name"])) ==
            norm(h.gold("SELECT * FROM employees WHERE dept NOT IN ('eng','hr')")))

    not_like = h.condition_filter("employees", {"column": "name", "op": "not_like", "value": "*A*"})
    t.check("condition_filter not_like wildcard", norm(h.rows(not_like["table_name"])) ==
            norm(h.gold("SELECT * FROM employees WHERE name NOT LIKE '%A%'")))

    shorthand = h.condition_filter(
        "employees",
        {"or": [{"contains": {"column": "dept", "value": "eng"}},
                {"like": {"column": "dept", "value": "sales"}}]},
    )
    t.check("condition_filter shorthand predicate tree",
            norm(h.rows(shorthand["table_name"])) ==
            norm(h.gold("SELECT * FROM employees WHERE dept LIKE '%eng%' OR dept LIKE 'sales'")))

    returned = h.condition_filter(
        "employees",
        {"column": "salary", "op": ">", "value": 1000},
        return_columns=["name", "salary"],
    )
    t.check("condition_filter accepts return_columns",
            returned["columns"] == ["name", "salary"] and
            norm(h.rows(returned["table_name"])) ==
            norm(h.gold("SELECT name,salary FROM employees WHERE salary > 1000")),
            str(returned))

    g = h.group_aggregate(f["table_name"], ["dept"], [{"op": "mean", "column": "salary", "as": "a"}])
    t.check("group_aggregate", norm(h.rows(g["table_name"])) ==
            norm(h.gold("SELECT dept,AVG(salary) FROM employees WHERE salary>1000 GROUP BY dept")))

    j = h.join_tables("employees", "depts", [{"left": "dept", "right": "dept"}], "inner", ["name", "location"])
    t.check("join_tables", norm(h.rows(j["table_name"])) ==
            norm(h.gold("SELECT e.name,d.location FROM employees e JOIN depts d ON e.dept=d.dept")))

    j_flat = h.join_tables(
        tables=["employees", "depts"],
        on=[{"left": "dept", "right": "dept"}],
        join_types="inner",
        prefixes=["e", "d"],
        return_columns=["e__name", "d__location"],
    )
    t.check("join_tables accepts n-way 2-table flat on", norm(h.rows(j_flat["table_name"])) ==
            norm(h.gold("SELECT e.name,d.location FROM employees e JOIN depts d ON e.dept=d.dept")))

    j_left = h.join_tables(
        tables=["employees", "depts"],
        on=[{"left": "dept", "right": "dept"}],
        join_types="left",
        prefixes=["e", "d"],
    )
    t.check("join_tables string join_types normalized",
            j_left["row_count"] == h.gold("SELECT COUNT(*) FROM employees LEFT JOIN depts ON employees.dept=depts.dept")[0][0],
            str(j_left))

    unique_pref = h.join_tables(
        tables=["employees", "depts"],
        on=[{"left": "dept", "right": "dept"}],
        join_types="inner",
        prefixes=["uemp", "udep"],
    )
    t_alias = h.project("uemp", ["uemp__name"])
    t.check("unique prefix alias resolves to produced handle",
            norm(h.rows(t_alias["table_name"])) == norm(h.gold("SELECT name FROM employees JOIN depts USING(dept)")),
            str(unique_pref))

    d = h.derive_column("sales", "total", "price*qty")
    t.check("derive_column + aggregate(sum expr)",
            abs(h.aggregate(d["table_name"], "total", "sum") -
                h.gold("SELECT SUM(price*qty) FROM sales")[0][0]) < 1e-9)

    t.check("aggregate count_distinct",
            h.aggregate("employees", "dept", "count_distinct") ==
            h.gold("SELECT COUNT(DISTINCT dept) FROM employees")[0][0])

    distinct = h.group_aggregate("employees", [], [{"op": "distinct", "column": "dept", "as": "dept"}])
    t.check("group_aggregate distinct compatibility",
            norm(h.rows(distinct["table_name"])) == norm(h.gold("SELECT DISTINCT dept FROM employees")),
            str(distinct))

    e = h.extreme_value_select("employees", ["salary DESC"], 2, ["name"])
    t.check("extreme_value_select (projection)",
            h.rows(e["table_name"]) == h.gold("SELECT name FROM employees ORDER BY salary DESC LIMIT 2"))

    p = h.project("employees", ["name", "salary"])
    t.check("project", norm(h.rows(p["table_name"])) == norm(h.gold("SELECT name,salary FROM employees")))

    casted = h.project("employees", ["age::integer AS age_int"])
    t.check("project accepts postgres cast syntax",
            h.rows(casted["table_name"]) == h.gold("SELECT CAST(age AS INTEGER) AS age_int FROM employees"),
            str(casted))

    pref = h.join_tables(
        tables=["employees", "depts"],
        on=[{"left": "dept", "right": "dept"}],
        join_types="inner",
        prefixes=["emp", "dep"],
    )
    suffix_filter = h.condition_filter(pref["table_name"], {"column": "location", "op": "=", "value": "SF"})
    t.check("condition_filter resolves unique prefixed suffix",
            suffix_filter["row_count"] == h.gold("SELECT COUNT(*) FROM employees JOIN depts USING(dept) WHERE location='SF'")[0][0],
            str(suffix_filter))

    # merged: extreme_value_select with no projection subsumes the old order_limit
    o = h.extreme_value_select("employees", ["salary DESC"], 2)
    t.check("extreme_value_select (no projection)",
            h.rows(o["table_name"]) == h.gold("SELECT * FROM employees ORDER BY salary DESC LIMIT 2"))

    pv = h.preview(f["table_name"])
    t.check("preview inlines small table",
            pv["row_count"] == 4 and len(pv["rows"]) == 4 and "is_truncated" not in pv, str(pv))

    s = h.set_op("employees", "employees", "union")
    t.check("set_op union (dedup)", norm(h.rows(s["table_name"])) ==
            norm(h.gold("SELECT * FROM employees UNION SELECT * FROM employees")))

    names = h.project("employees", ["name"])
    high_salary = h.condition_filter("employees", {"column": "salary", "op": ">", "value": 1000})
    except_aligned = h.set_op(names["table_name"], high_salary["table_name"], "except")
    t.check("set_op projects right side to left columns when alignable",
            norm(h.rows(except_aligned["table_name"])) ==
            norm(h.gold("SELECT name FROM employees EXCEPT SELECT name FROM employees WHERE salary > 1000")))

    ids_depts = h.project("employees", ["id", "dept"])
    id_members = h.condition_filter("employees", {"column": "id", "op": "in", "in_table": ids_depts["table_name"]})
    t.check("in_table multi-column resolves target column",
            id_members["row_count"] == h.gold("SELECT COUNT(*) FROM employees WHERE id IN (SELECT id FROM employees)")[0][0],
            str(id_members))

    no_op = h.condition_filter("employees", {"column": "*", "op": "=", "value": None})
    t.check("condition_filter star-null compatibility no-op", no_op["row_count"] == 6, str(no_op))

    dotted_except = h.set_op("employees.id", names["table_name"], "except")
    t.check("set_op accepts dotted table column side",
            norm(h.rows(dotted_except["table_name"])) ==
            norm(h.gold("SELECT id FROM employees EXCEPT SELECT name FROM employees")),
            str(dotted_except))

    weird = h.conn.execute('CREATE TABLE ratings(id INT, "18_49_Rating_Share" REAL)').rowcount
    h.conn.executemany('INSERT INTO ratings VALUES (?,?)', [(1, 3.5), (2, 4.1)])
    h.register_sources()
    joined_weird = h.join_tables("ratings", "ratings", [{"left": "id", "right": "id"}])
    t.check("join_tables quotes nonstandard identifiers",
            joined_weird["row_count"] == 2 and "18_49_Rating_Share" in joined_weird["columns"],
            str((weird, joined_weird)))

    return t.result()
