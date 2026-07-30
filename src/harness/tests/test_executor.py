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

    is_null = h.condition_filter("employees", {"column": "age", "op": "is_null"})
    is_not_null = h.condition_filter("employees", {"column": "age", "op": "is_not_null"})
    t.check("condition_filter normalizes null operators",
            is_null["row_count"] == 0 and is_not_null["row_count"] == 6,
            f"is_null={is_null} is_not_null={is_not_null}")

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

    conditional = h.group_aggregate(
        "employees",
        [],
        [
            {
                "op": "count",
                "column": "*",
                "as": "eng_count",
                "where": {"column": "dept", "op": "=", "value": "eng"},
            },
            {
                "op": "count_distinct",
                "column": "dept",
                "as": "high_salary_depts",
                "where": {"column": "salary", "op": ">", "value": 1000},
            },
            {
                "op": "mean",
                "column": "salary",
                "as": "sales_avg",
                "where": {"column": "dept", "op": "=", "value": "sales"},
            },
        ],
    )
    t.check(
        "group_aggregate supports several conditional metrics on one input grain",
        h.rows(conditional["table_name"]) == h.gold(
            "SELECT "
            "COUNT(CASE WHEN dept='eng' THEN 1 END), "
            "COUNT(DISTINCT CASE WHEN salary>1000 THEN dept END), "
            "AVG(CASE WHEN dept='sales' THEN salary END) "
            "FROM employees"
        ),
        str(conditional),
    )

    wide_department_counts = h.group_aggregate(
        "employees",
        ["dept"],
        [{
            "op": "count_distinct",
            "column": "name",
            "as": "employee_count",
            "where": {"column": "salary", "op": ">", "value": 1000},
        }],
        output_layout="columns",
        category_values=["eng", "sales", "hr"],
        output_columns=["engineering", "sales", "human_resources"],
    )
    t.check(
        "group_aggregate columns layout emits ordered one-row category metrics",
        wide_department_counts["columns"] == ["engineering", "sales", "human_resources"]
        and h.rows(wide_department_counts["table_name"]) == [(2, 2, 0)],
        str(wide_department_counts),
    )

    component = h.join_tables(
        base="employees",
        joins=[
            {"table": "depts", "on": [{"left": "employees.dept", "right": "dept"}]},
            {"table": "sales", "on": [{"left": "depts.dept", "right": "dept"}]},
        ],
    )
    t.check(
        "version5 join component uses flat stable namespaces",
        component["columns"] == [
            "employees.id", "employees.name", "employees.dept", "employees.salary", "employees.age",
            "depts.dept", "depts.location", "depts.budget",
            "sales.item", "sales.price", "sales.qty", "sales.dept",
        ] and norm(h.rows(component["table_name"])) == norm(h.gold(
            "SELECT e.*,d.*,s.* FROM employees e "
            "JOIN depts d ON e.dept=d.dept JOIN sales s ON d.dept=s.dept"
        )),
        str(component),
    )
    symmetric = h.join(
        left="employees",
        right="depts",
        on=[{"left": "employees.dept", "right": "depts.dept"}],
    )
    t.check(
        "symmetric join resolves qualified columns independently",
        symmetric["columns"] == [
            "employees.id", "employees.name", "employees.dept",
            "employees.salary", "employees.age",
            "depts.dept", "depts.location", "depts.budget",
        ]
        and norm(h.rows(symmetric["table_name"])) == norm(h.gold(
            "SELECT e.*,d.* FROM employees e JOIN depts d ON e.dept=d.dept"
        )),
        str(symmetric),
    )
    chained = h.join(
        left=symmetric["table_name"],
        right="sales",
        on=[{"left": "depts.dept", "right": "sales.dept"}],
    )
    t.check(
        "symmetric join chains through a derived left input without handle prefixes",
        "employees.id" in chained["columns"]
        and "depts.location" in chained["columns"]
        and "sales.item" in chained["columns"]
        and not any(column.startswith(f"{symmetric['table_name']}.") for column in chained["columns"]),
        str(chained),
    )
    symmetric_self = h.join(
        left="employees",
        right="employees",
        left_alias="employee",
        right_alias="manager",
        on=[{"left": "employee.dept", "right": "manager.dept"}],
    )
    t.check(
        "symmetric self join uses explicit aliases",
        "employee.id" in symmetric_self["columns"]
        and "manager.id" in symmetric_self["columns"],
        str(symmetric_self),
    )
    dotted_project = h.project(component["table_name"], ["employees.name", "depts.location"])
    t.check(
        "project resolves exact dotted logical columns",
        dotted_project["columns"] == ["employees.name", "depts.location"] and
        norm(h.rows(dotted_project["table_name"])) == norm(h.gold(
            "SELECT e.name,d.location FROM employees e "
            "JOIN depts d ON e.dept=d.dept JOIN sales s ON d.dept=s.dept"
        )),
        str(dotted_project),
    )
    dotted_expression = h.project(
        component["table_name"],
        [
            "employees.name",
            "sales.price * sales.qty AS revenue",
            "employees.name || ':' || sales.item AS label",
        ],
    )
    t.check(
        "project quotes dotted logical columns inside scalar expressions",
        dotted_expression["columns"] == ["employees.name", "revenue", "label"] and
        norm(h.rows(dotted_expression["table_name"])) == norm(h.gold(
            "SELECT e.name,s.price*s.qty,e.name||':'||s.item FROM employees e "
            "JOIN depts d ON e.dept=d.dept JOIN sales s ON d.dept=s.dept"
        )),
        str(dotted_expression),
    )
    unique_bare_aggregate = h.group_aggregate(
        component["table_name"],
        [],
        [{"op": "sum", "column": "budget", "as": "total_budget"}],
    )
    t.check(
        "downstream tools resolve a unique bare suffix from dotted logical columns",
        h.rows(unique_bare_aggregate["table_name"]) == h.gold(
            "SELECT SUM(d.budget) FROM employees e "
            "JOIN depts d ON e.dept=d.dept JOIN sales s ON d.dept=s.dept"
        ),
        str(unique_bare_aggregate),
    )
    dotted_filter = h.condition_filter(
        component["table_name"],
        {"column": "depts.location", "op": "=", "value": "SF"},
        return_columns=["employees.name", "sales.item"],
    )
    t.check(
        "filter resolves exact dotted logical columns",
        dotted_filter["columns"] == ["employees.name", "sales.item"] and
        norm(h.rows(dotted_filter["table_name"])) == norm(h.gold(
            "SELECT e.name,s.item FROM employees e "
            "JOIN depts d ON e.dept=d.dept JOIN sales s ON d.dept=s.dept "
            "WHERE d.location='SF'"
        )),
        str(dotted_filter),
    )
    first_component = h.join_tables(
        base="employees",
        joins=[{"table": "depts", "on": [{"left": "employees.dept", "right": "dept"}]}],
    )
    filtered_component = h.condition_filter(
        first_component["table_name"],
        {"column": "depts.location", "op": "=", "value": "SF"},
    )
    continued_component = h.join_tables(
        base=filtered_component["table_name"],
        joins=[{"table": "sales", "on": [{"left": "depts.dept", "right": "dept"}]}],
    )
    t.check(
        "continued join preserves prior namespaces instead of nesting the base handle",
        "employees.id" in continued_component["columns"] and
        "depts.location" in continued_component["columns"] and
        "sales.item" in continued_component["columns"] and
        not any(
            column.startswith(f"{filtered_component['table_name']}.")
            or column.startswith(f"{first_component['table_name']}.")
            for column in continued_component["columns"]
        ),
        str(continued_component),
    )
    self_join = h.join_tables(
        base="employees",
        base_role="employee",
        joins=[{
            "table": "employees",
            "role": "peer",
            "on": [{"left": "employee.dept", "right": "dept"}],
        }],
    )
    t.check(
        "version5 self join requires semantic roles and stays flat",
        "employee.id" in self_join["columns"] and "peer.id" in self_join["columns"] and
        not any(column.startswith("join_") for column in self_join["columns"]),
        str(self_join),
    )
    try:
        h.join_tables(
            base="employees",
            joins=[{
                "table": "employees",
                "on": [{"left": "employees.dept", "right": "dept"}],
            }],
        )
        duplicate_role_rejected = False
    except ValueError as exc:
        duplicate_role_rejected = "semantic role" in str(exc)
    t.check("version5 repeated relation rejects missing role", duplicate_role_rejected)

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
            norm(h.gold("SELECT e.name,d.location FROM employees e JOIN depts d ON e.dept=d.dept")) and
            j_flat["columns"] == ["e__name", "d__location"], str(j_flat))

    try:
        h.join_tables(
            tables=["employees", "depts"],
            on=[[{"left": "e__dept", "right": "dept"}]],
            prefixes=["e", "d"],
            return_columns=["e.name", "d.location"],
        )
        dotted_return_rejected = False
    except ValueError as exc:
        dotted_return_rejected = "never table.column" in str(exc)
    t.check("join_tables rejects unresolved dotted return columns", dotted_return_rejected)

    j_documented_prefix = h.join_tables(
        tables=["employees", "depts"],
        on=[[{"left": "e__dept", "right": "dept"}]],
        join_types="inner",
        prefixes=["e", "d"],
        return_columns=["e__name", "d__location"],
    )
    t.check("join_tables accepts documented first-edge prefix", norm(h.rows(j_documented_prefix["table_name"])) ==
            norm(h.gold("SELECT e.name,d.location FROM employees e JOIN depts d ON e.dept=d.dept")))

    try:
        h.join_tables(
            tables=["employees", "depts"],
            on=[[{"left": "L.e__dept", "right": "dept"}]],
            join_types="inner",
            prefixes=["e", "d"],
        )
        internal_alias_rejected = False
    except Exception:  # The SQL alias must stay invalid rather than being silently stripped.
        internal_alias_rejected = True
    t.check("join_tables rejects internal SQL aliases", internal_alias_rejected)

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

    distinct_project = h.project("employees", ["dept"], distinct=True)
    t.check(
        "project distinct removes duplicate projected rows",
        norm(h.rows(distinct_project["table_name"])) ==
        norm(h.gold("SELECT DISTINCT dept FROM employees")),
        str(distinct_project),
    )

    percentage = h.scalar_compute("percent", [18, 19], "percentage")
    difference = h.scalar_compute("subtract", [25, 89], "difference")
    t.check(
        "scalar_compute returns exact 1x1 table results",
        percentage["columns"] == ["percentage"] and
        h.rows(percentage["table_name"])[0][0] == 18 * 100 / 19 and
        h.rows(difference["table_name"]) == [(-64,)],
        str((percentage, difference)),
    )
    elapsed = h.scalar_compute(
        "date_diff_days",
        ["2008-02-15", "2008-02-26"],
        "taken_days",
    )
    t.check(
        "scalar_compute computes date differences without model arithmetic",
        h.rows(elapsed["table_name"]) == [(11.0,)],
        str(elapsed),
    )

    duplicate_names = h._new("dup", "SELECT name AS Name, dept AS Name FROM employees")
    renamed = h.project(duplicate_names["table_name"], ["Name", "Name:1 AS department"])
    t.check("project quotes exact SQLite duplicate-column names",
            renamed["columns"] == ["Name", "department"] and
            norm(h.rows(renamed["table_name"])) == norm(h.gold("SELECT name,dept FROM employees")),
            str(renamed))

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

    first_page = h.read_subtable(
        "employees",
        columns=["id", "name"],
        order_by=["id"],
        limit=2,
    )
    second_page = h.read_subtable(
        "employees",
        columns=["id", "name"],
        order_by=["id"],
        limit=2,
        offset=first_page["next_offset"],
    )
    t.check(
        "read_subtable returns deterministic non-overlapping pages",
        first_page["rows"] == [[1, "A"], [2, "B"]]
        and first_page["has_more"]
        and first_page["next_offset"] == 2
        and second_page["rows"] == [[3, "C"], [4, "D"]]
        and second_page["offset"] == 2,
        str((first_page, second_page)),
    )
    try:
        h.read_subtable("employees", limit=21)
    except ValueError as exc:
        bounded_read = "1 to 20" in str(exc)
    else:
        bounded_read = False
    t.check("read_subtable enforces the limit cap in the executor", bounded_read)

    return t.result()
