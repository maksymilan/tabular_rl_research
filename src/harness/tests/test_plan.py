"""Unit tests for plan.run_plan (step-id resolution + result extraction)."""
from common import T, employees_db, norm
from plan import Step, run_plan


def run():
    t = T("plan")
    h = employees_db()

    # s2 references s1's output by id -> must resolve to the real view name
    plan = [
        Step("s1", "condition_filter",
             {"table": "employees", "conditions": [{"column": "salary", "op": ">", "value": 1000}]}),
        Step("s2", "group_aggregate",
             {"table": "s1", "group_by": ["dept"], "aggregations": [{"op": "mean", "column": "salary", "as": "a"}]}),
    ]
    got = run_plan(h, plan)
    gold = h.gold("SELECT dept,AVG(salary) FROM employees WHERE salary>1000 GROUP BY dept")
    t.check("step-id threading s1->s2", norm(got) == norm(gold), str(got))

    # whole-table scalar aggregate returns one row as a table
    plan2 = [
        Step("s1", "condition_filter",
             {"table": "employees", "conditions": [{"column": "dept", "op": "=", "value": "eng"}]}),
        Step("s2", "group_aggregate",
             {"table": "s1", "group_by": [], "aggregations": [{"op": "count", "column": "*", "as": "count_1"}]}),
    ]
    t.check("scalar aggregate table returns one row", run_plan(h, plan2) == [(3,)], str(run_plan(h, plan2)))

    # source-table reference (no step id) passes through unchanged
    plan3 = [Step("s1", "project", {"table": "depts", "expressions": ["location"]})]
    t.check("source ref passthrough", norm(run_plan(h, plan3)) == norm(h.gold("SELECT location FROM depts")))

    # aggregation-level predicates retain direct scalar step references until execution
    plan4 = [
        Step(
            "s1",
            "group_aggregate",
            {
                "table": "employees",
                "group_by": [],
                "aggregations": [{"op": "mean", "column": "salary", "as": "mean_salary"}],
            },
        ),
        Step(
            "s2",
            "group_aggregate",
            {
                "table": "employees",
                "group_by": [],
                "aggregations": [{
                    "op": "count",
                    "column": "*",
                    "as": "above_mean",
                    "where": {"column": "salary", "op": ">", "value_ref": "s1"},
                }],
            },
        ),
    ]
    t.check(
        "conditional aggregate resolves a scalar value_ref",
        run_plan(h, plan4) == h.gold(
            "SELECT COUNT(CASE WHEN salary > (SELECT AVG(salary) FROM employees) THEN 1 END) "
            "FROM employees"
        ),
        str(run_plan(h, plan4)),
    )

    # A one-row multi-metric aggregate can feed scalar arithmetic without repeated aggregation.
    plan5 = [
        Step(
            "s1",
            "group_aggregate",
            {
                "table": "employees",
                "group_by": [],
                "aggregations": [
                    {"op": "count", "column": "*", "as": "total_employees"},
                    {
                        "op": "count",
                        "column": "*",
                        "as": "engineering_employees",
                        "where": {"column": "dept", "op": "=", "value": "eng"},
                    },
                ],
            },
        ),
        Step(
            "s2",
            "scalar_compute",
            {
                "operation": "percent",
                "operands": [
                    {"value_ref": "s1", "column": "engineering_employees"},
                    {"value_ref": "s1", "column": "total_employees"},
                ],
                "result_name": "engineering_percentage",
            },
        ),
    ]
    t.check(
        "scalar arithmetic resolves named columns from one aggregate row",
        run_plan(h, plan5) == [(50.0,)],
        str(run_plan(h, plan5)),
    )

    return t.result()
