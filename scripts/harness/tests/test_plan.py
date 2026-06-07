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

    # terminal scalar tool returns one row
    plan2 = [
        Step("s1", "condition_filter",
             {"table": "employees", "conditions": [{"column": "dept", "op": "=", "value": "eng"}]}),
        Step("s2", "aggregate", {"table": "s1", "column": "*", "op": "count"}),
    ]
    t.check("terminal scalar wraps as row", run_plan(h, plan2) == [(3,)], str(run_plan(h, plan2)))

    # source-table reference (no step id) passes through unchanged
    plan3 = [Step("s1", "project", {"table": "depts", "expressions": ["location"]})]
    t.check("source ref passthrough", norm(run_plan(h, plan3)) == norm(h.gold("SELECT location FROM depts")))

    return t.result()
