"""Tests for the trajectory emitter + validator (build steps 3 & 4)."""
from common import T, employees_db
from emitter import emit, validate

CASES = [
    ("How many employees work in engineering?",
     "SELECT COUNT(*) FROM employees WHERE dept = 'eng'"),
    ("What is the average salary per department?",
     "SELECT dept, AVG(salary) FROM employees GROUP BY dept"),
    ("Which departments have more than one employee?",
     "SELECT dept FROM employees GROUP BY dept HAVING COUNT(*) > 1"),
    ("Names and salaries of the two highest paid employees",
     "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 2"),
    ("Names and locations of employees earning over 1000",
     "SELECT name, location FROM employees JOIN depts ON employees.dept = depts.dept WHERE salary > 1000"),
]


def run():
    t = T("emitter")
    for q, sql in CASES:
        h = employees_db()
        traj = emit(h, q, sql, dataset="synthetic")
        t.check(f"verified: {q[:32]}", traj["label_status"] == "verified", str(traj.get("label_status")))
        t.check(f"legal:    {q[:32]}", validate(traj) == [], str(validate(traj)))
        t.check(f"terminal: {q[:32]}", traj["steps"][-1]["tool_call"]["tool"] == "answer_from_context")
        tools = [s["tool_call"]["tool"] for s in traj["steps"]]
        t.check(
            f"raw backbone has no perception: {q[:20]}",
            not ({"describe_table", "inspect_column", "read_subtable"} & set(tools)),
            str(tools),
        )

    # negative cases: the validator must reject tampered trajectories
    h = employees_db()
    bad = emit(h, "x", "SELECT name FROM employees")
    bad["steps"][0]["tool_call"]["tool"] = "frobnicate"
    t.check("rejects unknown tool", any("unknown tool" in e for e in validate(bad)), str(validate(bad)))

    bad2 = emit(h, "x", "SELECT name FROM employees")
    bad2["label_status"] = "mismatch"
    t.check("rejects unverified", any("verified" in e for e in validate(bad2)))

    bad3 = emit(h, "x", "SELECT name FROM employees")
    bad3["steps"] = bad3["steps"][:-1]  # drop the terminal answer step
    t.check("rejects missing terminal", any("answer_from_context" in e for e in validate(bad3)))

    return t.result()
