"""End-to-end round-trip suite: compile SQL -> run via harness -> must equal gold SQL.

Every case here MUST round-trip 'ok'. This is the correctness backbone for the supported subset.
"""
from common import T, employees_db
from verify import round_trip

CASES = [
    # projection / WHERE
    "SELECT name FROM employees WHERE salary > 1000",
    "SELECT name, dept FROM employees WHERE salary >= 1100 AND dept = 'eng'",
    "SELECT name FROM employees WHERE dept != 'hr'",
    "SELECT * FROM employees WHERE name LIKE '%A%'",
    "SELECT name FROM employees AS e WHERE e.age < 35",
    # scalar aggregates
    "SELECT COUNT(*) FROM employees",
    "SELECT COUNT(*) FROM employees WHERE dept = 'eng'",
    "SELECT SUM(salary) FROM employees",
    "SELECT AVG(salary) FROM employees",
    "SELECT MIN(age) FROM employees",
    "SELECT MAX(salary) FROM employees",
    "SELECT COUNT(DISTINCT dept) FROM employees",
    "SELECT SUM(price * qty) FROM sales",
    # group by / having
    "SELECT dept, COUNT(*) FROM employees GROUP BY dept",
    "SELECT dept, AVG(salary), MAX(salary) FROM employees GROUP BY dept",
    "SELECT dept, AVG(salary) AS a FROM employees GROUP BY dept HAVING AVG(salary) > 1000",
    "SELECT dept FROM employees GROUP BY dept HAVING COUNT(*) > 1",
    "SELECT dept, COUNT(*) AS c FROM employees GROUP BY dept ORDER BY c DESC",
    "SELECT dept, AVG(salary) FROM employees WHERE salary>1000 GROUP BY dept "
    "HAVING AVG(salary)>1200 ORDER BY AVG(salary) DESC LIMIT 2",
    # distinct / order / limit
    "SELECT DISTINCT dept FROM employees",
    "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 3",
    "SELECT name FROM employees ORDER BY age ASC LIMIT 2",
    "SELECT name, salary FROM employees ORDER BY age DESC LIMIT 2",
    # expression projection
    "SELECT item, price * qty AS rev FROM sales",
    # joins (qualified-column mode)
    "SELECT name, location FROM employees JOIN depts ON employees.dept = depts.dept WHERE salary > 1000",
    "SELECT e.name, d.location FROM employees AS e JOIN depts AS d ON e.dept = d.dept",
    # ON written right-table-first (exercises join-key L/R routing)
    "SELECT e.name FROM employees AS e JOIN depts AS d ON d.dept = e.dept WHERE d.budget > 1000",
    # non-grouped, non-aggregated SELECT column carried through (SQLite lenient + passthrough)
    "SELECT d.dept, d.location, COUNT(*) FROM employees AS e JOIN depts AS d ON e.dept = d.dept GROUP BY d.dept",
    # boolean tree: OR / NOT / IN / BETWEEN / LIKE
    "SELECT name FROM employees WHERE dept = 'eng' OR dept = 'hr'",
    "SELECT name FROM employees WHERE salary > 1000 AND (dept = 'eng' OR age > 40)",
    "SELECT name FROM employees WHERE dept NOT IN ('eng')",
    "SELECT name FROM employees WHERE dept IN ('eng', 'sales')",
    "SELECT name FROM employees WHERE age BETWEEN 30 AND 45",
    "SELECT name FROM employees WHERE name LIKE 'A%'",
    "SELECT name FROM employees WHERE NOT salary > 1000",
    # multiple scalar aggregates
    "SELECT MAX(salary), MIN(salary) FROM employees",
    "SELECT COUNT(*), AVG(age) FROM employees WHERE dept = 'eng'",
    # set operations
    "SELECT dept FROM employees INTERSECT SELECT dept FROM depts",
    "SELECT dept FROM employees EXCEPT SELECT dept FROM depts WHERE budget < 2000",
    "SELECT name FROM employees WHERE dept='eng' UNION SELECT name FROM employees WHERE dept='hr'",
    # scalar subqueries: aggregate -> filter(value_ref=that step) -> ...
    "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)",
    "SELECT name FROM employees WHERE age = (SELECT MAX(age) FROM employees)",
    "SELECT COUNT(*) FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)",
    "SELECT name FROM employees WHERE salary >= (SELECT MIN(salary) FROM employees WHERE dept='eng')",
    # IN / NOT IN (subquery): semi-join / anti-join via membership against the subquery's table
    "SELECT name FROM employees WHERE dept IN (SELECT dept FROM depts WHERE budget > 1000)",
    "SELECT name FROM employees WHERE dept NOT IN (SELECT dept FROM depts WHERE budget < 2000)",
    "SELECT name FROM employees WHERE salary IN (SELECT MAX(salary) FROM employees)",
    # mixed aggregate + bare column, no GROUP BY (SQLite extension)
    "SELECT dept, COUNT(*), SUM(salary) FROM employees",
    # bare SELECT * (must emit a step, not an empty plan)
    "SELECT * FROM employees",
    # IN (set-op subquery) and = (single-row LIMIT-1 subquery used as a scalar)
    "SELECT name FROM employees WHERE dept IN "
    "(SELECT dept FROM depts WHERE budget>1000 INTERSECT SELECT dept FROM depts WHERE budget<5000)",
    "SELECT name FROM employees WHERE salary = (SELECT salary FROM employees ORDER BY salary DESC LIMIT 1)",
]


def run():
    t = T("verify (round-trip)")
    for q in CASES:
        h = employees_db()
        st, info = round_trip(h, q)
        t.check(q, st == "ok", f"{st}: {info if st != 'ok' else ''}")
    return t.result()
