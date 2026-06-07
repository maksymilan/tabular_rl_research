# Harness + Compiler — Test Report

Generated 2026-06-07 20:44 · `python scripts/harness/run_all.py`

## Unit tests — 49/49 passed

| module | passed | failed |
|---|---|---|
| test_executor | 10 | 0 |
| test_plan | 3 | 0 |
| test_compiler | 10 | 0 |
| test_verify | 26 | 0 |

## Spider compile coverage — 1528/2000 (76.4%)

Parse + decompose of real Spider gold SQL. *Compile coverage only* — the HF parquet ships no SQLite DBs, so results are not execution-verified here.

Top unsupported-construct buckets:

| count | reason | example |
|---|---|---|
| 80 | predicate Not unsupported | `SELECT count(*) FROM department WHERE department_id NOT IN (SELECT dep…` |
| 66 | predicate RHS Subquery unsupported | `SELECT date ,  zip_code FROM weather WHERE min_dew_point_f  <  (SELECT…` |
| 62 | top-level Intersect unsupported (only SELECT) | `SELECT T3.born_state FROM department AS T1 JOIN management AS T2 ON T1…` |
| 60 | multiple scalar aggregates without GROUP BY unsu | `SELECT max(budget_in_billions) ,  min(budget_in_billions) FROM departm…` |
| 60 | predicate Or unsupported | `SELECT Official_Name FROM city WHERE Population  >  1500 OR Population…` |
| 60 | top-level Except unsupported (only SELECT) | `SELECT id FROM station WHERE lat  >  37.4 EXCEPT SELECT station_id FRO…` |
| 31 | predicate Like unsupported | `SELECT zip_code  ,  avg(mean_temperature_f) FROM weather WHERE date LI…` |
| 21 | predicate In unsupported | `SELECT avg(followers) FROM user_profiles WHERE UID IN (SELECT UID FROM…` |
| 16 | predicate Between unsupported | `SELECT avg(num_employees) FROM department WHERE ranking BETWEEN 10 AND…` |
| 10 | top-level Union unsupported (only SELECT) | `SELECT student_id FROM student_course_registrations UNION SELECT stude…` |
| 4 | mixed aggregate + non-aggregate without GROUP BY | `SELECT billing_state ,  COUNT(*) ,  SUM(total) FROM invoices WHERE bil…` |
| 2 | join ON supports equality / AND of equalities on | `SELECT count(*) FROM station AS T1 JOIN trip AS T2 JOIN station AS T3 …` |

## Notes

- Round-trip suite (`test_verify`) is the correctness backbone: each case is compiled, executed via the harness, and checked against gold SQL on a synthetic DB.
- Spider coverage measures how much of real-world SQL the compiler structurally handles; failure buckets are the prioritized backlog (joins-with-ambiguous-columns, subqueries, set ops, window, etc.).
- Pipeline: `compiler.py` (SQL→Plan) · `plan.py` (run) · `executor.py` (tools→SQL) · `verify.py` (round-trip). See `tool_design/final_tool_design.md`.
