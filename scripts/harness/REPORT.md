# Harness + Compiler — Test Report

Generated 2026-06-08 09:08 · `python scripts/harness/run_all.py`

## Unit tests — 83/83 passed

| module | passed | failed |
|---|---|---|
| test_executor | 10 | 0 |
| test_plan | 3 | 0 |
| test_compiler | 12 | 0 |
| test_verify | 40 | 0 |
| test_emitter | 18 | 0 |

## Spider compile coverage — 1824/2000 (91.2%)

Parse + decompose of real Spider gold SQL. *Compile coverage only* — the HF parquet ships no SQLite DBs, so results are not execution-verified here.

Top unsupported-construct buckets:

| count | reason | example |
|---|---|---|
| 101 | IN (subquery) unsupported | `SELECT count(*) FROM department WHERE department_id NOT IN (SELECT dep…` |
| 69 | predicate RHS Subquery unsupported | `SELECT date ,  zip_code FROM weather WHERE min_dew_point_f  <  (SELECT…` |
| 4 | mixed aggregate + non-aggregate without GROUP BY | `SELECT billing_state ,  COUNT(*) ,  SUM(total) FROM invoices WHERE bil…` |
| 2 | join ON supports equality / AND of equalities on | `SELECT count(*) FROM station AS T1 JOIN trip AS T2 JOIN station AS T3 …` |

## Notes

- Round-trip suite (`test_verify`) is the correctness backbone: each case is compiled, executed via the harness, and checked against gold SQL on a synthetic DB.
- Spider coverage measures how much of real-world SQL the compiler structurally handles; failure buckets are the prioritized backlog (joins-with-ambiguous-columns, subqueries, set ops, window, etc.).
- Pipeline: `compiler.py` (SQL→Plan) · `plan.py` (run) · `executor.py` (tools→SQL) · `verify.py` (round-trip). See `tool_design/final_tool_design.md`.
