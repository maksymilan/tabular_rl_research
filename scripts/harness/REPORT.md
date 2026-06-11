# Harness + Compiler — Test Report

Generated 2026-06-11 10:05 · `python scripts/harness/run_all.py`

## Unit tests — 91/91 passed

| module | passed | failed |
|---|---|---|
| test_executor | 11 | 0 |
| test_plan | 3 | 0 |
| test_compiler | 15 | 0 |
| test_verify | 44 | 0 |
| test_emitter | 18 | 0 |

## Spider compile coverage — 1885/2000 (94.2%)

Parse + decompose of real Spider gold SQL. *Compile coverage only* — the HF parquet ships no SQLite DBs, so results are not execution-verified here.

Top unsupported-construct buckets:

| count | reason | example |
|---|---|---|
| 101 | IN (subquery) unsupported | `SELECT count(*) FROM department WHERE department_id NOT IN (SELECT dep…` |
| 8 | only scalar (single-aggregate) subqueries are su | `SELECT t1.catalog_entry_name FROM Catalog_Contents AS t1 JOIN Catalog_…` |
| 4 | mixed aggregate + non-aggregate without GROUP BY | `SELECT billing_state ,  COUNT(*) ,  SUM(total) FROM invoices WHERE bil…` |
| 2 | join ON supports equality / AND of equalities on | `SELECT count(*) FROM station AS T1 JOIN trip AS T2 JOIN station AS T3 …` |

## Notes

- Round-trip suite (`test_verify`) is the correctness backbone: each case is compiled, executed via the harness, and checked against gold SQL on a synthetic DB.
- Spider coverage measures how much of real-world SQL the compiler structurally handles; failure buckets are the prioritized backlog (joins-with-ambiguous-columns, subqueries, set ops, window, etc.).
- Pipeline: `compiler.py` (SQL→Plan) · `plan.py` (run) · `executor.py` (tools→SQL) · `verify.py` (round-trip). See `tool_design/final_tool_design.md`.
