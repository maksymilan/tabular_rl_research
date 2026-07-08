# Harness + Compiler — Test Report

Generated 2026-07-07 23:27 · `python src/harness/run_all.py`

## Unit tests — 131/131 passed

| module | passed | failed |
|---|---|---|
| test_executor | 26 | 0 |
| test_environment_state | 12 | 0 |
| test_plan | 3 | 0 |
| test_compiler | 16 | 0 |
| test_verify | 51 | 0 |
| test_emitter | 23 | 0 |

## Spider compile coverage — 1998/2000 (99.9%)

Parse + decompose of real Spider gold SQL. *Compile coverage only* — the HF parquet ships no SQLite DBs, so results are not execution-verified here.

Top unsupported-construct buckets:

| count | reason | example |
|---|---|---|
| 2 | join ON supports equality / AND of equalities on | `SELECT count(*) FROM station AS T1 JOIN trip AS T2 JOIN station AS T3 …` |

## Notes

- Round-trip suite (`test_verify`) is the correctness backbone: each case is compiled, executed via the harness, and checked against gold SQL on a synthetic DB.
- Spider coverage measures how much of real-world SQL the compiler structurally handles; failure buckets are the prioritized backlog (joins-with-ambiguous-columns, subqueries, set ops, window, etc.).
- Pipeline: `compiler.py` (SQL→Plan) · `plan.py` (run) · `executor.py` (tools→SQL) · `verify.py` (round-trip). See `tool_design/final_tool_design.md`.
