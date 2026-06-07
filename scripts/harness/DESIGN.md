# Harness + SQL→tool Compiler — Design

Implementation of `tool_design/final_tool_design.md`. Turns Spider/BIRD **gold SQL** into
faithful **tool-call trajectories** and verifies them by execution. Test report: `REPORT.md`
(regenerate with `.venv/bin/python scripts/harness/run_all.py`).

## Pipeline & data flow

```
SQL string
  │  compiler.py        parse (sqlglot) → walk logical pipeline → emit Plan (list[Step])
  ▼
Plan = [Step(id, tool, args)]      args reference inputs by source-table name or earlier step id
  │  plan.run_plan(harness, plan)  resolve step ids → real view names; execute each tool
  ▼
result rows  ───────────────┐
                            │  verify.round_trip: compare to harness.gold(SQL)
gold SQL → harness.gold ─────┘     equal ⇒ faithful trajectory (status "ok")
```

The model never sees SQL: `executor.py` translates each abstract tool call into **composed SQL**
over a SQLite backend (Spider/BIRD DBs are SQLite already; JSON datasets load into SQLite). SQL is
the executor's internal substrate, like a CPU is for a code agent.

## Modules

| Module | Responsibility | Key API |
|---|---|---|
| `executor.py` | Relational core. Each table-producing tool registers a named view (`SELECT … FROM (prev)`); reading/scalar tools run a SELECT. | `Harness`: `condition_filter, group_aggregate, join_tables, derive_column, set_op, window, project, order_limit, aggregate, extreme_value_select, read_subtable, rows, gold` |
| `plan.py` | The IR between compiler and executor. `Step(id, tool, args)`; `run_plan` resolves step-id refs to real view names and returns the final rows. | `Step`, `run_plan`, `TABLE_REF_ARGS` |
| `compiler.py` | SQL → Plan. sqlglot AST → one Step per logical stage. Unsupported → `CompileError`. | `Compiler().compile(sql)`, `CompileError` |
| `verify.py` | Round-trip gate: compile → run → compare to gold SQL. | `round_trip(harness, sql) -> (status, info)` |
| `tests/` | Per-module unit tests + the round-trip suite. Each exposes `run() -> (passed, failed, fails)`. | `test_executor/plan/compiler/verify` |
| `run_all.py` | Integration: run all tests + Spider compile-coverage probe → write `REPORT.md`. | `main()` |

## The Plan IR

A `Step` is one abstract tool call. Table-reference arguments (`table`, or `left`/`right` for
joins/set-ops) hold either a **source table name** or the **id of an earlier step** (e.g. `"s2"`).
`run_plan` threads outputs: a table-producing step's id maps to the harness-assigned view name;
the final step's result is returned as rows (a scalar `aggregate` is wrapped as one row). This
keeps the compiler pure/inspectable and lets the same Plan drive both verification and (later)
trajectory emission.

## Compiler: logical-pipeline walk

Per SELECT, in logical order:
`FROM/JOIN → WHERE → GROUP BY → HAVING → ORDER BY/LIMIT → SELECT projection → DISTINCT`.

Notable decisions:
- **Bare columns**: table qualifiers are stripped (`e.salary`→`salary`) so columns resolve inside
  single-source views; this also absorbs most single-table aliases.
- **HAVING/ORDER aggregates** absent from SELECT are still materialized as group columns
  (`_extend_aggs`), then dropped by the **final projection** so the result matches the gold SELECT.
- **Scalar aggregate** with no GROUP BY (`SELECT COUNT(*) …`) compiles to a terminal `aggregate`.
- **DISTINCT** = `group_aggregate` with no aggregations.
- Anything outside scope raises `CompileError` — surfaced as a coverage bucket, never mis-compiled.

## Extending to new SQL

Two layers, both additive:
1. **Executor** (rarely needed): add a method that emits a SQL fragment over the input view. The
   composition-as-nested-views model already gives arbitrary nesting; `set_op`/`window` exist.
2. **Compiler**: add the AST case that recognizes the construct and emits the tool step(s).

Examples (executor already supports these — only the compiler case is missing):
- `WHERE a OR b` → `set_op(filter(a), filter(b), union)`, or an `OR` expression in one filter.
- `BETWEEN x AND y` → two predicates `>= x` and `<= y`.
- `UNION/INTERSECT/EXCEPT` → `set_op` on the two compiled sides.
- `top-3 per group` → `window(partition_by, order_by, row_number)` → `condition_filter(rn<=3)`.
- correlated subquery → `group_aggregate` → `join_tables` → `condition_filter` (column-vs-column).

## Testing strategy

- **Unit** per module (`tests/test_*.py`): each tool vs gold SQL; plan threading; compiler structure
  + `CompileError` on scope-outs.
- **Round-trip backbone** (`test_verify`): every supported pattern is compiled, executed, and
  checked against gold SQL on a synthetic DB. This is the correctness guarantee.
- **Spider compile-coverage** (`run_all`): parse+decompose 2000 real Spider gold SQL; report the
  pass rate and bucket failures (the prioritized backlog). *Compile-only* — the HF parquet ships no
  SQLite DBs, so these are not execution-verified.

## Status (see REPORT.md)

- Unit tests: all passing (executor/plan/compiler/verify); round-trip suite 26/26.
- Spider compile coverage: **~91%** of 2000 queries (after boolean condition trees, set ops,
  multiple scalar aggregates, and Spider's double-quoted-string convention). Remaining frontier
  (~9%, almost all subqueries): `IN (subquery)` (semi/anti-join), scalar subquery in WHERE —
  both need value/table threading from a subquery's steps into the outer query (a Plan-IR
  extension); plus a few multi-table self-joins and mixed agg/non-agg projections.

## Out of scope (v1)

Subqueries in WHERE / correlated subqueries (partial), set operations (executor ready, compiler
pending), window functions (executor ready, compiler pending), multi-table column-ambiguous joins,
free-form `semantic_match` data (built separately by literal-fuzzing augmentation — see
`tool_design/final_tool_design.md` §8). Row-id/provenance bookkeeping for citation + the trajectory
emitter are the next components.
