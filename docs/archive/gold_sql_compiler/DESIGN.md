# Harness + SQL→tool Compiler — Design

Implementation of `tool_design/final_tool_design.md`. Turns Spider/BIRD **gold SQL** into
faithful **tool-call trajectories** and verifies them by execution. Test report: `REPORT.md`
(regenerate with `.venv/bin/python src/harness/run_all.py`).

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
| `executor.py` | Relational core. Each table-producing tool registers a named view (`SELECT … FROM (prev)`); reading/scalar tools run a SELECT. `join_tables` prefixes columns internally (`left_prefix`/`right_prefix`); `preview` inlines a table's content (small in full, large truncated+flagged) so the model perceives intermediates. | `Harness`: `condition_filter, group_aggregate, join_tables, derive_column, set_op, window, project, extreme_value_select, aggregate, read_subtable, preview, rows, gold` |
| `plan.py` | The IR between compiler and executor. `Step(id, tool, args)`; `run_plan` resolves step-id refs to real view names and returns the final rows. | `Step`, `run_plan`, `TABLE_REF_ARGS` |
| `compiler.py` | SQL → Plan. sqlglot AST → one Step per logical stage. Unsupported → `CompileError`. | `Compiler().compile(sql)`, `CompileError` |
| `verify.py` | Round-trip gate: compile → run → compare to gold SQL. | `round_trip(harness, sql) -> (status, info)` |
| `emitter.py` | Verified Plan → training **trajectory** (ReAct steps + terminal `answer_from_context`, `dataset_overview` initial state) + structural **legality** check. | `emit(harness, question, gold_sql)`, `validate(traj)` |
| `tests/` | Per-module unit tests + the round-trip suite. Each exposes `run() -> (passed, failed, fails)`. | `test_executor/plan/compiler/verify` |
| `run_all.py` | Integration: run all tests + Spider compile-coverage probe + emit a sample trajectory → write `REPORT.md`. | `main()` |
| `run_spider.py` | **Execution-verified** eval on the real Spider SQLite DBs: round-trip each gold SQL against its actual database. | `.venv/bin/python src/harness/run_spider.py [N]` |

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
- **Column rendering, two modes**: *bare* (single-table / no schema) strips qualifiers so columns
  resolve inside single-source views; *qualified* (joins + a known schema, `Compiler(schema)`)
  renders every reference as `<alias>__<col>` — resolving shared column names, self-joins, and
  ambiguity. The prefixing is **internalized in `join_tables`** (`left_prefix`/`right_prefix`): the
  first base table is prefixed by the first join, each newly-joined right table by `right_prefix`,
  and the accumulated intermediate (already prefixed) passes through. The compiler emits **no
  separate rename steps**, so a join is one `join_tables` call, not a rename-then-join. Join `ON`
  keys are given in source terms (bare for an un-prefixed base side, `<owner>__<col>` for a
  prefixed intermediate) and routed to L/R by which table each column belongs to (not SQL position).
- **Passthrough columns**: a SELECT column that is neither grouped nor aggregated (SQLite's lenient
  bare-column extension; functionally dependent on the group key in practice) is carried through
  `group_aggregate`'s `passthrough` so the final projection can reference it.
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

## Trajectory emission & validation

`emitter.emit(harness, question, gold_sql)` compiles the SQL, executes the Plan while capturing
each step's real `tool_output` + the assigned table names, verifies the final result against the
gold SQL (`label_status = verified`), and packages a trajectory: `dataset_overview` initial state,
one ReAct step per tool call (`think` / `tool_call` / `tool_output`), and a terminal
`answer_from_context` citing the final evidence table. Each table-producing step's `tool_output`
**inlines the new table's content via `harness.preview`** (small tables in full, large ones a
truncated head with `is_truncated` + a note) so the trajectory is closed-loop — the model perceives
intermediate data, not just a table handle. `emitter.validate(traj)` is the legality
gate: required fields, every `tool_call.tool` in the known tool set, well-formed steps, a terminal
`answer_from_context`, citation integrity (answer cites a real source/derived table), and that the
trajectory was execution-verified. A sample is written to `sample_trajectory.json` by `run_all.py`.

## Testing strategy

- **Unit** per module (`tests/test_*.py`): each tool vs gold SQL; plan threading; compiler structure
  + `CompileError` on scope-outs.
- **Round-trip backbone** (`test_verify`): every supported pattern is compiled, executed, and
  checked against gold SQL on a synthetic DB. This is the correctness guarantee.
- **Spider compile-coverage** (`run_all`): parse+decompose 2000 real Spider gold SQL; report the
  pass rate and bucket failures (the prioritized backlog). *Compile-only* — the HF parquet ships no
  SQLite DBs, so these are not execution-verified.

## Status (see REPORT.md)

- Unit tests: all passing (executor/plan/compiler/verify/emitter).
- Spider **compile** coverage (`run_all.py`, no DB): **~91%** of 2000 queries.
- Spider **execution-verified** coverage (`run_spider.py`, the real SQLite DBs): **~91.6%** of
  train queries round-trip exactly to the gold SQL's result on the actual database. Path:
  78.5% (initial, case-insensitive ids) → 90.3% (qualified-column mode, ON-key L/R routing,
  passthrough of non-grouped SELECT columns) → **91.6%** after **internalizing prefixing into
  `join_tables`** (no separate rename steps). Remaining is almost entirely **subqueries** (`IN
  (subquery)` semi/anti-join, scalar subquery in WHERE) — need value/table threading in the Plan
  IR (the next major work) — plus a handful of tie-ordering mismatches to triage.
- Batch generation (`gen_trajectories.py`): **6,256 train + 914 dev** execution-verified + legal
  trajectories (lengths 2–14); one readable sample per length in `sample_trajectories/by_length/`.

## Out of scope (v1)

Subqueries in WHERE / correlated subqueries (partial), set operations (executor ready, compiler
pending), window functions (executor ready, compiler pending), multi-table column-ambiguous joins,
free-form `semantic_match` data (built separately by literal-fuzzing augmentation — see
`tool_design/final_tool_design.md` §8). Row-id/provenance bookkeeping for citation + the trajectory
emitter are the next components.
