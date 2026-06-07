# Harness + SQL→tool compiler (build plan)

Implements `tool_design/final_tool_design.md`. Goal: compile Spider/BIRD **gold SQL** into a
faithful **tool-call trajectory**, execute it through the harness, and verify the result equals
executing the gold SQL.

```
gold SQL ──(compiler)──> tool-call chain ──(harness executor)──> result
                                   gold SQL ──(SQLite)──────────> result
                              verify: results equal  → accepted trajectory
```

## Status

Modular pipeline DONE for the v1 SQL core (see `DESIGN.md`, test results in `REPORT.md`):
- `executor.py` — relational core (tools → composed SQL over SQLite).
- `plan.py` — Plan IR (`Step`) + `run_plan` (step-id threading).
- `compiler.py` — SQL → Plan via sqlglot (FROM/JOIN/WHERE/GROUP/HAVING/ORDER/LIMIT/DISTINCT/projection/scalar-agg).
- `verify.py` — round-trip gate (compile → run → compare to gold SQL).
- `run_all.py` — integration runner; runs all unit tests + the round-trip suite + Spider compile-coverage, writes `REPORT.md`.

Run: `.venv/bin/python scripts/harness/run_all.py`. Current: **49 unit tests pass**, round-trip
suite **26/26**, **Spider compile coverage ~76%** (2000 queries) with a prioritized backlog.
Needs the project venv with sqlglot: `python3 -m venv .venv && .venv/bin/pip install sqlglot`.

## To build (in order)

1. **JSON adapter** — load FeTaQA/TableBench/tqabench tables into SQLite; build `dataset_overview`
   (types, relations, stats). Spider/BIRD need none (already SQLite).
2. **SQL→tool compiler** (`compiler.py`) — parse gold SQL to an AST and emit a tool chain.
   - Parser: `sqlglot` (clean transpile AST) or duckdb `json_serialize_sql` (no new dep). Lean sqlglot.
   - Mapping (one AST node → tool step), materialize intermediates as named views:
     | SQL | tool |
     |---|---|
     | FROM + JOIN ... ON | `join_tables` |
     | WHERE (AND of predicates) | `condition_filter` |
     | scalar expr in SELECT/WHERE/agg | `derive_column` / expression sublanguage |
     | GROUP BY + agg | `group_aggregate` |
     | HAVING | `condition_filter` on the group table |
     | scalar subquery | `aggregate` → use value as a literal in `condition_filter` |
     | ORDER BY ... LIMIT k | `extreme_value_select` |
     | UNION/INTERSECT/EXCEPT | `set_op` |
     | window fn | `window` |
     | final projection / scalar | `read_subtable` / `aggregate` → `answer_from_context` |
   - v1 scope: SELECT-FROM-WHERE-JOIN-GROUPBY-HAVING-ORDERBY-LIMIT (covers most Spider easy/medium);
     defer nested set-ops / correlated subqueries / window.
3. **Round-trip verifier** (`verify.py`) — run compiler→executor, compare result set to gold SQL;
   bucket Spider by hardness; report compile/verify coverage. Only verified chains become trajectories.
4. **Trajectory emitter** — wrap verified chains into the final-design trajectory format
   (think/tool_call/tool_response/answer, uniform (table,row_id) citation), source provenance.
5. **Fuzzy (semantic_match) data** — SQL never emits semantic_match. Generate it by
   **literal-fuzzing augmentation**: for a compiled trajectory whose `condition_filter(col='V')`
   binds a question literal `V`, perturb `V` in the question to a synonym/alias/description
   (alias table or LLM, **verified to still resolve to V**), and swap that step to
   `semantic_match(col, perturbed_query)`. Gold answer (from the exact SQL) is unchanged.
   Also: naturally-fuzzy datasets (WTQ/FeTaQA) via sample-and-filter; LLM-generated semantic
   questions + rejection sampling. Invariant: **answer always from execution; only the question
   surface is fuzzy.** See `tool_design/final_tool_design.md` §8.

## Notes

- The model never sees SQL; SQL is the executor's internal substrate (like a CPU for a code agent).
- Row identity / provenance bookkeeping (stable row_ids per view, member-row tracking for
  group/join) is layered on the executor for citation + process reward — to add.
