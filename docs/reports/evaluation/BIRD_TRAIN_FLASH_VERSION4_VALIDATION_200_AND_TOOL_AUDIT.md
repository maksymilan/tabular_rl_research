# BIRD-train Flash version4 fresh-200 validation and tool audit

Date: 2026-07-23

## Scope

This run answers two questions:

1. Does the current `version4` typed-tool loop remain usable on a fresh stratified sample after the
   recent prompt, provider-feedback, projection, and join-name changes?
2. Are the public protocol, executor, resident state, RL environment, scorer, and current
   documentation actually one coherent contract?

The validation cohort is disjoint from the earlier 200-task SQL-versus-tools ablation.

## Frozen setup

- Source: `data/eval_inputs/bird_train_sft1_scale1000c.jsonl`
- Input: `data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`
- Seed: `20260723`
- Distribution: 80 easy, 60 medium, 60 hard
- Databases: 57
- Overlap with the prior 200-task sample: 0
- Model: `deepseek-v4-flash`
- Attempts: one complete episode per task
- Budget: 30 semantic actions, 3 recoverable errors per class
- Context: rolling legal history, last 4 successful turns, full prompt
- Parser: strict, with no XML/JSON repair
- Judge: `strict-multiset`
- Run protocol hash: `1673eaf158294de1`

An initial 30-result run was deliberately interrupted and retained with an `aborted30` suffix after
the audit found that `generate_teacher_rollouts.py` still used the old keyword classifier. DeepSeek
carrier feedback contains the literal JSON key `"arguments"`, so carrier failures were being
misclassified as `argument_validation_error`. The generator was aligned with `rollout.py`, a
regression test was added, and the reported 200-task run was then restarted from scratch.

## Main result

| Outcome | Count |
| --- | ---: |
| Correct | 125/200 (62.5%) |
| Legal terminal | 188/200 (94.0%) |
| Clean success | 78 |
| Recovered success | 47 |
| Legal wrong denotation | 63 |
| Protocol/carrier termination | 12 |
| Argument/execution terminal | 0 |

By difficulty:

| Difficulty | Correct | Legal | Mean actions |
| --- | ---: | ---: | ---: |
| Easy | 58/80 (72.5%) | 76/80 | 6.68 |
| Medium | 33/60 (55.0%) | 56/60 | 8.20 |
| Hard | 34/60 (56.7%) | 56/60 | 9.47 |

Hard being one point above medium is sample/model variance, not evidence that the hard stratum is
easier. The hard sample contains 57 join queries, 35 multi-join queries, and 20 subqueries.

The run used 1,594 API requests and 6.92M total tokens. There were no API transport retries or
terminal API failures.

## Error and recovery audit

There were 163 rejected semantic actions:

| Error event | Count |
| --- | ---: |
| Protocol/carrier | 135 |
| Argument validation | 6 |
| Execution | 22 |

All 135 protocol events are now counted against the correct budget. Their carrier causes are:

- visible prose before the tool call: 93;
- missing native reasoning content: 18;
- incomplete or suffixed visible tool call: 22;
- other strict-parser failures: 2.

Ninety-five episodes encountered at least one error. Forty-seven still ended correctly, confirming
that same-episode feedback recovery is active and useful. Rejected calls remain audit-only and are
not SFT targets.

## Terminal scorer audit

Every episode was replayed through the harness to identify the scorer branch that decided the final
result:

| Terminal route | Count |
| --- | ---: |
| Exact evidence table | 37 |
| Explicit scalar answer, `evidence=null` | 87 |
| Evidence accepted only after column permutation | 1 |
| Legal incorrect terminal | 63 |
| No legal terminal | 12 |

All 87 explicit-answer successes are 1-row x 1-column denotations, so they follow the documented
scalar rule. No success used a correct literal to override a nonmatching evidence table.

One task, `bird_train_06336`, is accepted only because `score()` permutes same-width evidence
columns. The question asks for words and ids; `top_001` contains `[wid, word]`, while gold is
`[word, wid]`. The prompt and current documentation say column order must be exact, so the truly
contract-strict result is **124/200 (62.0%)**, not 125/200. The scorer and prompt must choose one
rule.

## Join audit

This sample is deliberately informative for join design:

- 161/200 gold queries contain a join;
- 52/200 contain at least two joins;
- the model called `join_tables` 127 times in 97 episodes;
- 107 calls executed; 20 calls failed, a 15.7% call-level execution-error rate;
- 18 episodes had a join execution error, and 9 recovered to a correct final answer.

Every join execution error is an identifier-contract error:

| Join error | Count |
| --- | ---: |
| Dotted `table.column` in `return_columns` | 11 |
| `L.`/`R.` or dotted identifier in `on` | 9 |

Observed success rates:

| Gold SQL shape | Correct |
| --- | ---: |
| No join | 28/39 (71.8%) |
| One join | 68/109 (62.4%) |
| Multiple joins | 29/52 (55.8%) |

Episodes where the model used `join_tables` scored 54/97 (55.7%); episodes without a join call
scored 71/103 (68.9%). These raw rates are confounded by task difficulty and must not be read as the
causal cost of one tool. Within easy tasks, one-join queries actually score 38/52 versus 20/28 for
no-join tasks. The causal evidence is narrower but strong: all 20 join execution errors come from
the public naming convention rather than missing database facts.

### Why the current join interface is too complex

`version4` combines three positional structures: `tables`, `on`, and `prefixes`. Their meanings
change while the left fold advances:

- on the first edge, the left key may be a future `P1__column` name even though that name is not yet
  materialized;
- the newly attached right key must remain a bare source column;
- on later edges, the left key must use an already materialized `Pi__column`;
- `return_columns` must use final materialized names and reject dotted source names.

This is a small state machine embedded in string spelling. It saves join actions by allowing an
n-way path in one call, but it transfers bookkeeping from the harness to the model. The error data
confirms that this raises the probability of an otherwise semantically correct join being rejected.

The executor also contains dangerous silent behavior that is not visible in the successful-call
statistics:

- an unknown `join_types` value silently becomes an inner join;
- a short `on` chain can make a later fold execute without an ON clause, producing a Cartesian
  product;
- short `prefixes` are padded with `None`;
- bare mode drops a right-table column whenever its name matches any left-table column, even when
  that column is not the join key.

These were reproduced on a minimal SQLite database. They should be validation errors, never
successful state transitions.

### Recommended version5 join

Use one binary join per action and one identifier grammar everywhere. For example:

```json
{
  "tool": "join_tables",
  "arguments": {
    "left": {"table": "filter_001", "as": "language"},
    "right": {"table": "country", "as": "country"},
    "on": [{"left": "language.Country", "right": "country.Code"}],
    "join_type": "inner",
    "return_columns": ["country.Name"]
  }
}
```

The harness should validate both source instances and columns before SQL generation and should own
all materialized output names. A multi-table query then uses several simple, causally visible join
actions. That is slightly longer, but easier to learn, easier to recover, and better aligned with
step-local RL credit than one high-entropy n-way call.

## Contract alignment audit

At the top level, the design is wired correctly: every one of the 11 public tools either maps to a
`Harness` method or to the special `plan`/`answer_from_context` dispatch. The active suites pass:

- harness: 49/49;
- SFT: 47/47;
- RL: 30/30;
- eval: 17/17.

The current BIRD adapter also passes a fresh all-database smoke: 69/69 databases, 532 tables, zero
errors. The database layer is healthy. The following protocol/environment gaps remain.

### P0: fix before scaling data or process RL

1. **Execution errors are not transactionally state-preserving.** `Harness._new()` increments the
   handle counter and inserts a view before validating/counting the SQL. A failed project leaves an
   invalid `project_001` in `views`; the next successful output becomes `project_002`. Resident state
   hashes do not see this hidden mutation, so the recovery layer incorrectly calls the error
   state-preserving.
2. **Nested argument validation is largely absent.** The strict parser validates top-level keys, but
   accepts strings where `tables`, `on`, `expressions`, `group_by`, or `aggregations` require arrays,
   accepts invalid set/join operators, and does not validate predicate leaf shapes. This converts
   model argument mistakes into internal `KeyError`/`TypeError`/SQL errors. Two of this run's
   non-join execution errors are exactly malformed predicate objects.
3. **Join validation permits silent semantic changes.** Lengths, enums, edge shapes, column
   existence, source instances, and uniqueness must be checked before SQL construction.
4. **Teacher/eval and RL environments are not prompt/error equivalent.** The Accelerate backend
   requests `rolling-legal-history`, but `ToolUseEnv` defaults to plain `SYSTEM_PROMPT` without the
   rolling-history suffix. Its error feedback is also raw `type: message`, while teacher/eval use
   `format_tool_error()` with valid handles, available columns, and join-specific guidance.
5. **Terminal scoring contradicts the exact-order prompt.** Remove column permutation, or document
   and train the relaxed rule. This run changes by one task, but the contract should not be implicit.

### P1: remove active compatibility residue

- `_prefix_alias()` silently accepts a join column prefix as a table handle.
- `_resolve_col()` strips `table.` from dotted columns for most tools, even though join rejects the
  same spelling and documentation says dots are invalid in version4.
- `condition_filter` normalizes legacy predicate shorthands and operator spellings inside the
  executor rather than rejecting them at the public schema boundary.
- `condition_filter(return_columns=["missing"])` can produce a successful constant string column
  under SQLite's double-quoted-string compatibility instead of reporting a missing column.
- Plan nested fields are not schema checked: unsupported fields are silently ignored, and unknown
  evidence step ids are stored as `unresolved` rather than rejected.
- Active provenance constants still live in compiler-oriented `src/harness/plan.py` and include
  hidden `derive_column`, `aggregate`, and `window` tools. `process_credit.py` also names retired
  tools in active sets.
- `docs/current/research_plan.md` still describes older version1/state-only work, and
  `tool_protocol.md` contains stale `v2c-plan` wording. These are documentation residues, not runtime
  failures, but `docs/current` is supposed to be authoritative.

The replay parser may keep compatibility code, but live execution should receive an explicit
`mode=current|replay` boundary rather than relying on the strict parser to prevent hidden shims from
being reached.

## SQL semantic sufficiency

The current tool set is broadly sufficient for the dominant BIRD workload, but it is not a complete
SQL algebra.

This fresh sample contains 161 joins, 106 aggregates, 33 `GROUP BY`, 7 `HAVING`, 39 orderings, 28
`CASE` expressions, and 25 subqueries. The available tools can compose the common cases:

- selection and Boolean predicates: `condition_filter`;
- projection, arithmetic, `CASE`, casts, and scalar functions: `project`;
- equi-joins and self-joins: `join_tables`;
- grouped/scalar aggregates and distinct groups: `group_aggregate`;
- `HAVING`: aggregate first, then filter the aggregate handle;
- ordering/top-k: `extreme_value_select`;
- uncorrelated scalar/set subqueries: `value_ref` and `in_table`;
- union/intersection/difference: `set_op`.

The main semantic gaps are:

- no public window-function tool;
- no general non-equality/theta join;
- no direct correlated `EXISTS`/`NOT EXISTS` representation;
- no offset/pagination operator;
- no direct right/full outer join;
- multi-scalar arithmetic is possible only through raw SQL expressions in `project`;
- `project` therefore acts as an untyped SQL escape hatch, contradicting the otherwise typed-tool
  abstraction and the documentation statement that the model does not emit SQL fragments.

The 6,601-task filtered BIRD-train corpus contains only two window queries and no `EXISTS` in the
current local extraction, so missing window/exists support is not the present accuracy bottleneck.
Join ergonomics, validation, terminal semantics, and cross-environment parity are higher priority.
Do not add more tools until those boundaries are fixed.

## Conclusion

The core approach is viable: a fresh Flash pass reaches 62.5%, 94% legal termination, and 47 genuine
same-episode recoveries. The underlying BIRD SQLite adapter is healthy, and the operator set covers
most target queries.

The current environment is **not yet clean enough for large-scale regeneration or process RL**.
The most serious problems are hidden executor mutation after errors, permissive nested schemas,
teacher/RL prompt-feedback drift, and the join API's asymmetric naming state machine. The join tool
is not semantically unnecessary; it is ergonomically over-complex. Keep join, replace the n-way
prefix contract with a binary structured-source interface, then run a migration smoke before
regenerating training data.

## Artifacts

- Frozen input manifest:
  `data/eval_inputs/bird_train_tool_interface_validation200_version4.manifest.json`
- All attempts: `data/trajectories/bird_train_version4_validation200_flash_all.jsonl`
- Successes: `data/trajectories/bird_train_version4_validation200_flash_success.jsonl`
- Failures: `data/trajectories/bird_train_version4_validation200_flash_failures.jsonl`
- Generator manifest:
  `data/trajectories/bird_train_version4_validation200_flash_success.manifest.json`
- Audit summary: `data/results/bird_train_version4_validation200_flash_audit/summary.json`
- Join errors: `data/results/bird_train_version4_validation200_flash_audit/join_errors.jsonl`
- Terminal routes:
  `data/results/bird_train_version4_validation200_flash_audit/terminal_routes.jsonl`
