# BIRD-train Flash interface ablation (200 tasks)

Date: 2026-07-22

## Question

On the same stratified BIRD-train sample, compare DeepSeek v4 Flash using:

1. the current typed relational-tool protocol; and
2. iterative SQLite with execution feedback (`execute_sql`, `submit_sql`).

The purpose is to separate teacher-model limitations from avoidable protocol and tool-design losses.
This is not a BIRD dev leaderboard evaluation.

## Frozen setup

- Tasks: `data/eval_inputs/bird_train_tool_interface_ablation200.jsonl`
- Distribution: 80 easy, 60 medium, 60 hard; 57 databases
- Model: `deepseek-v4-flash`
- Sampling: one complete attempt per task
- Budget: 30 semantic actions; at most 3 errors of one class
- Context: rolling legal history, last 4 successful turns
- Judge: `strict-multiset`
- Gold SQL: hidden from the model and used only for terminal denotation
- Parser: strict, with no JSON/XML repair

The SQL prompt contains only `execute_sql` and `submit_sql`. Its provider-specific example is an
`execute_sql` call. It contains none of the typed-tool or retired tool names. The typed arm uses the
current `v2i-state-only-join-feedback-r2` protocol. The earlier SQL run whose provider example
mentioned `describe_table` is excluded.

The actual model inputs contain exactly one system-prompt variant per arm. Their SHA-256 hashes are
`75c8051b0249a80ba6dbc9a080095a564ab50d160ab34383e245c528ebd00d47` (SQL) and
`450f1fd8e702cac3149f6dd67f6ba82d02b097148dc5876982f92795880b164e` (typed tools).

## Main result

| Interface | Correct | Legal terminal | Correct among legal | Avg. actions | Total tokens |
|---|---:|---:|---:|---:|---:|
| Iterative SQL + feedback | 98/200 (49.0%) | 172/200 (86.0%) | 98/172 (57.0%) | 6.99 | 3,268,107 |
| Current typed tools | 90/200 (45.0%) | 166/200 (83.0%) | 90/166 (54.2%) | 8.23 | 6,346,367 |

Paired outcomes:

| Both correct | SQL only | Typed only | Neither |
|---:|---:|---:|---:|
| 67 | 31 | 23 | 79 |

The paired difference is +8 tasks for SQL, but exact McNemar `p = 0.341`. On 200 tasks this does not
establish that SQL is intrinsically more accurate. It does establish that the typed interface does
not currently improve Flash pass@1, and costs about 1.94x as many tokens.

### Difficulty

| Difficulty | N | SQL | Typed tools |
|---|---:|---:|---:|
| Easy | 80 | 48 (60.0%) | 41 (51.3%) |
| Medium | 60 | 35 (58.3%) | 31 (51.7%) |
| Hard | 60 | 15 (25.0%) | 18 (30.0%) |

Typed tools are not uniformly weaker: they are +3 on hard tasks and perform better on gold queries
with `GROUP BY` (11/32 versus 6/32) or `ORDER BY ... LIMIT` (10/36 versus 7/36). Most of their net
loss is on easy and medium tasks.

## Failure boundary

| Final outcome | SQL | Typed tools |
|---|---:|---:|
| Correct | 98 | 90 |
| Legal but wrong denotation | 74 | 76 |
| Protocol-error termination | 23 | 32 |
| Argument-validation termination | 3 | 1 |
| Execution-error termination | 2 | 1 |

API transport failures did not determine the result. There were 6 SQL and 8 typed transport retries,
with zero context retries and no terminal API failures.

The dominant per-turn protocol problem in both arms is a missing canonical `<think>` block after the
DeepSeek split-response adapter rejects a nonconforming carrier. Typed tools had 185 protocol-error
events in 1,646 turns (11.2%); SQL had 152 in 1,397 turns (10.9%). The per-turn rates are nearly the
same. Typed tools lose more complete episodes partly because their trajectories are longer.

## Concrete tool-design findings

### 1. Terminal evidence silently returns extra columns

Of 76 legal typed-tool wrong answers, 50 have a different output arity from gold. At least 26 sampled
outputs contain every gold value in order but also contain unwanted columns. These are strong
missing-final-projection cases, not failures to locate the answer.

Example `bird_train_05102`: the agent correctly filters Chicago Lawn and reads Brian P. McDermott and
the correct email. It cites `filter_001`, whose row has all 10 District columns. The harness scores
that entire row against the requested two columns, so the result is wrong. The SQL arm submits the
same correct two-column query.

Example `bird_train_01320`: the agent correctly identifies publishers with one game, but cites a
four-column joined table instead of projecting `publisher_name`. Again, the factual search is mostly
correct and the terminal shape is wrong.

This is the largest actionable loss. The current prompt says that `answer_from_context` reads the
evidence table, but does not make the exact-denotation consequence prominent enough. Either require
an explicit final `project`, or let the terminal action specify validated output columns and have the
harness create the final projection. Do not score free-text `reason` as evidence.

### 2. Free-text answer and evidence can disagree

Example `bird_train_05033`: the model's reason correctly names Uganda and lists its ethnic groups,
but the cited evidence table contains country code `EAU`, not full country name. The harness correctly
ignores the ungrounded prose and returns the table. The safety boundary is right; the interface makes
it too easy to believe prose repairs a malformed evidence table.

Example `bird_train_00625`: the computed average is `144.189681...`, but the model places string
`"144.19"` in the scalar answer. Strict denotation rejects the rounded string. Scalar terminals need
clearer typed-value guidance and should preferentially cite the computed scalar step.

### 3. Validation does not cover the full argument grammar

Ten typed execution errors surface internal `KeyError` or `TypeError` messages for malformed table or
condition objects. Examples include `table={"name":"movie"}`, unsupported `is_null` predicates, and
SQL-like boolean condition trees. These should be rejected before execution as structured
`argument_validation_error` messages that show the accepted shape. They do not prove missing
expressivity because the same filters can be composed as multiple atomic calls, but the present error
feedback is unnecessarily cryptic.

### 4. Join identifiers remain costly

Several calls still use `table.column` or `L./R.` keys despite the prompt's prefix rules. The harness
returns useful available-column feedback and some episodes recover, but the repeated confusion is
evidence that the n-way prefix contract has high cognitive cost. A join should accept source handle +
bare source column references and let the harness own all materialized output naming.

### 5. The SQL control also has avoidable friction

SQL produced 97 argument-validation events because `submit_sql.sql` had to be byte-for-byte identical
to a previously executed query. Models often changed a semicolon or formatting. A cleaner control is
`submit_sql(step_id)`, where the harness submits a successful workspace query by immutable id. The
current strict rule biases the SQL result downward, although 53 SQL tasks still recovered after at
least one error.

## Model semantic failures

Not all misses are tool defects:

- `bird_train_05717`: typed tools follow misleading external knowledge (`> 8000`) instead of the
  question (`>= 80000`) and return 172 rather than 2.
- `bird_train_01898`: typed tools inspect the wrong party field and return only one independent
  legislator instead of three.
- `bird_train_01607`: typed tools compute the average star threshold over all businesses instead of
  active Goodyear businesses.
- SQL also has 74 legal semantic failures, and both interfaces fail 79 of the same tasks. These shared
  failures are the strongest evidence of Flash's reasoning ceiling under pass@1.

## Conclusion

The present typed tool set is not catastrophically below iterative SQL: the paired gap is 4 points
and is not significant on this sample, while typed tools do slightly better on the hard stratum.
However, the current implementation leaves measurable performance on the table through terminal
shape ambiguity, incomplete argument validation, join-name complexity, longer trajectories, and a
large prompt/token footprint.

The first repair should be terminal output shaping plus strict validation, followed by a focused
replay of the 31 SQL-only cases. If the 15 clear SQL-only projection-superset cases recover without
hurting typed-only cases, the interface gap largely disappears without adding a new relational
operator. Only after that should a stronger teacher or pass@k be used to estimate the remaining model
ceiling.

## Artifacts

- SQL results: `data/results/bird_train_iterative_sql_feedback_flash_ablation200_cleanprompt/`
- Typed all attempts: `data/trajectories/bird_interface_ablation200_typed_flash_all.jsonl`
- Typed successes: `data/trajectories/bird_interface_ablation200_typed_flash_success.jsonl`
- Typed failures: `data/trajectories/bird_interface_ablation200_typed_flash_failures.jsonl`
