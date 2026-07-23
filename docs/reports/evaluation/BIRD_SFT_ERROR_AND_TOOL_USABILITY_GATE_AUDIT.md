# SFT Error Audit and Frozen-200 Tool-Usability Gate

Date: 2026-07-23

## Decision

Do **not** construct new SFT data from version12/version13 yet.

The last completed fixed-200 control is version11 at **130/200 = 65.0%** under
`strict-multiset`. Only four terminal failures are protocol/execution failures, so perfect recovery
of those failures would reach at most 134/200. Reaching the requested **150/200 = 75.0%** therefore
requires at least 16 additional semantic-answer gains, not another join-parser repair.

Version12 and version13 are useful interface experiments, but their fixed-prefix 30-task pilots did
not pass the small-sample expansion gate:

| Contract | Correct | Legal | Final failures |
| --- | ---: | ---: | --- |
| version11 control on the same first 30 tasks | 20/30 | 30/30 | 10 wrong |
| version12: distinct projection + scalar compute | 21/30 | 30/30 | 9 wrong |
| version13: exact grounded terminal table | 20/30 | 27/30 | 7 wrong, 3 protocol |

The version13 protocol failures were provider-carrier failures, dominated by 20 empty visible
responses. They were not rejections of the exact-evidence terminal shape. Even an optimistic
counterfactual in which all three terminal protocol failures recover gives only 23/30, below a
24/30 small-pilot threshold for a 75% target.

Version13 remains the current implementation because its exact-evidence terminal improves the
provenance contract, but it is not accuracy-qualified training-data infrastructure. Version11 is
the accuracy control.

## Audited Artifacts

The streaming audit reader is `src/eval/audit_rollout_errors.py`. It reads flat rollout attempts,
pass@K `samples[]`, and normalized accepted trajectories. Raw visible `"tool"` extraction is used
only for error attribution; the audit never repairs or replays a rejected action.

The audited cohorts are:

- SFT1 accepted causal trajectories:
  `data/trajectories/bird_sft1_grounded_v4d_all651.jsonl`;
- SFT2 raw student attempts:
  `data/results/bird_sft2_student_full1000_k4_canonical_all.jsonl`;
- SFT2 raw Flash fallback attempts:
  `data/trajectories/bird_sft2_flash_fallback_full1000_all368_all.jsonl`;
- fixed-200 version11 attempts:
  `data/trajectories/bird_train_version11_join_validation200_flash_rolling_json_unambiguous_success.all.jsonl`.

Generated audit JSON files live beside those inputs under `data/` and remain ignored experiment
artifacts.

## Historical SFT Error Counts

| Cohort | Episodes | Correct | Protocol events | Argument events | Execution events |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT1 accepted | 651 | 651 | 441 | 6 | 23 |
| SFT2 raw student | 4,000 | 1,745 | 360 | 205 | 1,593 + 2 nonrecoverable |
| SFT2 Flash fallback | 368 | 38 | 413 | 25 | 46 |
| version11 fixed 200 | 200 | 130 | 18 | 10 | 15 |

The accepted SFT1 file is selection-conditioned: all 651 episodes are verified successes. Its
errors show recovery behavior, not the failure rate of the original teacher population. All 306
episodes with a protocol event, 21 with an execution event, and six with an argument event
eventually recovered.

### Raw SFT2 student errors by attempted tool

The 1,593 execution events concentrate in:

| Tool | Execution events |
| --- | ---: |
| `condition_filter` | 604 |
| `join_tables` | 419 |
| `project` | 170 |
| `describe_table` | 112 |
| `inspect_column` | 110 |
| `group_aggregate` | 86 |
| `extreme_value_select` | 77 |
| `set_op` | 10 |

Root-cause clustering is more informative than the tool name:

- 1,128 column-reference failures;
- 140 unknown table/handle failures;
- 99 bad scalar/SQL-expression failures;
- 67 multi-column `in_table` failures;
- 41 predicates missing a value;
- 30 non-scalar `value_ref` failures;
- 19 predicates missing a column;
- 19 malformed historical join-shape `ValueError` failures;
- seven set-operation alignment failures.

The 360 protocol events include 77 unknown-tool calls. Most are variants of `compute`,
`compute_expression`, `compute_scalar`, percentage calculation, or Python scalar calculation. This
is direct evidence that the old action space lacked a natural grounded arithmetic atom; it motivated
version12 `scalar_compute`.

### Raw SFT2 Flash fallback errors

The fallback cohort is dominated by the retired carrier:

- 286 missing/invalid `<think>` events;
- 121 missing/invalid tool-call events;
- six invalid JSON events.

Its 46 execution events are mainly 26 column-reference failures, with 20 attributed to
`join_tables`. These failures describe the older prompt/carrier and prefix contracts; they are not
the current join-local rate.

## What Version11 Already Fixed

On the fixed 200, version11 has:

- 130 correct and 196 legal episodes;
- three final protocol failures and one final execution failure;
- 100 canonical join attempts, 94 executed joins, and **94.0% join-local success**;
- six join errors, all caused by adding a derived handle in front of an already-flat logical
  namespace;
- seven condition execution errors: three multi-column `in_table` cases and four invalid scalar
  `value_ref` cases;
- two project execution errors.

The historical carrier and column-interface problems have therefore fallen sharply. Current error
events also recover frequently: six correct episodes recovered after an execution error, five after
an argument error, and three after a protocol error.

The remaining 18 version11 protocol events are 12 empty visible responses, three non-JSON visible
responses, and three wrong top-level JSON shapes. These are provider/model carrier events after the
prompt was reduced to one explicit JSON-output contract; they are not evidence that another
relational operator is missing.

## The 66 Version11 Wrong Answers

The stored samples divide into:

| Observable mismatch | Tasks |
| --- | ---: |
| Same-width numeric mismatch | 23 |
| Same-width text mismatch | 12 |
| Same-width multi-column/row mismatch | 12 |
| Prediction has too few columns | 10 |
| Prediction has extra columns | 8 |
| Empty prediction against non-empty gold | 1 |

Thus 18/66 wrong answers have an obvious output-width mismatch. Examples include concatenating
first/middle/last name into one cell, preserving ID/ranking helper columns, returning a product name
instead of its ID, and citing a grouped table rather than its requested output column. This
motivated exact-output wording, `project(distinct=...)`, and the version13 table-only terminal.

The other failures are mostly query semantics: wrong aggregation grain, wrong denominator, selecting
all ties instead of SQL `LIMIT 1`, dirty-type ordering, joining the wrong relation instance, or
changing a stored code/case into a natural-language label. Gold queries containing casts, CASE,
ordering, limits, and grouping have markedly lower version11 accuracy than the full 65% cohort.

Some strict failures are underdetermined or internally inconsistent from model-visible input:

- `bird_train_01152` says in external knowledge that “player name refers to playerID,” while the gold
  query returns first, middle, and last name;
- `bird_train_06492` describes `COUNT(OBJ_SAMPLE_ID) < 15`, while the gold query counts rows whose
  `OBJ_SAMPLE_ID < 15` and does not group by image;
- `bird_train_04598` has no model-visible distinctness cue, while the gold query uses `DISTINCT`;
- `bird_train_05952` asks for one county name, while `strict-multiset` requires two identical rows.

These examples do not license changing the fixed metric after seeing results. They do show why a
75% tool-usability claim must name its metric and why a separate BIRD-reference `bird-set` report is
needed before interpreting duplicate-only misses as tool failures.

## Cross-Design Stability

The same 200 task IDs exist in the version4, version6, version7, version8, version10, and version11
artifacts. Their descriptive correct counts are 125, 121, 89, 106, 121, and 130. Provider carrier
and prompt controls differ across some rows, so this is a stability audit, not a clean six-arm
causal estimate.

Number of tasks correct in exactly N of those six runs:

| Successful runs out of six | Tasks |
| ---: | ---: |
| 0 | 50 |
| 1 | 9 |
| 2 | 10 |
| 3 | 14 |
| 4 | 19 |
| 5 | 43 |
| 6 | 55 |

Fifty tasks fail under all six designs. Of the 70 version11 misses, 20 succeeded under at least one
other design. The post-hoc union of every success from all six runs is exactly **150/200**.
Consequently, a single pass@1 design reaches 75% only if it retains every version11 success while
recovering all 20 interface-sensitive misses, or also solves some of the 50 universal failures.
Neither 30-task pilot demonstrated that no-regression behavior.

## Tool Changes Worth Keeping or Isolating

1. **Keep the version11 join shape and flat logical namespace.** Its local execution rate is already
   94%. Making join binary again would lengthen trajectories without addressing the dominant
   terminal errors.
2. **Keep exact-evidence terminal semantics as a provenance experiment.** It prevents a model from
   bypassing an exact table by manually concatenating, translating, or rounding answer values. The
   version13 pilot proved the model can use the shape, but did not prove an accuracy gain.
3. **Retain grounded scalar arithmetic, but isolate it in evaluation.** Historical SFT attempts
   explicitly ask for it, yet the version12 30-task pilot improved by only one task. It also exposes
   a separate missing row-wise typed expression for cases such as date differences over multiple
   rows.
4. **Evaluate explicit source-column selectors for subqueries.** A condition leaf such as
   `in_table + source_column` and `value_ref + source_column` would directly address the historical
   67 + 30 condition failures. On version11 these events are mostly recovered and do not explain the
   20-point accuracy gap, so this is an ergonomics change, not the next 75% lever.
5. **Consider safe join-prefix normalization only as bounded recovery.** Stripping a derived handle
   prefix is safe only when the remainder resolves to one exact logical column. It can remove the
   six current join errors, but those errors are mostly recovered and fixing every terminal
   protocol/execution failure still leaves at least 16 required semantic gains.
6. **Do not expand the public surface with general SQL escape tools.** `project` already behaves as
   an untyped scalar-expression escape hatch. The next expression change should be a narrow typed
   row operation, not another open-ended SQL field.

## Next Gate

Before any new SFT construction:

1. separate the version13 terminal-only change from `project(distinct)` and `scalar_compute`, so one
   variable is tested at a time against version11;
2. use the same DeepSeek v4 Flash request controls, rolling history, action/error budgets, and
   `strict-multiset` judge;
3. run a small fixed diagnostic first and expand only if it reaches at least 24/30 without
   regressing version11 successes;
4. run the frozen 200 only after that gate;
5. require at least 150/200 on the frozen strict-multiset result, while separately reporting
   provider/API failures and a BIRD-reference `bird-set` score;
6. replay every accepted success and require exact reference/backward-slice checks before it can
   become an SFT source.

Until those conditions are met, the correct result is “tool design improved but the 75% usability
gate is not passed,” not a new training mixture.

## Verification

At the audited HEAD:

- active harness: 60/60;
- SFT unit tests: 65/65;
- RL unit tests: 32/32;
- eval tests: 24/24.

