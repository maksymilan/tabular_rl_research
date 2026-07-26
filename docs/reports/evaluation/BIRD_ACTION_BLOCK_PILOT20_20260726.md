# BIRD action-block frozen-20 diagnostic (2026-07-26)

## Scope

This diagnostic uses the same frozen 20 BIRD-train tasks, DeepSeek v4 Flash, greedy decoding,
thinking enabled with high reasoning effort, rolling legal history of four turns, and the
`bird-set` denotation metric. Gold SQL was hidden from the model and used only for terminal
scoring.

Compared runs:

- `baseline`: original version24 one-tool-per-turn trajectories, 16/20.
- `batch-v1`: first batch-plan pilot (`batch_plan_v1_pilot20_r1`), 14/20.
- `action-v2`: hybrid action blocks with environment-derived local dependencies and branch-local
  recovery, 15/20.
- `action-v4`: action-v2 plus grounded terminal column selection and concise relational
  invariants, 16/20.

The intervening action-v3 run scored 14/20 with one provider-carrier failure. Its target shape
fixes worked, but it also exposed a cross-table `column_value` misuse and a population-ordering
failure. Action-v4 corrected those general invariants and reran all 20 tasks rather than replacing
selected failures.

## Aggregate result

| Run | Correct | Legal | Mean model turns | Mean atomic actions | Process errors | Blocked nodes | Total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 16/20 | 20/20 | 6.35 | 6.35 | 1 | n/a | 690,165 |
| batch-v1 | 14/20 | 19/20 | 5.15 | 13.65 | 45 | 32 | 583,494 |
| action-v2 | 15/20 | 20/20 | 4.40 | 8.35 | 8 | 15 | 398,665 |
| action-v4 | **16/20** | **20/20** | 4.75 | **8.15** | 9 | **10** | 450,896 |

The batch-v1 process-error count mixed attempted failures with dependency-blocked descendants.
Action-v2/v4 count only attempted root failures; blocked descendants are reported separately and
do not consume the primitive-action or error budget.

Relative to batch-v1, action-v4:

- retains every one of batch-v1's 14 correct tasks and additionally solves `00593` and `05149`;
- raises legal termination from 19/20 to 20/20;
- reduces mean atomic actions by 40.3%;
- reduces recorded process errors from 45 to 9 under the corrected root/blocked boundary;
- reduces total tokens by 22.7%.

Relative to the original baseline, action-v4 has the same 16/20 accuracy. The paired result is
15 both correct, 3 both wrong, 1 action-v4-only (`00593`), and 1 baseline-only (`06489`). It uses
25.2% fewer model turns and 34.7% fewer total tokens, while executing more relational primitives
inside those turns.

## Why action-v4 has eight more intermediate errors

The baseline has one attempted execution error on the frozen cohort. In `02935`, it cited a
non-scalar-producing step as a filter `value_ref`; the harness rejected it and the next turn
recovered.

Action-v4 has nine attempted root errors across six tasks. Ten additional calls were dependency
blocked and never executed, so they are not included in the nine:

| Task | Root errors | Exact cause | Recovered terminally? |
|---|---:|---|:---:|
| `03390` | 1 | After filtering `language`, join used stale `language.Country` instead of the actual derived namespace `filter_001.Country`. | yes |
| `06151` | 2 | First used stale `Paper.ConferenceId`; the next block then reused expired local id `$filter_paper` instead of resident handle `filter_001`. | yes |
| `00593` | 1 | `scalar_compute` cited a two-row medication table rather than one of the later one-row tables. | yes |
| `06171` | 1 | After filtering `Paper`, join used stale `Paper.Id` instead of `filter_001.Id`. | yes |
| `01152` | 3 | Used stale `awards_players.playerID`, then expired `$roty_players` across blocks, then ambiguous bare `playerID` after a namespaced join. | no |
| `00541` | 1 | Passed same-block call id `check_roles` as a literal table handle instead of `$check_roles`. | yes |

One action-v4 error is the same general scalar-grounding class as the baseline's only error. The
remaining eight are all reference/namespace friction introduced or amplified by the block
interface:

- four stale source-table namespaces after a unary tool created a runtime derived handle;
- two block-local ids incorrectly reused in a later block;
- one missing `$` on a same-block local reference;
- one ambiguous bare column after a namespaced join.

This is why equal 16/20 terminal accuracy coexists with eight more intermediate errors. A block
reduces model round trips, but the model must author downstream arguments before seeing the runtime
handle and namespace produced by earlier calls. It also has to obey two distinct lifetimes:
`$call_id` inside the current block and an environment handle such as `filter_001` in later blocks.
Most of these failures were recoverable, so they increased process-error count without necessarily
changing the final denotation.

## Per-task comparison

| Task | baseline | batch-v1 | action-v2 | action-v4 | Diagnostic |
|---|:---:|:---:|:---:|:---:|---|
| `00541` | ✓ | ✓ | ✗ | ✓ | v2 cited `person_id,birth_place`; terminal column selection restores exact `birth_place`. |
| `00593` | ✗ | ✗ | ✓ | ✓ | Batch-v1 exhausted 30 actions; hybrid execution preserves both 11- and 18-day rows. |
| `01152` | ✗ | ✗ | ✗ | ✗ | Persistent output-slot conflict: task knowledge maps name to ID, while gold requires first, middle, last names. v4 improves to first+last but misses middle. |
| `02437` | ✓ | ✓ | ✓ | ✓ | Stable in final runs. v3 transiently used cross-table `column_value`; v4's same-input invariant restored the join/filter path. |
| `02605` | ✓ | ✓ | ✓ | ✓ | Stable. |
| `02935` | ✓ | ✓ | ✓ | ✓ | Stable; v4 removes v2's recovery errors. |
| `03275` | ✓ | ✓ | ✓ | ✓ | Stable. |
| `03390` | ✓ | ✓ | ✓ | ✓ | Stable. |
| `04189` | ✓ | ✓ | ✓ | ✓ | Stable. |
| `04377` | ✓ | ✓ | ✓ | ✓ | Stable. |
| `04619` | ✓ | ✓ | ✓ | ✓ | Stable. |
| `04636` | ✓ | ✓ | ✓ | ✓ | Stable. |
| `05149` | ✓ | ✗ | ✓ | ✓ | Batch-v1 selected a different turkey recipe; hybrid schema/value feedback finds recipe 1074. |
| `05440` | ✗ | ✗ | ✓ | ✗ | v2 joined Paper to Journal before ranking and succeeded. v4 used a left join, retained unmatched high-year papers, and returned a null homepage. |
| `05952` | ✓ | ✓ | ✓ | ✓ | Stable; action-v4 completes without a root error. |
| `06026` | ✗ | ✗ | ✗ | ✗ | Persistent question/data ambiguity. All policies sum repeated West/South profit rows to 48.392; gold selects distinct South profit 33.8744. |
| `06151` | ✓ | ✓ | ✓ | ✓ | Stable. |
| `06171` | ✓ | ✓ | ✓ | ✓ | Stable. |
| `06336` | ✓ | ✓ | ✗ | ✓ | v2 retained ranking helper `occurrences`; terminal column selection returns only `word,wid`. |
| `06489` | ✓ | ✗ | ✗ | ✗ | Model interprets “object” as class `paper` despite external knowledge mapping it to `OBJ_SAMPLE_ID`; this is semantic-slot selection, not block scheduling. |

## What was fixed

The current method fixes the main engineering failures of the first batch implementation:

1. The model no longer maintains a resident DAG or mandatory plan status. It emits ordered calls;
   the environment derives dependencies from local references.
2. Attempted root errors and unattempted dependency descendants are separate. Independent branches
   continue, successful outputs remain reusable, and blocked calls do not consume action/error
   budget.
3. Exploration and execution can share a block when arguments are already grounded.
4. Terminal `evidence.columns` makes answer-slot selection an environment-grounded projection in
   the same terminal action. It does not alter row population, grain, order, aggregation, or
   distinctness.
5. `column_value` is explicitly limited to two columns in the same input relation; cross-relation
   comparison must use a join, `in_table`, or grounded scalar reference.

These changes remove the first batch run's accuracy/legal regression on this cohort while
preserving the intended reduction in model round trips.

## Remaining boundary and next optimization

The remaining failures do not justify adding more global prompt prose:

- `06489` and `01152` need a more reliable answer-slot commitment. A teacher-data experiment can
  supervise a concise declared output contract derived causally from the question, schema, and
  external knowledge. It must not use gold SQL or later trajectory facts.
- `05440` needs population-aware join supervision: when requested outputs come from two relations,
  train joining the intended population before ranking and distinguish inner from left joins using
  fact-only unmatched-row feedback.
- `06026` should remain classified as a question/data/gold conflict unless the task context is
  enriched causally; choosing South or `DISTINCT Profit` is not entailed by the visible question.

The next meaningful gate should therefore compare causal teacher trajectories with and without the
declared output/population contract. It should not expand this 20-task prompt with task-specific
rules or treat a single greedy run as proof of a general accuracy gain.

## Rejected global evidence-gate prompt ablation

A follow-up tested the hypothesis that the prompt should explicitly require more observation and
feedback reflection before relational execution:

- action-v5 added a verbose evidence gate and asked native reasoning to identify supporting
  observations and unresolved ambiguity. The first four completed tasks all ended in provider
  carrier failure after repeated empty visible actions, so the run was stopped and is not scored.
- action-v6 shortened the gate to one targeted check at an observation boundary and restored the
  brief native-reasoning instruction. Its complete frozen-20 run scored **13/20**, with 19/20 legal
  termination, versus action-v4's 16/20 and 20/20.

| Metric | action-v4 | evidence-gate v6 | Change |
|---|---:|---:|---:|
| Correct | 16/20 | 13/20 | -3 |
| Legal | 20/20 | 19/20 | -1 |
| Mean atomic actions | 8.15 | 9.35 | +14.7% |
| Process errors | 9 | 13 | +44.4% |
| `inspect_column` calls | 10 | 20 | +100% |
| API requests | 96 | 108 | +12.5% |
| Reasoning tokens | 27,592 | 40,570 | +47.0% |
| Total tokens | 450,896 | 532,789 | +18.2% |
| Carrier retries | 0 | 9 | +9 |

The paired result has no v6-only recovery and three v4-only correct tasks: `00593`, `03275`, and
`04189`.

- `03275` constructed the correct three-column relation but selected only the country column at
  terminal time. More observation did not improve answer-slot commitment.
- `04189` changed to a left join and enlarged the answer population.
- `00593` expanded to nine model turns, attempted unsupported SQL `DATEDIFF`, then used non-step
  local ids as scalar `value_ref` and exhausted the execution-error budget.

This ablation therefore did not reduce any of the four v4 semantic failures. It increased
exploration, reasoning, invalid calls, and provider-carrier pressure while creating three control
regressions. The evidence gate is rejected and action-v4 remains the implementation default.
Targeted fact feedback or causal supervision should be tested instead of a global “explore more”
instruction.

## Fact-only error-feedback experiment

Two follow-ups tested whether the environment could reduce the reference ambiguity without adding
another global reasoning instruction or rewriting model arguments:

- action-v7 included structured error facts but also changed prompt wording and the
  `reusable_outputs` shape on successful blocks. It scored 13/20 with 19/20 legal termination, 14
  process errors, and 493,159 tokens. Because policy input changed on every block, it is a
  confounded ablation and was rejected.
- action-v8 restored the action-v4 visible prompt byte-for-byte (same system-prompt SHA256
  `2e12ccb8492d52339f415cab41d4e584f353bf0682d9bfb0b484c49a22a657c7`) and changed observations
  only after a partial failure. It returned harness-derived facts: the actual input table, exact
  suffix candidates for unresolved columns, prior binding metadata for expired local ids, the
  required `$call_id` form for a missing local-reference sigil, scalar-source shape, and exact
  reusable successful outputs.

| Metric | action-v4 | error-feedback v8 | Change |
|---|---:|---:|---:|
| Correct | **16/20** | 13/20 | -3 |
| Legal | 20/20 | 20/20 | 0 |
| Process errors | 9 | **7** | -2 |
| Blocked nodes | 10 | **8** | -2 |
| Mean model turns | 4.75 | **4.20** | -11.6% |
| Mean atomic actions | 8.15 | **7.90** | -3.1% |
| API requests | 96 | **88** | -8 |
| Total tokens | 450,896 | **402,464** | -10.7% |

The local recovery signal worked, but terminal accuracy did not. The paired result is 13 both
correct, 4 both wrong, no v8-only recovery, and three v4-only correct tasks (`00593`, `04189`,
`06171`; exact paired two-sided p = 0.25):

- `00593`: feedback identified the two-row scalar source. The model then correctly computed both
  11-day and 18-day scalar tables, but terminal evidence cited only the 11-day table.
- `06171`: feedback supplied `filter_001.Id`, and the model successfully rebuilt the three-table
  join. It then added `distinct=true`, collapsing the requested author-name population.
- `04189`: the model repaired a project-column error, but retained a left join with unmatched rows,
  producing the wrong full denotation even though the displayed first five rows matched.

The evaluated v8 implementation also emitted one inaccurate auxiliary fact on `06171`: it checked
the second join edge against only the initial base columns and labeled
`PaperAuthor.AuthorId` missing even though the first edge would introduce it. The model nevertheless
used the correct edge on the next turn, so the subsequent `distinct` decision—not that fact—caused
the scored failure. The implementation now evaluates join edges progressively and has a regression
test for this case. Because that changes feedback semantics, the corrected form is named
action-v9 and is available only through `--structured-error-feedback`; it is not an accuracy
promotion. Action-v4 remains the default.

The key result is therefore narrower than “better feedback improves ability”: fact-only feedback
can reduce local reference errors, blocked work, turns, and tokens, but it does not by itself
preserve answer completeness, row population, or deduplication semantics after recovery. Those
downstream decisions need causal supervision or an independently grounded semantic contract, not
more column hints.

## Low-friction interface follow-up

Action-v10 deterministically resolved the eight audited batch-reference failures and expanded to
the frozen 200-task cohort. It scored 143/200 versus the paired version24 baseline's 145/200, while
reducing model turns by 38.4% and total tokens by 45.8%. The paired 10 gains versus 12 regressions
are not significant. See
`docs/reports/evaluation/BIRD_ACTION_BLOCK_V10_LOW_FRICTION_FIXED200_20260726.md`.

## Artifacts

- `data/trajectories/batch_plan_20260726/action_block_v4_pilot20_r1.all.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v4_pilot20_r1.all.manifest.json`
- `data/trajectories/batch_plan_20260726/action_block_v4_pilot20_r1.paired.json`
- `data/trajectories/batch_plan_20260726/action_block_v6_evidence_gate_pilot20_r1.all.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v6_vs_v4.paired.json`
- `data/trajectories/batch_plan_20260726/action_block_v7_structured_feedback_pilot20_r1.all.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v8_error_only_feedback_pilot20_r1.all.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v8_vs_v4.paired.json`
