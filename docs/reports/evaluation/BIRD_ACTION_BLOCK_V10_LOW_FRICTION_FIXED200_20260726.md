# BIRD action-block v10 low-friction fixed-200 evaluation (2026-07-26)

## Decision

Action-block v10 is an execution-efficiency improvement, not an accuracy promotion.

Against the paired version24 one-tool baseline under `bird-set`, DeepSeek v4 Flash scored
143/200 versus 145/200. The paired result has 10 gains and 12 regressions
(`p=0.8318119049072266`, exact two-sided binomial). The nominal -2 is not significant, but there is
also no evidence of an accuracy gain.

V10 reduced model round trips by 38.4%, API requests by 37.4%, and total tokens by 45.8%. It
removed all eight previously audited batch-reference errors in two frozen-20 repeats. However,
the fixed-200 run still had 30 process errors versus the baseline's 29, because schema
hallucinations, malformed actions, and invalid relational arguments replaced the eliminated
reference errors.

Do not make v10 the default and do not use it as an SFT source.

## Scope

Both runs use the same frozen 200 BIRD-train tasks, DeepSeek v4 Flash, greedy decoding, thinking
enabled with high reasoning effort, rolling legal history of four turns, and `bird-set`
denotation. Gold SQL is hidden from the model and used only for scoring.

Compared runs:

- `baseline`: version24, one atomic tool per model turn.
- `v10`: the action-v4 visible prompt byte-for-byte, with deterministic environment-side
  resolution enabled. The system-prompt SHA256 is
  `2e12ccb8492d52339f415cab41d4e584f353bf0682d9bfb0b484c49a22a657c7`; the v10 protocol hash is
  `bafd9fd60dfdced5`.

The environment records every normalization in `interface_resolution_events`; original and
resolved arguments remain separate. It does not use gold SQL, task-specific rules, or model
reasoning as factual authority.

## Deterministic interface rules

V10 accepts four additional structurally grounded cases:

1. A stale relation prefix such as `Paper.Id` is mapped to `filter_001.Id` only when the current
   introduced relation has exactly one matching suffix.
2. A successful `$call_id` from a prior block resolves to its audited resident table or step
   binding when the id has not been shadowed in the current block.
3. A same-block call id written without `$` in a table or `value_ref` position resolves to the
   earlier successful call.
4. Multiple same-suffix columns resolve only when an inner-join equality derivation proves all
   candidates equivalent.

Ambiguous, missing, or non-equivalent candidates remain errors.

## Frozen-20 gate

| Run | Correct | Legal | Process errors | Blocked | Mean model turns | Total tokens |
|---|---:|---:|---:|---:|---:|---:|
| action-v4 | **16/20** | 20/20 | 9 | 10 | 4.75 | 450,896 |
| v10 repeat 1 | 15/20 | 20/20 | 6 | 4 | **4.05** | **379,201** |
| v10 repeat 2 | 13/20 | 20/20 | **5** | **3** | 4.80 | 461,323 |

The eight v4 interface-friction errors—four stale derived namespaces, two expired cross-block
local ids, one missing same-block `$`, and one ambiguous joined bare column—were absent in both
v10 repeats. Accuracy nevertheless fell in both samples. The regressed task identities were not
stable across repeats, showing substantial provider/policy trajectory variance even with
temperature zero.

## Fixed-200 aggregate result

| Metric | version24 baseline | action-v10 | Change |
|---|---:|---:|---:|
| Correct (`bird-set`) | **145/200 (72.5%)** | 143/200 (71.5%) | -2 / -1.0 pp |
| Legal termination | 197/200 | **198/200** | +1 |
| Process errors | **29** | 30 | +1 |
| Mean model turns | 7.45 | **4.59** | -38.4% |
| Mean primitive actions | **7.45** | 9.04 | +21.3% |
| API requests | 1,509 | **944** | -37.4% |
| Total tokens | 8,420,861 | **4,567,874** | -45.8% |

V10's two non-legal records are one exhausted `max_atomic_actions` episode and one API disconnect.
The API task (`02052`) was retried once as a complete fresh episode; it again disconnected after
11 successful semantic tool actions, so it remains an external failure rather than being relabeled
as a policy answer.

V10 executed 1,808 attempted primitive actions. Its 30 process errors are 13 execution errors,
9 argument-validation errors, and 8 top-level protocol errors. Twelve dependency descendants were
blocked and not attempted.

## Interface-resolution audit

The environment performed 92 deterministic resolutions across 63 tasks:

| Rule | Events |
|---|---:|
| Unique column suffix | 72 |
| Missing same-block `$` for a step reference | 13 |
| Prior-block successful call binding | 5 |
| Inner-join-equivalent columns | 2 |

Forty-five of the unique-suffix events repaired the first `join_tables.on.left` edge. Two
fixed-200 errors remained close to the interface boundary:

- `01599` used stale `Business.business_id` after both `filter_002.business_id` and
  `Business_Categories.business_id` had been introduced. V10 did not propagate their proven
  inner-join equivalence while constructing later edges.
- `02438` used `$join1.order_id` when the local join output contained two equal `order_id`
  candidates. V10 rejected it before consulting the resident join derivation.

The other errors are malformed carrier/actions, nonexistent schema columns, invalid tool
arguments, or semantic use of columns that had already been projected away.

## Paired capability comparison

The 10 v10-only correct tasks are:

- output-shape recoveries: `02512`, `03262`, `03724`, `03969`, `05316`;
- relational or semantic recoveries: `00074`, `01692`, `03636`, `03977`, `05440`.

The 12 baseline-only correct tasks are:

- output/answer-slot regressions: `00600`, `02507`, `03131`, `03373`, `05053`, `06489`;
- population or relational regressions: `00362`, `02179`, `02513`, `04189`, `04244`, `04906`.

Thus the net -2 is not one dominant operator regression: v10 loses one more task in output
shape/slot selection and one more in relational/population semantics.

## Resolution-triggered subset

| Subset | Tasks | v10 | baseline | v10-only | baseline-only |
|---|---:|---:|---:|---:|---:|
| At least one v10 interface resolution | 63 | 37 | **43** | 0 | 6 |
| No v10 interface resolution | 137 | **106** | 102 | 10 | 6 |

For the resolution-triggered subset, the paired exact p-value is 0.03125. This is a
post-treatment partition: tasks enter the subset because the v10 model authored a noncanonical
reference, while the baseline uses a different one-tool protocol. It must not be interpreted as a
causal estimate of the resolver. It is nevertheless a strong warning: auto-resolution marks
fragile trajectories and produced no observed task-level gain on this run.

One event was directly unsafe. On `02179`, the model placed `$filter_geo.LocationID` in a literal
`value` position. V10 converted it to the column-name string `LocationID`, executed an empty
filter, and returned a wrong answer without a process error. A column reference is not a grounded
cell value.

## Safe v11 follow-up

The evaluated v10 implementation is retained behind `--low-friction-interface` for artifact
reproduction. The unpromoted v11 follow-up is behind `--safe-low-friction-interface` and adds:

- rejection of `$id.column` in literal `value`/range positions, with an explicit path to
  `value_ref` or observed literals;
- progressive equivalence classes while constructing multi-edge inner joins;
- fact-proven resolution of ambiguous `$joined.column` local references.

Unit tests cover all three boundaries. Three targeted, non-combinable diagnostics were run:

- `02179`: v10 wrong with zero errors; v11 raised one recoverable literal-reference error, replanned,
  and answered correctly.
- `02438`: v10 wrong with two errors; v11 removed the errors but remained semantically wrong.
- `01599`: v10 correct after one join error; v11 had zero errors but was wrong on a different
  sampled semantic path.

These targeted runs validate the safety boundary but are not a v11 fixed-200 score.

## Interpretation

The action-block hypothesis is only partially supported:

- Parallel action blocks substantially compress model/API round trips and prompt-token cost.
- Deterministic reference resolution removes the specific batch-interface failures it targets.
- Neither change improves end-to-end denotation by itself.

In the atomic baseline, an execution error often creates an observation boundary that forces the
model to reconsider its population, grain, or output slots. V10 removes that boundary and continues
executing already-authored descendants. This can turn a recoverable interface mistake into a fast,
error-free semantic failure.

The next design should preserve the successful resolution without silently consuming the rest of
the branch. A promising environment-owned policy is:

1. Continue automatically for missing `$` and exact prior table/step bindings.
2. Reject column references used as literal values.
3. When namespace/equivalence resolution changes a relational column identity, execute the
   corrected producer but defer its dependent descendants as `deferred_after_resolution`, not
   `blocked` or `error`.
4. Return the canonical binding and resume from a new model observation boundary.

This tests whether the useful part of the old error—adaptive replanning—can be retained without
charging an interface error. Gate it first on a resolution-heavy paired subset before another
fixed-200 run.

## Artifacts

- `data/trajectories/batch_plan_20260726/action_block_v10_low_friction_pilot20_r1.all.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v10_low_friction_pilot20_r2.all.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v10_low_friction_fixed200_r1.all.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v10_vs_version24_fixed200.paired.json`
- `data/trajectories/batch_plan_20260726/action_block_v10_low_friction_fixed200_api_retry_02052.all.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v11_target_02179.all.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v11_target_02438.all.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v11_target_01599.all.jsonl`
