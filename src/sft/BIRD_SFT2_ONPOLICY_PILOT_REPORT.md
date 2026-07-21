# BIRD SFT-2 On-policy Construction Pilot

Date: 2026-07-19

## Protocol

The frozen construction contract is `draft/bird_sft2_onpolicy_data_protocol.md`.  SFT-1 student
pass@K always runs first.  External teacher generation is restricted to tasks with no correct
student sample after K.  Flash is the default teacher; Pro is a matched-ID fallback after an
explicit quality gate.  All accepted actions are strict-parser, database-replay, and terminal-
denotation verified.  Error actions and bad student actions remain audit-only.

## Student on-policy lane

Inputs are the completed easy100, medium30, and hard30 K=4 BIRD-train runs: 160 tasks / 640 samples.
There are 318 verifier-correct student samples.  Fresh harness replay reconstructs authoritative
state and provenance for all 318; no sample is lost to replay.  Quality gates reject three episodes
with repeated identical calls and one with overlong reasoning.  The accepted pool contains:

- 314 student-success episodes;
- 1,953 last-turn-only targets;
- 45 Feedback Recovery targets (a tagged subset, not duplicated records);
- difficulty targets: easy 1,222, medium 473, hard 258.

Exact Qwen2.5/LLaMA-Factory token audit with `mask_history=true`, Qwen template, and cutoff 6400
retains 1,953/1,953 targets.  Encoded length min/p50/p90/max is 1670/2752/3481/6311.

## Historical error replay finding

Nine recovered student episodes initially failed offline reconstruction.  The historical v2i
executor increments its hidden materialized-handle counter before SQL validation, even when the
model-visible EnvironmentState remains unchanged.  For example, a failed operation consumes
`filter_001` and the next legal output is `filter_002`.  Canonical replay now re-executes audited
`execution_error` events only to reproduce this historical counter progression and verifies that
the visible state remains unchanged.  No argument, JSON, handle, or stored action is rewritten.
A regression test covers this behavior.

## Flash fallback pilot

Six tasks were selected deterministically from pass@4 failures: two easy, two medium, and two hard.
DeepSeek v4 Flash ran one strict closed-loop attempt per task with rolling-4/full/resident context,
2048 output tokens, and no parser relaxation:

- 1/6 verifier-correct (hard);
- 3/6 legal wrong answers;
- 2/6 protocol-error terminals;
- zero API transport or context retries.

The one successful branch has 10 verified teacher-fallback actions.  Exact token audit retains all
10 at cutoff 6400.  This is enough to validate the pipeline but not enough to estimate teacher
quality reliably.

## Decision Correction

Teacher and failed-student branches are paired at an exact shared state.  `describe_table` table
order is treated as semantically unordered, preventing a false correction based only on argument
order.  Mere action reordering is also insufficient: the bad student action requires a deterministic
diagnosis.  The pilot admits one correction:

- student inspected `Business_Hours` and irrelevant base table `Users`;
- teacher instead described `Business` and `Business_Hours`;
- diagnosis: `extraneous_base_table:Users`;
- complete teacher branch replay and terminal denotation pass;
- the student action is audit-only and never a target.

The correction target passes strict/no-leak validation and exact 6400-token audit.

## Candidate mixture

`bird_sft2_onpolicy_pilot160_flash6_mixture.jsonl` contains 2,613 unique records:

- 650 SFT-1 demo replay targets;
- 1,908 ordinary student-success targets;
- 45 Feedback Recovery targets;
- 9 non-duplicate teacher-fallback targets;
- 1 Decision Correction target.

The correction copy has priority over its identical teacher-fallback record, so it appears once.
Canonical records are not duplicated to force ratios; lane and transition tags support later
sampler weighting and ablation.

## Scale decision

The construction pipeline is operational, but the correction lane is too small for SFT-2 training.
The frozen fallback cohort was therefore expanded to 20 pass@4-failed tasks (8/6/6 easy/medium/hard,
14 databases), retaining the original six IDs.  Flash was resumed without reissuing the first six:

- Flash: **3/20 = 15%** verified, 14 legal-wrong, 3 protocol-terminal;
- Pro on the exact same 20 IDs: **2/20 = 10%** verified, 17 legal-wrong, 1 protocol-terminal;
- paired union: four raw successful tasks; Pro adds one success not solved by Flash;
- after max-20-step/max-300-word/repetition quality gates: three unique teacher episodes / 24 targets
  remain (one Flash success rejected for overlong reasoning, one duplicate cross-provider success).

Thus Pro improves carrier reliability but not semantic yield on this cohort, and should not replace
Flash globally.  The economical policy remains student K=4 first, Flash fallback second, and Pro
only as a rescue on Flash failures.  Even this two-provider union produces only three training-grade
episodes from 20 student-failed tasks, so hundreds of fallback calls should not be launched yet.

The three accepted teacher branches produce two deterministic Decision Correction targets; the
third has no semantically divergent action at a shared state and is correctly rejected.  The updated
deduplicated mixture `bird_sft2_onpolicy_pilot160_teacher20_mixture.jsonl` has **2,627** records:
650 SFT-1 replay, 1,908 normal student successes, 45 recoveries, 22 non-duplicate teacher fallback
targets, and two corrections.  Exact token audit keeps teacher 24/24 and correction 2/2 at cutoff
6400.  This is a validated construction artifact, not yet a correction-rich final SFT-2 corpus.

## Flash-only completion of the residual set

Following the decision to prefer Flash because Pro showed no semantic advantage, the teacher cohort
was expanded from the matched 20 to **all 53** SFT-1 student pass@4 failures.  The original Flash20
audit records were reused unchanged and only the remaining 33 tasks were requested.  The residual
on-policy distribution is easy/medium/hard 32/8/13 across 31 databases.

- Flash raw verifier success: **6/53 = 11.32%**;
- terminal failures: 42 legal-wrong and 5 protocol-terminal;
- API transport/context retries: zero;
- max-20-step/max-300-word/repetition quality gates retain **5 episodes / 35 targets**;
- one raw success is rejected by the think-length gate;
- accepted difficulty episodes: easy 3, medium 1, hard 1;
- accepted teacher targets include one Feedback Recovery action.

Shared-state deterministic diagnosis still admits only two Decision Corrections; three newly
accepted teacher branches have no provably bad student action at an identical state and are not
forced into the correction lane.  The final Flash-only candidate mixture
`bird_sft2_onpolicy_pilot160_flash_all53_mixture.jsonl` has **2,638 unique records**: 650 SFT-1
replay, 1,908 ordinary student-success, 45 student Feedback Recovery, 33 non-duplicate Flash
fallback, and two Decision Correction targets.  Strict parser/uniqueness tests pass; exact
Qwen2.5/LLaMA-Factory cutoff-6400 audit retains Flash 35/35 and correction 2/2, with maximum encoded
lengths 4634 and 2197.  All student pass@4 failures in this 160-task pilot have now received one
Flash teacher attempt; no further Pro generation is planned for this pool.
