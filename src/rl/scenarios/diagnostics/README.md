# Diagnostic scenarios

## Cross-run static rollout audit

`audit_rollout_corpus.py --sources inventory.json --output NEW_DIRECTORY` composes
the shared `rl.diagnostics.rollout_corpus` statistics and streaming JSONL I/O.
Inputs identify immutable rollout/run-manifest snapshots and the source-local IDs
of previously reviewed questions. Question partitions use database plus question
content, not a cohort's renumbered example index. Outputs include input hashes,
K=8 coverage, logged failures, bounded hash-selected review pairs, and a reserved
task partition. It never invents probabilities, semantic labels, or rewards.

`audit_distance_credit.py` preserves a failed development-only heuristic for audit.
It is not an admitted classifier or training entrypoint; zero feature distance
does not mean equal semantics. See the 2026-09-08 corpus report before using it.

One-off audits, cohort screens, replay comparisons, and training reports are
organized here by experiment scenario. They may depend on a frozen historical
artifact, but must import reusable I/O, ranking, reward, and environment
functions from stable modules under `rl.diagnostics`, `rl.runtime`, or
`rl.objectives`. New current experiments should add one clearly named scenario
module instead of extending the shared runtime.

## Harness-conditioned SMDP/IQL feasibility audit

`audit_harness_smdp_iql.py` consumes a flat Atomic v26 `rollouts.jsonl` and
writes an offline transition dataset plus a diagnostic manifest. It keeps only
the exact model-visible `model_input`, typed tool action, Harness terminal
reward, and episode outcome; reference fields such as `gold_sql` are rejected.
The output is diagnostic only and never updates the actor.

The audit must show repeated exact states with both action variation and outcome
variation before a critic is trained. Otherwise an IQL critic has only learned
a trajectory-success baseline, which is not evidence that it can concentrate
pass@k correctness into greedy behavior.
