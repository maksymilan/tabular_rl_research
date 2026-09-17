# Module extraction status

2026-09-16 screened RL candidate pool: `rl.data_selection.passk_artifact` now owns
raw pass-k artifact reduction (input/result pairing by `(example_index, db_id,
question)`, K completeness, protocol identity) and `rl.shared.io.iter_jsonl` owns
streaming reads for multi-gigabyte results. `scenarios/data/export_passk_screen_snapshot.py`
is the thin export CLI; `screened_pool` additionally emits
`retained_non_candidate_observations.jsonl` for out-of-band outcomes. No scenario
parsing, hashing, or candidate predicate is duplicated. Covered by
`src/rl/tests/test_passk_artifact.py`; the 2026-09-12 inventory rebuilds byte-identically.

2026-09-14 fixed half-strength later-error credit is implemented in the shared
TRL mechanism/SAAM layer. The diagnostic launcher only composes shared training,
checkpoint preservation and the frozen matched evaluator. No new training loop
or GPU resource helper is introduced; existing credit modes retain their behavior.

2026-09-12 RL execution options remain in the shared TRL runner/trainer/server;
no scenario training loop was added. `docs/current/rl_performance.md` is the
launch-time performance record. The checkpointing resolver now preserves both
replicated BF16 and 4-bit historical defaults, covered by 3 CPU-only tests in
`src/rl/tests/test_trl_execution_options.py`. The 8B A100 launcher still overrides
eager execution; 8B CUDA Graph validation remains pending.

2026-09-11 screened RL candidate pool: `rl.data_selection.screened_pool` now owns
stable-ID mapping, K=8 observation merging, duplicate/conflict checks, and trainer-shaped
candidate views. `scenarios/data/build_screened_rl_candidate_pool.py` is the thin rebuild
entrypoint; source snapshots and manifests remain under the versioned inventory directory.
Diagnostic rescreens are retained as evidence but cannot silently change strict membership.

2026-09-11 Qwen3-4B vanilla GRPO reuses the shared TRL trainer, paired-GPU
launcher and Atomic v26 runtime. The only new experiment surface is a frozen YAML
configuration. Base-model identity validation now accepts a complete numbered
shard set and still rejects missing, extra or discontinuous shards; this covers
the 4B three-shard and 8B five-shard layouts without model-specific trainer code.

2026-09-11 Qwen3-4B evaluation reuses shared feedback preflight, data-parallel
launcher, shard creation and aggregation. The model identity verifier validates
the pinned shard set against the model index for both 3-shard 4B and 5-shard 8B.
Checkpoint step/epoch and serving concurrency are manifest-bound; epoch queueing
only supplies arguments. The old supervisor remains unchanged. Relevant tests:
17 passed, with compile/bash/diff smoke checks. See
`docs/reports/sft/QWEN3_4B_CUMULATIVE_DP_EVAL_20260911.md`.

2026-09-10 hybrid diagnostic: `trl.serving_contract` owns the read-only TRL server
capability gate; `frameworks/launcher/run_trl_gpu_pair.sh` composes existing GPU and
cleanup primitives. The diagnostic scenario only supplies frozen configuration.
Turn-error provenance lives in transition accounting, span routing in the trainer,
and signed-binary in runtime terminal scoring. CPU preflight and exact canonical
factory-path audit preserve frozen protocol checks. Remote core tests and CPU
identity preflight and an isolated CUDA kernel passed; actual trainer startup is
waiting for two idle GPUs. The shared cleanup helper
captures its run root explicitly and preserves TERM/INT exit codes, covered by
subprocess tests without real GPU PIDs. See `LEGAL_REASON_HYBRID_RESTART_20260910.md`.

This is the current boundary audit for the RL refactor. Shared code belongs in
`rl.*`; scenario files should only parse scenario arguments, compose shared
APIs, and write scenario-specific artifacts.

| Package | Status | Current boundary |
|---|---|---|
| `rl.runtime` | ready for current v26 path | environment, task loading, replay, terminal scoring, rollout scoring, failure normalization, and fixed-suffix action counterfactual primitive |
| `rl.frameworks` | ready with a large runner pending | TRL transition training primitives are shared; `run_transition_grpo.py` still owns orchestration; `trl/mdp_critic.py` and `trl/iql.py` provide the diagnostic SMDP transition/equation boundary |
| `rl.evaluation` | partial | plans, contracts, identity construction, and shard aggregation are shared; the formal runner still combines asset checks, serving, and execution |
| `rl.data_selection` | in migration | pass@k and pilot policies are shared; boundary/cohort and task artifact selectors remain scenario adapters |
| `rl.fixed_pool` | in migration | generation, assembly, rescoring, validation, stratified selection, and common I/O/identity are shared; admission/freeze operations remain scenario adapters |
| `rl.configuration` | ready for current config schema | identity and optimizer validation are centralized; hardware/cohort manifest validation remains launcher-specific |
| `rl.experiments` | registry introduced | named config resolution and active-contract checks are centralized; scenario launchers still provide process orchestration |

2026-09-08 error feedback: `rl.runtime.error_feedback` owns public, bounded rejected-action
facts and message merging. `rl.evaluation.runners.feedback_overlay` applies the same renderer
to a pinned runtime with per-sample ContextVar isolation; the original parser/executor/scorer
remain authoritative. `feedback_regression` composes existing asset and identity checks;
`scenarios/evaluation/run_sft_feedback_regression.py` composes preflight, existing dual-GPU
launcher, owned-process cleanup and paired reporting. No reward or tool compatibility is added.
The shared preflight also records an explicitly supplied existing Triton driver-library path
and its resolved hash; the scenario restores this process-only setting without package changes.

The first migrated policies are `rl.data_selection.pilot` and
`rl.fixed_pool.selection`. Their historical scenario entrypoints remain as
thin compatibility CLIs. Future migrations must preserve output schema,
deterministic seeds, source hashes, and manifest identity before removing the
old implementation.

2026-09-08: the alpha=0.25 diagnostic scenario composes the explicitly frozen
remote training launcher, shared launcher helpers, and existing data-parallel
evaluation runner. Its one-off deployment/receipt audit is archived under
`archive/diagnostics/20260908_saam_alpha025.py`; no training loop is added to the
scenario and no legacy trainer is imported by the active experiment registry.

2026-09-08 static corpus audit: `rl.diagnostics.rollout_corpus` owns factual K=8
grouping, cohort-renumbering-safe question partitions and bounded review selection;
`audit_rollout_corpus.py` is the thin provenance/output CLI. Large files use the
shared `rl.diagnostics.io.iter_jsonl` streaming reader. The `distance_credit`
prototype is diagnostic-only and explicitly not a validated semantic classifier
or actor reward; its known zero-distance counterexample is regression-tested.

2026-09-09 teacher process-credit pilot: `rl.diagnostics.teacher_credit` owns the
allowlisted model-visible trajectory payload, fixed discrete rule prompt, strict
judgment schema and diagnostic-only coefficient map; `audit_teacher_process_credit.py`
is the bounded CLI. It is not imported by the actor trainer. Provider transport is
the official DeepSeek helper in `sft.provider_client`; API failures and incomplete
judgments fail closed, and no teacher output can override Harness correctness.

2026-09-09 action counterfactual audit: `rl.runtime.action_counterfactual` owns
stable-step/handle fixed-suffix replay and conservative labels; corpus selection,
runtime/database identity gates, bounded audit and manifest assembly live in
`rl.diagnostics.action_counterfactual`. The scenario is a thin CLI only. The
primitive reports structural suffix effects and is not imported by the actor
trainer or reward path.
