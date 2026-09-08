# Module extraction status

This is the current boundary audit for the RL refactor. Shared code belongs in
`rl.*`; scenario files should only parse scenario arguments, compose shared
APIs, and write scenario-specific artifacts.

| Package | Status | Current boundary |
|---|---|---|
| `rl.runtime` | ready for current v26 path | environment, task loading, replay, terminal scoring, rollout scoring, and failure normalization |
| `rl.frameworks` | ready with a large runner pending | TRL transition training primitives are shared; `run_transition_grpo.py` still owns orchestration; `trl/mdp_critic.py` and `trl/iql.py` provide the diagnostic SMDP transition/equation boundary |
| `rl.evaluation` | partial | plans, contracts, identity construction, and shard aggregation are shared; the formal runner still combines asset checks, serving, and execution |
| `rl.data_selection` | in migration | pass@k and pilot policies are shared; boundary/cohort and task artifact selectors remain scenario adapters |
| `rl.fixed_pool` | in migration | generation, assembly, rescoring, validation, stratified selection, and common I/O/identity are shared; admission/freeze operations remain scenario adapters |
| `rl.configuration` | ready for current config schema | identity and optimizer validation are centralized; hardware/cohort manifest validation remains launcher-specific |
| `rl.experiments` | registry introduced | named config resolution and active-contract checks are centralized; scenario launchers still provide process orchestration |

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
