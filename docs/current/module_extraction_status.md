# Module extraction status

This is the current boundary audit for the RL refactor. Shared code belongs in
`rl.*`; scenario files should only parse scenario arguments, compose shared
APIs, and write scenario-specific artifacts.

| Package | Status | Current boundary |
|---|---|---|
| `rl.runtime` | ready for current v26 path | environment, task loading, replay, terminal scoring, rollout scoring, and failure normalization |
| `rl.frameworks` | ready with a large runner pending | TRL transition training primitives are shared; `run_transition_grpo.py` still owns orchestration |
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
