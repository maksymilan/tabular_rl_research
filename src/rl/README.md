# Active RL source

This directory is the active Atomic version26 source boundary. Historical
branches and one-off diagnostics are being migrated out; until each dependency
chain is moved, legacy files remain explicitly marked in the ownership reports
and must not be used by new workflows.

The dependency closure for the current screened500 run starts at
`scenarios/rl_main/run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_single_gpu_a100.sh`
and `configs/experiments/qwen3_8b_atomic_v26_saam_screened500_single_gpu.yaml`,
then enters `frameworks/trl/`, `runtime/`, `objectives/`, and the shared task and
rollout modules. A file outside that closure needs an owner, an entrypoint,
and a manifest before it is added here.

Fixed reusable code belongs in owned packages such as `shared/`, `runtime/`,
`objectives/`, `frameworks/trl/`, and `data_selection/`. A new experiment
scenario belongs in `scenarios/` and should compose these packages through
configuration; it should not copy their I/O, hashing, cohort predicates, GPU
checks, or trainer logic.
