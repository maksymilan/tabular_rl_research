# Historical experiment shelf

The active RL experiment is the named scenario at
`src/rl/scenarios/rl_main/run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_single_gpu_a100.sh`.
All superseded launchers and experiment-specific utilities were moved to
`archive/experiments/legacy_20260906/` with their filenames intact. Shared GPU
resource and process handling is fixed infrastructure at
`src/rl/frameworks/launcher/launch_common.sh`.
