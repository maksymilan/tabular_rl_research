# Remote runtime snapshot (archived)

These files were copied from the remote A100 runtime for comparison during the
2026-09-03 debugging session. They are retained for audit only and are not an
importable module or an experiment entrypoint.

Source paths:

- `/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_8b_v26_rl_dynamic_micro_20260902/src/rl/frameworks/trl/rollout.py`
- `/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_8b_v26_rl_dynamic_micro_20260902/src/rl/frameworks/trl/transition_grpo.py`
- `/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_8b_v26_rl_dynamic_micro_20260902/src/rl/frameworks/trl/tool_loss_mask.py`

The files were formerly left as `*.remote.current.py` in the repository root
or under ignored `tmp/`. Their hashes are recorded below so an old comparison
can be reproduced without treating the snapshot as current code.

| file | SHA-256 |
|---|---|
| `src/rl/frameworks/trl/rollout.py` | `9fb39423e1c3b4b155febe3eaeee87aa3d3940d4c0907d8f7eda20f6497fb68b` |
| `src/rl/frameworks/trl/transition_grpo.py` | `3aa40dabf95f2424272c6f7337a45575baa5a7124779fb53bca92b76690d7b86` |
| `src/rl/frameworks/trl/tool_loss_mask.py` | `b15072423600c663d91651e640f7738087bdc1ea551758704f98cc6882d14abb` |

Do not import from this directory. Active Atomic version26 code lives under
`src/rl/` and uses the immutable version26 runtime specified by the project
contract.
