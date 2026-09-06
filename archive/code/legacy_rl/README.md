# Legacy RL branches

These directories were previously placed under `src/rl` but are not part of the
Atomic version26 screened500 mainline. They are preserved with their original
filenames for explicit historical replay only.

- `action_dpo/`: frozen exact-prefix/action-DPO experiments.
- `distillation/`: historical teacher/distillation pipeline.

Nothing in the active launcher imports these directories. Restore an explicit
historical experiment together with its matching runtime and manifest; do not
use them to initialize the current version26 RL run.
