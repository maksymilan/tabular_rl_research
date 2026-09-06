# Evaluation runners

`runners/` is the single home for reusable evaluation executables and pure
evaluation primitives: dataset planning, identity construction, shard
creation/merge, and the Atomic v26 matched evaluator. A
scenario selects these runners and supplies its dataset, checkpoint, GPU
partition, and contract path. Experiment-specific wrappers belong in
`rl.scenarios.evaluation`, never alongside the reusable runners.

Evaluation JSON contracts are generated or archived artifacts. They are not
tracked under `src/rl`; set `ATOMIC_V26_EVAL_CONTRACT` when replaying one.
