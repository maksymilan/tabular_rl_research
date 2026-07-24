# Active architecture

## Runtime layers

- `src/harness/`: SQLite tool execution, resident environment state, catalog construction,
  relation derivation, provenance, scalar grounding, and dataset adapters.
- `src/sft/`: protocol rendering/parsing, causal teacher rollouts, replay and quality gates, SFT
  dataset export, and current BIRD SFT-2 assembly.
- `src/eval/`: closed-loop tool evaluation, direct-SQL controls, pass@k aggregation, and artifact
  contracts. `denotation.py` owns the named result-comparison registry independently of candidate
  generation; `candidate_selection.py` owns optional multi-candidate selection independently of
  correctness scoring.
- `src/rl/`: task loading, tool environment, hidden target support, terminal reward, process credit,
  SFT-index reward audits, policy objective, and the supported Accelerate backend.
- `experiment_dashboard/`: local inspection UI; it is not part of training semantics.

## Shared ownership

`src/sft/protocol.py` owns the model-visible tool contract and context renderer.
`src/harness/executor.py` owns tool execution. `src/harness/relation_derivation/` owns validated,
fact-only formal semantics attached to derived relation handles. `src/harness/provenance.py` owns
data/value/grounding edges. `src/harness/environment_state.py` owns resident state.
`src/eval/rollout.py` orchestrates those harness services and provides shared scoring helpers used
by evaluation, teacher rollout, and RL. `src/harness/catalog.py` owns the bounded opening catalog.

## Archived code

- `archive/code/gold_sql_compiler/`: retired SQL→complete-tool-trajectory pipeline.
- `archive/code/trajectory_enrichment/`: retired complete-trajectory enrichment pipeline.
- `archive/code/experimental_backends/verl/`: unsupported historical Verl backend.
- `archive/experiments/`: completed launchers, configs, and launchd definitions.

Archived modules are audit artifacts. Active code must not import from `archive/`.
