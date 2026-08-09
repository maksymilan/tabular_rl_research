# Active architecture

## Runtime layers

- `src/harness/`: SQLite tool execution, resident environment state, catalog construction,
  relation derivation, provenance, scalar grounding, and dataset adapters.
- `src/sft/`: protocol rendering/parsing, causal teacher rollouts, replay and quality gates, SFT
  dataset export, and current BIRD SFT-2 assembly.
- `src/tool_modules/`: independently selectable, scheme-owned protocols, execution loops, audits,
  exporters, and tests. `action_block/`, `relational_program/`, `direct_sql_search/`, and
  `iterative_sql/` and `native_tool_bundle/` do not import one another except through an explicit
  semantic dependency;
  `sql_common/` is the named immutable-SQL execution layer shared by the two SQL schemes.
- `src/eval/`: cross-scheme evaluation infrastructure, atomic/direct-SQL controls, pass@k
  aggregation, denotation/candidate selection, artifact contracts, and thin compatibility entry
  points only. Scheme implementations no longer live here. `denotation.py` owns the named
  result-comparison registry independently of candidate generation; `candidate_selection.py` owns
  optional multi-candidate selection independently of correctness scoring.
- `src/rl/`: task loading, tool environment, hidden target support, terminal reward, process credit,
  SFT-index reward audits, policy objective, and the supported Accelerate backend.
- `experiment_dashboard/`: local inspection UI; it is not part of training semantics.

## Shared ownership

`src/sft/protocol.py` owns the model-visible tool contract and context renderer.
Non-atomic scheme contracts are owned by their package under `src/tool_modules/`; the exclusive
scheme registry is `src/tool_modules/registry.py`.
`src/harness/executor.py` owns tool execution. `src/harness/relation_derivation/` owns validated,
fact-only formal semantics attached to derived relation handles. `src/harness/provenance.py` owns
data/value/grounding edges. `src/harness/environment_state.py` owns resident state.
`src/eval/rollout.py` orchestrates those harness services and provides shared scoring helpers used
by evaluation, teacher rollout, and RL. `src/harness/catalog.py` owns the bounded opening catalog.

## Scheme package map

| Scheme | Canonical protocol | Canonical runtime / exporter |
|---|---|---|
| atomic | `src/sft/protocol.py` | `src/eval/rollout.py` and the shared harness |
| action-block | `src/tool_modules/action_block/protocol.py` | `evaluator.py`, `sft_export.py` |
| relational-program | `src/tool_modules/relational_program/protocol.py` | `evaluator.py` |
| direct-sql-search | `src/tool_modules/direct_sql_search/protocol.py` | `src/tool_modules/sql_common/runner.py` |
| iterative-sql | `src/tool_modules/iterative_sql/protocol.py` | `src/tool_modules/sql_common/runner.py`, `iterative_sql/audit.py` |
| native-tool-bundle | `src/tool_modules/native_tool_bundle/protocol.py` | `provider_tools.py`, `audit.py` |

Atomic remains the promoted project-wide protocol consumed jointly by SFT, evaluation, and RL; it
is intentionally not duplicated under `tool_modules/`. The package split isolates the independent
experimental action surfaces that were previously mixed into flat evaluation/SFT directories.

Historical paths such as `src/eval/evaluate_batch_plan.py`,
`src/eval/evaluate_relational_program.py`, `src/eval/iterative_sql.py`, and the former flat protocol
modules, plus `src/sft/tool_schemes.py`, are compatibility aliases only. New imports, launchers,
manifests, and documentation must use `tool_modules.*`; compatibility aliases may not be imported
from inside `src/tool_modules/`.

## Archived code

- `archive/code/gold_sql_compiler/`: retired SQL→complete-tool-trajectory pipeline.
- `archive/code/trajectory_enrichment/`: retired complete-trajectory enrichment pipeline.
- `archive/code/experimental_backends/verl/`: unsupported historical Verl backend.
- `archive/experiments/`: completed launchers, configs, and launchd definitions.

Archived modules are audit artifacts. Active code must not import from `archive/`.
