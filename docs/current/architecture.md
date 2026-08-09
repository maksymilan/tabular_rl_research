# Active architecture

## Runtime layers

- `src/harness/`: SQLite tool execution, resident environment state, catalog construction,
  relation derivation, provenance, scalar grounding, and dataset adapters.
- `src/sft/`: protocol rendering/parsing, causal teacher rollouts, replay and quality gates, SFT
  dataset export, and current BIRD SFT-2 assembly.
- `src/tool_modules/`: independently selectable, scheme-owned protocols, execution loops, audits,
  exporters, and tests. `checkpoint_relalg/`, `action_block/`, `relational_program/`,
  `direct_sql_search/`, `iterative_sql/`, and `native_tool_bundle/` do not import one another except
  through an explicit semantic dependency;
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

`src/sft/protocol.py` owns the retained atomic model-visible contract and context renderer.
`src/tool_modules/checkpoint_relalg/` owns the forward protocol, mode schemas, prompt layers,
state/artifacts/checkpoints, execution loop, and audit contract. Other non-atomic scheme contracts
are owned by their package under `src/tool_modules/`; the exclusive scheme registry is
`src/tool_modules/registry.py`.
`src/harness/executor.py` owns retained atomic tool execution; checkpoint-relalg uses its
package-owned typed relational executor. `src/harness/relation_derivation/` owns validated,
fact-only formal semantics attached to derived relation handles. `src/harness/provenance.py` owns
data/value/grounding edges. `src/harness/environment_state.py` owns resident state.
`src/eval/rollout.py` orchestrates those services for the retained atomic line and provides shared
scoring helpers used by evaluation, teacher rollout, and RL. `src/harness/catalog.py` owns the
bounded atomic opening catalog.

## Scheme package map

| Scheme | Canonical protocol | Canonical runtime / exporter |
|---|---|---|
| checkpoint-relalg | `src/tool_modules/checkpoint_relalg/protocol.py` | `runner.py`, `runtime.py`, `audit.py` |
| atomic | `src/sft/protocol.py` | `src/eval/rollout.py` and the shared harness |
| action-block | `src/tool_modules/action_block/protocol.py` | `evaluator.py`, `sft_export.py` |
| relational-program | `src/tool_modules/relational_program/protocol.py` | `evaluator.py` |
| direct-sql-search | `src/tool_modules/direct_sql_search/protocol.py` | `src/tool_modules/sql_common/runner.py` |
| iterative-sql | `src/tool_modules/iterative_sql/protocol.py` | `src/tool_modules/sql_common/runner.py`, `iterative_sql/audit.py` |
| native-tool-bundle | `src/tool_modules/native_tool_bundle/protocol.py` | `provider_tools.py`, `audit.py` |

`checkpoint-relalg-v1` is the forward implementation boundary for all new tool and experiment
development. Its package owns one shared state/artifact/checkpoint runtime and three explicitly
selected public surfaces (`direct`, `atomic`, and `hybrid`); the mode is part of every manifest and
cannot be inferred from observed calls. The complete design specification is a Harness and
implementation contract. Provider requests contain only a short shared core, a short mode prompt,
compact native function schemas, and the current dynamic context.

This forward designation does not open training admission. `checkpoint-relalg-v1` remains
diagnostic-only until its own fresh replay, structure, provider-history, no-leak, behavior, and
export/admission gates pass. The original atomic runtime is intentionally not duplicated under
`tool_modules/`; it remains supported for existing RL experiments, frozen controls, and exact
reproduction. Version54 / `native-tool-bundle` is a frozen diagnostic predecessor with no RL
admission. Neither legacy line may be silently mixed with or relabeled as `checkpoint-relalg`.

The package split isolates independent action surfaces that were previously mixed into flat
evaluation/SFT directories.

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
