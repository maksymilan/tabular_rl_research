# Active architecture

The repository has two deliberately separate axes:

- the **training mainline** freezes Atomic version26 SFT1 for Qwen3 SFT expansion, matched local
  evaluation, and the next RL stage;
- the **tool-research line** retains Direct/Atomic/Hybrid, checkpoint, alternative carriers and
  other schemes as identity-separated diagnostics.

They may share Harness implementation, but they may not share a model parser, prompt renderer,
result directory, checkpoint, or training admission merely because both contain a tool named
“Atomic”. See `training_mainline.md`.

## Runtime layers

- `src/harness/`: SQLite tool execution, resident environment state, catalog construction,
  relation derivation, provenance, scalar grounding, and dataset adapters.
- `src/sft/`: protocol rendering/parsing, causal teacher rollouts, replay and quality gates, the
  current version26 SFT1 preparation/training path, and retained SFT-2/control assembly code.
- `src/tool_modules/`: independently selectable, scheme-owned protocols, execution loops, audits,
  exporters, and tests. `checkpoint_relalg/`, `action_block/`, `relational_program/`,
  `direct_sql_search/`, `iterative_sql/`, and `native_tool_bundle/` do not import one another except
  through an explicit semantic dependency;
  `sql_common/` is the named immutable-SQL execution layer shared by the two SQL schemes.
  `checkpoint_relalg/qwen3_carrier.py` is retained only for the frozen failed Atomic-v24 diagnostic
  projection; it is not the current version26 local evaluator.
- `src/eval/`: cross-scheme evaluation infrastructure, atomic/direct-SQL controls, pass@k
  aggregation, denotation/candidate selection, artifact contracts, and thin compatibility entry
  points only. Scheme implementations no longer live here. `denotation.py` owns the named
  result-comparison registry independently of candidate generation; `candidate_selection.py` owns
  optional multi-candidate selection independently of correctness scoring.
- `src/rl/`: task loading, tool environments, hidden target support, terminal reward, process credit,
  SFT-index reward audits, policy objective, TRL/Accelerate backends, and the canonical cross-arm
  evaluation analyzer. `tool_environment_v26.py` is the current Qwen3 SFT-to-RL handoff; retained
  atomic/action-block and checkpoint-relalg experiments remain identity-separated controls.
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

This broad forward designation does not open every mode/profile for training. Current training has
explicitly returned to the isolated Atomic version26 SFT1 identity described in
`training_mainline.md`; Direct, Hybrid, checkpointed semantic profiles, Atomic-v24-frozen
projections, and alternative carriers remain diagnostic. The original atomic runtime is intentionally not duplicated under
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
