# SFT pipeline

## Accepted sources

SFT examples come from causal student or external-teacher model↔harness episodes. A successful
episode must pass terminal denotation scoring, fresh replay, protocol/quality gates, and the rolling
single-action export checks before entering a mixture.

Version51 is the frozen provider baseline, version52 is its compact-prompt successor, version53 is
the reviewed-prompt control, and version54 / `native-tool-bundle` is the forward no-plan
diagnostic. All four are diagnostic-only. Their
multi-call turns must not enter the existing single-action exporter or SFT/RL
mixtures by flattening them into fake assistant turns. Admission requires a scheme-aware exporter
and an explicit training promotion gate. Its fixed-200 behavior gate passed versus version50 at
147/200 correct and 200/200 legal, but it remained statistically tied with version24 and this does
not change the export boundary.

`iterative-sql-v6` is likewise excluded. Its SQL actions, resident SQL state, and recoverable
submission errors have a distinct protocol identity and must not be relabeled as atomic or
direct-sql-search targets. No current SFT exporter accepts this scheme.

The denotation comparison used to admit an episode is stored in
`rollout_generation.denotation_comparison`, and fresh replay reuses that exact comparison.
Historical artifacts that predate this field require an explicit replay override; they must never
be silently replayed under a comparison that conflicts with their manifest. The current default
for every new BIRD episode, replay, grounding gate, and reward audit is `bird-set`.

The current BIRD SFT-2 construction is student-first pass@k. Teacher fallback is restricted to true
student pass@k failures. Provider attempts, transport failures, rejected trajectories, and duplicate
actions remain auditable but do not become training targets.

Task-level admission happens before rollout. `gold-denotation-nonempty-task-filter-v1` privately
executes hidden gold SQL in read-only SQLite and retains only tasks with at least one returned row.
It does not use the source `gold_exec_results` field and does not store SQL text, result rows, or
values in its status audit. Empty tasks and execution/input errors are excluded before any provider
request. A scalar result row containing numeric zero remains eligible. The active teacher cohort is
`bird_train_atomic_teacher1500_v2_nonempty`; it preserves all v1 ids and order because the complete
old 1,500 was certified nonempty, while its manifest binds the filtered 6,599/5,915 source pools.

## Export contract

`src/sft/export_sft_dataset.py` and `src/sft/build_rolling_sft_data.py` render each supervised action
from the state immediately before that action. The input must exclude the current tool output,
future actions or observations, hidden gold SQL, and future factual provenance.

The exporter always uses the concise student runtime prompt from
`src/sft/prompt_contract.py`. External-teacher guidance and worked examples affect generation only;
they are neither copied into the ShareGPT `system` field nor treated as learned factual context.
The exported manifest binds the student prompt SHA-256 and public tool-schema SHA-256; source
teacher prompt hashes remain audit metadata.

Recovered trajectories retain erroneous turns in the audit record, but rejected actions are not SFT
targets. The first later legal action can be labeled as feedback recovery when it causally uses the
structured error.

Training export also enforces `causal-empty-result-target-filter-v1`:

- a successful intermediate action whose result explicitly reports `row_count=0` is retained in
  the causal prefix but excluded as an SFT target;
- a trajectory whose terminal evidence table has `row_count=0` is excluded wholesale;
- a one-row scalar table containing the value zero, such as `COUNT=0`, is not an empty result and
  remains eligible.

This rule is checked both when trajectories are generated/audited and defensively by the rolling
exporter. Empty feedback is never deleted when a later correction depends on seeing it.

## Current entry points

- `src/sft/generate_teacher_rollouts.py`: closed-loop external-teacher generation.
- `src/sft/build_bird_sft2_dataset.py`: replay student successes and select fallback tasks.
- `src/sft/assemble_bird_sft2_mixture.py`: deterministic mixture assembly.
- `src/sft/export_sft_dataset.py`: final ShareGPT-style export.
- `src/sft/train_bird_sft2_qwen25_7b.sh`: current training launcher.
