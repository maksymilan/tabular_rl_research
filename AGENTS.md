# AGENTS.md — current shared memory

This is the concise source of truth read by Claude Code and Codex. Historical milestones and
superseded instructions are preserved in `docs/archive/project_log.md`. Do not put secrets here;
local external-provider credentials live in ignored `api.md` and must never be committed or echoed.

## Project

This repository studies reinforcement learning for an LLM tool-use agent over relational tables.
The actor calls typed planning, perception, relational, and terminal tools. The harness executes
legal calls over SQLite, maintains resident state, returns structured observations/errors, and
verifies terminal denotation.

Research focus: **tool design and grounded, dense process credit**. The central algorithmic problem
is assigning credit to the actor's own tool trajectory without treating model-authored reasoning as
factual authority or aligning to one privileged gold path.

Start at `docs/current/README.md`.

## Non-negotiable current decisions

### Causal data generation only

- New SFT data comes from a real model↔harness loop.
- Every model/teacher turn sees only the legal episode prefix represented by current state and the
  latest environment feedback.
- Do not compile gold SQL into a complete tool trajectory and then enrich its plans, thoughts,
  observations, or perception steps.
- The retired compiler/enrichment path leaked later trajectory information into earlier turns and
  produced low-quality supervision. It lives under `archive/` for audit only.
- Gold SQL is hidden from the actor and teacher. On training tasks it may be used only by the
  harness for compatibility and terminal-denotation checks.
- No trajectory enters SFT without fresh replay, execution verification, and quality/no-leak gates.

### Model-visible protocol

- Source of truth: `src/sft/protocol.py`; index: `docs/current/tool_protocol.md`.
- Current implementation: `version20`; it has a completed fixed-200 validation. The former
  `v2i-state-only-join-feedback-r2` contract is `version1`;
  `version2` added canonical calls/final shape, `version3` added precise provider feedback and
  stable projected join columns, and `version4` counted provider-carrier failures precisely.
  `version5` replaced prefix-heavy joins with `base + joins[]`, a flat `relation.column` namespace,
  and semantic roles only for repeated relations. `version6` keeps that call shape and renders
  wide dotted columns as compact `column_namespaces` only in model-visible context; canonical
  harness state remains unchanged for replay. `version7` makes downstream relational tools consume
  those logical columns consistently, including dotted identifiers inside project expressions and
  safe unique-bare-column resolution. `version8` removes competing canonical/split response
  instructions from DeepSeek-facing prompts and renders rolling assistant history in the same
  provider carrier; canonical stored trajectories are unchanged. `version9` explicitly enables
  DeepSeek thinking mode, fixes reasoning effort, and records request/response provider identity
  metadata so split-carrier behavior no longer depends on unaudited API defaults. `version10`
  additionally uses the provider's JSON Output constraint for visible action content, then wraps
  that unmodified JSON in the internal canonical tool-call envelope. `version11` removes the last
  generic visible-`<tool_call>` wording from the DeepSeek-facing generation and retry clauses, so
  the API-facing prompt names only the native-reasoning + raw-JSON carrier. `version12` adds
  `project(distinct=...)` and grounded `scalar_compute`; `version13` makes terminal answers cite
  one exact result table, including 1x1 scalar tables, instead of carrying model-authored answer
  values. `version14` keeps canonical grounded plan evidence intact but renders only its
  `step_id + tool` identity in model-visible resident plan state, avoiding repeated large
  schema/table payloads. `version15` adds an optional aggregation-level `where` predicate to
  `group_aggregate`, allowing several conditional metrics to share one fixed input population and
  grain instead of being split across drifting filter branches. `version16` briefly exposed a
  separate `pivot` reshape atom and `version17` clarified its boundary with `project`. `version18`
  folds that reshape into `group_aggregate(output_layout="columns", category_values=[...])`;
  `pivot` is replay-only, so the model chooses one aggregate tool rather than two competing tools.
  `version19` lets `scalar_compute` cite a named column from a prior one-row multi-metric table as
  `{"value_ref":"step_k","column":"metric"}`, avoiding duplicate aggregation while preserving
  harness-owned cell grounding and per-column provenance. `version20` removes copy-prone provider
  field labels from both DeepSeek carrier examples and adds short argument invariants beside the
  affected public tools (`join_tables`, `scalar_compute`, `group_aggregate`, `project`, and
  `read_subtable`). It also treats a length-truncated provider completion as a bounded, audited
  same-turn client retry rather than a semantic agent action; tool execution semantics and action
  boundaries are unchanged.
  Future versions increment numerically.
- The canonical model action contains exactly one `<think>` block and one strict `<tool_call>` JSON
  object. A provider-native reasoning adapter may carry the same authored reason in a separate API
  field, but its API-facing prompt and history must describe only that one carrier.
- Context is rebuilt from catalog + question + optional external knowledge + current resident state
  + optional `LAST TOOL ERROR`; it is not an appended observation transcript.
- Current tools: `plan`, `describe_table`, `inspect_column`, `read_subtable`,
  `condition_filter`, `project`, `scalar_compute`, `join_tables`, `group_aggregate`,
  `extreme_value_select`, `set_op`, and `answer_from_context`.
- No `add_to_memory`, `refine_memory`, reflection, invalidate, or model-visible sidecar state.
- Plans are control state, not factual evidence. Scalar reuse is grounded through direct step-id
  `value_ref`.
- Do not construct new version12-version20 SFT data until the frozen 200-task tool-usability gate
  is explicitly passed. Small pilots are diagnostic only and are not SFT sources.

### Recovery and provenance

- Protocol, argument-validation, and state-preserving execution errors are bounded recoverable
  actions. They spend the shared action budget and are stored as audit-only error events.
- Rejected actions are never SFT targets. The first later legal action may be marked
  `feedback_recovery`.
- Data/value/grounding references are harness-owned. Model reasoning and plan text never control
  factual provenance or reward.
- API transport retries are client events, not semantic agent actions.

### Reward boundary

- `src/rl/terminal_reward.py` is the deliberate terminal-{0,1} control.
- `src/rl/process_credit.py` owns replay-derived tool-local credit.
- Full-turn result-only credit can positively train erroneous turns in a recovered successful
  episode; this is acceptable only for the coarse control. Process training must keep local errors
  negative/zero and credit later recovery separately.
- Do not enable process-RL optimization until deterministic completeness and independent grounding
  edge-precision gates pass.

### Evaluation

- Always name the denotation metric. BIRD reference EX uses `bird-set`; training replay and process
  audits retain normalized `strict-multiset` unless explicitly overridden.
- Do not mix result directories across task selections, model/checkpoint, protocol, decoding,
  timeout, external knowledge, or denotation comparison.
- Report API/transport failures separately from semantic policy failures.
- The completed fixed-200 DeepSeek v4 Flash version11 control is **130/200 strict-multiset**.
  Manual audit of its 70 failures labels 21 exact-output-shape, 13 relational/tool-use,
  17 semantic-understanding, 18 prompt/gold-conflict, and 1 provider-protocol case.
- The completed version19 fixed-200 canonical rerun is **132/200 strict-multiset**. A global
  relational-invariants prompt reached 137/200 but had 8 paired regressions, lower legal rate,
  more process errors, and no significant paired gain; it remains an ablation rather than the
  canonical prompt. Neither run passed the required 150/200 SFT-construction gate.
- The completed version20 fixed-200 canonical run is **138/200 strict-multiset**, with 199/200 legal
  termination and 24 process errors versus version19's 190/200 and 91. It removes all 71 observed
  carrier protocol errors, but still misses the 150/200 gate; 53 of its 62 failures are error-free
  legal trajectories. See
  `docs/reports/evaluation/BIRD_VERSION20_CARRIER_SCHEMA_FIXED200_20260723.md`.
- A deterministic-resolution version21 ablation reached 139/200 but reduced legal termination to
  197/200, raised process errors from 24 to 38, and used 8.0% more tokens. Its 9 paired gains versus
  8 regressions were not significant, so it was not promoted and the active code remains version20.
  See `docs/reports/evaluation/BIRD_VERSION21_RESOLUTION_ABLATION_FIXED200_20260723.md`.
- A paired six-hard-task diagnostic found no gain from forced resident planning:
  optional and required were both 0/6, while required planning increased mean actions by 16.4%
  and tokens by 18.5%. Keep `required-resident` experimental; do not expand it or make it default
  without a new small-pilot signal. See
  `docs/reports/evaluation/BIRD_VERSION11_FAILURE_TRAJECTORY_AND_RESIDENT_PLAN_AUDIT.md`.

## Current BIRD reference points (2026-07-22)

- Direct SQL, fresh greedy Qwen2.5-7B under BIRD reference EX: **650/1534 = 42.37%**.
- Direct SQL K=4 sampling under BIRD reference EX: pass@1/2/4 =
  **624/713/785 = 40.68/46.48/51.17%**.
- Frozen SFT-2 checkpoint: `checkpoint-743`, trained for one epoch on the 11,874-record causal
  on-policy mixture. Its completed strict-multiset K=4 evaluation was
  **38.79/46.87/55.74%** at pass@1/2/4; do not compare this directly with `bird-set` results.
- A fresh SFT-2 BIRD-reference-EX re-evaluation and a result-only-RL checkpoint-70 held-out
  evaluation were active when the historical log was compacted. Verify remote state and manifests
  before launching or reporting either; never start duplicates from this note alone.

Detailed experiment chronology, artifact paths, and scorer audits are in
`docs/archive/project_log.md` and `docs/reports/`.

## Active layout

- `src/harness/`: executor, catalog, environment state, provenance, grounding, dataset adapters.
- `src/sft/`: protocol, causal teacher rollout, replay/quality filters, SFT export and assembly.
- `src/eval/`: closed-loop tool evaluation and direct-SQL controls.
- `src/rl/`: tool environment, task loader, terminal reward, process credit, objective, backend.
- `docs/current/`: active contracts and workflows.
- `docs/decisions/`: rationale; not executable contracts.
- `docs/reports/`: immutable experiment and audit reports.
- `docs/archive/` and `archive/`: historical only; active code must not import from them.

## Active entry points

- Tool rollout: `src/eval/rollout.py`, `src/eval/rollout_passk.py`.
- Direct SQL: `src/eval/text2sql.py`, `src/eval/text2sql_passk.py`.
- Teacher data: `src/sft/generate_teacher_rollouts.py`.
- BIRD SFT-2 assembly: `src/sft/build_bird_sft2_dataset.py`,
  `src/sft/assemble_bird_sft2_mixture.py`.
- SFT export: `src/sft/export_sft_dataset.py`.
- Current SFT-2 training: `src/sft/train_bird_sft2_qwen25_7b.sh` with
  `src/sft/configs/bird_sft2_qwen25_7b_qlora_6400.yaml`.
- RL backend: `src/rl/frameworks/accelerate/group_reinforce.py`.

## Naming rules

- Reusable modules use role-based snake_case names, not experiment versions.
- Dataset split belongs in adapter names (`bird_train_adapter.py`, `bird_dev_adapter.py`).
- Experiment versions, model names, sample counts, and dates belong in configs, manifests, result
  directories, and reports rather than reusable module names.
- Completed launchers/configs move to `archive/experiments/`; do not leave them beside current
  entry points with names such as `v9`, `v13`, or `after_train`.
- New active code must not use imports from `archive/` or depend on retired compiler/enrichment
  modules.

## Verification

Run checks proportionate to the changed layer. The normal local suite is:

```bash
PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python src/harness/run_tests.py
PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python -m unittest discover -s src/sft/tests
PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python -m unittest discover -s src/rl/tests
PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python src/eval/test_eval.py
```

Compiler/emitter/enrichment tests under `archive/` are frozen historical tests and are not part of
the active suite.

## Operational safety

- Existing worktree changes belong to the user; preserve unrelated edits.
- Before starting remote GPU work, inspect running services, tunnels, result manifests, and resume
  state. Do not infer that a historical `active` entry is still active.
- Do not kill unrelated GPU processes. Physical GPU ownership notes in the historical log may be
  stale and must be rechecked.
- Data under `data/` is ignored and currently large; never delete or reorganize it without a
  separate explicit data-retention decision.
