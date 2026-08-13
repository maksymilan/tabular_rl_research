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

### External provider: official DeepSeek only

- New DeepSeek teacher, evaluation, audit, and recovery requests must use the official OpenAI-
  compatible base URL `https://api.deepseek.com` and the repository's existing
  `/chat/completions` transport.
- AimixHub/AIHubMix (`aihubmix.com`) is deprecated as of 2026-08-05. Do not use it as a primary
  endpoint, fallback, proxy, credential source, or source for copied API examples. If the official
  service is unavailable, fail explicitly rather than silently routing through a third party.
- The current agent loop is Chat Completions, not FIM. The official `/beta/completions` FIM API may
  be used only by a separately scoped code-completion feature and must not replace the tool-use
  provider loop.
- Runtime credentials remain only in ignored `api.md` with
  `BASE_URL=https://api.deepseek.com`. Never copy keys into documentation, tracked configuration,
  commands, logs, or chat. See `docs/current/provider_api.md`.

### Pre-authorized external evaluation

- Official DeepSeek API evaluation or teacher batches with fewer than 200 episodes are
  pre-authorized and do not require additional user confirmation.
- The endpoint must remain `https://api.deepseek.com`; the requested model, cohort, per-episode
  limits, and total token cap must be recorded before launch.
- Run these pre-authorized batches inside the existing `workspace-write` sandbox with its enabled
  network access. Do **not** mark the command `require_escalated` merely because it calls the
  official API; that unnecessarily routes the run through desktop approval/auto-review. Request
  escalation only when the command genuinely needs access beyond the configured writable roots or
  another separately protected capability.
- Ask again only for 200 or more episodes, a different provider or endpoint, destructive actions,
  or a material expansion of scope.

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

- The forward implementation is `checkpoint-relalg-v1` under scheme `checkpoint-relalg`, with a
  mandatory `mode=direct|atomic|hybrid`. New tool and protocol experiments start from this package,
  not from atomic or version54. Every assistant turn authors exactly one official DeepSeek native
  tool call; all modes share the same Harness-owned relation artifacts, current environment state,
  semantic checkpoint/restore path, and exact-artifact terminal answer. The complete 40k-character
  specification is for the Harness and implementation agent, not a model prompt. The model sees
  only a short shared core, one short mode prompt, compact schemas for that mode, and causal dynamic
  context in this order: question, optional external knowledge, current phase targets, checkpoint
  history, current environment state, and optional last error. The external teacher adds only short
  checkpoint-use guidance. See `docs/current/checkpoint_relalg_v1_zh.md`.
- Atomic operator granularity is now an explicit, identity-bound profile. `micro-v1` remains the
  frozen default for exact replay. `semantic-v2` is an isolated diagnostic profile under the same
  `mode=atomic`; it replaces several mechanical chains with `shape_rows`, conditional
  `group_aggregate`, grounded `scalar_compute`, and tie-explicit `rank_select`. It retains
  model-visible `commit_checkpoint` and `restore_checkpoint` plus the same Harness snapshot and
  audit semantics. `semantic-v3-v24` is a Text-JSON-only diagnostic repair profile over exactly
  the same semantic-v2 executable tool/schema surface. It borrows version24's compact operational
  signatures, validator-checked canonical examples, exact-output constraints, and recent-4 causal
  provider history, while retaining current typed execution, artifacts, and checkpoint semantics.
  Its prompt/history/protocol identity is distinct and must not be mixed with v2. Neither semantic
  profile has behavior or SFT/RL promotion until a paired causal gate passes. See
  `docs/reports/evaluation/CHECKPOINT_RELALG_SEMANTIC_V3_V24_GAP_REPAIR_20260811_ZH.md`.
- The fresh semantic-v2 versus semantic-v3-v24 paired Gate20 used teacher1500 positions 200–219,
  official Flash, Atomic, Text-JSON, and identical model-choice-v6 checkpoint guidance. Both arms
  scored 14/20 `bird-set`, 9/20 strict artifact, 11/20 schema match, and 20/20 legal with identical
  per-task outcome vectors. V3 reduced provider tokens from 2,481,912 to 881,327 (-64.49%), turns
  from 160 to 151, and tool errors from 11 to 9; every one of the 20 v3 episodes used fewer tokens.
  All 40 records passed current identity, structure, cohort/budget, no-leak, and fresh-replay audit.
  Retain v3 as the next diagnostic Atomic candidate, but do not claim an accuracy gain or admit it
  to SFT/RL. See
  `docs/reports/evaluation/CHECKPOINT_RELALG_SEMANTIC_V3_V24_PAIRED_GATE20_RESULT_20260811_ZH.md`.
- “Forward” identifies the development base; it is not a behavior or training promotion.
  `checkpoint-relalg-v1` remains diagnostic-only until fresh replay, structure, provider-history,
  no-leak, behavior, scheme-aware export, and explicit SFT/RL admission gates pass. Gold SQL and
  gold results remain Harness-only and never enter any provider request. Do not report a pilot
  result until an actual run and its audits complete.
- The first actual official checkpoint-relalg request on 2026-08-09 passed `/models` identity
  verification but the Direct Chat Completions call returned `HTTP 402 / Insufficient Balance`
  before any authored tool call. Its failure artifact passed structure audit and fresh replay.
  This is a transport-blocked attempt with zero semantic episodes, not a Direct failure or a
  three-mode pilot result. Atomic/Hybrid were not sent redundant requests and no third-party
  fallback was used. See
  `docs/reports/evaluation/CHECKPOINT_RELALG_V1_CAUSAL_SMOKE_20260809_ZH.md`.
- After balance recovery, the frozen teacher1500 Prefix100 ran causally with official
  `deepseek-v4-flash`, Hybrid, and the diagnostic A/Text-JSON carrier. It scored 63/100 `bird-set`,
  42/100 strict artifact, 56/100 schema match, and 99/100 legal termination. All 100 records passed
  structure and fresh replay; 581 provider attempts used 11,677,199 tokens and all response model
  identities were Flash. No episode used checkpoint/restore. Treat at most the 42 strict records as
  scheme-local training candidates: checkpoint-relalg still has no approved exporter or SFT/RL
  admission, and none may be mixed into the atomic pipeline. See
  `docs/reports/evaluation/CHECKPOINT_RELALG_V1_FLASH_TEXT_JSON_HYBRID_PREFIX100_20260809_ZH.md`.
- A later Atomic `semantic-v2` paired Gate20 compared model-chosen checkpointing with checkpoint
  disabled while leaving Harness checkpoint eligibility at `none` and restore disabled. The
  model-choice arm attempted zero checkpoints across 20/20 tasks and 223 model turns; it scored
  9/20 `bird-set` versus 11/20 disabled, while both arms scored 6/20 strict artifact. All 40
  records passed structure, cohort/manifest binding, and fresh replay; seven overall audits failed
  only at the frozen per-episode token gate. This is an underexposed prompt-manipulation failure,
  not evidence that checkpoint helps or hurts. Do not restore producer-count gating or remove
  checkpoint based on this result. See
  `docs/reports/evaluation/CHECKPOINT_RELALG_MODEL_CHOICE_VS_DISABLED_GATE20_20260811_ZH.md`.
- A follow-up Atomic `semantic-v2` fresh paired Gate8 tested `model-choice-commit-v2` against
  checkpoint-disabled with Harness eligibility still `none`, restore disabled, and max checkpoints
  eight. The v2 arm now achieved checkpoint exposure on 5/8 tasks with seven accepted commits, but
  scored 4/8 `bird-set` versus 6/8 disabled (paired v2-only 0, disabled-only 2). Legal termination
  was 8/8 versus 7/8 and tokens were 1.928M versus 1.863M. All 16 records passed structure,
  cohort/manifest binding, and fresh replay; the only overall audit failure was one disabled
  per-episode token-cap stop. This proves model-selected staging can trigger checkpoint without a
  producer-count Harness gate, but it does not show accuracy benefit. Keep it diagnostic-only; do
  not promote, remove checkpoint, or increase commit pressure from this sample. See
  `docs/reports/evaluation/CHECKPOINT_RELALG_MODEL_CHOICE_V2_VS_DISABLED_GATE8_20260811_ZH.md`.
- `model-choice-commit-v3` then required a successful `read_rows`/`inspect_column` verification
  before every candidate commit. On a disjoint fresh Gate8 it again produced zero checkpoint
  attempts across 8/8 tasks and 90 turns, scored 2/8 versus 3/8 disabled, used 2.133M versus 1.624M
  tokens, and had 11 versus six errors. All 16 structures and fresh replays passed; two v3 records
  stopped at the per-episode token cap. Freeze v3 as an over-suppressed manipulation failure. Do
  not add more mechanical preconditions; the next diagnostic must return to v2-style semantic
  model choice with only narrow exclusions for unresolved post-error state and useless late
  commits. See
  `docs/reports/evaluation/CHECKPOINT_RELALG_MODEL_CHOICE_V3_HOLDOUT_GATE8_20260811_ZH.md`.
- `model-choice-commit-v4` removed that mandatory perception and retained only two model-side
  vetoes for an unresolved error correction or a useless checkpoint with one relation operator
  remaining. A fresh candidate-only Gate4 nevertheless produced zero checkpoint attempts across
  4/4 tasks and 36 model turns; it scored 2/4 `bird-set`, 1/4 strict artifact, and 3/4 legal, using
  740,755 tokens. All four structures, identities, batch controls, and fresh replays passed. The
  preregistered exposure gate failed, so no disabled control was launched. Freeze v4 as another
  underexposed prompt manipulation. `model-choice-commit-v5` restores one concise positive
  model-selected trigger while keeping Harness eligibility `none`: the model privately selects a
  reusable semantic relation, then must commit when it exists and at least two distinct relational
  decisions remain; unresolved correction and one-operator-to-answer remain narrow exceptions.
  v5 is implementation-complete but has no live result until its independent exposure gate runs.
  See
  `docs/reports/evaluation/CHECKPOINT_RELALG_MODEL_CHOICE_V4_EXPOSURE_GATE4_20260811_ZH.md`.
- `model-choice-commit-v5` restored checkpoint exposure on a fresh paired Gate4: 2/4 tasks made
  five accepted commits with Harness eligibility still `none`. It scored 1/4 `bird-set` versus
  disabled 2/4 (one paired regression, no gain), with both arms 3/4 legal and 0/4 strict. On the
  two checkpoint-used tasks v5 was 0/2 versus 1/2, used two extra turns, and 2.2% more tokens.
  One trajectory committed twice consecutively with no intervening relation production; another
  made a late second commit that had only three tool actions left to amortize the reset. Keep v5
  diagnostic-only. `model-choice-commit-v6` preserves the first model-selected trigger but makes
  later commits model-side cost-aware: never consecutive, at least one new successful relation
  producer since the prior checkpoint, and at least four expected tool actions remaining. Harness
  eligibility remains `none`, max checkpoints remains eight, and restore remains disabled. See
  `docs/reports/evaluation/CHECKPOINT_RELALG_MODEL_CHOICE_V5_VS_DISABLED_GATE4_20260811_ZH.md`.
- A fresh `model-choice-commit-v6` versus v5 Gate4 preserved identical outcomes (2/4 correct,
  1/4 strict, 4/4 legal) while reducing accepted commits from 8 to 4, turns from 53 to 44, and
  provider tokens from 860,854 to 717,127 (-16.7%). Checkpoint coverage remained 3/4. However,
  v6 still produced one consecutive accepted commit with no intervening non-control action,
  despite an explicit prompt prohibition; v5 produced this defect on two tasks. All eight records
  and audits passed. Freeze v6 without expansion: prompt-only guidance cannot guarantee the
  no-progress invariant. The next isolated change should reject only a commit made in an empty
  phase since the previous accepted checkpoint. It must not count producers, select a turn, or
  otherwise transfer semantic boundary ownership from the model to the Harness. See
  `docs/reports/evaluation/CHECKPOINT_RELALG_MODEL_CHOICE_V6_VS_V5_GATE4_20260811_ZH.md`.
- The user-authorized historical fixed-200 expansion then ran `model-choice-commit-v6` with
  Atomic `semantic-v2` on all 200 frozen version24 task identities. V6 scored 142/200 `bird-set`
  versus historical Atomic version24 at 145/200; paired outcomes were 13 v6-only and 16
  version24-only (`p=0.7111`). V6 used checkpoint on 107 tasks with 130 accepted commits and no
  consecutive commits, but scored 70/107 versus version24's 71/107 on those same IDs. Legal
  termination was 188 versus 197, tool errors 96 versus 29, and provider tokens 28.61M versus
  8.42M. All 200 records passed structure and fresh replay; one 44-record shard correctly failed
  only its batch-level audit after an in-flight response crossed the local token cap, and the
  never-requested final position was completed in a separate audited top-up. This historical
  comparison is not a fresh single-variable checkpoint ablation and does not establish checkpoint
  benefit. Keep v6 diagnostic-only. See
  `docs/reports/evaluation/CHECKPOINT_RELALG_MODEL_CHOICE_V6_HISTORICAL_ATOMIC_FIXED200_20260811_ZH.md`.
- The original atomic protocol remains supported for ongoing RL work, frozen controls, and exact
  reproduction. Version54 / `native-tool-bundle` remains a diagnostic control/reproduction line
  and does not gain RL admission. Do not delete either line, mix their
  trajectories or result directories with `checkpoint-relalg`, relabel their checkpoints, or use
  them as the starting point for new protocol work. Atomic shared prompt semantics remain in
  `src/sft/prompt_contract.py`; atomic protocol/validation remains in `src/sft/protocol.py`; the
  cross-scheme index is `docs/current/tool_protocol.md`.
- In the frozen predecessor lineage, `version51` is the provider-tool-call baseline,
  `version52` is the frozen compact-prompt predecessor, and `version53` is the frozen reviewed-
  prompt control for the version54 no-plan ablation; version54 is the frozen no-plan diagnostic.
  `version26` is now historical checkpoint-560 control only: do not
  start new feature work from it or describe it as the current destination. The original atomic
  local diagnostic baseline remains `version39`; `version40`-`version50` are frozen
  prompt/interface/provider diagnostics. None may be mixed into version26 result directories.
  Version26 retains version25's prompt-role and
  public-contract refactor, but replaces the model-visible tagged action carrier with one
  non-empty `<think>` block followed directly by a strict raw JSON action object. The former
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
  boundaries are unchanged. `version21` was a deterministic-resolution ablation that was audited
  but not promoted. `version22` keeps the version20 public calls and argument schemas unchanged and
  adds compact harness-derived feedback for multi-column projection collapse, aggregate
  grain/layout/count semantics, and actual unmatched rows after a single-edge left join. Only the
  latest structural feedback is resident. Its frozen 16-task pilot preserved all eight controls
  but recovered only 2/7 unambiguous targets, below the expansion gate. `version23` leaves tools,
  arguments, execution, and resident-state boundaries unchanged and makes the existing feedback
  operationally explicit about named output fields, join multiplicity/global aggregate grain, and
  unmatched left-join rows. Its frozen 16-task pilot recovered 3/7 targets with 1/8 control
  regression, 16/16 legal termination, and a +0.125 mean-action change versus the original; this is
  a weak pass authorizing only the frozen 50-task gate. The completed 50-task gate scored 35/50
  versus the paired original 30/50: 6/20 capability recoveries, 29/30 controls retained, 50/50
  legal termination, unchanged 6.96 mean actions, and six recoverable errors. It missed the
  predeclared 7/20 recovery and +7 net-gain thresholds, so do not launch a version23 fixed-200 run.
  `version24` is an engineering-semantic cleanup, not an accuracy promotion: it removes
  advice-heavy feedback and the global latest-feedback sidecar, then records one validated,
  fact-only `relation-derivation-v1` object on every derived table. The schema covers every active
  table-producing tool and is implemented in `src/harness/relation_derivation/`, separate from
  SQL execution, provenance, resident storage, protocol rendering, and model policy.
  Its staged DeepSeek v4 Flash evaluation used the frozen 200-task cohort, rolling history,
  optional plan, JSON Output, and `bird-set`: the first 50 were 42/50 versus the paired version20
  39/50, then the full result was 145/200 versus 143/200. The +2 is not significant (10 gains,
  8 regressions); legal termination fell from 199 to 197, process errors rose from 24 to 29, and
  total tokens rose 5.8%. It remains an engineering boundary, not a 75%-validated SFT protocol.
  `version25` keeps the version24 tools, state, execution, and relation-derivation semantics, but
  separates prompt roles: one concise student runtime contract is shared by SFT export, evaluation,
  and RL, while an external teacher receives that same contract plus generation-only guidance and
  examples. `MODEL_ARG_SCHEMA` is the public model-call schema; replay-only compatibility remains
  separate. Manifests record teacher/student prompt hashes and the public tool-schema hash. A
  matched 400-episode causal pilot rejected adding formal JSON grammar to the student prompt:
  full grammar induced 154 plan calls at checkpoint 10, while restricting grammar to high-entropy
  tools removed that loop but reached 0/10 legal termination at checkpoint 20 with 10/10 joins
  invalid. Keep the 1,050-token canonical student prompt; test action coverage through causal
  supervision rather than more prompt prose. See
  `docs/reports/sft/BIRD_STUDENT_FORMAL_PROMPT_CAUSAL_GATE_20260725.md`.
  `version26` preserves all version25 tools, arguments, execution, state, grounding, and
  relation-derivation semantics. The active strict parser accepts only `<think>...</think>` plus
  one direct `{"tool":...,"arguments":...}` object; retired tagged actions are parsed only by a
  named offline migration path and re-rendered without changing structured actions or reasoning.
  `version27` added stable structured carrier/argument error details and rejected only two
  consecutive parsed actions whose canonical `tool + arguments` were exactly equal. Its fresh
  48-task checkpoint-560 gate scored 12/48 versus a matched fresh version26 control at 13/48;
  legal termination rose from 21 to 24 and mean steps fell from 11.00 to 7.58, but accuracy did
  not improve. `version28` retains the original rejection root cause across an identical retry
  and states that identical `read_subtable` calls cannot paginate because the tool has no
  offset/cursor. It scored 12/48 with 25 legal terminations and 8.06 mean steps on the same gate.
  Do not expand either version to 120/240 or full greedy, and do not use them as SFT sources.
  See
  `docs/reports/evaluation/BIRD_CP560_ADJACENT_REPEAT_FEEDBACK_GATE_20260727_ZH.md`.
  `version29` tested typed row expressions and explicit rank/read offsets but was not promoted.
  `version30` restores the version28 public action surface and adds teacher-only causal evidence
  discipline; it has no accuracy promotion and does not alter frozen version26 artifacts.
  `version31` keeps the version30 public tool schema and valid-call execution semantics, but adds
  state-aware pre-execution validation for table handles, table/column ownership, predicate
  operands, join-edge columns, and terminal evidence handles. Invalid references now return
  structured `argument_validation_error` feedback before SQLite executes; version31 is
  diagnostic-only pending paired failure/control gates. `version32` keeps those semantics and adds
  only a concise instruction on `unknown_column` errors to choose the correct table or column from already
  observed schemas; it does not claim global column absence or expose unseen schema. `version33`
  additionally prepares `project` expressions against the current relation before registering any
  derived state, translating missing columns or malformed expressions into structured validation
  errors instead of raw SQLite failures. `version34` keeps the public contract and relational
  semantics unchanged while lazily materializing connection-local copies of derived relations
  smaller than 50,000 rows when they are reused as join inputs; this prevents repeated nested SQL
  evaluation without changing model-visible handles or values. `version35` keeps version34's
  public contract, validation, and execution semantics. An exactly repeated adjacent call is still
  rejected, recorded, and charged to the shared action budget, but `no_progress_error` is exempt
  from the generic three-errors-per-type early abort. The episode may recover until `max_steps`;
  all other recoverable error limits remain unchanged. The first version35 full-evaluation launch
  exposed a runner inconsistency: `rollout.py` used the exemption but `rollout_passk.py` retained
  the generic limit. That partial artifact is frozen. `version36` applies the same shared
  `error_limit_reached` policy to both atomic runners; public calls and model-visible feedback are
  unchanged. `version37` keeps version36's execution/error policy and adds two narrowly typed
  capabilities. `read_subtable` remains a read-only perception tool but can now select rows with
  typed conditions, exact-column ordering, and deterministic ordered offsets; it never creates a
  filtered handle. `project` accepts only two typed row-date expressions,
  `date_diff_days(start,end)` and `extract_year(date)`, alongside its existing string expressions.
  The action-block and relational-program schemes remain frozen. Version37 is diagnostic-only
  pending a larger protocol gate. Its frozen seven-task DeepSeek v4 Flash recovery diagnostic
  compared recent-4 with first-5 plus recent-5 legal pairs after revalidating every historical
  anchor error under the active version37 contract. Recent-4 scored 5/7 versus 4/7, had one paired
  gain and no regression, used 76 versus 93 legal teacher actions, and used 709,739 versus
  1,051,252 tokens. Both policies exercised typed row reads and the date task exercised
  `date_diff_days`; teacher continuations had no adjacent exact repeats. Keep recent-4 as the
  active history policy and do not promote these diagnostic trajectories to SFT. See
  `docs/reports/evaluation/BIRD_VERSION37_ROW_READ_DATE_HISTORY_GATE7_20260728_ZH.md`.
  `version38` keeps version37's student runtime prompt, public tools, argument schemas, execution,
  state, feedback, carrier, and recent-4 history policy unchanged. It adds only external-teacher
  generation guidance derived from the frozen 54-failure audit: obey explicit question/external-
  knowledge mappings, do not invent singleton/time/aggregate restrictions, fix population and
  row grain before relational commitments, preserve exact aggregation units, investigate
  anomalous observations rather than rationalizing them, and audit the exact terminal table
  against requested output slots. Its frozen 16-target/16-control DeepSeek v4 Flash prompt gate
  required an infrastructure-only retry because the initial three-request budget left 19 empty
  provider carriers and one disconnected request. With a ten-request retry budget, all 32 tasks
  formed semantic completions: version38 scored 18/32 versus paired version37 at 16/32, with six
  target recoveries, four control regressions, 32/32 legal termination, seven process errors, and
  exact paired `p=0.7539`. It missed every semantic expansion threshold. Keep version38
  diagnostic-only, do not expand it, and do not admit its prompt-only trajectories to SFT. See
  `docs/reports/evaluation/BIRD_ATOMIC_VERSION38_SEMANTIC_DISCIPLINE_GATE32_20260729_ZH.md`.
  `version39` keeps version38's public tools, argument schemas, execution, canonical resident
  state, replay semantics, feedback, carrier, and recent-4 history policy unchanged. It changes
  only model-visible resident-state rendering: completely equivalent `read_subtable` observations
  are shown once with all equivalent source step ids, and groups of at least two unreferenced,
  unread zero-row `condition_filter` handles sharing one input and output shape are folded into a
  fact-only summary that retains handle, creator step, predicate, input, and columns. Referenced,
  read, or singleton empty handles stay fully rendered. Version39 is diagnostic-only pending a
  context-efficiency and behavior-preservation gate; frozen version26 runs and artifacts remain
  unchanged.
  `version40` is an isolated external-teacher diagnostic selected with
  `--atomic-protocol-version version40 --diagnostic-only`. It keeps version39 execution, state,
  grounding, feedback, carrier, and the exact multi-edge `join_tables` rule. It removes `plan`
  from its public surface, renames the read-only row observer to `inspect_rows`, uses one concise
  layered prompt, retains all successful and rejected reasoning, and limits exact successful calls
  plus unabridged observations to recent-4. Its frozen first-50 gate on the version24 fixed-200
  cohort scored 37/50 versus the paired version24 42/50, with 0 gains, 5 regressions, 50/50 legal
  termination, nine process errors, and exact paired `p=0.0625`. All five new regressions selected
  the correct entity/row set but cited the wrong output columns, order, or representation. Do not
  run the remaining 150 or use version40 for SFT/RL. See
  `docs/reports/evaluation/BIRD_ATOMIC_VERSION40_CONCISE_HISTORY_GATE50_20260729_ZH.md`.
  `version41` keeps every version40 public tool, argument, execution, state, feedback, carrier,
  join, and full-reasoning recent-4 history behavior unchanged. It adds one prompt-only module:
  a consolidated exact-output contract restoring separate-field and ID/code representation
  boundaries, plus legal correction examples for the five tool families that produced version40
  Gate50 process errors. Its frozen 8-target/8-control output-shape Gate16 scored 10/16 versus
  paired version40 8/16, with two gains, no regressions, 8/8 controls retained, 16/16 legal
  termination, and zero versus six process errors. It recovered only 2/8 targets, below the
  predeclared 4/8 threshold. Do not expand it to the fixed first 50 or use it for SFT/RL. See
  `docs/reports/evaluation/BIRD_ATOMIC_VERSION41_OUTPUT_CORRECTION_GATE16_20260729_ZH.md`.
  `version42` keeps every version41 nonterminal tool and behavior unchanged, but requires
  `answer_from_context.evidence` to contain a source table plus an ordered non-empty list of exact
  existing columns. The harness deterministically projects only those grounded columns before
  scoring and records the lowering; it never consults question semantics or gold. Version42 is
  diagnostic-only and has no SFT/replay admission. Its frozen output-shape Gate16 scored 12/16
  versus version41 10/16, with two gains, no regressions, 4/8 targets correct, 8/8 controls
  retained, 16/16 legal termination, and zero terminal projection failures. It nevertheless
  produced four process errors versus the allowed two, including two exact-column rejections on
  uniquely resolvable bare names. Do not expand version42. See
  `docs/reports/evaluation/BIRD_ATOMIC_VERSION42_TERMINAL_COLUMNS_GATE16_20260729_ZH.md`.
  `version43` keeps every version42 behavior and changes only terminal column-name resolution.
  Full logical names match first; a bare name may match a dotted logical-column suffix only when
  exactly one candidate exists in the cited table. Ambiguous, missing, and incorrectly qualified
  names remain structured errors. The resolver never consults the question, external knowledge,
  model reason, or gold and cannot alter rows, values, or declared order. Version43 is
  diagnostic-only and has no SFT/replay admission. Its frozen Gate16 scored 13/16 and passed every
  local threshold, but the original frozen first-50 scored 39/50 versus paired version24 42/50:
  one gain, four regressions, 50/50 legal termination, zero terminal projection errors, and nine
  process errors. All 12 exercised unique-bare resolutions succeeded, so the resolver fixed its
  narrow interface issue without producing a general accuracy gain. Do not run the remaining 150.
  A fresh second attempt on each of its 11 first-50 failures recovered three tasks, making the
  verifier-selected union 42/50, but the recovery gate still failed: only 10/11 fresh attempts
  terminated semantically and they produced seven process errors versus the allowed three. The
  selective two-attempt total used 1,588,917 tokens and 346 model actions, slightly more tokens and
  17.7% more actions than version24's single-attempt 42/50. Do not expand version43 K=2.
  See
  `docs/reports/evaluation/BIRD_ATOMIC_VERSION43_UNIQUE_BARE_TERMINAL_COLUMNS_20260729_ZH.md`.
  `version44` returns to version39's full prompt/recent-4/exact-table terminal semantics, removes
  public `plan`, renames the row observer to `inspect_rows`, adds deterministic read-only
  `search_values(table, query, column?, limit?, offset?)`, and exposes BIRD semantic name and
  description only through `inspect_column`. Its frozen DeepSeek v4 Flash paired Gate50 scored
  **38/50** versus version39 at **39/50**, with four gains, five regressions, 50/50 legal
  termination, and three versus two process errors. Search was used on 19 tasks, but both protocols
  scored 16/19 on that subset. The current implementation also scans every distinct value before
  fuzzy ranking; two approximately 1.98M-distinct-title searches took about 380 seconds each.
  Do not run the remaining 150 or use version44 for SFT/RL before bounded candidate retrieval is
  implemented and the same Gate50 is rerun. See
  `docs/reports/evaluation/BIRD_ATOMIC_VERSION44_SEARCH_VALUES_PAIRED_GATE50_20260730_ZH.md`.
  `version45` keeps version44's public calls and all non-search semantics but routes search through
  isolated deterministic `bounded-v1`: SQLite exact/case-insensitive-exact recall first, otherwise
  at most 4,096 stable token/trigram candidates per column before fuzzy ranking. Exact hits suppress
  broader alternatives; observations expose candidate scope and truncation. Offline replay of all
  22 version44 Gate50 searches retained 22/22 top candidates and cut their total search time to
  1.033 seconds; the two approximately 1.98M-distinct-title cases fell from about 380 seconds each
  to 0.374 and 0.132 seconds. After provider recovery, the current-hash fresh Gate50 scored
  **37/50** versus frozen version39 at **39/50**, with two gains, four regressions, 50/50 legal
  termination, and three versus two process errors. All four regressions were non-search
  semantic/output failures. Wall time fell from version44's 405.83 seconds to 317.21 seconds, so
  the latency defect is fixed but accuracy is not promoted. Do not run the remaining 150 or use
  version45 for SFT/RL. See
  `docs/reports/evaluation/BIRD_ATOMIC_VERSION45_BOUNDED_SEARCH_PAIRED_GATE50_20260730_ZH.md`.
  `version46` returns to version39's public tools and behavior and changes only model-visible
  resident derivations into Harness-authored single-line handle cards. `version47` additionally
  keeps the latest successful observation full while replacing resident row payloads with read
  cards that require re-observation for exact values. `version48` adds only external-teacher
  interpret-before-act guidance. Their frozen read-heavy Gate16 used the updated DeepSeek v4 Flash;
  one task with an unresolved version46 provider failure was excluded from every arm, leaving 15
  semantic pairs. Version39/version46/version47/version48 scored **8/15, 8/15, 9/15, 9/15**.
  Version47 had two target gains and one control regression versus version46 (`p=1.0`) but legal
  termination fell to 13/15; reads rose from baseline 27 to 71 with 24 exact re-reads. Version48
  had one gain and one regression versus version47, used 1.49x baseline actions and 1.62x tokens,
  and produced 35 exact re-reads. Offline fixed-200 rendering saved only 478/233 characters on the
  final turn for version46/version47, while version48 added 145. `version49` implements the next
  dependency-aware variant without model-authored branch parameters: it derives active handles
  from references in the retained recent-4 action/observation/error structures, recursively keeps
  exact rows for their derivation dependencies, and archives rows only outside that closure. Its
  raw Gate16 scored 9/16 with 15/16 legal termination. On the same 15-task semantic subset it
  scored **8/15**, exactly matching every version39 outcome, retaining all 8/8 controls and
  recovering 0/7 targets. It used 138 actions, 26 reads, zero exact re-reads, and 1,169,008 tokens
  versus baseline 146, 27, zero, and 1,201,127. It fixed the version47/48 churn cases but had no
  capability gain. Offline fixed-200 next-action literal loss fell from version47's 12 turns/20
  values to one dormant-branch turn/four values; that raw version49 task was correct but still used
  10 reads and two exact re-reads. A later frozen Gate8 K=3 stability check rejected the apparent
  single-run efficiency: version39 and version49 had identical per-run correct counts and both
  totaled 20/24, with all 15/15 controls retained, but version49 used 23.0% more mean actions,
  71.9% more row reads, and 28.7% more tokens. It produced three max-step failures versus zero for
  version39; on the two hard context-sensitive tasks, reads repeatedly revisited the same evidence
  and failed to advance the relational program. Version49 failed five of nine preregistered
  stability checks. Do not expand version46-version49 to Gate50, use them for SFT/RL, or preserve
  version49 as a candidate renderer. Further work should first establish an evidence-sufficiency
  invariant or target semantic/output errors rather than compressing more resident rows. See
  `docs/reports/evaluation/BIRD_ATOMIC_CONTEXT_HANDLE_CARD_ABLATION_GATE16_20260731_ZH.md` and
  `docs/reports/evaluation/BIRD_ATOMIC_VERSION49_CONTEXT_STABILITY_K3_GATE8_20260731_ZH.md`.
  A later strict single-variable audit rebuilt historical `version24` from commit `22196e2`:
  its current-hash fresh Gate50 reproduced **41/50** versus the original **42/50**, with 47/50
  identical task outcomes. On that isolated branch, adding only bounded
  `search_values(table, query, column?, limit?, offset?)` while retaining `plan`,
  `read_subtable`, raw schema, and exact-table terminal scored **39/50**. It had two gains and
  four regressions versus the fresh baseline (exact paired `p=0.6875`), used search on 13 tasks
  with 10/13 correct versus 11/13 for baseline, and increased actions 9.4% and tokens 14.3%.
  Search recall itself had no execution errors or candidate truncation; the only search-used
  paired regression returned the correct exact literal but kept an extra ranking column. Do not
  promote or expand this ablation. See
  `docs/reports/evaluation/BIRD_VERSION24_BOUNDED_SEARCH_SINGLE_VARIABLE_GATE50_20260730_ZH.md`.
  `version50` is a completed, rejected provider-carrier diagnostic. It returns to every version39
  public tool, argument, execution, canonical state, recent-4 rendering, exact-table terminal, and
  teacher semantic-guidance rule, but exposes those same 12 calls to DeepSeek through the official
  native `tools`/`tool_calls` interface. The provider prompt contains no positive raw-action
  carrier example; canonical teacher examples are rendered as function name plus arguments only.
  Thinking-mode history must return the original `reasoning_content`, tool call, and call id, while
  harness observations use the matching `role=tool`. Because DeepSeek thinking mode rejects
  `tool_choice=required`, the request uses `auto`; zero-call/provider-shape defects receive bounded
  client retries, while model-authored multiple calls, non-empty assistant content, unknown
  functions, or malformed argument JSON are rejected before execution as state-preserving semantic
  protocol errors. Canonical stored trajectories remain `<think>` plus raw JSON for replay/export
  compatibility. Its user-authorized frozen Flash fixed-200 run scored **133/200** versus
  historical version24 **145/200**, with 13 gains, 25 regressions, 120 both correct, 42 both wrong,
  and exact paired `p=0.0730`. Legal termination fell from 197/200 to **183/200** (2 gains,
  16 regressions, `p=0.00131`); process errors rose from 29 to **166** and tokens from 8,420,861
  to **16,158,278**, while actions were nearly unchanged at 1,487 versus 1,490. All 133 correct
  trajectories passed fresh replay and structural/no-leak/native-lowering audits, but this does not
  rescue the failed behavior and reliability gates. Do not promote version50, use its trajectories
  for SFT/RL, or replace the JSON-Output default.
  `version51` is the frozen behavior baseline and the first `native-tool-bundle` scheme. It keeps
  version39's 12 primitive functions, harness execution, resident state, and exact-table terminal,
  but follows the official provider semantics instead of forcing one call: one actual assistant
  turn may contain 1..8 native `tool_calls`, and the client returns one `role=tool` result for each
  call id in provider order. Non-empty assistant content is retained for audit but is never
  executable evidence or a training target. Every call is statically and state-validity checked
  against the bundle's shared pre-state before any call executes, so a later call cannot consume a
  handle created by an earlier call that the model had not yet observed. Primitive calls execute
  and are audited in order; `answer_from_context` must be the sole call in its turn. Stored steps
  carry both `model_turn_index` and `native_tool_call_id`; they must not be flattened into fake
  assistant turns. Version51 remains diagnostic-only until a scheme-aware exporter and explicit
  training-admission gates pass. Its frozen carrier-failure Gate32 scored **22/32**
  versus version50 **16/32**, with six gains, zero regressions, 32/32 versus 26/32 legal
  termination, 11 versus 37 errors, and 1.0846x tokens. It recovered 6/16 targets and retained
  16/16 controls; 37 multi-call turns and 92 non-empty-content turns produced zero carrier
  rejection. All 22 correct trajectories replayed; structure and the full 32-task/271-call
  no-leak, lowering, call/result order, reasoning-history, state-mutation, and bundle-pre-state
  audits had zero issues. The subsequent fixed-200 scored **147/200** with **200/200 legal**,
  54 process errors, 1,414 real model turns, 1,650 primitive calls, and 16,165,179 tokens. Against
  version50 it had 20 gains and six regressions (`p=0.00936`), 17 legal gains and no regression,
  and reduced carrier protocol errors from 124 to two. Against historical version24's 145/200 it
  had 15 gains and 13 regressions (`p=0.8506`): accuracy is tied rather than promoted, while errors
  remain 54 versus 29 and tokens 1.92x. All 147 successes replayed, and structural plus full
  200-task native-history/no-leak audits had zero issues. Version51 therefore passed as the
  provider-tool-call behavior baseline for its frozen lineage, not as an SFT/RL protocol. New
  protocol experiments now branch from `checkpoint-relalg-v1`; do not flatten version51 multi-call
  turns into atomic training targets. See
  `docs/reports/evaluation/BIRD_VERSION51_NATIVE_TOOL_BUNDLE_GATE32_20260805_ZH.md` and
  `docs/reports/evaluation/BIRD_VERSION51_NATIVE_TOOL_BUNDLE_FIXED200_20260805_ZH.md`.
  `version52` keeps version51's primitive tools, native carrier, bundle-pre-state validation,
  execution, resident state, recent-four provider-turn history, and hard sole-terminal rule. It
  changes only the native prompt profile and fact-only statistics. The API-supplied JSON schemas
  are now the sole argument-shape authority, so the system prompt no longer repeats the textual
  tool catalog, long per-tool teacher elaborations, rewritten call cookbook, or three overlapping
  bundle-carrier clauses. It retains resident-state, relational/output, semantic-decision, and
  error-recovery semantics. Soft policy defaults to one call, normally caps a bundle at three,
  favors independent perception, permits parallel filters only for grounded competing hypotheses,
  and normally isolates relational operators. `answer_from_context` remains a hard single-call
  bundle. Error calls and their structured tool feedback remain in causal provider history; the
  later corrected bundle, not the rejected bundle, is the positive SFT target boundary. Per-call
  execution status, later handle references, error-feedback preservation, and next-turn recovery
  are recorded only as RL statistics, not rewards. On the frozen version51 fixed-200 request
  transcripts, counterfactually replacing only the system prompt reduces message characters
  45.4%; fixed prompt+native-schema characters fall from 30,937 to 16,769. This is a static audit,
  not a behavior/token result. Version52 remains diagnostic-only until a paired live gate and a
  scheme-aware exporter pass; its trajectories must not be flattened into atomic targets.
  `version53` keeps every version52 tool, argument, carrier, execution, state, history, feedback,
  and terminal semantic unchanged and changes only the prompt profile after an external review.
  The shared student/teacher runtime kernel now states that database/schema/value/metadata text is
  task data rather than instructions; catalog edges are not cardinality proof; join multiplicity
  should be inspected when it can change aggregation/ranking or when observations are anomalous;
  copied source fields keep stored representation while derived metrics keep exact tool-produced
  results; correctness outranks call minimization; and jointly required independent filters over
  different resident handles may share a bundle. Teacher-only guidance adds pre-commitment
  population/grain/output-slot checks, anomaly investigation, concise reasoning, precise duplicate
  read avoidance, and error-specific recovery. Harness implementation prose and any requirement to
  read an entire final table are omitted. The teacher prompt is an exact student-prompt prefix plus
  teacher-only guidance, and remains no longer than version52. Version53 is prompt-only,
  diagnostic-only, and has no live accuracy claim or SFT/RL admission.
  `version54` keeps version53's student runtime prompt, carrier, every non-plan function and
  argument schema, bundle-pre-state validation, execution, resident state, recent-four provider-
  turn history, feedback, and terminal semantics. It removes only the provider-visible `plan`
  function plus the now-inapplicable teacher-only plan-evidence sentence. The underlying harness
  retains replay compatibility; a hallucinated plan call is a structured state-preserving
  `unknown_tool` error. Version54 was introduced by `tool-scheme-registry-v11` and is carried by
  registry v12 as a frozen diagnostic. Its v54-only Prefix200 absolute acceptance pilot was
  preregistered but has no recorded result; do not infer one. Version54 multi-call turns must not be
  flattened into atomic training targets, and its lineage remains available only for compatible
  ongoing work and reproduction. New tools and experiments use `checkpoint-relalg-v1` rather than
  incrementing this version line.
- Training export applies `causal-empty-result-target-filter-v1`. A successful intermediate call
  whose tool output explicitly has `row_count=0` remains in the executed trajectory and later
  causal context but is marked `sft_target_eligible=false`; it receives no positive target loss.
  If the table cited by `answer_from_context` has `row_count=0`, exclude the entire trajectory from
  SFT. Do not confuse an empty relation with a grounded one-row scalar whose stored value is zero:
  COUNT=0 in a 1x1 table remains eligible. Generation, cohort audit, and exporters must record and
  independently enforce this boundary; never delete an empty observation needed by a later
  recovery target.
- Training-task admission applies `gold-denotation-nonempty-task-filter-v1` before rollout. Hidden
  gold SQL is executed only by the local read-only SQLite harness and only an at-least-one-row bit
  is retained; the source `gold_exec_results` placeholder is never trusted. Zero-row tasks and
  tasks with execution/input errors are excluded before any student/teacher request, while a
  one-row scalar containing value zero remains eligible. Gold SQL, result rows/values, and private
  empty/nonempty status must never be model-visible or sent to an external provider. The active
  source pools are the frozen 6,599-task `bird_train_filtered_nonempty_v1` and 5,915-task
  `bird_train_tool_compatible_nonempty_v1`; the active teacher candidate is
  `bird_train_atomic_teacher1500_v2_nonempty`. It preserves all v1 ids/order and the same task-file
  hash because all prior 1,500 were certified nonempty. New training-task selectors must require a
  hash-bound successful filter manifest; the unfiltered bypass is historical reproduction only.
- Non-atomic scheme implementations are package-owned under `src/tool_modules/`:
  checkpoint-relalg, action-block, relational-program, direct-SQL-search, iterative-SQL, and
  native-tool-bundle each own their
  protocol directory;
  `sql_common` is the explicit shared immutable-SQL runner and `tool_modules.registry` is the
  exclusive cross-scheme registry. `src/eval` contains cross-scheme
  evaluation infrastructure plus thin historical CLI/import aliases only. New code, launchers,
  manifests, tests, and docs must use `tool_modules.*` canonical paths and must not import the old
  flat `batch_plan_protocol`, `evaluate_batch_plan`, `relational_program_protocol`,
  `evaluate_relational_program`, `direct_sql_search_protocol`, `iterative_sql_protocol`, or
  `iterative_sql`, `atomic_version51`, `deepseek_native_tools`, or `audit_native_tool_bundle`
  aliases, nor the old `tool_schemes` registry alias, from inside `src/tool_modules/`. See
  `docs/current/architecture.md`.
- The canonical atomic model action contains exactly one non-empty `<think>` block followed by one strict
  raw JSON object with exact `tool` and `arguments` keys. A provider-native adapter may carry the
  same authored reason and action in separate reasoning/function-call fields, but its API-facing
  prompt and history must describe only that selected carrier; the canonical form is reconstructed
  without editing function arguments before the shared parser and harness. Version51-version54 are named
  non-atomic exception: its structured provider assistant turn is the authoritative carrier and is
  never reconstructed as multiple causal text turns.
- The retained atomic context contract remains bounded recent legal history with
  `history_turns=4`: catalog,
  question, and optional external knowledge are followed by at most four successful
  assistant/observation pairs, and the latest observation carries rebuilt resident state plus
  optional `LAST TOOL ERROR`. Rejected assistant text is never added to the retained atomic
  history. Version40 is an explicit diagnostic exception for reasoning only: rejected reasons are
  retained with `status=rejected`, while the authored failed call is represented by the
  harness-owned attempted action/error. Version37 may render first-5 plus recent-5 for a paired
  diagnostic, but that policy is not promoted unless it beats recent-4 on the same frozen tasks.
  Version51-version54 instead bound history by four provider assistant turns; each retained assistant turn
  is followed by all of its matching tool-result messages, including structured per-call errors.
- Retained atomic tools: `plan`, `describe_table`, `inspect_column`, `read_subtable`,
  `condition_filter`, `project`, `scalar_compute`, `join_tables`, `group_aggregate`,
  `extreme_value_select`, `set_op`, and `answer_from_context`.
- Forward checkpoint-relalg surfaces: every mode has `describe_table`, `inspect_column`,
  `read_rows`, `commit_checkpoint`, `restore_checkpoint`, and `answer`; direct adds read-only
  `execute_sql`; atomic adds `filter_rows`, `project`, `join`, `aggregate`, `distinct`,
  `set_operation`, `sort`, `limit`, and `add_rank`; hybrid has both data-operation sets. Exact
  arguments come only from the mode's executable native schemas.
- Version54 exposes that same surface except that `plan` is absent; the other eleven functions and
  their argument schemas are unchanged.
- Version40 exposes the same set except that `plan` is absent and `read_subtable` is named
  `inspect_rows`; all other signatures and execution semantics are unchanged.
- The frozen experimental `action-block-v32` scheme does not change that atomic contract. A
  work turn is one top-level `action_block` with one to five ordered nonterminal primitive calls;
  termination is one separate top-level `answer_from_context`. Terminal calls cannot be nested or
  mixed with work. The model does not see `plan`, `update_plan`, dependency fields, handles before
  execution, or mutable environment state. The harness executes in list order, resolves only
  strict same-block backward `$id` references, maintains resident factual context and handles, and
  reports one full atomic output/error/blocked result per submitted call in model list order.
  The active adapter performs no spelling, schema, column, predicate, order, handle, or argument
  shape rewrite. Provider-specific carrier constraints are API-facing only. Its completed frozen
  200-task gate scored 144/200 and remains ineligible for SFT.
- The rejected `action-block-v33` diagnostic kept v32's structured action, primitive tools,
  ordered execution, full per-call feedback, backward `$id` references, and atomic audit boundary.
  It simplifies only the model-facing abstraction: a block is a short 1..8 sequential operation
  segment, not a program; the model declares no DAG, dependencies, result root, exports, plan,
  status, handle, or environment state. Its concise prompt lists every exact tool signature,
  gives one complete best-practice example, and requires the block to end whenever unseen feedback
  could change the next operation or argument. The external-teacher diagnostic permits up to
  40 model actions. Its frozen 20-task diagnostic scored 11/20 with 18 process errors and
  20 blocked descendants, so it was not expanded. It is diagnostic-only and ineligible for SFT.
- The completed `action-block-v34` diagnostic keeps v33's short 1..8 consecutive-operation blocks,
  full ordered per-call feedback, bounded history, and atomic credit boundary. It replaces only
  the public multi-edge `join_tables(base, joins[])` call with one unambiguous
  `join(left, right, left_on, right_on, how?)` edge; the harness deterministically lowers it to the
  frozen executor without semantic guessing and hides the private executor vocabulary from model
  prompts and feedback. Its frozen 20-task DeepSeek v4 Flash diagnostic scored **14/20**, with
  20/20 legal termination, seven process errors, two blocked descendants, 113 model turns,
  165 attempted primitives, and 479,019 tokens. It had three paired gains and no regressions
  versus v33 (11/20), and eliminated observed join-call errors, but remained below atomic
  version24 (16/20) and historical action-block v18 (17/20). It is diagnostic-only and ineligible
  for SFT; see
  `docs/reports/evaluation/BIRD_ACTION_BLOCK_V34_ONE_EDGE_JOIN_PILOT20_20260727_ZH.md`.
- The active `action-block-v35` diagnostic changes only the public `scalar_compute` operand
  carrier on top of v34. A literal is `{"value":...}`; a same-block one-cell result is
  `"$id.exact_column"`; and a prior resident one-cell result is `"step_id.exact_column"`.
  The adapter accepts a cell reference only after verifying a successful producer, exactly one
  source row, and one exact named output column, then deterministically lowers it to the frozen
  grounded `value_ref + column` executor carrier. It never chooses a row, resolves a suffix,
  aggregates, or guesses a producer. Evaluation and RL use the same validator/lowering path.
  Its frozen four-task DeepSeek v4 Flash gate scored 3/4 with 4/4 legal termination, one process
  error, zero blocked calls, and six verified scalar-cell lowerings. All three v34 zero-error
  controls remained correct, but the scalar target remained wrong after deriving both matching
  row durations and citing only one. It failed the predeclared expansion gate and must not be
  expanded or used for SFT. See
  `docs/reports/evaluation/BIRD_ACTION_BLOCK_V35_SCALAR_CELL_GATE4_20260727_ZH.md`.
- The separate experimental `relational-program-v6` scheme exposes only `observe`,
  `relational_program`, and `answer_from_context`; its prompt contains no atomic or action-block
  tool definitions. A program contains up to eight deterministic `filter`, `select`, `scalar`,
  `join`, `aggregate`, `rank`, or `combine` nodes. Source tables, prior resident tables/steps, and
  current-program nodes use disjoint `typed-relational-reference-v2` objects. Only current-program
  node references create DAG edges; the harness deterministically lowers the public references to
  internal atomic executors, derives and validates the DAG, topologically schedules execution, and
  propagates root errors to blocked descendants. Primitive nodes remain the execution/audit unit.
  Its frozen 20-task DeepSeek v4 Flash diagnostic scored 10/20 with 19/20 legal termination,
  eight process errors, three blocked descendants, 119 model turns, and 508,471 tokens. V3 scored
  12/20 on the same cohort; v4 halved process errors and reduced tokens 9.5% but lost two net
  correct tasks. Do not expand v4 or use it as an SFT/RL source; see
  `docs/reports/evaluation/BIRD_RELATIONAL_PROGRAM_V4_TYPED_REFERENCES_PILOT20_20260727_ZH.md`.
  V5 changes only typed scalar sources: scalar operands directly use `node`/`resident_step`
  objects with optional named columns, and predicates use `value_from`; private executor names and
  sigiled references are removed from model-visible errors. Its frozen two-target/four-control
  micro-gate scored 3/6: 0/2 target recoveries and 3/4 controls retained, with seven process errors
  and two blocked nodes. It failed its expansion gate. Do not expand or train on v5; see
  `docs/reports/evaluation/BIRD_RELATIONAL_PROGRAM_V5_TYPED_SCALAR_GATE6_20260727_ZH.md`.
  V6 changes no public call shape; it fixes only the deterministic lowering of a typed bare
  `on.left` column when a current-node join declares `base_role`, and records that lowering in the
  work graph. Its frozen one-target/three-control gate scored 3/4: all controls retained but the
  target remained wrong after 11 successful primitive actions. The live target did not exercise
  `base_role`; the bug fix is covered by deterministic SQLite tests only. Stop further prompt/schema
  expansion, and do not use v6 for SFT/RL; see
  `docs/reports/evaluation/BIRD_RELATIONAL_PROGRAM_V6_BASE_ROLE_GATE4_20260727_ZH.md`.
  The scheme has evaluation and causal-rollout plumbing only, was introduced in registry v4 and is
  carried forward by `tool-scheme-registry-v12`, and remains
  `diagnostic_only_pending_protocol_scale_gate`; it has no SFT exporter or RL environment.
- The separate direct-SQL-search scheme is the action scheme introduced by registry v4 and exposes
  exactly `search_values(query, table?, column?, limit?, offset?)` plus
  `execute_sql(sql, mode)`. Search is deterministic out-of-band bounded lexical retrieval over
  exact stored database values; it is not claimed to be theoretically inexpressible in SQL and
  does not create a relation. `execute_sql(mode="inspect")` returns read-only SQLite feedback;
  `mode="final"` is terminal and must reuse a previously inspected SELECT/WITH. Its frozen paired
  Gate16 on the first half of the version38 semantic cohort scored **7/16** with 14/16 legal
  termination versus fresh atomic version39 at **10/16** and 16/16. It used 538,791 versus
  1,359,421 tokens but 154 versus 141 actions, and had two max-step failures versus zero. Both
  value-search calls occurred in those failures; the sole paired gain did not use search. All
  seven successes freshly replayed and passed the independent structural audit. Keep it only as a
  diagnostic low-token SQL control; do not expand it or use it for SFT/RL. The active diagnostic
  `direct-sql-search-v2` preserves those two calls and the v1 compatibility path, but adds
  external-teacher semantic decision discipline, structured fact-only errors, bounded preview
  shape facts, a recent-exact-6/prior-cards-6 state renderer with resident-pointer history, and
  rejection of any exact previously successful action on the immutable database. Its disjoint
  Holdout Gate15 scored **6/15** versus fresh atomic version39 at **10/15**, with 15/15 legal
  termination in both arms, 88 versus 115 actions, and 327,401 versus 854,227 tokens. V2 had zero
  repeated actions and zero max-step failures, so its engineering optimizations worked, but it
  missed the accuracy expansion threshold by four tasks. The single search call occurred in a
  failure. Keep v2 as a diagnostic low-token SQL control; do not expand it or use it for SFT/RL. See
  `docs/current/direct_sql_search_tool_scheme_zh.md`.
- The separate active `iterative-sql-v6` scheme is carried by `tool-scheme-registry-v12` and exposes
  exactly `execute_sql(sql)` plus `submit_sql(sql)`. The prompt requires causal SQL exploration
  until enough schema/value/join/population/grain/output evidence is available, then requires the
  final SELECT/WITH to match a previously successful `execute_sql` query. Safety, prior-inspection,
  timeout, syntax, and SQLite execution failures return structured state-preserving `LAST SQL
  ERROR` feedback and the same episode may recover; only a successfully executed submission
  terminates for hidden `bird-set` scoring, and an executable wrong answer receives no verifier
  feedback. V3 added a strict schema-PRAGMA allowlist, exact-success no-progress rejection, compact
  resident SQL state, and hidden-verifier isolation. Its frozen 15-task Flash baseline scored
  7/15 with 15/15 legal termination. V4 keeps the same two calls and execution semantics, repeats
  exact question/external knowledge at the latest context boundary, strengthens binding output/
  formula/source/grain rules, and adds a deterministic syntax-only query-shape audit to successful
  SQL observations. On the same 15-task development set it scored 12/15 with 15/15 legal
  termination, six gains, one regression, 91 versus 97 actions, one versus four process errors,
  and 346,664 versus 341,223 tokens. All 15 outcomes passed fresh replay, structural, and hidden-
  input-key audit. Because v4 was designed after inspecting these same v3 failures, this is an
  in-sample optimization result; freeze v4 and require a new independent holdout before expansion.
  V5 keeps those public tools and execution semantics, but incorporates an external prompt review:
  QUESTION and external knowledge receive non-overlapping binding roles; database facts cannot
  invent semantic restrictions; bounded previews cannot establish absence/order/completeness/
  extrema/ties; exploration targets one fact or tightly related uncertainty set; alternative
  source tables are inspected only when relevant; top-N defaults to N rows unless ties are
  requested; and database-derived final answers must remain SQL-data-dependent. Its v5-only
  query-shape facts add `has_from` and `literal_only_select`, without semantic rejection. DeepSeek
  sees only native-reasoning/JSON transport instructions, not internal canonical-envelope details.
  Its frozen active-baseline300 Prefix20 DeepSeek v4 Flash gate is independent of the old fixed-200
  and initially scored 14/20 versus frozen v4 at 12/20. The completed Prefix50 scored **36/50**
  versus **34/50**, with four gains, two regressions, exact paired `p=0.6875`, 49/50 versus 50/50
  legal, 10 versus seven process errors, 263 versus 265 actions, and 984,082 versus 877,576 tokens
  (+12.1%). Both arms passed 50/50 fresh replay/structural/no-hidden-input-key audit. The two
  regressions both concatenated `full name refers to field1, field2...` into one string, while v4
  preserved separate columns. One irreconcilable question/external/reference formula task caused
  v5's sole illegal termination and 139,858 tokens. Stop v5 expansion at Prefix50; do not tune on
  the consumed tasks, infer SFT/RL promotion, or describe v5 as a reliable v4 replacement.
  V6 keeps the v5 tools, execution, feedback, context, and audit semantics and changes only the
  model-visible output rule: every answer is a SQL result table; a scalar is 1x1; one mapped field
  is one column; multiple mapped fields remain separate columns in their stated order; and
  concatenation is allowed only when QUESTION or EXTERNAL KNOWLEDGE explicitly requests one
  formatted, combined, or string value. This is a general representation rule, not a task-specific
  exception. Its frozen disjoint tasks 51-70 DeepSeek Flash Gate20 scored 12/20 versus v5 12/20,
  with one gain, one regression, 20/20 legal in both arms, 97 versus 100 actions, two process errors
  each, and 340,873 versus 331,641 tokens (+2.8%). Both arms passed 20/20 fresh replay/structure.
  The slice had no multi-field-name target or explicit-concatenation control, so it does not
  validate the target rule and provides no promotion evidence. Do not tune on the consumed first
  70 tasks. A subsequent public-text-selected Target Gate20 excluded the complete baseline300 and
  historical fixed-200. V6 scored **15/20** versus v5 **10/20**, with five gains, zero regressions,
  20/20 legal in both arms, 99 versus 111 actions, one versus six errors, and 339,600 versus
  371,036 tokens. Separate-field targets improved from 1/6 to 5/6; all ten valid single-field and
  ordinary controls were retained, and both arms passed 20/20 replay/structure. All numeric gates
  passed. However, the predeclared `field1+field2` explicit-combined controls were invalid: local
  hidden references also required separate columns, so the anti-overseparation claim remains
  untested. V6 is authorized only for a new representative paired gate; do not tune on either
  consumed Gate20 or use v6 for SFT/RL. V5 remains available as
  frozen `execute-sql-submit-sql-v5` with DeepSeek Flash lazy-catalog hash
  `b6465c6e222c12c2`; v6 uses `execute-sql-submit-sql-v6` with hash
  `176ce977b411636e`. Frozen v4/v3 and the historical
  `execute_sql_submit_sql_v2` interface are reproduction-only. Iterative SQL remains
  diagnostic-only, has no SFT exporter or RL environment, and must not be mixed with
  direct-SQL-search or atomic artifacts. See
  `docs/current/iterative_sql_tool_scheme_zh.md`.
  The subsequent user-authorized v6-only baseline300 run reused the audited tasks 51-70 and
  requested the missing 280 episodes. It scored **215/300 (71.67%)** with **298/300 legal**;
  all 300 records passed fresh replay and structure audit. Because tasks 1-70 were already consumed
  during design/diagnosis, the primary generalization read is untouched tasks 71-300:
  **168/230 (73.04%)**, **228/230 legal**; both clean 115-task halves scored 84/115. The full run
  used 1,698 actions, had 72 process errors, and consumed 7,000,561 tokens. No v5 arm was run on
  the clean 230, so this is an absolute v6 diagnostic rather than evidence that v6 beats v5 or
  another scheme. It does not change the no-SFT/RL boundary. See
  `docs/reports/evaluation/BIRD_ITERATIVE_SQL_V6_FLASH_FULL300_20260805_ZH.md`.
- No `add_to_memory`, `refine_memory`, reflection, invalidate, or model-visible sidecar state.
- Plans are control state, not factual evidence. Scalar reuse is grounded through direct step-id
  `value_ref`.
- Do not construct new version12-version26 SFT data until the frozen 200-task tool-usability gate
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
- Both mandatory process-RL gates passed on 2026-07-30 for the frozen 23-task cohort. The promoted
  `process-counterfactual-suite-v2` manifest binds 46 full-schema databases; known-correct replay
  passed 102/105 with coverage on 23/23 tasks, all four shortcut regressions were rejected, and the
  independent Codex audit labeled 63/63 grounding edges valid with no missing edge. The formal
  audit SHA-256 is `6ba130933ec4ce20e58117b506d57cd80181f9b1c7c044765572ca8b04ba8e49`;
  only configs marked `allowed_process_after_gates` may use it through strict
  `counterfactual-completeness`. The historical `phase1_process_current` remains evaluation-only.

### RTX 3090 long-context training memory profile (2026-08-04)

- The reusable OOM-safe Action-DPO implementation is
  `src/rl/action_dpo/train_fixed_prefix_action_dpo.py`. For long exact prefixes on a 24 GiB RTX
  3090, enable the safeguards progressively rather than changing data, truncating sequences, or
  changing the loss: selected tool-position logits, sequential positive/negative scoring,
  single-graph exact-DPO derivative recomputation, and CPU-resident AdamW moments during
  forward/backward. The moments move back to the parameter device only for the unchanged AdamW
  update. The corresponding flags are `--selected-tool-logits`, `--sequential-pair-scoring`,
  `--memory-safe-dpo-backward`, and `--cpu-offload-optimizer-state`.
- On `table_rl`'s current PyTorch 2.9 / Transformers 4.57 Qwen2 stack, `sdpa` falls back to math
  attention; its quadratic backward workspace requested 2.69 GiB at 5,078 tokens and still OOMed
  at optimizer step 16 with both expandable segments and `cudaMallocAsync`. Use
  `--attention-implementation flex_attention` with the checked-in conservative 32x32 forward and
  backward blocks. Set
  `TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda`; do not create a
  system `libcuda.so` symlink. The 5,827-token dataset maximum passed a post-Adam-state backward
  stress test at 13.17 GiB allocated / 13.89 GiB reserved, and the real run passed the former step
  16 OOM point with about 10 GiB free. The recovered run then completed 90/90 optimizer steps over
  183 pairs / 90 questions at learning rate 1e-6 with zero gradient clipping. Recovery evidence is frozen in
  `training/oom_recovery_audit.json` under the experiment root.
- These options trade compute or PCIe time for memory. Selected logits may also save compute when
  the loss mask is sparse; sequential pair scoring, graph recomputation, CPU optimizer offload,
  and small Flex blocks usually reduce raw examples/second. Use the recovered memory to raise a
  single job's micro-batch only after a measured smoke test proves higher end-to-end throughput.
  Do not infer that lower memory alone permits more same-size training processes: two observed
  approximately 13-15 GiB jobs do not fit safely on one 24 GiB card. Keep at least 2-3 GiB hard
  headroom and measure peak allocated/reserved memory before adding a smaller concurrent job.
- Apply this profile to training only. RTX 3090 evaluation/inference continues to use the separate
  vLLM continuous/dynamic-batching policy below. Never change question grouping, optimizer-step
  count, sequence length, beta, learning rate, or pair order merely to claim a memory optimization;
  such changes are new experiments rather than systems-equivalent OOM fixes.

### Evaluation

- Always name the denotation metric. All current/new BIRD evaluation, SFT replay, grounding gates,
  and RL/process audits use `bird-set`. `strict-multiset` remains only for immutable historical
  artifacts and explicitly named compatibility audits; never silently mix the two.
- The promoted single-GPU full-BIRD-dev greedy inference configuration is dynamic rather than
  statically sharded: `n=1534`, `n_samples=1`, `pass_k=1`, `temperature=0`, `top_p=1`,
  `max_steps=30`, `max_tokens=1024`, `record_logprobs=true`, `top_logprobs=20`, protocol
  `version36`, and denotation comparison `bird-set`. Run `rollout_passk.py` with `workers=24`,
  `sample_workers=1`, and `max_inflight_requests=24`; completed questions dynamically release a
  worker for the next question, while each question's tool/environment turns remain sequential.
  Serve with vLLM continuous batching using `max_num_seqs=24`,
  `max_num_batched_tokens=8192`, `max_model_len=8192`, BF16, and
  `gpu_memory_utilization=0.90`. This is the preferred RTX 3090 evaluation setup unless a named
  hardware test proves a safer or faster replacement; do not replace it with fixed question
  batches or static worker partitions. This evaluation policy is separate from training batch
  sizes and repair-generation microbatches.
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
  8 regressions were not significant, so it was not promoted.
  See `docs/reports/evaluation/BIRD_VERSION21_RESOLUTION_ABLATION_FIXED200_20260723.md`.
- Under the baseline-aligned `bird-set` metric, deterministic replay of the version20 trajectories
  with the corrected percentage operator scores 143/200. Version23 passed its 16-task pilot weakly
  but failed the frozen 50-task expansion gate at 35/50 versus the paired original 30/50: +5 net
  rather than the required +7. Do not launch its fixed-200 run.
- The completed version24 fixed-200 `bird-set` evaluation is **145/200 = 72.5%**, with 197/200
  legal termination, 29 process errors, 10 paired gains, and 8 paired regressions versus
  version20. It fails the 150/200 gate and is ineligible as an SFT source. See
  `docs/reports/evaluation/BIRD_VERSION24_RELATION_DERIVATION_FIXED200_20260724.md`.
- A frozen model-capability ceiling follow-up first reproduced version24 on the fixed 20-task
  DeepSeek v4 Flash gate at **16/20**, then ran DeepSeek v4 Pro on the same fixed 200 cohort.
  Pro also scored **145/200**, with 15 Pro-only gains and 15 Flash-only regressions (`p=1.0`).
  It improved legal termination from 197 to 199, reduced process errors from 29 to 22, actions
  from 1,490 to 1,454, and tokens by 3.1%, but did not raise denotation accuracy. All 145 correct
  Pro trajectories passed structural, fresh-replay, and provider-payload no-leak audits. Current
  evidence therefore does not support raw model capacity as the version24 accuracy bottleneck;
  keep the result diagnostic-only. See
  `docs/reports/evaluation/BIRD_ATOMIC_VERSION24_FLASH20_PRO200_CAPABILITY_CEILING_20260803_ZH.md`.
- The version40 concise-prompt/full-reasoning-history diagnostic stopped after the frozen first
  50 tasks: **37/50** versus paired version24 **42/50**, with 50/50 legal termination, zero gains,
  five output-shape regressions, and nine process errors. Do not expand it to the remaining 150.
  Its total tokens fell 12.4%, but completion reasoning tokens rose 81.3%; removing concrete
  output-slot and high-entropy argument guidance was not redundant.
- The version41 prompt-only output-correction Gate16 scored **10/16** versus paired version40
  **8/16**, with two gains, no regressions, all eight controls retained, and zero versus six
  process errors. It nevertheless recovered only 2/8 output-shape targets, below the required 4/8;
  do not expand it to the fixed first 50. See
  `docs/reports/evaluation/BIRD_ATOMIC_VERSION41_OUTPUT_CORRECTION_GATE16_20260729_ZH.md`.
- The version42 explicit-terminal-columns Gate16 scored **12/16** versus paired version41
  **10/16**, with two gains, no regressions, 4/8 targets correct, 8/8 controls retained, and
  16/16 legal termination. It failed only the predeclared total-process-error gate: 4 errors
  versus an allowed 2, of which two were exact-column rejections for uniquely resolvable bare
  names. Do not expand version42. See
  `docs/reports/evaluation/BIRD_ATOMIC_VERSION42_TERMINAL_COLUMNS_GATE16_20260729_ZH.md`.
- The version43 unique-bare terminal-column resolver passed Gate16 at **13/16**, then failed the
  original frozen first-50 at **39/50 = 78%** versus paired version24 **42/50**. It had one paired
  gain and four regressions, 50/50 legal termination, zero terminal projection errors, and nine
  process errors. All 12 unique-bare resolutions succeeded, but semantic population, omitted
  constraints, tie handling, and output-slot choices remained the dominant failures. Do not run
  the remaining 150. See
  `docs/reports/evaluation/BIRD_ATOMIC_VERSION43_UNIQUE_BARE_TERMINAL_COLUMNS_20260729_ZH.md`.
- A paired six-hard-task diagnostic found no gain from forced resident planning:
  optional and required were both 0/6, while required planning increased mean actions by 16.4%
  and tokens by 18.5%. Keep `required-resident` experimental; do not expand it or make it default
  without a new small-pilot signal. See
  `docs/reports/evaluation/BIRD_VERSION11_FAILURE_TRAJECTORY_AND_RESIDENT_PLAN_AUDIT.md`.
- The separate `action-block-v18` fixed-20 diagnostic scored 17/20 `bird-set`, with 19/20 legal
  termination, five process errors, and one provider-carrier failure. It was +1 versus both the
  paired action-block v4 and atomic version24 controls, but the paired differences were not
  significant. v16/v17 prompt ablations on the multi-row date task increased loops without
  recovering the target and were rejected. Keep v18 diagnostic-only and ineligible for SFT; see
  `docs/reports/evaluation/BIRD_ACTION_BLOCK_V18_OPTIMIZATION_PILOT20_20260726_ZH.md`.
- The historical action-block v21 fixed-40 diagnostic scored 31/40 `bird-set`, with 40/40 legal
  termination, seven process errors, one blocked descendant, 181 model turns, and 734,815 total
  tokens. Atomic version24 scored 33/40 on the same cohort but used 240 turns and 1,301,937 tokens.
  A v19 final-check prompt ablation also scored 31/40 and produced no paired gain, so it was
  rejected. v21 instead fixes an action-block-only silent semantic hazard: a cross-result
  `column_value` is converted to `value_ref` only for a harness-verified one-row/one-column source
  and is otherwise rejected. Keep v21 diagnostic-only and ineligible for SFT; see
  `docs/reports/evaluation/BIRD_ACTION_BLOCK_V21_FIXED40_SEMANTIC_AUDIT_20260726_ZH.md`.
- The completed action-block v21 fixed-200 gate scored **136/200 `bird-set`**, with 194/200 legal
  termination, 63 process errors, 25 blocked descendants, 1,092 model turns, and 4,982,816 total
  tokens. Paired atomic version24 scored 145/200 with 197/200 legal termination, 29 process errors,
  1,490 turns, and 8,420,861 tokens. V21 had 10 paired gains and 19 regressions (net -9,
  exact two-sided `p=0.1360`): it reduced turns 26.7% and tokens 40.8% but failed the 150/200 gate.
  Of its 58 legal wrong answers, 42 had no process error, so further interface normalization cannot
  recover the semantic gap. Do not construct action-block v21 SFT data; see
  `docs/reports/evaluation/BIRD_ACTION_BLOCK_V21_FIXED200_20260726_ZH.md`.
- The active action-block v32 fixed-40 diagnostic scored 31/40 `bird-set`, with 40/40 legal
  termination, 10 process errors, two blocked descendants, 182 model turns, and 792,488 total
  tokens. Its completed frozen 200-task gate scored **144/200**, with 197/200 legal termination,
  112 process errors, 51 blocked descendants, 1,115 model turns, and 5,031,851 total tokens.
  Paired atomic version24 scored 145/200 with the same legal rate; v32 had 13 gains and 14
  regressions (`p=1.0`). It reduced model turns 25.2% and tokens 40.2%, while increasing attempted
  primitive actions from 1,490 to 1,756. It fully removed nested/mixed terminal errors and
  single-object join-`on` errors, but failed the 150/200 gate. Do not construct action-block v32
  SFT data; see
  `docs/reports/evaluation/BIRD_ACTION_BLOCK_V32_FIXED200_20260726_ZH.md`.

## Baseline cohort lifecycle

- The active BIRD-train baseline for all new experiments is
  `data/eval_inputs/bird_train_baseline300_v1.jsonl` (300 tasks, all 69 train databases, SHA-256
  `87b37b307ea2710acdff7d03bdc474a6ba2cdb8ed51b005cff0fe8fa54c67c03`). Its frozen manifest is
  `data/eval_inputs/bird_train_baseline300_v1.manifest.json`.
- `data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl` is deprecated for new
  experiments. Keep it unchanged only for reproducing historical fixed-200 reports; never call an
  old-200 score a current baseline or pair it statistically with the new 300.
- The selection and lifecycle source of truth is `docs/current/baseline_datasets.md`. Every future
  cohort change must create a new versioned JSONL/manifest, preserve prior artifacts, mark the
  predecessor deprecated, update that document and this section in the same change, and record the
  exact cohort path/hash in result manifests.

## Current BIRD reference points (2026-07-24)

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
  Fact-only table derivation metadata is owned by `src/harness/relation_derivation/`.
- `src/tool_modules/checkpoint_relalg/`: forward `checkpoint-relalg-v1` protocol, compact prompts,
  native schemas, relation artifacts, checkpoint state, runtime, causal runner, and audits.
- Other `src/tool_modules/` packages: frozen or independent action-scheme implementations retained
  under their exact protocol identities.
- `src/sft/`: protocol, causal teacher rollout, replay/quality filters, SFT export and assembly.
- `src/eval/`: closed-loop tool evaluation and direct-SQL controls.
- `src/rl/`: tool environment, task loader, terminal reward, process credit, objective, backend.
- `docs/current/`: active contracts and workflows.
- `docs/decisions/`: rationale; not executable contracts.
- `docs/reports/`: immutable experiment and audit reports.
- `docs/archive/` and `archive/`: historical only; active code must not import from them.

## Active entry points

- Forward checkpoint-relalg diagnostic rollout:
  `src/tool_modules/checkpoint_relalg/runner.py --mode direct|atomic|hybrid`.
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
