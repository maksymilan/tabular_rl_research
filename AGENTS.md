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

- Shared prompt semantics: `src/sft/prompt_contract.py`; protocol/validation:
  `src/sft/protocol.py`; index: `docs/current/tool_protocol.md`.
- Current atomic local diagnostic default: `version39`; `version40` and `version41` are opt-in
  prompt/tool-name diagnostics. The production checkpoint-560 evaluation chain remains frozen on
  `version26`; none of version39-version41 has received an accuracy promotion and they must
  not be mixed into its result directories. Version26 retains version25's prompt-role and
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
  Gate50 process errors. Its tool-schema hash is identical to version40. Version41 is
  diagnostic-only pending a paired output-shape gate and is ineligible for SFT/RL. See
  `docs/current/atomic_version41_output_corrections_zh.md`.
  Future versions increment numerically.
- The canonical model action contains exactly one non-empty `<think>` block followed by one strict
  raw JSON object with exact `tool` and `arguments` keys. A provider-native reasoning adapter may
  carry the same authored reason in a separate API field, but its API-facing prompt and history
  must describe only that one carrier.
- The promoted/default context contract remains bounded recent legal history with
  `history_turns=4`: catalog,
  question, and optional external knowledge are followed by at most four successful
  assistant/observation pairs, and the latest observation carries rebuilt resident state plus
  optional `LAST TOOL ERROR`. Rejected assistant text is never added to the promoted/default
  history. Version40 is an explicit diagnostic exception for reasoning only: rejected reasons are
  retained with `status=rejected`, while the authored failed call is represented by the
  harness-owned attempted action/error. Version37 may render first-5 plus recent-5 for a paired
  diagnostic, but that policy is not promoted unless it beats recent-4 on the same frozen tasks.
- Current tools: `plan`, `describe_table`, `inspect_column`, `read_subtable`,
  `condition_filter`, `project`, `scalar_compute`, `join_tables`, `group_aggregate`,
  `extreme_value_select`, `set_op`, and `answer_from_context`.
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
  The scheme has evaluation and causal-rollout plumbing only, uses `tool-scheme-registry-v3`, and remains
  `diagnostic_only_pending_protocol_scale_gate`; it has no SFT exporter or RL environment.
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

### Evaluation

- Always name the denotation metric. All current/new BIRD evaluation, SFT replay, grounding gates,
  and RL/process audits use `bird-set`. `strict-multiset` remains only for immutable historical
  artifacts and explicitly named compatibility audits; never silently mix the two.
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
- The version40 concise-prompt/full-reasoning-history diagnostic stopped after the frozen first
  50 tasks: **37/50** versus paired version24 **42/50**, with 50/50 legal termination, zero gains,
  five output-shape regressions, and nine process errors. Do not expand it to the remaining 150.
  Its total tokens fell 12.4%, but completion reasoning tokens rose 81.3%; removing concrete
  output-slot and high-entropy argument guidance was not redundant.
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
