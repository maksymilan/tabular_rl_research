# Atomic Tool-Scheme Trajectory Protocol (version39 diagnostic)

Status: index-level contract for the currently implemented trajectory format. This file does not
replace code; it points to the source of truth and records what must not drift.

This document describes the original `atomic` scheme and indexes its version51-version54
primitive-compatible `native-tool-bundle` successors. The independently selectable non-atomic schemes are indexed in
`docs/current/tool_schemes.md`; the new two-action SQL protocol is specified by
`docs/current/iterative_sql_tool_scheme_zh.md` and
`src/tool_modules/iterative_sql/protocol.py`. A model is
given exactly one scheme; top-level action spaces are never merged in one prompt.

## Sources of Truth

- Shared atomic prompt semantics: `src/sft/prompt_contract.py`
- Public atomic tool structure: `src/sft/public_tool_contract.py`
- Model-visible protocol and validation: `src/sft/protocol.py`
  - `PROTOCOL_VERSION`
  - `STUDENT_SYSTEM_PROMPT` / `TEACHER_SYSTEM_PROMPT`
  - `TOOL_SPECS`
  - `MODEL_ARG_SCHEMA`
  - message rendering/parsing
- Isolated atomic version40 diagnostic: `src/sft/atomic_version40.py`
- Isolated atomic version41 prompt module: `src/sft/atomic_version41_prompt.py`
- Isolated atomic version42 terminal-column protocol: `src/sft/atomic_version42.py`
- Isolated atomic version43 unique-bare terminal resolver: `src/sft/atomic_version43.py`
- Isolated atomic version50 native function-call carrier: `src/sft/atomic_version50.py` and
  `src/tool_modules/native_tool_bundle/provider_tools.py`
- Forward version51 native bundle protocol: `src/tool_modules/native_tool_bundle/protocol.py`,
  `src/tool_modules/native_tool_bundle/provider_tools.py`, and
  `src/sft/generate_teacher_rollouts.py`
- Version52 compact native prompt and statistics:
  `src/tool_modules/native_tool_bundle/compact_protocol.py`,
  `src/tool_modules/native_tool_bundle/compact_prompt.py`, and
  `src/tool_modules/native_tool_bundle/bundle_credit.py`
- Version53 reviewed role-separated native prompt:
  `src/tool_modules/native_tool_bundle/reviewed_protocol.py` and
  `src/tool_modules/native_tool_bundle/reviewed_prompt.py`
- Harness provenance sidecars: `src/harness/provenance.py`
- Visible-cell binding semantics: `src/harness/observation_binding.py`
- Scalar `value_ref` grounding: `src/harness/scalar_grounding.py`
- Provenance redesign decision record: `docs/decisions/provenance_redesign.md`
- SFT export rendering: `src/sft/export_sft_dataset.py`
- Live rollout rendering: `src/eval/rollout.py` and `src/eval/rollout_passk.py`
- Canonical execution/ownership contract: `docs/current/execution_contract.md`
- Provider transport adapters: `src/sft/provider_adapter.py` (raw provider fields are retained;
  adapters are not parser repair)

If these disagree, fix the code and this index together. Do not infer the active protocol from
anything under `docs/archive/`.

## Model-Visible Format

Use `execution_contract.md` for the single current/new-episode contract. This file is the
protocol index; historical v2h and older documents are replay references only.

Before every assistant turn, the promoted/default SFT/evaluation/RL contract is rendered by
`src/sft/protocol.py::rolling_legal_history_messages` with recent `history_turns=4`:

```text
system: protocol/system prompt
user: DATASET OVERVIEW + QUESTION + optional EXTERNAL KNOWLEDGE
[up to four successful assistant action + harness observation pairs]
user: latest observation + CURRENT ENVIRONMENT STATE + optional LAST TOOL ERROR
assistant: <think>...</think>
           {"tool": "...", "arguments": {...}}
```

The environment state is harness-managed resident context. It groups the current task plan and
known table context by source table or derived handle. Legal history is bounded and is not the
source of factual provenance; harness state and references remain authoritative. Rejected
assistant text is never appended, and its structured error appears only in the current user
message. A rolling SFT record masks prior assistant turns and applies loss only to the final
supervised assistant action; the final input never includes that action's output.

The completed version37 diagnostic also tested `history_policy=head-tail`,
`history_head_turns=5`, and `history_turns=10`. It retained the first five and latest five legal
pairs with overlap removed. Resident state and rejected-action handling were unchanged; only legal
transcript selection differed. It scored 4/7 versus recent-4's 5/7 while using more actions,
errors, and tokens, so recent-4 remains the active policy.

Every new model turn is generated causally inside a real model↔harness episode. The model sees only
the legal prefix represented by current resident state and the latest environment feedback; it
chooses perception, relational, planning, and terminal actions itself. Gold SQL and future actions,
outputs, plans, or observations are never visible to the model or teacher. The retired
gold-SQL-compilation and complete-trajectory-enrichment pipelines live under `archive/` only.

### Prompt roles

The tool semantics, public argument schema, resident-state authority, grounding rules, and terminal
contract are shared. Prompt roles differ only in generation guidance:

- the student runtime prompt is used by SFT export, evaluation, and RL. It contains the shared
  typed-tool contract, canonical action envelope, and concise runtime invariants, but no call
  cookbook or case-specific repair advice;
- the external teacher receives that same student contract plus teacher-only elaborations,
  edge-condition guidance, canonical examples, and causal-generation quality controls;
- SFT export stores the canonical executed trajectory, then re-renders each legal prefix with the
  student runtime prompt. It never copies the teacher prompt into student records.

Teacher-rollout, SFT, evaluation, and RL manifests record the applicable teacher/student prompt
SHA-256 values and the public tool-schema SHA-256. A teacher addition cannot introduce a tool,
argument, state field, or execution behavior absent from the shared contract.

The model emits only one non-empty `<think>` block followed by one raw JSON object containing
exactly `tool` and `arguments`. The active carrier has no `tool_call` tags.

The model does not emit provenance, references, produces, quality status, repair metadata, or reward
fields.

## Current Tool Set

Public version mapping:

- `version1`: the former `v2i-state-only-join-feedback-r2` contract;
- `version2`: canonical calls and exact final-answer shape;
- `version3`: precise DeepSeek transport feedback, stable projected join-column names, and a
  canonical multi-table join call;
- `version4`: provider carrier failures are explicitly counted against the
  protocol-error budget rather than argument-validation errors;
- `version5`: joins use `base + joins[]`, flat `relation.column` output names,
  and semantic roles only for repeated relations;
- `version6`: keeps the version5 action shape and groups wide dotted columns as
  `column_namespaces` in model-visible context without changing canonical harness snapshots;
- `version7`: makes filters, projection expressions, grouping, and ordering consume those logical
  columns consistently, with safe unique-bare resolution;
- `version8`: gives a provider exactly one API-facing response envelope and
  renders bounded assistant history in that same carrier, while retaining the canonical stored
  `<think>` + `<tool_call>` action;
- `version9`: explicitly enables DeepSeek thinking mode, fixes reasoning effort,
  and audits the request controls plus returned provider identity metadata;
- `version10`: constrains provider-visible actions with JSON Output and wraps the unchanged returned
  JSON in the internal canonical `<tool_call>` envelope;
- `version11`: removes the remaining generic visible-`<tool_call>` wording from
  DeepSeek-facing generation and retry instructions, leaving one raw-JSON visible contract;
- `version12`: adds `project(distinct=...)` and a grounded `scalar_compute` atom;
- `version13`: terminal actions cite one exact result table, including a 1x1 table for scalar
  answers, and cannot carry model-authored answer values;
- `version14`: model-visible resident plan evidence keeps only its grounded
  `step_id + tool` identity, while canonical snapshots retain the full evidence output for
  replay/audit. This prevents a plan item from duplicating large schemas or row payloads on every
  later turn;
- `version15`: each `group_aggregate.aggregations[]` item may carry an
  optional `where` predicate, so several conditional metrics can be computed over one fixed input
  population and grain;
- `version16`: `pivot` reshapes a grouped key/value table from one row per
  category into one row with ordered category columns;
- `version17`: the `project` contract explicitly states that it preserves
  row orientation and directs category-row-to-column reshaping to `pivot`;
- `version18`: the separate public `pivot` action is folded into
  `group_aggregate(output_layout="columns", category_values=[...], output_columns=[...])`.
  Historical pivot calls remain replay-compatible only;
- `version19`: `scalar_compute` can cite a named column from a prior
  one-row multi-metric result using `{"value_ref":"step_k","column":"metric"}`. The harness
  resolves the exact non-NULL cell and records the output column on each value edge;
- `version20`: provider carrier examples contain only the final action
  shape and no copy-prone field labels. Short schema invariants live beside the affected tools:
  join `right` is bare, scalar references cite producing steps rather than observation steps,
  aggregate `where` is per-aggregation, and project/read arguments remain closed. Length-truncated
  provider completions receive a bounded, audited same-turn client retry and are not themselves
  semantic agent actions;
- `version21`: audited deterministic resolver ablation; not promoted because its fixed-200 run
  increased process errors and trajectory length without a stable accuracy gain;
- `version22`: keeps the version20 public tools and argument schemas
  unchanged and adds compact, harness-derived structural feedback after high-risk legal actions:
  multi-column projection collapse, aggregate result grain/layout/count semantics, and unmatched
  rows for a single-edge left join. Only the latest such feedback is resident, so it cannot
  accumulate across a trajectory. Its frozen 16-task pilot was non-destructive but recovered only
  2/7 unambiguous targets, below the 3/7 expansion gate;
- `version23`: advice-heavy feedback experiment; keeps every version22 tool, argument, execution, and state
  boundary unchanged while making those three existing feedback payloads operationally explicit:
  named component fields remain separate unless formatting is requested, row counts preserve join
  multiplicity and global totals are not per-entity totals, and unmatched left-join rows lack the
  right-side attribute. It remains an advice-heavy experiment and was not authorized for a
  fixed-200 run;
- `version24`: removes the advice-heavy feedback and global latest-feedback
  sidecar. Every table-producing operator now emits one validated, fact-only
  `relation-derivation-v1` record bound to its output handle. Derivation construction lives in the
  harness, is complete over the active table action space, and is separate from SQL execution,
  provenance, state storage, protocol rendering, and model policy. Its completed frozen-200
  DeepSeek v4 Flash evaluation is 145/200 under `bird-set`, versus the paired version20 143/200,
  but it fails the 150/200 gate and slightly regresses legal termination and process errors;
- `version25`: preserves the version24 tools, execution, state, and
  relation-derivation semantics while separating prompt roles. The student runtime prompt is the
  concise shared tool contract used identically by SFT export, evaluation, and RL. External
  teachers receive a strict superset containing generation guidance and examples. Public model
  argument validation is now isolated in `MODEL_ARG_SCHEMA`, separate from replay-only legacy
  schemas. This is a contract/engineering change and has no accuracy promotion yet;
- `version26`: preserves all version25 tools, arguments, execution, state, grounding, and
  relation-derivation semantics, but replaces the model-visible tagged action carrier with one
  non-empty `<think>` block followed directly by the raw `{"tool":...,"arguments":...}` object.
  The strict runtime accepts only this carrier. Retired tagged actions are parsed only by an
  explicitly named offline migration function, then re-rendered without changing their structured
  action or reasoning;
- `version27`: adds structured carrier/argument error codes and rejects only two consecutive
  parsed actions whose canonical `tool + arguments` are exactly equal. It ignores `<think>` and
  JSON key order, and never searches farther back;
- `version28`: retains the original rejection details when the same rejected action is retried and
  makes the existing `read_subtable` no-pagination boundary explicit: without an offset/cursor,
  an identical call reads the same prefix. Its frozen 48-task checkpoint-560 diagnostic improved
  legal termination and mean steps but scored 12/48 versus fresh version26 at 13/48, so it is not
  accuracy-promoted and must not be expanded to full greedy or used for SFT;
- `version29`: adds typed row expressions to `project`, a non-negative `read_subtable.offset`, and
  explicit rank offset/partition arguments to `extreme_value_select`. It remains a local
  diagnostic and has no accuracy or SFT promotion;
- `version30`: restores the version28 public action surface and adds teacher-only causal evidence
  discipline before semantic commitments and termination. The student tool schema is unchanged;
  it remains diagnostic-only;
- `version31`: keeps version30's public tools and valid-call semantics, but validates current table
  handles, table/column ownership, predicate operands, join-edge columns, and terminal evidence
  handles before executing model-authored SQL. Failures are structured
  `argument_validation_error` events with exact argument paths and available columns;
- `version32`: keeps version31 validation and all valid-call semantics unchanged. An
  `unknown_column` message now adds one concise instruction to choose the correct table or column from
  already observed schemas; it does not add global-absence wording or reveal unseen schema;
- `version33`: keeps version32 feedback and valid-call semantics unchanged, but prepares
  model-authored `project` expressions against the current relation without registering a derived
  handle. Missing columns and malformed expressions now become structured, state-preserving
  validation errors instead of raw SQLite failures;
- `version34`: keeps version33's public calls, feedback, and relational semantics unchanged. The
  executor lazily materializes only small derived relations when they are reused as join inputs,
  preventing nested composed SQL from being recomputed inside a later join. The cache is
  connection-local, bounded to 50,000 rows, and does not change model-visible handles or values;
- `version35`: keeps version34's public calls, feedback, validation, and execution semantics.
  Exactly repeated adjacent calls are still rejected and charged to the shared action budget, but
  `no_progress_error` no longer triggers the generic three-errors-per-type early abort. The model
  can recover until `max_steps`; other recoverable error limits are unchanged. Its first full
  launch revealed that the pass@k runner had retained the old generic limit, so that partial
  artifact is frozen rather than mixed with corrected output;
- `version36`: applies version35's no-progress action-budget policy through one shared helper in
  both atomic runners (`rollout.py` and `rollout_passk.py`). Public tools, feedback, validation,
  and valid-call execution remain unchanged;
- `version37`: retains version36's error policy. `read_subtable` remains observation-only but adds
  typed `conditions`, exact-column `order_by`, and a non-negative `offset` that requires ordering.
  `project` adds only typed per-row `date_diff_days(start,end)` and `extract_year(date)`
  expressions. No other arithmetic/ranking extension from the rejected version29 experiment is
  restored. A frozen seven-task recovery diagnostic scored 5/7 with recent-4 versus 4/7 with
  first-5 plus recent-5, so the larger history renderer is not promoted;
- `version38`: keeps version37's student runtime prompt, public tools, argument schemas, execution,
  state, feedback, carrier, and recent-4 history policy unchanged. It adds only external-teacher
  semantic decision discipline: explicit question/external-knowledge mappings are binding;
  answer population, row grain, aggregation unit, and output slots are fixed before semantic
  commitments; unsupported singleton/time/mean/current restrictions are forbidden; anomalous
  observations trigger grounded inspection; and the cited terminal table is checked slot by slot.
  Its frozen 16-target/16-control DeepSeek v4 Flash gate scored 18/32 versus paired version37 at
  16/32 after infrastructure-only recovery, with six gains and four regressions (`p=0.7539`).
  It missed its recovery, control-retention, and net-gain thresholds. Keep it diagnostic-only,
  do not expand it, and do not use its trajectories for SFT;
- `version39`: keeps version38's public tools, argument schemas, execution, canonical resident
  state, replay, feedback, carrier, and recent-4 history policy unchanged. Only model-visible
  resident-state rendering changes: completely equivalent `read_subtable` observations are
  rendered once with their equivalent source steps, while groups of at least two unread and
  unreferenced zero-row `condition_filter` handles sharing an input and output shape are folded
  into a fact-only summary retaining their handles, creator steps, predicates, input, and columns.
  Referenced, observed, or singleton empty handles remain fully expanded. It is diagnostic-only
  pending context-efficiency and behavior-preservation gates;
- `version40`: an opt-in external-teacher diagnostic selected with
  `--atomic-protocol-version version40 --diagnostic-only`. It keeps version39 relational
  execution and the exact `join_tables(base, joins[], base_role?)` contract, removes `plan` from
  the public surface, and renames the read-only row observer to `inspect_rows`. Its single layered
  prompt states each policy once and has no appended teacher-policy duplicate. All successful and
  rejected reasoning is retained as explicitly labeled, non-factual continuity context. The
  latest rejected attempted action and complete error remain in `LAST TOOL ERROR`; exact
  successful actions and unabridged results are fixed to the four most recent pairs. Its frozen
  first-50 gate scored 37/50 versus paired version24 42/50, with zero gains and five output-shape
  regressions. Do not expand it or use it for SFT/RL;
- `version41`: keeps every version40 public tool, argument, execution, state, feedback, history,
  carrier, and join rule unchanged. It changes only the prompt through
  `src/sft/atomic_version41_prompt.py`: one consolidated output contract restores exact output-slot
  discriminators, and concise legal examples correct the five tool families that produced
  Gate50 process errors. Its frozen output-shape Gate16 scored 10/16 versus paired version40 8/16,
  with two gains, no regressions, 16/16 legal termination, and zero versus six process errors, but
  only 2/8 targets recovered. It failed the predeclared 4/8 recovery gate. Do not expand it or use
  it for SFT/RL;
- `version42`: keeps every version41 nonterminal tool and behavior unchanged, but requires
  `answer_from_context.evidence={"table":handle,"columns":[...]}`. The harness validates exact
  existing columns and deterministically projects only those columns in order before scoring. It
  never consults gold or changes rows/values. Its frozen output-shape Gate16 scored 12/16 versus
  paired version41 10/16, with two gains, no regressions, all eight controls retained, and 16/16
  legal termination. It met the 4/8 target-recovery threshold but produced four process errors,
  above the allowed two, including two avoidable exact-column errors on uniquely resolvable bare
  names. Do not expand it or use it for SFT/RL;
- `version43`: keeps version42's explicit ordered terminal columns and changes only deterministic
  column-name resolution. Full logical names match first; an unqualified name may match a dotted
  logical-column suffix only when exactly one candidate exists in the cited table. Ambiguous,
  missing, or incorrectly qualified names remain errors. It never consults question semantics,
  external knowledge, reason text, or gold, and cannot change rows, values, or declared order.
  Its frozen Gate16 scored 13/16 and passed all local thresholds, but the original frozen first-50
  scored 39/50 versus paired version24 42/50: one gain, four regressions, 50/50 legal termination,
  zero terminal projection errors, and nine process errors. The resolver succeeded on all 12
  unique-bare columns exercised, but the protocol failed the overall accuracy, paired-regression,
  and process-error gates. Do not run the remaining 150 or use version43 for SFT/RL;
- `version44`: returns to version39's full prompt, recent-4 legal history, exact-table terminal
  evidence, execution, state, and grounding. It removes public `plan`, renames the public row
  observer to `inspect_rows`, adds deterministic read-only
  `search_values(table, query, column?, limit?, offset?)`, and exposes BIRD semantic name and
  column description only through `inspect_column`. Its frozen DeepSeek v4 Flash paired Gate50
  scored 38/50 versus version39 at 39/50, with four gains, five regressions, 50/50 legal
  termination, and 3 versus 2 process errors. Search was used on 19 tasks, but both protocols
  scored 16/19 on that subset. Million-distinct-value fuzzy scans also caused a severe latency
  regression. Do not run the remaining 150 or use version44 for SFT/RL before bounded retrieval
  is implemented and the same gate is rerun;
- `version45`: keeps version44's public call signatures and all non-search behavior, but selects
  an isolated `bounded-v1` executor path. SQLite first recalls exact/case-insensitive-exact values;
  an exact hit suppresses broader fuzzy alternatives. Otherwise stable token/trigram anchors
  recall at most 4,096 candidates per column before the existing lexical matcher runs.
  Observations expose candidate count, scope, and truncation. Replaying all 22 searches from the
  version44 Gate50 preserved 22/22 top candidates and reduced total search time to 1.033 seconds;
  the two approximately 1.98M-distinct-title searches fell from about 380 seconds each to 0.374
  and 0.132 seconds. After the provider recovered, its fresh full Gate50 scored 37/50 versus the
  frozen version39 baseline at 39/50, with two gains, four regressions, 50/50 legal termination,
  and three process errors. All four regressions were non-search semantic/output failures. Wall
  time fell from version44's 405.83 seconds to 317.21 seconds, but accuracy did not improve. Keep
  version45 diagnostic-only; do not run the remaining 150 or use it for SFT/RL;
- `version46`: returns to the version39 tools, prompt, execution, canonical state, grounding,
  recent-4, and exact-table terminal semantics. It changes only model-visible resident rendering:
  every canonical relation derivation becomes one Harness-authored fact-only handle card. On the
  15-task infrastructure-complete subset of a frozen read-heavy Gate16, it had exactly the same
  outcomes as version39 at 8/15, but used 10.3% more actions, 38.6% more tokens, and 11 versus six
  process errors. It has no accuracy promotion;
- `version47`: adds to version46 a full latest successful observation while older successful
  observations remain compact; resident row payloads become deterministic read cards requiring a
  fresh row observation before exact reuse. It scored 9/15 versus version46 at 8/15, with two
  gains and one control regression (`p=1.0`), but legal termination fell to 13/15, row reads rose
  from baseline 27 to 71, and 24 reads exactly repeated earlier arguments. It is rejected;
- `version48`: changes only version47's external-teacher prompt by requiring one brief
  interpretation of the latest Harness output before the next action. It remained 9/15 with one
  gain and one regression versus version47, while actions reached 1.49x baseline, tokens 1.62x,
  row reads 101, and exact re-reads 35. It is rejected;
- `version49`: returns to version46's compact successful-observation policy and adds automatic
  Harness-owned relation focus. Exact resident rows remain only for table handles referenced by
  recent-4 structured calls/observations/errors and their recursive derivation dependencies; rows
  on all other branches become read cards. The model declares no branch id or lifecycle. Its raw
  Gate16 scored 9/16 with 15/16 legal termination. On the 15-task infrastructure-complete subset,
  it exactly matched version39 at 8/15 and retained 8/8 controls but recovered 0/7 targets. It used
  138 actions, 26 reads, zero exact re-reads, and 1,169,008 tokens versus version39's 146, 27, zero,
  and 1,201,127. It recovered both observed churn cases, but had no accuracy gain. Offline
  fixed-200 rendering reduced next-action literal loss from 12 turns/20 values under version47 to
  one dormant-branch turn/four values. That dormant-branch task was correct in the raw version49
  run but still used 10 reads and two exact re-reads. A frozen Gate8 K=3 follow-up then scored
  20/24 for both version39 and version49 with 15/15 controls retained, but version49 increased mean
  steps 23.0%, reads 71.9%, and tokens 28.7%, and produced three max-step failures versus zero.
  It failed five of nine preregistered stability checks and is rejected even as a candidate
  engineering renderer. Do not expand version46-version49 to Gate50 or use their trajectories for
  SFT/RL. See
  `docs/reports/evaluation/BIRD_ATOMIC_CONTEXT_HANDLE_CARD_ABLATION_GATE16_20260731_ZH.md` and
  `docs/reports/evaluation/BIRD_ATOMIC_VERSION49_CONTEXT_STABILITY_K3_GATE8_20260731_ZH.md`;
- `version50`: returns to version39's public tools, prompt semantics, execution, canonical state,
  recent-4 context, and exact-table terminal. It changes only the DeepSeek-facing action carrier
  from native reasoning plus visible JSON Output to native reasoning plus official function calls.
  The tools schema is deterministically rendered from the same `MODEL_ARG_SCHEMA`; native responses
  are converted without argument repair to the canonical internal action before the shared parser.
  Thinking-mode history returns original reasoning and call ids. The diagnostic uses
  `tool_choice=auto` because the provider rejects `required`, then locally enforces exactly one
  call. Its completed user-authorized fixed-200 scored 133/200 correct and 183/200 legal versus
  historical version24 at 145/200 and 197/200; process errors rose from 29 to 166 and tokens nearly
  doubled. Replay, structural, no-leak, and native-lowering audits passed, but the behavioral gate
  failed. Version50 is rejected and ineligible for SFT/RL;
- `version51`: keeps the version39 primitive function schema and harness semantics, but moves to
  the distinct `native-tool-bundle` scheme. A provider assistant turn contains 1..8 calls chosen
  from one visible pre-state; every call is prevalidated before any executes and receives a
  matching tool-result message. Non-empty assistant content is audit-only. Terminal calls must be
  alone. Primitive steps retain their shared model-turn identity rather than pretending the model
  observed intermediate results. Its frozen Gate32 scored 22/32 versus version50 16/32, with six
  gains and no regressions. Its completed fixed-200 scored 147/200 with 200/200 legal versus
  version50 at 133/200 and 183/200; 20 paired gains and six regressions were significant, carrier
  protocol errors fell from 124 to two, and all replay/structure/provider-history/no-leak audits
  passed. Historical version24 remained statistically tied at 145/200 and used about half the
  tokens. Version51 is the frozen diagnostic base for version52+, but is not SFT/RL eligible;
- `version52`: keeps every version51 tool, argument, execution, state, carrier, bundle-pre-state,
  recent-four history, and terminal behavior. Native JSON schemas become the sole argument-shape
  authority, eliminating the duplicate textual tool catalog, long teacher tool descriptions, and
  call cookbook. Soft scheduling defaults to one call, normally uses at most three, favors
  independent perception, restricts parallel filters to grounded hypotheses, and normally isolates
  relational operators; the harness still permits at most eight and enforces terminal as exactly
  one call. Structured error results remain in provider history and the later correction is the
  learnable target boundary. Fact-only call annotations are statistics, not reward. Static
  fixed-200 counterfactual request rendering falls 45.4% in message characters; no live gate has
  yet promoted accuracy or measured provider tokens. Version52 remains diagnostic-only;
- `version53`: keeps version52 tools and execution unchanged, separates the shared runtime kernel
  from teacher-only generation, adds data-as-data, risk-triggered join-cardinality,
  source-versus-derived representation, correctness-first scheduling, and distinct-handle
  independent-filter rules, and removes harness-only prompt prose. Static teacher prompt plus
  native schema falls from 16,769 to 15,168 characters. This is not a live behavior result and
  version53 remains diagnostic-only;
- `version54`: keeps the version53 student prompt, carrier, eleven non-plan functions, execution,
  state, history, feedback, and terminal semantics; removes only public `plan` and its stale
  teacher-only evidence sentence. Its v54-only teacher1500 Prefix200 absolute gate is
  preregistered, not yet completed; passing every gate conditionally opens the remaining 1,300
  frozen tasks without a v53 comparison;
- future changes increment only the integer (`version55`, `version56`, ...).

Historical-control note: a separate branch at `deaa0e4` rebuilt original version24 and added only
bounded `search_values`, retaining plan, `read_subtable`, raw schema, and exact-table terminal
semantics. Fresh version24 reproduced 41/50 versus the original 42/50; the search-only arm scored
39/50, with two gains and four regressions (`p=0.6875`) and 14.3% more tokens. This is a rejected
historical ablation and not part of the current protocol progression. See
`docs/reports/evaluation/BIRD_VERSION24_BOUNDED_SEARCH_SINGLE_VARIABLE_GATE50_20260730_ZH.md`.

The version39 atomic baseline and version51-version53 native-bundle line share the complete tool
set in `src/sft/protocol.py::TOOL_SPECS`; version54 uses the same set minus `plan`:

- `condition_filter`
- `plan`
- `project`
- `scalar_compute`
- `join_tables`
- `group_aggregate`
- `extreme_value_select`
- `set_op`
- `describe_table`
- `inspect_column`
- `read_subtable`
- `answer_from_context`

The shared version40/version41 nonterminal public set is defined by
`src/sft/atomic_version40.py::TOOL_SPECS`. It removes `plan`, replaces
`read_subtable` with `inspect_rows`, and leaves every other signature—including `join_tables`—
unchanged. Version41 reuses the exact same tool-schema hash. The executor keeps `read_subtable`
only as a private/replay-compatible implementation name. Version42 reuses those nonterminal specs
but has a distinct tool-schema hash because its terminal evidence object requires `columns`.

The model-visible relation derivation schema is indexed separately in
`docs/current/relation_derivation.md`.

For the promoted/default version39 contract, only the teacher-generation prompt includes canonical
JSON examples for complex operation families; its student runtime prompt has no worked cases. The
isolated external-teacher-only version40/version41 diagnostics instead use one fixed prompt and
remain ineligible for student SFT/RL.

For every terminal answer, the cited evidence table is scored as the answer. Its rows, columns, and
column order must match the requested output exactly. `read_subtable(columns=..., conditions=...)`
only limits observation and does not change table shape or create a new handle. If helper columns
remain, the model must call `project`
before `answer_from_context(evidence={"table": ...})`. Scalar aggregates and `scalar_compute`
produce 1x1 evidence tables and use the same terminal shape. The terminal call contains no
model-authored answer data; think/reason text cannot repair answer data.

`read_subtable.limit` is an integer in `1..20`. An invalid value is an explicit
`argument_validation_error`; the harness never clamps it. `conditions` uses the same typed
predicate tree as `condition_filter`, including `on_date` to match the calendar date of a stored
date/timestamp. `order_by` names exact visible columns and may append `ASC`/`DESC`; a positive
`offset` is accepted only with `order_by`. Repeating the exact same arguments still requests the
same rows and is rejected only when immediately adjacent. In bounded rolling mode, prior successful
actions retain compact result summaries while full factual payloads remain in resident state.

`project` typed date expressions have exactly `{op, operands, as}`. `date_diff_days` has two
ordered operands (start, end); `extract_year` has one. Each operand is exactly `{column}` or
`{value}`. The harness validates referenced columns before SQLite, deterministically lowers the
operation, and records exact input-column lineage on the derived table.

Each `group_aggregate.aggregations[]` item has `op`, `column`, and `as`, plus an optional `where`
using the same predicate tree as `condition_filter`. A call may therefore produce a one-row,
multi-column result such as female and male counts without creating two filtered handles. Literal,
`value_ref`, `in_table`, schema-grounding, and domain-grounding dependencies inside `where` remain
harness-owned references. Conditional `count_distinct` requires a named column rather than `*`.

For category comparisons, the same `group_aggregate` call may set `output_layout="columns"`.
This mode requires exactly one `group_by` column and one aggregation. Ordered `category_values`
become one output slot each; optional `output_columns` renames those slots one-for-one without
changing their order. Default `output_layout="rows"` retains ordinary SQL GROUP BY row output.
`project` cannot change row orientation.

When such an aggregate returns exactly one row with several named metrics, `scalar_compute` can
reuse its cells without splitting the aggregation:

```json
{
  "operation": "percent",
  "operands": [
    {"value_ref": "step_7", "column": "usa_nominees"},
    {"value_ref": "step_7", "column": "total_nominees"}
  ],
  "result_name": "percentage"
}
```

The step must identify a table with exactly one row, each column name must match exactly one output
column case-insensitively, and each cell must be non-NULL. The harness reads the cell from its
resident table rather than trusting a model-authored value. Omitting `column` retains the original
1x1-only behavior.

No `add_to_memory` tool exists in v2c-plan. No reflection or invalidate tool exists.

### Join Identifier Rule (version20; unchanged from version8)

`join_tables(base, joins, base_role?)` joins a connected component in one model action. Every
`joins[]` item attaches exactly one new table. Its `on.left` values are exact already-introduced
logical columns (`relation.column`); its `on.right` values are bare columns of the new table.
The default namespace is the visible source table or handle name. `base_role` and item `role` are
used only to disambiguate repeated relations such as employee/manager.

Join failures report both the referenced table schemas and this model-facing identifier rule. The
harness does not strip arbitrary SQL aliases or invent roles to rescue an invalid model action.
Output columns stay flat across the full component (`orders.id`, not `join_003.orders.id`), and
projection remains a separate tool action.

For context efficiency, a derived handle may render these flat names as
`column_namespaces={"orders":["id",...], ...}`. This is lossless model-visible compression:
`orders` remains the logical namespace for subsequent column arguments even when the table argument
is a handle such as `join_003`. The canonical resident state and replay record retain full names.
Downstream tools quote exact dotted names even inside scalar expressions. A bare suffix is accepted
only when it maps to one available logical column; joins themselves still require the exact
namespace-qualified left key.

## Plan Status

`plan(ops)` is model-visible task-control state managed by the harness. The model can create,
add, update, or delete subgoals. Each plan item contains only a goal, status, and an optional prior
step-id evidence reference:

- `status`: `pending|in_progress|done|blocked` says whether the subtask is complete;
- `evidence`: a prior step id expanded by the harness to actual tool output.

The plan is not factual evidence:

- it cannot be used through `value_ref`;
- it cannot support the final answer;
- it is excluded from data/value provenance slices;
- it may receive separate planning/process rewards.

Plan wording and updates are produced online from prefix-visible context. Deterministic checks ensure
that plan items cannot become factual evidence or contain unsupported answer values.

## Memory Status

Memory is not part of the current model-visible protocol:

- no `add_to_memory`;
- no `memory_id`;
- no `mem_` references;
- no `supporting_memory_ids`.

Computed scalars are reused through direct producing-step `value_ref`. Predicates require a
scalar-shaped producing step. `scalar_compute` additionally accepts `value_ref+column` for one
named cell of a one-row multi-metric table. The harness performs scalar/cell extraction and
validation in both cases.

## Harness-Owned Fields

Trajectory JSON may contain harness-owned fields that are not model actions:

- `step_id`
- `tool_status`
- `tool_output`
- `references`
- `produces`
- `schema_version`
- `label_status`

These fields are used for validation, replay, SFT export, provenance slicing, and analysis.
They should not be copied into model tool arguments.

When a later predicate literal exactly copies a scalar from an earlier model-visible row, the
harness may add a `row_observation` grounding reference. A declared FK or column-equivalence match
is preferred. Because BIRD omits some FK declarations, a unique exact visible-cell copy is also
valid even when the two column names differ. The reference records the consuming argument path and
the source row/column locator. Counterfactual replay binds only from a singleton observation;
repeated cells and multi-row selections do not receive a direct replay binding.

This remains harness-side semantics:

- the model does not emit a source pointer or any new tool argument;
- literals stated by the question or external knowledge remain task constants;
- counterfactual replay may rebind only those unambiguous harness-proven visible-cell copies;
- a row-copy edge proves value flow, not an unexecuted higher-order operation such as `argmax`.

## Recovery Data Rule

Recoverable model errors stay inside the same episode. The harness preserves the resident factual
state, spends one action from the shared `max_steps` budget, and supplies the next turn with the
bounded rolling renderer plus structured `LAST TOOL ERROR` in the current user message:

```json
{"step_id":"step_4","status":"error","error":{"type":"argument_validation_error","code":"argument_validation_error","message":"...","details":{"expected_arguments":{"required":["table"],"optional":["columns","conditions","limit","offset","order_by"]}}},"attempted_action":{"tool":"read_subtable","arguments":{"table":"T","limit":21}}}
```

The bounded recoverable classes are `protocol_error`, `argument_validation_error`, and an
`execution_error` whose environment-state snapshot is unchanged. `no_progress_error` is the
recoverable rejection for one exactly repeated adjacent structured call. It is recorded and spends
one shared action, but does not trigger an early per-type abort; recovery remains possible until
`max_steps`. The other recoverable classes retain their per-type error limits. A mutated-state
execution failure is terminal as `nonrecoverable_execution_error`.
API transport retries are client-side requests, not semantic actions or recovery events.

Every rejected model action is retained in the full audit record as an `error_event`, with action
index and before/after state hashes. It is never included in `trajectory.steps` or exported as an
SFT target. The first later legal step carries `feedback_recovery: true` and its prior error type.
A verifier-correct episode is `clean_success` only when it had no error events; otherwise it is
`recovered_success`.

Strict recovery parsing accepts exactly one non-empty `<think>` block followed by one complete raw
JSON object with exact `tool`/`arguments` keys and nothing else. It does not accept `tool_call`
tags, balance JSON, normalize arguments, or otherwise repair model output. Whole-episode restarts,
when deliberately enabled for pass@k, are distinct attempts and must be reported as such.

Recovery behavior is represented by ordinary first-person `<think>` text and existing tools:

```text
The previous result is empty, so it does not support answering yet. I should inspect the relevant
column values and try a grounded filter instead.
```

Do not add:

- an `invalidate` field;
- a `reflection` tool;
- model-visible sidecar state;
- a new memory object.

Any recovery analysis outside the dialogue must be derived from the execution log, tool outputs, and
the final answer dependency slice.
