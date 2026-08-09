# Selectable tool schemes

## Contract

The repository exposes seven complete and independently selectable model action schemes:

| Scheme id | Model turn | Top-level actions | Student carrier |
|---|---|---|---|
| `checkpoint-relalg` | exactly one native function call under mandatory `mode=direct|atomic|hybrid` | mode-specific perception/relational actions plus shared `commit_checkpoint`, `restore_checkpoint`, and exact-artifact `answer` | official DeepSeek `reasoning_content` + exactly one `tool_calls` item, followed by its matching tool result |
| `atomic` | exactly one primitive tool | the original planning, perception, relational, and terminal tools | `<think>` followed directly by one raw JSON action |
| `native-tool-bundle` | one provider assistant turn containing 1..8 direct primitive calls from one shared pre-state | the version39 perception, relational, and terminal functions; frozen version54 excludes `plan` | structured DeepSeek `reasoning_content` + `tool_calls`, followed by one tool result per call id |
| `action-block` | one ordered work block of 1..8 primitive calls, or one terminal action | `action_block` for work; `answer_from_context` for termination | `<think>` followed directly by one raw JSON action |
| `relational-program` | one interactive observation, one declarative relation program of 1..8 nodes, or one terminal action | `observe`; `relational_program`; `answer_from_context` | `<think>` followed directly by one raw JSON action |
| `direct-sql-search` | one database-value retrieval or one read-only SQL inspection/final execution | `search_values`; `execute_sql` | `<think>` followed directly by one raw JSON action |
| `iterative-sql` | one read-only SQL exploration or one recoverable final SQL proposal | `execute_sql`; `submit_sql` | `<think>` followed directly by one raw JSON action |

The ids are defined by `tool-scheme-registry-v12` in `src/tool_modules/registry.py`. A model sees exactly
one scheme. Do not combine top-level schemas from different schemes in one prompt and do not infer
a scheme from trajectory shape. Registry v2 unifies both prior student schemes on the active
`think-json-v1` carrier;
registry v3 adds `relational-program` without changing either prior scheme; registry v4 adds the
diagnostic-only `direct-sql-search` scheme without changing the first three; registry v5 adds the
diagnostic-only `native-tool-bundle` scheme without changing the prior four; registry v6 registers
the diagnostic-only `iterative-sql-v3` scheme without changing the prior five; registry v7 advances
only that scheme to the failure-derived `iterative-sql-v4`; registry v8 advances only that scheme
to the externally reviewed `iterative-sql-v5`; registry v9 advances only that scheme to
`iterative-sql-v6`, adding an explicit relational output-shape rule; registry v10 advances only
the native bundle to the version53 reviewed prompt profile; registry v11 advances only that bundle
to version54 by removing the public `plan` function; registry v12 adds the independent forward
`checkpoint-relalg-v1` scheme without changing any frozen predecessor. Registry v1's
tagged atomic carrier is retired.

The first five listed schemes share Harness-owned relational semantics while retaining distinct
state, action, and trajectory contracts.
`direct-sql-search` and `iterative-sql` instead share the immutable SQLite database, provider
transport, causal loop, and hidden terminal scorer. The former exposes raw read-only SQL plus a
bounded retrieval service; the latter exposes only SQL execution and recoverable final submission.
Schemes do not share:

- model-visible system prompts;
- top-level argument schemas;
- structured-action validators;
- trajectory or manifest identities;
- SFT targets;
- model adapters/checkpoints or result directories.

## Runtime adapters

`src/tool_modules/registry.py` builds a complete `ToolScheme` object containing the prompt, protocol
version/hash, carrier, top-level actions, primitive actions, and batch bound. It also provides
strict render/parse round trips for local student models.

All text-carried schemes use the same `think-json-v1` carrier while retaining different structured
JSON action schemas. The DeepSeek action-block evaluator may transport reasoning in its
provider-native field plus raw visible JSON, then records the same canonical carrier:

```text
<think>one non-empty reason</think>
{"tool":"action_block","arguments":{"calls":[...]}}
```

The provider and student carriers are adapters around the same structured action. They are
separately hashed and recorded.

### checkpoint-relalg-v1

`checkpoint-relalg-v1` is the forward implementation line for all new tool and experiment work.
The mode is explicit in the launcher, protocol identity, manifests, prompt hash, schema hash, and
result directory; it cannot be inferred or changed inside an episode. The consolidated Chinese
contract is `docs/current/checkpoint_relalg_v1_zh.md`.

- `direct` exposes perception plus read-only `execute_sql`.
- `atomic` exposes perception plus typed relational-algebra operators.
- `hybrid` exposes both direct SQL and typed relational algebra.
- every mode shares the same relation-artifact store, current environment state, semantic
  checkpoint/restore controls, and exact-artifact terminal answer.

Every assistant turn authors exactly one native tool call. Unlike `native-tool-bundle`, there is no
multi-call turn and no shared-pre-state bundle lowering. Checkpoint history is a compact working-
memory path; it is not database evidence and cannot override the current environment or artifact
contents. Simple tasks may answer without creating a checkpoint. A commit records a semantic
milestone; restore is reserved for an explicit contradiction and creates a new active checkpoint
path rather than erasing audit history.

The full approximately 40k-character design specification is a Harness/implementation contract,
not a provider prompt. The model receives a short shared core, one short mode prompt, compact
schemas for the selected mode, and bounded dynamic context. Teacher generation adds only short
checkpoint guidance. The scheme is the forward mainline but remains
`diagnostic-only`: it has no authorized SFT exporter or RL
environment until fresh replay, structure, provider-history, no-leak, behavior, and explicit
admission gates pass.

Atomic `version50` is an isolated exception on the provider side only: DeepSeek receives the
version39 atomic functions through native `tools` and answers with one `tool_calls` item plus
`reasoning_content`. The stored/replay/student carrier remains `think-json-v1`. This keeps the
harness and future SFT representation unchanged while testing whether native function selection
improves external-teacher trajectories. Version50 is selectable only with
`--deepseek-carrier native-tool-calls --diagnostic-only`; it is not the registry default. Its
completed fixed-200 scored 133/200 correct and 183/200 legal versus historical version24 at
145/200 and 197/200, so it is rejected and its trajectories cannot enter SFT/RL.

Version51 is the frozen behavior baseline of the `native-tool-bundle` lineage. Unlike version50, it does not reject
provider-authored parallel calls or non-empty assistant content. Content is retained only for
audit; execution consumes the structured calls. All arguments are checked against the bundle's
shared pre-state before any primitive executes, preventing same-bundle use of unseen derived
handles. Primitive results remain individually replayable and carry `model_turn_index` plus the
original call id, while the causal history retains one assistant message followed by all matching
tool messages. `answer_from_context` must be the sole call. Version51 is diagnostic-only pending
scheme-specific export/admission work; it is nevertheless the base for subsequent provider-tool-
call experiments instead of version26.

The frozen carrier-failure/control Gate32 passed at 22/32 correct and 32/32 legal versus version50
at 16/32 and 26/32, with six paired gains and no regressions. The completed fixed-200 scored
147/200 correct and 200/200 legal versus version50 at 133/200 and 183/200, with 20 gains, six
regressions, 54 versus 166 errors, and essentially unchanged tokens. It is statistically tied with
historical version24's 145/200 and still uses 1.92x its tokens. This promotes version51 only as the
frozen diagnostic baseline for version52+, not to SFT/RL.

Version53 is the frozen reviewed-prompt control. It preserves version52's
function schema, carrier, shared-pre-state validation, execution, state, history, feedback, and
terminal behavior. The shared runtime kernel adds data-as-data, risk-triggered JOIN cardinality,
copied-source versus derived-metric representation, correctness-first scheduling, and independent
filters over distinct handles. Teacher-only rules cover trajectory generation; harness-only
message mechanics are omitted. No live accuracy result or SFT/RL admission is claimed.

Version54 is the frozen no-plan diagnostic. It keeps the exact version53 student runtime prompt,
all eleven non-plan functions and argument schemas, carrier, execution, state, feedback, history,
and terminal behavior. Only the native `plan` schema and its stale teacher-only evidence sentence
are removed. Its v54-only absolute acceptance pilot on the first 200 tasks of the frozen
teacher1500 training-candidate cohort was preregistered but had no result when the lineage was
superseded as the forward development base. Version54 has no live scale claim and remains
ineligible for SFT/RL; keep it for exact reproduction and already-running compatible work.

## Evaluation and causal rollout

Preferred unified evaluation entry:

```bash
PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python src/eval/run_tool_scheme.py \
  --tool-scheme checkpoint-relalg -- --mode hybrid <checkpoint-relalg runner arguments>

PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python src/eval/run_tool_scheme.py \
  --tool-scheme atomic -- <atomic rollout.py arguments>

PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python src/eval/run_tool_scheme.py \
  --tool-scheme native-tool-bundle -- <generate_teacher_rollouts.py arguments>

PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python src/eval/run_tool_scheme.py \
  --tool-scheme action-block -- <action-block evaluator arguments>

PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python src/eval/run_tool_scheme.py \
  --tool-scheme relational-program -- <relational-program evaluator arguments>

PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python src/eval/run_tool_scheme.py \
  --tool-scheme direct-sql-search -- <sql_common/runner.py arguments>

PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python src/eval/run_tool_scheme.py \
  --tool-scheme iterative-sql -- <sql_common/runner.py arguments>
```

The active action-block diagnostic is `action-block-v35`. It keeps v34's short sequential
1..8-operation block, concise prompt, one complete best-practice example, full per-operation
feedback, frozen primitive execution, and the public single-edge
`join(left,right,left_on,right_on,how?)`. V35 changes only `scalar_compute` operands: literals use
`{"value":...}`, same-block one-cell results use `"$id.exact_column"`, and resident one-cell
results use `"step_id.exact_column"`. The harness requires a successful producer, exactly one
source row, and an exact column match before deterministically lowering the reference to the
private grounded scalar carrier. It never chooses a row, matches a suffix, aggregates, or guesses
intent. V32-v34 and historical v4-v18 artifacts remain frozen compatibility inputs for
replay/audit. DeepSeek v4 uses the default provider-native carrier. A local or student-model
OpenAI-compatible endpoint uses:

```bash
--assistant-carrier think-json-v1 --model <student-model-name>
```

The evaluator then parses the model's complete inline turn directly; it does not require a
provider-native reasoning field.

The matching causal generation entry is
`src/sft/generate_tool_scheme_rollouts.py`. It dispatches to the original external-teacher loop for
`atomic`, the real model↔harness action-block loop for `action-block`, and the separate
model↔harness relational-program loop for `relational-program`. `direct-sql-search` dispatches to
the causal iterative-SQL loop with its active interface pinned to `search-values-execute-sql-v2`;
`iterative-sql` dispatches to the same execution runner with the distinct
`execute-sql-submit-sql-v6` protocol.

Every new episode and manifest records:

- `tool_scheme`;
- `tool_scheme_registry_version`;
- protocol version/hash;
- assistant carrier;
- scheme-specific budgets.

## Direct-SQL-search boundary

The active diagnostic `direct-sql-search-v2` exposes exactly the same two top-level tools as the
frozen v1. `search_values` is deterministic,
read-only, and bounded; it retrieves exact stored literals across the database or a declared table/
column using the version45 bounded candidate-recall engine, but it does not create a relation or
choose answer rows. `execute_sql(sql, mode)` runs one read-only SQLite statement. `mode="inspect"`
returns at most 20 rows; `mode="final"` is terminal and must reuse a SQL statement that already
succeeded in inspect mode. There is no third submit or terminal tool.

Frozen v1's paired DeepSeek v4 Flash Gate16 scored 7/16 with 14/16 legal termination, versus fresh atomic
version39 at 10/16 and 16/16. All seven successes replayed and passed the scheme-specific structural
audit. Only two searches were called and both occurred in max-step failures. Keep the scheme as a
low-token diagnostic SQL control; do not expand it or use it for SFT/RL. V2 changes no public
arguments: it adds external-teacher semantic discipline, structured state-preserving errors,
bounded preview-shape facts, resident-pointer history with recent exact outputs, and rejection of
any exact prior successful action on the immutable database. Its preregistered disjoint Holdout
Gate15 scored 6/15 versus fresh atomic version39 at 10/15, with 15/15 legal termination in both
arms, 88 versus 115 actions, and 327,401 versus 854,227 tokens. It passed engineering stability
but failed the accuracy expansion threshold by four tasks. The only search call occurred in a
failure. V2 therefore remains a low-token SQL control and has no accuracy promotion. See
`docs/current/direct_sql_search_tool_scheme_zh.md`.

## Iterative-SQL boundary

The diagnostic `iterative-sql-v6` exposes exactly `execute_sql(sql)` and `submit_sql(sql)`.
`execute_sql` runs one read-only SQLite query and returns a bounded factual preview without ending
the episode. `submit_sql` accepts only `SELECT`/`WITH`, must match an exact previously successful
`execute_sql` query after outer-whitespace/trailing-semicolon normalization, and executes the full
result under the generated-query deadline. Syntax, safety, prior-inspection, timeout, and SQLite
execution failures become structured, state-preserving `LAST SQL ERROR` feedback; the model may
then issue another action in the same episode. A successfully executed submission is terminal and
receives hidden `bird-set` scoring; an executable wrong answer receives no judge feedback.

The active implementation restricts PRAGMA exploration to an allowlist of schema-inspection calls,
rejects exact repeated successful exploration, uses one compact resident SQL state, and keeps gold
SQL and verifier results outside model context. V4 repeats the exact task specification at the
latest context boundary and adds a deterministic syntax-only query-shape audit. Its same-set
development Gate15 scored 12/15 versus v3 7/15, with 15/15 legal termination, six gains, one
regression, and all outcomes passing fresh replay/structure/no-hidden-input-key audit. Because v4
was designed from those same 15 v3 failures, this is not an independent holdout or a training
promotion. V5 keeps the public tools and execution state machine but clarifies evidence authority,
bounded-preview limits, data-dependent finals, targeted exploration, and top-N ties; its audit adds
syntax-only `has_from`/`literal_only_select` facts. On the active baseline300 frozen Prefix20, v5
initially scored 14/20 versus v4 12/20. The completed Prefix50 scored 36/50 versus 34/50, with four
gains, two regressions, exact paired `p=0.6875`, 49/50 versus 50/50 legal, 10 versus seven process
errors, nearly identical actions, and 12.1% more tokens. Stop expansion; this does not change
training admission or establish v5 as a reliable v4 replacement. Frozen v4/v3 and the older
unregistered
`execute_sql_submit_sql_v2` remain explicit reproduction paths. V6 changes only the model-visible
output rule: every answer is a SQL relation, a scalar is 1x1, and multiple mapped fields remain
separate ordered columns unless the task explicitly requests one formatted/combined string. It has
since scored 12/20 versus v5 12/20 on the disjoint frozen tasks 51–70, with one gain, one regression,
20/20 legal in both arms, and 2.8% more tokens. That slice contained no direct multi-field-name
target or explicit-concatenation control, so it does not validate the target rule and provides no
promotion evidence. A subsequent disjoint Target Gate20 directly sampled public multi-field
mappings and controls: v6 scored 15/20 versus v5 10/20, with five gains, zero regressions, 20/20
legal in both arms, 99 versus 111 actions, one versus six errors, and 8.5% fewer tokens. The six
separate-field targets scored 5/6 versus 1/6. All replay/structure audits passed. However, the
predeclared `field1+field2` anti-overseparation controls were invalid because the reference also
required separate columns. V6 therefore passes only to a new representative paired gate and
remains excluded from SFT/RL. See
`docs/current/iterative_sql_tool_scheme_zh.md`.

The completed v6-only baseline300 run reused the audited tasks 51–70 and requested the other 280
episodes. It scored **215/300 (71.67%)** with **298/300 legal**, and all 300 records passed fresh
replay/structure audit. Since tasks 1–70 were consumed by prior design or diagnostics, use the
untouched tasks 71–300 as the primary generalization read: **168/230 (73.04%)**, **228/230 legal**.
The two clean 115-task halves both scored 84/115. This run has no clean230 v5 arm and therefore
establishes only v6 absolute behavior, not a causal v6-over-v5 or cross-scheme improvement. It does
not change training admission.

## Sequential action-block boundary

The action-block model never creates or updates a task plan, DAG, execution status, resident
handles, or environment state. It receives a read-only `AVAILABLE TOOL CONTEXT`, reasons about
the question and returned facts, and emits:

```text
<think>one non-empty reason</think>
{"tool":"action_block","arguments":{"calls":[...]}}
```

`calls` contains one to eight entries and list order is both execution order and feedback order.
It is not a model-authored program: there are no dependency declarations, DAGs, result roots,
exports, plans, or statuses. Every later operation must already be fully determined from facts
visible before the block. It may use `$call_id` or `$call_id.column` to consume an earlier result,
but if an unseen schema, value, row, or error could change the next operation or any argument, the
model ends the block and waits. A one-call block is always valid. Forward and cross-block local
references are invalid; later turns use exact resident handles or step ids returned by the harness.

Every turn chooses one of two actions. A work action is an `action_block` containing one to eight
observation or relational calls. A terminal action is a standalone top-level
`answer_from_context`; its evidence must already be resident before the turn. If any producer or
observation is still needed, the model emits a work action and waits for feedback before answering.

The active adapter performs no spelling, schema, column, predicate, order-by, handle, or intent
repair. It resolves only strict block-local references, deterministically compiles the public
one-edge `join` into exactly one private executor edge, and lowers only verified one-row
`scalar_compute` cell references. A failed primitive is an attempted root error. A later call
referencing it is reported as blocked and is not executed; independent later calls still run.

`join.left_on` is one exact column of the left input and `join.right_on` is one bare column of the
right table. `how` is `inner` by default, or `left`; a cross join sets `how="cross"` and omits both
keys. Multi-hop work uses consecutive one-edge `join` calls and `$id` references. The model never
authors `base`, `joins[]`, `on[]`, `base_role`, or executor namespaces.

For `scalar_compute`, operand order is semantic. A resident duration call has this public form:

```json
{"operation":"date_diff_days",
 "operands":["step_19.START","step_19.STOP"],
 "result_name":"duration_days"}
```

This is legal only when `step_19` produced exactly one row with exact columns `START` and `STOP`.
Multi-row elementwise computation is not silently mapped. The private `value_ref` object is not
part of the v35 `scalar_compute` public schema.

The terminal response is a separate top-level action:

```json
{"tool":"answer_from_context","arguments":{
  "evidence":{"table":"project_001"}
}}
```

`answer_from_context` cannot be nested in `action_block.calls`, so it cannot be mixed with
observation or relational calls. It must cite a result produced in an earlier work action using
its exact resident handle. Its arguments and semantics are identical to atomic
`answer_from_context`: the cited table must already have the exact final rows, columns, column
order, grain, ordering, and duplicates. The harness performs no hidden terminal projection.
When a model cites an observation-only local id such as `read_subtable` as a table, the adapter
rejects the reference without rewriting it. The fact-only error identifies the observation tool
and its observed input relation so the next block can cite a relational producer explicitly.
The action-block model cannot call `plan` or `update_plan`. The shared harness may retain internal
compatibility state for the atomic scheme, but the action-block renderer removes plan state before
constructing model context.

After every nonterminal block, `ACTION BLOCK RESULTS.results[]` contains one entry per submitted
primitive call in the model's list order. Every successful call includes the same complete
model-visible `output` payload as its atomic immediate observation; every attempted failure
includes its error, and every unexecuted dependent includes `blocked_by` and `root_causes`.
There is no duplicate `reusable_outputs` sidecar; successful resident handles are present in the
individual outputs and current harness state.

The bounded history retains four model actions, not four primitive calls. One model action consumes
one action-budget unit regardless of whether it is a one-to-eight-call work block or the standalone
terminal. The v35 external-teacher diagnostic permits up to 40 model actions; primitive counts
remain separate audit metrics.

Result directories must remain isolated by scheme.

## Relational-program boundary

`relational-program-v6` is a separate diagnostic scheme implemented by
`src/tool_modules/relational_program/protocol.py` and
`src/tool_modules/relational_program/evaluator.py`. It does
not add raw SQL or merge the atomic and action-block prompts.

The model-visible prompt is exclusive to three top-level tools: `observe`,
`relational_program`, and `answer_from_context`. It does not include the atomic or action-block
tool definitions. `observe` exposes the `schema`, `column`, and `rows` variants, so a model can
inspect an intermediate result and then submit another program. A deterministic work turn has:

```json
{"tool":"relational_program","arguments":{
  "calls":[
    {"id":"filtered","operation":"filter","arguments":{"table":{"source_table":"orders"},"conditions":{"column":"amount","op":">","value":100}}},
    {"id":"exact","operation":"select","arguments":{"table":{"node":"filtered"},"expressions":["order_id"],"distinct":true}}
  ],
  "result":"exact",
  "exports":[]
}}
```

The model supplies only node ids, program operations, typed references, one primary result, and
optional exported roots. Source relations use `{"source_table":"..."}`, earlier resident outputs
use `{"resident_table":"..."}` or `{"resident_step":"..."}`, and current-program dependencies use
`{"node":"..."}`. A current node's exact join column uses
`{"node":"...","column":"..."}`. These shapes are disjoint: bare strings and the former
`$id`/`$id.column` syntax are invalid in reference positions. The harness derives dependency edges,
validates
undefined references, cycles, result/export roots, and disconnected nodes, and computes a stable
topological order. It then maps each operation to an existing internal relational primitive and
executes the primitives one by one. A failed node is
one root error; transitive descendants are blocked without execution, while independent nodes
continue. Primitive calls—not the program envelope—remain the unit of execution audit and future
process credit.

Only deterministic program operations may appear inside a program:
`filter`, `select`, `scalar`, `join`, `aggregate`, `rank`, and `combine`. Observation, planning,
and termination cannot be nested.
`answer_from_context` stays a separate terminal action citing an already resident exact result.

Version 1 established the graph boundary. Version 2 keeps execution unchanged and makes local
reference lifetime plus the exact local-result join argument shape explicit after the frozen
20-task v1 diagnostic exposed repeated join-shape errors. Version 3 removes all atomic tool names
and definitions from the model-visible prompt and feedback, replacing them with one observation
tool and named program-operation variants while retaining the same internal atomic execution and
credit boundary. Version 4 keeps that exclusive prompt and execution boundary, but replaces
ambiguous sigiled strings with `typed-relational-reference-v1`. The deterministic compiler lowers
the typed public syntax into the private executor carrier while recording every authored reference
and deriving edges only from current-program node references. It also makes the existing
`observe.rows` no-filter contract explicit when a model incorrectly supplies conditions.
Version 5 changes only scalar-value ergonomics: scalar operands directly use typed `node` or
`resident_step` references with an optional named `column`, while computed predicate values use
`value_from` with the same typed source. The compiler lowers both to the unchanged grounded atomic
value-reference semantics. Model-visible errors map private executor names and sigiled references
back to public operations and typed references.
Version 6 fixes one deterministic compiler defect without changing the public call shape: when a
join consumes a current-program node and supplies `base_role`, typed bare columns of that base are
lowered under the declared role rather than the runtime-generated table handle. The lowering is
recorded in the work graph.
The scheme is evaluation/causal-rollout plumbing only. It has no SFT exporter and
no RL environment; all outputs are
`diagnostic_only_pending_protocol_scale_gate` until a frozen scale gate explicitly promotes the
protocol.

## SFT

Atomic bounded-history SFT continues through `src/sft/build_rolling_sft_data.py`.

`checkpoint-relalg-v1` is not an SFT source yet. Its one-native-call turns, explicit mode, relation
artifacts, checkpoint path, provider history, and state hashes require a scheme-aware causal
exporter and explicit admission gate. Do not relabel them as atomic records or infer eligibility
from the fact that both schemes expose relational operations.

Action-block bounded-history SFT is exported by
`src/tool_modules/action_block/sft_export.py`. It:

- requires the source episode to be explicitly marked `sft_export_eligible=true`;
- requires `bird-set` and correct legal termination;
- freshly replays every primitive action and terminal denotation;
- converts provider-native legal history to the inline student carrier;
- creates one last-turn-only record per causal decision prefix;
- records source and target protocol identities;
- permits verified recovery episodes while never making a failed or blocked block a target; later
  clean recovery blocks may be targets with the error block retained only as causal context.

The frozen action-block-v32 fixed-200 evaluation scored 144/200 and remains ineligible because the
150/200 promotion gate was not passed. On the same frozen 20-task diagnostic, v33 scored 11/20
with 18 process errors and 20 blocked descendants; v34 scored 14/20 with seven process errors and
two blocked descendants. V34 had three paired gains and no regressions versus v33, and removed all
observed join-call errors, but remained below atomic version24 (16/20) and historical
action-block-v18 (17/20). V35's frozen four-task scalar-cell gate retained all three controls but
left its single target wrong, so it was not expanded. The exporter remains pinned to the frozen
v32 replay contract; v32-v35 diagnostics are not authorized training sources. See
`docs/reports/evaluation/BIRD_ACTION_BLOCK_V34_ONE_EDGE_JOIN_PILOT20_20260727_ZH.md` and
`docs/reports/evaluation/BIRD_ACTION_BLOCK_V35_SCALAR_CELL_GATE4_20260727_ZH.md`.

Datasets and adapters from different schemes must not be mixed. `assert_record_tool_scheme`
provides the mandatory pre-export guard.

Relational-program diagnostics are never SFT sources. A future exporter would require its own
fresh replay, graph/grounding audit, last-turn-only causal rendering, and a separate accuracy
promotion; it must not reuse the action-block exporter by relabeling records.

Direct-SQL-search diagnostics are also never SFT sources. They have distinct action semantics and
an independent replay audit, and cannot be relabeled as atomic trajectories.

Iterative-SQL v6 diagnostics are never SFT sources until a separate replay/no-leak audit and paired
behavior gate explicitly open admission. They cannot be relabeled as direct-SQL-search or atomic
trajectories.

## RL

There is no `checkpoint-relalg` RL environment yet. The forward-mainline designation applies to
new protocol/tool development, not to training admission. Existing atomic and action-block RL
runs remain supported under their exact stored scheme/protocol hashes; they must not be silently
resumed under `checkpoint-relalg-v1`.

`src/rl/tool_environment.py::create_tool_use_env` constructs either:

- `ToolUseEnv` for `atomic`; or
- `ActionBlockToolUseEnv` for `action-block`.

There is no `relational-program` RL environment in v3. This is deliberate: tool usability and the
primitive-local graph credit boundary must pass evaluation before result-only or process RL is
enabled.

There is no `direct-sql-search` RL environment. Its Gate16 failed the accuracy/legal comparison with
atomic and provided no positive search-use signal.

There is no `iterative-sql` RL environment. Version 6 currently establishes only the causal
evaluation/teacher loop, recoverable submission contract, and a diagnostic-only full300 absolute
result; it has no training admission.

`group_reinforce.py --tool-scheme ...` can therefore run matched result-only training for the two
RL-enabled schemes. Checkpoint metadata, rollout logs, and metrics include the scheme and scheme-specific
budgets, carrier, protocol version, and protocol hash, so resume cannot silently switch protocols
or continue a retired tagged-carrier checkpoint. Such a checkpoint requires an explicit offline
carrier-repair/migration decision rather than metadata fallback.

The action-block RL environment uses the same v35 public join/scalar validators, deterministic
lowering, public feedback renderer, and protocol hash as evaluation. A prompt/runtime mismatch is
a failing test, not a supported configuration.

Atomic process credit is the RL-only mechanism that assigns replay-derived reward to individual
primitive tool actions rather than giving every turn the terminal result reward. No action-block
process-credit objective is selected. Action-block work remains at the tool-feasibility stage, so
only result-only evaluation/RL plumbing exists and process optimization stays disabled.

## Paired experiment rule

For parallel comparison, pin task ids, base model, initialization checkpoint, decoding,
denotation metric, and provider/runtime limits. Use separate:

- SFT datasets and dataset registry names;
- adapters and checkpoint directories;
- rollout/result directories;
- protocol hashes and scheme ids.

Compare terminal `bird-set`, legal rate, transport failures, semantic process errors, model turns,
primitive actions, and tokens. For action-block additionally report submitted calls per block,
root errors, blocked descendants, and information-barrier violations.
