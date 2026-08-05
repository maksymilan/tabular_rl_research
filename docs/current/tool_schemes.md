# Selectable tool schemes

## Contract

The repository exposes four complete and independently selectable model action schemes:

| Scheme id | Model turn | Top-level actions | Student carrier |
|---|---|---|---|
| `atomic` | exactly one primitive tool | the original planning, perception, relational, and terminal tools | `<think>` followed directly by one raw JSON action |
| `action-block` | one ordered work block of 1..8 primitive calls, or one terminal action | `action_block` for work; `answer_from_context` for termination | `<think>` followed directly by one raw JSON action |
| `relational-program` | one interactive observation, one declarative relation program of 1..8 nodes, or one terminal action | `observe`; `relational_program`; `answer_from_context` | `<think>` followed directly by one raw JSON action |
| `direct-sql-search` | one database-value retrieval or one read-only SQL inspection/final execution | `search_values`; `execute_sql` | `<think>` followed directly by one raw JSON action |

The ids are defined by `tool-scheme-registry-v4` in `src/sft/tool_schemes.py`. A model sees exactly
one scheme. Do not combine top-level schemas from different schemes in one prompt and do not infer
a scheme from trajectory shape. Registry v2 unifies both prior student schemes on the active
`think-json-v1` carrier;
registry v3 adds `relational-program` without changing either prior scheme; registry v4 adds the
diagnostic-only `direct-sql-search` scheme without changing the first three. Registry v1's tagged
atomic carrier is retired.

The first three schemes share the harness-owned primitive relational semantics and resident facts.
`direct-sql-search` instead shares the immutable SQLite database, provider transport, causal loop,
and hidden terminal scorer, but exposes raw read-only SQL plus a bounded retrieval service. Schemes do
not share:

- model-visible system prompts;
- top-level argument schemas;
- structured-action validators;
- trajectory or manifest identities;
- SFT targets;
- model adapters/checkpoints or result directories.

## Runtime adapters

`src/sft/tool_schemes.py` builds a complete `ToolScheme` object containing the prompt, protocol
version/hash, carrier, top-level actions, primitive actions, and batch bound. It also provides
strict render/parse round trips for local student models.

Both local student schemes use the same `think-json-v1` carrier. Their structured JSON action
schemas remain different. The DeepSeek action-block evaluator may transport reasoning in its
provider-native field plus raw visible JSON, then records the same canonical carrier:

```text
<think>one non-empty reason</think>
{"tool":"action_block","arguments":{"calls":[...]}}
```

The provider and student carriers are adapters around the same structured action. They are
separately hashed and recorded.

## Evaluation and causal rollout

Preferred unified evaluation entry:

```bash
PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python src/eval/run_tool_scheme.py \
  --tool-scheme atomic -- <atomic rollout.py arguments>

PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python src/eval/run_tool_scheme.py \
  --tool-scheme action-block -- <action-block evaluator arguments>

PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python src/eval/run_tool_scheme.py \
  --tool-scheme relational-program -- <relational-program evaluator arguments>

PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python src/eval/run_tool_scheme.py \
  --tool-scheme direct-sql-search -- <iterative_sql.py arguments>
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
the causal iterative-SQL loop with its active interface pinned to `search-values-execute-sql-v2`.

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
`src/eval/relational_program_protocol.py` and `src/eval/evaluate_relational_program.py`. It does
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

Action-block bounded-history SFT is exported by
`src/sft/build_action_block_sft_data.py`. It:

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

## RL

`src/rl/tool_environment.py::create_tool_use_env` constructs either:

- `ToolUseEnv` for `atomic`; or
- `ActionBlockToolUseEnv` for `action-block`.

There is no `relational-program` RL environment in v3. This is deliberate: tool usability and the
primitive-local graph credit boundary must pass evaluation before result-only or process RL is
enabled.

There is no `direct-sql-search` RL environment. Its Gate16 failed the accuracy/legal comparison with
atomic and provided no positive search-use signal.

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
