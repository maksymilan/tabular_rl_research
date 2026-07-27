# Selectable tool schemes

## Contract

The repository exposes two complete and independently selectable model action schemes:

| Scheme id | Model turn | Top-level actions | Student carrier |
|---|---|---|---|
| `atomic` | exactly one primitive tool | the original planning, perception, relational, and terminal tools | `<think>` followed directly by one raw JSON action |
| `action-block` | one ordered work block of 1..5 primitive calls, or one terminal action | `action_block` for work; `answer_from_context` for termination | `<think>` followed directly by one raw JSON action |

The ids are defined by `tool-scheme-registry-v2` in `src/sft/tool_schemes.py`. A model sees exactly
one scheme. Do not combine both top-level schemas in one prompt and do not infer a scheme from
trajectory shape. Registry v2 unifies both student schemes on the active `think-json-v1` carrier;
registry v1's tagged atomic carrier is retired.

Both schemes share the harness-owned primitive relational semantics and resident facts. They do
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
```

The active action-block protocol is `action-block-v32`. It uses the strict ordered
interface. Historical v4-v18 artifacts and explicitly named parsers remain compatibility inputs
for replay/audit; the active launcher does not select them. DeepSeek v4 uses the default
provider-native carrier. A local or student-model OpenAI-compatible endpoint uses:

```bash
--assistant-carrier think-json-v1 --model <student-model-name>
```

The evaluator then parses the model's complete inline turn directly; it does not require a
provider-native reasoning field.

The matching causal generation entry is
`src/sft/generate_tool_scheme_rollouts.py`. It dispatches to the original external-teacher loop for
`atomic` and the real model↔harness action-block loop for `action-block`.

Every new episode and manifest records:

- `tool_scheme`;
- `tool_scheme_registry_version`;
- protocol version/hash;
- assistant carrier;
- scheme-specific budgets.

## Unified action-block boundary

The action-block model never creates or updates a task plan, DAG, execution status, resident
handles, or environment state. It receives a read-only `AVAILABLE TOOL CONTEXT`, reasons about
the question and returned facts, and emits:

```text
<think>one non-empty reason</think>
{"tool":"action_block","arguments":{"calls":[...]}}
```

`calls` contains one to five entries and list order is both execution order and feedback order. A
block is one semantic action: every included call must be fully selectable from facts visible
before the block. An execution-dependent later call may use `$call_id` or `$call_id.column` only
for an earlier call in the same list and only when its tools, columns, predicates, literals, grain,
and outputs were already grounded before the block. Forward and cross-block local references are
invalid; later blocks use exact resident handles or step ids returned by the harness. If unseen
feedback could change a choice, the block ends before that choice.

Every turn chooses one of two actions. A work action is an `action_block` containing one to five
observation or relational calls. A terminal action is a standalone top-level
`answer_from_context`; its evidence must already be resident before the turn. If any producer or
observation is still needed, the model emits a work action and waits for feedback before answering.

The active adapter performs no spelling, schema, column, predicate, order-by, handle, or argument
shape repair. It resolves only the strict block-local reference syntax required to connect ordered
primitive calls. A failed primitive is an attempted root error. A later call referencing it is
reported as blocked and is not executed; independent later calls still run.

When a derived handle is the base of `join_tables`, its exact handle is the default logical
namespace for `on.left`: for example, base `filter_001` uses `filter_001.column`. Supplying
`base_role:"orders"` instead makes `orders.column` the left namespace. `on.right` remains the new
table's bare column.

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
one action-budget unit regardless of whether it is a one-to-five-call work block or the standalone
terminal. Primitive counts remain audit metrics and do not terminate an episode.

Result directories must remain isolated by scheme.

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

The active action-block-v32 fixed-200 evaluation scored 144/200 and remains ineligible because the
150/200 promotion gate was not passed. The exporter is an implementation path for a future
promoted protocol, not an authorization to train on current or historical diagnostics.

Datasets and adapters from different schemes must not be mixed. `assert_record_tool_scheme`
provides the mandatory pre-export guard.

## RL

`src/rl/tool_environment.py::create_tool_use_env` constructs either:

- `ToolUseEnv` for `atomic`; or
- `ActionBlockToolUseEnv` for `action-block`.

`group_reinforce.py --tool-scheme ...` can therefore run matched result-only training for both
schemes. Checkpoint metadata, rollout logs, and metrics include the scheme and scheme-specific
budgets, carrier, protocol version, and protocol hash, so resume cannot silently switch protocols
or continue a retired tagged-carrier checkpoint. Such a checkpoint requires an explicit offline
carrier-repair/migration decision rather than metadata fallback.

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
