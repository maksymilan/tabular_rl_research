# Canonical Execution Contract

Status: active contract for **new** SFT generation, evaluation, and RL episodes under
`v2i-state-only-join-feedback-r2`. `src/sft/protocol.py` is executable authority; this document makes
the ownership boundaries explicit. Old trajectory artifacts remain replay inputs, not examples of
the public action interface.

## One Episode, One State Machine

An episode is a sequence of model actions against one immutable source database and one
harness-managed resident state.

1. In state-only mode, the harness sends `system + user` only. The user message contains the catalog, question,
   optional external knowledge, current environment state, and, only after an error, `LAST TOOL ERROR`.
2. The model emits exactly one non-empty `<think>` block and one complete `<tool_call>` JSON object.
3. The harness strictly parses, validates, executes, records the action, and updates resident state.
4. The next model input is rebuilt from resident state. In bounded rolling mode, up to four prior
   successful assistant actions plus structured result summaries are also retained; full schemas,
   values, and rows remain present only once in resident state.
5. `answer_from_context` is terminal and is execution-scored against the hidden gold SQL result.

```text
<think>brief reason for the next action</think>
<tool_call>{"tool":"tool_name","arguments":{...}}</tool_call>
```

There is no parser repair for a new episode: no closing-tag insertion, JSON completion, shorthand
tool syntax, argument normalization, or silent parameter rewrite.

### Provider Transport Adapters

A provider adapter is allowed only before the strict parser when an API explicitly transports a
model-authored action across separate fields. For example, the DS Flash adapter accepts exactly
`reasoning_content` plus visible content consisting of one complete tool-call block, preserves both
raw fields in the audit record, and carries that existing reasoning into the canonical `<think>`
field. It never invents reasoning, edits JSON/arguments, balances tags, or accepts partial tags.
All other shapes still go to the strict parser unchanged and fail normally. Adapter use is recorded
in the audit turn and successful trajectory metadata.

## Ownership Boundary

| Layer | May author | Must not author |
| --- | --- | --- |
| Model | `think`, `tool`, `arguments`, plan goals/statuses | SQL aliases, step ids, provenance, handles, result rows claimed as facts |
| Harness | execution SQL, handles, `step_id`, state, scalar grounding, references, audit events | semantic guesses that change an invalid model action |
| Dataset adapter | question, DB path, dialect, optional gold SQL | model-facing execution state |
| SFT exporter | state-before-action and legal assistant action | rejected actions as targets |

`plan` is control state only. It cannot supply a factual answer or a `value_ref`. A scalar
`value_ref` names an earlier producing `step_id`; the harness extracts and validates that scalar.

## Public Tool API

New actions may use only:

`plan`, `describe_table`, `inspect_column`, `condition_filter`, `project`, `join_tables`,
`group_aggregate`, `extreme_value_select`, `set_op`, `read_subtable`, and `answer_from_context`.

`aggregate`, `derive_column`, old two-table join fields (`left`, `right`, `join_type`,
`left_prefix`, `right_prefix`), parser shorthands, and truncated-answer repair are **replay-only
compatibility**. They may be read in historical artifacts but are rejected by the strict parser and
must not occur in new SFT/RL/eval actions.

All table-producing tools return a harness-created handle such as `filter_002` or `join_003`.
Handles are the only derived-table names the model may use later. `read_subtable` is the explicit
way to view row values; handles alone expose schema and row-count metadata. Its public `limit` must
be an integer from 1 through 20. Larger or non-integer values are explicit argument-validation
errors and are never silently clamped.

## Current v2i Column Naming

For source tables and unambiguous derived handles, arguments use the schema column name shown by
`describe_table` or the state handle. The model never emits SQLite aliases (`L.`, `R.`) or SQL
fragments.

`join_tables` has one public n-way form:

```json
{
  "tables": ["game", "publisher", "platform"],
  "on": [
    [{"left":"G__id", "right":"game_id"}],
    [{"left":"P__id", "right":"publisher_id"}]
  ],
  "join_types": "inner",
  "prefixes": ["G", "P", "PL"]
}
```

`prefixes` declares stable *column* labels, not SQL aliases and not table handles. With prefixes:

- First edge: the left key may be `P1__column`; the next-table key is its bare source column.
- Later edges: the accumulated left key is a materialized `Pi__column`; the right key is the bare
  source column of the newly attached table.
- Output columns are `Pi__column`; later tools must use exactly the names in the returned handle state.
- Dotted aliases such as `L.G__id`, `R.id`, or `game.id` are invalid.

The first-edge exception exists because v2i prefixes are materialized after that predicate executes.
The harness accepts exactly that documented spelling and never strips arbitrary aliases. It is a
compatibility rule, not an ideal long-term public API.

## Errors and Recovery

Every generated assistant message consumes one action from `max_steps`, including a rejected one.
The recoverable types are `protocol_error`, `argument_validation_error`, and an `execution_error`
whose resident state hash is unchanged.

- The harness preserves state and sends structured `LAST TOOL ERROR` on the next state-only turn.
- Error counts are capped per type; `nonrecoverable_execution_error`, max-step exhaustion, API failure,
  and a wrong terminal denotation end the attempt.
- API transport retries happen inside the client request and are reported separately; they are not
  model actions or recovery events.
- Each error becomes an audit-only `error_event` with action index and before/after state hashes.
- Rejected actions are never SFT targets. The next legal action is marked `feedback_recovery`.
- A correct terminal answer after any error is `recovered_success`; otherwise it is `clean_success`.

Whole episode restarts, when used for pass@k, are separate attempts and retain separate logs.

The active shared implementation is `src/eval/rollout.py`, `src/eval/rollout_passk.py`,
`src/sft/rollout_external_data.py`, and `src/rl/env.py` (used by the Accelerate backend). The
historical Verl token-concatenating adapter is not a v2i training entry point, because state-only
rebuilding needs per-turn loss accounting rather than one appended transcript.

## Migration Rule: v2i Now, v2j Next

Do not mix two naming contracts inside one dataset or evaluation. New v2i data uses the explicit
`prefixes`/`__` rule above. Historical v2h and earlier artifacts are frozen and replay-only.

The proposed v2j migration will replace `prefixes` with harness-derived source-instance names and
use one dotted identifier spelling (`source_instance.column`) everywhere. It requires an atomic
protocol bump and regeneration of compiler output, environment-state schemas, SFT data, tests, and
evaluation artifacts. Until that migration passes replay and tool-smoke gates, dots are not accepted
v2i join identifiers. This avoids the harmful middle state where documentation says one thing and
the executor accepts another.
