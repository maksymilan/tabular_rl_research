# Canonical Execution Contract

Status: active contract for **new** SFT generation, evaluation, and RL episodes under
`version19`. `src/sft/protocol.py` is executable authority; this document makes
the ownership boundaries explicit. Old trajectory artifacts remain replay inputs, not examples of
the public action interface.

## One Episode, One State Machine

An episode is a sequence of model actions against one immutable source database and one
harness-managed resident state.

1. In state-only mode, the harness sends `system + user` only. The user message contains the catalog, question,
   optional external knowledge, current environment state, and, only after an error, `LAST TOOL ERROR`.
2. The canonical action has exactly one non-empty `<think>` block and one complete `<tool_call>`
   JSON object. An API with native reasoning transport receives one provider-specific prompt and
   emits the same reason/tool action across its two fields.
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
`reasoning_content` plus the one visible action shape selected for that experiment: either one raw
JSON action under JSON Output or one complete tool-call block without that constraint. It preserves
both raw fields in the audit record and carries that existing reasoning into the canonical
`<think>` field. It never invents reasoning, edits JSON/arguments, balances tags, or accepts partial
tags. All other shapes still go to the strict parser unchanged and fail normally. The selected
carrier, request controls, adapter use, and raw provider identity are recorded in turns and
manifests.

The API-facing prompt must contain only the selected provider envelope. For DS Flash, positive
instructions to emit a visible `<think>` block are removed before the split-field contract and its
interface-specific example are appended. Bounded rolling history likewise sends prior assistant
content in the selected raw-JSON or tool-call carrier only; canonical `<think>` text remains in the
stored trajectory but is not replayed as a contradictory visible example.

## Ownership Boundary

| Layer | May author | Must not author |
| --- | --- | --- |
| Model | `think`, `tool`, `arguments`, plan goals/statuses | SQL aliases, step ids, provenance, handles, result rows claimed as facts |
| Harness | execution SQL, handles, `step_id`, state, scalar grounding, references, audit events | semantic guesses that change an invalid model action |
| Dataset adapter | question, DB path, dialect, optional gold SQL | model-facing execution state |
| SFT exporter | state-before-action and legal assistant action | rejected actions as targets |

`plan` is control state only. It cannot supply a factual answer or a `value_ref`. A scalar
`value_ref` names an earlier producing `step_id`; the harness extracts and validates that scalar.
For `scalar_compute`, an optional `column` selects one unambiguous non-NULL cell from a prior
one-row table; model-authored values are still outside the trust boundary.
Canonical plan evidence retains the grounded step output for replay and audit. Its model-visible
resident rendering exposes only the evidence `step_id` and tool identity; the authoritative table
or scalar payload already appears under resident tables/values and is not duplicated inside plan.

## Public Tool API

New actions may use only:

`plan`, `describe_table`, `inspect_column`, `condition_filter`, `project`, `scalar_compute`, `join_tables`,
`group_aggregate`, `extreme_value_select`, `set_op`, `read_subtable`, and `answer_from_context`.

`aggregate`, `pivot`, `derive_column`, old two-table join fields (`left`, `right`, `join_type`,
`left_prefix`, `right_prefix`), parser shorthands, and truncated-answer repair are **replay-only
compatibility**. They may be read in historical artifacts but are rejected by the strict parser and
must not occur in new SFT/RL/eval actions.

All table-producing tools return a harness-created handle such as `filter_002` or `join_003`.
Handles are the only derived-table names the model may use later. `read_subtable` is the explicit
way to view row values; handles alone expose schema and row-count metadata. Its public `limit` must
be an integer from 1 through 20. Larger or non-integer values are explicit argument-validation
errors and are never silently clamped.

## Current version19 Conditional Aggregation and Wide Output

`group_aggregate` keeps one input table and one `group_by` grain. Each aggregation may add an
optional `where` predicate:

```json
{
  "table": "patients",
  "group_by": [],
  "aggregations": [
    {
      "op": "count",
      "column": "*",
      "as": "female_count",
      "where": {"column": "gender", "op": "=", "value": "F"}
    },
    {
      "op": "count",
      "column": "*",
      "as": "male_count",
      "where": {"column": "gender", "op": "=", "value": "M"}
    }
  ]
}
```

The harness compiles these predicates as conditional aggregate expressions over the same source
rows. This preserves denominator and row-grain identity while returning one row with the requested
metric columns. `where` accepts the same predicate tree, direct scalar `value_ref`, and computed-set
`in_table` references as `condition_filter`; execution resolves those values while canonical
trajectory arguments retain the authored references. Provenance records the corresponding value,
data, schema, and literal-grounding edges.

The same aggregate action can request ordered category columns instead of category rows:

```json
{
  "table": "hypertension_patients",
  "group_by": ["gender"],
  "aggregations": [
    {"op": "count_distinct", "column": "patient", "as": "patient_count"}
  ],
  "output_layout": "columns",
  "category_values": ["M", "F"]
}
```

The result is one row with `M` followed by `F`; optional `output_columns` can rename those slots
when a later tool needs semantic aliases. This wide layout requires
exactly one grouping column, one aggregation, no passthrough, and a non-empty ordered category list.
The harness compiles category predicates into the same aggregate input, so there is no intermediate
grouped table and no second model action. `project` remains an orthogonal selection/expression atom:
it preserves row orientation and cannot replace wide aggregation. The version16/17 standalone
`pivot` call remains replay-only.

### Named Scalar Cells

A one-row multi-metric aggregate can feed arithmetic directly:

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

The canonical trajectory retains these authored references. During execution, the harness verifies
that `step_7` produced a resident table with exactly one row, resolves each case-insensitive column
name uniquely against that table's canonical output columns, reads the real cell, and rejects NULL.
It records two value-provenance edges, each carrying its operand index and selected column. The
model therefore cannot forge the values, while a single conditional aggregate can support later
percent, difference, ratio, or percent-change arithmetic without redundant branches.

Without `column`, the existing rule remains strict: the producing result must be 1x1. Named-column
references do not apply to predicates, which continue to require a scalar-producing step.

## Current version19 Join and Column Naming

For source tables and unambiguous derived handles, arguments use the schema column name shown by
`describe_table` or the state handle. The model never emits SQLite aliases (`L.`, `R.`) or SQL
fragments.

`join_tables` has one public connected-component form:

```json
{
  "base": "game",
  "joins": [
    {
      "table": "publisher",
      "on": [{"left": "game.publisher_id", "right": "id"}]
    },
    {
      "table": "platform",
      "on": [{"left": "game.platform_id", "right": "id"}],
      "type": "left"
    }
  ]
}
```

The harness assigns each ordinary relation its visible table/handle name as its namespace:

- `on.left` is an exact logical column already introduced, such as `game.publisher_id`.
- `on.right` is the bare source column of the new `joins[]` table, such as `id`.
- `type` is optional and defaults to `inner`; allowed values are `inner`, `left`, and `cross`.
- Output columns are one flat list such as `game.id`, `publisher.name`, and `platform.name`.
  A later join never rewrites these as `join_003.game.id`.
- The public call has no `return_columns`; use the separate `project` atom when narrowing is
  semantically required.

For a repeated relation, and only then, the model supplies semantic instance names:

```json
{
  "base": "employees",
  "base_role": "employee",
  "joins": [{
    "table": "employees",
    "role": "manager",
    "on": [{"left": "employee.manager_id", "right": "id"}]
  }]
}
```

SQL implementation aliases such as `L.` and `R.` are never model-visible. Historical
`tables/on/prefixes/return_columns` and binary join forms remain replay-only.

Version6 keeps the version5 relational call unchanged but compacts wide dotted columns in
model-visible observations and resident state:

```json
{
  "table": "join_003",
  "column_namespaces": {
    "game": ["id", "publisher_id", "platform_id"],
    "publisher": ["id", "name"],
    "platform": ["id", "name"]
  },
  "row_count": 42
}
```

The exact column spelling is reconstructed as `namespace.column`. When continuing a join from
`join_003`, the handle remains the `base` table argument, but `on.left` must use one of these
logical namespaces, never `join_003.column`. This is a rendering-only compression: canonical
harness snapshots and replay artifacts retain their full flat column lists.

Version7 completes downstream consumption of the same logical names. Exact dotted identifiers are
quoted as one physical column when they appear inside `project` scalar expressions. Filters,
projection, grouping, and ordering may also use a bare suffix only when it resolves to exactly one
available logical column; ambiguous bare names remain explicit errors. `join_tables.on.left`
continues to require the exact namespace-qualified form because it selects a relation instance.

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
`src/sft/generate_teacher_rollouts.py`, and `src/rl/tool_environment.py` (used by the Accelerate backend). The
historical Verl token-concatenating adapter is not a version19 training entry point, because state-only
rebuilding needs per-turn loss accounting rather than one appended transcript.

## Migration Rule

Do not mix naming contracts inside one dataset or evaluation. New diagnostic episodes use version19,
but new SFT construction remains gated on the frozen 200-task accuracy requirement. Historical
version1-version18 artifacts retain their original model-visible contracts and may only enter
replay-compatible paths. Every version19 teacher/eval result directory and manifest must record its
version19 protocol hash and provider request controls; historical data is not relabeled or mutated
in place.
