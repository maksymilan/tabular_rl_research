# Tool Usage And State Format Spec

This document is the local interface contract for Table Agent Harness tool design. Future tool design changes and harness implementations should follow the formats in this folder unless a new versioned spec is explicitly added.

The machine-readable single source of truth for tool I/O and state field formats is `tool_design/tool_io_spec.json`. To change any format, edit that spec first, then update the matching `tool_json_examples/*.json` and conform every trajectory. `scripts/tool_design/validate_trajectories.py` loads the spec and enforces it; it does not hardcode field sets. This document is the human-readable contract; the spec is the enforced one. For a field-by-field dictionary (meaning, who fills each field, enum value meanings, and a dataset-conversion checklist) see `tool_design/tool_io_spec.md`.

Concrete examples live in `tool_design/tool_json_examples/`. Each tool has one JSON file containing:
- `tool_name`: canonical tool name;
- `purpose`: what the tool is for;
- `tool_call`: model-emitted action format;
- `tool_output`: harness-returned output format;
- `state_delta`: how the harness state changes;
- `notes`: constraints and implementation notes.

Full question trajectories live in `tool_design/trajectory/`. Each trajectory should include a readable `trajectory.md` plus a structured `trajectory.json` that records every tool call and state change.

## Two-Tier Table Model

Tables live in two tiers. This separation keeps potentially large tables out of the model prompt while still letting the model operate on them.

1. Harness-side `data_view` (not in the model prompt). This is the actual table store. It holds both source tables and derived intermediate tables produced by `group_aggregate` and `join_tables`. Each `data_view` table has:
   - `table_name`: canonical id used by every tool call;
   - `kind`: one of `source`, `group`, `join`;
   - column names and semantic types, row count, and full-table numeric statistics;
   - for derived tables, a harness-side `definition` (the operation that produced it) so the harness can re-materialize on demand. The `definition` is never put in the model prompt.

   A `data_view` table may be large. Its full rows are never injected into the model context; the model reads them only through bounded retrieval.

2. Model-visible `dynamic_table_context`. This is a small, budget-bounded reading buffer that holds materialized slices retrieved from `data_view` tables. It is the only place the model reads actual rows.

The model-visible `dataset_overview` is the catalog of lightweight handles over `data_view`: table name, columns and types, row count, table relations, and an optional one-line `intent` for derived tables. It does not contain rows or definitions. The catalog starts with the source tables and grows as `group_aggregate`/`join_tables` append derived-table handles.

Any tool addresses a table by `table_name`, resolved against the `data_view` registry, so source tables and derived intermediate tables are operated on through the same path.

## Model-Visible Intermediate State

The harness should construct the initial model-visible state before the first model action. Dataset overview is not a callable tool because every trajectory needs the same schema-level information.

The model-visible prompt state should contain only the information needed for the next decision:

```json
{
  "question": "年龄在 15-20 的用户对哪些类别的产品更感兴趣？",
  "dataset_overview": {},
  "static_task_memory": [],
  "dynamic_table_context": {},
  "context_budget": {}
}
```

`dataset_overview` is the model-visible catalog of `data_view` handles. At initialization it should include only compact column-level global information:
- canonical table names used by all later tool calls;
- row counts;
- column names and simple semantic types;
- full-table numeric statistics inside the relevant column object when available;
- `relations`: primary/foreign-key links between source tables, each with `from`, `to`, and a `type` cardinality (`one_to_one`, `one_to_many`, `many_to_one`, `many_to_many`).

`relations` is provided at initialization so the model does not have to guess join keys. Cardinality is included because it carries real semantics: it warns against fan-out double-counting in join-then-aggregate, marks the unique primary-key side (so an `entity_match` on it returns at most one row), and identifies the fact-versus-dimension side of a relationship.

When `group_aggregate` or `join_tables` creates a derived table, the harness appends its handle to this catalog with `kind`, columns and types, row count, and an optional one-line `intent`. The catalog must not include sample rows, page titles, dataset annotations, long adapter explanations, or derived-table definitions. This keeps the initial context compact when a dataset contains many columns.

Do not include full `tool_history` in the model-visible prompt by default. Tool history is maintained internally by the harness for duplicate detection, trajectory analysis, and reward computation. If the model needs a reminder, expose only compact summaries, not full raw history.

See `tool_json_examples/state_snapshot.json` for the canonical state snapshot example.

## Dynamic Table Context Format

`dynamic_table_context` is the materialized table view currently visible to the model. The default invariant is rectangular:

```text
selected_rows x selected_columns
```

Canonical fields:
- `context_id`: stable id for this context snapshot.
- `tables`: one or more materialized table views.
- `columns`: visible columns for each table.
- `rows`: visible row ids for each table.
- `data`: actual visible row data.
- `source_step_ids`: tool steps that created or updated this context.

Every materialized slice here is retrieved from a `data_view` table and stays small enough to respect `context_budget`. Group and join results are not stored here directly; they are `data_view` tables (potentially large) that the model reads through bounded retrieval, the same way it reads a source table. Cell-level and scalar derived-fact evidence is stored in Static Task Memory rather than forced into the rectangular table context.

Important tool boundary:
- `retrieve_column_context` may update `dynamic_table_context`, but only with a fixed small number of sample rows for column-level awareness.
- Its only model-controlled input is `requested_columns`, where the model specifies exact table/column names.
- Sample row count and sampling policy are fixed by the harness.
- It must not return the full table or all evidence rows.
- Full-table column statistics belong in the initial `dataset_overview`. `retrieve_column_context` should not repeat or recompute them.
- `retrieve_row_context` is the tool responsible for entity matching, condition filtering, semantic row matching, and returning matched evidence rows.
- All `retrieve_row_context` modes share `table_name`, `search_scope`, `return_columns`, and `top_k`.
- `search_scope` is either `full_table` or `dynamic_table_context`. `full_table` targets the complete `data_view` table named by `table_name`, which may be a source table or a derived `group`/`join` table; `dynamic_table_context` restricts to the currently visible slice. This lets the model operate uniformly on source and derived tables, and recursively on the visible slice.
- Every `retrieve_row_context` result should automatically include `matched_set_statistics` for numeric columns in `return_columns` over the complete matched row set. These statistics are metadata of the newly selected subtable, not a separate aggregation action.
- `matched_set_statistics` should remain complete even when the visible row payload is truncated by `context_budget`. This allows filtered count, sum, mean, min, max, variance, and stddev questions to be answered without exposing thousands of rows.
- All retrieved rows and columns are merged into the existing Dynamic Table Context. Irrelevant rows are removed later through `drop_context`.
- One `retrieve_row_context` call should express one row-retrieval intent. If the answer needs multiple independent evidence targets, use multiple row retrieval calls.
- `query` means the natural-language retrieval intent for one call. It is not a place to bundle a multi-step plan.
- `condition_filter.conditions` may contain multiple predicates only when they form one logical filter, such as an age range `age >= 15 AND age <= 20`.
- Supported first-version condition operators are `=`, `!=`, `>`, `>=`, `<`, `<=`, `contains`, and `is_numeric`.
- Ordered comparisons such as `>`, `<`, `max`, and `min` require a compatible numeric, date, datetime, or otherwise orderable column semantic type.
- `retrieve_row_context.mode` currently supports `entity_match`, `condition_filter`, `semantic_match`, and `extreme_value_select`.
- Use `extreme_value_select` when the goal is to find rows with the maximum or minimum value of a numeric/date/orderable column, such as winner by highest votes.
- If a question asks for a margin/lead/difference and the model has not observed a summary row, prefer retrieving the top two relevant rows with `extreme_value_select(top_k=2)` and computing the difference in memory.
- Adapters must not invent semantic row types such as `candidate_result_rows`. The model must construct a valid row scope with explicit tools and conditions.
- A nested call with `search_scope=dynamic_table_context` operates only on rows currently visible in the dynamic table. If the previous result was truncated, the model cannot claim that a later nested maximum/minimum covers hidden rows.
- Model-visible `retrieve_row_context` output should not expose detailed rows that were excluded from the search scope. Detailed exclusion lists belong in internal tool history.

## Static Task Memory Format

`static_task_memory` is a task-level structured scratchpad. It is observable and editable through tools, but every memory item should include support/check fields to avoid unsupported hallucinations.

Canonical memory item fields:
- `id`: stable memory item id assigned by the harness.
- `type`: one of `schema_fact`, `evidence`, `intermediate_result`, `plan`, `exclusion`, `derived_fact`.
- `content`: concise natural language or structured content.
- `supporting_step_ids`: tool steps that support this memory item.
- `supporting_rows_or_columns`: row, column, cell, table, or result references.
- `confidence_or_check`: why this memory item is believed true.
- `invalidating_condition`: optional condition that would make the item unreliable.

`supporting_rows_or_columns` may contain only `tables`, `rows`, `columns`, and `cells`.

Group and join results are materialized as `data_view` tables, not memory items, so memory no longer uses `group_result`/`join_result` types. When a scalar conclusion is read from a group/join table, store it as a `derived_fact` that references the derived table by `table_name` in `supporting_rows_or_columns.tables` and the producing step in `supporting_step_ids`.

See `tool_json_examples/memory_format.json`.

## Internal Tool History Format

Tool history is internal harness state, not default prompt context. It should record:
- `step_id`;
- `tool_name`;
- `normalized_arguments`;
- `output_summary`;
- `returned_rows_or_columns`;
- `state_delta`;
- `duplicate_check`;
- `timestamp_or_turn_index`.

Duplicate tool-call analysis should compare both normalized parameters and output overlap.

See `tool_json_examples/tool_history_entry.json`.

## Current Tool Set

Tool JSON examples:
- `retrieve_column_context.json`
- `retrieve_row_context.json`
- `retrieve_row_context_entity_match.json`
- `retrieve_row_context_extreme_value_select.json`
- `retrieve_row_context_semantic_match.json`
- `group_aggregate.json`
- `join_tables.json`
- `drop_context.json`
- `add_to_memory.json`
- `refine_memory.json`
- `answer_from_context.json`

Single source of truth (enforced):
- `../tool_io_spec.json` — authoritative field sets and enums for every tool and state object, loaded by `scripts/tool_design/validate_trajectories.py`.

Canonical format examples (human-readable):
- `state_snapshot.json`
- `memory_format.json`
- `tool_history_entry.json`
- `trajectory_format.json`

When adding a tool:
1. Add the tool's `arguments`/`output`/`state_delta` field sets and any new enums to `tool_io_spec.json` first (the single source of truth).
2. Add one JSON example file in `tool_json_examples/`.
3. Update the tool list in this document.
4. Extend `scripts/tool_design/validate_trajectories.py` only with semantic checks the flat spec cannot express (qualified columns, derived-table tracking, etc.).
5. Update `tool_design/agent.md` with the research/design reason.
6. Keep the model-visible output small enough to respect `context_budget`.

When adding a trajectory:
1. Create `trajectory/<question_type>/<case_id>/`.
2. Add `trajectory.md` for readable steps.
3. Add `trajectory.json` with actual tool calls, outputs, state deltas, memory updates, and final answer.
4. Record the source sample path and record index.
5. Put the initialized `dataset_overview` in `initial_state`; do not add an `inspect_dataset` action.
6. Follow `tool_json_examples/trajectory_format.json` exactly; keep Reason/Observation prose in `trajectory.md`.
