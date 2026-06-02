# Tool Usage And State Format Spec

This document is the local interface contract for Table Agent Harness tool design. Future tool design changes and harness implementations should follow the formats in this folder unless a new versioned spec is explicitly added.

Concrete examples live in `tool_design/tool_json_examples/`. Each tool has one JSON file containing:
- `tool_name`: canonical tool name;
- `purpose`: what the tool is for;
- `tool_call`: model-emitted action format;
- `tool_output`: harness-returned output format;
- `state_delta`: how the harness state changes;
- `notes`: constraints and implementation notes.

Full question trajectories live in `tool_design/trajectory/`. Each trajectory should include a readable `trajectory.md` plus a structured `trajectory.json` that records every tool call and state change.

## Model-Visible Intermediate State

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

Cell-level, group-level, join-result-level, and derived-fact evidence should be stored in Static Task Memory rather than forcing all evidence into the rectangular table context.

Important tool boundary:
- `retrieve_column_context` may update `dynamic_table_context`, but only with a fixed small number of sample rows for column-level awareness.
- Its primary input should be `requested_columns`, where the model specifies exact table/column names and what information it wants for each column.
- `request_reason` may explain why those columns are needed, but it should not replace explicit column requests.
- It must not return the full table or all evidence rows.
- For numeric columns, `retrieve_column_context` should return precomputed statistics when available, such as min, max, mean, variance/stddev, count, and parse coverage. These statistics support later row-selection and aggregation decisions without exposing all rows.
- If `retrieve_column_context` already returns the exact statistic needed by the question, the model may use that statistic directly and write it to memory.
- Use `aggregate_column` when the needed statistic is not returned by column context, when a filter/group-by is required, or when the trajectory needs an explicit auditable recomputation step.
- `aggregate_column` should return compact operation results and optional audit metadata, not full row values. This prevents context collapse on large tables and reduces arithmetic hallucination.
- `retrieve_row_context` is the tool responsible for entity matching, condition filtering, semantic row matching, and returning matched evidence rows.
- One `retrieve_row_context` call should express one row-retrieval intent. If the answer needs multiple independent evidence targets, use multiple row retrieval calls.
- `query` means the natural-language retrieval intent for one call. It is not a place to bundle a multi-step plan.
- `condition_filter.conditions` may contain multiple predicates only when they form one logical filter, such as an age range `age >= 15 AND age <= 20`.
- `retrieve_row_context.mode` currently supports `entity_match`, `condition_filter`, `semantic_match`, and `extreme_value_select`.
- Use `extreme_value_select` when the goal is to find rows with the maximum or minimum value of a numeric/date/orderable column, such as winner by highest votes.
- If a question asks for a margin/lead/difference and the model has not observed a summary row, prefer retrieving the top two relevant rows with `extreme_value_select(top_k=2)` and computing the difference in memory.
- Model-visible `retrieve_row_context` output should not expose detailed rows that were excluded from the search scope. Detailed exclusion lists belong in internal tool history; model-visible output may include only compact scope summaries.

## Static Task Memory Format

`static_task_memory` is a task-level structured scratchpad. It is observable and editable through tools, but every memory item should include support/check fields to avoid unsupported hallucinations.

Canonical memory item fields:
- `id`: stable memory item id assigned by the harness.
- `type`: one of `schema_fact`, `evidence`, `intermediate_result`, `plan`, `exclusion`, `derived_fact`, `join_result`, `group_result`.
- `content`: concise natural language or structured content.
- `supporting_step_ids`: tool steps that support this memory item.
- `supporting_rows_or_columns`: row, column, cell, table, or result references.
- `confidence_or_check`: why this memory item is believed true.
- `invalidating_condition`: optional condition that would make the item unreliable.

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
- `inspect_dataset.json`
- `retrieve_column_context.json`
- `aggregate_column.json`
- `retrieve_row_context.json`
- `retrieve_row_context_extreme_value_select.json`
- `drop_context.json`
- `add_to_memory.json`
- `refine_memory.json`
- `answer_from_context.json`

When adding a tool:
1. Add one JSON example file in `tool_json_examples/`.
2. Update the tool list in this document.
3. Update `tool_design/agent.md` with the research/design reason.
4. Keep the model-visible output small enough to respect `context_budget`.

When adding a trajectory:
1. Create `trajectory/<question_type>/<case_id>/`.
2. Add `trajectory.md` for readable steps.
3. Add `trajectory.json` with actual tool calls, outputs, state deltas, memory updates, and final answer.
4. Record the source sample path and record index.
