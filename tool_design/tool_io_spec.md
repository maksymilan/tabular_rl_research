# Tool I/O Field Reference & Dataset-Construction Standard

This is the authoritative, human-readable dictionary for every field in a trajectory.
When converting **any** new dataset into trajectory data, follow the field meanings and
ownership defined here.

Companion files:
- `tool_io_spec.json` — machine-readable single source of truth (field sets + enums), enforced by `scripts/tool_design/validate_trajectories.py`.
- `tool_usage.md` — the design contract and rationale.

## Who fills a field: the location rule

Every field belongs to exactly one of three roles. The role is determined by **where the field sits**:

| Location in a trajectory | Role | Who produces it |
| --- | --- | --- |
| Trajectory top level (except `initial_state`/`steps`) | **DATA** | Source dataset + human author/annotator |
| `initial_state.*` | **HARNESS** | The environment, constructed at t=0 |
| `step.tool_call.*` | **MODEL** | The policy's action |
| `step.tool_output.*` | **HARNESS** | The environment's response to the action |
| `step.state_delta.*` | **HARNESS** | The environment's bookkeeping |
| All ids (`step_id`, `context_id`, `mem_*`, derived table names) | **HARNESS** | Assigned by the environment |

Legend used below: **[DATA]** dataset/author · **[MODEL]** policy action · **[HARNESS]** environment.

When you hand-construct a trajectory (no live harness yet), you wear all three hats in turn:
author the DATA metadata, write each MODEL `tool_call`, then write the HARNESS
`tool_output`/`state_delta` that a correct environment would have returned.

---

## 1. Trajectory-level fields — [DATA]

| Field | Meaning | How the value is obtained |
| --- | --- | --- |
| `trajectory_id` | Unique name of this trajectory | Author chooses |
| `question_type` | Coarse reasoning-type label (e.g. `EntityLookup`, `NumericalReasoning/FilterAggregation`, `MultiTableJoinAggregation`). For organizing/analysis only; the harness loop ignores it | Author classifies |
| `source_sample` | Provenance block (see below) | Author + dataset |
| `source_sample.path` | Sample file the record came from | Author (e.g. `data_sample/xxx.json`) |
| `source_sample.record_index` | Index of the record inside that file | Author |
| `source_sample.dataset` | Which dataset (`fetaqa` / `tablebench` / `tqabench` / ...) | Author |
| `source_sample.record_id` | Original id in the source dataset | From dataset |
| `question` | The question to answer | From dataset (or authored, if the dataset's native question is unusable — document this in `dataset_annotations`) |
| `dataset_gold_answer` | The dataset's claimed gold answer label. May be wrong/unreproducible — see `label_assessment` | From dataset (or author-verified, when authored over a snapshot) |
| `dataset_annotations` | Container: dataset-provided evidence + author's judgment | See §1.1 |
| `initial_state` | The t=0 model-visible state (HARNESS — see §2) | Harness constructs |
| `steps` | The ordered tool-use steps (MODEL + HARNESS, see §3) | Authored/simulated |

### 1.1 `dataset_annotations` — [DATA]

| Field | Meaning | How obtained |
| --- | --- | --- |
| `evidence` | Dataset-provided evidence annotation (e.g. FeTaQA `highlighted_cell_ids`); `{}` if none | From dataset |
| `label_assessment.status` | `accepted` or `conflict` (see enum below) | Author judges |
| `label_assessment.reason` | Why; may be `null` | Author writes |

`label_assessment.status` controls the final-answer check:
- **`accepted`** — the table reproduces the dataset label; the validator requires `final_answer == dataset_gold_answer`.
- **`conflict`** — the dataset label conflicts with the table evidence; the validator requires `final_answer != dataset_gold_answer`. Keeps the original label while honestly flagging it.

---

## 2. `initial_state` — [HARNESS]

Constructed by the environment before the first action; the model only reads it.

| Field | Meaning | How obtained |
| --- | --- | --- |
| `dataset_overview` | Model-visible catalog of table handles | Harness computes from source tables (see §2.1) |
| `static_task_memory` | Task memory list | Starts empty `[]`; filled later by `add_to_memory` |
| `dynamic_table_context` | Visible table-reading buffer | Starts `{context_id, tables: []}`; filled by retrieval |
| `context_budget` | Context-usage meters | Harness maintains (see §2.2) |

### 2.1 `dataset_overview` — [HARNESS]

| Field | Meaning | How obtained |
| --- | --- | --- |
| `tables[]` | One handle per source table | Harness |
| `tables[].table_name` | Canonical table id used by every tool | Harness/author names it |
| `tables[].num_rows` | Row count of the table | Harness counts the source table |
| `tables[].columns[].name` | Column name | From source schema |
| `tables[].columns[].semantic_type` | `identifier` / `numeric` / `text` / `datetime` / `date` | Harness infers from the column |
| `tables[].columns[].statistics` | Full-table numeric stats (optional; numeric columns only) | **Harness pre-computes over the whole table** (gives the model a baseline without exposing rows) |
| `tables[].columns[].unit` / `format` | Optional unit/format hints | Harness/author, if known |
| `relations[]` | Primary/foreign-key links between source tables (optional) | Harness/author declares (see §2.3) |

`columns[].statistics` keys: `count`, `non_null_count`, `parse_coverage`, `sum`, `mean`, `min`, `max`, `variance`, `stddev`.

### 2.2 `context_budget` — [HARNESS]

| Field | Meaning |
| --- | --- |
| `max_rows_visible` | Cap on visible rows |
| `max_columns_visible` | Cap on visible columns |
| `current_rows_visible` | Rows currently in the buffer |
| `current_columns_visible` | Columns currently visible |
| `estimated_tokens` | Estimated tokens of the currently visible content |

### 2.3 `relations[]` — [HARNESS]

| Field | Meaning | How obtained |
| --- | --- | --- |
| `from` | Foreign-key column, `table.column` | Harness/author |
| `to` | Referenced (usually primary-key) column, `table.column` | Harness/author |
| `type` | Cardinality (see enum) | Harness/author |

Provided at t=0 so the model does not guess join keys. Cardinality also warns about
fan-out double-counting in join-then-aggregate, and marks the unique (PK) side.

---

## 3. Steps — `tool_call` [MODEL] · `tool_output` [HARNESS] · `state_delta` [HARNESS]

Each step: `step_id` [HARNESS], `tool_name` (= `tool_call.tool`), then the model's action
and the harness's response.

### 3.1 What the MODEL fills (`tool_call.arguments`)

The model chooses these based on the question and the current visible state.

| Tool | Model-filled arguments |
| --- | --- |
| `retrieve_column_context` | `requested_columns` (list of `{table_name, column}`) |
| `retrieve_row_context` | `mode`, `table_name`, `search_scope`, `return_columns`, `top_k` + mode-specific: `conditions` / `query`+`entity_columns` / `target_column`+`order` / `query`+`semantic_columns` |
| `group_aggregate` | `table_name`, `search_scope`, `group_by`, `aggregations` (`{op, column?, as}`) |
| `join_tables` | `left_table`, `right_table`, `on` (`[{left, right}]`), `join_type`, `return_columns` |
| `drop_context` | `drop_rows`, `drop_columns`, `reason` |
| `add_to_memory` | `items` — memory items **without** `id` (see §4.1) |
| `refine_memory` | `operations` (`{op, source_ids, new_item?/reason?}`) |
| `answer_from_context` | `answer`, `evidence_rows`, `evidence_columns`, `supporting_memory_ids`, `reason` |

### 3.2 What the HARNESS returns (`tool_output`)

| Output field | Meaning | How obtained |
| --- | --- | --- |
| `retrieved_columns` | Columns materialized (column-retrieval) | Harness |
| `sample_row_policy` | `{type, num_rows}` — fixed sample policy | Harness |
| `matched_rows` / `matched_row_count` | Visible matched rows / full matched count | Harness executes the match |
| `matched_set_statistics` | Numeric stats over the **complete** matched set (see §4.2) | **Harness computes** |
| `visible_row_policy` | `{top_k, matched_rows_total, visible_rows_returned, truncated}` | Harness |
| `extreme_value_result` | `{target_column, order, selected_values:[{row,value,rank}], tie_policy}` | Harness |
| `semantic_match_result` | Match scores for `semantic_match` | Harness (adapter) |
| `current_table_context` | Materialized visible slice (see §4.3) | Harness |
| `created_table` | Derived-table handle for group/join (see §4.4) | **Harness builds the derived table** |
| `group_sample` / `row_sample` | A bounded sample of the derived table's rows | Harness |
| `group_sample_policy` | `{type, num_rows, groups_total, truncated}` | Harness |
| `row_sample_policy` | `{type, num_rows, rows_total, truncated}` | Harness |
| `join_diagnostics` | `{join_type, on, relation_used, left_rows, right_rows, result_rows, fanout_warning, unmatched_left_rows}` | Harness |
| `dropped` | `{rows, columns}` actually removed | Harness |
| `task_memory` | Stored memory list — **now each item has an `id`** | Harness assigns ids and stores |
| `refine_trace` | `[{op, source_ids, target_id?}]` | Harness |
| `final_answer` | Echo of the submitted answer | Harness |
| `evidence` | `{rows, columns, memory_ids}` cited evidence | Harness echoes the model's citation |

### 3.3 `state_delta` — [HARNESS]

Bookkeeping of what changed. Shapes by tool family:

| Tool family | `state_delta` fields |
| --- | --- |
| `retrieve_column_context`, `retrieve_row_context` | `dynamic_table_context_changed`, `context_id_before`, `context_id_after`, `added_rows`, `added_columns`, `static_task_memory_changed` |
| `group_aggregate`, `join_tables` | `data_view_changed`, `added_table`, `dynamic_table_context_changed`, `static_task_memory_changed` |
| `drop_context` | `dynamic_table_context_changed`, `context_id_before`, `context_id_after`, `removed_rows`, `removed_columns` |
| `add_to_memory` | `static_task_memory_changed`, `added_memory_ids`, `dynamic_table_context_changed` |
| `refine_memory` | `static_task_memory_changed`, `removed_memory_ids`, `added_memory_ids`, `dynamic_table_context_changed` |
| `answer_from_context` | `final_answer_submitted`, `dynamic_table_context_changed`, `static_task_memory_changed` |

---

## 4. Shared sub-objects

### 4.1 `memory_item` — content [MODEL], `id` [HARNESS]

| Field | Meaning | Who |
| --- | --- | --- |
| `id` | Stable memory id (`mem_001`, ...) | **[HARNESS]** — present only in stored output, never in the model's `add_to_memory` input |
| `type` | One of the `memory_type` enum | [MODEL] |
| `content` | NL string or structured object | [MODEL] |
| `supporting_step_ids` | Steps that support this item | [MODEL] |
| `supporting_rows_or_columns` | Evidence refs; keys only `tables`/`rows`/`columns`/`cells` | [MODEL] |
| `confidence_or_check` | Why it is believed true | [MODEL] |
| `invalidating_condition` | What would make it unreliable | [MODEL] |

A memory item is therefore **model content + harness storage**: the model writes everything
except `id`, which the harness assigns on store.

### 4.2 `matched_set_statistic` — [HARNESS]

`table_name`, `column`, `scope`, `count`, `non_null_count`, `sum`, `mean`, `min`, `max`,
`variance`, `stddev`, `parse_coverage`. Computed over the full matched set (the zero-group
case of an aggregation).

### 4.3 `context_table` (inside `current_table_context`) — [HARNESS]

`table_name`, `columns`, `rows` (visible row ids), `data` (each row carries `_row_id`),
`source_step_ids`. `rows` and `data` must align by `_row_id`.

### 4.4 `created_table` (group/join result handle) — [HARNESS]

| Field | Meaning |
| --- | --- |
| `table_name` | Derived-table id (e.g. `g_flights_by_carrier`, `j_carrier_named`) |
| `kind` | `group` or `join` |
| `intent` | One-line natural-language description of the derived table |
| `columns` | `[{name, semantic_type}]` flat output columns |
| `num_rows` | Row count of the derived table |
| `definition_in_harness` | `true` — the re-materializable definition lives harness-side, never in the prompt |

### 4.5 Model-filled argument sub-objects — [MODEL]

| Object | Fields | Used by |
| --- | --- | --- |
| `requested_column` | `table_name`, `column` | `retrieve_column_context` |
| `condition` | `column`, `operator`, `value` | `retrieve_row_context` (condition_filter) |
| `join_condition` | `left`, `right` | `join_tables.on` |

---

## 5. Enum value meanings

### `memory_type`
- `schema_fact` — schema/linkage understanding (e.g. which columns join).
- `evidence` — a directly observed fact from the table.
- `intermediate_result` — an intermediate computation.
- `plan` — a multi-step plan.
- `exclusion` — something deliberately ruled out.
- `derived_fact` — a derived conclusion (e.g. a computed sum/mean/margin, or a value read from a group/join table).

### `search_scope`
- `full_table` — the complete `data_view` table named by `table_name` (source **or** derived).
- `dynamic_table_context` — only the currently visible slice.

### `retrieve_row_context` modes (the keys of `mode_arguments`)
The valid modes are defined by the keys of `tools.retrieve_row_context.mode_arguments` in the spec (there is no separate `row_mode` enum).
- `entity_match` — find rows matching an entity string in given columns.
- `condition_filter` — filter rows by structured predicates.
- `semantic_match` — rank rows by semantic similarity.
- `extreme_value_select` — pick max/min rows of an orderable column.

### `condition_operator`
`=`, `!=`, `>`, `>=`, `<`, `<=`, `contains`, `is_numeric`. Ordered ops (`> < >= <=`) require a numeric/date/orderable column.

### `extreme_order`
- `max` — largest value(s); `min` — smallest value(s).

### `aggregation_op` (per group in `group_aggregate`; whole set when zero group keys)
- `count` — number of rows per group (no column needed). E.g. flights per carrier.
- `sum` — total of a numeric column. E.g. total GDP, total goals.
- `mean` — average. E.g. average price, average cyclones per season.
- `min` — smallest value. E.g. earliest date, lowest price.
- `max` — largest value. E.g. highest score, latest date.

### `join_type`
- `inner` — only matched rows; `left` — all left rows, nulls where unmatched.

### `derived_table_kind`
- `group` — a `group_aggregate` result; `join` — a `join_tables` result.

### `relation_cardinality` (for `from -> to`)
- `many_to_one` — many `from` rows map to one `to` row; the `to` side is unique (PK). E.g. `orders.user_id -> users.user_id`, `Airlines.OP_CARRIER_AIRLINE_ID -> Air Carriers.Code`.
- `one_to_many` — the reverse: one `from` maps to many `to`. E.g. `users.user_id -> orders.user_id`.
- `one_to_one` — both sides unique. E.g. `users.user_id -> user_profiles.user_id`.
- `many_to_many` — many on both sides (usually via a junction table). E.g. students <-> courses.

`one_to_many` and `many_to_many` joins multiply rows (fan-out) and can inflate `sum`/`count`
after a join; `many_to_one` does not.

### `label_status`
- `accepted` — final answer must equal `dataset_gold_answer`.
- `conflict` — final answer must differ from `dataset_gold_answer` (dataset label disputed).

---

## 6. Dataset-conversion workflow (use this standard)

To turn a record from a new dataset into a trajectory:

1. **Build `dataset_overview`** (HARNESS role) from the new dataset's table(s): `table_name`, `num_rows`, every `column` with a `semantic_type`, numeric `statistics`, and `relations` (with cardinality) for multi-table data.
2. **Fill trajectory metadata** (DATA role): `question`, `dataset_gold_answer`, `source_sample`, `question_type`, and `dataset_annotations` (including `label_assessment`).
3. **Verify the gold against the table.** If the table reproduces it, set `label_assessment.status = accepted`. If not (e.g. the dataset gold is computed over data not present in the provided table), either set `conflict`, or author a snapshot-verified question and record both the native and authored Q/A in `dataset_annotations.evidence`.
4. **Author the steps** in order: each `tool_call` (MODEL) using only the argument fields above; each `tool_output` and `state_delta` (HARNESS) exactly as a correct environment would return them. Reference columns as `table.column`; derived-table columns as `derived_table.column`.
5. **End with `answer_from_context`** whose `final_answer` satisfies the `label_status` rule.
6. **Validate**: `python3 scripts/tool_design/validate_trajectories.py`. It loads `tool_io_spec.json` and checks every field set, enum, qualified column, derived-table reference, and the final-answer/label rule.

Any field added or changed must be edited in `tool_io_spec.json` **first**, then reflected here and in `tool_json_examples/`.
