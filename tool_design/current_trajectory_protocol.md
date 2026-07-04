# Current Trajectory Protocol (v2c-plan / v3+state)

Status: index-level contract for the currently implemented trajectory format. This file does not
replace code; it points to the source of truth and records what must not drift.

## Sources of Truth

- Model-visible protocol: `src/sft/protocol.py`
  - `PROTOCOL_VERSION`
  - `SYSTEM_PROMPT`
  - `TOOL_SPECS`
  - per-tool argument schema
  - message rendering/parsing
- Harness provenance sidecars: `src/harness/provenance.py`
- Scalar `value_ref` grounding: `src/harness/scalar_grounding.py`
- Frozen provenance redesign and implementation status: `draft/provenance_redesign.md`
- SFT export rendering: `src/sft/build_sft_data.py`
- Live rollout rendering: `src/eval/rollout.py` and `src/eval/rollout_passk.py`

If these disagree, fix the code and this index together. Do not infer the active protocol from old
`tool_design/tool_usage.md` examples.

## Model-Visible Format

The model sees a normal ReAct dialogue:

```text
system: protocol/system prompt
user: DATASET OVERVIEW + QUESTION
assistant: <think>...</think>
           <tool_call>{"tool": "...", "arguments": {...}}</tool_call>
user: {"step_id":"step_1","status":"success","output":{...}}
user: CURRENT ENVIRONMENT STATE
     {"plan":[...],"tables":{...}}     # ephemeral, only in the next model input
...
assistant: final answer_from_context tool call
```

The environment state is harness-managed resident context. It groups the current task plan and
known table context by source table or derived handle. It is NOT part of historical tool
observations. Online rollout/RL constructs each model input as:

```
system + initial user + historical assistant/tool-output transcript + latest CURRENT ENVIRONMENT STATE
```

The latest environment message is a mutable side channel for the next model call; it is not appended
to the durable conversation history. SFT full-transcript exports omit resident state rather than
duplicating stale environment snapshots in every observation. If we train directly on the side
channel later, use per-turn examples or another format that can replace the state message.

Raw SQL-compiled trajectories contain the verified relational backbone only. They do not
mechanically inject `describe_table`, `inspect_column`, or `read_subtable`; those perception steps
are model-visible tools and should be inserted by the external-model enrichment pass when the
current reasoning context needs them.

Plan steps are added by `src/sft/enrich_plan.py`: an external model proposes the initial `plan`
operation and later updates, then the sequence is replayed through the harness. The script's
`--dry-run-template` mode is for smoke tests only.

The model emits only:

- `<think>` text;
- one `<tool_call>` JSON object.

The model does not emit provenance, references, produces, quality status, repair metadata, or reward
fields.

## Current Tool Set

The current v2b tool set is the one in `src/sft/protocol.py::TOOL_SPECS`:

- `condition_filter`
- `plan`
- `project`
- `join_tables`
- `group_aggregate`
- `aggregate`
- `extreme_value_select`
- `set_op`
- `describe_table`
- `inspect_column`
- `read_subtable`
- `answer_from_context`

No `add_to_memory` tool exists in v2c-plan. No reflection or invalidate tool exists.

## Plan Status

`plan(ops)` is model-visible task-control state managed by the harness. The model can create,
add, update, or delete subgoals. Each plan item separates completion from conclusion:

- `status`: `pending|in_progress|done|blocked` says whether the subtask is complete;
- `result`: optional model-authored subtask answer/conclusion, such as
  `{type:"boolean", value:true}`, `{type:"scalar", value:3}`, or
  `{type:"text", summary:"..."}`.

The plan is not factual evidence:

- it cannot be used through `value_ref`;
- it cannot support the final answer;
- it is excluded from data/value provenance slices;
- it may receive separate planning/process rewards.

External-model plan enrichment should generate natural plan wording and updates, while deterministic
checks enforce that plan items do not leak facts unsupported by later observations.

## Memory Status

Memory is not part of the current model-visible protocol:

- no `add_to_memory`;
- no `memory_id`;
- no `mem_` references;
- no `supporting_memory_ids`.

Computed scalars are reused through `condition_filter.conditions[*].value_ref`, and `value_ref`
points directly to the step id that produced the scalar. The harness performs scalar extraction and
validation.

## Harness-Owned Fields

Trajectory JSON may contain harness-owned fields that are not model actions:

- `step_id`
- `tool_status`
- `tool_output`
- `references`
- `produces`
- `schema_version`
- `label_status`
- `enrichment`

These fields are used for validation, replay, SFT export, provenance slicing, and analysis.
They should not be copied into model tool arguments.

## Recovery Data Rule

Second-stage recovery data must keep this same protocol.

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
