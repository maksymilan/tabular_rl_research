# Current Trajectory Protocol (v2i-state-only-join-feedback-r2 / v3)

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

Before every assistant turn, the model sees a state-only context rebuilt by
`src/sft/protocol.py::model_context_messages`:

```text
system: protocol/system prompt
user: DATASET OVERVIEW + QUESTION + optional EXTERNAL KNOWLEDGE
      + CURRENT ENVIRONMENT STATE + optional LAST TOOL ERROR
assistant: <think>...</think>
           <tool_call>{"tool": "...", "arguments": {...}}</tool_call>
```

The environment state is harness-managed resident context. It groups the current task plan and
known table context by source table or derived handle. It is NOT a historical tool-observation
transcript. Online rollout/RL constructs each model input as:

```
system + (catalog + question + optional evidence + current state + optional last error)
```

Old tool observations may exist in debug artifacts, but must never be appended to model input. A
single-step SFT record uses the same renderer and contains exactly one human/gpt pair; the human
message is the state before the supervised call and never includes that call's output.

Every new model turn is generated causally inside a real model↔harness episode. The model sees only
the legal prefix represented by current resident state and the latest environment feedback; it
chooses perception, relational, planning, and terminal actions itself. Gold SQL and future actions,
outputs, plans, or observations are never visible to the model or teacher. The retired
gold-SQL-compilation and complete-trajectory-enrichment pipelines live under `archive/` only.

The model emits only:

- `<think>` text;
- one `<tool_call>` JSON object.

The model does not emit provenance, references, produces, quality status, repair metadata, or reward
fields.

## Current Tool Set

The current v2i tool set is the one in `src/sft/protocol.py::TOOL_SPECS`:

- `condition_filter`
- `plan`
- `project`
- `join_tables`
- `group_aggregate`
- `extreme_value_select`
- `set_op`
- `describe_table`
- `inspect_column`
- `read_subtable`
- `answer_from_context`

`read_subtable.limit` is an integer in `1..20`. An out-of-range value is an explicit
`argument_validation_error`; the harness never clamps it. In bounded rolling mode, prior successful
actions retain compact result summaries while full factual payloads remain in resident state.

No `add_to_memory` tool exists in v2c-plan. No reflection or invalidate tool exists.

### Join Identifier Rule (v2i Compatibility)

`join_tables.on` uses model-facing column identifiers, not SQL implementation aliases. Never emit
`L.` / `R.` or `table.column` inside `on` edges. With `prefixes=[P1,P2,...]`, the first edge may use
`P1__column` on its left side even though the source SQL column is physically bare at that instant;
the harness maps this documented first-edge form exactly. The edge's right key is the bare column of
the next table. On later folds, left keys use the actual accumulated `prefix__column` names.

Join failures report both the referenced table schemas and this model-facing identifier rule. The
harness does not strip arbitrary aliases or invent prefixes to rescue an invalid model action.
The proposed v2j dotted-name migration in `canonical_execution_contract.md` is not active in v2i.

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

These fields are used for validation, replay, SFT export, provenance slicing, and analysis.
They should not be copied into model tool arguments.

## Recovery Data Rule

Recoverable model errors stay inside the same episode. The harness preserves the resident factual
state, spends one action from the shared `max_steps` budget, and supplies the next turn with the
normal state-only renderer plus structured `LAST TOOL ERROR`:

```json
{"step_id":"step_4","status":"error","error":{"type":"argument_validation_error","message":"..."}}
```

The bounded recoverable classes are `protocol_error`, `argument_validation_error`, and an
`execution_error` whose environment-state snapshot is unchanged. Each class has its own error
limit; a mutated-state execution failure is terminal as `nonrecoverable_execution_error`. API
transport retries are client-side requests, not semantic actions or recovery events.

Every rejected model action is retained in the full audit record as an `error_event`, with action
index and before/after state hashes. It is never included in `trajectory.steps` or exported as an
SFT target. The first later legal step carries `feedback_recovery: true` and its prior error type.
A verifier-correct episode is `clean_success` only when it had no error events; otherwise it is
`recovered_success`.

Strict recovery parsing accepts exactly one complete `<tool_call>` JSON object with a non-empty
`<think>` block and exact `tool`/`arguments` keys. Do not close tags, balance JSON, normalize
arguments, or otherwise repair model output. Whole-episode restarts, when deliberately enabled for
pass@k, are distinct attempts and must be reported as such.

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
