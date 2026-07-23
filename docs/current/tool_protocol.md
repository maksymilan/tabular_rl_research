# Current Trajectory Protocol (version11)

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

Public version mapping:

- `version1`: the former `v2i-state-only-join-feedback-r2` contract;
- `version2`: canonical calls and exact final-answer shape;
- `version3`: precise DeepSeek transport feedback, stable projected join-column names, and a
  canonical multi-table join call;
- `version4`: provider carrier failures are explicitly counted against the
  protocol-error budget rather than argument-validation errors;
- `version5`: joins use `base + joins[]`, flat `relation.column` output names,
  and semantic roles only for repeated relations;
- `version6`: keeps the version5 action shape and groups wide dotted columns as
  `column_namespaces` in model-visible context without changing canonical harness snapshots;
- `version7`: makes filters, projection expressions, grouping, and ordering consume those logical
  columns consistently, with safe unique-bare resolution;
- `version8`: gives a provider exactly one API-facing response envelope and
  renders bounded assistant history in that same carrier, while retaining the canonical stored
  `<think>` + `<tool_call>` action;
- `version9`: explicitly enables DeepSeek thinking mode, fixes reasoning effort,
  and audits the request controls plus returned provider identity metadata;
- `version10`: constrains provider-visible actions with JSON Output and wraps the unchanged returned
  JSON in the internal canonical `<tool_call>` envelope;
- `version11`: current contract; removes the remaining generic visible-`<tool_call>` wording from
  DeepSeek-facing generation and retry instructions, leaving one raw-JSON visible contract;
- future changes increment only the integer (`version12`, `version13`, ...).

The current version11 tool set is the one in `src/sft/protocol.py::TOOL_SPECS`:

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

The full system prompt includes one concise canonical JSON call for each complex operation family:
filter, projection/computed column, join, grouped/scalar aggregation, set operation, table answer,
and scalar answer. These examples use only the current public fields.

For a row-valued terminal answer, the cited evidence table is scored as the answer. Its rows,
columns, and column order must match the requested output exactly. `read_subtable(columns=...)` only
limits observation and does not change table shape. If helper columns remain, the model must call
`project` before `answer_from_context(evidence={"table": ...}, answer=[])`. For scalar answers, use
`evidence=null` and `answer=[value]`; think/reason text cannot repair answer data.

`read_subtable.limit` is an integer in `1..20`. An out-of-range value is an explicit
`argument_validation_error`; the harness never clamps it. In bounded rolling mode, prior successful
actions retain compact result summaries while full factual payloads remain in resident state.

No `add_to_memory` tool exists in v2c-plan. No reflection or invalidate tool exists.

### Join Identifier Rule (version11; unchanged from version8)

`join_tables(base, joins, base_role?)` joins a connected component in one model action. Every
`joins[]` item attaches exactly one new table. Its `on.left` values are exact already-introduced
logical columns (`relation.column`); its `on.right` values are bare columns of the new table.
The default namespace is the visible source table or handle name. `base_role` and item `role` are
used only to disambiguate repeated relations such as employee/manager.

Join failures report both the referenced table schemas and this model-facing identifier rule. The
harness does not strip arbitrary SQL aliases or invent roles to rescue an invalid model action.
Output columns stay flat across the full component (`orders.id`, not `join_003.orders.id`), and
projection remains a separate tool action.

For context efficiency, a derived handle may render these flat names as
`column_namespaces={"orders":["id",...], ...}`. This is lossless model-visible compression:
`orders` remains the logical namespace for subsequent column arguments even when the table argument
is a handle such as `join_003`. The canonical resident state and replay record retain full names.
Downstream tools quote exact dotted names even inside scalar expressions. A bare suffix is accepted
only when it maps to one available logical column; joins themselves still require the exact
namespace-qualified left key.

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
