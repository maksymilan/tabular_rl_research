# Manual teacher agent prompt

Copy everything between `BEGIN PROMPT` and `END PROMPT` into a fresh Codex agent. This prompt is
for a diagnostic hard-task pilot. It does not authorize adding generated trajectories to an SFT
mixture while the active protocol's frozen scale gate remains unpromoted.

## BEGIN PROMPT

You are the external teacher agent for `/Users/hudou/Research/tabular_rl_research`. Your job is to
solve BIRD-train questions by interacting causally with the repository's real table-tool harness
and to retain only verifier-correct, replay-correct, quality-gated trajectories.

Read `/Users/hudou/Research/tabular_rl_research/AGENTS.md` and
`/Users/hudou/Research/tabular_rl_research/docs/current/README.md` before acting. Preserve all
existing worktree changes. Work only inside the existing repository. Do not commit, train a model,
launch a remote job, or merge any output into an SFT dataset.

### Scope and stopping rule

- Run a diagnostic pilot on up to 20 previously uncovered tasks whose
  `metadata.difficulty_proxy` is `hard`.
- A task gets one semantic attempt. Protocol/execution errors may be repaired within the same live
  episode from `LAST TOOL ERROR`; a verifier-correctness failure is terminal and must not be used
  to search for the hidden label.
- Stop early and report if five consecutive selected tasks fail semantically, if the environment
  itself is broken, or if the active shared contract says the protocol is ineligible for this
  diagnostic.
- Every artifact from this prompt remains
  `diagnostic_only_pending_protocol_scale_gate`, even if its local replay and quality gates pass.

### Non-negotiable information boundary

- Never read, print, search, summarize, or infer from `gold_sql`, `query`, `gold_exec_results`,
  `gold_sql_path`, or any gold answer rows.
- Never run SQLite, direct SQL, a database browser, or an ad-hoc script against a task database.
- Never inspect another model's successful trajectory for the selected example.
- Never turn gold SQL into a tool trajectory.
- Select candidates only from safe metadata: `example_id`, `db_id`,
  `metadata.difficulty_proxy`, `question`, and `external_knowledge`.
- During an episode, reason only from each emitted `model_input`: its system contract, dataset
  overview, question, optional external knowledge, rolling legal history, current environment
  state, and latest tool error.
- Gold may exist internally in the session process only for terminal `bird-set` verification and
  deterministic replay. It must not be visible to you.

### Candidate selection

From the repository root, use this exact read-only command to list uncovered hard candidates
without exposing gold fields:

```bash
jq -n \
  --slurpfile tasks data/eval_inputs/bird_train_batch2_student_rollout_disjoint3000.jsonl \
  --slurpfile accepted data/trajectories/batch2_cp560_rollout3000_20260726/final_repeated_only_v1/verified_trajectories.jsonl \
  '($accepted | map(select((.source|type)=="object") | .source.example_id) | unique) as $done
   | [$tasks[]
      | select(.metadata.difficulty_proxy=="hard")
      | select(.example_id as $id | ($done | index($id))==null)
      | {example_id,db_id,difficulty_proxy:.metadata.difficulty_proxy,question,external_knowledge}]'
```

Do not replace that projection with `jq .`, `head`, `sed`, Python loading/printing, or any command
that could expose the hidden task fields. Before selecting an example, also exclude any
`example_id` for which this pilot directory already contains either a verified or audit artifact.

### Start one live episode

Set `EXAMPLE_ID` to one selected ID and give every attempt unique output paths. Do not overwrite or
delete existing artifacts.

```bash
EXAMPLE_ID=replace_with_one_selected_uncovered_id
PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python -u src/sft/manual_teacher_session.py \
  --tasks-json data/eval_inputs/bird_train_batch2_student_rollout_disjoint3000.jsonl \
  --example-id "$EXAMPLE_ID" \
  --trajectory-out "data/trajectories/manual_teacher_pilot/${EXAMPLE_ID}.verified.jsonl" \
  --audit-out "data/trajectories/manual_teacher_pilot/${EXAMPLE_ID}.audit.json" \
  --max-steps 30 \
  --max-errors-per-type 3 \
  --max-think-words 300
```

Run it in an interactive terminal and retain its session ID. The process prints `session_started`
and then one `model_input`. The script itself supplies the exact current full teacher prompt,
public tool contract, rolling-history policy, and current environment state; do not paste a stale
copy of the model-facing tool prompt from a report.

### Respond one action at a time

For every `model_input`, submit exactly one JSON line whose only transport key is `assistant`:

```json
{"assistant":"<think>One non-empty, state-specific reason for the next action.</think>{\"tool\":\"describe_table\",\"arguments\":{\"tables\":[\"customer\"]}}"}
```

The value of `assistant` must contain exactly:

1. one non-empty `<think>...</think>` block; then
2. one raw JSON object with exactly the top-level keys `tool` and `arguments`.

Do not emit Markdown, `<tool_call>` tags, multiple actions, future steps, an authored final answer,
or text after the action object. Wait for the next `model_input` before choosing another action.

Follow the tool schema printed in the current system message exactly. In particular:

- treat explicit mappings in the question and EXTERNAL KNOWLEDGE as binding; do not replace an ID,
  field, aggregation, formula, or output slot with a more natural proxy;
- before filtering, joining, aggregating, or ranking, identify the requested answer unit, current
  row grain, eligible population, and exact output columns from visible evidence;
- do not invent earliest/latest/current/active/top-1/mean/same-year restrictions to force a
  singleton; retain all rows satisfying the stated conditions unless a grounded selector exists;
- distinguish row count, entity count, COUNT, and COUNT DISTINCT, and keep numerator and denominator
  on the same population and grain unless the specification says otherwise;
- treat empty joins, unexpected multiplicity, NULLs, impossible dates, or implausible arithmetic as
  reasons to inspect assumptions rather than reasons to switch targets or accept a plausible value;
- call `describe_table` before using unresolved columns;
- use `inspect_column` to ground uncertain filter literals;
- after a join, use the exact flat `relation.column` names shown by the environment;
- in `join_tables`, `left` is an already-introduced exact logical column and `right` is the bare
  column of the newly attached table;
- cite only producing step IDs in `value_ref`;
- use `read_subtable` when seeing rows is necessary; use its typed conditions for row lookup and
  ordered offsets for deterministic pagination, and never repeat identical arguments adjacently;
- use typed `project` date expressions for row-wise `date_diff_days(start,end)` or
  `extract_year(date)` instead of authoring SQLite date syntax;
- derive the exact requested row set, column set, column order, and field representation before
  terminating; remove helper counts, ranking keys, and join identifiers not requested by the
  question, and keep separate source fields separate unless formatting is explicit;
- terminate only with
  `{"tool":"answer_from_context","arguments":{"evidence":{"table":"HANDLE"},"reason":"..."}}`.

If an action is rejected, use only the new `LAST TOOL ERROR` and unchanged environment state to
repair the next action. The rejected text is audit-only and must never become an SFT target. Do not
repeat the identical action.

### Accept or reject the episode

At terminal completion, accept the trajectory only if the emitted summary contains all four facts:

```text
correct=true
replay_passed=true
quality_gate_passed=true
trajectory_written=true
```

If `correct=false`, retain only the audit JSON, do not alter it, do not inspect gold, do not retry
the same example with a different interpretation, and move to a new candidate. If replay or
quality fails, retain the audit, report the failure reason, and do not hand-edit the trajectory.

For each written trajectory, run the independent structural audit:

```bash
PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python src/sft/audit_verified_rollouts.py \
  --input "data/trajectories/manual_teacher_pilot/${EXAMPLE_ID}.verified.jsonl" \
  --out "data/trajectories/manual_teacher_pilot/${EXAMPLE_ID}.structural_audit.json" \
  --required-prompt-variant full
```

The episode is structurally retained only when both `structural_gate` and `prompt_variant_gate`
are `pass`.

### Pilot-level verification and report

Do not concatenate verified files into an SFT dataset. Instead, report:

- selected, attempted, terminal-correct, replay-passed, quality-passed, and structurally passed
  episode counts;
- clean success versus recovered success;
- error-event counts by type;
- task IDs and artifact paths for retained diagnostic trajectories;
- each failed task's public failure type, without gold or hidden-answer details;
- the active `PROTOCOL_VERSION`, teacher/student prompt hashes, tool-schema hash, history length,
  and denotation metric from the artifacts;
- an explicit final sentence that the pilot is diagnostic-only and was not admitted to training.

Before finishing, run:

```bash
PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python -m py_compile src/sft/manual_teacher_session.py
```

Never claim success from terminal correctness alone. The minimum retained unit is:
hidden `bird-set` terminal success + fresh deterministic replay + local quality gate + independent
structural gate.

## END PROMPT
