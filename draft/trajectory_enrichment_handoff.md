# Trajectory enrichment handoff for Claude

Date: 2026-06-24

This note records the current shared understanding between the user and Codex about observation
trajectory enrichment. Read this before changing `src/sft/enrich_traj.py`, regenerating data, or
starting another SFT run from enriched trajectories.

## Goal

The goal is not merely to produce trajectories that pass a hard-coded validator. The goal is to
teach a table-tool agent how to work under a large-database / limited-context setting:

1. Start from a lazy catalog: table names, row counts, and foreign-key relations only.
2. Notice what information is missing before acting.
3. Use observation tools to acquire global schema first, then local values or intermediate rows.
4. Execute relational actions only after their information preconditions are grounded.
5. Explain each decision in first-person task reasoning, so a human can understand why that tool and
   those columns were selected.

The final SFT data should make the agent behave like it is actively reading and verifying a table,
not like it is replaying a SQL plan with decorative observations inserted.

## Current data under review

The most recent Claude-generated 40-example probe is:

`data/trajectories/probe40_v3_gated.jsonl`

Codex audit summary:

- 40 total trajectories.
- 39 are marked `enriched`.
- 1 is `fallback_skeleton`: `spider_train_4563`.
- Current hard gates mostly pass, but that is not enough. Many accepted examples still contain
  reasoning that violates the catalog-only state or overstates what an observation proved.

Do not interpret "validator passed" as "ready for SFT".

## Main problems observed

### 1. Schema leakage before `describe_table`

Many first-step `describe_table` thoughts name exact columns before the simulated agent has observed
the schema.

Examples:

- `spider_train_10`: says the management table may contain `temporary_acting` before describing it.
- `spider_train_4753`: says describing `Customer_Orders` will reveal `order_status_code` and
  `order_date`.
- `spider_train_3450`: mentions `COMMISSION_PCT` and `DEPARTMENT_ID` before schema observation.
- `spider_train_2674`: says the catalog includes `Nationality`, which is false under the lazy
  catalog assumption.

Allowed before `describe_table`:

- Natural question concepts: "order status", "order date", "commission information",
  "department identifier", "host nationality".
- Table names and FK endpoints visible in the lazy catalog.

Not allowed before `describe_table`:

- Exact hidden schema identifiers such as `order_status_code`, `order_date`, `COMMISSION_PCT`,
  `temporary_acting`, `Nationality`, unless they are actually visible in the lazy catalog.

After `describe_table`, exact column names are required and desirable.

### 2. Observation evidence is sometimes overstated

Some trajectories say an observation confirmed a value even when the returned observation did not
show that value.

Example:

- `spider_train_3293`: `inspect_column(DEPARTMENT.DEPT_NAME)` returns frequent values with
  `truncated: true`, and the visible list does not contain `Accounting`. The later think says the
  model confirmed `Accounting` exists. That is not grounded by the displayed observation.

Rule:

- If `inspect_column` returns the literal in `frequent_values`, the model may say it confirmed the
  literal.
- If `truncated: true` and the literal is not visible, the model may only say the column is the right
  semantic field; it must not claim the literal was confirmed.
- If a literal is required but not visible in the observation, either the observation tool must be
  extended to support targeted lookup, or the generated reasoning must acknowledge the limitation.

### 3. Some observations are still decorative or missing

The desired pattern is not "always read before every action"; it is "observe when a precondition is
unresolved." Still, some generated examples skip useful verification or fall back to the old skeleton.

Examples:

- `spider_train_4563` is a fallback skeleton with no observations and templated thoughts. It should
  not be used as enriched SFT data.
- Some simple sort/project tasks do not use `read_subtable`. That can be acceptable if the task is
  simple and the final table-producing operation is direct. Do not force `read_subtable` everywhere.
- For multi-hop joins, filters, set operations, or ambiguous intermediate results, `read_subtable`
  should usually appear before the next dependent action or final answer.

### 4. Reasoning sometimes references wrong step numbers

Examples:

- `spider_train_104`: says "from step 0" after the inserted observation shifts step numbering.
- `spider_train_4753`: says "minimum order_date from step 1" when the aggregate producing the value is
  a later step.

Rule:

- Do not let the LLM freely author exact step references unless they are remapped after insertion.
- If step references are used in think text, they must match the final rendered step ids.
- Prefer semantic references such as "the previous aggregate result" unless the generation pipeline
  can reliably rewrite step ids after splicing.

### 5. Validator passing is currently too weak as a quality signal

The current validator has been relaxed to reduce false rejections, which is good. However, relaxed
hard gates mean more responsibility must move to human-readable audit fields and soft warnings.

Do not optimize only for hard-gate acceptance. The quality bar is:

- executable and gold-correct;
- no state-visibility violation;
- first-person reasoning;
- clear semantic bridge from question cue -> observed evidence -> chosen tool/column -> next action;
- no unsupported claim about what an observation proved.

## Desired trajectory pattern

A good trajectory should look like this:

1. **Catalog-level orientation**
   - "The question asks for customers with cancelled orders. `Customer_Orders` is the likely source
     because it is the order-level table; I need its schema to find the exact status and date fields."
   - No hidden exact column names before schema observation.

2. **Schema acquisition**
   - `describe_table({"tables": ["Customer_Orders"]})`
   - Output reveals `order_status_code`, `order_date`, `customer_id`.

3. **Local value grounding when needed**
   - `inspect_column({"table": "Customer_Orders", "column": "order_status_code"})`
   - If `Cancelled` is visible, the think may say the literal is confirmed.

4. **Relational action**
   - `condition_filter` using `order_status_code = "Cancelled"`.
   - Think explains that this keeps cancelled orders because the schema and observed values support
     the predicate.

5. **Intermediate result verification when downstream action depends on it**
   - `read_subtable(filter_...)` before aggregate, join, set operation, or final answer when the
     intermediate cardinality/content matters.

6. **Final answer**
   - `answer_from_context` cites the actual evidence table or scalar-producing step.

## Acceptance categories

Use three categories instead of a single pass/fail label:

- `ready`: execution-verified, state-faithful, clear reasoning, no unsupported observation claims.
- `repairable`: executable but has reasoning issues such as pre-schema identifier leakage, stale step
  references, or unsupported "confirmed value" language.
- `reject`: fallback skeleton, execution failure, wrong final answer, illegal tool call, or major
  state-visibility violation that cannot be safely repaired.

The 40-example probe should be treated roughly as:

- `ready`: many simple count/sort/filter cases, after a light manual/automatic check.
- `repairable`: cases with pre-schema exact column names but otherwise good tool paths.
- `reject`: `spider_train_4563` fallback skeleton.

## Implementation guidance

### Prompt

The prompt must define the model identity and state boundary explicitly:

- The LLM is writing the table-tool agent's own first-person reasoning.
- The agent initially sees only the lazy catalog.
- Exact column identifiers are forbidden before schema observation unless visible in FK relations.
- After schema observation, exact identifiers are required.
- Observation claims must match the actual returned observation.

Avoid prompt wording that asks the LLM to "explain the fixed backbone" using hidden SQL/schema. That
causes it to leak future columns into earlier thoughts.

### Validator

Keep hard gates for:

- execution/gold correctness;
- source-table actions before `describe_table`;
- text literal filters before value/domain grounding when required;
- third-person/meta narration;
- fallback skeletons in an enriched dataset.

Track soft warnings for:

- lack of first-person wording;
- missing exact column names after schema observation;
- weak semantic bridge;
- missing or incomplete rationale;
- pre-schema exact identifier mentions that may be repairable.

Add a specific warning/error for unsupported confirmation:

- If a think says "confirmed", "verified", or "exists" for a literal, that literal must appear in the
  prior observation output or the observation must explicitly report a successful targeted lookup.

### Repair strategy

Prefer repairing reasoning over discarding executable trajectories:

- Rewrite pre-`describe_table` exact identifiers into natural concepts.
- Rewrite stale step references after perception insertion.
- Downgrade "confirmed literal exists" to "identified the relevant column" when the literal is not
  visible due to truncation.
- Remove fallback skeletons from enriched SFT data.

Do not repair by adding fake evidence. The harness owns observations; the LLM cannot invent values,
rows, step ids, or provenance.

## Next recommended action

Before generating more data:

1. Fix the `quality_check` call sites so the code consistently treats its return value as
   `(hard_issues, soft_issues)`.
2. Add a repair pass for pre-schema exact identifier leakage in `describe_table` thoughts.
3. Add the unsupported-literal-confirmation check.
4. Re-run a 40-example probe and produce a per-example quality manifest with `ready` /
   `repairable` / `reject`.
5. Only then scale to larger SFT data.

## Implemented after this review

The enrichment code now has a stricter post-generation audit layer:

- `quality_check` treats stale observation wording as `repairable`: a non-observation action should
  not say "I should inspect/read first" after the observation should already be in the transcript.
- `quality_check` treats annotator-style final answer wording as `repairable`: final thoughts should
  cite the evidence table/result, not say "the verified/expected answer is ...".
- `src/sft/enrich_traj.py --audit-file <jsonl>` recomputes quality for an existing enriched file
  without calling the external LLM and writes a per-example quality manifest.
- Normal generation also writes the same quality manifest next to the output by default.
- The manifest records computed `quality_status`, stored `quality_status`, hard/repairable/style
  issues, generator model, tool sequence, and a recommended action (`use_for_sft`,
  `repair_then_recheck`, or `drop_or_manual_review`).

Current audit of `data/trajectories/probe40_v6_gated.jsonl` under the stricter rules:

- `ready`: 22
- `repairable`: 17
- `reject`: 1
- repairable issue counts: stale observation wording 18, final answer leak 10, unsupported
  confirmation 1.

This confirms the v6 prompt greatly reduced schema leaks, but the next bottleneck is temporal
wording and final-answer annotation style.

Follow-up probe `data/trajectories/probe40_v7_gated.jsonl` was regenerated with the stricter prompt
and quality gate:

- `ready`: 20 under the newer audit rules
- `repairable`: 19 under the newer audit rules
- `reject`: 1
- repairable issue counts: stale observation wording 17, tool-action mismatch 17, missing action
  reference 38, missing action column 11, final answer leak 2.

Interpretation: prompt changes alone only slightly improved readiness. Schema leakage and unsupported
confirmation are largely solved, but stale action wording and shifted action explanations remain.

Implemented next:

- `quality_check` now treats tool/action mismatch as `repairable`, with emphasis on the actual
  `think` text rather than allowing a correct structured rationale to hide a bad action sentence.
- Non-observation actions must name the exact action columns in the `think` text. Missing action
  columns are `repairable`, not merely style.
- Non-observation actions must name their key table/reference handles in the `think` text. For
  `join_tables`, both left and right inputs are required; this catches shifted long-chain joins such
  as a step whose actual call is `join_009` with `department` but whose think still describes
  `join_008` with `course`.
- `--repair-file` performs a deterministic text-only repair pass. It changes only `think` and
  `rationale`; it never changes tool calls, tool outputs, answers, step ids, provenance, or
  observations.
- New generation now runs this same repair pass automatically before writing each trajectory. Use
  `--no-auto-repair` only when intentionally auditing the raw external-LLM output; the raw model
  responses are still preserved in `annotation_history`.

Current repaired output:

- input: `data/trajectories/probe40_v7_gated.jsonl`
- output: `data/trajectories/probe40_v7_gated_repaired.jsonl`
- manifest: `data/trajectories/probe40_v7_gated_repaired.quality_manifest.json`
- repaired steps: 58
- final computed quality: `ready` 39, `reject` 1, `repairable` 0
- repairable issue counts: none
- soft issue count: 0

Manual spot check: `spider_train_3293` previously had shifted text in the Accounting / Computer Info.
Systems path (`project` described as filtering; later joins described previous joins). After repair,
the affected steps explicitly match their actual calls, e.g. `join_007` with `class`,
`join_008` with `course`, and `join_009` with `department`.

Do not interpret the deterministic repaired text as the ideal final training style. It is a safe
fallback for mechanical defects in otherwise execution-verified trajectories. For large-scale
generation, prefer improving the upstream LLM prompt and use this repair pass as an export gate and
cleanup layer.

## V8 pilot generation result

Command used:

```bash
.venv/bin/python src/sft/enrich_traj.py \
  --which subset \
  --mode staged_perception \
  --model deepseek-v4-flash \
  --out data/trajectories/subset180_v8_pilot.jsonl \
  --api-timeout 240 \
  --api-retries 2 \
  --max-attempts 5
```

Result:

- source subset: 180 trajectories (`data/trajectories/subset_180.ids.json`, key `subset`)
- enriched: 178
- fallback/reject: 2 (`spider_train_2854`, `spider_train_4564`)
- computed quality after auto-repair: `ready` 178, `reject` 2, `repairable` 0
- no external API errors recorded in `annotation_history`
- ready-only trajectory file:
  `data/trajectories/subset180_v8_pilot_ready.jsonl`
- SFT train file:
  `data/sft/spider_v8_pilot_ready_train.jsonl`
- LLaMA-Factory dataset name:
  `spider_tools_v8_pilot_ready`

SFT build stats from the ready-only file:

- source/kept: 178 / 178
- dropped_overlong: 0
- dropped_low_quality: 0
- estimated tokens p50 / p95 / max: 2475 / 3786 / 6001
- tool histogram: `describe_table` 318, `read_subtable` 108, `inspect_column` 62,
  `join_tables` 88, `condition_filter` 93, `project` 167, `group_aggregate` 65,
  `aggregate` 33, `extreme_value_select` 66, `set_op` 11, `answer_from_context` 178.

Manual spot checks:

- `spider_train_3293` is the hardest sample in this pilot: 21 steps, 7 perception steps, 11 repaired
  reasoning steps. After repair, the long join chain explicitly names the correct handles/tables
  (`join_001` -> `class`, `join_002` -> `course`, etc.) and no longer has the earlier shifted-action
  issue.
- Long/high-repair samples are usable as a pilot, but deterministic repair introduces more templated
  action text. Treat this pilot as a seed set for behavior cloning and failure-driven iteration, not
  as the final high-diversity SFT corpus.

Recommended next experiment:

1. Fine-tune a small pilot adapter on `spider_tools_v8_pilot_ready`.
2. Roll out the adapter on the remaining Spider train trajectories, excluding the 178 ready pilot
   ids.
3. Save full model I/O, tool observations, and failure categories.
4. Use external LLM repair/correction only on real failed rollouts to construct the next correction
   dataset.
