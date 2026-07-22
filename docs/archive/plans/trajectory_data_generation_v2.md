# Trajectory Data Generation v2 Strategy

Status: implementation handoff for Claude. Read this document before changing the compiler,
emitter, protocol, rollout environment, or SFT data. Do not start a new SFT run until all blocking
acceptance checks in section 8 pass.

The V2a memory repair and V2-ctx context layer described here have now been implemented and
evaluated. The next data iteration focuses on observation-guided planning and recovery; read
`draft/reflection_trajectory_data_plan.md` before generating correction or reflection trajectories.

## 1. Objective

Generate execution-verified trajectories that support both:

1. SFT learning of legal and meaningful table-tool use.
2. RL process credit computed from harness-authored provenance rather than model-authored claims.

The protocol should remain deterministic and strict. Diversity should come from tasks, operation
chains, equivalent valid plans, memory reuse patterns, and verified model rollouts, not from making
tool schemas inconsistent.

## 2. Current v2 Audit

The current v2 files contain 6,773 train and 998 dev trajectories. Their offline structure is
mostly sound:

- all trajectories pass the current emitter validator;
- references point to earlier steps;
- the backward slice reaches the full compiled answer path;
- 969/969 non-memory dev trajectories replay and score correctly;
- the SFT builder produces 6,771 train and 998 dev records.

Two issues block retraining:

1. All 29 memory dev trajectories fail online replay because model-visible
   `add_to_memory.arguments` contains `references`, while the protocol/executor accepts only
   `key`, `value`, and `content`.
2. The model currently supplies `value` and provenance claims. During RL it could submit a false
   value or false reference and receive process credit unless the harness independently derives
   and validates them.

There are also 36 weakly described single-row memories (30 train, 6 dev) with:

```json
{"key": "value_subquery", "content": "single-row value from subquery"}
```

These arise from non-aggregate scalar subqueries such as:

```text
extreme_value_select -> project -> scalar memory
group_aggregate -> extreme_value_select -> project -> scalar memory
```

## 3. Canonical Memory Model

The scalar protocol below is the first required implementation, but it is not the whole memory
design. Preserve non-scalar working memory through explicit types and trust levels.

### 3.1 Grounded scalar memory

The model should identify an already computed scalar result. It must not author the scalar value or
claim its provenance.

Recommended model-visible call:

```json
{
  "tool": "add_to_memory",
  "arguments": {
    "key": "accelerate_at_max_horsepower",
    "source_step_id": "step_2"
  }
}
```

Optional later extension:

```json
{
  "key": "accelerate_at_max_horsepower",
  "source_step_id": "step_2",
  "extract": {"mode": "single_cell", "column": "Accelerate"}
}
```

For the first implementation, require the referenced output to be unambiguously scalar: either a
scalar tool result or a one-row, one-column table. Reject zero-row, multi-row, or multi-column
sources rather than guessing.

The harness returns:

```json
{
  "memory": {
    "key": "accelerate_at_max_horsepower",
    "value": 19.0,
    "content": "Accelerate of the row with maximum Horsepower in CARS_DATA",
    "source_step_id": "step_2",
    "derivation": {
      "type": "argmax_lookup",
      "source_table": "CARS_DATA",
      "selected_column": "Accelerate",
      "selector": {"column": "Horsepower", "direction": "max"},
      "supporting_step_ids": ["step_1", "step_2"]
    }
  }
}
```

Ownership:

- model: `key`, `source_step_id`;
- harness: `value`, `content`, `derivation`, validated references;
- tool history: normalized call, output, dependencies, and extraction result.

### 3.2 Grounded non-scalar result memory

If a conclusion refers to multiple rows, a list, a group result, or a join result, do not copy the
payload into memory. The source table remains the authority. Store a semantic pointer:

```json
{
  "tool": "add_to_memory",
  "arguments": {
    "items": [{
      "type": "evidence_pointer",
      "key": "high_value_customers",
      "source_step_id": "step_4",
      "intent": "customers whose total order value exceeds the threshold"
    }]
  }
}
```

The harness returns a grounded item containing the real table handle, columns, row scope/count,
source steps, and normalized derivation. This supports later operations without copying thousands
of values into the prompt.

### 3.3 Plan memory

A plan is useful non-scalar memory, but it is not a factual conclusion:

```json
{
  "tool": "add_to_memory",
  "arguments": {
    "items": [{
      "type": "plan",
      "key": "task_plan",
      "content": {
        "goals": [
          {"id": "g1", "intent": "compute average age", "status": "pending"},
          {"id": "g2", "intent": "filter singers above that value", "status": "pending"},
          {"id": "g3", "intent": "project song names", "status": "pending"}
        ]
      }
    }]
  }
}
```

The harness marks this `model_authored`. It may be updated as goals complete, but:

- it cannot be used through `value_ref`;
- it cannot support a final answer as factual evidence;
- it does not receive provenance credit merely for being written;
- it can receive separate consistency/completion rewards.

### 3.4 Hypotheses and interpretations

Tentative claims should use `hypothesis` with `model_authored_unverified` authority. They become
grounded only after a later tool result explicitly validates them. Hypothesis data is not required
for the first SFT/RL smoke test and should be deferred rather than filled with synthetic certainty.

### 3.5 How to construct non-scalar memory data

An external LLM is optional, not mandatory:

| Memory type | Primary construction source | External LLM role |
| --- | --- | --- |
| `derived_value` | Compiler + trusted tool history | Optional wording only |
| `evidence_pointer` | Compiler dependencies and produced `data_view` | Optional intent paraphrase |
| `plan` | Remaining gold Plan converted to abstract goals; verified rollout plans | Optional alternate wording/plans |
| `hypothesis` | Model rollouts or multi-turn datasets, then validation | Useful candidate generator |

For plan SFT, derive the structure programmatically from the gold Plan. Keep it abstract: record
subgoals and dependencies, not gold answer values. To avoid teaching one rigid planning template:

- include plans only in a controlled subset of trajectories;
- vary plan granularity deterministically;
- include partial plans and later extensions;
- obtain alternate valid plans from execution-verified model rollouts;
- train plan revision primarily from SParC/CoSQL or verified failed-to-corrected rollouts.

The first repaired Spider SFT may implement only `derived_value` and `evidence_pointer`. Add plan
memory as a separately measured data mixture so its effect can be ablated.

## 4. Semantic Derivation

Do not infer semantics from free-form text and do not ask an external LLM to produce authoritative
memory facts. Build a structured derivation from the normalized operation graph, then render a
compact model-readable sentence.

Initial deterministic derivation types:

| Operation graph | Type |
| --- | --- |
| `aggregate` | `aggregate_stat` |
| `condition_filter -> aggregate` | `filtered_aggregate` |
| `extreme_value_select -> project` | `argmax_lookup` / `argmin_lookup` |
| `group_aggregate -> extreme_value_select -> project` | `group_argmax` / `group_argmin` |
| `condition_filter -> project`, exactly one cell | `filtered_lookup` |
| `derive_column -> aggregate` | `derived_aggregate` |

Unknown valid graphs should use a structured fallback:

```json
{
  "type": "operation_result",
  "selected_column": "...",
  "operations": [
    {"step_id": "step_1", "tool": "...", "arguments": {...}},
    {"step_id": "step_2", "tool": "...", "arguments": {...}}
  ]
}
```

The fallback must still generate a semantic key from the selected column and the most informative
selector, never the opaque `value_subquery`.

Natural-language `content` may use deterministic templates. It is a rendering of `derivation`, not
the source of truth. Keep the structured object for validation and RL.

Suggested shared module:

```text
src/harness/memory_semantics.py
```

It should provide:

```python
extract_scalar(tool_history, source_step_id)
build_derivation(tool_history, source_step_id)
build_memory_key(derivation)
render_memory_content(derivation)
```

The emitter and live rollout must call the same functions.

## 5. Online Provenance

The live environment must assign a stable `step_id` to every successfully executed action and keep
an internal tool history. The model does not emit top-level trajectory `references` or `produces`.

For each action, the harness derives:

- referenced source tables;
- referenced earlier result tables;
- referenced memory entries;
- produced table/scalar/memory handle;
- normalized arguments;
- execution status and compact output summary.

When `add_to_memory` references a source step, the harness:

1. verifies that the step exists and is earlier;
2. extracts its real scalar result;
3. constructs the derivation from the trusted history;
4. writes the memory item;
5. records the dependency edge automatically.

At the terminal answer, the harness constructs the backward slice from the cited evidence table and
supporting memory IDs. Process rewards must use this harness graph, never model-provided references.

## 6. Data Diversity

Protocol stability is desirable. Avoid randomizing tool field names or derivation schemas.
Diversify the decision situations instead.

### 6.1 Programmatic Sources

Generate execution-equivalent variants and retain them only when they reproduce the gold result:

- split or merge compatible filter predicates;
- reorder independent filters and projections when semantics are preserved;
- vary valid join order;
- use memory only when a scalar must survive across later actions;
- reuse one memory in multiple later predicates;
- create trajectories with two independent memory entries;
- vary the distance between memory write and use;
- include aggregate, filtered aggregate, argmax lookup, group argmax, and derived aggregate memory;
- generate `evidence_pointer` items for non-scalar intermediate results that are reused later;
- compile optional abstract plan memory from the remaining gold Plan without embedding answer values;
- retain multiple correct plans instead of enforcing one canonical path.

Do not add no-op steps to positive SFT trajectories merely for variety. Legal but wasteful calls are
useful as negative RL/DPO examples.

### 6.2 Dataset Sources

- Spider: protocol repair and first RL smoke test.
- BIRD Mini-Dev: cross-database compiler/tool coverage evaluation; do not train on Mini-Dev.
- BIRD filtered train: later source of harder joins, expressions, dirty values, and external
  evidence after the Mini-Dev adapter is validated.
- SParC/CoSQL: later source of multi-turn memory reuse and refinement.

### 6.3 Verified Model Rollouts

After the repaired SFT model exists, sample multiple trajectories per train question. Execute every
action and retain:

- correct, legal, deduplicated paths as alternate positive trajectories;
- wrong or wasteful paths as RL experience or preference pairs;
- multiple distinct correct paths for studying lenient provenance credit.

An external LLM is optional for question paraphrases or candidate plans. Its outputs must pass
schema validation, execution verification, final-answer verification, and provenance validation.

## 7. Implementation Order

1. Add schema/protocol versioning to trajectories and SFT manifests.
2. Implement shared memory semantics and scalar extraction.
3. Change compiler Plan memory steps to carry source Plan IDs and derivation metadata.
4. Change emitter model-visible memory arguments to `key + source_step_id`.
5. Update `protocol.py` tool specification and strict argument validation.
6. Add online step IDs/tool history/provenance construction to `rollout.py`.
7. Make the SFT builder invoke the complete trajectory validator.
8. Regenerate Spider v2 and splice/rebuild grounded think only after structural equivalence checks.
9. Run full replay and build exact-token-filtered SFT data.
10. Retrain, evaluate, then implement the minimal RL trainer bridge.

Do not patch the executor to silently ignore unknown `references`; that would hide SFT/rollout
protocol drift without fixing provenance ownership.

## 8. Acceptance Gates

Before SFT:

- no model-visible `add_to_memory.value`;
- no model-authored provenance sidecar;
- every memory item declares `type` and harness-assigned `authority`;
- non-scalar grounded memory points to `data_view` instead of copying result rows;
- `plan`/`hypothesis` memory cannot be used as `value_ref` or factual answer evidence;
- no `value_subquery` key or `single-row value from subquery` content;
- every memory source resolves to a real earlier scalar;
- all trajectory tool arguments pass strict per-tool schemas;
- all train/dev trajectories pass structural and reference validation;
- Spider dev v2 replay: 998/998 execution and score success;
- SFT and rollout use the same protocol hash/version;
- Qwen tokenizer audit is used for cutoff filtering.

Before process-reward RL:

- live rollouts record harness-authored `references` and `produces`;
- terminal backward slices can be recomputed from recorded history;
- forged values, future references, missing steps, and ambiguous scalar sources are rejected;
- terminal-only reward works first;
- process reward is introduced only after trusted online provenance passes adversarial tests.

## 9. BIRD Mini-Dev Handoff

Official resources are downloaded under:

```text
data/bird_mini_dev/
```

Canonical local entry points:

```text
data/bird_mini_dev/mini_dev_sqlite.json  # latest HF SQLite annotations, 500 examples
data/bird_mini_dev/databases/            # 11 SQLite databases
data/bird_mini_dev/dev_tables.json       # schema metadata
```

Download provenance:

- GitHub repository commit: `b3d4bcbbae9a96934ad812551eb400c7a3b23c12`;
- official `minidev.zip`: 800,943,648 bytes;
- ZIP MD5 / official ETag: `7beb6dab11e65f0fde563e644a1ea319`;
- ZIP SHA-256: `cc48ba16838204e4e214512030cb572eeb5f7bcdd999bae4b9b6ff12ec13b92f`;
- all 11 SQLite databases pass `PRAGMA integrity_check`;
- license declared by the official repository: CC BY-SA 4.0.

The older annotation bundled in the ZIP and the current Hugging Face annotation both contain 500
examples over the same 11 databases. Only 496 `(db_id, question, SQL)` triples are identical, so
use the Hugging Face annotation through the canonical symlink above and retain the ZIP copy only
for provenance.

Use the 500-example SQLite Mini-Dev first. Keep it evaluation-only. Add a BIRD adapter that maps:

- `db_id`;
- `question`;
- `SQL`;
- optional expert `evidence`;
- database path and schema descriptions.

Report compile, execution-verification, and tool coverage separately. Unsupported BIRD SQL should
reduce coverage, never be approximated into a wrong trajectory.
