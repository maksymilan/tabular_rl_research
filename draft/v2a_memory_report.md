# V2a Memory & Step-Level Backward-Derivation — Report

Status: implemented + verified (2026-06-14). Scope: V2a = scalar `derived_value` memory + the
harness-authored provenance backbone. `evidence_pointer` (V2b) and `plan`/`hypothesis` (V2c) are
deferred to separate, separately-ablated data mixtures.

Verification (all local): Spider dev replay **998/998** execution+score · **30,534** tool calls pass
the strict per-tool schema with **0** violations · backward-slice invariant **7767/7767** (0
violations, 0 illegal) · unit tests 98 (harness) + 6 (sft) + 4 (eval) · real-DB execution-verified
**98.4%** · v1 trajectory/SFT files byte-identical (untouched).

Running example: `spider_train_160`.
- Question: *Which days had a minimum dew point smaller than any day in zip code 94107, and in which
  zip codes were those measurements taken?*
- Gold SQL: `SELECT date, zip_code FROM weather WHERE min_dew_point_f < (SELECT min(min_dew_point_f)
  FROM weather WHERE zip_code = 94107)`

---

## 1. How the memory tool is called — the trust boundary

Principle: **the model only names a source; it never authors the fact.** The model does not write the
value, key, or provenance — the harness derives all of them from the verified tool history.

Model-visible call (what actually appears in an assistant turn):
```json
{"tool":"add_to_memory","arguments":{"type":"derived_value","source_step_id":"step_2"}}
```
Two fields only: `type` (V2a: only `derived_value`) and `source_step_id` (a step, seen in an earlier
observation, whose output is a single scalar). Optional `alias` is a human-readable label only.

Harness-grounded entry returned in the observation:
```json
{"memory_id":"mem_step_2","key":"min_min_dew_point_f","value":5,
 "authority":"harness_grounded",
 "content":"min(min_dew_point_f) over rows where zip_code = 94107",
 "source_step_id":"step_2",
 "derivation":{"type":"filtered_aggregate","op":"min","column":"min_dew_point_f",
               "filter":"zip_code = 94107","source_table":"weather",
               "supporting_step_ids":["step_1","step_2"]}}
```

Field ownership:

| Field | Owner | Notes |
|---|---|---|
| `type`, `source_step_id`, optional `alias` | model | only "remember this step's result"; alias is read-only |
| `value` | harness | extracted from `source_step_id`'s real output; not model-forgeable |
| `key`, `content` | harness | deterministic from `derivation` (`build_memory_key` / `render_memory_content`) |
| `memory_id` | harness | stable identity = `mem_<source_step_id>` |
| `authority` | harness | `derived_value` is always `harness_grounded` |
| `derivation` | harness | structured semantics classified from the operation graph (for validation/RL) |

Strict scalar validation (`memory_semantics.extract_scalar`): the source must be a scalar tool output
or an exactly-1×1 table, non-NULL. Zero-row / multi-row / multi-column / NULL → `MemoryGroundingError`,
rejected and counted in the manifest (V2a dropped 4 `memory_reject_null_source` trajectories). This
guarantees a `derived_value` is always trustworthy.

Semantics come from the operation graph, not an LLM. `build_derivation` walks references back from the
source and classifies deterministically: `aggregate_stat` / `filtered_aggregate` / `argmax_lookup` /
`group_argmax` / `filtered_lookup` / `derived_aggregate`, with an `operation_result` fallback (still a
real column name). For the example, `condition_filter → aggregate` ⟹ `filtered_aggregate`, so
key=`min_min_dew_point_f`, content=`min(min_dew_point_f) over rows where zip_code = 94107`. The opaque
`v3` / `value_subquery` of v1 can no longer occur.

Use site: a predicate references the stable `memory_id` (not the semantic key):
```json
{"column":"min_dew_point_f","op":"<","value_ref":"mem_step_2"}
```
Execution resolves `mem_step_2` back to the real value 5; the trajectory display keeps `mem_step_2`.

---

## 2. How memory appears in the model's context

Memory shows up in 5 places across one dialogue (`[gpt]` = model output, loss-bearing;
`[observation]` = harness-written, masked):

1. **System prompt** tells the model the contract: *"You do NOT supply the value — the harness extracts
   it and returns {memory_id, key, value, …}. Use the returned memory_id as {value_ref: memory_id}."*
2. **The scalar-producing step + its observation** — the observation carries a `step_id`:
   ```
   [observation] {"step_id":"step_2","status":"success","output":{"result_sample":[[5]],"row_count":1}}
   ```
3. **The add_to_memory call** — the model copies `step_2` from the previous observation:
   ```
   [gpt]         <think>Register the scalar computed by the cited source step …</think>
                 <tool_call>{"tool":"add_to_memory","arguments":{"type":"derived_value","source_step_id":"step_2"}}</tool_call>
   [observation] {"step_id":"step_3","status":"success","output":{"memory":{"memory_id":"mem_step_2","key":"min_min_dew_point_f","value":5,"authority":"harness_grounded","content":"min(min_dew_point_f) over rows where…"}}}
   ```
   The model sees the real value 5 and the semantics for the first time **in this harness-written
   observation**, not in its own output.
4. **Use**: `condition_filter … "value_ref":"mem_step_2"`.
5. **Terminal answer**: `"supporting_memory_ids":["mem_step_2"]`.

Two points:
- The observation envelope is uniformly `{step_id, status, output}`, rendered by the single shared
  `protocol.tool_output_message` (same offline for SFT and online for rollout). The model learns to
  **copy** a `step_id` from an observation as the next `source_step_id`, never to predict ids; failed
  actions yield `{status:"error"}` and cannot be cited.
- Loss is computed only on `[gpt]` turns; the memory value/semantics live in `[observation]` (harness),
  which is masked — so the model learns "name the source + state the purpose", never "invent a value".

---

## 3. Step-level backward derivation

Every step carries two harness-written sidecar fields (NOT inside `tool_call.arguments`; the model
never emits them):
- `references` — incoming edges (what this step consumed), as `step_id`s or `{source: table}`. No
  name-matching.
- `produces` — outgoing edge (table handle / scalar / memory_id / answer).

Real reference graph of `spider_train_160`:
```
step_1 condition_filter  refs=[{source: weather}]                                  produces={table: filter_001}
step_2 aggregate         refs=[{step: step_1}]                                     produces={scalar}
step_3 add_to_memory     refs=[{step: step_2, via: source}]                        produces={memory_id: mem_step_2}
step_4 condition_filter  refs=[{source: weather}, {step: step_3, via: value_ref}]  produces={table: filter_002}
step_5 project           refs=[{step: step_4}]                                     produces={table: project_003}
step_6 answer            refs=[{step: step_5}]                                     produces={answer}
```

Three edge kinds cover all dependencies: **table consumption** (`step`/`source`), **`value_ref`**
(predicate → the step that produced that memory), **`source`** (memory → the step that produced the
scalar).

Backward derivation = transitive closure of `references` from the answer (`emitter.backward_slice`):
```python
def backward_slice(traj):
    by_id = {s["step_id"]: s for s in traj["steps"]}
    frontier = [r["step"] for r in traj["steps"][-1]["references"] if "step" in r]
    seen = set()
    while frontier:
        x = frontier.pop()
        if x in seen: continue
        seen.add(x)
        frontier += [r["step"] for r in by_id[x]["references"] if "step" in r]
    return seen
```

Trace for the example:
```
answer ─▶ step_5 ─▶ step_4 ─┬▶ {source: weather}
                            └▶ step_3 ─▶ step_2 ─▶ step_1     ← the chain through memory
backward_slice(answer) = {step_1, step_2, step_3, step_4, step_5}
```
Key: the chain `step_4 →(value_ref)→ step_3 →(source)→ step_2` — "where did the filter threshold come
from" — was severed in v1 (emitter dropped the source edge, memory was an opaque `v3`). It is now fully
traceable.

Dual trust guarantee (model-claimed provenance is never trusted):
- **offline (emitter)**: `_references()` builds edges from plan step-ids at compile time.
- **online (rollout)**: `_online_references()` derives edges from args at execution time (table handle
  → producing step, value_ref → memory step, source → scalar step), and `add_to_memory` is re-grounded
  online via `ground_derived_value` — so during RL a model that submits a forged value/reference still
  earns no process credit.

Dataset invariant: compiled-from-gold trajectories contain no dead-end steps, so every
`backward_slice(answer)` should equal the full set of prior steps — **measured 7767/7767 on V2a, 0
violations**. This proves the reference graph is connected end-to-end. Its discriminative power
(excluding wasted steps from the slice) is exercised at RL time, where model rollouts do produce
dead ends: the soundness property `t ∉ Slice(E) ⟹ c_t = 0` is a reliable waste-step detector and the
basis for dense process-reward credit assignment.

---

## 4. Artifacts / reproduction / gates

- Data (gitignored; v1 preserved): `data/trajectories/spider_{train,dev}_v2{,_think}.jsonl`
  (6769+998, `schema_version:"v2a"`) · SFT `data/sft/spider_v2_{train,dev}.jsonl` (6767+998; manifest
  records `protocol_version:v2a`, `protocol_hash:eedbb946aa0f2cb7`).
- Code: `src/harness/memory_semantics.py` (shared grounding), `compiler.py` (emits
  `{type, source_step_id}`), `plan.py` (run_plan threading), `emitter.py` (references/produces/
  backward_slice/validate), `src/sft/protocol.py` (observation envelope, specs, strict schema,
  protocol hash), `src/eval/rollout.py` (online provenance + grounding), `src/sft/splice_think.py`.
- Reproduce:
  ```
  gen_trajectories.py {dev,train} --tag=_v2
  splice_think.py --split {train,dev}
  build_sft_data.py both --input-pattern 'data/trajectories/spider_{split}_v2_think.jsonl' \
      --output-prefix spider_v2 --dataset-name spider_tools_v2 --max-est-tokens 8900
  rollout.py --replay 998        # 998/998 acceptance gate
  ```
- §8 gates: ✅ no model-visible value/provenance · ✅ typed + harness authority · ✅ no `value_subquery`
  · ✅ strict scalar source (NULL rejected) · ✅ 998/998 replay · ✅ shared `protocol_hash` · ✅ schema
  version · ✅ strict per-tool schemas · ⬜ exact Qwen-tokenizer cutoff audit (server-side; transformers
  not in the local venv — same as v1).
