# Context Management Design — large-DB readiness

Status: design for review (confirm before implementing). Decisions locked with the user 2026-06-14.
This is the layer that makes the abstract tools + memory actually deliver their reason for existing:
**work over very large databases without the context being stuffed with table data.** It is
separable from and parallel to the V2a 7B SFT/eval (V2a data is already trainable).

## 1. Problem (measured, not asserted)

On the CURRENT Spider v2 SFT data the large-DB property does NOT hold:
- median record ~2,152 tok, but the longest records (~7,200 tok) are dominated by `dataset_overview`
  — the full schema dump of every table — at ~3,800 tok (≈53% of the record); the assistant turns are
  only ~600 tok.
- a second cluster is dominated by per-step inlined `preview` content (~3,344 tok of observations).
- raw records over the build cutoff are simply dropped (data loss hiding the bloat).

Root cause is the v0/v1 context RENDERING, not the tool abstraction: full schema upfront + every tool
output inlines rows + an unbounded transcript + no compaction. The tools compress state inside
`data_view`; the rendering re-expands it into the prompt. On a wide BIRD/enterprise schema the
overview alone overflows regardless of how well the tools compress.

## 2. Two-layer context model

### A. Resident world-model — replace-in-place snapshot each turn; bounded by RELEVANCE, not DB size

Rendered as ONE current-state block that REPLACES the previous snapshot each turn (it does not
accumulate in the transcript). Contents:

- **Catalog** (always present): table names + `row_count` + relations (FK edges). NO columns.
- **Schemas**: columns + types of tables the model has `describe_table`'d. Lazily accumulated, resident.
- **Column domains**: distinct/frequent values + format of columns the model has `inspect_column`'d.
  Resident.
- **Active handles**: intermediate tables created so far (name + schema + row_count).
- **Memory**: deliberate grounded conclusions (`derived_value`; later `evidence_pointer` handles).

Key property: resident size ∝ (#described tables + #inspected columns + #handles + #memory) — the
RELEVANT subset — not the size of the DB. A 1000-table DB where the model describes 3 tables →
resident ≈ catalog + 3 schemas, not 1000.

### B. Scrolling action transcript — bounded by compaction

- Per turn: `<think>` + `<tool_call>`.
- Transient observations: `read_subtable` rows; metadata acks for table-producing tools.
- Older than a window → compacted to a handle stub (§5).

## 3. Tool surface

| Tool | Persistence of its info | Status |
|---|---|---|
| `describe_table(tables: [name, ...])` | resident (joins Schemas) | **NEW — confirmed**; multi-table in one call to avoid long describe chains |
| `inspect_column(table, column)` | resident (joins Column domains) | canon tool; needs executor impl |
| `read_subtable(table, rows[head:k\|ids\|sample:k], columns)` | transient (compacted) | un-deleted; executor has a basic version |
| table-producing tools (filter/join/group/aggregate/derive/setop/window/extreme) | metadata handle only `{table_name, columns, row_count}` — NO inlined rows | change output: drop per-step `preview` dump |

Three perception tools read three different things and do not overlap: `describe_table` = structure
(columns/types), `inspect_column` = a column's value domain, `read_subtable` = actual rows. Memory and
relational tools are unchanged (V2a memory intact).

## 4. Data construction (critical — do NOT re-create feedback-blindness)

Metadata-only outputs mean compiled gold trajectories no longer "show" the model any data. The SFT
data must INJECT read-only perception steps so the model learns to look explicitly. All injected steps
are read-only and do not change the result, so trajectories stay execution-verified:

- `describe_table([...])` for the tables the gold SQL touches, near the start.
- `inspect_column` before a string-literal equality/`contains` filter (grounding; also fixes the
  v0 "country='French' → 0 rows → ignores it" feedback-blindness).
- `read_subtable` on the evidence table before `answer_from_context` (the model SEES the answer rows —
  this is the feedback signal).

These injections add modest length per Spider trajectory; the win is at scale (describe a few relevant
tables vs dump all). Aggressiveness is a knob (§7).

## 5. Compaction

A harness rendering policy — no tool, no model action. When rendering turn t, replace row-bearing
transient observations older than the last K turns with a stub:
```json
{"step_id":"step_k","status":"success","output":{"table":"filter_002","columns":[...],"row_count":37,
 "note":"rows elided — re-read with read_subtable if needed"}}
```
The data stays in `data_view` and is deterministically re-readable. This is primarily an inference/RL
policy; SFT trajectories (≤14 steps) rarely trigger it, but the SFT data should include a few
compacted examples so the model learns the stub semantics.

## 6. Memory interface (what lives where)

- Raw perceived rows → NEVER memory (canon: memory never stores verbatim source data). Transient;
  `data_view` is the authority; re-readable.
- Schema / column-domain → resident world-model (NOT memory).
- A scalar to reuse as a literal across distance → `derived_value` memory (V2a).
- A result-set/table to reuse → `evidence_pointer` memory = a HANDLE, not rows (V2b).
- A conclusion that informs the next action → no store; it flows into the action, whose output handle
  persists.

So memory stays deliberate and sparse; reads NEVER auto-write memory (that rigidity was explicitly
rejected). Resident ≠ memory: the resident world-model is harness-maintained perception/structure;
memory is the model's deliberate, harness-grounded conclusions.

## 7. Decided defaults (locked 2026-06-14)

- Catalog includes `row_count` per table (one int — small; helps plan small-vs-huge tables).
- `compaction_window_k` is a hyperparameter, **default 3** (keep the last 3 row-bearing observations
  full; older ones compact to a handle stub).
- Injection is **CONSERVATIVE**: `describe_table` the touched tables once; `inspect_column` ONLY before
  a string-literal equality/`contains` filter (numeric filters not injected); `read_subtable` the
  evidence table once before `answer_from_context`.
- `describe_table` output = columns + types + PK/FK (NO values; values are `inspect_column`'s job).
- `inspect_column` value-domain info is RESIDENT (joins the world-model), like schema — not transient.

## 8. Implementation order (after approval; does not block V2a 7B SFT)

1. executor: table-producing tools return metadata-only; add `describe_table` (multi-table); implement
   `inspect_column`; bounded `read_subtable`.
2. protocol: resident-block renderer + transient transcript; specs for the 3 perception tools;
   observation envelope already `{step_id, status, output}`; bump `PROTOCOL_VERSION`.
3. compiler/emitter: inject perception steps; emit resident/transient split; provenance for perception
   steps (read-only; on the answer's slice only if actually consumed).
4. rollout: maintain the resident world-model + compaction; same renderer as SFT (no drift).
5. regenerate (new `schema_version`, e.g. `v2-ctx`) + replay gate; keep prior data untouched.

## 9. Relation to the canon

This refines `final_tool_design.md` §1 rather than contradicting it: the canon's "no persistent
`dynamic_table_context`" was about ROW content not being resident — which still holds (rows are
transient, §2B). `dataset_overview` (catalog + schema) was always model-visible/resident; we make it
LAZY (catalog always, schema/column-domain on demand) so it scales to wide DBs. "Compression =
acquisition restraint" and "harness may auto-compact old row-bearing results" are now made concrete.
