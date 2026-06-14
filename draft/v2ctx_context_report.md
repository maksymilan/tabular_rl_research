# V2-ctx Context-Management — Transformation Report

Status: implemented + verified (2026-06-14). The layer that makes the abstract tools deliver their
reason for existing: **work over large databases without the context being stuffed with table data.**
Schema `schema_version: "v2-ctx"`; data files `spider_*_v2ctx*` (v2a files untouched).

Verification: Spider dev replay **998/998** execute+score · round-trip keep-rate 6769 train / 998 dev
(same as v2a; 4 NULL-source memory rejects) · unit tests 98 (harness) · real-DB exec-verified 98.4%
(compiler unchanged) · strict per-tool schema covers the 3 new perception tools.

## 1. What changed (the transformation)

| Before (v2a) | After (v2-ctx) |
|---|---|
| Opening `dataset_overview` dumps **every column of every table** upfront | Opening = **CATALOG**: table names + row_counts + FK relations ONLY (no columns) |
| Every table-producing tool **inlines its rows** into the observation | Table outputs are **metadata-only** handles `{table, columns, row_count}` (a 1×1 scalar result keeps its one cell) |
| Model never explicitly looks (rows arrive for free) | **Injected resident perception**: `describe_table` the touched tables, `inspect_column` before a string-literal filter, `read_subtable` the evidence once before answering |
| n/a | New tools `describe_table(tables)` (multi-table), `inspect_column`, `read_subtable`; shared schema/validation/protocol_hash |

Property that matters: the opening context now scales with **table COUNT**, and schema acquisition
with **TOUCHED tables** — not with the DB's total column count. That is what makes huge DBs fit.

## 2. Concrete case (real-shape trajectory)

Question: *What are the names of singers from France?*  Gold: `SELECT name FROM singer WHERE country='France'`

Opening CATALOG (no columns — vs v2a which listed every column of singer+country):
```json
{"tables":[{"table_name":"country","num_rows":3},{"table_name":"singer","num_rows":4}],
 "relations":[{"from":"singer.country","to":"country.code"}]}
```
Steps (note: the model now describes, grounds, and reads explicitly; tool outputs are handles):
```
step_1 describe_table([singer])   -> singer columns/types/PK/FK         (resident schema)
step_2 inspect_column(singer,country) -> {distinct 3, frequent:[France,...]}  (grounds 'France')
step_3 condition_filter(country='France') -> {table:filter_001, columns, row_count:2}   (HANDLE, no rows)
step_4 project([name])            -> {table:project_002, columns:[name], row_count:2}    (HANDLE, no rows)
step_5 read_subtable(project_002) -> rows [[Alice],[Cara]]              (the ONE place rows enter)
step_6 answer_from_context(evidence=project_002, ...)
```
`backward_slice(answer) = {step_3, step_4}` — the computational chain; the perception steps
(describe/inspect/read) are correctly OFF the slice (read-only, they let the model see, not compute).
Round-trip verified.

## 3. Measured results

### Spider (small DBs, 2–5 tables): the long tail is compressed
Matched dev trajectories (998, same think), est tokens of the rendered SFT record:

| | p50 | p90 | p95 | max |
|---|---|---|---|---|
| v2a | 2242 | 3174 | 3610 | 6532 |
| v2-ctx | 2258 | 2832 | 3108 | 6896 |
| Δ | +1% | **-11%** | **-14%** | +6% (one whole-DB query) |

The **records near the cutoff shrink** (p90/p95 down ~11–14%) — exactly the ones that were the
problem. The median is flat (injection overhead offsets the catalog saving on small queries). On the
server Qwen-tokenizer audit, v2a had 8 records over the 8192 cutoff (filtered to build
`spider_tools_v2_8k`); v2-ctx's tail is shorter.

### Large DB (the design's real payoff): synthetic 60 tables × 12 cols, query touches 2

| opening schema context | tokens |
|---|---|
| v2a (full schema, all 60 tables upfront) | **7,817** |
| v2-ctx catalog (60 names + relations) | 656 |
| v2-ctx describe_table(2 touched) | 371 |
| v2-ctx total | **1,027** |
| **reduction** | **-87%** |

v2a grows with **total** tables (a 600-table DB ≈ 78k tokens → overflow); v2-ctx grows with **touched**
tables (≈ 6.9k). This is the property the abstract-tools design was supposed to deliver, now realized.

## 4. Honest scope

- On small Spider DBs the benefit is **tail-only** (p95 -14%) — the median is flat because the
  injected describe/inspect/read add ~as much as the lazy catalog saves when a query touches most of
  a tiny DB. The value is **structural and scales with DB size**; the synthetic case shows it.
- The grounding injection (`inspect_column` before a literal filter, `read_subtable` of the evidence)
  is a second, independent benefit: it directly targets the v0 **feedback-blindness** (filter
  `country='French'` → 0 rows → ignore). The model now sees the column domain and the result rows.
- **Not implemented (deferred to the RL env):** the "replace-in-place resident block" rendering and
  the compaction window (`compaction_window_k`, design §5). For SFT trajectories (≤14 steps) the
  resident info (schema, domains) simply stays in the short transcript; compaction matters only for
  long-horizon RL rollouts. The SFT-relevant parts (lazy catalog, metadata-only, perception injection)
  are done and verified.

## 5. Artifacts / remaining

- Code: `executor.py` (describe_table/inspect_column), `emitter.py` (catalog + metadata-only +
  injection, `schema_version=v2-ctx`), `protocol.py` (3 perception specs, system prompt, schema),
  `rollout.py` (perception execution + replay). Data (gitignored): `spider_*_v2ctx*`; SFT
  `spider_v2ctx_*` (template think).
- Remaining before a v2-ctx SFT run: think-fill/splice for the new step structure, then train —
  **sequenced after the running v2a SFT is evaluated** (don't stack unvalidated layers). Compaction +
  resident-block rendering land with the RL env.
