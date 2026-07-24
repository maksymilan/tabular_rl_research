# BIRD version24 relation-derivation fixed-200 evaluation

Date: 2026-07-24

## Decision

Version24 does **not** pass the requested 75% tool-usability gate:

- **145/200 = 72.5% `bird-set`**
- required gate: **150/200 = 75.0%**
- legal termination: **197/200**
- process errors: **29 over 23 tasks**

It does not show an aggregate accuracy regression relative to the paired version20 trajectories
rescored under the same `bird-set` metric (143/200), but the observed +2 is not statistically
meaningful: 10 gains versus 8 regressions, exact two-sided paired p = 0.815. Interface reliability
is slightly worse (197 versus 199 legal; 29 versus 24 process errors), and the package uses 5.8%
more total tokens. Do not promote version24 as a 75%-validated tool design and do not use these
evaluation traces as SFT data.

The engineering conclusion is narrower: the new relation-derivation layer did not cause a
systematic accuracy collapse, did not lengthen trajectories, and produced no derivation-schema or
resident-state execution failures over 200 tasks. It is therefore a sound semantic refactor, but
not an accuracy solution for the remaining model-policy errors.

## Frozen setup

All 200 tasks are the frozen cohort, in its original selection order:

`data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`

Both batches used:

- model: `deepseek-v4-flash`
- decoding: temperature 0, thinking enabled, reasoning effort `high`
- provider carrier: JSON Output
- context: `rolling-legal-history`, 4 successful assistant/tool pairs
- rolling prompt: full
- policy prompt: canonical
- plan policy: optional
- maximum actions/tokens: 30 / 2048
- attempts per task: 1
- strict parser: no response or argument repair
- denotation metric: `bird-set`
- protocol hash: `25ac4c10ef96365c`
- system-prompt length: 16,400 characters

Artifacts:

- first 50:
  `data/trajectories/tool_usability_20260724/version24_fixed200_first50_bird_set.*`
- remaining 150:
  `data/trajectories/tool_usability_20260724/version24_fixed200_remaining150_bird_set.*`

The two manifests have the same controlled configuration and protocol hash. Their task sets are
disjoint, contain no duplicate attempt, and cover all 200 frozen task IDs exactly. Concurrent
workers make JSONL completion order nondeterministic; pairing uses `trajectory_id`, never physical
line order.

The version20 comparison is:

`data/trajectories/tool_usability_20260723/version20_final_fixed200_json.all.jsonl`

That artifact's original 138/200 score was normalized `strict-multiset`. Current baseline-aligned
deterministic replay under `bird-set`, including the corrected SQLite percentage operation order,
is **143/200**. This report compares version24 only with that 143/200 task-level contract.

## Staged gate

The first 50 frozen tasks were evaluated before authorizing the remaining 150.

| Batch | version20 `bird-set` | version24 `bird-set` | Delta |
|---|---:|---:|---:|
| First 50 gate | 39/50 | **42/50** | +3 |
| Remaining 150 | 104/150 | **103/150** | -1 |
| Full 200 | 143/200 | **145/200** | +2 |

The first-50 paired result was 4 gains and 1 regression, with 50/50 legal termination and no final
API failure. It therefore passed the predeclared aggregate no-regression gate, and the remaining
150 were launched without changing any parameter.

## Full paired result

| Pair outcome | Count |
|---|---:|
| Both correct | 135 |
| Both wrong | 47 |
| version24-only correct | 10 |
| version20-only correct | 8 |

Version24 gains:

`06489, 04189, 02901, 06454, 01425, 03131, 03216, 04244, 04906, 02507`

Version24 regressions:

`05316, 06246, 03969, 03262, 03042, 03977, 03724, 03636`

The gains and regressions do not support a claim that derivation metadata reliably fixes output
shape. Four gains correct historical shape/representation failures (`02901`, `06454`, `03131`,
`02507`), but five regressions are also final-shape decisions:

- `05316`: keeps the correct restaurant but adds the ranking `review` helper column.
- `03969`: keeps the correct gender but also returns age.
- `03262`: returns station, item, and units instead of station and item.
- `03042`: concatenates first/last name into one column instead of two stored slots.
- `03724`: keeps rental counts beside the five requested film titles.

The other regressions are model-policy choices, not derivation execution failures:

- `06246`: changes the Delaware population/residential interpretation and counts 149 instead of 173.
- `03977`: intersects unrelated one-column sets instead of joining them by `device_id`, producing
  21 instead of 19.
- `03636`: repeatedly uses a multi-column table as a predicate scalar `value_ref` and reaches the
  execution-error cap.

No regression contains a relation-derivation schema, lineage, validation, or resident-state
exception.

## Reliability and trajectory cost

| Metric | version20 | version24 | Delta |
|---|---:|---:|---:|
| `bird-set` correct | 143 | **145** | +2 |
| Legal termination | **199** | 197 | -2 |
| Mean semantic actions | 7.46 | **7.45** | -0.01 |
| Total semantic actions | 1,492 | **1,490** | -2 |
| Process errors | **24** | 29 | +5 |
| Tasks with process errors | **21** | 23 | +2 |
| Protocol errors | **0** | 3 | +3 |
| Argument-validation errors | 10 | **9** | -1 |
| Execution errors | **14** | 17 | +3 |
| API request attempts | 1,510 | **1,509** | -1 |
| Final API failures | 0 | 0 | 0 |

The three non-legal tasks are:

- `02438`: three invalid join-key shapes exhaust the argument-validation budget.
- `03636`: three repeated non-scalar predicate `value_ref` calls exhaust the execution-error
  budget.
- `06299`: reaches 30 actions without a terminal answer; this was also the version20 max-steps
  failure.

The protocol errors consist of one invalid `group_aggregate` layout shape and two provider-visible
JSON objects with extra/wrong top-level shape. Both carrier errors recovered. Eleven API transport
retries and eight completion-length retries also recovered; all 1,490 semantic responses identify
the provider model as `deepseek-v4-flash`.

## Context and token cost

| Usage | version20 | version24 | Change |
|---|---:|---:|---:|
| Prompt tokens | 7,590,770 | 8,003,435 | +5.4% |
| Completion tokens | 371,058 | 417,426 | +12.5% |
| Total tokens | 7,961,828 | 8,420,861 | +5.8% |
| API requests | 1,510 | 1,509 | -0.1% |

The system prompt itself grows only from 16,145 to 16,400 characters (+1.6%). Because request and
action counts are unchanged, the larger prompt-token total is consistent with table-bound
derivation metadata accumulating in resident state, plus different model paths. This run measures
the complete version24 package; it does not isolate those two sources causally. The rolling
observation correctly avoids duplicating derivation outside resident state, but the resident
payload still has a measurable context cost.

## Interpretation

1. **Atomic execution remains intact.** No SQL operator, public tool argument, reference resolver,
   terminal scorer, or reward rule changed; no derivation-layer execution failure occurred.
2. **The metadata is not a reliable policy corrector.** It is fact-only by design. The model still
   makes output-slot, grain, population, join, and aggregation choices correctly on some runs and
   incorrectly on others.
3. **Accuracy is slightly positive but inconclusive.** The +2 paired result has eight
   counter-regressions and does not reach 75%.
4. **Reliability and context cost need attention.** Legal termination and process errors regress
   slightly, while total tokens increase 5.8%.
5. **Do not add more advisory fields to the derivation schema.** Remaining failures should be
   addressed through causal SFT/RL or a separately named task-aware validator, not by mixing policy
   recommendations into atomic relation metadata.

## Next decision

Keep the modular, fact-only derivation implementation as an engineering boundary, but do not treat
version24 as the validated training-data protocol. Before another paid 200-task run:

1. perform a same-action rendering ablation to measure resident derivation token overhead without
   model-policy variance;
2. decide whether derivation should remain model-visible for every historical handle or be
   losslessly compacted to only the dependencies needed by live handles;
3. target the remaining model-capability failures with a small causal SFT/RL experiment while
   keeping the public tools unchanged.
