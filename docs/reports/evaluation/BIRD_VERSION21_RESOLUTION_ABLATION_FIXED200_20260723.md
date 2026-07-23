# BIRD version21 deterministic-resolution ablation

Date: 2026-07-23

## Decision

Do **not** promote version21. Restore version20 as the current canonical protocol.

Version21 made several deterministic argument forms easier to execute, and its targeted pilot
looked positive. On the full frozen 200-task cohort, however, it produced only one additional
correct task while reducing legal termination, increasing process errors, lengthening trajectories,
and increasing token use. It still missed the 150/200 gate by 11 tasks.

## Change under test

Starting from committed version20 (`5dc152c`), version21 tested:

- safe unique-bare resolution for `join_tables.joins[].on[].left`;
- one removable current-handle prefix before an otherwise exact logical join column;
- safe unique dotted-suffix resolution for named `scalar_compute` columns;
- optional call-level join `type` as the default for all edges, with edge-level overrides.

Ambiguous names remained errors. Authored arguments and provenance references were preserved. Plan,
right-hand join identifiers, aggregate grammar, and gold visibility were unchanged.

## Setup

- frozen input:
  `data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`
- model: `deepseek-v4-flash`
- temperature 0, thinking enabled, reasoning effort `high`
- JSON Output carrier
- rolling legal history = 4
- optional resident plan
- max actions = 30
- initial max completion = 2048
- normalized `strict-multiset`
- version21 protocol hash: `d3111e8b87ef4432`

Full artifact:

`data/trajectories/tool_usability_20260723/version21_resolution_fixed200_json.*`

## Targeted eight-task pilot

The pilot selected all version20 tasks that had exercised the intended gaps: bare join left,
call-level join type, redundant handle prefix, or named scalar suffix.

| Metric | version20 fixed-200 records on the 8 IDs | version21 fresh pilot |
| --- | ---: | ---: |
| Correct | 4/8 | **5/8** |
| Legal | 7/8 | 7/8 |
| Process errors | 10 | **2** |
| Argument-validation errors | 5 | **0** |
| Execution errors | 5 | **2** |

No previously correct pilot task regressed. `02052` changed from wrong to correct in the fresh
pilot; `00593` and `06299` remained a grain error and a strategy loop respectively. This was enough
to justify the full run, but it was not treated as a stable accuracy estimate.

## Fixed-200 result

| Metric | version20 canonical | version21 ablation | Delta |
| --- | ---: | ---: | ---: |
| Strict-multiset correct | 138/200 (69.0%) | 139/200 (69.5%) | +1 |
| Legal termination | **199/200** | 197/200 | -2 |
| Mean semantic actions | **7.460** | 7.920 | +0.460 |
| Process errors | **24** | 38 | +14 |
| Tasks with process errors | **21** | 30 | +9 |
| Protocol errors | **0** | 2 | +2 |
| Argument-validation errors | **10** | 12 | +2 |
| Execution errors | **14** | 24 | +10 |
| API requests | **1510** | 1598 | +88 |
| Total tokens | **7,961,828** | 8,596,293 | +634,465 (+8.0%) |

Paired denotation:

- both correct: 130
- both wrong: 53
- version21-only correct: 9
- version20-only correct: 8
- exact two-sided McNemar/binomial p = 1.0

Legal termination has zero gains and two regressions. The regressions were `02438` (four unrelated
carrier/read/right-key validation events) and `06165` (a long strategy loop with an invalid
multi-column `in_table`). `06299` again reached max steps. These are not evidence that deterministic
left/scalar resolution changed SQL semantics, but they still count against the observed interface
run and prevent promotion.

## Interpretation

The intended forms did become executable: the full run had successful call-level join types,
successful named scalar calls, and no recurrence of version20's call-level-type rejection,
bare-left validation error, or named-scalar missing-column error. The targeted mechanism works.

That mechanism did not translate into a stable full-cohort gain. Independent greedy rollouts
changed many later semantic choices: 9 gains and 8 regressions are ordinary trajectory variance,
while process errors shifted to plan calls, oversized reads, aggregate argument placement,
multi-column `in_table`, stale relation names, and malformed project expressions.

The result strengthens the version20 conclusion:

1. safe grammar tolerance can remove local friction;
2. local friction is no longer the dominant accuracy bottleneck;
3. broadening accepted call forms does not recover the 53+ legal semantic failures;
4. chasing 75% through more resolver aliases would weaken the strict, teachable contract without
   evidence of a reliable denotation gain.

Version20 therefore remains canonical. Version21 is an audited negative ablation, not an active
training or evaluation protocol.
