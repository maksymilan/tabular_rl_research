# BIRD-train Flash version2 failure replay (110 tasks)

Date: 2026-07-22

## Purpose

Replay every failed typed-tool task from the version1 200-task interface ablation after two scoped
changes:

1. add concise canonical JSON calls and an exact final-table shape contract to the full prompt; and
2. fix `condition_filter` normalization so public `is_null` becomes executable `IS NULL` instead of
   falling through to `KeyError: value`.

The public protocol uses numeric versions from this point onward. The former
`v2i-state-only-join-feedback-r2` contract is `version1`; this run is `version2`.

## Setup

- Input: all 110 version1 failures from the frozen 200-task sample
- Distribution: 39 easy, 29 medium, 42 hard
- Model: `deepseek-v4-flash`
- Attempts: one new complete episode per task
- Budget: 30 actions, 3 errors per class
- Context: rolling legal history, 4 successful turns
- Judge: `strict-multiset`
- Version2 protocol hash: `532ed9857e74065c`
- Parser repair: disabled

This is a failure-conditioned replay, not an independent pass@1 evaluation. It measures recovery on
known version1 failures and includes ordinary sampling variance.

## Result

| Metric | Result |
|---|---:|
| Recovered correct | 46/110 (41.8%) |
| Legal terminal | 98/110 (89.1%) |
| New wrong denotation | 52 |
| New protocol termination | 10 |
| New execution termination | 2 |

Recovery by difficulty:

| Difficulty | Recovered |
|---|---:|
| Easy | 18/39 (46.2%) |
| Medium | 14/29 (48.3%) |
| Hard | 14/42 (33.3%) |

Recovery by version1 failure type:

| Version1 failure | N | Recovered in version2 |
|---|---:|---:|
| Wrong denotation | 76 | 29 (38.2%) |
| Protocol termination | 32 | 15 (46.9%) |
| Execution termination | 1 | 1 |
| Argument-validation termination | 1 | 1 |

Combining the original 90 version1 successes with these 46 second-attempt recoveries gives 136/200,
but **68.0% is conditional success@2, not version2 pass@1**. A fresh full-200 version2 run is required
for a pass@1 claim.

## Final-shape effect

| Diagnostic on the same selected tasks | Version1 | Version2 |
|---|---:|---:|
| Parsed `project` calls | 11 | 52 |
| Legal `project` calls | 11 | 49 |
| Legal wrong answers with column-arity mismatch | 50 | 14 |
| Clear ordered projection-superset failures | 26 | 3 |

Eighteen of the 26 clear version1 projection-superset cases became correct. Recovered examples include
`bird_train_05102`, `06227`, `04680`, `02582`, `03289`, `03175`, `03628`, `01625`, `00093`,
`00578`, `02652`, `03189`, `03751`, `02475`, `02919`, `04576`, `01307`, and `03085`.

The earlier Chicago Lawn case (`bird_train_05102`) now projects commander and email before terminal
submission. The one-game publisher case (`bird_train_01320`) also becomes correct after producing an
exact final table. This is direct evidence that the former terminal wording caused avoidable losses.

## Error-feedback effect

Across the selected tasks, protocol-error events fell from 146/988 turns (14.8%) to 80/973 turns
(8.2%). Final protocol terminations fell from 32 to 10. Internal `KeyError` execution events fell
from six to zero; the old repeated `is_null` failure `bird_train_00594` is now correct.

The remaining execution errors are primarily join-identifier mistakes (`table.column`, prefixed
right-side keys, or internal `L./R.` forms). Canonical examples helped final shaping but did not make
the prefix-based n-way join contract intuitive enough. This remains the next tool-design issue.

## Cost

| Same 110 selected tasks | Version1 | Version2 |
|---|---:|---:|
| Total tokens | 3,925,264 | 4,363,997 |
| API requests | 993 | 974 |
| Semantic turns | 988 | 973 |

The examples increased total token use by 11.2% on this replay while slightly reducing actions and
requests. The measured recovery is substantial, but the next prompt revision should replace or
compress existing prose rather than continuing to append examples.

## Artifacts

- Input: `data/eval_inputs/bird_train_tool_interface_ablation200_typed_failures110_version2.jsonl`
- All attempts: `data/trajectories/bird_interface_ablation110_version2_prompt_flash_all.jsonl`
- Successes: `data/trajectories/bird_interface_ablation110_version2_prompt_flash_success.jsonl`
- Failures: `data/trajectories/bird_interface_ablation110_version2_prompt_flash_failures.jsonl`
