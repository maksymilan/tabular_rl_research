# Scale500b process-reward v2 audit

> Historical v2 saturation audit. The distributed and recalibrated successor is documented in
> `PROCESS_REWARD_V3_DISTRIBUTED_REPORT.md`; this file and its artifacts remain unchanged controls.

Date: 2026-07-16

## 1. Batch status

The second disjoint BIRD-train batch completed all 500 unique examples with no duplicate attempts:

| outcome | count |
|---|---:|
| verified success | 139 |
| wrong answer | 127 |
| protocol terminal | 228 |
| execution terminal | 1 |
| API transport failure | 5 |

The 5 API failures are not model-semantic transitions and are excluded from reward. The new
`external_failure_adapter.py` converts the other 356 failures from raw `turns/error_events` into the
normalized replay shape without promoting rejected actions into legal steps or SFT targets.

Reward population: 139 successes + 356 semantic failures = 495 episodes.

## 2. Positive reward on 139 successes

All 139 replay correctly. Harness-inferred grounding is 139/139, with no terminal fallback.

| metric | value |
|---|---:|
| reward min / mean / max | 0.30 / 0.8878 / 1.00 |
| positive / negative / zero steps | 628 / 56 / 201 |
| BackSlice hits | 606 |
| new-evidence hits | 439 |
| search-reduction hits | 175 |
| feedback-response hits | 57 |
| max positive share p50 / p90 / max | 0.277 / 0.422 / 0.600 |

Perception credit remains selective:

| tool | total | BackSlice | E hits | positive |
|---|---:|---:|---:|---:|
| describe_table | 152 | 148 | 148 | 148 |
| inspect_column | 53 | 22 | 22 | 27 |
| read_subtable | 141 | 131 | 131 | 131 |

Positive mass by major tool: filter 40.91, describe 36.95, read 26.70, aggregate 11.73, project
5.45, join 5.14, inspect 5.08. Applied error penalties total 15.60, leaving net success return 123.40.

Representative `bird_train_03230`:

```text
describe_table          +0.2514
condition_filter        +0.2128   S=0.693
extreme_value_select    +0.2844   S=0.262
read_subtable           +0.2514
answer_from_context      0
total                   +1.0000
```

## 3. Negative reward on 356 failures

All 356 match the expected failure label and satisfy `sum(r)=C-P`. Failed trajectories receive no
applied positive reward even when a legal feedback recovery occurred, because `C=0`.

| metric | value |
|---|---:|
| reward min / mean / max | -0.80 / -0.80 / -0.80 |
| negative / zero steps | 889 / 882 |
| tool-error events | 762 |
| ignored-feedback hits | 449 |
| legal feedback responses | 92 |

Actual negative reward by cause, after per-episode normalization and cap:

| cause | mass | share |
|---|---:|---:|
| terminal failure | 176.81 | 62.1% |
| tool error | 74.63 | 26.2% |
| ignored feedback | 33.36 | 11.7% |

Failure-type behavior:

| type | n | mean raw penalty | mean negative steps | total reward |
|---|---:|---:|---:|---:|
| clean/mixed wrong answer | 127 | 1.144 | 1.54 | always -0.80 |
| protocol terminal | 228 | 2.142 | 3.02 | always -0.80 |
| execution terminal | 1 | 2.600 | 4.00 | -0.80 |

Representative allocations:

- Clean wrong answer `bird_train_04236`: final answer gets -0.80.
- Recovered-but-wrong `bird_train_02898`: four errors get -0.091/-0.091/-0.164/-0.091 and the
  final wrong answer gets -0.364. Legal recovery actions get 0 because the episode ultimately fails.
- Protocol terminal `bird_train_01221`: two prior protocol errors get -0.103 each; the terminal
  protocol error gets -0.595 because it also carries ignored-feedback and terminal-failure terms.
- Execution terminal `bird_train_00003`: four consecutive failed actions share -0.80, with the
  terminal failed join receiving -0.446.

## 4. Saturation problem

The current config has `lambda_terminal_failure=1.0` and `Pmax=0.8`. Terminal failure alone exceeds
the cap, so every failed episode has total reward -0.8. Extra tool errors only redistribute the same
negative mass across steps; they cannot make a trajectory total more negative.

This is algebraically correct but removes episode-level severity. A feature-only counterfactual with
`lambda_terminal_failure=0.4`, `lambda_tool_error=0.15`, and `lambda_ignored_feedback=0.1` gives:

| type | current mean | candidate mean |
|---|---:|---:|
| successes | 0.8878 | 0.9338 |
| wrong answers | -0.8000 | -0.4819 |
| protocol terminals | -0.8000 | -0.8000 |
| execution terminal | -0.8000 | -0.8000 |

The candidate distinguishes clean wrong answers from error-heavy failures, while repeated protocol
failures still reach the cap. It is an audit candidate, not the committed training config.

## 5. Difficulty imbalance

| difficulty | success | semantic failures | API excluded |
|---|---:|---:|---:|
| easy | 96/200 | 104 | 0 |
| medium | 43/150 | 106 | 1 |
| hard | 0/150 | 146 | 4 |

Hard failures are 145 protocol terminals and one wrong answer. Under current rewards, mean semantic
return is easy +0.0177, medium -0.3232, and hard -0.8000. This batch must not be used as a balanced
RL population: hard tasks supply no positive behavior and mostly teach format avoidance.

## 6. Conclusion

The mechanism is validated structurally at larger scale:

- positive credit is dense, harness-grounded, and stable across a second disjoint success set;
- negative credit lands on terminal failure, actual tool errors, and ignored feedback;
- API transport failures remain separate;
- all 495 semantic episodes satisfy reward conservation and expected reward sign.

It is not ready for process RL with the current sampling/configuration. Before training:

1. lower terminal-failure weight below `Pmax` and run sensitivity comparisons;
2. repair or regenerate the hard-task protocol bucket until it contains verified successes;
3. manually audit inferred grounding-edge precision;
4. keep the result-only RL baseline and compare against the same sampled tasks and budget.

Artifacts:

- `data/rl/bird_scale500b_process_reward_v2/success139/`
- `data/rl/bird_scale500b_process_reward_v2/failures356/`
- `data/rl/bird_scale500b_process_reward_v2/normalized_failures.jsonl`
- `data/rl/bird_scale500b_process_reward_v2/excluded_nonsemantic_failures.jsonl`
