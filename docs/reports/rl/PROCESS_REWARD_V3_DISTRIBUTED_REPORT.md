# Process-reward v3: distributed outcome responsibility

Date: 2026-07-16

## Goal

V2 already distributed positive credit over harness-derived dependency and perception edges, but
its terminal-failure term made every failed episode total `-0.8`. V3 separates two questions:

1. How much total reward or penalty should this episode receive?
2. Which environment-observable steps are responsible for that total, and by how much?

The terminal outcome controls a bounded budget. It is not itself a reason to place that whole
budget on the final action.

## Allocation

Positive `B/E/S/F` mass is unchanged. If a correct episode has no positive feature mass, fallback
credit uses the environment-derived weight

```text
q+_t = 0.25 * legal_success + 1.00 * state_changed + 0.50 * is_terminal
```

and is linearly normalized across the episode. It no longer defaults to terminal-only credit.

Failure penalty has two independent components:

```text
p_t = p_outcome_t + p_local_t
```

`p_outcome` distributes a fixed `0.30` failure budget. When an attempted dependency slice exists,
its actions and the legal terminal answer are eligible. Each eligible action gets base weight 1.0,
plus 0.25 for a real state change and 0.50 for the terminal/last responsibility boundary. If no
slice exists but legal actions do, only legal actions receive outcome responsibility. A trajectory
with no legal actions distributes the outcome budget over its rejected attempts.

`p_local` remains attached to the event that the harness can identify exactly:

| event | raw penalty |
|---|---:|
| tool/protocol/argument/execution error | 0.08 |
| repeated action without feedback | 0.06 |
| legal action without state change | 0.03 |
| ignored error/empty-result feedback | 0.05 |

The episode penalty is still capped at 0.8 and linearly normalized, so
`sum_t r_t = C - P`. Splitting one useful operation into more calls cannot increase the outcome
budget. Local penalties can increase only when the harness observes additional undesirable events.

## Scale500b replay

The audit replays 139 verified successes and 356 semantic failures. Five API transport failures
remain excluded.

### Successes

| metric | value |
|---|---:|
| replay correct | 139/139 |
| reward min / mean / max | 0.79 / 0.9649 / 1.00 |
| positive / negative / zero steps | 628 / 56 / 201 |
| max positive share p50 / p90 / max | 0.277 / 0.422 / 0.600 |

Example `bird_train_05615`:

```text
describe_table          +0.1857   B=1 E=1
inspect_column           0.0000   not used later
condition_filter        +0.1763   B=1 S=0.900
read_subtable           +0.1857   B=1 E=1
condition_filter        +0.2667   B=1 E=1 S=0.873
read_subtable           +0.1857   B=1 E=1
answer_from_context      0.0000
total                   +1.0000
```

The different values follow environment features, not action position. An unused observation stays
zero; the second filter gets more credit because it contributes dependency, evidence, and search
reduction simultaneously.

### Failures

| metric | v2 | v3 |
|---|---:|---:|
| total reward min / p50 / p90 / max | -0.80 / -0.80 / -0.80 / -0.80 | -0.80 / -0.64 / -0.30 / -0.30 |
| negative steps | 889 | 1,613 |
| episodes with multiple negative steps | not recorded | 356/356 |
| negative steps per episode p50 / p90 / max | not recorded | 3 / 8 / 19 |
| max single-step negative share p50 / p90 / max | not recorded | 0.404 / 0.404 / 0.556 |

There are 73 clean/low-error failures at `-0.30`, 214 typical three-error failures at `-0.64`, and
only one episode reaches the `-0.80` cap. That capped episode has six actual error events and raw
penalty 0.98; its largest individual step reward is only `-0.1224`.

Example `bird_train_00622` totals `-0.38`:

```text
describe_table          -0.0484   attempted dependency + state change
inspect_column           0.0000   not on attempted dependency chain
condition_filter        -0.0484   attempted dependency + state change
read_subtable            0.0000   not used by the attempted final chain
protocol_error          -0.0800   local error only
join_tables             -0.0484   attempted dependency + state change
project                 -0.0484   attempted dependency + state change
read_subtable           -0.0484   attempted dependency + state change
answer_from_context     -0.0581   terminal outcome responsibility
total                   -0.3800
```

## Decision

The allocation behavior now matches the intended mechanics: reward and penalty are multi-step,
bounded, non-uniform, and derived without an external model at runtime. External model review is
only an offline QA mechanism for improving deterministic harness rules.

This does not clear process RL yet. The external grounding audit still found deterministic false
edges, and the Scale500b hard subset has no positive trajectories. V3 fixes concentration and
severity calibration; grounding precision and sampling balance remain separate release gates.

Artifacts:

- `configs/process_reward_v3_distributed.json`
- `data/rl/bird_scale500b_process_reward_v3_distributed/success139/`
- `data/rl/bird_scale500b_process_reward_v3_distributed/failures356/`

Verification: 19 RL tests, 136 harness tests, Spider compile coverage 1998/2000, and reward
conservation on all 495 semantic episodes.
