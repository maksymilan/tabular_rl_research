# BIRD hard Pro rescue probe under the current protocol

Date: 2026-07-17

## Setup

This is a paired rescue test, not a new random benchmark. A deterministic seed (`20260717`)
selected 30 tasks from the 145 Scale500b examples that were all:

- BIRD-train difficulty proxy `hard`;
- terminal `protocol_error` under `deepseek-v4-flash`;
- generated under the current rolling-history-4, full-prompt, 2048-token strict protocol.

The rerun changed only the teacher model to `deepseek-v4-pro`. It retained one trajectory attempt,
`max_steps=30`, three errors per class, strict no-repair parsing, and the same protocol hash
`d18c629372c8a7ab`.

Selection input and manifest:

- `data/eval_inputs/bird_scale500b_hard_protocol_pro30.jsonl`
- `data/eval_inputs/bird_scale500b_hard_protocol_pro30.manifest.json`

## Result

| metric | paired Flash | Pro |
|---|---:|---:|
| tasks | 30 | 30 |
| legal terminal episodes | 0 | 30 |
| execution-correct | 0 | 20 |
| wrong answer | 0 | 10 |
| terminal protocol error | 30 | 0 |
| protocol-error events | 90 | 5 |
| execution-error events | 0 | 2 |

Pro therefore reaches **20/30 = 66.67%** hard execution accuracy and **30/30 = 100%** legal
terminal rate on tasks where Flash previously exhausted its protocol-error allowance. All 20
accepted trajectories replay from the initial SQLite database successfully. They contain 16
`clean_success` and 4 `recovered_success` episodes; the four accepted recovery targets follow three
protocol errors and one execution error.

Under the current SFT quality gate (`<=12` legal steps, `<=300` words per think, no repeated
identical call), 18/20 successes remain eligible and yield 149 single-step targets. Two otherwise
correct trajectories exceed the think-length gate.

## Usage

The Pro run used four workers and completed in 433.8 seconds:

- 278 API request attempts;
- 1 API transport retry;
- 1,081,658 prompt tokens, of which 604,928 were cached;
- 38,052 completion tokens, including 21,264 reasoning tokens;
- 1,119,710 total tokens.

The paired Flash failures used only 224,144 tokens because they terminated after three rejected
turns each. The roughly 5x token difference is therefore mainly the cost of Pro completing long
closed-loop trajectories, not a controlled per-token price comparison.

## Interpretation

For this paired hard protocol-failure bucket, the dominant bottleneck was Flash's output-channel
reliability, not an inherent inability of the current tools to solve every task. Pro removes the
format bottleneck and exposes the next error layer: 10/30 trajectories are legal but semantically
wrong. This result does not imply 66.67% accuracy over all BIRD hard tasks because the sample was
conditioned on Flash protocol failures and contains one hosted-model draw per task.

Artifacts:

- `data/trajectories/bird_ds_v4pro_scale500b_hard_protocol30_rolling4_full_all.jsonl`
- `data/trajectories/bird_ds_v4pro_scale500b_hard_protocol30_rolling4_full_success.jsonl`
- `data/trajectories/bird_ds_v4pro_scale500b_hard_protocol30_rolling4_full_failures.jsonl`
- `data/trajectories/bird_ds_v4pro_scale500b_hard_protocol30_rolling4_full_success.manifest.json`
- `data/trajectories/bird_ds_v4pro_scale500b_hard_protocol30_rolling4_full_success.audit.json`
