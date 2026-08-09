# OmniSQL SFT union epoch 4 — BIRD-dev evaluation

This directory contains the compact evaluation artifacts copied from NewGNN on 2026-08-05.

## Greedy

- Protocol: atomic `version36`, BIRD set denotation, temperature 0, top-p 1, one sample, max 30 steps.
- Accuracy: 789/1534 = 51.4342%.
- Valid/legal: 1264/1534 = 82.3990%.
- Mean steps: 9.4544.
- SFT2 matched baseline: 762/1534 = 49.6741%; OmniSQL SFT is +27 questions / +1.7601 pp.
- Original OmniSQL Direct SQL system reference: 983/1534 = 64.0808%; OmniSQL SFT tool use is -194 questions / -12.6467 pp.

## Pass@4

- Protocol: atomic `version36`, BIRD set denotation, temperature 0.7, top-p 0.95, four samples, max 30 steps.
- Pass@1: 817/1534 = 53.2595%.
- Pass@2: 946/1534 = 61.6688%.
- Pass@4: 1033/1534 = 67.3403%.
- All-sample valid/legal: 5297/6136 = 86.3266%.
- At least one valid sample: 1485/1534 = 96.8057%.
- Mean steps per sample: 8.5841.
- Pass@4 over Pass@1: +216 questions / +14.0808 pp.

## Files

- `accuracy.json`: Pass@k accuracy.
- `valid_rate.json`: legal/valid completion rates.
- `avg_steps.json`: mean tool-step counts.
- `failure_types.json`: terminal outcome counts.
- `per_question_result.json`: compact per-question results.
- `trajectory.json`: compact trajectory records.
- `manifest.json` and `evaluation_protocol.json`: frozen evaluation configuration.

The large raw `all.jsonl` files remain on NewGNN:

- Greedy: `/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728/data/results/omnisql_sft_union_epoch4_version36_dev1534_greedy_t0_logprobs20_20260805/all.jsonl` (about 260 MiB).
- Pass@4: `/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728/data/results/omnisql_sft_union_epoch4_version36_dev1534_passk4_t07_p095_logprobs20_20260805/all.jsonl` (about 887 MiB).
