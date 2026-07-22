# Grounded R2 SFT evaluation

Date: 2026-07-18

## Setup

- Model: Qwen2.5-7B-Instruct QLoRA.
- Training: 206 BIRD-train episodes, 1,303 step targets; 22 disjoint eval episodes, 130 targets.
- Protocol: full system prompt, rolling legal history 4, R2 resident observations, strict parser.
- Closed-loop gate: fixed BIRD-dev 30 sample, seed 20260717, 11 databases, difficulty
  simple/moderate/challenging = 12/9/9.
- Comparison: old 40-episode adapter on the same 30 task ids. The old adapter uses its matching
  pre-R2 full-observation protocol, so this is a behavioral model comparison, not a pure protocol
  ablation.

## Results

| Model | Correct | Legal terminal | Simple | Moderate | Challenging | Avg. steps |
|---|---:|---:|---:|---:|---:|---:|
| Old 40 episodes | 6/30 (20.0%) | 24/30 | 3/12 | 1/9 | 2/9 | 8.07 |
| Grounded R2 epoch 1 (`checkpoint-82`) | **8/30 (26.7%)** | 23/30 | 3/12 | 2/9 | 3/9 | 10.73 |
| Grounded R2 epoch 2 (final) | 7/30 (23.3%) | 24/30 | 4/12 | 1/9 | 2/9 | 8.77 |

Epoch 1 outcomes: 8 success, 15 legal wrong answers, 3 execution-error terminals, and 4 max-step
terminals. Epoch 2 outcomes: 7 success, 17 legal wrong answers, 2 execution-error terminals, 2
protocol terminals, 1 context overflow, and 1 max-step terminal.

Epoch 1 and epoch 2 share five correct tasks. Epoch 1 alone solves dev indices 46, 1194, and 1479;
epoch 2 alone solves 789 and 1462. This is not a monotonic checkpoint improvement.

## Single-step held-out evaluation

On all 130 disjoint R2 eval targets, the final adapter has:

- strict-valid actions: 126/130 = 96.9%
- teacher tool match: 100/130 = 76.9%
- exact tool + arguments: 48/130 = 36.9%
- first-step tool match: 22/22 = 100%
- later-step tool match: 78/108 = 72.2%

On the same 20 episode ids / 117 states used by the old-adapter gate, the final R2 adapter has
95.7% valid actions, 75.2% tool match, and 35.0% exact action match. These do not improve on the
old adapter's approximate 99.1% / 77.8% / 38.5%; the input protocols differ in observation
compaction, so the comparison is directional rather than a clean ablation.

Training validation loss does improve from 0.4753 at epoch 1 to 0.4608 at epoch 2. The disagreement
with closed-loop execution confirms that token loss is not sufficient for checkpoint selection.

## Decision

Use `checkpoint-82` as the current behavior-selected adapter and retain the final adapter as a
comparison. The larger dataset gives a modest dev-30 gain (20.0% to 26.7%), but it does not improve
average teacher-forced action matching. The remaining bottleneck is semantic action/argument
selection and long-horizon error accumulation, not basic protocol acquisition.

Before RL or a strong scaling claim, evaluate epoch 1 and epoch 2 on a larger fixed BIRD-dev sample
(at least 100 tasks). Dev-30 is a useful gate but too small for a stable 1-task difference.

## Artifacts

- Epoch 1: `data/results/qwen2.5_7b_bird_grounded_r2_epoch1_bird_dev_stratified30/`
- Epoch 2: `data/results/qwen2.5_7b_bird_grounded_r2_bird_dev_stratified30/`
- Held-out step 130: `data/results/qwen2.5_7b_bird_grounded_r2_step_eval130/`
- Matched step 117: `data/results/qwen2.5_7b_bird_grounded_r2_step_eval_unseen20_matched/`

