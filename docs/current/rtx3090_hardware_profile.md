# RTX 3090 hardware profile for table-agent RL

## Status and scope

This is the active hardware policy for Qwen2.5-Coder-7B-Instruct plus the frozen SFT2 LoRA on
24 GiB RTX 3090 cards. It changes scheduling and throughput only; it does not change model,
adapter, prompts, sampling, rewards, task order, or evaluation semantics. The sourceable defaults
are in `src/rl/configs/hardware/rtx3090_24gb.sh`.

The profile was measured on NewGNN GPU 6 (RTX 3090, 24,576 MiB, 350 W) using the `trl-table`
runtime (PyTorch 2.9.0+cu128, vLLM 0.12.0, Transformers 4.57.6). The fixed benchmark contains 12
BIRD-train questions (four per difficulty), K=4, temperature 0.7, top-p 0.95, version26,
max_steps 30, max_new_tokens 1,024, max context 8,192, prefix caching, and eager execution.
Benchmark outputs are operational diagnostics and are excluded from every training pool.

## Selected configurations

### Interactive rollout collection

Use the rolling dynamic scheduler with `question_window=8` and `group_size=4`, for at most 32
concurrently advancing trajectories. As soon as one question's four trajectories finish, admit
the next question instead of waiting for the slowest question in a static batch. Use
`gpu_memory_utilization=0.82`, max model length and max batched tokens 8,192, prefix caching
enabled, and eager execution. If an OOM or allocator failure occurs, retry the untouched task
group with `question_window=4`; do not change sampling or overwrite complete group files.

The initial concurrency sweep produced:

| task batch | active trajectories | tasks/hour | completion tok/s | peak MiB | result |
|---:|---:|---:|---:|---:|---|
| 2 | 8 | 98.44 | 107.89 | 20,796 | stable |
| 3 | 12 | 119.37 | 135.40 | 21,310 | stable |
| 4 | 16 | 122.29 | 142.55 | 20,930 | stable, fallback |
| 6 | 24 | 105.61 | 130.49 | 22,954 | stable, slower |
| 8 | 32 | 209.44 | 205.58 | 21,928 | stable, selected |
| 12 | 48 | 140.08 | 170.06 | 20,930 | stable, tail-limited |

Batch 8 is 112.8% faster than batch 2 and 71.3% faster than batch 4 by end-to-end tasks/hour.
Batch 12 is stable but slower because the static batch waits for its slowest trajectory. GPU
utilization alone is not the
selection metric because every tool turn alternates model inference with CPU/SQLite execution.
The selected metric is end-to-end tasks/hour; completion-token throughput is the cross-check.

An isolated batch-8 retest with `gpu_memory_utilization=0.90` increased the KV cache from 67,680
to 103,152 tokens but fell to 167.77 tasks/hour and 167.37 completion tok/s, with a 22,834 MiB
peak. Therefore 0.90 is rejected for interactive rollouts; it remains appropriate for the formal
evaluator, whose request scheduler and traffic pattern differ.

The dynamic scheduler was then validated on the same 12-question/K=4 workload with stable
per-turn seeds derived from `(base_seed, task_id, sample_index, turn_index)`. The final audited
run completed 12 groups, 48 episodes, and 337 turns in 343.51 seconds (125.76 tasks/hour and
154.14 completion tokens/s), versus 556.84 seconds and 77.58 tasks/hour for the stable-seed
static batch-8 run: **62.1% higher task throughput and 38.3% lower wall time**. All 48 episodes
retained `sample_index`, unique sequence ids, prompt/response token ids, and aligned sampled-token
logprobs. Stochastic decoding can still change exact sampled paths when GPU batching order changes;
the scheduler is therefore an operational throughput setting, not a promise of bitwise-identical
rollouts. It leaves the declared model, sampling distribution, task order, and scoring unchanged.

### Formal greedy and K=4 evaluation

Use one vLLM instance per GPU with `gpu_memory_utilization=0.90`, max model length 8,192,
`max_num_batched_tokens=8192`, `max_inflight=24`, and `max_num_seqs=24`.

- Greedy@1: `workers=24`, `sample_workers=1`.
- K=4: `workers=6`, `sample_workers=4` (24 active samples).
- When two cards are free, run different checkpoints or disjoint question shards on the two
  cards. Do not tensor-parallelize this 7B evaluator across both 3090s.

The full-dev Exp10 greedy run on NewGNN GPU 7 is the saturation check for this profile: the card
runs near full compute utilization at concurrency 24 without sharing it with another model.

### Training

Online RL uses exactly two distinct 3090s: one card holds the vLLM rollout policy and the other
holds the trainable QLoRA policy/optimizer. Keep vLLM memory utilization at 0.82 and transition
micro-batch size at 1. The rollout card receives refreshed adapter weights through the existing
TRL synchronization path; do not place trainer and vLLM on the same card.

Frozen-pool RL has no online rollout server. Run one independent experiment per free 3090, with
each experiment initialized directly from the same frozen SFT2 checkpoint. Two free cards may run
two independent candidates in parallel; never initialize one candidate from another RL result.

## Allocation and failure rules

- A card is free only below 512 MiB with no compute PID. On a shared host, require ten consecutive
  30-second checks and recheck immediately at launch.
- Never stop, preempt, or replace another user's process. NewGNN remains restricted to explicitly
  authorized GPUs.
- One GPU hosts at most one model-serving or training process. CPU/SQLite workers may be concurrent.
- Resume interrupted rollout generation with identical model/sampling parameters and preserve
  complete atomic group files. Restarting alone is not an experimental variable.
- On OOM, lower only the hardware concurrency tier. Do not silently change model length,
  max steps, sampling, prompts, or rewards.
- NewGNN outputs belong under `/home/dengyan`; do not add large artifacts to the nearly full
  `/data` filesystem.

## Evidence

The benchmark launcher is `src/rl/experiments/benchmark_rtx3090_agent_rollout_newgnn.sh`.
Server artifacts live under
`/home/dengyan/tabular_rl_outputs/benchmarks/rtx3090_agent_rollout_20260801` and include the fixed
task manifest, per-variant generation logs, one-second GPU samples, and JSON summaries. Any later
hardware retuning must use an isolated benchmark id and preserve these results.

The final dynamic validation is preserved under
`/home/dengyan/tabular_rl_outputs/benchmarks/rtx3090_agent_rollout_dynamic_window8_validated_20260801`;
the stable-seed static comparator is under
`/home/dengyan/tabular_rl_outputs/benchmarks/rtx3090_agent_rollout_static_seeded_batch8_20260801`.
