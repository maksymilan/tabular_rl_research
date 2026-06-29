# RL pilot environment

This folder prepares the first small RL verification loop for the table-tool agent.
It does **not** yet commit us to PPO, GRPO, or a final reward design. The immediate goal is
to make the environment and reward auditable before spending GPU time.

## What is ready

- `env.py` wraps one Spider question as a closed-loop tool environment.
  The trainer generates an assistant message; the environment parses it, executes the tool,
  appends the model-visible observation, and scores terminal answers.
- `reward.py` defines a transparent pilot reward:
  final correctness dominates, legal final answers get a small bonus, repeated calls and
  tool/protocol errors are penalized, and reading evidence before answering gets a small bonus.
- `build_reward_report.py` scores existing `rollout.py` or `rollout_passk.py` artifacts.
- `select_pilot_tasks.py` selects tasks for the first RL batch from pass@k artifacts.

## First verification loop

1. Run or reuse SFT rollouts.

   For pass@2:

   ```bash
   EVAL_ENABLE_THINKING=0 python src/eval/rollout_passk.py \
     --base-url http://127.0.0.1:8002/v1 \
     --model qwen35_v9_e4_tool \
     --n 1034 --workers 4 --sample-workers 2 \
     --n-samples 2 --pass-k 2 \
     --temperature 0.7 --top-p 0.95 \
     --result-dir data/results/qwen3.5_9b_sft_v9_ready_4k_epoch4/tool_pass2_dev1034 \
     --resume
   ```

2. Score rollout samples with the pilot reward.

   ```bash
   .venv/bin/python src/rl/build_reward_report.py \
     --input data/results/qwen3.5_9b_sft_v9_ready_4k_epoch4/tool_pass2_dev1034/all.jsonl \
     --output-dir data/rl/qwen35_v9_pass2_reward_report
   ```

3. Select a first RL task set.

   ```bash
   .venv/bin/python src/rl/select_pilot_tasks.py \
     --input data/results/qwen3.5_9b_sft_v9_ready_4k_epoch4/tool_pass2_dev1034/all.jsonl \
     --output data/rl/qwen35_v9_pilot_tasks.jsonl \
     --limit 200 --include-hard
   ```

## Acceptance criteria before training

- The selected batch should mostly be `mixed_success` or legal-but-wrong cases, not API/protocol
  failures.
- Reward ranking should match manual inspection on a 20-case sample: correct legal trajectories
  above legal wrong trajectories, legal wrong above protocol/execution failures, and repeated
  useless calls penalized.
- The rollout legal rate should remain high enough that RL is not merely learning XML/JSON format.
  As a rule of thumb, start pilot RL only if legal rate is above 90%, preferably 95%.

## Expected first RL result

For the first small run, a useful result is not necessarily a huge accuracy jump. It is enough if:

- pass@1 improves on the held-out dev slice,
- legal rate does not drop,
- `all_samples_failed` shrinks on the pilot failure buckets,
- sampled trajectories show better use of observations after empty or surprising tool outputs.

If the reward report does not align with manual judgment, fix the reward before running PPO/GRPO.
