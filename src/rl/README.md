# RL pilot environment

This folder prepares the first small RL verification loop for the table-tool agent.
It does **not** yet commit us to PPO, GRPO, or a final reward design. The immediate goal is
to make the environment and reward auditable before spending GPU time.

## What is ready

- `env.py` wraps one dataset-adapter task as a closed-loop tool environment. The trainer generates
  one assistant action; the environment strictly parses it, executes it, rebuilds the next
  model-visible context from resident state plus an optional `LAST TOOL ERROR`, and scores terminal
  answers. It accepts adapter-provided `db_path`/`gold_sql`, so Spider and BIRD use the same loop.
- `reward.py` defines a transparent pilot reward:
  final correctness dominates, legal final answers get a small bonus, repeated calls and
  tool/protocol errors are penalized, and reading evidence before answering gets a small bonus.
- `build_reward_report.py` scores existing `rollout.py` or `rollout_passk.py` artifacts.
- `select_pilot_tasks.py` selects tasks for the first RL batch from pass@k artifacts.

## Layout

- `env.py`, `reward.py`, `task_data.py`, candidate selection, and reward reports are framework-neutral.
  They define the tool episode, hidden execution labels, and rewards once for every backend.
- `frameworks/<backend>/` contains only backend glue.  The current `frameworks/verl/` adapter owns
  Verl's token loop, Parquet serialization, Hydra config, launcher, and machine-specific patch.

This separation lets a compatible training adapter reuse the same `ToolUseEnv`, task records, and
`terminal_result_reward` without copying harness behavior or changing the experimental definition.

## Process reward audits

`PROCESS_REWARD_V1_REPORT.md` records the original conservative implementation and why its
model-declared evidence dependency was rejected. `PROCESS_REWARD_V2_REPORT.md` records the first
harness-inferred design. `PROCESS_REWARD_V3_DISTRIBUTED_REPORT.md` is the current allocation design:
outcome responsibility is distributed over harness dependency/state evidence and local errors stay
on their event steps. `PROCESS_REWARD_V4_GROUNDING_REPORT.md` records the column-aware grounding,
multi-handle final-answer support, deterministic completeness gate, and current 316-episode audit.
`SCALE500B_REWARD_AUDIT.md` is the historical v2 positive/negative audit.
`GROUNDING_EXTERNAL_AUDIT.md` records
the independent Flash/Pro grounding review, audit-package corrections, confirmed false edges, and
the current process-RL no-go decision.

`process_reward.py` implements the framework-neutral step allocation
`r_t = C c_t^+ - P c_t^-`. Its offline adapter replays normalized trajectories through the real
harness to rebuild data/value/grounding references, semantic state changes, row counts, empty
results, and error/recovery timing. Final dependency tables and perception edges are inferred from
legal execution parameters and harness-owned outputs; model-authored `evidence` and reasoning text
do not control reward. The fixed-root log row reduction is enabled only for comparable
`condition_filter` and `extreme_value_select` steps.

The current pilot weights live in `configs/process_reward_v3_distributed.json`; v1/v2 configs remain
historical controls. `process_objective.py` implements the per-step weighted policy loss and
requires callers to provide KL values from a genuinely frozen SFT-2 reference whenever `beta > 0`.

Run a reward audit before wiring the scorer into an optimizer:

```bash
.venv/bin/python src/rl/build_process_reward_report.py \
  --input data/trajectories/bird_ds_flash_v4_scale500_rolling4_full_train40.jsonl \
  --output-dir data/rl/bird_scale500_train40_process_reward_v3_distributed \
  --config-json src/rl/configs/process_reward_v3_distributed.json --quiet
```

The report separates structural coverage, deterministic completeness, and precision approval.
`structural_grounding_gate_passed`
requires at least 90% non-empty harness slices and zero ungrounded non-fallback episodes.
`deterministic_grounding_gate_passed` additionally requires every episode's explicit final values
and non-task action literals to have a harness-visible source. `process_reward_ready` also requires
`--grounding-audit-approved`, which must only be set after a separate edge-precision audit.

## Result-only GRPO baseline

`reward.terminal_result_reward` defines the intentionally minimal RL baseline:

```
reward = 1.0  if the final answer's executed denotation equals Spider gold
         0.0  otherwise
```

There is no tool bonus, no error penalty, no length term, and no process/evidence/plan reward.
The harness records protocol, argument-validation, and state-preserving execution errors as
structured `LAST TOOL ERROR` feedback so a rollout may recover before its terminal answer. Every
rejected action spends one shared action-budget slot and remains audit-only.
`build_result_only_candidates.py` exports verified **training**
questions; `select_pilot_tasks.py` then chooses mixed-success examples from pass@4, which gives GRPO
both positive and negative completions without using held-out dev labels for training.

The reproducible server sequence is:

```bash
# 1. Export candidate training questions and collect current-policy pass@4 rollouts.
python src/rl/build_result_only_candidates.py \
  --input data/trajectories/external_rollout_flash_train1k_v3_mixedpass_success881.jsonl \
  --output data/rl/verl_result_only_candidates_300.json --limit 300

# 2. Select only training cases with at least one success and one failure.
python src/rl/select_pilot_tasks.py --input <pass4/all.jsonl> \
  --output data/rl/verl_result_only_selected.jsonl --limit 200 --include-hard

# 3. The Verl adapter materializes Parquet and starts GRPO.
python src/rl/frameworks/verl/prepare_data.py --split train \
  --selection data/rl/verl_result_only_selected.jsonl \
  --output /home/dengyan/tabular_rl_outputs/rl/result_only/train.parquet
bash src/rl/frameworks/verl/run_result_only_grpo.sh
```

Use an independently sampled Spider dev parquet for validation and the standard closed-loop
`rollout.py` for the final held-out evaluation.

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

## Backend choice on the current server

The failed Verl smoke was not caused by the result reward or the table harness. Its historical
token-concatenating adapter is not a supported v2i training entry point: state-only context is
rebuilt per turn and requires a per-turn loss adapter. Current Verl uses a
colocated FSDP actor plus tensor-parallel vLLM rollout engine and dynamically transfers LoRA weights.
On this server, the two RTX 3090s do not support CUDA peer access.  FSDP's required all-gather and
the current vLLM dynamic-LoRA path therefore conflict even though ordinary two-GPU all-reduce works.

For the first usable baseline on this host, prefer a backend that can run **single-GPU QLoRA actor
rollouts** and avoids colocated FSDP/vLLM synchronization:

- **TRL**: the most direct option for a small custom GRPO/PPO-style loop.  It can use
  `ToolUseEnv` directly and run one sampled episode at a time with Transformers generation.
- **OpenRLHF**: useful later when scaling to a machine with more compatible interconnect, but it also
  leans on distributed actor/inference separation and is not the shortest path on these two 3090s.
- **Unsloth/TRL**: a memory-efficient variant of the TRL route if its installed versions support the
  chosen Qwen2.5 model; it is an optimization, not a different experiment.

The first runnable backend is now `frameworks/accelerate/`. It consumes `task_data` records, calls
`ToolUseEnv.apply_model_output`, and assigns only `terminal_result_reward` at episode end. It uses a
single-GPU 4-bit LoRA model and a group-relative REINFORCE update, so it avoids all cross-GPU weight
transfer. Its algorithm intentionally has no PPO clipping or KL term; this makes it the cleanest
comparison point for later reward-function experiments.

```bash
# Server: 4 completions per question, terminal reward in {0, 1}, only LoRA parameters updated.
CUDA_VISIBLE_DEVICES=0 \
  bash src/rl/frameworks/accelerate/run_result_only_group_reinforce.sh \
  --steps 200 --group-size 4 --limit 200 --save-every 25
```

For the full 7B baseline, use the complete 7,000-question Spider train split and launch
`frameworks/accelerate/run_full_result_only_baseline.sh`. It performs one full-train pass with
four sampled episodes per question, then automatically evaluates the final adapter on all 1,034
Spider development examples using the standard rollout runner. Homogeneous reward groups are logged
but correctly skip an optimizer update; the resulting update rate is an important baseline metric.
The full launcher uses a 7,680-token training ceiling on the 24GB card and records a rare
`gradient_oom` group without aborting the remaining full-train pass.

`frameworks/trl/` can be added later using the same environment and task records. The present TRL
installation needs an additional `mergekit` dependency before its `GRPOTrainer` can be imported, and
its stock completion API would still need an interactive rollout adapter.
