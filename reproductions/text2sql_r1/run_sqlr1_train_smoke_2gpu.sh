#!/usr/bin/env bash
set -euo pipefail

# Infrastructure-only one-step GRPO smoke test for two 24 GB GPUs.
# This is intentionally smaller than the paper's 8x80 GB, n=8 configuration.

REPRO_ROOT="${REPRO_ROOT:-/home/dengyan/tabular_rl_outputs/text2sql_reproduction}"
SQLR1_CODE="${SQLR1_CODE:-$REPRO_ROOT/third_party/SQL-R1}"
# SQL-R1's vendored verl only supports vLLM through 0.6.3. Keep the
# compatibility stack isolated from the server's current vLLM 0.19 runtime.
TRAIN_PYTHON="${TRAIN_PYTHON:-$REPRO_ROOT/envs/sqlr1_train_py311/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/dengyan/models/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPRO_ROOT/results/sqlr1/train_smoke_2gpu}"
TRAIN_FILE="${TRAIN_FILE:-$REPRO_ROOT/data/sqlr1_train_smoke_short2.parquet}"
VAL_FILE="${VAL_FILE:-$SQLR1_CODE/example_data/test.parquet}"

mkdir -p "$OUTPUT_DIR"
cd "$SQLR1_CODE"
test -x "$TRAIN_PYTHON"
test -d "$MODEL_PATH"
test -f "$TRAIN_FILE"
test -f "$VAL_FILE"

export CUDA_VISIBLE_DEVICES=0,1
# The two 3090s on table_rl do not expose CUDA peer access to one another.
# NCCL must use its host-memory transport for this topology.
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1

"$TRAIN_PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=2 \
  data.val_batch_size=2 \
  data.max_prompt_length=640 \
  data.max_response_length=128 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=3e-7 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.grad_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.temperature=1.1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name=SQL-R1-reproduction \
  trainer.experiment_name=train-smoke-2gpu \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
  trainer.default_hdfs_dir=null \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  +trainer.val_before_train=false 2>&1 | tee "$OUTPUT_DIR/train.log"
