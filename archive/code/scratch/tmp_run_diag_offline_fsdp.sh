#!/usr/bin/env bash
set -u

BASE=${BASE:-/home/dengyan/tabular_rl_outputs/qwen3_8b_v26_rl_two_phase_diag_20260902}
POOL=${POOL:-$BASE/pool2_k8}
OUT=${OUT:-$BASE/train2_rows8}
TRAIN_OUT=${TRAIN_OUT:-$OUT/train}
PROMPTS_PER_UPDATE=${PROMPTS_PER_UPDATE:-1}
EXPECTED_RECORDS=${EXPECTED_RECORDS:-1}
TASKS_JSON=${TASKS_JSON:-$POOL/tasks2.jsonl}
TASK_LIMIT=${TASK_LIMIT:-1}
RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_8b_v26_rl_dynamic_micro_20260902}
PROTOCOL=${PROTOCOL:-/home/dengyan/tabular_rl_outputs/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL=${MODEL:-/data2/shared/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER=${ADAPTER:-/data4/dengyan/checkpoints/qwen3_atomic_v26_cumulative_augmented_fresh4ep_4gpu_newgnn_20260827/checkpoint-6380}
MASTER_PORT=${MASTER_PORT:-29618}

mkdir -p "$OUT"
if [[ -e "$OUT/run.started" ]]; then
  echo "refusing existing diagnostic output: $OUT" >&2
  exit 75
fi
date +%s.%N > "$OUT/run.started"
printf 'gpu_pair=5,7\\nmax_rows=%s\\ntoken_budget=%s\\n' "${MAX_ROWS:-8}" "${TOKEN_BUDGET:-24576}" > "$OUT/config.txt"

export CUDA_VISIBLE_DEVICES=5,7
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$PROTOCOL"
export TABLE_RL_TRAINER_SHARDING=fsdp
export TABLE_RL_FSDP_BASE_STORAGE=bf16
export ACCELERATE_MIXED_PRECISION=no
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-LOC}
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_LAUNCH_MODE=GROUP
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTHONPATH="$RUNTIME/src/rl:$PROTOCOL/src/eval:$PROTOCOL/src/harness:$PROTOCOL/src/sft"
unset PYTORCH_CUDA_ALLOC_CONF

echo "start_epoch=$(date +%s.%N)" >> "$OUT/events.log"
(
  while [[ -f "$OUT/trainer.pid" ]] && kill -0 "$(cat "$OUT/trainer.pid")" 2>/dev/null; do
    printf '%s,' "$(date +%s.%N)" >> "$OUT/gpu_trace.csv"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\\n' ';' >> "$OUT/gpu_trace.csv"
    printf '\\n' >> "$OUT/gpu_trace.csv"
    sleep 5
  done
) &
watcher=$!

set +e
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 --master_port "$MASTER_PORT" \
  /tmp/tmp_rl_diag_offline_fsdp_20260902.py \
  --model-path "$MODEL" --adapter-path "$ADAPTER" \
  --examples-json "$TASKS_JSON" --limit "$TASK_LIMIT" --expected-records "$EXPECTED_RECORDS" \
  --output-dir "$TRAIN_OUT" --protocol-runtime-root "$PROTOCOL" \
  --fixed-rollout-pool "$POOL/trajectories.jsonl" \
  --fixed-pool-manifest "$POOL/diagnostic_manifest.json" \
  --reward-mode result-only --result-reward-profile four-level \
  --credit-assignment saam-asymmetric-error --policy-reduction trajectory_token_mean \
  --error-penalty 1.0 --span-balance-alpha 0.5 --train-turns all \
  --optimizer-steps 1 --ppo-iterations 1 --prompts-per-update "$PROMPTS_PER_UPDATE" --group-size 8 \
  --transition-micro-batch-size "${MAX_ROWS:-8}" \
  --transition-micro-batch-tokens "${TOKEN_BUDGET:-24576}" \
  --optimizer-name adamw_torch --learning-rate 4e-7 --weight-decay 0.1 --kl-beta 0 \
  --trainer-sharding fsdp --fsdp-base-storage bf16 \
  --max-agent-steps 30 --max-batch-calls 5 --max-new-tokens 2048 \
  --max-context-tokens 16384 --history-turns 4 --temperature 0.8 --top-p 1.0 \
  --top-k 0 --enable-thinking --save-steps 1 --save-total-limit 1 \
  --seed 20260829 --vllm-host 127.0.0.1 --vllm-port 8251 --vllm-group-port 51401 \
  > "$OUT/train.log" 2>&1 &
trainer=$!
echo "$trainer" > "$OUT/trainer.pid"
wait "$trainer"
code=$?
printf 'exit_code=%s\\nend_epoch=%s\\n' "$code" "$(date +%s.%N)" >> "$OUT/events.log"
rm -f "$OUT/trainer.pid"
kill "$watcher" 2>/dev/null || true
wait "$watcher" 2>/dev/null || true
exit "$code"
