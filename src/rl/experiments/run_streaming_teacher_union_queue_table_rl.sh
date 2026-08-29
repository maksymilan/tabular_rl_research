#!/usr/bin/env bash
# Dedicated-GPU queue: Exp15 train-K4 prepass -> streaming OPD/repair-DPO -> full BIRD-dev greedy.
set -euo pipefail

O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TR=${TRAIN_RUNTIME:-$O/rl_runtime_streaming_teacher_union_20260803}
ER=${EVAL_RUNTIME:-$O/eval_runtime_version36_20260728}
DATA_PROJECT_ROOT=${DATA_PROJECT_ROOT:-/home/dengyan/tabular_rl_project}
ROOT=${RUN_ROOT:-$O/streaming_teacher_union_20260803}
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
SFT2=${SFT2_ADAPTER:-$O/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
EXP15=${EXP15_ADAPTER:-$O/checkpoints/trl-transition-v26-exp15-fixed-prefix-action-dpo-sft2-seed101-20260801/final}
SOURCE_POOL=${SOURCE_POOL:-$O/phase8_dense_stage2_20260802/balanced_mixed120_dense_uniform_seed101}
EXP15_POOL=${EXP15_POOL:-$ROOT/exp15_k4_balanced120}
STATUS=${STATUS:-$O/logs/streaming_teacher_union_queue_table_rl_20260803.status}
RUN_LOG=${RUN_LOG:-$O/logs/streaming_teacher_union_queue_table_rl_20260803.log}
LOCK=${LOCK:-$O/logs/streaming_teacher_union_queue_table_rl_20260803.lock}
GPU_ID=${GPU_ID:-0}
WAIT_FOR_STAGE2=${WAIT_FOR_STAGE2:-0}
FORMAL_TRAIN_DIR=${FORMAL_TRAIN_DIR:-$ROOT/formal60_batch2_repairk4}
REPAIR_GENERATION_BATCH_SIZE=${REPAIR_GENERATION_BATCH_SIZE:-4}
SFT2_SHA=d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e
# PyTorch 2.x on table_rl reads the CUDA-specific spelling (and its OOM
# diagnostics explicitly recommend it).  Keep the newer alias too so the
# frozen launcher behaves consistently across runtime upgrades.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-$PYTORCH_CUDA_ALLOC_CONF}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
state_of() { awk -F '\t' 'NR==1 {print $2}' "$1" 2>/dev/null || printf missing; }
gpu_used() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | sed -n "$((GPU_ID + 1))p"
}
wait_gpu_free() {
  while true; do
    used=$(gpu_used)
    if [[ -n "$used" && "$used" -le 512 ]]; then return 0; fi
    set_status waiting_gpu "gpu=$GPU_ID mib=${used:-unknown} stage=$1"
    sleep 30
  done
}
on_exit() {
  code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then
    set_status failed "exit=$code see=$RUN_LOG"
  fi
  exit "$code"
}
trap on_exit EXIT

mkdir -p "$O/logs" "$ROOT"
exec 9>"$LOCK"
flock -n 9 || exit 0
exec >>"$RUN_LOG" 2>&1
cd "$TR"
[[ "$(grep -c '^PROTOCOL_VERSION = "version26"' src/sft/protocol.py)" -eq 1 ]]
[[ -f src/rl/action_dpo/generate_exact_prefix_branches.py ]]
[[ "$(sha256sum "$SFT2/adapter_model.safetensors" | awk '{print $1}')" == "$SFT2_SHA" ]]
[[ -f "$EXP15/adapter_model.safetensors" ]]
[[ -f "$SOURCE_POOL/tasks.jsonl" && -f "$SOURCE_POOL/trajectories.jsonl" ]]

# When explicitly sharing GPU1 with Stage2, do not take the brief gap between
# seed202 training and its full evaluation.  A dedicated GPU0 lane skips this.
SEED202_DIAG=$O/logs/stage2_exp18_dense_uniform_full_response_scale120_seed202.diagnostics.status
if [[ "$WAIT_FOR_STAGE2" == 1 ]]; then
  while [[ "$(state_of "$SEED202_DIAG")" != complete ]]; do
    set_status waiting_stage2 "seed202_diagnostics=$(state_of "$SEED202_DIAG") gpu=$GPU_ID mib=$(gpu_used)"
    sleep 60
  done
fi
wait_gpu_free initial

if [[ ! -f "$EXP15_POOL/manifest.pending.json" ]]; then
  set_status teacher_prepass "teacher=Exp15 tasks=120 K4 gpu=$GPU_ID protocol=version26 scheduler=dynamic question_window=8"
  CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 \
  TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda \
    "$PY" src/rl/fixed_pool/generate_fixed_rollout_pool.py \
      --model-path "$MODEL" --adapter-path "$EXP15" \
      --tasks "$SOURCE_POOL/tasks.jsonl" --output-dir "$EXP15_POOL" \
      --group-size 4 --temperature 0.7 --top-p 0.95 --max-steps 30 \
      --max-new-tokens 1024 --max-context-tokens 8192 --history-turns 4 \
      --seed 1515 --gpu-memory-utilization 0.82 \
      --scheduler dynamic --question-window 8 --task-batch-size 4
fi
"$PY" - "$EXP15_POOL/manifest.pending.json" "$EXP15_POOL/trajectories.jsonl" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); rows=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
assert p["protocol_version"]=="version26"
assert p["tasks"]==120 and p["group_size"]==4 and p["trajectories"]==480
assert len(rows)==480 and [r["sequence"] for r in rows]==list(range(480))
assert all(r["environment"]["dataset_split"]=="train" for r in rows)
PY

ELIGIBILITY=$ROOT/teacher_eligibility_balanced120.jsonl
ELIGIBILITY_MANIFEST=$ROOT/teacher_eligibility_balanced120.manifest.json
if [[ ! -f "$ELIGIBILITY_MANIFEST" ]]; then
  set_status freezing_teacher_routes "tasks=120 SFT2_K4+Exp15_K4 dense_threshold=2"
  "$PY" src/rl/distillation/build_teacher_eligibility.py \
    --sft2-rollouts "$SOURCE_POOL/trajectories.jsonl" \
    --exp15-rollouts "$EXP15_POOL/trajectories.jsonl" \
    --output "$ELIGIBILITY" --manifest "$ELIGIBILITY_MANIFEST" \
    --expected-k 4 --minimum-dense-correct 2 --protocol-version version26 \
    --order-seed 101
fi

# A two-question hardware/integration smoke is always a fresh SFT2 initialization.
SMOKE=$ROOT/smoke2_batch2_repairk2
if [[ "$("$PY" - "$SMOKE/status.json" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get("state","missing"))
except Exception: print("missing")
PY
)" != complete ]]; then
  [[ ! -e "$SMOKE" ]] || { set_status blocked "incomplete smoke artifact requires audit: $SMOKE"; exit 4; }
  wait_gpu_free smoke
  set_status hardware_smoke "gpu=$GPU_ID tasks=2 batch=2 repairK=2"
  CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 \
  TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda \
    "$PY" src/rl/distillation/train_streaming_teacher_union.py \
      --project-root "$DATA_PROJECT_ROOT" --eligibility "$ELIGIBILITY" \
      --eligibility-manifest "$ELIGIBILITY_MANIFEST" --model-path "$MODEL" \
      --sft2-adapter "$SFT2" --exp15-adapter "$EXP15" --output-dir "$SMOKE" \
      --limit 2 --question-batch-size 2 --repair-k 2 --seed 101 \
      --experiment-name streaming_teacher_union_hardware_smoke2
fi

TRAIN=$FORMAL_TRAIN_DIR
if [[ "$("$PY" - "$TRAIN/status.json" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get("state","missing"))
except Exception: print("missing")
PY
)" != complete ]]; then
  RESUME_ARGS=()
  if [[ -e "$TRAIN" ]]; then
    LATEST_CHECKPOINT=$("$PY" - "$TRAIN/checkpoints" <<'PY'
import sys
from pathlib import Path
from src.rl.distillation.streaming_checkpoint import latest_complete_checkpoint
value = latest_complete_checkpoint(Path(sys.argv[1]))
print(value or "")
PY
)
    [[ -n "$LATEST_CHECKPOINT" ]] || {
      set_status blocked "incomplete formal artifact has no complete checkpoint: $TRAIN"
      exit 4
    }
    RESUME_ARGS=(--resume-latest)
    set_status formal_resuming "gpu=$GPU_ID checkpoint=$LATEST_CHECKPOINT"
  fi
  wait_gpu_free formal_train
  if [[ ${#RESUME_ARGS[@]} -eq 0 ]]; then
    set_status formal_training "gpu=$GPU_ID tasks=60 batch=2 online_student=1 repairK=4"
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 \
  TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda \
    "$PY" src/rl/distillation/train_streaming_teacher_union.py \
      --project-root "$DATA_PROJECT_ROOT" --eligibility "$ELIGIBILITY" \
      --eligibility-manifest "$ELIGIBILITY_MANIFEST" --model-path "$MODEL" \
      --sft2-adapter "$SFT2" --exp15-adapter "$EXP15" --output-dir "$TRAIN" \
      --limit 60 --question-batch-size 2 --repair-k 4 --seed 101 \
      --repair-generation-batch-size "$REPAIR_GENERATION_BATCH_SIZE" \
      --checkpoint-every-batches 5 --checkpoint-total-limit 3 \
      "${RESUME_ARGS[@]}" \
      --experiment-name streaming_sft2_exp15_opd_repair_dpo_60_batch2
fi

FINAL_ADAPTER=$("$PY" - "$TRAIN/run_manifest.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p["status"]=="complete"; print(p["final_adapter"])
PY
)
[[ -f "$FINAL_ADAPTER/adapter_model.safetensors" ]]

RESULT=$ER/data/results/streaming_sft2_exp15_opd_repair_dpo60_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260803
EVAL_STATUS=$O/logs/streaming_sft2_exp15_opd_repair_dpo60.full_dev_greedy.status
set_status full_dev_greedy "gpu=$GPU_ID questions=1534 temperature=0 top_p=1 concurrency=24"
ADAPTER="$FINAL_ADAPTER" \
SERVED_MODEL=streaming-sft2-exp15-opd-repair-dpo60-full-dev-greedy \
RESULT_DIR="$RESULT" STATUS="$EVAL_STATUS" \
RUN_LOG=$O/logs/streaming_sft2_exp15_opd_repair_dpo60.full_dev_greedy.log \
PORT=18193 EVAL_GPU_ID="$GPU_ID" RUNTIME="$ER" OUTPUT_ROOT="$O" \
  bash "$TR/src/rl/experiments/run_full_dev_greedy_table_rl.sh"
[[ "$(state_of "$EVAL_STATUS")" == complete ]]

SUMMARY=$ROOT/full_dev_summary.json
if [[ ! -f "$SUMMARY" ]]; then
  "$PY" src/rl/diagnostics/analyze_evaluation_results.py \
    --examples "$ER/data/eval_inputs/bird_dev_20240627.jsonl" \
    --arm "candidate=$RESULT/all.jsonl" \
    --arm "sft2=$ER/data/results/sft2_version36_dev1534_greedy_t0_logprobs20_20260801/all.jsonl" \
    --arm "exp14=$ER/data/results/stage1_exp14_fixed_process_rank_action_mean_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260802/all.jsonl" \
    --arm "exp15=$ER/data/results/stage1_exp15_fixed_prefix_action_dpo_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260802/all.jsonl" \
    --compare candidate:sft2 --compare candidate:exp14 --compare candidate:exp15 \
    --expected-count 1534 --protocol-version version36 \
    --temperature 0 --top-p 1 --denotation-comparison bird-set \
    --output "$SUMMARY"
fi
set_status complete "teacher_prepass+formal60+full_dev complete summary=$SUMMARY K4=not_run"
