#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${GPU_ID:-1}
RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/rl_runtime_exp15_branch_dynamic_20260803}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs/exp15_exact_prefix_12h_dynamic_20260803}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
ADAPTER_PATH=${ADAPTER_PATH:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
SOURCE_DIR=${SOURCE_DIR:-/home/dengyan/tabular_rl_outputs/phase8_controlled_20260801/balanced_mixed60_rank_seed101}
SOURCE_POOL=$SOURCE_DIR/validated_trajectories.jsonl
SOURCE_MANIFEST=$SOURCE_DIR/manifest.json
STATUS=$OUTPUT_ROOT/pipeline.status
RUN_LOG=$OUTPUT_ROOT/pipeline.log
LOCK=$OUTPUT_ROOT/pipeline.lock
SMOKE_DIR=$OUTPUT_ROOT/smoke_dynamic2
DATA_DIR=$OUTPUT_ROOT/data
TIME_BUDGET=${TIME_BUDGET:-12h}

EXPECTED_SOURCE_SHA=bae41e0951b649d7fd94596c8d303076aca0d1586d34bb25e0690365eabc7d97
EXPECTED_ADAPTER_SHA=d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e

if [[ "$GPU_ID" != 1 ]]; then
  printf 'this launch is authorized only for table_rl GPU 1\n' >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
exec 9>"$LOCK"
if ! flock -n 9; then
  printf 'another Exp15 12h dynamic pipeline holds %s\n' "$LOCK" >&2
  exit 3
fi

set_status() {
  printf 'state=%s detail=%s time=%s\n' "$1" "$2" "$(date -Is)" >"$STATUS"
}

on_exit() {
  exit_code=$?
  trap - EXIT
  if [[ $exit_code -ne 0 ]]; then
    set_status failed "exit_code=$exit_code log=$RUN_LOG"
  fi
  exit "$exit_code"
}
trap on_exit EXIT
exec >>"$RUN_LOG" 2>&1

test -f "$RUNTIME/src/rl/action_dpo/generate_exact_prefix_branches.py"
test -f "$SOURCE_POOL"
test -f "$SOURCE_MANIFEST"
test -f "$ADAPTER_PATH/adapter_model.safetensors"
test "$(sha256sum "$SOURCE_POOL" | awk '{print $1}')" = "$EXPECTED_SOURCE_SHA"
test "$(sha256sum "$ADAPTER_PATH/adapter_model.safetensors" | awk '{print $1}')" = "$EXPECTED_ADAPTER_SHA"

stable=0
while [[ "$stable" -lt 10 ]]; do
  used=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  pids=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits \
    | awk '/^[0-9]+$/ {print}' | paste -sd, - || true)
  if [[ -n "$used" && "$used" -le 512 && -z "$pids" ]]; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  set_status waiting_gpu_stability "gpu=$GPU_ID samples=$stable/10 mib=${used:-unknown} pids=${pids:-none}"
  if [[ "$stable" -lt 10 ]]; then sleep 30; fi
done

cd "$RUNTIME"
COMMON_ARGS=(
  --source-pool "$SOURCE_POOL"
  --source-manifest "$SOURCE_MANIFEST"
  --model-path "$MODEL_PATH"
  --adapter-path "$ADAPTER_PATH"
  --candidate-count 4
  --candidate-draws 8
  --continuation-count 2
  --max-pairs-per-anchor 2
  --temperature 0.7
  --top-p 0.95
  --max-steps 30
  --max-new-tokens 1024
  --max-context-tokens 8192
  --history-turns 4
  --seed 30303
  --gpu-memory-utilization 0.82
  --scheduler dynamic
  --max-num-batched-tokens 8192
  --max-num-seqs 32
)

set_status smoke "gpu=$GPU_ID anchors=2 scheduler=dynamic window=2"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 \
PYTHONPATH=src/rl:src/eval:src/harness:src/sft \
  "$PYTHON_BIN" src/rl/action_dpo/generate_exact_prefix_branches.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "$SMOKE_DIR" \
    --limit-trajectories 2 \
    --anchors-per-trajectory 1 \
    --anchor-window 2
test -f "$SMOKE_DIR/manifest.json"

while true; do
  used=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ -n "$used" && "$used" -le 512 ]] && break
  set_status waiting_after_smoke "gpu=$GPU_ID mib=${used:-unknown}"
  sleep 15
done

set_status generating "gpu=$GPU_ID budget=$TIME_BUDGET scheduler=dynamic anchor_window=8 max_num_seqs=32"
set +e
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 \
PYTHONPATH=src/rl:src/eval:src/harness:src/sft \
  timeout --signal=TERM --kill-after=120s "$TIME_BUDGET" \
  "$PYTHON_BIN" src/rl/action_dpo/generate_exact_prefix_branches.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "$DATA_DIR" \
    --limit-trajectories 0 \
    --anchors-per-trajectory 30 \
    --anchor-window 8 \
    --no-finalize
generation_code=$?
set -e
if [[ "$generation_code" -ne 0 && "$generation_code" -ne 124 && "$generation_code" -ne 143 ]]; then
  set_status failed "generation_exit=$generation_code"
  exit "$generation_code"
fi

set_status finalizing_partial "generation_exit=$generation_code"
PYTHONPATH=src/rl:src/eval:src/harness:src/sft \
  "$PYTHON_BIN" src/rl/action_dpo/generate_exact_prefix_branches.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "$DATA_DIR" \
    --limit-trajectories 0 \
    --anchors-per-trajectory 30 \
    --anchor-window 8 \
    --finalize-partial-only

summary=$(
  "$PYTHON_BIN" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); print("anchors=%s/%s pairs=%s strict=%s questions=%s" % (p["attempted_anchors"],p["planned_anchors"],p["retained_pairs"],p["pair_views"]["online_consistent"]["pairs"],p["pair_views"]["online_consistent"]["questions"]))' \
    "$DATA_DIR/manifest.json"
)
set_status complete "$summary generation_exit=$generation_code"
trap - EXIT
