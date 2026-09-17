#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
SCRIPT=${SCRIPT:?set SCRIPT}
ROLLOUTS_PATH=${ROLLOUTS:?set ROLLOUTS}
MODEL=${MODEL:?set MODEL}
ADAPTER=${ADAPTER:?set ADAPTER}
GPU=${GPU:-0}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}

mkdir -p "$RUN_ROOT/logs"
exec 9>"$RUN_ROOT/gate.lock"
flock -n 9 || exit 75
exec 8>"/tmp/table-rl-fixed-transition-gpu-$GPU.lock"
flock -n 8 || exit 75

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  printf '%s\n' 'GPU is not process-level idle; refusing to start.' >"$RUN_ROOT/logs/preflight_error"
  exit 75
fi

nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv >"$RUN_ROOT/logs/gpus_before.csv"
printf '%s\n' "$(sha256sum "$SCRIPT")" >"$RUN_ROOT/implementation.sha256"
printf '%s\n' "$(sha256sum "$ROLLOUTS_PATH")" >"$RUN_ROOT/rollouts.sha256"
trap 'rc=$?; printf "%s\n" "$rc" >"$RUN_ROOT/exit_status"; nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv >"$RUN_ROOT/logs/gpus_after.csv"' EXIT

overall_rc=0

run_arm() {
  local name=$1
  shift
  local arm_root="$RUN_ROOT/$name"
  mkdir -p "$arm_root"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" "$SCRIPT" \
      --rollouts "$ROLLOUTS_PATH" --model "$MODEL" --adapter "$ADAPTER" \
      --mode "$name" --output "$arm_root" --device cuda:0 "$@" \
      >"$arm_root/run.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "$rc" >"$arm_root/exit_status"
  if [[ "$rc" -ne 0 ]]; then
    overall_rc=1
  fi
}

run_arm 4bit
run_arm bf16

FUSION_ROOT="$RUN_ROOT/4bit_fusion"
mkdir -p "$FUSION_ROOT"
set +e
CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" "$SCRIPT" \
    --rollouts "$ROLLOUTS_PATH" --model "$MODEL" --adapter "$ADAPTER" \
    --mode 4bit --output "$FUSION_ROOT" --device cuda:0 --fusion \
    >"$FUSION_ROOT/run.log" 2>&1
fusion_rc=$?
set -e
printf '%s\n' "$fusion_rc" >"$FUSION_ROOT/exit_status"
if [[ "$fusion_rc" -ne 0 ]]; then
  overall_rc=1
fi

if [[ "$overall_rc" -eq 0 ]]; then
  printf '%s\n' complete >"$RUN_ROOT/status"
else
  printf '%s\n' partial_failed >"$RUN_ROOT/status"
fi
exit "$overall_rc"
