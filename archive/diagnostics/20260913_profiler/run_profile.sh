#!/usr/bin/env bash
set -Eeuo pipefail
RUN_ROOT=${RUN_ROOT:?}
PROJECT_DIR=${PROJECT_DIR:?}
PYTHON=/home/dengyan/miniconda3/envs/trl-table/bin/python
source "$PROJECT_DIR/src/rl/frameworks/launcher/launch_common.sh"
exec 9>/tmp/atomic-v26-gpu-0.lock
flock -n 9 || exit 75
exec 8>/tmp/table-rl-perf-audit-gpu-0.lock
flock -n 8 || exit 75
gpu_idle 0 || exit 75
mkdir -p "$RUN_ROOT"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv > "$RUN_ROOT/gpus_before.csv"
sha256sum "$RUN_ROOT/profile_4b_forward_backward.py" > "$RUN_ROOT/implementation.sha256"
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN_ROOT/exit_status"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > "$RUN_ROOT/gpus_after.csv"' EXIT
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True "$PYTHON" \
  "$RUN_ROOT/profile_4b_forward_backward.py" \
  --model /home/dengyan/models/Qwen3-4B-TrustSQL-baseline \
  --adapter /home/dengyan/tabular_rl_outputs/checkpoints/qwen3-4b-atomic-v26-cumulative-fresh4ep-table-rl-formal-b1/checkpoint-6380 \
  --output "$RUN_ROOT" --lengths "${LENGTHS:-2048,4096,8192}" --storage "${STORAGE:-4bit}"
