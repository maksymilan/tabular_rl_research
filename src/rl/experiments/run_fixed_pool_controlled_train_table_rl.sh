#!/usr/bin/env bash
# Train one Exp12-Exp14 candidate from SFT2 using the exact frozen 60xK4 pool.
set -euo pipefail
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${TRAIN_RUNTIME:-$O/rl_runtime_rank_score_v3_20260731}
POOL=${POOL_DIR:-$O/phase8_controlled_20260801/fixed_pool_60_seed101}
MODEL=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
SFT2=${ADAPTER_PATH:-$O/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
GPU_ID=${GPU_ID:-1}
TRAIN_SEED=${TRAIN_SEED:-101}
EXPECTED_POOL_TASKS=${EXPECTED_POOL_TASKS:-60}
EXPECTED_POOL_TRAJECTORIES=${EXPECTED_POOL_TRAJECTORIES:-$((EXPECTED_POOL_TASKS * 4))}
EXPECTED_POOL_STATUS=${EXPECTED_POOL_STATUS:-}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
TRANSITION_MICRO_BATCH_SIZE=${TRANSITION_MICRO_BATCH_SIZE:-}
for name in EXPERIMENT_NAME EXPERIMENT_CONFIG ARTIFACT STATUS RUN_LOG; do
  [[ -n "${!name:-}" ]] || { printf 'missing %s\n' "$name" >&2; exit 2; }
done
timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
on_exit() { local c=$?; trap - EXIT; if [[ "$c" -ne 0 ]]; then set_status failed "exit=$c see=$RUN_LOG"; fi; exit "$c"; }
trap on_exit EXIT
mkdir -p "$O/logs"
exec 9>"$O/logs/${ARTIFACT}.train.lock"
if ! flock -n 9; then exit 0; fi
exec >>"$RUN_LOG" 2>&1
output="$O/checkpoints/$ARTIFACT"
if [[ -f "$output/final/adapter_model.safetensors" && -f "$output/run_manifest.json" ]]; then
  set_status complete "already trained artifact=$ARTIFACT"; exit 0
fi
train_args=()
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  case "$RESUME_FROM_CHECKPOINT" in
    "$output"/checkpoint-*) ;;
    *) set_status failed "resume checkpoint is outside artifact path=$RESUME_FROM_CHECKPOINT"; exit 4 ;;
  esac
  for required in adapter_model.safetensors trainer_state.json optimizer.pt scheduler.pt; do
    [[ -f "$RESUME_FROM_CHECKPOINT/$required" ]] || {
      set_status failed "resume checkpoint missing $required path=$RESUME_FROM_CHECKPOINT"; exit 4;
    }
  done
  train_args+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
elif [[ -e "$output" ]]; then
  set_status failed "partial output exists; audit before resume path=$output"; exit 4
fi
if [[ -n "$TRANSITION_MICRO_BATCH_SIZE" ]]; then
  [[ "$TRANSITION_MICRO_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
    set_status failed "invalid transition micro batch size=$TRANSITION_MICRO_BATCH_SIZE"; exit 4;
  }
  train_args+=(--transition-micro-batch-size "$TRANSITION_MICRO_BATCH_SIZE")
fi
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((GPU_ID + 1))p")
  [[ -n "$used" && "$used" -le 512 ]] && break
  set_status waiting_gpu "gpu=$GPU_ID mib=${used:-unknown}"; sleep 30
done
set_status training "experiment=$EXPERIMENT_NAME gpu=$GPU_ID fixed_pool=$EXPECTED_POOL_TRAJECTORIES train_seed=$TRAIN_SEED resume=${RESUME_FROM_CHECKPOINT:-none} transition_micro_batch=${TRANSITION_MICRO_BATCH_SIZE:-default}"
cd "$R"
CUDA_VISIBLE_DEVICES="$GPU_ID" PROJECT_DIR="$R" PYTHON="$PY" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
MODEL_PATH="$MODEL" ADAPTER_PATH="$SFT2" EXAMPLES_JSON="$POOL/tasks.jsonl" \
OUTPUT_DIR="$output" EXPERIMENT_CONFIG="$EXPERIMENT_CONFIG" \
FIXED_ROLLOUT_POOL="$POOL/validated_trajectories.jsonl" \
FIXED_POOL_MANIFEST="$POOL/manifest.json" \
  bash src/rl/frameworks/trl/run_atomic_transition_grpo.sh \
    --seed "$TRAIN_SEED" "${train_args[@]}"
"$PY" - "$output/run_manifest.json" "$EXPERIMENT_NAME" \
  "$EXPECTED_POOL_STATUS" "$EXPECTED_POOL_TASKS" "$EXPECTED_POOL_TRAJECTORIES" \
  "$TRAIN_SEED" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); expected=sys.argv[2]
assert p["experiment_config"]["experiment_name"] == expected
expected_status = sys.argv[3] or {
    "exp13_fixed_rank_only_action_mean": "frozen_rank_ready",
    "exp12_fixed_process_only": "frozen_process_screened",
    "exp14_fixed_process_rank_action_mean": "frozen_process_screened",
    "exp16_dense_uniform_full_response": "frozen_dense_ready",
    "exp17_dense_strategic_full_response": "frozen_dense_ready",
}[expected]
expected_tasks=int(sys.argv[4]); expected_trajectories=int(sys.argv[5]); expected_seed=int(sys.argv[6])
assert p["fixed_pool_status"] == expected_status
assert p["fixed_pool_reuse_contract"]["online_resampling_forbidden"] is True
assert p["records"] == expected_tasks and p["group_size"] == 4
assert p["optimizer_steps"] == expected_tasks
assert expected_trajectories == expected_tasks * p["group_size"]
assert p["seed"] == expected_seed
assert p["adapter_path"].endswith("checkpoint-1682")
PY
set_status complete "experiment=$EXPERIMENT_NAME artifact=$output/final"
