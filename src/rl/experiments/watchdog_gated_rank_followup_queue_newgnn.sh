#!/usr/bin/env bash
# Shared-server watchdog; it never touches occupied GPUs or other users' processes.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-"$OUTPUT_ROOT/rl_runtime_process_gated_v2_20260730"}
QUEUE_SCRIPT=${QUEUE_SCRIPT:-"$RUNTIME/src/rl/experiments/run_gated_rank_followup_queue_newgnn.sh"}
QUEUE_STATUS=${QUEUE_STATUS:-"$OUTPUT_ROOT/logs/gated_rank_followup_queue_newgnn_20260731.status"}
READY_MARKER=${READY_MARKER:-"$OUTPUT_ROOT/logs/newgnn_rank_runtime_ready_20260731"}
SMOKE_MARKER=${SMOKE_MARKER:-"$OUTPUT_ROOT/logs/newgnn_sft2_runtime_smoke_20260731.passed"}
SFT2_ADAPTER_PATH=${SFT2_ADAPTER_PATH:-"$OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682"}
WATCHDOG_LOG=${WATCHDOG_LOG:-"$OUTPUT_ROOT/logs/gated_rank_followup_watchdog_newgnn_20260731.log"}
WATCHDOG_LATEST=${WATCHDOG_LATEST:-"$OUTPUT_ROOT/logs/gated_rank_followup_watchdog_newgnn_20260731.latest"}
WATCHDOG_LOCK=${WATCHDOG_LOCK:-"$OUTPUT_ROOT/logs/gated_rank_followup_watchdog_newgnn_20260731.v2.lock"}
IDLE_MARKER=${IDLE_MARKER:-"$OUTPUT_ROOT/logs/gated_rank_followup_watchdog_newgnn_20260731.idle_since"}
QUEUE_SUPERVISOR_LOG=${QUEUE_SUPERVISOR_LOG:-"$OUTPUT_ROOT/logs/gated_rank_followup_queue_newgnn_20260731.supervisor.log"}
TRAIN_GPU_ID=${TRAIN_GPU_ID:-6}
VLLM_GPU_ID=${VLLM_GPU_ID:-7}
GPU_FREE_THRESHOLD_MIB=${GPU_FREE_THRESHOLD_MIB:-512}
STABLE_FREE_SECONDS=${STABLE_FREE_SECONDS:-300}

ARTIFACTS=(
  trl-transition-v26-phase4-process-rank-action-only-lr1e6-23x4-gated-v2-newgnn-20260731
  trl-transition-v26-phase5-process-rank-conservative-lr1e6-23x4-gated-v2-newgnn-20260731
)

mkdir -p "$OUTPUT_ROOT/logs"
exec 9>"$WATCHDOG_LOCK"
if ! flock -n 9; then exit 0; fi

state_of() {
  if [[ -f "$1" ]]; then awk -F '\t' 'NR==1 {print $2}' "$1"; else printf 'missing\n'; fi
}
status_line() {
  if [[ -f "$1" ]]; then tr '\n' ' ' <"$1"; else printf 'missing'; fi
}
line_count() {
  if [[ -f "$1" ]]; then wc -l <"$1" | tr -d '[:space:]'; else printf '0\n'; fi
}
gpu_memory() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    sed -n "$(( $1 + 1 ))p" | tr -d '[:space:]'
}
all_complete() {
  local artifact
  for artifact in "${ARTIFACTS[@]}"; do
    [[ "$(state_of "$OUTPUT_ROOT/logs/${artifact}.train.status")" == complete ]] || return 1
    [[ "$(state_of "$OUTPUT_ROOT/logs/${artifact}.eval.status")" == complete ]] || return 1
  done
}
verify_launch_contract() {
  /home/dengyan/miniconda3/envs/trl-table/bin/python - "$RUNTIME" "$READY_MARKER" <<'PY'
import json
import sys
from pathlib import Path

import yaml

runtime = Path(sys.argv[1])
ready = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
assert ready["schema_version"] == "newgnn-sft2-rank-runtime-ready-v1"
assert ready["sft2_adapter"].endswith("checkpoint-1682")
assert ready["sft2_adapter_sha256"] == "d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e"
assert ready["counterfactual_manifest_sha256"] == "5624c75c6ebbca556f1e0946069fdb95e923dc95d2d7692c8d929bedebd9568b"
expected = {
    "phase4_process_rank_action_only.yaml": "full_trajectory",
    "phase5_process_rank_conservative.yaml": "conservative_legal",
}
for name, scope in expected.items():
    config = yaml.safe_load(
        (runtime / "src/rl/configs/experiments" / name).read_text(encoding="utf-8")
    )
    reward = json.loads(
        (runtime / config["process_reward_config"]).read_text(encoding="utf-8")
    )
    assert config["trainable_part"] == "all"
    assert config["rank_loss"] == {
        "enabled": True,
        "coefficient": 0.5,
        "beta": 0.1,
        "score_tokens": "tool_only",
        "update_scope": scope,
    }
    assert reward["w_back_slice"] == 0.0
    assert reward.get("normalize_positive", True) is True
PY
}

queue_script_name=${QUEUE_SCRIPT##*/}
queue_pids=$(
  ps -u "$(id -un)" -o pid=,args= |
    awk -v self="$$" -v name="$queue_script_name" '
      $1 != self && ($2 == "bash" || $2 ~ /\/bash$/) &&
      $3 ~ ("(^|/)" name "$") { print $1 }
    ' | tr '\n' ' ' || true
)
train_gpu_memory=$(gpu_memory "$TRAIN_GPU_ID")
vllm_gpu_memory=$(gpu_memory "$VLLM_GPU_ID")
gpu_lines=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits)
now=$(date +%s)

action=observe
reason=unknown
if [[ "$(state_of "$QUEUE_STATUS")" == complete ]] && all_complete; then
  action=stop
  reason=followup_complete
elif [[ -n "$queue_pids" ]]; then
  reason=queue_alive
elif [[ "$(state_of "$QUEUE_STATUS")" == failed ]]; then
  action=stop
  reason=failed_requires_audit
elif [[ ! -f "$READY_MARKER" ]] || \
     [[ ! -f "$SMOKE_MARKER" ]] || \
     [[ ! -f "$SFT2_ADAPTER_PATH/adapter_model.safetensors" ]]; then
  action=wait
  reason=sft2_runtime_not_ready
elif [[ "$train_gpu_memory" -gt "$GPU_FREE_THRESHOLD_MIB" || \
        "$vllm_gpu_memory" -gt "$GPU_FREE_THRESHOLD_MIB" ]]; then
  printf '0\n' >"$IDLE_MARKER"
  action=wait
  reason="candidate_gpus_occupied gpu${TRAIN_GPU_ID}=${train_gpu_memory} gpu${VLLM_GPU_ID}=${vllm_gpu_memory}"
else
  idle_since=$(cat "$IDLE_MARKER" 2>/dev/null || printf '0')
  if [[ ! "$idle_since" =~ ^[0-9]+$ ]] || [[ "$idle_since" -eq 0 ]]; then
    printf '%s\n' "$now" >"$IDLE_MARKER"
    action=wait
    reason=recorded_first_idle_observation
  elif [[ $((now - idle_since)) -lt "$STABLE_FREE_SECONDS" ]]; then
    action=wait
    reason="waiting_stable_idle seconds=$((now - idle_since))/$STABLE_FREE_SECONDS"
  else
    verify_launch_contract
    cd "$RUNTIME"
    nohup setsid bash "$QUEUE_SCRIPT" 9>&- </dev/null >>"$QUEUE_SUPERVISOR_LOG" 2>&1 &
    queue_pid=$!
    action=start
    reason="stable_idle_and_ready_pid=$queue_pid"
  fi
fi

temp_path=$(mktemp "$OUTPUT_ROOT/logs/.gated-rank-newgnn-watchdog.XXXXXX")
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'action=%s reason=%s\n' "$action" "$reason"
  printf 'queue_status=%s\n' "$(status_line "$QUEUE_STATUS")"
  printf 'queue_pids=%s ready_marker=%s\n' "${queue_pids:-none}" "$([[ -f "$READY_MARKER" ]] && printf yes || printf no)"
  printf 'candidate_gpus=%s,%s threshold_mib=%s stable_seconds=%s\n' \
    "$TRAIN_GPU_ID" "$VLLM_GPU_ID" "$GPU_FREE_THRESHOLD_MIB" "$STABLE_FREE_SECONDS"
  printf 'gpu=index,memory_used_mib,memory_total_mib,utilization_percent,power_watts\n%s\n' "$gpu_lines"
  for artifact in "${ARTIFACTS[@]}"; do
    checkpoint="$OUTPUT_ROOT/checkpoints/$artifact"
    result_dir="$OUTPUT_ROOT/eval_runtime_version36_20260728/data/results/${artifact}_version36_equal300_seed20260729_passk4_t07_p095_logprobs20_bird_set"
    latest_checkpoint=$(
      if [[ -d "$checkpoint" ]]; then
        find "$checkpoint" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null |
          sort -V | tail -n 1
      fi
    )
    printf 'experiment=%s train_status=%s eval_status=%s rollouts=%s checkpoint=%s eval_rows=%s\n' \
      "$artifact" \
      "$(status_line "$OUTPUT_ROOT/logs/${artifact}.train.status")" \
      "$(status_line "$OUTPUT_ROOT/logs/${artifact}.eval.status")" \
      "$(line_count "$checkpoint/rollouts.jsonl")" \
      "${latest_checkpoint:-none}" \
      "$(line_count "$result_dir/all.jsonl")"
  done
} >"$temp_path"
mv "$temp_path" "$WATCHDOG_LATEST"
{
  printf '\n===== watchdog %s =====\n' "$(date --iso-8601=seconds)"
  cat "$WATCHDOG_LATEST"
} >>"$WATCHDOG_LOG"
