#!/usr/bin/env bash
# Read-only 30-minute monitor for the sequential RL experiment queue on table_rl.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-1800}
STALL_SECONDS=${STALL_SECONDS:-2700}
MODE=${1:-daemon}

MONITOR_LOG=${MONITOR_LOG:-"$OUTPUT_ROOT/logs/rl_experiment_monitor_30m.log"}
LATEST_STATUS=${LATEST_STATUS:-"$OUTPUT_ROOT/logs/rl_experiment_monitor_latest.txt"}
LOCK_FILE=${LOCK_FILE:-"$OUTPUT_ROOT/logs/rl_experiment_monitor_30m.lock"}
PID_FILE=${PID_FILE:-"$OUTPUT_ROOT/logs/rl_experiment_monitor_30m.pid"}

TRAIN_OUTPUT=${TRAIN_OUTPUT:-"$OUTPUT_ROOT/checkpoints/trl-transition-v26-phase1-result-only-lr1e6-23x4-20260729"}
EVAL_RESULT=${EVAL_RESULT:-"$OUTPUT_ROOT/eval_runtime_version36_20260728/data/results/qwen25_coder7b_sft2_trl_resultonly_lr1e6_step23_tool_version36_equal300_seed20260729_passk4_t07_p095_logprobs20_bird_set"}
TRAIN_STATUS=${TRAIN_STATUS:-"$OUTPUT_ROOT/logs/phase1_result_only_trl_23x4_20260729.status"}
EVAL_STATUS=${EVAL_STATUS:-"$OUTPUT_ROOT/logs/phase1_result_only_equal300_passk4_20260729.status"}
QUEUE_STATUS=${QUEUE_STATUS:-"$OUTPUT_ROOT/logs/phase1_experiment_queue_20260729.status"}
TRAIN_LOG=${TRAIN_LOG:-"$OUTPUT_ROOT/logs/phase1_result_only_trl_23x4_20260729.run.log"}
VLLM_LOG=${VLLM_LOG:-"$OUTPUT_ROOT/logs/phase1_result_only_trl_23x4_20260729.vllm.log"}

mkdir -p "$OUTPUT_ROOT/logs"

line_count() {
  if [[ -f "$1" ]]; then
    wc -l <"$1" | tr -d '[:space:]'
  else
    printf '0\n'
  fi
}

mtime_epoch() {
  if [[ -f "$1" ]]; then
    stat -c '%Y' "$1"
  else
    printf '0\n'
  fi
}

status_value() {
  if [[ -f "$1" ]]; then
    tr '\n' ' ' <"$1"
  else
    printf 'missing'
  fi
}

snapshot() {
  local now_epoch newest_epoch age_seconds activity
  local rollout_rows completed_groups eval_rows
  local trainer_pid vllm_pid trainer_alive vllm_alive
  local gpu_lines gpu0_util gpu1_util latest_checkpoint

  now_epoch=$(date +%s)
  rollout_rows=$(line_count "$TRAIN_OUTPUT/rollouts.jsonl")
  completed_groups=$((rollout_rows / 4))
  eval_rows=$(line_count "$EVAL_RESULT/all.jsonl")
  latest_checkpoint=$(
    find "$TRAIN_OUTPUT" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
      2>/dev/null | sort -V | tail -n 1
  )
  latest_checkpoint=${latest_checkpoint:-none}

  trainer_pid=$(
    pgrep -f "run_transition_grpo.py.*trl-transition-v26-phase1-result-only-lr1e6-23x4-20260729" \
      | head -n 1 || true
  )
  vllm_pid=$(
    pgrep -f "trl vllm-serve.*--port 8030" | head -n 1 || true
  )
  trainer_alive=$([[ -n "$trainer_pid" ]] && printf yes || printf no)
  vllm_alive=$([[ -n "$vllm_pid" ]] && printf yes || printf no)

  gpu_lines=$(
    nvidia-smi \
      --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits 2>/dev/null || true
  )
  gpu0_util=$(awk -F ',' '$1 ~ /^0/ {gsub(/ /, "", $4); print $4}' <<<"$gpu_lines")
  gpu1_util=$(awk -F ',' '$1 ~ /^1/ {gsub(/ /, "", $4); print $4}' <<<"$gpu_lines")
  gpu0_util=${gpu0_util:-0}
  gpu1_util=${gpu1_util:-0}

  newest_epoch=0
  for path in "$TRAIN_OUTPUT/rollouts.jsonl" "$TRAIN_LOG" "$VLLM_LOG"; do
    current_epoch=$(mtime_epoch "$path")
    if [[ "$current_epoch" -gt "$newest_epoch" ]]; then
      newest_epoch=$current_epoch
    fi
  done
  age_seconds=$((now_epoch - newest_epoch))

  if [[ -f "$TRAIN_OUTPUT/final/adapter_model.safetensors" ]]; then
    activity=training_complete
  elif [[ "$trainer_alive" == no ]]; then
    activity=trainer_not_running
  elif [[ "$gpu0_util" -gt 0 || "$gpu1_util" -gt 0 ]]; then
    activity=active_gpu_compute
  elif [[ "$age_seconds" -le "$STALL_SECONDS" ]]; then
    activity=active_between_phases
  else
    activity=possible_stall
  fi

  temp_path=$(mktemp "$OUTPUT_ROOT/logs/.rl-monitor-latest.XXXXXX")
  {
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
    printf 'activity=%s\n' "$activity"
    printf 'queue_status=%s\n' "$(status_value "$QUEUE_STATUS")"
    printf 'train_status=%s\n' "$(status_value "$TRAIN_STATUS")"
    printf 'eval_status=%s\n' "$(status_value "$EVAL_STATUS")"
    printf 'trainer_alive=%s trainer_pid=%s\n' "$trainer_alive" "${trainer_pid:-none}"
    printf 'vllm_alive=%s vllm_pid=%s\n' "$vllm_alive" "${vllm_pid:-none}"
    printf 'rollout_rows=%s completed_update_groups=%s/23\n' \
      "$rollout_rows" "$completed_groups"
    printf 'latest_checkpoint=%s eval_rows=%s/300\n' \
      "$latest_checkpoint" "$eval_rows"
    printf 'seconds_since_artifact_activity=%s\n' "$age_seconds"
    printf 'gpu=index,memory_used_mib,memory_total_mib,utilization_percent,power_watts\n'
    printf '%s\n' "$gpu_lines"
  } >"$temp_path"
  mv "$temp_path" "$LATEST_STATUS"
  {
    printf '\n===== snapshot %s =====\n' "$(date --iso-8601=seconds)"
    cat "$LATEST_STATUS"
  } >>"$MONITOR_LOG"
}

if [[ "$MODE" == once ]]; then
  snapshot
  cat "$LATEST_STATUS"
  exit 0
fi
if [[ "$MODE" != daemon ]]; then
  printf 'usage: %s [once|daemon]\n' "$0" >&2
  exit 2
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'monitor already running; lock=%s\n' "$LOCK_FILE" >&2
  exit 1
fi
printf '%s\n' "$$" >"$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT INT TERM

while true; do
  snapshot
  sleep "$INTERVAL_SECONDS"
done
