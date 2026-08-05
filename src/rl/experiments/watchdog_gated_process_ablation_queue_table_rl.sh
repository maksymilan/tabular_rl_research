#!/usr/bin/env bash
# Server-side cron watchdog for the gated Exp3 -> Exp7 process-RL queue.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-"$OUTPUT_ROOT/rl_runtime_process_gated_v2_20260730"}
QUEUE_SCRIPT=${QUEUE_SCRIPT:-"$RUNTIME/src/rl/experiments/run_gated_process_ablation_queue_table_rl.sh"}
QUEUE_STATUS=${QUEUE_STATUS:-"$OUTPUT_ROOT/logs/gated_process_ablation_queue_20260730.status"}
QUEUE_LOG=${QUEUE_LOG:-"$OUTPUT_ROOT/logs/gated_process_ablation_queue_20260730.log"}
STALL_SECONDS=${STALL_SECONDS:-2700}

WATCHDOG_LOG=${WATCHDOG_LOG:-"$OUTPUT_ROOT/logs/gated_process_ablation_watchdog_20260730.log"}
WATCHDOG_LATEST=${WATCHDOG_LATEST:-"$OUTPUT_ROOT/logs/gated_process_ablation_watchdog_20260730.latest"}
WATCHDOG_LOCK=${WATCHDOG_LOCK:-"$OUTPUT_ROOT/logs/gated_process_ablation_watchdog_20260730.lock"}
QUEUE_SUPERVISOR_LOG=${QUEUE_SUPERVISOR_LOG:-"$OUTPUT_ROOT/logs/gated_process_ablation_queue_20260730.supervisor.log"}

ARTIFACTS=(
  trl-transition-v26-phase1-process-no-backslice-lr1e6-23x4-gated-v2-20260730
  trl-transition-v26-phase1-process-tool-only-lr1e6-23x4-gated-v2-20260730
  trl-transition-v26-phase2-process-rank-lr1e6-23x4-gated-v2-20260730
  trl-transition-v26-phase2-process-no-normalize-lr1e6-23x4-gated-v2-20260730
  trl-transition-v26-phase3-process-strong-penalty-lr1e6-23x4-gated-v2-20260731
)

mkdir -p "$OUTPUT_ROOT/logs"
exec 9>"$WATCHDOG_LOCK"
if ! flock -n 9; then
  exit 0
fi

line_count() {
  if [[ -f "$1" ]]; then
    wc -l <"$1" | tr -d '[:space:]'
  else
    printf '0\n'
  fi
}

status_line() {
  if [[ -f "$1" ]]; then
    tr '\n' ' ' <"$1"
  else
    printf 'missing'
  fi
}

queue_state() {
  if [[ -f "$QUEUE_STATUS" ]]; then
    awk -F '\t' 'NR == 1 {print $2}' "$QUEUE_STATUS"
  else
    printf 'missing\n'
  fi
}

latest_activity_epoch() {
  local newest=0 path epoch
  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    epoch=$(stat -c '%Y' "$path" 2>/dev/null || printf '0')
    if [[ "$epoch" -gt "$newest" ]]; then
      newest=$epoch
    fi
  done < <(
    find \
      "$OUTPUT_ROOT/logs" \
      "$OUTPUT_ROOT/checkpoints" \
      "$OUTPUT_ROOT/eval_runtime_version36_20260728/data/results" \
      -maxdepth 4 -type f \
      \( \
        -name 'gated_process_ablation_queue_20260730.status' -o \
        -name 'gated_process_ablation_queue_20260730.log' -o \
        -name 'trl-transition-v26-phase*-process-*-gated-v2-2026073[01]*' -o \
        -path '*trl-transition-v26-phase*-process-*-gated-v2-2026073[01]*/*' \
      \) \
      -print 2>/dev/null
  )
  printf '%s\n' "$newest"
}

verify_future_configs() {
  /home/dengyan/miniconda3/envs/trl-table/bin/python - "$RUNTIME" <<'PY'
import json
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
expected = {
    "phase1_process_tool_only.yaml": ("tool_only", False, True),
    "phase2_process_rank.yaml": ("all", True, True),
    "phase2_process_no_normalize.yaml": ("all", False, False),
    "phase3_process_strong_penalty.yaml": ("all", False, True),
}
for name, (trainable_part, rank_enabled, normalize_positive) in expected.items():
    config = yaml.safe_load(
        (root / "src/rl/configs/experiments" / name).read_text(encoding="utf-8")
    )
    reward_path = root / config["process_reward_config"]
    reward = json.loads(reward_path.read_text(encoding="utf-8"))
    assert reward["w_back_slice"] == 0.0, (name, reward_path)
    assert reward.get("normalize_positive", True) is normalize_positive, name
    assert config["trainable_part"] == trainable_part, name
    assert bool(config["rank_loss"]["enabled"]) is rank_enabled, name
    assert config["process_admission_policy"] == "counterfactual-completeness", name
    if name == "phase3_process_strong_penalty.yaml":
        assert reward["lambda_terminal_failure"] == 0.0, reward_path
        assert reward["lambda_tool_error"] == 0.24, reward_path
        assert reward["lambda_adjacent_repeat"] == 0.18, reward_path
        assert reward["lambda_legal_no_state_change"] == 0.09, reward_path
        assert reward["lambda_ignored_feedback"] == 0.0, reward_path
        assert reward["lambda_unsupported_guess"] == 0.0, reward_path
        assert reward["penalty_cap"] == 0.95, reward_path
PY
}

all_experiments_complete() {
  local artifact train_status eval_status
  for artifact in "${ARTIFACTS[@]}"; do
    train_status="$OUTPUT_ROOT/logs/${artifact}.train.status"
    eval_status="$OUTPUT_ROOT/logs/${artifact}.eval.status"
    if [[ "$(queue_state_from_file "$train_status")" != complete ]] || \
       [[ "$(queue_state_from_file "$eval_status")" != complete ]]; then
      return 1
    fi
  done
  return 0
}

queue_state_from_file() {
  if [[ -f "$1" ]]; then
    awk -F '\t' 'NR == 1 {print $2}' "$1"
  else
    printf 'missing\n'
  fi
}

start_queue() {
  verify_future_configs
  cd "$RUNTIME"
  nohup setsid bash "$QUEUE_SCRIPT" 9>&- </dev/null >>"$QUEUE_SUPERVISOR_LOG" 2>&1 &
  restarted_pid=$!
  action=restart
  reason="$1_pid=${restarted_pid}"
  sleep 2
  if ! kill -0 "$restarted_pid" 2>/dev/null; then
    action=restart_failed
    reason="queue_exited_after_safe_restart_pid=${restarted_pid}"
  fi
}

now_epoch=$(date +%s)
state=$(queue_state)
queue_script_name=${QUEUE_SCRIPT##*/}
queue_pids=$(
  ps -eo pid=,args= |
    awk -v self="$$" -v name="$queue_script_name" '
      $1 != self &&
      ($2 == "bash" || $2 ~ /\/bash$/) &&
      $3 ~ ("(^|/)" name "$") {
        print $1
      }
    ' | tr '\n' ' ' || true
)
heavy_processes=$(
  ps -eo pid=,args= |
    awk -v self="$$" '
      $1 != self &&
      $0 !~ /watchdog_gated_process_ablation_queue_table_rl/ &&
      $0 ~ /(run_transition_grpo.py|rollout_passk.py|run_bird_lora_tool_passk_local_gpu.sh|run_gated_process_equal300_passk4_table_rl.sh|trl vllm-serve|vllm.entrypoints.openai.api_server)/ {
        print
      }
    ' || true
)
gpu_lines=$(
  nvidia-smi \
    --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits 2>/dev/null || true
)
gpu0_util=$(awk -F ',' '$1 ~ /^0/ {gsub(/ /, "", $4); print $4}' <<<"$gpu_lines")
gpu1_util=$(awk -F ',' '$1 ~ /^1/ {gsub(/ /, "", $4); print $4}' <<<"$gpu_lines")
gpu0_memory=$(awk -F ',' '$1 ~ /^0/ {gsub(/ /, "", $2); print $2}' <<<"$gpu_lines")
gpu1_memory=$(awk -F ',' '$1 ~ /^1/ {gsub(/ /, "", $2); print $2}' <<<"$gpu_lines")
gpu0_util=${gpu0_util:-0}
gpu1_util=${gpu1_util:-0}
gpu0_memory=${gpu0_memory:-0}
gpu1_memory=${gpu1_memory:-0}
newest_epoch=$(latest_activity_epoch)
if [[ "$newest_epoch" -eq 0 ]]; then
  newest_epoch=$now_epoch
fi
age_seconds=$((now_epoch - newest_epoch))

action=observe
reason=unknown
if [[ "$state" == complete ]] && all_experiments_complete; then
  action=stop
  reason=queue_complete
elif [[ -n "$queue_pids" ]]; then
  reason=queue_alive
elif [[ -n "$heavy_processes" ]]; then
  reason=heavy_process_alive_without_queue
elif [[ "$gpu0_util" -gt 0 || "$gpu1_util" -gt 0 || \
        "$gpu0_memory" -gt 512 || "$gpu1_memory" -gt 512 ]]; then
  reason=gpu_resources_in_use_without_known_process
elif [[ "$state" == complete ]]; then
  start_queue queue_complete_with_pending_experiment
elif [[ "$age_seconds" -lt "$STALL_SECONDS" ]]; then
  reason=recent_artifact_activity
else
  start_queue safe_stall_recovery
fi

temp_path=$(mktemp "$OUTPUT_ROOT/logs/.gated-process-watchdog.XXXXXX")
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'action=%s reason=%s\n' "$action" "$reason"
  printf 'queue_status=%s\n' "$(status_line "$QUEUE_STATUS")"
  printf 'queue_pids=%s\n' "${queue_pids:-none}"
  printf 'heavy_processes=%s\n' "$([[ -n "$heavy_processes" ]] && printf yes || printf no)"
  printf 'seconds_since_artifact_activity=%s stall_threshold=%s\n' \
    "$age_seconds" "$STALL_SECONDS"
  printf 'gpu=index,memory_used_mib,memory_total_mib,utilization_percent,power_watts\n'
  printf '%s\n' "$gpu_lines"
  for artifact in "${ARTIFACTS[@]}"; do
    checkpoint="$OUTPUT_ROOT/checkpoints/$artifact"
    result_dir="$OUTPUT_ROOT/eval_runtime_version36_20260728/data/results/${artifact}_version36_equal300_seed20260729_passk4_t07_p095_logprobs20_bird_set"
    train_status="$OUTPUT_ROOT/logs/${artifact}.train.status"
    eval_status="$OUTPUT_ROOT/logs/${artifact}.eval.status"
    latest_checkpoint=$(
      if [[ -d "$checkpoint" ]]; then
        find "$checkpoint" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
          2>/dev/null | sort -V | tail -n 1
      fi
    )
    printf 'experiment=%s train_status=%s eval_status=%s rollouts=%s checkpoint=%s eval_rows=%s\n' \
      "$artifact" \
      "$(status_line "$train_status")" \
      "$(status_line "$eval_status")" \
      "$(line_count "$checkpoint/rollouts.jsonl")" \
      "${latest_checkpoint:-none}" \
      "$(line_count "$result_dir/all.jsonl")"
  done
} >"$temp_path"
mv "$temp_path" "$WATCHDOG_LATEST"
{
  printf '\n===== watchdog %s =====\n' "$(date --iso-8601=seconds)"
  cat "$WATCHDOG_LATEST"
  if [[ -n "$heavy_processes" ]]; then
    printf 'heavy_process_detail:\n%s\n' "$heavy_processes"
  fi
} >>"$WATCHDOG_LOG"

if [[ "$action" == restart_failed ]]; then
  exit 1
fi
