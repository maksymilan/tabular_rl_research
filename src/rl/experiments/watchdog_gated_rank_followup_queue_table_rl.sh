#!/usr/bin/env bash
# Cron watchdog: launch Exp8 -> Exp9 only after Exp7 and the main queue finish.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-"$OUTPUT_ROOT/rl_runtime_process_gated_v2_20260730"}
QUEUE_SCRIPT=${QUEUE_SCRIPT:-"$RUNTIME/src/rl/experiments/run_gated_rank_followup_queue_table_rl.sh"}
QUEUE_STATUS=${QUEUE_STATUS:-"$OUTPUT_ROOT/logs/gated_rank_followup_queue_20260731.status"}
PREDECESSOR_QUEUE_STATUS=${PREDECESSOR_QUEUE_STATUS:-"$OUTPUT_ROOT/logs/gated_process_ablation_queue_20260730.status"}
PREDECESSOR_EVAL_STATUS=${PREDECESSOR_EVAL_STATUS:-"$OUTPUT_ROOT/logs/trl-transition-v26-phase3-process-strong-penalty-lr1e6-23x4-gated-v2-20260731.eval.status"}
WATCHDOG_LOG=${WATCHDOG_LOG:-"$OUTPUT_ROOT/logs/gated_rank_followup_watchdog_20260731.log"}
WATCHDOG_LATEST=${WATCHDOG_LATEST:-"$OUTPUT_ROOT/logs/gated_rank_followup_watchdog_20260731.latest"}
WATCHDOG_LOCK=${WATCHDOG_LOCK:-"$OUTPUT_ROOT/logs/gated_rank_followup_watchdog_20260731.lock"}
QUEUE_SUPERVISOR_LOG=${QUEUE_SUPERVISOR_LOG:-"$OUTPUT_ROOT/logs/gated_rank_followup_queue_20260731.supervisor.log"}
HANDOFF_MARKER=${HANDOFF_MARKER:-"$OUTPUT_ROOT/logs/rank_followup_handed_off_to_newgnn_20260731"}

ARTIFACTS=(
  trl-transition-v26-phase4-process-rank-action-only-lr1e6-23x4-gated-v2-20260731
  trl-transition-v26-phase5-process-rank-conservative-lr1e6-23x4-gated-v2-20260731
)

mkdir -p "$OUTPUT_ROOT/logs"
exec 9>"$WATCHDOG_LOCK"
if ! flock -n 9; then
  exit 0
fi

state_of() {
  if [[ -f "$1" ]]; then
    awk -F '\t' 'NR==1 {print $2}' "$1"
  else
    printf 'missing\n'
  fi
}
status_line() {
  if [[ -f "$1" ]]; then
    tr '\n' ' ' <"$1"
  else
    printf 'missing'
  fi
}
line_count() {
  if [[ -f "$1" ]]; then
    wc -l <"$1" | tr -d '[:space:]'
  else
    printf '0\n'
  fi
}
all_complete() {
  local artifact
  for artifact in "${ARTIFACTS[@]}"; do
    [[ "$(state_of "$OUTPUT_ROOT/logs/${artifact}.train.status")" == complete ]] || return 1
    [[ "$(state_of "$OUTPUT_ROOT/logs/${artifact}.eval.status")" == complete ]] || return 1
  done
}
verify_configs() {
  /home/dengyan/miniconda3/envs/trl-table/bin/python - "$RUNTIME" <<'PY'
import json
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
expected = {
    "phase4_process_rank_action_only.yaml": "full_trajectory",
    "phase5_process_rank_conservative.yaml": "conservative_legal",
}
for name, scope in expected.items():
    config = yaml.safe_load(
        (root / "src/rl/configs/experiments" / name).read_text(encoding="utf-8")
    )
    reward = json.loads(
        (root / config["process_reward_config"]).read_text(encoding="utf-8")
    )
    assert config["admission_status"] == "allowed_process_after_gates", name
    assert config["process_admission_policy"] == "counterfactual-completeness", name
    assert config["trainable_part"] == "all", name
    assert config["rank_loss"] == {
        "enabled": True,
        "coefficient": 0.5,
        "beta": 0.1,
        "score_tokens": "tool_only",
        "update_scope": scope,
    }, name
    assert reward["w_back_slice"] == 0.0, name
    assert reward.get("normalize_positive", True) is True, name
PY
}

queue_script_name=${QUEUE_SCRIPT##*/}
queue_pids=$(
  ps -eo pid=,args= |
    awk -v self="$$" -v name="$queue_script_name" '
      $1 != self &&
      ($2 == "bash" || $2 ~ /\/bash$/) &&
      $3 ~ ("(^|/)" name "$") { print $1 }
    ' | tr '\n' ' ' || true
)
heavy_processes=$(
  ps -eo pid=,args= |
    awk -v self="$$" '
      $1 != self &&
      $0 !~ /watchdog_gated_rank_followup_queue_table_rl/ &&
      $0 ~ /(run_transition_grpo.py|rollout_passk.py|run_bird_lora_tool_passk_local_gpu.sh|run_gated_process_equal300_passk4_table_rl.sh|trl vllm-serve|vllm.entrypoints.openai.api_server)/ { print }
    ' || true
)
gpu_lines=$(
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits 2>/dev/null || true
)
gpu0_memory=$(awk -F ',' '$1 ~ /^0/ {gsub(/ /, "", $2); print $2}' <<<"$gpu_lines")
gpu1_memory=$(awk -F ',' '$1 ~ /^1/ {gsub(/ /, "", $2); print $2}' <<<"$gpu_lines")
gpu0_memory=${gpu0_memory:-0}
gpu1_memory=${gpu1_memory:-0}

action=observe
reason=unknown
if [[ -f "$HANDOFF_MARKER" ]]; then
  action=stop
  reason=handed_off_to_newgnn
elif [[ "$(state_of "$QUEUE_STATUS")" == complete ]] && all_complete; then
  action=stop
  reason=followup_complete
elif [[ -n "$queue_pids" ]]; then
  reason=followup_queue_alive
elif [[ "$(state_of "$QUEUE_STATUS")" == failed ]]; then
  action=stop
  reason=failed_requires_audit
elif [[ "$(state_of "$PREDECESSOR_QUEUE_STATUS")" != complete ]] || \
     [[ "$(state_of "$PREDECESSOR_EVAL_STATUS")" != complete ]]; then
  action=wait
  reason=predecessor_not_complete
elif [[ -n "$heavy_processes" ]]; then
  action=wait
  reason=heavy_process_alive
elif [[ "$gpu0_memory" -gt 512 || "$gpu1_memory" -gt 512 ]]; then
  action=wait
  reason=gpu_resources_in_use
else
  verify_configs
  cd "$RUNTIME"
  nohup setsid bash "$QUEUE_SCRIPT" 9>&- </dev/null >>"$QUEUE_SUPERVISOR_LOG" 2>&1 &
  queue_pid=$!
  action=start
  reason="predecessor_complete_resources_free_pid=$queue_pid"
fi

temp_path=$(mktemp "$OUTPUT_ROOT/logs/.gated-rank-followup-watchdog.XXXXXX")
{
  printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
  printf 'action=%s reason=%s\n' "$action" "$reason"
  printf 'predecessor_queue=%s\n' "$(status_line "$PREDECESSOR_QUEUE_STATUS")"
  printf 'predecessor_eval=%s\n' "$(status_line "$PREDECESSOR_EVAL_STATUS")"
  printf 'followup_queue=%s\n' "$(status_line "$QUEUE_STATUS")"
  printf 'queue_pids=%s heavy_processes=%s\n' \
    "${queue_pids:-none}" "$([[ -n "$heavy_processes" ]] && printf yes || printf no)"
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
  if [[ -n "$heavy_processes" ]]; then
    printf 'heavy_process_detail:\n%s\n' "$heavy_processes"
  fi
} >>"$WATCHDOG_LOG"
