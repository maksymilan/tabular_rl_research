#!/usr/bin/env bash
set -euo pipefail
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${TRAIN_RUNTIME:-$O/rl_runtime_rank_score_v3_20260731}
V=$O/logs/dense_outcome_views_table_rl_20260802.status
S=$O/logs/dense_full_response_queue_table_rl_20260802.status
L=$O/logs/dense_full_response_watchdog_table_rl_20260802.latest
H=$O/logs/dense_full_response_watchdog_table_rl_20260802.history
K=$O/logs/dense_full_response_watchdog_table_rl_20260802.lock
mkdir -p "$O/logs"; exec 9>"$K"; flock -n 9 || exit 0
state_of() { awk -F '\t' 'NR==1 {print $2}' "$1" 2>/dev/null || printf missing; }
view_state=$(state_of "$V")
queue_state=$(state_of "$S")
queue_pids=$(pgrep -f '[r]un_dense_full_response_queue_table_rl.sh' | tr '\n' ',' | sed 's/,$//' || true)
action=observe
exp16_diag=$O/logs/stage1_exp16_dense_uniform_full_response.diagnostics.status
exp17_diag=$O/logs/stage1_exp17_dense_strategic_full_response.diagnostics.status
exp17_output=$O/checkpoints/trl-transition-v26-exp17-dense-strategic-full-response-sft2-60xk4-seed101-20260802
exp17_pool=$O/phase8_controlled_20260801/balanced_mixed60_dense_strategic_seed101
exp17_lane_status=$O/logs/dense_exp17_lane_table_rl_20260802.status
exp17_lane_state=$(state_of "$exp17_lane_status")
exp17_lane_pids=$(pgrep -f '[r]un_dense_single_lane_table_rl.sh' | tr '\n' ',' | sed 's/,$//' || true)
safe_exp17_recovery=0
if [[ "$view_state" == complete \
  && "$(state_of "$exp16_diag")" == complete \
  && ! -e "$exp17_output" \
  && -z "$exp17_lane_pids" ]]; then
  safe_exp17_recovery=1
fi

if [[ "$(state_of "$exp16_diag")" == complete \
  && "$(state_of "$exp17_diag")" == complete \
  && -z "$queue_pids" && -z "$exp17_lane_pids" ]]; then
  printf '%s\tcomplete\tExp16 and Exp17 train+full-dev complete\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >"$S"
  queue_state=complete
  action=done
elif [[ "$exp17_lane_state" == failed || "$exp17_lane_state" == blocked ]]; then
  action=fail_closed_exp17_lane
elif [[ ( "$queue_state" == failed || "$queue_state" == blocked ) \
  && "$safe_exp17_recovery" == 1 && -z "$queue_pids" ]]; then
  cd "$R"
  nohup bash src/rl/experiments/run_dense_full_response_queue_table_rl.sh \
    >>"$O/logs/dense_full_response_queue_table_rl_20260802.nohup.log" \
    2>&1 </dev/null &
  queue_pids=$!
  action=restarted_after_verified_exp16
elif [[ "$view_state" == failed || "$view_state" == blocked \
  || "$queue_state" == failed || "$queue_state" == blocked ]]; then
  action=fail_closed
elif [[ "$queue_state" == complete ]]; then
  action=done
elif [[ -z "$queue_pids" ]]; then
  cd "$R"
  nohup bash src/rl/experiments/run_dense_full_response_queue_table_rl.sh \
    >>"$O/logs/dense_full_response_queue_table_rl_20260802.nohup.log" \
    2>&1 </dev/null &
  queue_pids=$!
  action=started_queue
fi
if [[ -f "$exp17_pool/manifest.json" \
  && "$exp17_lane_state" != complete \
  && "$exp17_lane_state" != failed \
  && "$exp17_lane_state" != blocked \
  && -z "$exp17_lane_pids" \
  && ! -e "$exp17_output" ]]; then
  cd "$R"
  CANDIDATE_NAME=exp17_dense_strategic_full_response \
  EXPERIMENT_CONFIG=exp17_dense_strategic_full_response.yaml \
  ARTIFACT=trl-transition-v26-exp17-dense-strategic-full-response-sft2-60xk4-seed101-20260802 \
  POOL_DIR="$exp17_pool" GPU_ID=0 PORT=18082 \
  STATUS="$exp17_lane_status" \
  RUN_LOG="$O/logs/dense_exp17_lane_table_rl_20260802.log" \
  nohup bash src/rl/experiments/run_dense_single_lane_table_rl.sh \
    >>"$O/logs/dense_exp17_lane_table_rl_20260802.nohup.log" \
    2>&1 </dev/null &
  exp17_lane_pids=$!
  action="${action}+started_exp17_lane"
fi
if [[ "$view_state" != complete \
  && "$view_state" != failed && "$view_state" != blocked ]]; then
  cd "$R"
  bash src/rl/experiments/prepare_dense_outcome_views_table_rl.sh
  if [[ "$action" == observe ]]; then action=prepared_or_waiting_views; fi
fi
line="$(date -u '+%Y-%m-%dT%H:%M:%SZ')\t$action\tviews=$(state_of "$V") queue=$(state_of "$S") queue_pids=${queue_pids:-none} exp17_lane=$(state_of "$exp17_lane_status") exp17_pids=${exp17_lane_pids:-none}"
printf '%b\n' "$line" >"$L"; printf '%b\n' "$line" >>"$H"
