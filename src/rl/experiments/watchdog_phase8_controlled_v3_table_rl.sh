#!/usr/bin/env bash
set -euo pipefail
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${TRAIN_RUNTIME:-$O/rl_runtime_rank_score_v3_20260731}
S=$O/logs/phase8_controlled_queue_v3_table_rl_20260801.status
L=$O/logs/phase8_controlled_v3_watchdog_20260801.latest
H=$O/logs/phase8_controlled_v3_watchdog_20260801.history
K=$O/logs/phase8_controlled_v3_watchdog_20260801.lock
mkdir -p "$O/logs"; exec 9>"$K"; flock -n 9 || exit 0
state=$(awk -F '\t' 'NR==1 {print $2}' "$S" 2>/dev/null || printf missing)
pids=$(pgrep -f '[r]un_phase8_controlled_queue_v3_table_rl.sh' | tr '\n' ',' | sed 's/,$//' || true)
action=observe
parallel_recovery_ready=0
exp12_status=$O/logs/stage1_exp12_fixed_process_only.train.status
exp13_status=$O/logs/stage1_exp13_fixed_rank_only_action_mean.train.status
exp12_final=$O/checkpoints/trl-transition-v26-exp12-fixed-process-only-sft2-60xk4-seed101-20260801/final/adapter_model.safetensors
exp13_final=$O/checkpoints/trl-transition-v26-exp13-fixed-rank-only-action-mean-sft2-60xk4-seed101-20260801/final/adapter_model.safetensors
if grep -q $'\tcomplete\t' "$exp12_status" 2>/dev/null \
  && grep -q $'\tcomplete\t' "$exp13_status" 2>/dev/null \
  && [[ -f "$exp12_final" && -f "$exp13_final" ]]; then
  parallel_recovery_ready=1
fi
if [[ "$state" == stage1_complete ]]; then action=done
elif [[ ( "$state" == failed || "$state" == blocked ) && "$parallel_recovery_ready" == 1 && -z "$pids" ]]; then
  cd "$R"; nohup bash src/rl/experiments/run_phase8_controlled_queue_v3_table_rl.sh \
    >>"$O/logs/phase8_controlled_queue_v3_table_rl_20260801.nohup.log" 2>&1 </dev/null &
  pids=$!; action=restarted_after_verified_parallel_train
elif [[ "$state" == failed || "$state" == blocked ]]; then action=fail_closed
elif [[ -z "$pids" ]]; then
  cd "$R"; nohup bash src/rl/experiments/run_phase8_controlled_queue_v3_table_rl.sh \
    >>"$O/logs/phase8_controlled_queue_v3_table_rl_20260801.nohup.log" 2>&1 </dev/null &
  pids=$!; action=restarted_queue
fi
line="$(date -u '+%Y-%m-%dT%H:%M:%SZ')\t$action\tstate=$state pids=${pids:-none}"
printf '%b\n' "$line" >"$L"; printf '%b\n' "$line" >>"$H"
