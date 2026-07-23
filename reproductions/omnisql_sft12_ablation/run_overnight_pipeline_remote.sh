#!/usr/bin/env bash
set -euo pipefail

QWEN_TRAIN_PID=${QWEN_TRAIN_PID:-1925378}
OMNI_TRAIN_PID=${OMNI_TRAIN_PID:-1935094}
PROJECT_DIR=/home/dengyan/tabular_rl_experiments/omnisql_sft12_ablation
OUTPUT_ROOT=/home/dengyan/tabular_rl_outputs/ablation/omnisql_sft12_1k
RUN_DIR="$PROJECT_DIR/reproductions/omnisql_sft12_ablation"
STATUS="$OUTPUT_ROOT/overnight_pipeline.status"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT/results"

set_status() {
  printf '%s %s\n' "$1" "$(date --iso-8601=seconds)" > "$STATUS"
}
fail() {
  set_status "failed:$1"
  echo "$1" >&2
  exit 1
}
trap 'code=$?; if (( code != 0 )); then set_status "failed:exit-$code"; fi' EXIT

wait_for_training() {
  local label=$1
  local pid=$2
  local output_dir=$3
  while true; do
    if [[ -f "$output_dir/trainer_state.json" && -f "$output_dir/adapter_config.json" ]]; then
      /home/dengyan/miniconda3/envs/sft/bin/python - "$output_dir/trainer_state.json" <<'PY'
import json
import sys
state = json.load(open(sys.argv[1]))
if int(state.get("global_step", -1)) != 128:
    raise SystemExit(f"unexpected global_step: {state.get('global_step')}")
PY
      echo "$label training complete"
      return
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      fail "$label training process $pid exited without a complete checkpoint"
    fi
    sleep 60
  done
}

set_status waiting-for-training
wait_for_training qwen "$QWEN_TRAIN_PID" "$OUTPUT_ROOT/checkpoints/qwen25_coder_7b" &
qwait=$!
wait_for_training omni "$OMNI_TRAIN_PID" "$OUTPUT_ROOT/checkpoints/omnisql_7b" &
owait=$!
wait "$qwait"
wait "$owait"

set_status evaluating-tools
bash "$RUN_DIR/evaluate_tool_fixed200_remote.sh" qwen25-coder sft 0 8024 \
  >"$LOG_DIR/qwen25_coder_sft_tool_fixed200.log" 2>&1 &
qtool=$!
bash "$RUN_DIR/evaluate_tool_fixed200_remote.sh" omnisql sft 1 8025 \
  >"$LOG_DIR/omnisql_sft_tool_fixed200.log" 2>&1 &
otool=$!
wait "$qtool"
wait "$otool"

set_status merging-omnisql-and-evaluating-qwen-direct
/home/dengyan/miniconda3/envs/sft/bin/python "$RUN_DIR/merge_adapter.py" \
  --base-model /home/dengyan/models/text2sql_reproduction/OmniSQL-7B-af4eed67 \
  --adapter "$OUTPUT_ROOT/checkpoints/omnisql_7b" \
  --out "$OUTPUT_ROOT/merged/omnisql_7b" \
  >"$LOG_DIR/omnisql_merge.log" 2>&1 &
merge_pid=$!
bash "$RUN_DIR/evaluate_qwen_direct_sql_remote.sh" 0 8026 \
  >"$LOG_DIR/qwen25_coder_sft_direct_sql.log" 2>&1
wait "$merge_pid"

set_status evaluating-omnisql-direct
bash "$RUN_DIR/evaluate_omnisql_direct_sql.sh" smoke \
  >"$LOG_DIR/omnisql_sft_direct_sql_smoke.log" 2>&1
bash "$RUN_DIR/evaluate_omnisql_direct_sql.sh" full \
  >"$LOG_DIR/omnisql_sft_direct_sql_full.log" 2>&1

set_status summarizing
/home/dengyan/miniconda3/envs/sft/bin/python "$RUN_DIR/summarize_overnight.py" \
  >"$LOG_DIR/overnight_summary.log" 2>&1
set_status completed
