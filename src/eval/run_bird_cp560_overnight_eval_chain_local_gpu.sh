#!/usr/bin/env bash
set -euo pipefail

# Unattended GPU-1 chain for checkpoint-560 and its untrained Coder-7B
# direct-SQL control. Run this on table_rl inside a dedicated tmux session.

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
GPU_ID="${GPU_ID:-1}"
CONCURRENCY="${CONCURRENCY:-24}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}"
PYTHON_BIN="${PYTHON_BIN:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
BASE_MODEL=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct
ADAPTER=/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-coder-7b-bird-student-raw-json-6400-qlora/checkpoint-560
EXPECTED_TASKS=1534
CHAIN_LOG="${CHAIN_LOG:-$OUTPUT_ROOT/logs/bird_cp560_overnight_eval_chain.log}"
STATUS_FILE="${STATUS_FILE:-$OUTPUT_ROOT/logs/bird_cp560_overnight_eval_chain.status}"

[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || {
  printf 'ERROR: CONCURRENCY must be a positive integer, got %s\n' "$CONCURRENCY" >&2
  exit 2
}

mkdir -p "$(dirname "$CHAIN_LOG")"
exec >>"$CHAIN_LOG" 2>&1

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

set_status() {
  local stage="$1"
  local detail="$2"
  printf '%s\t%s\t%s\n' "$(timestamp)" "$stage" "$detail" >"$STATUS_FILE"
}

completed_rows() {
  local result_dir="$1"
  if test -f "$result_dir/all.jsonl"; then
    wc -l <"$result_dir/all.jsonl" | tr -d '[:space:]'
  else
    printf '0\n'
  fi
}

require_complete() {
  local result_dir="$1"
  local rows
  rows="$(completed_rows "$result_dir")"
  if test "$rows" -ne "$EXPECTED_TASKS"; then
    printf 'ERROR: expected %s rows in %s, found %s\n' \
      "$EXPECTED_TASKS" "$result_dir" "$rows" >&2
    return 1
  fi
}

run_tool_stage() {
  local stage="$1"
  local served_model="$2"
  local port="$3"
  local result_dir="$4"
  local n_samples="$5"
  local pass_k="$6"
  local workers="$7"
  local sample_workers="$8"
  local server_config_id="$9"

  if test "$(completed_rows "$result_dir")" -eq "$EXPECTED_TASKS"; then
    printf '%s stage=%s already complete; skipping\n' "$(timestamp)" "$stage"
    return 0
  fi

  set_status "$stage" "running concurrency=$CONCURRENCY"
  printf '%s starting stage=%s concurrency=%s\n' \
    "$(timestamp)" "$stage" "$CONCURRENCY"
  PROJECT_DIR="$PROJECT_DIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  BASE_MODEL="$BASE_MODEL" \
  ADAPTER="$ADAPTER" \
  SERVED_MODEL="$served_model" \
  GPU_ID="$GPU_ID" \
  PORT="$port" \
  RESULT_DIR="$result_dir" \
  N="$EXPECTED_TASKS" \
  N_SAMPLES="$n_samples" \
  PASS_K="$pass_k" \
  WORKERS="$workers" \
  SAMPLE_WORKERS="$sample_workers" \
  MAX_INFLIGHT="$CONCURRENCY" \
  MAX_NUM_SEQS="$CONCURRENCY" \
  MAX_NUM_BATCHED_TOKENS=8192 \
  MAX_MODEL_LEN=8192 \
  GPU_MEMORY_UTILIZATION=0.90 \
  MAX_STEPS=30 \
  MAX_TOKENS=1024 \
  TEMPERATURE="$(
    if test "$n_samples" -eq 1; then printf '0'; else printf '0.7'; fi
  )" \
  TOP_P="$(
    if test "$n_samples" -eq 1; then printf '1'; else printf '0.95'; fi
  )" \
  HISTORY_TURNS=4 \
  EVAL_ENABLE_THINKING=0 \
  SERVER_CONFIG_ID="$server_config_id" \
  ALLOW_OPERATIONAL_CONCURRENCY_RESUME=1 \
  TOOL_EXECUTION_TIMEOUT_SECONDS=20 \
  ALLOW_OPERATIONAL_TOOL_TIMEOUT_RESUME=1 \
  VLLM_LOG="$OUTPUT_ROOT/logs/${served_model}.vllm.log" \
  VLLM_PID_FILE="$OUTPUT_ROOT/logs/${served_model}.vllm.pid" \
  EVAL_LOG="$OUTPUT_ROOT/logs/${served_model}.eval.log" \
    bash src/eval/run_bird_lora_tool_passk_local_gpu.sh
  require_complete "$result_dir"
  printf '%s completed stage=%s\n' "$(timestamp)" "$stage"
}

run_direct_stage() {
  local stage="$1"
  local served_model="$2"
  local port="$3"
  local result_dir="$4"

  if test "$(completed_rows "$result_dir")" -eq "$EXPECTED_TASKS"; then
    printf '%s stage=%s already complete; skipping\n' "$(timestamp)" "$stage"
    return 0
  fi

  set_status "$stage" "running concurrency=$CONCURRENCY"
  printf '%s starting stage=%s concurrency=%s\n' \
    "$(timestamp)" "$stage" "$CONCURRENCY"
  PROJECT_DIR="$PROJECT_DIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  BASE_MODEL="$BASE_MODEL" \
  SERVED_MODEL="$served_model" \
  GPU_ID="$GPU_ID" \
  PORT="$port" \
  RESULT_DIR="$result_dir" \
  N="$EXPECTED_TASKS" \
  N_SAMPLES=8 \
  PASS_K=1,2,4,8 \
  WORKERS="$(( (CONCURRENCY + 7) / 8 ))" \
  MAX_NUM_SEQS="$CONCURRENCY" \
  MAX_NUM_BATCHED_TOKENS=8192 \
  MAX_MODEL_LEN=8192 \
  GPU_MEMORY_UTILIZATION=0.90 \
  MAX_TOKENS=1024 \
  TEMPERATURE=0.7 \
  TOP_P=0.95 \
  REPETITION_PENALTY=1.05 \
  VLLM_LOG="$OUTPUT_ROOT/logs/${served_model}.vllm.log" \
  VLLM_PID_FILE="$OUTPUT_ROOT/logs/${served_model}.vllm.pid" \
  EVAL_LOG="$OUTPUT_ROOT/logs/${served_model}.eval.log" \
    bash src/eval/run_bird_direct_sql_passk_local_gpu.sh
  require_complete "$result_dir"
  printf '%s completed stage=%s\n' "$(timestamp)" "$stage"
}

cd "$PROJECT_DIR"
set_status preflight "checking GPU and artifacts"
printf '%s starting checkpoint-560 overnight evaluation chain\n' "$(timestamp)"

greedy_dir=data/results/qwen25_coder7b_raw_json_cp560_tool_dev1534_greedy1_bird_set
tool_k4_dir=data/results/qwen25_coder7b_raw_json_cp560_tool_dev1534_passk4_bird_set
base_k8_dir=data/results/qwen2.5_coder_7b_bird_direct_sql_base_passk8_rp105_dev1534_bird_ex
tool_k8_dir=data/results/qwen25_coder7b_raw_json_cp560_tool_dev1534_passk8_bird_set

run_tool_stage \
  greedy \
  qwen25-coder7b-rawjson-cp560-tool-dev1534-greedy1 \
  18035 \
  "$greedy_dir" \
  1 \
  1 \
  "$CONCURRENCY" \
  1 \
  vllm-generation-config-vllm-cp560-tool-dev1534-greedy1

run_tool_stage \
  tool_passk4 \
  qwen25-coder7b-rawjson-cp560-tool-dev1534-passk4 \
  18036 \
  "$tool_k4_dir" \
  4 \
  1,2,4 \
  "$(( (CONCURRENCY + 3) / 4 ))" \
  4 \
  vllm-generation-config-vllm-cp560-tool-dev1534-passk4

# The matching Coder-7B direct-SQL K=4 artifact is already complete locally
# (1534/1534). Run its new K=8 cohort before the much longer tool K=8 stage so
# an overnight run returns another complete control even if tool K=8 continues.
run_direct_stage \
  baseline_direct_sql_passk8 \
  qwen25-coder7b-direct-sql-base-passk8-rp105 \
  18038 \
  "$base_k8_dir"

run_tool_stage \
  tool_passk8 \
  qwen25-coder7b-rawjson-cp560-tool-dev1534-passk8 \
  18037 \
  "$tool_k8_dir" \
  8 \
  1,2,4,8 \
  "$(( (CONCURRENCY + 7) / 8 ))" \
  8 \
  vllm-generation-config-vllm-cp560-tool-dev1534-passk8

set_status complete "all queued evaluations completed"
printf '%s overnight evaluation chain complete\n' "$(timestamp)"
