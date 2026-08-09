#!/usr/bin/env bash
set -euo pipefail

# Detached table_rl queue: after the two current Qwen3-4B full evaluations
# complete successfully, launch the missing Qwen3-8B Direct and iterative-SQL
# full controls on the newly idle GPUs. Inputs stay server-local.

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PYTHON="/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python"
SOURCE="/home/dengyan/tabular_rl_project/data/eval_inputs/bird_dev_20240627.jsonl"
EXPECTED_SOURCE_SHA="8bf5a8bfe93ab49788656e2cc789bf80e729e0ec5f7f40159be01a1ab7b923e0"
RUN_ROOT="/home/dengyan/tabular_rl_outputs/evaluations/qwen3_sql_controls"
FOUR_B_DIRECT_STATUS="$RUN_ROOT/qwen3_4b_direct_sql_bird_dev1534_20260808/status.json"
FOUR_B_AUTHOR_SCORE="/home/dengyan/tabular_rl_outputs/reproductions/trust_sql/results/qwen3_4b_base_unknown_greedy_dev1534_seed20260808/score.json"
FOUR_B_AUTHOR_PID_FILE="/home/dengyan/tabular_rl_outputs/reproductions/trust_sql/logs/qwen3_4b_base/launcher_full_seed20260808.nohup.log.pid"
DIRECT_RUN="$RUN_ROOT/qwen3_8b_direct_sql_bird_dev1534_table_rl_20260808"
ITERATIVE_RUN="$RUN_ROOT/qwen3_8b_iterative_sql_v6_bird_dev1534_table_rl_20260808"
MODEL_ROOT="/home/dengyan/models/Qwen3-8B-TrustSQL-baseline"
QUEUE_STATUS="$HERE/queue.status"
RUNTIME_SHA="$(sha256sum "$HERE/runtime.tar.gz" | awk '{print $1}')"

printf 'waiting_for_4b_full_jobs %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$QUEUE_STATUS"
while true; do
  direct_ready=0
  author_ready=0
  if [[ -f "$FOUR_B_DIRECT_STATUS" ]]; then
    direct_state="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state"))' "$FOUR_B_DIRECT_STATUS")"
    if [[ "$direct_state" == failed ]]; then
      printf 'failed prerequisite_4b_direct_failed %s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$QUEUE_STATUS"
      exit 10
    fi
    [[ "$direct_state" == completed ]] && direct_ready=1
  fi
  if [[ -f "$FOUR_B_AUTHOR_SCORE" ]]; then
    author_ready=1
  elif [[ -f "$FOUR_B_AUTHOR_PID_FILE" ]]; then
    author_pid="$(tr -d '[:space:]' <"$FOUR_B_AUTHOR_PID_FILE")"
    if [[ "$author_pid" =~ ^[0-9]+$ ]] && ! kill -0 "$author_pid" 2>/dev/null; then
      printf 'failed prerequisite_4b_author_exited_without_score %s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$QUEUE_STATUS"
      exit 11
    fi
  fi
  (( direct_ready == 1 && author_ready == 1 )) && break
  sleep 60
done

"$PYTHON" - "$FOUR_B_DIRECT_STATUS" "$FOUR_B_AUTHOR_SCORE" <<'PY'
import json, sys
direct = json.load(open(sys.argv[1], encoding="utf-8"))
if direct.get("success") is not True or (direct.get("result") or {}).get("records") != 1534:
    raise SystemExit(f"4B direct prerequisite did not pass: {direct}")
author = json.load(open(sys.argv[2], encoding="utf-8"))
summary = author.get("summary") or {}
if summary.get("tasks") != 1534 or summary.get("observed_rollout_records") != 1534:
    raise SystemExit(f"4B author prerequisite is incomplete: {summary}")
PY

sha256sum -c "$HERE/STATIC_SHA256SUMS" >/dev/null
actual_source="$(sha256sum "$SOURCE" | awk '{print $1}')"
[[ "$actual_source" == "$EXPECTED_SOURCE_SHA" ]] || exit 12
for gpu in 0 1; do
  used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 512 )) || {
    printf 'failed gpu_%s_not_idle_%sMiB %s\n' "$gpu" "$used" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$QUEUE_STATUS"
    exit 13
  }
done

for run in "$DIRECT_RUN" "$ITERATIVE_RUN"; do
  [[ ! -e "$run" ]] || exit 14
  mkdir -p "$run/controller" "$run/input" "$run/runtime"
  cp "$HERE/runtime.tar.gz" "$run/runtime.tar.gz"
  cp "$HERE/remote_supervisor.py" "$run/controller/remote_supervisor.py"
  cp "$HERE/prepare_remote_eval_inputs.py" "$run/controller/prepare_remote_eval_inputs.py"
  cp "$HERE/verify_pinned_qwen3_model.py" "$run/controller/verify_pinned_qwen3_model.py"
  cp "$HERE/qwen3_model_specs.json" "$run/controller/qwen3_model_specs.json"
  cp "$HERE/remote_eval_lock.json" "$run/controller/remote_eval_lock.json"
  cp "$SOURCE" "$run/input/bird_dev_20240627.jsonl"
  tar -xzf "$run/runtime.tar.gz" -C "$run/runtime"
done

nohup setsid "$PYTHON" -u "$DIRECT_RUN/controller/remote_supervisor.py" \
  --mode direct --model-size 8b --model-root "$MODEL_ROOT" \
  --run-dir "$DIRECT_RUN" --gpu 1 --port 8040 --runtime-sha256 "$RUNTIME_SHA" \
  --n 1534 --max-model-len 32768 --gpu-memory-utilization 0.92 \
  >"$DIRECT_RUN/supervisor.log" 2>&1 </dev/null &
direct_pid=$!
printf '%s\n' "$direct_pid" >"$DIRECT_RUN/supervisor.pid"

nohup setsid "$PYTHON" -u "$ITERATIVE_RUN/controller/remote_supervisor.py" \
  --mode iterative --model-size 8b --model-root "$MODEL_ROOT" \
  --run-dir "$ITERATIVE_RUN" --gpu 0 --port 8041 --runtime-sha256 "$RUNTIME_SHA" \
  --n 1534 --max-model-len 32768 --gpu-memory-utilization 0.92 \
  >"$ITERATIVE_RUN/supervisor.log" 2>&1 </dev/null &
iterative_pid=$!
printf '%s\n' "$iterative_pid" >"$ITERATIVE_RUN/supervisor.pid"

printf 'submitted direct_pid=%s iterative_pid=%s %s\n' \
  "$direct_pid" "$iterative_pid" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$QUEUE_STATUS"
