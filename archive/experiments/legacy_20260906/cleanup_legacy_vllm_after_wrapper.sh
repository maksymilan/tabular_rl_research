#!/usr/bin/env bash
# One-shot cleanup for a vLLM server launched before process-group cleanup existed.
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  printf 'usage: %s WRAPPER_PID RESOURCE_TRACKER_PID ENGINE_CORE_PID\n' "$0" >&2
  exit 2
fi

wrapper_pid=$1
resource_tracker_pid=$2
engine_core_pid=$3

if [[ "$(ps -p "$wrapper_pid" -o args= 2>/dev/null)" \
  != "bash src/rl/experiments/run_gated_process_trl_table_rl.sh" ]]; then
  printf 'wrapper identity mismatch: pid=%s\n' "$wrapper_pid" >&2
  exit 1
fi

while kill -0 "$wrapper_pid" 2>/dev/null; do
  sleep 30
done
sleep 5

terminate_if_orphan() {
  local pid=$1
  local expected=$2
  local ppid=""
  local args=""
  if ! kill -0 "$pid" 2>/dev/null; then
    return
  fi
  ppid=$(ps -p "$pid" -o ppid= | tr -d '[:space:]')
  args=$(ps -p "$pid" -o args=)
  if [[ "$ppid" != 1 ]] || [[ "$args" != *"$expected"* ]]; then
    printf 'skip identity mismatch: pid=%s ppid=%s args=%s\n' \
      "$pid" "$ppid" "$args" >&2
    return
  fi
  kill -TERM "$pid" 2>/dev/null || true
}

terminate_if_orphan "$engine_core_pid" "VLLM::EngineCore"
terminate_if_orphan "$resource_tracker_pid" "multiprocessing.resource_tracker"
sleep 3

for pid in "$engine_core_pid" "$resource_tracker_pid"; do
  if ! kill -0 "$pid" 2>/dev/null; then
    continue
  fi
  ppid=$(ps -p "$pid" -o ppid= | tr -d '[:space:]')
  args=$(ps -p "$pid" -o args=)
  if [[ "$ppid" == 1 ]] \
    && { [[ "$args" == *"VLLM::EngineCore"* ]] \
      || [[ "$args" == *"multiprocessing.resource_tracker"* ]]; }; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
done

printf '%s cleanup_complete wrapper=%s resource_tracker=%s engine_core=%s\n' \
  "$(date --iso-8601=seconds)" \
  "$wrapper_pid" \
  "$resource_tracker_pid" \
  "$engine_core_pid"
