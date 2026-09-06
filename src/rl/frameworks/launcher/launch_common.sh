#!/usr/bin/env bash
# Shared process and resource helpers for RL launchers.
# Experiment scripts should provide configuration and call these helpers;
# topology, locking, and cleanup logic must not be reimplemented per arm.

set -euo pipefail

gpu_idle() {
  local gpu=$1 pids used
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  [[ -z "$pids" && "$used" =~ ^[0-9]+$ && "$used" -le 512 ]]
}

assert_distinct_allowlisted_gpus() {
  local trainer_gpu=$1 serving_gpu=$2 max_gpu=${3:-3}
  [[ "$trainer_gpu" =~ ^[0-9]+$ && "$serving_gpu" =~ ^[0-9]+$ ]] || {
    printf 'trainer and serving GPUs must be numeric\n' >&2
    return 2
  }
  [[ "$trainer_gpu" -ge 0 && "$trainer_gpu" -le "$max_gpu" &&
     "$serving_gpu" -ge 0 && "$serving_gpu" -le "$max_gpu" ]] || {
    printf 'GPUs must be in allowlist 0-%s\n' "$max_gpu" >&2
    return 2
  }
  [[ "$trainer_gpu" != "$serving_gpu" ]] || {
    printf 'trainer and serving GPUs must be distinct\n' >&2
    return 2
  }
}

set_status() {
  local run_root=$1 state=$2 detail=$3
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" "$detail" >"$run_root/status.next"
  mv "$run_root/status.next" "$run_root/status"
}

stop_group() {
  local pgid=${1:-}
  [[ -n "$pgid" ]] || return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 -- "-$pgid" 2>/dev/null || break
      sleep 1
    done
    kill -0 -- "-$pgid" 2>/dev/null && kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
}

wait_for_health() {
  local pid=$1 url=$2 attempts=${3:-300} interval=${4:-2}
  for _ in $(seq 1 "$attempts"); do
    kill -0 "$pid" 2>/dev/null || return 1
    curl -fsS "$url" >/dev/null 2>&1 && return 0
    sleep "$interval"
  done
  return 1
}

install_process_cleanup_trap() {
  local run_root=$1
  # Callers set trainer_pgid/vllm_pgid in the same shell. The trap is shared so
  # every launcher terminates both process groups and records failure uniformly.
  _rl_cleanup() {
    local code=$?
    trap - EXIT INT TERM
    stop_group "${trainer_pgid:-}"
    stop_group "${vllm_pgid:-}"
    [[ "$code" -eq 0 ]] || set_status "$run_root" failed "exit=$code"
    exit "$code"
  }
  trap _rl_cleanup EXIT INT TERM
}
