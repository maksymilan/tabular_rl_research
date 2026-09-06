#!/usr/bin/env bash
# Score independent fixed-prefix checkpoints on an explicitly free public-server GPU.
set -euo pipefail

O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
R=${RUNTIME:-$O/rl_runtime_phase8_parallel_20260801}
WORK=${WORK_DIR:-$O/phase8_parallel_20260801/fixed_prefix_scores}
DATASET=${DATASET:-$WORK/fixed_prefix_dev300.jsonl}
BASE_MODEL=${BASE_MODEL:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
for name in GPU_ID STATUS RUN_LOG; do
  [[ -n "${!name:-}" ]] || { printf 'missing %s\n' "$name" >&2; exit 2; }
done
[[ "$GPU_ID" == 6 || "$GPU_ID" == 7 ]] || {
  printf 'NewGNN scoring is restricted to GPU6/7, got %s\n' "$GPU_ID" >&2
  exit 2
}
[[ "$#" -gt 0 ]] || { printf 'pass NAME=ADAPTER assignments\n' >&2; exit 2; }

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
on_exit() {
  local code=$?
  trap - EXIT
  local state
  state=$(awk -F '\t' 'NR==1 {print $2}' "$STATUS" 2>/dev/null || true)
  if [[ "$code" -ne 0 && "$state" != blocked ]]; then
    set_status failed "exit=$code see=$RUN_LOG"
  fi
  exit "$code"
}
trap on_exit EXIT
mkdir -p "$O/logs" "$WORK/results"
exec >>"$RUN_LOG" 2>&1

used=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits)
compute_pids=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[0-9]+$' || true)
if [[ "$used" -gt 512 || -n "$compute_pids" ]]; then
  set_status blocked "gpu=$GPU_ID mib=$used compute_pids=${compute_pids:-none}"
  exit 3
fi
expected=$(wc -l <"$DATASET" | tr -d '[:space:]')
[[ "$expected" -gt 0 ]]
cd "$R"

ordinal=0
for assignment in "$@"; do
  ordinal=$((ordinal + 1))
  name=${assignment%%=*}
  adapter=${assignment#*=}
  [[ -n "$name" && "$adapter" != "$assignment" ]]
  [[ -f "$adapter/adapter_model.safetensors" ]]
  jsonl="$WORK/results/$name.jsonl"
  parquet="$WORK/results/$name.parquet"
  if [[ -f "$jsonl" && -f "$parquet" ]] \
    && [[ "$(wc -l <"$jsonl" | tr -d '[:space:]')" -eq "$expected" ]]; then
    set_status scoring "gpu=$GPU_ID ordinal=$ordinal/$# checkpoint=$name already_complete=1"
    continue
  fi
  set_status scoring "gpu=$GPU_ID ordinal=$ordinal/$# checkpoint=$name pairs=$expected"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" src/rl/diagnostics/score_action_candidates.py \
    --dataset "$DATASET" --checkpoint-name "$name" --base-model "$BASE_MODEL" \
    --adapter "$adapter" --output-jsonl "$jsonl" --output-parquet "$parquet" \
    --device cuda:0
  [[ "$(wc -l <"$jsonl" | tr -d '[:space:]')" -eq "$expected" ]]
  "$PY" - "$parquet" "$expected" <<'PY'
import sys
import pyarrow.parquet as pq
assert pq.read_table(sys.argv[1]).num_rows == int(sys.argv[2])
PY
done
set_status complete "gpu=$GPU_ID checkpoints=$# pairs=$expected"
