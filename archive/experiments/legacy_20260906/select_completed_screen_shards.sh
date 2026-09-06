#!/usr/bin/env bash
# Complete the deterministic boundary-cohort selection after two fresh screen
# shards finish.  The selected tasks are new training inputs; screen
# trajectories are never reused by RL.
set -euo pipefail
SCREEN_ROOT=${SCREEN_ROOT:?set SCREEN_ROOT}
TASKS=${TASKS:?set TASKS}
SELECTOR=${SELECTOR:?set SELECTOR}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
POLL_SECONDS=${POLL_SECONDS:-60}
LOG=${LOG:-$SCREEN_ROOT/selection_queue.log}
mkdir -p "$(dirname "$LOG")"
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
while [[ ! -f "$SCREEN_ROOT/shard-0-r2/result/summary.json" || ! -f "$SCREEN_ROOT/shard-1-r2/result/summary.json" \
  || $(wc -l < "$SCREEN_ROOT/shard-0-r2/result/all.jsonl" 2>/dev/null || echo 0) -lt 300 \
  || $(wc -l < "$SCREEN_ROOT/shard-1-r2/result/all.jsonl" 2>/dev/null || echo 0) -lt 300 ]]; do
  sleep "$POLL_SECONDS"
done
OUT="$SCREEN_ROOT/selection/boundary_cohort"
if [[ -e "$OUT" ]]; then log "selection output already exists: $OUT"; exit 0; fi
mkdir -p "$(dirname "$OUT")"
export PYTHONPATH="$RUNTIME/src:$RUNTIME/src/rl:$RUNTIME"
"$PYTHON" "$SELECTOR" \
  --tasks "$TASKS" \
  --screen "$SCREEN_ROOT/shard-0-r2/result/all.jsonl" \
  --screen "$SCREEN_ROOT/shard-1-r2/result/all.jsonl" \
  --output-dir "$OUT" --count 300 \
  --seed "qwen3-v26-screen600-newgnn-20260905"
log "NewGNN screen selection completed"
