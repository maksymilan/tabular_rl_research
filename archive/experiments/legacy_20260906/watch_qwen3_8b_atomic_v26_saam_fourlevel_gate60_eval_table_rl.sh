#!/usr/bin/env bash
# Durable handoff for the small four-level SAAM run: wait for training, then
# run the frozen-contract full BIRD-dev candidate-only evaluation.
set -euo pipefail

LAUNCHER_PID=${LAUNCHER_PID:?set the four-level launcher PID}
TRAIN_OUT=${TRAIN_OUT:-/home/dengyan/tabular_rl_outputs/qwen3_8b_atomic_v26_saam_fourlevel_gate60_20260901/train_two_pass_seed20260901}
EVAL_ROOT=${EVAL_ROOT:-/home/dengyan/tabular_rl_outputs/evaluations/qwen3_8b_atomic_v26_saam_fourlevel_gate60_20260901}
RUN_DIR=${RUN_DIR:-$EVAL_ROOT/fourlevel_final_dev_20260901}
EVAL_RUNTIME=${EVAL_RUNTIME:-/home/dengyan/tabular_rl_outputs/eval_runtime_qwen3_8b_v26_saam_gate60_20260829}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-8087}
LOG=${LOG:-$EVAL_ROOT/fourlevel_eval_watcher.log}
EVAL_ENTRY=$EVAL_RUNTIME/src/rl/evaluation/run_qwen3_8b_v26_saam_gate60_candidate_only_eval.py
EVAL_FORMAL=$EVAL_RUNTIME/src/rl/evaluation/formal_v26_matched_eval.py
EVAL_CONTRACT=$EVAL_RUNTIME/src/rl/evaluation/qwen3_8b_v26_vanilla_formal_matched_contract.json

mkdir -p "$EVAL_ROOT"
printf '%s waiting_for_training launcher=%s\n' "$(date -Is)" "$LAUNCHER_PID" >>"$LOG"
while kill -0 "$LAUNCHER_PID" 2>/dev/null; do sleep 30; done
printf '%s launcher_finished\n' "$(date -Is)" >>"$LOG"

for required in run_manifest.json implementation_lock.json training_precision.json \
  checkpoint-4/trainer_state.json checkpoint-4/adapter_model.safetensors \
  final/adapter_model.safetensors final/adapter_config.json rollouts.jsonl; do
  [[ -f "$TRAIN_OUT/$required" ]] || {
    printf '%s missing_training_artifact=%s\n' "$(date -Is)" "$TRAIN_OUT/$required" >>"$LOG"
    exit 3
  }
done

while true; do
  used=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  apps=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ -z "$apps" && "$used" =~ ^[0-9]+$ && "$used" -le 512 ]] && break
  sleep 30
done

# Validate the frozen source tree before allocating a GPU.  The verifier hashes
# admitted source content and ignores only known interpreter/test/OS caches;
# it must never mutate the runtime or require a hard-coded directory count.
if ! env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$EVAL_RUNTIME" "$PYTHON" - "$EVAL_CONTRACT" <<'PY' >>"$LOG" 2>&1
import sys
from pathlib import Path

from rl.scenarios.evaluation import formal_v26_matched_eval as formal

contract = formal.load_object(Path(sys.argv[1]))
formal.verify_runtime(contract, Path(contract["host_paths"]["runtime"]))
print("runtime_identity_check=ok")
PY
then
  printf '%s runtime_identity_check_failed\n' "$(date -Is)" >>"$LOG"
  exit 3
fi

[[ ! -e "$RUN_DIR" ]] || {
  printf '%s refusing_existing_eval_dir=%s\n' "$(date -Is)" "$RUN_DIR" >>"$LOG"
  exit 3
}
printf '%s starting_eval gpu=%s port=%s\n' "$(date -Is)" "$GPU_ID" "$PORT" >>"$LOG"
cd "$EVAL_RUNTIME"
exec env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$EVAL_RUNTIME" "$PYTHON" \
  "$EVAL_RUNTIME/src/rl/evaluation/run_qwen3_8b_v26_saam_gate60_candidate_only_eval.py" \
  --contract "$EVAL_RUNTIME/src/rl/evaluation/qwen3_8b_v26_vanilla_formal_matched_contract.json" \
  --training-run "$TRAIN_OUT" --run-dir "$RUN_DIR" \
  --gpu-id "$GPU_ID" --port "$PORT" --maximum-used-mib 512 \
  >>"$LOG" 2>&1
