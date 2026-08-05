#!/usr/bin/env bash
# Queue the expanded Exp15 checkpoint for the frozen stage-1 evaluation on NewGNN.
# Fixed-prefix scoring (GPU 7) and full BIRD-dev1534 greedy@1 (GPU 6) run in
# parallel.  The predeclared SFT2 gate runs only after both artifacts complete.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-$OUTPUT_ROOT/exp15_exact_prefix_branch_20260802}
EVAL_RUNTIME=${EVAL_RUNTIME:-$OUTPUT_ROOT/eval_runtime_version36_20260728}
LAUNCH_RUNTIME=${LAUNCH_RUNTIME:-$OUTPUT_ROOT/rl_runtime_exp15_exact_prefix_branch_20260802}
SCORE_RUNTIME=${SCORE_RUNTIME:-$OUTPUT_ROOT/rl_runtime_phase8_parallel_20260801}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
CANDIDATE_NAME=${CANDIDATE_NAME:-exp15_exact_prefix_stage5_expanded_union_action_dpo}
ADAPTER=${ADAPTER:-$OUTPUT_ROOT/checkpoints/trl-transition-v26-exp15-exact-prefix-stage5-expanded-union-action-dpo-sft2-seed101-20260802/final}
ADAPTER_SHA256=${ADAPTER_SHA256:-0486de6a0f2ba1a24b94448755fa6382878ffef16b83425db142b38d57d2b3bb}
PREFIX_GPU_ID=${PREFIX_GPU_ID:-7}
GREEDY_GPU_ID=${GREEDY_GPU_ID:-6}
PORT=${PORT:-18086}
SERIAL_EVALUATION=${SERIAL_EVALUATION:-0}
GPU_FREE_THRESHOLD_MIB=${GPU_FREE_THRESHOLD_MIB:-512}
GPU_WAIT_MAX_CHECKS=${GPU_WAIT_MAX_CHECKS:-1440}

FIXED_PREFIX_DIR=$EVAL_RUNTIME/data/diagnostics/fixed_prefix_dev300_union4_v2
FIXED_PREFIX_DATASET=$FIXED_PREFIX_DIR/fixed_prefix_dev300.jsonl
SFT2_PREFIX=$FIXED_PREFIX_DIR/scores/sft2.jsonl
PREFIX_SCORE=$FIXED_PREFIX_DIR/scores/$CANDIDATE_NAME.jsonl
PREFIX_PARQUET=$FIXED_PREFIX_DIR/scores/$CANDIDATE_NAME.parquet
PREFIX_MANIFEST=$FIXED_PREFIX_DIR/scores/$CANDIDATE_NAME.manifest.json
SFT2_GREEDY=$EVAL_RUNTIME/data/results/sft2_version36_dev1534_greedy_t0_logprobs20_20260801/all.jsonl
GREEDY_RESULT=$EVAL_RUNTIME/data/results/stage1_${CANDIDATE_NAME}_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260802
EVALUATION_DIR=$EXPERIMENT_ROOT/evaluation
DECISION=$EVALUATION_DIR/decision_requirements.json
COMPARISON=$EVALUATION_DIR/comparison_vs_original_exp15.json
ORIGINAL_DECISION=$EVALUATION_DIR/original_exp15_decision_requirements.json
STATUS=${STATUS:-$OUTPUT_ROOT/logs/${CANDIDATE_NAME}.evaluation_queue.status}
RUN_LOG=${RUN_LOG:-$OUTPUT_ROOT/logs/${CANDIDATE_NAME}.evaluation_queue.run.log}
PREFIX_STATUS=$OUTPUT_ROOT/logs/${CANDIDATE_NAME}.fixed_prefix.status
FULL_STATUS=$OUTPUT_ROOT/logs/${CANDIDATE_NAME}.full_dev_greedy.status
FULL_LOG=$OUTPUT_ROOT/logs/${CANDIDATE_NAME}.full_dev_greedy.run.log
LAUNCH_MANIFEST=$EVALUATION_DIR/evaluation_launch_manifest.json
LOCK=${LOCK:-$STATUS.lock}

EXPECTED_QUESTIONS=1534
EXPECTED_PREFIX_PAIRS=122
EXPECTED_EVAL_INPUT_SHA256=636e096babe2db9dde096b655ed01a0e1e6aae1ea5cbf4e952770e02841721f5
EXPECTED_SFT2_GREEDY_SHA256=f1ad3c5b36854e3e934fd6645c82116d745677cd5f65e8f77bbfb0092c5f0fc0
EXPECTED_PREFIX_DATASET_SHA256=c6a402089aaca2a1e48a0b477856c074d3869cc8203b67336cee49f21fbf8ebb
EXPECTED_SFT2_PREFIX_SHA256=43b63c3a83596f93e556112c94e3b7377e5fa75994fef837910acb7070b1d964
EXPECTED_FULL_LAUNCHER_SHA256=419103479ba3b7436961a36579c9c50a66f8b379d5cb6d87459269810baadaf5
EXPECTED_GATE_SHA256=44cfe1b84ef77e34ea753eba4d32bbf6d11497c84cd57a2d7c2f80975f37161f
EXPECTED_HARDWARE_SHA256=8f03b24e9295f9b8a896ae83c8ea43e22910d2d322964ae389430e66c78bf2f1
EXPECTED_SCORER_SHA256=8dde8c7f22400e5a55e36e278d3fc790f1a90a43bce0ac6b923c09dce3fac9c8
EXPECTED_METRICS_BUILDER_SHA256=917196ff5ea1776bb266bf4172b923186a990db6e8959e499d07561734cdd314

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
set_prefix_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$PREFIX_STATUS"; }
sha256() { sha256sum "$1" | awk '{print $1}'; }
nonempty_lines() { grep -cve '^[[:space:]]*$' "$1"; }
state_of() { awk -F '\t' 'NR==1 {print $2}' "$1" 2>/dev/null || printf missing; }

on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then
    set_status failed "exit=$code see=$RUN_LOG"
  fi
  exit "$code"
}
trap on_exit EXIT

mkdir -p "$OUTPUT_ROOT/logs" "$FIXED_PREFIX_DIR/scores" "$EVALUATION_DIR" "$GREEDY_RESULT"
exec >>"$RUN_LOG" 2>&1
exec 9>"$LOCK"
if ! flock -n 9; then
  printf '%s queue already owns %s; current_state=%s\n' "$(timestamp)" "$LOCK" "$(state_of "$STATUS")"
  exit 0
fi

require_file_hash() {
  local path=$1 expected=$2 label=$3
  test -f "$path" || { printf 'missing %s: %s\n' "$label" "$path" >&2; return 1; }
  local actual
  actual=$(sha256 "$path")
  [[ "$actual" == "$expected" ]] || {
    printf '%s hash mismatch: expected=%s actual=%s path=%s\n' "$label" "$expected" "$actual" "$path" >&2
    return 1
  }
}

wait_for_gpu() {
  local gpu=$1 stage=$2 used pids
  for ((check = 1; check <= GPU_WAIT_MAX_CHECKS; check++)); do
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
    pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | awk '/^[0-9]+$/ {print}' | paste -sd, - || true)
    if [[ -n "$used" && "$used" -le "$GPU_FREE_THRESHOLD_MIB" && -z "$pids" ]]; then
      printf '%s gpu_ready stage=%s gpu=%s memory_mib=%s\n' "$(timestamp)" "$stage" "$gpu" "$used"
      return 0
    fi
    set_status waiting_gpu "stage=$stage gpu=$gpu mib=${used:-unknown} pids=${pids:-none}"
    sleep 30
  done
  printf 'timed out waiting for GPU %s for %s\n' "$gpu" "$stage" >&2
  return 1
}

set_status preflight "candidate=$CANDIDATE_NAME"
test -d "$ADAPTER"
require_file_hash "$ADAPTER/adapter_model.safetensors" "$ADAPTER_SHA256" adapter
require_file_hash "$EVAL_RUNTIME/data/eval_inputs/bird_dev_20240627.jsonl" "$EXPECTED_EVAL_INPUT_SHA256" eval_input
require_file_hash "$SFT2_GREEDY" "$EXPECTED_SFT2_GREEDY_SHA256" sft2_greedy
require_file_hash "$FIXED_PREFIX_DATASET" "$EXPECTED_PREFIX_DATASET_SHA256" fixed_prefix_dataset
require_file_hash "$SFT2_PREFIX" "$EXPECTED_SFT2_PREFIX_SHA256" sft2_prefix
require_file_hash "$LAUNCH_RUNTIME/src/rl/experiments/run_full_dev_greedy_table_rl.sh" "$EXPECTED_FULL_LAUNCHER_SHA256" full_launcher
require_file_hash "$LAUNCH_RUNTIME/src/rl/experiments/evaluate_stage1_candidate_gate.py" "$EXPECTED_GATE_SHA256" gate
require_file_hash "$LAUNCH_RUNTIME/src/rl/configs/hardware/rtx3090_24gb.sh" "$EXPECTED_HARDWARE_SHA256" hardware_profile
require_file_hash "$SCORE_RUNTIME/src/rl/diagnostics/score_action_candidates.py" "$EXPECTED_SCORER_SHA256" fixed_prefix_scorer
require_file_hash "$EVAL_RUNTIME/src/eval/build_experiment_eval_metrics.py" "$EXPECTED_METRICS_BUILDER_SHA256" metrics_builder
test "$(nonempty_lines "$EVAL_RUNTIME/data/eval_inputs/bird_dev_20240627.jsonl")" -eq "$EXPECTED_QUESTIONS"
test "$(nonempty_lines "$SFT2_GREEDY")" -eq "$EXPECTED_QUESTIONS"
test "$(nonempty_lines "$FIXED_PREFIX_DATASET")" -eq "$EXPECTED_PREFIX_PAIRS"
test "$(nonempty_lines "$SFT2_PREFIX")" -eq "$EXPECTED_PREFIX_PAIRS"

if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${PORT}$"; then
  printf 'evaluation port is already in use: %s\n' "$PORT" >&2
  exit 1
fi

"$PYTHON_BIN" - "$LAUNCH_MANIFEST" <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "schema_version": "expanded-exp15-stage1-evaluation-launch-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "candidate": "$CANDIDATE_NAME",
    "adapter": "$ADAPTER",
    "adapter_sha256": "$ADAPTER_SHA256",
    "evaluation_protocol": {
        "atomic_protocol_version": "version36",
        "primary": "BIRD-dev1534 greedy@1",
        "temperature": 0.0,
        "top_p": 1.0,
        "denotation_comparison": "bird-set",
        "fixed_prefix_pairs": $EXPECTED_PREFIX_PAIRS,
        "k4": "deferred_to_finalist_after_stage1_gate",
    },
    "resources": {
        "full_dev_gpu": $GREEDY_GPU_ID,
        "fixed_prefix_gpu": $PREFIX_GPU_ID,
        "serial_evaluation": bool(int("$SERIAL_EVALUATION")),
        "vllm_port": $PORT,
    },
    "artifacts": {
        "fixed_prefix_score": "$PREFIX_SCORE",
        "full_dev_result": "$GREEDY_RESULT",
        "decision": "$DECISION",
    },
}
Path("$LAUNCH_MANIFEST").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY

# This is the queue-level live allocation check.  Each child performs another
# check immediately before loading a model.
wait_for_gpu "$PREFIX_GPU_ID" prefix
wait_for_gpu "$GREEDY_GPU_ID" full_dev

run_fixed_prefix() {
  set -euo pipefail
  if [[ -f "$PREFIX_SCORE" && -f "$PREFIX_PARQUET" && -f "$PREFIX_MANIFEST" \
    && "$(nonempty_lines "$PREFIX_SCORE")" -eq "$EXPECTED_PREFIX_PAIRS" ]]; then
    set_prefix_status complete "already complete pairs=$EXPECTED_PREFIX_PAIRS"
    return 0
  fi
  wait_for_gpu "$PREFIX_GPU_ID" prefix
  set_prefix_status running "gpu=$PREFIX_GPU_ID pairs=$EXPECTED_PREFIX_PAIRS"
  cd "$EVAL_RUNTIME"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$PREFIX_GPU_ID" \
    "$PYTHON_BIN" "$SCORE_RUNTIME/src/rl/diagnostics/score_action_candidates.py" \
      --dataset "$FIXED_PREFIX_DATASET" \
      --checkpoint-name "$CANDIDATE_NAME" \
      --base-model /home/dengyan/models/Qwen2.5-Coder-7B-Instruct \
      --adapter "$ADAPTER" \
      --output-jsonl "$PREFIX_SCORE" \
      --output-parquet "$PREFIX_PARQUET" \
      --manifest "$PREFIX_MANIFEST" \
      --device cuda:0
  test "$(nonempty_lines "$PREFIX_SCORE")" -eq "$EXPECTED_PREFIX_PAIRS"
  set_prefix_status complete "pairs=$EXPECTED_PREFIX_PAIRS score=$PREFIX_SCORE"
}

run_full_dev() {
  set -euo pipefail
  wait_for_gpu "$GREEDY_GPU_ID" full_dev
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ADAPTER="$ADAPTER" ADAPTER_SHA256="$ADAPTER_SHA256" \
  SERVED_MODEL="stage1-$CANDIDATE_NAME-full-dev-greedy" \
  RESULT_DIR="$GREEDY_RESULT" STATUS="$FULL_STATUS" RUN_LOG="$FULL_LOG" \
  PORT="$PORT" EVAL_GPU_ID="$GREEDY_GPU_ID" RUNTIME="$EVAL_RUNTIME" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
    bash "$LAUNCH_RUNTIME/src/rl/experiments/run_full_dev_greedy_table_rl.sh"
}

if [[ "$SERIAL_EVALUATION" == 1 ]]; then
  if [[ "$PREFIX_GPU_ID" != "$GREEDY_GPU_ID" ]]; then
    printf 'serial evaluation requires one shared GPU id\n' >&2
    exit 2
  fi
  set_status running_serial "gpu=$PREFIX_GPU_ID stage=fixed_prefix_then_full_dev"
  set +e
  run_fixed_prefix
  prefix_code=$?
  if [[ "$prefix_code" -eq 0 ]]; then
    run_full_dev
    full_code=$?
  else
    full_code=125
  fi
  set -e
else
  set_status running_parallel "prefix_gpu=$PREFIX_GPU_ID full_dev_gpu=$GREEDY_GPU_ID"
  run_fixed_prefix &
  prefix_pid=$!
  run_full_dev &
  full_pid=$!
  set +e
  wait "$prefix_pid"
  prefix_code=$?
  wait "$full_pid"
  full_code=$?
  set -e
fi
if [[ "$prefix_code" -ne 0 || "$full_code" -ne 0 ]]; then
  printf 'evaluation child failed: prefix_exit=%s full_exit=%s\n' "$prefix_code" "$full_code" >&2
  exit 1
fi

set_status checking_requirements "candidate=$CANDIDATE_NAME"
"$PYTHON_BIN" "$LAUNCH_RUNTIME/src/rl/experiments/evaluate_stage1_candidate_gate.py" \
  --candidate-name "$CANDIDATE_NAME" \
  --sft2-greedy "$SFT2_GREEDY" \
  --candidate-greedy "$GREEDY_RESULT/all.jsonl" \
  --sft2-prefix "$SFT2_PREFIX" \
  --candidate-prefix "$PREFIX_SCORE" \
  --expected-questions "$EXPECTED_QUESTIONS" \
  --output "$DECISION"

if [[ -f "$ORIGINAL_DECISION" ]]; then
  "$PYTHON_BIN" - "$DECISION" "$ORIGINAL_DECISION" "$COMPARISON" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

expanded = json.loads(Path(sys.argv[1]).read_text())
original = json.loads(Path(sys.argv[2]).read_text())

def metric(payload, family, name):
    return payload["candidate_metrics"][family][name]

payload = {
    "schema_version": "expanded-vs-original-exp15-stage1-comparison-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "expanded_candidate": expanded["candidate"],
    "original_candidate": original["candidate"],
    "expanded_gate_status": expanded["status"],
    "original_gate_status": original["status"],
    "expanded": expanded["candidate_metrics"],
    "original": original["candidate_metrics"],
    "expanded_minus_original": {
        "greedy_correct": metric(expanded, "greedy", "correct") - metric(original, "greedy", "correct"),
        "greedy_at_1_pp": 100 * (metric(expanded, "greedy", "greedy_at_1") - metric(original, "greedy", "greedy_at_1")),
        "valid_rate_pp": 100 * (metric(expanded, "greedy", "valid_rate") - metric(original, "greedy", "valid_rate")),
        "fixed_prefix_top1_pp": 100 * (metric(expanded, "fixed_prefix", "top1_accuracy") - metric(original, "fixed_prefix", "top1_accuracy")),
        "fixed_prefix_mean_margin": metric(expanded, "fixed_prefix", "mean_margin") - metric(original, "fixed_prefix", "mean_margin"),
    },
    "note": "Aggregate comparison on the same frozen evaluation artifacts; not a paired significance test.",
}
Path(sys.argv[3]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY
fi

passed=$("$PYTHON_BIN" - "$DECISION" <<'PY'
import json
import sys
print(int(json.loads(open(sys.argv[1]).read())["status"] == "passed"))
PY
)
if [[ "$passed" == 1 ]]; then
  set_status complete "requirements=passed full_dev=complete fixed_prefix=complete k4=deferred_to_finalist"
else
  set_status complete "requirements=failed full_dev=complete fixed_prefix=complete k4=skipped"
fi
