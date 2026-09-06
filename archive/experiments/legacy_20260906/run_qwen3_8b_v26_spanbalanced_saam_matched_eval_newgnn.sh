#!/usr/bin/env bash
# Fail-closed matched evaluation handoff for the span-balanced SAAM diagnostic.
#
# The script intentionally evaluates only completed adapters.  It validates
# both training artifacts before starting vLLM, isolates generated Python
# caches from the frozen runtime, launches the two arms on separate GPUs, and
# propagates each evaluator's real exit status instead of treating process
# creation as success.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODE=${1:---run}
if [[ "$MODE" != --run && "$MODE" != --plan ]]; then
  echo "usage: $0 [--plan|--run]" >&2
  exit 2
fi

PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_8b_v26_saam_spanbalanced_gate60_20260830}
CONTRACT=${CONTRACT:-$RUNTIME/src/rl/evaluation/qwen3_8b_v26_vanilla_formal_matched_contract.json}
EVALUATOR=${EVALUATOR:-$RUNTIME/src/rl/evaluation/run_qwen3_8b_v26_saam_candidate_only_eval.py}
FINAL_STEP=${FINAL_STEP:-4}
ANALYZER=${ANALYZER:-$RUNTIME/src/rl/diagnostics/analyze_evaluation_results.py}
EXAMPLES=${EXAMPLES:-/home/dengyan/tabular_rl_project/data/eval_inputs/bird_dev_20240627.jsonl}
EXPECTED_COUNT=${EXPECTED_COUNT:-1534}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs/evaluations/qwen3_8b_atomic_v26_saam_spanbalanced_gate60_20260830}
ANALYSIS_OUTPUT=${ANALYSIS_OUTPUT:-$OUTPUT_ROOT/matched_analysis.json}
CANDIDATE_RUN=${CANDIDATE_RUN:-/home/dengyan/tabular_rl_outputs/qwen3_8b_atomic_v26_saam_spanbalanced_gate60_20260830/run1}
SAAM_RUN=${SAAM_RUN:-/home/dengyan/tabular_rl_outputs/saam_reference_run3}
CANDIDATE_DIR=${CANDIDATE_DIR:-$OUTPUT_ROOT/candidate_full_handoff}
SAAM_DIR=${SAAM_DIR:-$OUTPUT_ROOT/saam_reference_full_handoff}
CANDIDATE_GPU=${CANDIDATE_GPU:-0}
SAAM_GPU=${SAAM_GPU:-3}
CANDIDATE_PORT=${CANDIDATE_PORT:-8087}
SAAM_PORT=${SAAM_PORT:-8088}

die() {
  echo "matched eval handoff: $*" >&2
  exit 2
}

[[ -x "$PYTHON" ]] || die "Python executable is missing: $PYTHON"
[[ -d "$RUNTIME" ]] || die "runtime is missing: $RUNTIME"
[[ -f "$CONTRACT" ]] || die "contract is missing: $CONTRACT"
[[ -f "$EVALUATOR" ]] || die "candidate-only evaluator is missing: $EVALUATOR"
[[ -f "$ANALYZER" ]] || die "evaluation analyzer is missing: $ANALYZER"
[[ -f "$EXAMPLES" ]] || die "evaluation examples are missing: $EXAMPLES"
[[ ! -e "$CANDIDATE_DIR" ]] || die "candidate output already exists: $CANDIDATE_DIR"
[[ ! -e "$SAAM_DIR" ]] || die "SAAM output already exists: $SAAM_DIR"
[[ ! -e "$ANALYSIS_OUTPUT" ]] || die "analysis output already exists: $ANALYSIS_OUTPUT"

require_completed_run() {
  local run=$1
  [[ -f "$run/run_manifest.json" ]] || die "missing run manifest: $run"
  [[ -f "$run/implementation_lock.json" ]] || die "missing implementation lock: $run"
  [[ -f "$run/training_precision.json" ]] || die "missing precision contract: $run"
  [[ -f "$run/checkpoint-$FINAL_STEP/trainer_state.json" ]] || die "missing checkpoint-$FINAL_STEP trainer state: $run"
  [[ -f "$run/checkpoint-$FINAL_STEP/adapter_model.safetensors" ]] || die "missing checkpoint-$FINAL_STEP adapter: $run"
  [[ -f "$run/checkpoint-$FINAL_STEP/adapter_config.json" ]] || die "missing checkpoint-$FINAL_STEP adapter config: $run"
  [[ -f "$run/final/adapter_model.safetensors" ]] || die "missing final adapter: $run"
  [[ -f "$run/final/adapter_config.json" ]] || die "missing final adapter config: $run"
  [[ -d "$run/implementation_source_snapshot" ]] || die "missing source snapshot: $run"
}

require_completed_run "$CANDIDATE_RUN"
require_completed_run "$SAAM_RUN"

if [[ "$MODE" == --plan ]]; then
  cat <<EOF
matched-eval handoff plan
contract=$CONTRACT
evaluator=$EVALUATOR
candidate_run=$CANDIDATE_RUN
saam_run=$SAAM_RUN
candidate_dir=$CANDIDATE_DIR
saam_dir=$SAAM_DIR
final_step=$FINAL_STEP
analyzer=$ANALYZER
examples=$EXAMPLES expected_count=$EXPECTED_COUNT
analysis_output=$ANALYSIS_OUTPUT
candidate_gpu=$CANDIDATE_GPU candidate_port=$CANDIDATE_PORT
saam_gpu=$SAAM_GPU saam_port=$SAAM_PORT
EOF
  exit 0
fi

# The frozen runtime hash intentionally excludes Python bytecode.  Training or
# a previous evaluator can create a cache file in the exported tree, so move
# only those generated files to a private, reversible quarantine.  The trap
# restores them even when either arm fails.
CACHE_BACKUP=$(mktemp -d /tmp/v26-runtime-cache.XXXXXX)
restore_caches() {
  local status=$?
  if [[ -d "$CACHE_BACKUP" ]]; then
    while IFS= read -r -d '' backup; do
      local relative=${backup#"$CACHE_BACKUP"/}
      local target="$RUNTIME/$relative"
      mkdir -p "$(dirname "$target")"
      mv "$backup" "$target"
    done < <(find "$CACHE_BACKUP" -type f -print0)
    find "$CACHE_BACKUP" -depth -type d -empty -delete 2>/dev/null || true
    rmdir "$CACHE_BACKUP" 2>/dev/null || true
  fi
  exit "$status"
}
trap restore_caches EXIT INT TERM

while IFS= read -r -d '' cache; do
  relative=${cache#"$RUNTIME"/}
  destination="$CACHE_BACKUP/$relative"
  mkdir -p "$(dirname "$destination")"
  mv "$cache" "$destination"
done < <(find "$RUNTIME/src/eval" "$RUNTIME/src/sft" "$RUNTIME/src/harness" \
  -type f \( -path '*/__pycache__/*' -o -name '*.pyc' \) -print0)

mkdir -p "$OUTPUT_ROOT"

run_arm() {
  local name=$1
  local training_run=$2
  local run_dir=$3
  local gpu=$4
  local port=$5
  local log=$6
  mkdir -p "$(dirname "$run_dir")"
  (
    cd "$RUNTIME"
    PYTHONPATH="$RUNTIME" "$PYTHON" "$EVALUATOR" \
      --contract "$CONTRACT" \
      --training-run "$training_run" \
      --run-dir "$run_dir" \
      --expected-global-step "$FINAL_STEP" \
      --gpu-id "$gpu" \
      --port "$port"
  ) >"$log" 2>&1
  local status=$?
  echo "$name exit=$status $(date -Is)" >>"$OUTPUT_ROOT/eval_status.log"
  return "$status"
}

set +e
run_arm candidate "$CANDIDATE_RUN" "$CANDIDATE_DIR" "$CANDIDATE_GPU" \
  "$CANDIDATE_PORT" "$OUTPUT_ROOT/candidate_handoff.log" &
candidate_pid=$!
run_arm saam "$SAAM_RUN" "$SAAM_DIR" "$SAAM_GPU" \
  "$SAAM_PORT" "$OUTPUT_ROOT/saam_handoff.log" &
saam_pid=$!
wait "$candidate_pid"
candidate_status=$?
wait "$saam_pid"
saam_status=$?
set -e

if [[ "$candidate_status" -ne 0 || "$saam_status" -ne 0 ]]; then
  echo "candidate_status=$candidate_status saam_status=$saam_status" >&2
  exit 1
fi

MATCH_FIELDS=(
  base_model base_model_revision base_model_identity_sha256
  dataset input_sha256 task_identity_sha256_without_db_path
  protocol_version protocol_hash runtime runtime_sha256
  serving concurrency decode agent tool_execution_timeout_seconds
)
analysis_args=(
  --examples "$EXAMPLES"
  --arm "candidate=$CANDIDATE_DIR/final/result/all.jsonl"
  --arm "saam=$SAAM_DIR/final/result/all.jsonl"
  --compare candidate:saam
  --expected-count "$EXPECTED_COUNT"
  --protocol-version version26
  --protocol-hash 4da19387399bd3a5
  --temperature 0 --top-p 1
  --denotation-comparison bird-set
  --identity "candidate=$CANDIDATE_DIR/final/evaluation_identity.json"
  --identity "saam=$SAAM_DIR/final/evaluation_identity.json"
  --require-identities --require-distinct-adapters
  --output "$ANALYSIS_OUTPUT"
  --overwrite
)
for field in "${MATCH_FIELDS[@]}"; do
  analysis_args+=(--match-identity-field "$field")
done
(
  cd "$RUNTIME"
  PYTHONPATH="$RUNTIME" "$PYTHON" "$ANALYZER" "${analysis_args[@]}"
)
echo "matched evals completed $(date -Is)" >>"$OUTPUT_ROOT/eval_status.log"
