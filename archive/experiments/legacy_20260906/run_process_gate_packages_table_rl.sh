#!/usr/bin/env bash
# Rebuild the 703-trajectory grounding packages with the current provenance code.
# This is a CPU-only package-construction step; it does not perform or approve
# the independent external review.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
GATE_ROOT=${GATE_ROOT:-"$OUTPUT_ROOT/process_gate_v2_20260730"}
RUNTIME=${RUNTIME:-"$GATE_ROOT/runtime"}
INPUT_DIR=${INPUT_DIR:-"$GATE_ROOT/inputs"}
RESULT_DIR=${RESULT_DIR:-"$GATE_ROOT/packages_current_provenance"}
STATUS=${STATUS:-"$OUTPUT_ROOT/logs/process_gate_packages_703_20260730.status"}
RUN_LOG=${RUN_LOG:-"$OUTPUT_ROOT/logs/process_gate_packages_703_20260730.run.log"}
LOCK_FILE=${LOCK_FILE:-"$OUTPUT_ROOT/logs/process_gate_packages_703_20260730.lock"}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
on_exit() {
  local exit_status=$?
  trap - EXIT
  if [[ "$exit_status" -ne 0 ]]; then
    set_status failed "exit=$exit_status see=$RUN_LOG"
  fi
  exit "$exit_status"
}
trap on_exit EXIT

mkdir -p "$OUTPUT_ROOT/logs" "$RESULT_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'process gate package builder already running\n' >&2
  exit 1
fi
exec >>"$RUN_LOG" 2>&1

required=(
  "$INPUT_DIR/version24_fixed200_first50_bird_set.jsonl"
  "$INPUT_DIR/version24_fixed200_remaining150_bird_set.jsonl"
  "$INPUT_DIR/additional800_bird_set.jsonl"
  "$INPUT_DIR/provider_retry_bird_set.jsonl"
  "$INPUT_DIR/provider_retry_second_bird_set.jsonl"
  "$INPUT_DIR/bird_external_teacher_fixed1000_full_prefix_qwen25_6400.index.jsonl"
)
for path in "${required[@]}"; do
  if [[ ! -s "$path" ]]; then
    set_status failed "missing_input=$path"
    exit 1
  fi
done
if [[ ! -f "$RUNTIME/src/rl/review_grounding_edges_external.py" ]]; then
  set_status failed "missing_runtime=$RUNTIME"
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  set_status failed "missing_python=$PYTHON"
  exit 1
fi

packages="$RESULT_DIR/packages703_current_provenance.jsonl"
reviews="$RESULT_DIR/reviews.placeholder.jsonl"
summary="$RESULT_DIR/packages_summary.json"
set_status running "packages_only expected=703"
cd "$RUNTIME"
PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  "$PYTHON" -u src/rl/review_grounding_edges_external.py \
  --input "$INPUT_DIR/version24_fixed200_first50_bird_set.jsonl" \
  --input "$INPUT_DIR/version24_fixed200_remaining150_bird_set.jsonl" \
  --input "$INPUT_DIR/additional800_bird_set.jsonl" \
  --input "$INPUT_DIR/provider_retry_bird_set.jsonl" \
  --input "$INPUT_DIR/provider_retry_second_bird_set.jsonl" \
  --sft-index "$INPUT_DIR/bird_external_teacher_fixed1000_full_prefix_qwen25_6400.index.jsonl" \
  --packages-output "$packages" \
  --reviews-output "$reviews" \
  --summary-output "$summary" \
  --packages-only

rows=$(wc -l <"$packages" | tr -d '[:space:]')
"$PYTHON" - "$summary" "$rows" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
rows = int(sys.argv[2])
assert rows == 703, rows
assert summary["trajectories"] == 703, summary
assert summary["packages_only"] is True, summary
assert summary["grounding_precision_audit_approved"] is False, summary
PY
set_status complete "packages=$rows external_review_pending=true result=$RESULT_DIR"
