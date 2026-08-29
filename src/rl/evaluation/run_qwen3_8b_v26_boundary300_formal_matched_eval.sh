#!/usr/bin/env bash
# Strict fresh SFT1-vs-boundary300-final full-dev evaluation.  The default is
# a read-only local dry run; preflight/run require five explicit frozen hashes.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODE=${1:-dry-run}
if [[ $# -gt 1 ]]; then
  printf 'usage: %s [dry-run|preflight|run]\n' "$0" >&2
  exit 2
fi
case "$MODE" in
  dry-run|preflight|run) ;;
  *) printf 'usage: %s [dry-run|preflight|run]\n' "$0" >&2; exit 2 ;;
esac

HERE=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$HERE/../../.." && pwd)
BOUNDARY_CONTROLLER=$HERE/boundary300_v26_matched_eval.py
FORMAL_CONTROLLER=$HERE/formal_v26_matched_eval.py
WRAPPER=$HERE/formal_v26_rollout_passk.py
CONTRACT=$HERE/qwen3_8b_v26_boundary300_formal_matched_contract.json
ANALYZER=$PROJECT_ROOT/src/rl/diagnostics/analyze_evaluation_results.py
GATE=$PROJECT_ROOT/src/rl/diagnostics/audit_vanilla_grpo_matched_gate.py

EXPECTED_BOUNDARY_CONTROLLER_SHA256=8b0dc5c31cc87385a87dba6d8be7a2775d560bd6a55772b5c52fe277c6ca53b6
EXPECTED_FORMAL_CONTROLLER_SHA256=2d8fe217368ef95924f4296a2a11fb3beb3c550ad9ef549614346ef11f87dc01
EXPECTED_WRAPPER_SHA256=b55dc000504a3f3633db02d9cb3afc6a80b6b86d79c7cfc5a3a7f8a7985df1fd
EXPECTED_CONTRACT_SHA256=8e14f965c3c5fb088f13ea23ea2051680bca9bbf92a7646b2b77b90b3462cce7
EXPECTED_ANALYZER_SHA256=abbfa3d2786d14b5ab20542e4f77d6d41d636f50d25a8c1d7bacbb901685d274
EXPECTED_GATE_SHA256=ef3551706fa6f9bdde7e389a9bc69998a0a4c886d4ddc0710d4966a0abaa41ca

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
require_sha() {
  local path=$1 expected=$2 actual
  [[ -f "$path" && ! -L "$path" ]] || {
    printf 'missing/non-regular pinned file: %s\n' "$path" >&2
    exit 3
  }
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] || {
    printf 'SHA-256 mismatch: expected=%s actual=%s path=%s\n' \
      "$expected" "$actual" "$path" >&2
    exit 3
  }
}

require_sha "$BOUNDARY_CONTROLLER" "$EXPECTED_BOUNDARY_CONTROLLER_SHA256"
require_sha "$FORMAL_CONTROLLER" "$EXPECTED_FORMAL_CONTROLLER_SHA256"
require_sha "$WRAPPER" "$EXPECTED_WRAPPER_SHA256"
require_sha "$CONTRACT" "$EXPECTED_CONTRACT_SHA256"
require_sha "$ANALYZER" "$EXPECTED_ANALYZER_SHA256"
require_sha "$GATE" "$EXPECTED_GATE_SHA256"

HOST_PYTHON=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
if [[ "$MODE" == dry-run && ! -x "$HOST_PYTHON" ]]; then
  HOST_PYTHON=$(command -v python3)
fi
[[ -x "$HOST_PYTHON" ]] || {
  printf 'evaluation Python is missing: %s\n' "$HOST_PYTHON" >&2
  exit 3
}

if [[ "$MODE" == dry-run ]]; then
  exec env PYTHONPATH="$PROJECT_ROOT" "$HOST_PYTHON" "$BOUNDARY_CONTROLLER" \
    --print-plan --contract "$CONTRACT"
fi

[[ -n "${EVAL_GPU_ID:-}" && "$EVAL_GPU_ID" =~ ^[0-9]+$ ]] || {
  printf 'EVAL_GPU_ID must explicitly select one physical GPU\n' >&2
  exit 2
}

FROZEN_BOUNDARY_MANIFEST_SHA256=${FROZEN_BOUNDARY_MANIFEST_SHA256:-}
FROZEN_TRAIN300_SHA256=${FROZEN_TRAIN300_SHA256:-}
FROZEN_TRAIN_RUN_MANIFEST_SHA256=${FROZEN_TRAIN_RUN_MANIFEST_SHA256:-}
FROZEN_TRAIN_IMPLEMENTATION_LOCK_SHA256=${FROZEN_TRAIN_IMPLEMENTATION_LOCK_SHA256:-}
FROZEN_FINAL_CHECKPOINT20_ADAPTER_SHA256=${FROZEN_FINAL_CHECKPOINT20_ADAPTER_SHA256:-}
BOUNDARY_MANIFEST_PATH=${BOUNDARY_MANIFEST_PATH:-}
TRAIN300_PATH=${TRAIN300_PATH:-}
TRAINING_RUN_PATH=${TRAINING_RUN_PATH:-}
for binding_name in \
  FROZEN_BOUNDARY_MANIFEST_SHA256 \
  FROZEN_TRAIN300_SHA256 \
  FROZEN_TRAIN_RUN_MANIFEST_SHA256 \
  FROZEN_TRAIN_IMPLEMENTATION_LOCK_SHA256 \
  FROZEN_FINAL_CHECKPOINT20_ADAPTER_SHA256
do
  binding_value=${!binding_name}
  [[ "$binding_value" =~ ^[0-9a-f]{64}$ ]] || {
    printf '%s must be an explicit 64-character lowercase SHA-256\n' \
      "$binding_name" >&2
    exit 2
  }
done
for path_name in BOUNDARY_MANIFEST_PATH TRAIN300_PATH TRAINING_RUN_PATH; do
  path_value=${!path_name}
  [[ "$path_value" == /* ]] || {
    printf '%s must be an explicit absolute path\n' "$path_name" >&2
    exit 2
  }
done

PORT=${EVAL_PORT:-8087}
MAXIMUM_USED_MIB=${EVAL_MAXIMUM_USED_MIB:-512}
COMMON_ARGS=(
  --contract "$CONTRACT"
  --gpu-id "$EVAL_GPU_ID"
  --port "$PORT"
  --maximum-used-mib "$MAXIMUM_USED_MIB"
  --boundary-manifest-sha256 "$FROZEN_BOUNDARY_MANIFEST_SHA256"
  --train300-sha256 "$FROZEN_TRAIN300_SHA256"
  --training-run-manifest-sha256 "$FROZEN_TRAIN_RUN_MANIFEST_SHA256"
  --training-implementation-lock-sha256 "$FROZEN_TRAIN_IMPLEMENTATION_LOCK_SHA256"
  --final-checkpoint20-adapter-sha256 "$FROZEN_FINAL_CHECKPOINT20_ADAPTER_SHA256"
  --boundary-manifest-path "$BOUNDARY_MANIFEST_PATH"
  --train300-path "$TRAIN300_PATH"
  --training-run-path "$TRAINING_RUN_PATH"
)

if [[ "$MODE" == preflight ]]; then
  exec env PYTHONPATH="$PROJECT_ROOT" "$HOST_PYTHON" "$BOUNDARY_CONTROLLER" \
    --preflight "${COMMON_ARGS[@]}"
fi

RUN_DIR=${RUN_DIR:-/home/dengyan/tabular_rl_outputs/evaluations/qwen3_8b_atomic_v26_boundary300_vanilla_formal/formal_sft1_vs_boundary300_final_20260812}
exec env PYTHONPATH="$PROJECT_ROOT" "$HOST_PYTHON" "$BOUNDARY_CONTROLLER" \
  --run --run-dir "$RUN_DIR" "${COMMON_ARGS[@]}"
