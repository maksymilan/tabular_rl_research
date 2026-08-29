#!/usr/bin/env bash
# Formal fresh SFT1-vs-final matched evaluation. The default is a local plan;
# only an explicit --run may allocate the caller-selected, already-idle GPU.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODE=${1:---plan}
if [[ $# -gt 1 || ( "$MODE" != --plan && "$MODE" != --preflight && "$MODE" != --run ) ]]; then
  printf 'usage: %s [--plan|--preflight|--run]\n' "$0" >&2
  exit 2
fi

HERE=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$HERE/../../.." && pwd)
CONTROLLER=$HERE/formal_v26_matched_eval.py
WRAPPER=$HERE/formal_v26_rollout_passk.py
CONTRACT=$HERE/qwen3_8b_v26_vanilla_formal_matched_contract.json
ANALYZER=$PROJECT_ROOT/src/rl/diagnostics/analyze_evaluation_results.py
GATE=$PROJECT_ROOT/src/rl/diagnostics/audit_vanilla_grpo_matched_gate.py

EXPECTED_CONTROLLER_SHA256=2d8fe217368ef95924f4296a2a11fb3beb3c550ad9ef549614346ef11f87dc01
EXPECTED_WRAPPER_SHA256=b55dc000504a3f3633db02d9cb3afc6a80b6b86d79c7cfc5a3a7f8a7985df1fd
EXPECTED_CONTRACT_SHA256=b0f15dff66e2e19f4baeb8df358b3a3eeeb9cb523c7871a4ecb23202754eb6e3
EXPECTED_ANALYZER_SHA256=abbfa3d2786d14b5ab20542e4f77d6d41d636f50d25a8c1d7bacbb901685d274
EXPECTED_GATE_SHA256=ef3551706fa6f9bdde7e389a9bc69998a0a4c886d4ddc0710d4966a0abaa41ca

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
require_sha() {
  local path=$1 expected=$2 actual
  [[ -f "$path" ]] || { printf 'missing pinned file: %s\n' "$path" >&2; exit 3; }
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] || {
    printf 'SHA-256 mismatch: expected=%s actual=%s path=%s\n' "$expected" "$actual" "$path" >&2
    exit 3
  }
}

require_sha "$CONTROLLER" "$EXPECTED_CONTROLLER_SHA256"
require_sha "$WRAPPER" "$EXPECTED_WRAPPER_SHA256"
require_sha "$CONTRACT" "$EXPECTED_CONTRACT_SHA256"
require_sha "$ANALYZER" "$EXPECTED_ANALYZER_SHA256"
require_sha "$GATE" "$EXPECTED_GATE_SHA256"

HOST_PYTHON=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
if [[ "$MODE" == --plan && ! -x "$HOST_PYTHON" ]]; then
  HOST_PYTHON=$(command -v python3)
fi
[[ -x "$HOST_PYTHON" ]] || { printf 'evaluation Python is missing: %s\n' "$HOST_PYTHON" >&2; exit 3; }

case "$MODE" in
  --plan)
    exec env PYTHONPATH= "$HOST_PYTHON" "$CONTROLLER" \
      --print-plan --contract "$CONTRACT"
    ;;
  --preflight|--run)
    [[ -n "${EVAL_GPU_ID:-}" && "$EVAL_GPU_ID" =~ ^[0-9]+$ ]] || {
      printf 'EVAL_GPU_ID must explicitly select one physical GPU\n' >&2
      exit 2
    }
    PORT=${EVAL_PORT:-8087}
    MAXIMUM_USED_MIB=${EVAL_MAXIMUM_USED_MIB:-512}
    if [[ "$MODE" == --preflight ]]; then
      exec env PYTHONPATH= "$HOST_PYTHON" "$CONTROLLER" \
        --preflight --contract "$CONTRACT" --gpu-id "$EVAL_GPU_ID" \
        --port "$PORT" --maximum-used-mib "$MAXIMUM_USED_MIB"
    fi
    RUN_DIR=${RUN_DIR:-/home/dengyan/tabular_rl_outputs/evaluations/qwen3_8b_atomic_v26_vanilla_formal/formal_sft1_vs_final_20260812}
    exec env PYTHONPATH= "$HOST_PYTHON" "$CONTROLLER" \
      --run --contract "$CONTRACT" --run-dir "$RUN_DIR" \
      --gpu-id "$EVAL_GPU_ID" --port "$PORT" \
      --maximum-used-mib "$MAXIMUM_USED_MIB"
    ;;
esac
