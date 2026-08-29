#!/usr/bin/env bash
# Strict fresh SFT1-vs-Arm-B-final32 full-dev evaluation.  Dry-run is the
# default and is read-only; active modes require every produced artifact hash.
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
ARM_B_CONTROLLER=$HERE/arm_b_v26_matched_eval.py
FORMAL_CONTROLLER=$HERE/formal_v26_matched_eval.py
WRAPPER=$HERE/formal_v26_rollout_passk.py
OVERLAY=$HERE/qwen3_8b_v26_arm_b_formal_matched_contract.json
COMMON_BASE=$HERE/qwen3_8b_v26_boundary300_formal_matched_contract.json
ANALYZER=$PROJECT_ROOT/src/rl/diagnostics/analyze_evaluation_results.py
GATE=$PROJECT_ROOT/src/rl/diagnostics/audit_vanilla_grpo_matched_gate.py
ARM_B_SHARED_CONTRACT=$PROJECT_ROOT/src/rl/vanilla_grpo_arm_b.py
ARM_B_INPUT_VALIDATOR=$PROJECT_ROOT/src/rl/diagnostics/validate_arm_b_training_inputs.py
TRIGGER_PREPARER=$PROJECT_ROOT/src/rl/diagnostics/prepare_vanilla_grpo_confirmatory_arm_trigger.py

EXPECTED_ARM_B_CONTROLLER_SHA256=76ecf74c741551c61b3c5abf4da004239eef6f3c0997ffbe39333ea37b90c020
EXPECTED_FORMAL_CONTROLLER_SHA256=2d8fe217368ef95924f4296a2a11fb3beb3c550ad9ef549614346ef11f87dc01
EXPECTED_WRAPPER_SHA256=b55dc000504a3f3633db02d9cb3afc6a80b6b86d79c7cfc5a3a7f8a7985df1fd
EXPECTED_OVERLAY_SHA256=c7ad3326573345cf8bb642a7bc6be92ac2ea573212503dde8f10fbd57ec7af33
EXPECTED_COMMON_BASE_SHA256=8e14f965c3c5fb088f13ea23ea2051680bca9bbf92a7646b2b77b90b3462cce7
EXPECTED_ANALYZER_SHA256=abbfa3d2786d14b5ab20542e4f77d6d41d636f50d25a8c1d7bacbb901685d274
EXPECTED_GATE_SHA256=ef3551706fa6f9bdde7e389a9bc69998a0a4c886d4ddc0710d4966a0abaa41ca
EXPECTED_ARM_B_SHARED_CONTRACT_SHA256=d596a01a89a125ec0375af7d3a6dbdd2c1b5a65a2b625dc8d07cb502c29e5a80
EXPECTED_ARM_B_INPUT_VALIDATOR_SHA256=a416b9490e8c6287c7f05fff5c2c314c528b8524dddbd782c5e25ed6eb7125e0
EXPECTED_TRIGGER_PREPARER_SHA256=554fe25b39fe7c012a73b8d4045762476347bfd6fbcc70355f98f4cb8bdd7b98

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

require_sha "$ARM_B_CONTROLLER" "$EXPECTED_ARM_B_CONTROLLER_SHA256"
require_sha "$FORMAL_CONTROLLER" "$EXPECTED_FORMAL_CONTROLLER_SHA256"
require_sha "$WRAPPER" "$EXPECTED_WRAPPER_SHA256"
require_sha "$OVERLAY" "$EXPECTED_OVERLAY_SHA256"
require_sha "$COMMON_BASE" "$EXPECTED_COMMON_BASE_SHA256"
require_sha "$ANALYZER" "$EXPECTED_ANALYZER_SHA256"
require_sha "$GATE" "$EXPECTED_GATE_SHA256"
require_sha "$ARM_B_SHARED_CONTRACT" "$EXPECTED_ARM_B_SHARED_CONTRACT_SHA256"
require_sha "$ARM_B_INPUT_VALIDATOR" "$EXPECTED_ARM_B_INPUT_VALIDATOR_SHA256"
require_sha "$TRIGGER_PREPARER" "$EXPECTED_TRIGGER_PREPARER_SHA256"

HOST_PYTHON=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
if [[ "$MODE" == dry-run && ! -x "$HOST_PYTHON" ]]; then
  HOST_PYTHON=$(command -v python3)
fi
[[ -x "$HOST_PYTHON" ]] || {
  printf 'evaluation Python is missing: %s\n' "$HOST_PYTHON" >&2
  exit 3
}

if [[ "$MODE" == dry-run ]]; then
  exec env PYTHONPATH="$PROJECT_ROOT" "$HOST_PYTHON" "$ARM_B_CONTROLLER" \
    --print-plan --overlay "$OVERLAY" --base-contract "$COMMON_BASE"
fi

[[ -n "${EVAL_GPU_ID:-}" && "$EVAL_GPU_ID" =~ ^[0-9]+$ ]] || {
  printf 'EVAL_GPU_ID must explicitly select one physical GPU\n' >&2
  exit 2
}

FROZEN_ARM_B_SELECTION_MANIFEST_SHA256=${FROZEN_ARM_B_SELECTION_MANIFEST_SHA256:-}
FROZEN_TRAIN320_SHA256=${FROZEN_TRAIN320_SHA256:-}
FROZEN_VALIDATION64_AUDIT_SHA256=${FROZEN_VALIDATION64_AUDIT_SHA256:-}
FROZEN_TRAIN_RUN_MANIFEST_SHA256=${FROZEN_TRAIN_RUN_MANIFEST_SHA256:-}
FROZEN_TRAIN_IMPLEMENTATION_LOCK_SHA256=${FROZEN_TRAIN_IMPLEMENTATION_LOCK_SHA256:-}
FROZEN_ARM_B_TRAINING_CONTRACT_SHA256=${FROZEN_ARM_B_TRAINING_CONTRACT_SHA256:-}
FROZEN_FINAL_CHECKPOINT32_ADAPTER_SHA256=${FROZEN_FINAL_CHECKPOINT32_ADAPTER_SHA256:-}
FROZEN_CONFIRMATORY_TRIGGER_SHA256=${FROZEN_CONFIRMATORY_TRIGGER_SHA256:-}
ARM_B_SELECTION_MANIFEST_PATH=${ARM_B_SELECTION_MANIFEST_PATH:-}
TRAIN320_PATH=${TRAIN320_PATH:-}
VALIDATION64_AUDIT_PATH=${VALIDATION64_AUDIT_PATH:-}
TRAINING_RUN_PATH=${TRAINING_RUN_PATH:-}
CONFIRMATORY_TRIGGER_PATH=${CONFIRMATORY_TRIGGER_PATH:-}

for binding_name in \
  FROZEN_ARM_B_SELECTION_MANIFEST_SHA256 \
  FROZEN_TRAIN320_SHA256 \
  FROZEN_VALIDATION64_AUDIT_SHA256 \
  FROZEN_TRAIN_RUN_MANIFEST_SHA256 \
  FROZEN_TRAIN_IMPLEMENTATION_LOCK_SHA256 \
  FROZEN_ARM_B_TRAINING_CONTRACT_SHA256 \
  FROZEN_FINAL_CHECKPOINT32_ADAPTER_SHA256 \
  FROZEN_CONFIRMATORY_TRIGGER_SHA256
do
  binding_value=${!binding_name}
  [[ "$binding_value" =~ ^[0-9a-f]{64}$ ]] || {
    printf '%s must be an explicit 64-character lowercase SHA-256\n' \
      "$binding_name" >&2
    exit 2
  }
done
for path_name in \
  ARM_B_SELECTION_MANIFEST_PATH TRAIN320_PATH VALIDATION64_AUDIT_PATH \
  TRAINING_RUN_PATH CONFIRMATORY_TRIGGER_PATH
do
  path_value=${!path_name}
  [[ "$path_value" == /* ]] || {
    printf '%s must be an explicit absolute path\n' "$path_name" >&2
    exit 2
  }
done

PORT=${EVAL_PORT:-8088}
MAXIMUM_USED_MIB=${EVAL_MAXIMUM_USED_MIB:-512}
COMMON_ARGS=(
  --overlay "$OVERLAY"
  --base-contract "$COMMON_BASE"
  --gpu-id "$EVAL_GPU_ID"
  --port "$PORT"
  --maximum-used-mib "$MAXIMUM_USED_MIB"
  --selection-manifest-sha256 "$FROZEN_ARM_B_SELECTION_MANIFEST_SHA256"
  --train320-sha256 "$FROZEN_TRAIN320_SHA256"
  --validation64-audit-sha256 "$FROZEN_VALIDATION64_AUDIT_SHA256"
  --training-run-manifest-sha256 "$FROZEN_TRAIN_RUN_MANIFEST_SHA256"
  --training-implementation-lock-sha256 "$FROZEN_TRAIN_IMPLEMENTATION_LOCK_SHA256"
  --arm-b-training-contract-sha256 "$FROZEN_ARM_B_TRAINING_CONTRACT_SHA256"
  --final-checkpoint32-adapter-sha256 "$FROZEN_FINAL_CHECKPOINT32_ADAPTER_SHA256"
  --confirmatory-trigger-sha256 "$FROZEN_CONFIRMATORY_TRIGGER_SHA256"
  --selection-manifest-path "$ARM_B_SELECTION_MANIFEST_PATH"
  --train-tasks-path "$TRAIN320_PATH"
  --validation64-audit-path "$VALIDATION64_AUDIT_PATH"
  --training-run-path "$TRAINING_RUN_PATH"
  --confirmatory-trigger-path "$CONFIRMATORY_TRIGGER_PATH"
)

if [[ "$MODE" == preflight ]]; then
  exec env PYTHONPATH="$PROJECT_ROOT" "$HOST_PYTHON" "$ARM_B_CONTROLLER" \
    --preflight "${COMMON_ARGS[@]}"
fi

RUN_DIR=${RUN_DIR:-/home/dengyan/tabular_rl_outputs/evaluations/qwen3_8b_atomic_v26_arm_b_vanilla_formal/formal_sft1_vs_arm_b_final32_20260812}
exec env PYTHONPATH="$PROJECT_ROOT" "$HOST_PYTHON" "$ARM_B_CONTROLLER" \
  --run --run-dir "$RUN_DIR" "${COMMON_ARGS[@]}"
