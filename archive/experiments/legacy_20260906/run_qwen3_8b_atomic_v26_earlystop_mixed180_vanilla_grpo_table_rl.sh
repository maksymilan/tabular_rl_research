#!/usr/bin/env bash
# Operator-requested early-stop vanilla GRPO arm. This is intentionally named
# separately from the preregistered 600-task/boundary300 confirmatory arm.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_earlystop_mixed180_grpo_20260813}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_earlystop_mixed180_vanilla_grpo_20260813}
TRAIN_OUT=$RUN_ROOT/train180_two_pass_seed20260812
TASKS=${TASKS:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_boundary_screen_s1_earlystop_568_20260813/cohort/train180.jsonl}
COHORT_MANIFEST=${COHORT_MANIFEST:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_boundary_screen_s1_earlystop_568_20260813/cohort/earlystop_mixed180_manifest.json}
CONFIG=$RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_vanilla_grpo_earlystop_mixed180.yaml
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER_PATH=${ADAPTER_PATH:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
TRAIN_GPU=${TRAIN_GPU:-0}
VLLM_GPU=${VLLM_GPU:-1}
VLLM_PORT=${VLLM_PORT:-8079}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51279}
ADMISSION_ROOT=${ADMISSION_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_vanilla_confirmatory_20260813}
ADMISSION=$ADMISSION_ROOT/admission.json
ADMISSION_HELPER=$RUNTIME/src/rl/diagnostics/manage_vanilla_grpo_confirmatory_admission.py

EXPECTED_TASKS_SHA256=a015c6513b850576ff95388238d45ad5cc0130da53823bf5d43869668592fb5d
EXPECTED_MANIFEST_SHA256=e940c996d854a1756ec9787d375517125d9b7b35872b6ce9251232cd7526eaad
EXPECTED_CONFIG_SHA256=58a9096030d242a36069ea026f5f27d3f12823b5912f25785756f217a100c6f6
EXPECTED_RUNNER_SHA256=2990f15e303b110845d222e76c9e5ae92edc8111b9942cd8c89b655688b28caa
EXPECTED_ADMISSION_HELPER_SHA256=b422e2248356b71fe91aff07844aa75f42d151a9349da0f724a332c9a8ea576e

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
require_sha() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "$path" && ! -L "$path" ]] || { printf 'missing %s: %s\n' "$label" "$path" >&2; exit 3; }
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] || {
    printf '%s SHA mismatch expected=%s actual=%s\n' "$label" "$expected" "$actual" >&2
    exit 3
  }
}
atomic_status() {
  local state=$1 detail=$2 temporary=$RUN_ROOT/status.next
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" "$detail" >"$temporary"
  mv "$temporary" "$RUN_ROOT/status"
}
gpu_idle() {
  local gpu=$1
  [[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]
}

mkdir -p "$RUN_ROOT/logs"
exec 9>"$RUN_ROOT/launcher.lock"
flock -n 9 || { printf 'launcher already running\n' >&2; exit 75; }
exec >>"$RUN_ROOT/logs/launcher.log" 2>&1

require_sha "$TASKS" "$EXPECTED_TASKS_SHA256" train180
require_sha "$COHORT_MANIFEST" "$EXPECTED_MANIFEST_SHA256" cohort_manifest
require_sha "$CONFIG" "$EXPECTED_CONFIG_SHA256" experiment_config
require_sha "$RUNTIME/src/rl/frameworks/trl/run_transition_grpo.py" "$EXPECTED_RUNNER_SHA256" runner
require_sha "$ADMISSION_HELPER" "$EXPECTED_ADMISSION_HELPER_SHA256" admission_helper
[[ -d "$PROTOCOL_RUNTIME" && ! -L "$PROTOCOL_RUNTIME" ]] || { printf 'missing protocol runtime\n' >&2; exit 3; }
[[ ! -e "$TRAIN_OUT" ]] || { printf 'fresh train output already exists: %s\n' "$TRAIN_OUT" >&2; exit 3; }
[[ "$TRAIN_GPU" != "$VLLM_GPU" ]] || { printf 'trainer and vLLM GPUs must differ\n' >&2; exit 3; }
gpu_idle "$TRAIN_GPU" || { printf 'trainer GPU%s is busy\n' "$TRAIN_GPU" >&2; exit 75; }
gpu_idle "$VLLM_GPU" || { printf 'vLLM GPU%s is busy\n' "$VLLM_GPU" >&2; exit 75; }
if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
  printf 'refusing occupied vLLM port %s\n' "$VLLM_PORT" >&2
  exit 75
fi

mkdir -p "$ADMISSION_ROOT"
"$PYTHON" "$ADMISSION_HELPER" claim --admission "$ADMISSION" --arm arm_a >/dev/null

vllm_pgid=""
trainer_pgid=""
stop_group() {
  local pgid=${1:-}
  [[ -n "$pgid" ]] || return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 -- "-$pgid" 2>/dev/null || break; sleep 1; done
    kill -0 -- "-$pgid" 2>/dev/null && kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
}
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  stop_group "$trainer_pgid"
  stop_group "$vllm_pgid"
  if [[ "$code" -ne 0 ]]; then atomic_status failed "exit=$code"; fi
  exit "$code"
}
trap cleanup EXIT INT TERM

atomic_status starting_vllm "gpu=$VLLM_GPU port=$VLLM_PORT"
setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
  PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table \
  MODEL_PATH="$MODEL_PATH" VLLM_PORT="$VLLM_PORT" \
  VLLM_GPU_MEMORY_UTILIZATION=0.82 MAX_MODEL_LEN=16384 \
  bash "$RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" \
  >>"$RUN_ROOT/logs/vllm.log" 2>&1 &
vllm_pgid=$!
ready=0
for _ in $(seq 1 240); do
  kill -0 "$vllm_pgid" 2>/dev/null || break
  if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 2
done
[[ "$ready" -eq 1 ]] || { printf 'vLLM readiness failed\n' >&2; exit 1; }

atomic_status training 'earlystop568 mixed180 records=180 k=8 prompts=30 updates=12 passes=2 trajectories=2880'
setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PROJECT_DIR="$RUNTIME" PYTHON="$PYTHON" \
  MODEL_PATH="$MODEL_PATH" ADAPTER_PATH="$ADAPTER_PATH" EXAMPLES_JSON="$TASKS" \
  OUTPUT_DIR="$TRAIN_OUT" EXPERIMENT_CONFIG="$CONFIG" VLLM_PORT="$VLLM_PORT" \
  VLLM_GROUP_PORT="$VLLM_GROUP_PORT" PYTHONPATH= \
  bash "$RUNTIME/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" \
    --optimizer-steps 12 --ppo-iterations 1 --prompts-per-update 30 --group-size 8 \
    --kl-beta 0 --protocol-runtime-root "$PROTOCOL_RUNTIME" \
    --transition-micro-batch-size 1 --save-steps 2 --save-total-limit 1 --seed 20260812 \
  >>"$RUN_ROOT/logs/train.log" 2>&1 &
trainer_pgid=$!
wait "$trainer_pgid"
trainer_pgid=""
stop_group "$vllm_pgid"
vllm_pgid=""
[[ -f "$TRAIN_OUT/run_manifest.json" && -f "$TRAIN_OUT/implementation_lock.json" ]] || {
  printf 'training exited without immutable manifests\n' >&2
  exit 1
}
[[ -d "$TRAIN_OUT/checkpoint-12" && -d "$TRAIN_OUT/final" ]] || {
  printf 'training exited without final step12 artifacts\n' >&2
  exit 1
}
atomic_status complete "final=$TRAIN_OUT/final checkpoint=12"
trap - EXIT INT TERM
