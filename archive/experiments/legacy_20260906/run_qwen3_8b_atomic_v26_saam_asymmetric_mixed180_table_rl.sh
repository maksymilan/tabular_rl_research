#!/usr/bin/env bash
# Matched-budget SAAM expansion: frozen SFT1 -> exact historical mixed180,
# K=8, two passes, 12 optimizer updates, with bounded full/reason/tool gradient geometry.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_rl_optimized_20260831}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_saam_asymmetric_mixed180_optimized_20260831}
TRAIN_OUT=${TRAIN_OUT:-$RUN_ROOT/train180_two_pass_seed20260812}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
TASKS=${TASKS:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_boundary_screen_s1_earlystop_568_20260813/cohort/train180.jsonl}
COHORT_MANIFEST=${COHORT_MANIFEST:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_boundary_screen_s1_earlystop_568_20260813/cohort/earlystop_mixed180_manifest.json}
CONFIG=${CONFIG:-$RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_asymmetric_mixed180.yaml}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER_PATH=${ADAPTER_PATH:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
TRAIN_GPU=${TRAIN_GPU:-0}
VLLM_GPU=${VLLM_GPU:-1}
VLLM_PORT=${VLLM_PORT:-8089}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51289}
TRANSITION_MICRO_BATCH_SIZE=${TRANSITION_MICRO_BATCH_SIZE:-1}

EXPECTED_TASKS_SHA256=a015c6513b850576ff95388238d45ad5cc0130da53823bf5d43869668592fb5d
EXPECTED_COHORT_SHA256=e940c996d854a1756ec9787d375517125d9b7b35872b6ce9251232cd7526eaad
EXPECTED_CONFIG_SHA256=d460d7513ea1ad3cc769d2e794c5e1639d1784a85247cf1b4bb7450caffab479
EXPECTED_RUNNER_SHA256=014aaea719b5d4579ffdfe3885a228e7ae800c265efa9b5959c7259f07bf0273
EXPECTED_TRAINER_SHA256=07430cff871296fd1d17fd541a07fc58a314bdb8bcffba7889671290aa783736
EXPECTED_SAAM_SHA256=eab36eb3cb17b182bcf1668ae47531a53c799a098d9ae5afcfa8903944b7cb71
EXPECTED_GRADIENT_SHA256=b36b01ed2f2defa586a20e54f5f4ca4451093cbcb999abca3536c7f765986807
EXPECTED_SHELL_SHA256=b2fa3a0b6277fcfb30d30f0f14b08a2a288c6f07435efbb6a1c4135864a97657
EXPECTED_ADAPTER_SHA256=3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
require_sha() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "$path" && ! -L "$path" ]] || {
    printf 'missing/non-regular %s: %s\n' "$label" "$path" >&2
    exit 3
  }
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] || {
    printf '%s SHA mismatch expected=%s actual=%s\n' "$label" "$expected" "$actual" >&2
    exit 3
  }
}
gpu_idle() {
  local gpu=$1 pids used
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  [[ -z "$pids" && "$used" =~ ^[0-9]+$ && "$used" -le 512 ]]
}
atomic_status() {
  local state=$1 detail=$2 temporary=$RUN_ROOT/status.next
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" "$detail" >"$temporary"
  mv "$temporary" "$RUN_ROOT/status"
}

[[ -x "$PYTHON" && -d "$RUNTIME" && -d "$MODEL_PATH" && -d "$ADAPTER_PATH" ]] || {
  printf 'missing Python/runtime/model/SFT1 adapter\n' >&2
  exit 3
}
[[ "$TRAIN_GPU" =~ ^[0-9]+$ && "$VLLM_GPU" =~ ^[0-9]+$ && "$TRAIN_GPU" != "$VLLM_GPU" ]] || {
  printf 'trainer and vLLM GPUs must be distinct non-negative integers\n' >&2
  exit 2
}

mkdir -p "$RUN_ROOT/logs"
exec 9>"$RUN_ROOT/launcher.lock"
flock -n 9 || { printf 'SAAM mixed180 launcher already running\n' >&2; exit 75; }
exec >>"$RUN_ROOT/logs/launcher.log" 2>&1

require_sha "$TASKS" "$EXPECTED_TASKS_SHA256" train180
require_sha "$COHORT_MANIFEST" "$EXPECTED_COHORT_SHA256" cohort_manifest
require_sha "$CONFIG" "$EXPECTED_CONFIG_SHA256" experiment_config
require_sha "$RUNTIME/src/rl/frameworks/trl/run_transition_grpo.py" "$EXPECTED_RUNNER_SHA256" runner
require_sha "$RUNTIME/src/rl/frameworks/trl/transition_grpo.py" "$EXPECTED_TRAINER_SHA256" trainer
require_sha "$RUNTIME/src/rl/frameworks/trl/state_action_ambiguity.py" "$EXPECTED_SAAM_SHA256" saam
require_sha "$RUNTIME/src/rl/frameworks/trl/gradient_conflict.py" "$EXPECTED_GRADIENT_SHA256" gradient_conflict
require_sha "$RUNTIME/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" "$EXPECTED_SHELL_SHA256" training_shell
require_sha "$ADAPTER_PATH/adapter_model.safetensors" "$EXPECTED_ADAPTER_SHA256" initial_sft1_adapter
[[ -d "$PROTOCOL_RUNTIME" && ! -L "$PROTOCOL_RUNTIME" ]] || {
  printf 'missing/non-regular protocol runtime\n' >&2
  exit 3
}
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  [[ -d "$TRAIN_OUT" && -f "$RESUME_FROM_CHECKPOINT/trainer_state.json" ]] || {
    printf 'resume output/checkpoint is incomplete: output=%s checkpoint=%s\n' \
      "$TRAIN_OUT" "$RESUME_FROM_CHECKPOINT" >&2
    exit 3
  }
else
  [[ ! -e "$TRAIN_OUT" && ! -L "$TRAIN_OUT" ]] || {
    printf 'refusing to overwrite training output: %s\n' "$TRAIN_OUT" >&2
    exit 3
  }
fi

env PYTHONPATH= "$PYTHON" - "$TASKS" "$COHORT_MANIFEST" "$CONFIG" <<'PY'
import hashlib, json, sys
from pathlib import Path
import yaml

tasks_path, manifest_path, config_path = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 180:
    raise SystemExit(f"expected 180 training tasks, got {len(rows)}")
ids = [row.get("example_id") for row in rows]
if len(set(ids)) != 180 or any(not isinstance(value, str) for value in ids):
    raise SystemExit("train180 example identities are missing or duplicated")
if any(row.get("split") != "train" for row in rows):
    raise SystemExit("train180 contains a non-training record")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("status") != "frozen_operator_requested_earlystop_mixed180":
    raise SystemExit("unexpected train180 cohort status")
output = (manifest.get("outputs") or {}).get("train180") or {}
if output.get("records") != 180:
    raise SystemExit("cohort manifest record count mismatch")
digest = hashlib.sha256(tasks_path.read_bytes()).hexdigest()
if output.get("sha256") != digest:
    raise SystemExit("cohort manifest does not bind train180 bytes")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
expected = {
    "experiment_name": "qwen3_8b_atomic_v26_saam_asymmetric_mixed180",
    "expected_records": 180,
    "credit_assignment": "saam-asymmetric-error",
    "error_penalty": 1.0,
    "record_gradient_conflicts": False,
    "gradient_conflict_save_vectors": False,
    "result_reward_profile": "binary",
    "policy_reduction": "trajectory_token_mean",
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"config {key}: expected {value!r}, got {config.get(key)!r}")
if (config.get("optimizer") or {}).get("steps") != 12:
    raise SystemExit("config optimizer.steps must be 12")
rollout = config.get("rollout") or {}
if rollout.get("prompts_per_update") != 30 or rollout.get("group_size") != 8:
    raise SystemExit("config must use prompts_per_update=30 and group_size=8")
print("train180/config identity audit: ok")
PY

gpu_idle "$TRAIN_GPU" || { printf 'trainer GPU%s is busy\n' "$TRAIN_GPU" >&2; exit 75; }
gpu_idle "$VLLM_GPU" || { printf 'vLLM GPU%s is busy\n' "$VLLM_GPU" >&2; exit 75; }
if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
  printf 'refusing occupied vLLM port %s\n' "$VLLM_PORT" >&2
  exit 75
fi

vllm_pgid=""
trainer_pgid=""
stop_group() {
  local pgid=${1:-}
  [[ -n "$pgid" ]] || return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 -- "-$pgid" 2>/dev/null || break
      sleep 1
    done
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
for _ in $(seq 1 300); do
  kill -0 "$vllm_pgid" 2>/dev/null || break
  if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 2
done
[[ "$ready" -eq 1 ]] || { printf 'vLLM readiness failed\n' >&2; exit 1; }

atomic_status training "mixed180 records=180 k=8 prompts=30 updates=12 passes=2 trajectories=2880 gradient_diagnostics=off transition_micro_batch=$TRANSITION_MICRO_BATCH_SIZE"
resume_args=()
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  resume_args+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi
setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PROJECT_DIR="$RUNTIME" PYTHON="$PYTHON" \
  MODEL_PATH="$MODEL_PATH" ADAPTER_PATH="$ADAPTER_PATH" EXAMPLES_JSON="$TASKS" \
  OUTPUT_DIR="$TRAIN_OUT" EXPERIMENT_CONFIG="$CONFIG" VLLM_PORT="$VLLM_PORT" \
  VLLM_GROUP_PORT="$VLLM_GROUP_PORT" PYTHONPATH= \
  bash "$RUNTIME/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" \
    --optimizer-steps 12 --ppo-iterations 1 --prompts-per-update 30 --group-size 8 \
    --credit-assignment saam-asymmetric-error --error-penalty 1.0 \
    --kl-beta 0 --no-record-gradient-conflicts \
    --protocol-runtime-root "$PROTOCOL_RUNTIME" --transition-micro-batch-size "$TRANSITION_MICRO_BATCH_SIZE" \
    --save-steps 1 --save-total-limit 12 --seed 20260812 \
    "${resume_args[@]}" \
  >>"$RUN_ROOT/logs/train.log" 2>&1 &
trainer_pgid=$!
wait "$trainer_pgid"
trainer_pgid=""
stop_group "$vllm_pgid"
vllm_pgid=""

env PYTHONPATH= "$PYTHON" - "$TRAIN_OUT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
required = [
    root / "run_manifest.json",
    root / "implementation_lock.json",
    root / "training_precision.json",
    root / "rollouts.jsonl",
    root / "checkpoint-12/trainer_state.json",
    root / "checkpoint-12/adapter_model.safetensors",
    root / "final/adapter_model.safetensors",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"completed training artifacts missing: {missing}")
manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
expected = {
    "records": 180,
    "expected_records": 180,
    "optimizer_steps": 12,
    "group_size": 8,
    "prompts_per_update": 30,
    "credit_assignment": "saam-asymmetric-error",
    "error_penalty": 1.0,
    "seed": 20260812,
}
for field, value in expected.items():
    if manifest.get(field) != value:
        raise SystemExit(f"training manifest {field}: expected {value!r}, got {manifest.get(field)!r}")
state = json.loads((root / "checkpoint-12/trainer_state.json").read_text(encoding="utf-8"))
if int(state.get("global_step", -1)) != 12:
    raise SystemExit("checkpoint-12 does not bind global_step=12")
if (manifest.get("gradient_conflict_logging") or {}).get("enabled") is not False:
    raise SystemExit("gradient conflict logging must be disabled")
rollout_count = sum(1 for line in (root / "rollouts.jsonl").open(encoding="utf-8") if line.strip())
if rollout_count != 2880:
    raise SystemExit(f"expected 2880 rollout rows, got {rollout_count}")
print("completed SAAM mixed180 training audit: ok")
PY

atomic_status complete "final=$TRAIN_OUT/final checkpoint=12"
trap - EXIT INT TERM
