#!/usr/bin/env bash
# Matched two-pass Gate60: vanilla trajectory GRPO versus SAAM-strict.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODE=${1:-queue}
if [[ $# -gt 1 || ( "$MODE" != queue && "$MODE" != trajectory && "$MODE" != saam ) ]]; then
  printf 'usage: %s [queue|trajectory|saam]\n' "$0" >&2
  exit 2
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_saam_gate60_20260828}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_saam_gate60_20260828}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER_PATH=${ADAPTER_PATH:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
TASKS=${TASKS:-$RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_saam_gate60_train_v1.jsonl}
COHORT_MANIFEST=${COHORT_MANIFEST:-$RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_saam_gate60_train_v1.manifest.json}
TRAJECTORY_CONFIG=${TRAJECTORY_CONFIG:-$RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_gate60_trajectory.yaml}
SAAM_CONFIG=${SAAM_CONFIG:-$RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_gate60_strict.yaml}
TRAJECTORY_OUT=${TRAJECTORY_OUT:-$RUN_ROOT/trajectory_two_pass_seed20260828}
SAAM_OUT=${SAAM_OUT:-$RUN_ROOT/saam_strict_two_pass_seed20260828}
TRAIN_GPU=${TRAIN_GPU:-0}
VLLM_GPU=${VLLM_GPU:-1}
VLLM_PORT=${VLLM_PORT:-8088}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51288}

EXPECTED_TASKS_SHA256=47e9d369bf6506d13b2432ac92bb971bf52912cb759c133846801f5411ad79d5
EXPECTED_COHORT_MANIFEST_SHA256=5bd96f416edc6fc4845db867cb6171d9bb3b7a28a74d11fedc9cf4ae88f72758
EXPECTED_TRAJECTORY_CONFIG_SHA256=45cc06691550b37a30500bcb55f6ce46d356a3c50a8889539e8218830e20d410
EXPECTED_SAAM_CONFIG_SHA256=4ee72a7e5ada1068ff3c6392667b40690384508ad6bc26f4abfe2c1614a6b3b4
EXPECTED_EXPERIMENT_CONFIG_PY_SHA256=ad6c5c1df2b1550c73c99a4da4fd45114596fedbe9a5b87308e16ab25d032d14
EXPECTED_RUNNER_SHA256=d7b6f0064cf2427846c19e4eaeab9b1b9c51a2d1b3b2ba5b716ab4e29a0f5240
EXPECTED_TRAINER_SHA256=cb387785164f245c066061ec15e081b39b6e98250868a38d848538af79edc594
EXPECTED_SAAM_PY_SHA256=f212ed053e3213ba599902059a7b26c2060f155d49f547e84c4c43dbfb24e040
EXPECTED_OSCILLATION_AUDITOR_SHA256=3e1e7a5d432fc8de4cf986002ed348f91ace2744897368b287ea2ed31b3672da
EXPECTED_SFT1_SHA256=3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
require_sha() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "$path" && ! -L "$path" ]] || { printf 'missing %s: %s\n' "$label" "$path" >&2; exit 3; }
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] || {
    printf '%s SHA mismatch expected=%s actual=%s path=%s\n' "$label" "$expected" "$actual" "$path" >&2
    exit 3
  }
}
atomic_status() {
  local state=$1 detail=$2 temporary=$RUN_ROOT/status.next
  printf '%s\t%s\t%s\n' "$(timestamp)" "$state" "$detail" >"$temporary"
  mv "$temporary" "$RUN_ROOT/status"
}
gpu_idle() {
  local gpu=$1
  [[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]
}

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
    if kill -0 -- "-$pgid" 2>/dev/null; then
      kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
  fi
}
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  stop_group "$trainer_pgid"
  trainer_pgid=""
  stop_group "$vllm_pgid"
  vllm_pgid=""
  if [[ "$code" -ne 0 ]]; then atomic_status failed "mode=$MODE exit=$code"; fi
  exit "$code"
}
trap cleanup EXIT INT TERM

validate_inputs() {
  [[ -x "$PYTHON" ]] || { printf 'missing Python: %s\n' "$PYTHON" >&2; exit 3; }
  [[ -d "$MODEL_PATH" && -d "$ADAPTER_PATH" && -d "$PROTOCOL_RUNTIME" ]] || {
    printf 'missing model, adapter, or protocol runtime\n' >&2
    exit 3
  }
  [[ "$TRAIN_GPU" != "$VLLM_GPU" ]] || { printf 'trainer and vLLM GPUs must differ\n' >&2; exit 3; }
  require_sha "$TASKS" "$EXPECTED_TASKS_SHA256" gate60_tasks
  require_sha "$COHORT_MANIFEST" "$EXPECTED_COHORT_MANIFEST_SHA256" gate60_manifest
  require_sha "$TRAJECTORY_CONFIG" "$EXPECTED_TRAJECTORY_CONFIG_SHA256" trajectory_config
  require_sha "$SAAM_CONFIG" "$EXPECTED_SAAM_CONFIG_SHA256" saam_config
  require_sha "$RUNTIME/src/rl/experiment_config.py" "$EXPECTED_EXPERIMENT_CONFIG_PY_SHA256" experiment_config_py
  require_sha "$RUNTIME/src/rl/frameworks/trl/run_transition_grpo.py" "$EXPECTED_RUNNER_SHA256" runner
  require_sha "$RUNTIME/src/rl/frameworks/trl/transition_grpo.py" "$EXPECTED_TRAINER_SHA256" trainer
  require_sha "$RUNTIME/src/rl/frameworks/trl/state_action_ambiguity.py" "$EXPECTED_SAAM_PY_SHA256" saam_implementation
  require_sha "$RUNTIME/src/rl/diagnostics/audit_saam_credit_oscillation.py" "$EXPECTED_OSCILLATION_AUDITOR_SHA256" oscillation_auditor
  require_sha "$ADAPTER_PATH/adapter_model.safetensors" "$EXPECTED_SFT1_SHA256" sft1_adapter

  env PYTHONPATH= "$PYTHON" - "$RUNTIME" "$TASKS" "$COHORT_MANIFEST" \
    "$TRAJECTORY_CONFIG" "$SAAM_CONFIG" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

runtime, tasks_path, manifest_path, trajectory_path, saam_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(runtime / "src" / "rl"))
from rl.configuration.experiment_config import RLExperimentConfig

rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
assert len(rows) == 60
ids = [str(row["example_id"]) for row in rows]
assert len(set(ids)) == 60 and all(row["split"] == "train" for row in rows)
assert all("/bird/train/" in row["db_path"] for row in rows)
manifest = json.loads(manifest_path.read_text())
assert manifest["schema_version"] == "qwen3-v26-saam-gate60-cohort-v1"
assert manifest["dataset"] == "BIRD-train" and manifest["dev1534_used"] is False
assert manifest["output"]["task_ids_in_order"] == ids
assert manifest["output"]["sha256"] == hashlib.sha256(tasks_path.read_bytes()).hexdigest()

a = RLExperimentConfig.load(trajectory_path)
b = RLExperimentConfig.load(saam_path)
pa, pb = dict(a.payload), dict(b.payload)
pa.pop("experiment_name")
pb.pop("experiment_name")
assert pa.pop("credit_assignment") == "trajectory"
assert pb.pop("credit_assignment") == "saam-strict"
assert pa == pb
for config, credit in ((a, "trajectory"), (b, "saam-strict")):
    defaults = config.argparse_defaults(runtime)
    assert defaults["expected_records"] == 60
    assert defaults["optimizer_steps"] == 4
    assert defaults["prompts_per_update"] == 30
    assert defaults["group_size"] == 8
    assert defaults["credit_assignment"] == credit
    assert defaults["result_reward_profile"] == "binary"
    assert defaults["policy_reduction"] == "trajectory_token_mean"
    assert defaults["kl_beta"] == 0.0
PY
}

verify_output() {
  local output=$1 expected_credit=$2
  [[ -f "$output/run_manifest.json" && -f "$output/implementation_lock.json" ]] || {
    printf 'training exited without immutable manifests: %s\n' "$output" >&2
    return 1
  }
  [[ -d "$output/checkpoint-4" && -d "$output/final" && -f "$output/rollouts.jsonl" ]] || {
    printf 'training exited without final step4 artifacts: %s\n' "$output" >&2
    return 1
  }
  "$PYTHON" - "$output/run_manifest.json" "$expected_credit" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1]))
assert manifest["credit_assignment"] == sys.argv[2]
assert manifest["records"] == 60
assert manifest["optimizer_steps"] == 4
assert manifest["prompts_per_update"] == 30
assert manifest["group_size"] == 8
assert manifest["seed"] == 20260828
assert manifest["initial_adapter_sha256"] == "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5"
PY
}

run_arm() {
  local arm=$1 credit=$2 config=$3 output=$4
  [[ ! -e "$output" ]] || { printf 'fresh output already exists: %s\n' "$output" >&2; return 3; }
  gpu_idle "$TRAIN_GPU" || { printf 'trainer GPU%s is busy\n' "$TRAIN_GPU" >&2; return 75; }
  gpu_idle "$VLLM_GPU" || { printf 'vLLM GPU%s is busy\n' "$VLLM_GPU" >&2; return 75; }
  if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
    printf 'refusing occupied vLLM port %s\n' "$VLLM_PORT" >&2
    return 75
  fi

  atomic_status starting_vllm "arm=$arm gpu=$VLLM_GPU port=$VLLM_PORT"
  setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
    PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table \
    MODEL_PATH="$MODEL_PATH" VLLM_PORT="$VLLM_PORT" \
    VLLM_GPU_MEMORY_UTILIZATION=0.82 MAX_MODEL_LEN=16384 \
    bash "$RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" \
    >>"$RUN_ROOT/logs/${arm}_vllm.log" 2>&1 &
  vllm_pgid=$!
  local ready=0
  for _ in $(seq 1 300); do
    kill -0 "$vllm_pgid" 2>/dev/null || break
    if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 2
  done
  [[ "$ready" -eq 1 ]] || { printf 'vLLM readiness failed for %s\n' "$arm" >&2; return 1; }

  atomic_status training "arm=$arm credit=$credit records=60 k=8 updates=4 passes=2"
  setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PROJECT_DIR="$RUNTIME" PYTHON="$PYTHON" \
    MODEL_PATH="$MODEL_PATH" ADAPTER_PATH="$ADAPTER_PATH" EXAMPLES_JSON="$TASKS" \
    OUTPUT_DIR="$output" EXPERIMENT_CONFIG="$config" VLLM_PORT="$VLLM_PORT" \
    VLLM_GROUP_PORT="$VLLM_GROUP_PORT" PYTHONPATH= \
    bash "$RUNTIME/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" \
      --optimizer-steps 4 --ppo-iterations 1 --prompts-per-update 30 --group-size 8 \
      --credit-assignment "$credit" --kl-beta 0 \
      --protocol-runtime-root "$PROTOCOL_RUNTIME" \
      --transition-micro-batch-size 1 --save-steps 1 --save-total-limit 4 \
      --seed 20260828 >>"$RUN_ROOT/logs/${arm}_train.log" 2>&1 &
  trainer_pgid=$!
  wait "$trainer_pgid"
  trainer_pgid=""
  stop_group "$vllm_pgid"
  vllm_pgid=""

  verify_output "$output" "$credit"
  "$PYTHON" "$RUNTIME/src/rl/diagnostics/audit_saam_credit_oscillation.py" \
    "$output/rollouts.jsonl" --credit-assignment "$credit" --group-size 8 \
    --output "$output/credit_oscillation_audit.json"
  atomic_status arm_complete "arm=$arm final=$output/final"
}

mkdir -p "$RUN_ROOT/logs"
exec 9>"$RUN_ROOT/launcher.lock"
flock -n 9 || { printf 'SAAM Gate60 launcher already running\n' >&2; exit 75; }
exec >>"$RUN_ROOT/launcher.log" 2>&1
validate_inputs

case "$MODE" in
  trajectory) run_arm trajectory trajectory "$TRAJECTORY_CONFIG" "$TRAJECTORY_OUT" ;;
  saam) run_arm saam saam-strict "$SAAM_CONFIG" "$SAAM_OUT" ;;
  queue)
    run_arm trajectory trajectory "$TRAJECTORY_CONFIG" "$TRAJECTORY_OUT"
    run_arm saam saam-strict "$SAAM_CONFIG" "$SAAM_OUT"
    ;;
esac

atomic_status complete "mode=$MODE trajectory=$TRAJECTORY_OUT/final saam=$SAAM_OUT/final"
trap - EXIT INT TERM
