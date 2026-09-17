#!/usr/bin/env bash
# Resume only checkpoint-3 of the isolated NewGNN 8B recovery. SDPA changes
# the attention kernel, not the reward, cohort, rollout budget, or objective.
set -Eeuo pipefail

PROJECT_DIR=${PROJECT_DIR:?set PROJECT_DIR to the frozen isolated source copy}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to a fresh recovery output root}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:?set RESUME_CHECKPOINT to checkpoint-3}
TRAIN_GPU=${1:?usage: resume_8b_sdpa_pair.sh TRAIN_GPU VLLM_GPU [VLLM_PORT] [VLLM_GROUP_PORT]}
VLLM_GPU=${2:?usage: resume_8b_sdpa_pair.sh TRAIN_GPU VLLM_GPU [VLLM_PORT] [VLLM_GROUP_PORT]}
VLLM_PORT=${3:-18314}
VLLM_GROUP_PORT=${4:-51414}

PYTHON_ENV=${PYTHON_ENV:-/home/dengyan/miniconda3/envs/trl-table}
PYTHON="$PYTHON_ENV/bin/python"
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER_PATH=${ADAPTER_PATH:-/home/dengyan/tabular_rl_outputs/checkpoints/qwen3_atomic_v26_cumulative_augmented_fresh4ep_4gpu_newgnn_20260827/checkpoint-6380}
EXAMPLES_JSON=${EXAMPLES_JSON:-$PROJECT_DIR/data/rl_inputs/qwen3_v26_checkpoint6380_k8_correct2to6_balanced60_20260911.jsonl}
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-$PROJECT_DIR/src/rl/configs/experiments/qwen3_8b_atomic_v26_signed_vanilla_grpo_c2to6_balanced60_newgnn.yaml}
OUTPUT_DIR="$RUN_ROOT/train"

export PROJECT_DIR RUN_ROOT RESUME_CHECKPOINT PYTHON PYTHON_ENV MODEL_PATH
export ADAPTER_PATH EXAMPLES_JSON EXPERIMENT_CONFIG OUTPUT_DIR
export VLLM_PORT VLLM_GROUP_PORT
export MAX_MODEL_LEN=16384
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.88}
export VLLM_ENFORCE_EAGER=1
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TRITON_LIBCUDA_PATH=${TRITON_LIBCUDA_PATH:-$PYTHON_ENV/var/triton-libcuda}
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

source "$PROJECT_DIR/src/rl/frameworks/launcher/launch_common.sh"
assert_distinct_allowlisted_gpus "$TRAIN_GPU" "$VLLM_GPU" 7

mkdir -p "$RUN_ROOT/logs" "$OUTPUT_DIR"
exec 9>"$RUN_ROOT/pipeline.lock"
flock -n 9 || exit 2
exec 8>"/tmp/atomic-v26-gpu-$TRAIN_GPU.lock"
flock -n 8 || exit 2
exec 7>"/tmp/atomic-v26-gpu-$VLLM_GPU.lock"
flock -n 7 || exit 2

if [[ -e "$OUTPUT_DIR/rollouts.jsonl" ]]; then
  set_status "$RUN_ROOT" already_materialized "fresh SDPA recovery root required"
  exit 2
fi
"$PYTHON" - "$RESUME_CHECKPOINT" <<'PY'
import json
import pathlib
import sys
checkpoint = pathlib.Path(sys.argv[1])
state = json.loads((checkpoint / "trainer_state.json").read_text())
if int(state.get("global_step", -1)) != 3:
    raise SystemExit(f"expected checkpoint global_step=3, got {state.get('global_step')}")
if not (checkpoint / "adapter_model.safetensors").is_file():
    raise SystemExit("checkpoint-3 is missing adapter_model.safetensors")
PY

set_status "$RUN_ROOT" preflight "checkpoint-3 recovery; trainer SDPA; rollout/context unchanged"
COMMON_ARGS=(
  --seed 20260911
  --expected-records 60
  --expected-examples-sha256 1a6cb257081da58ce66e09ab4e0783f296e3fdc4aa5674da178ae4c86756b4ca
  --save-steps 1
  --save-total-limit 10
  --transition-micro-batch-size 1
  --transition-micro-batch-tokens 4096
  --replicated-base-storage 4bit
  --attn-implementation sdpa
  --max-new-tokens 2048
  --max-context-tokens 16384
  --resume-from-checkpoint "$RESUME_CHECKPOINT"
)
CUDA_VISIBLE_DEVICES="" bash "$PROJECT_DIR/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" \
  --preflight-only "${COMMON_ARGS[@]}" >"$RUN_ROOT/logs/preflight.json"

gpu_idle "$TRAIN_GPU" && gpu_idle "$VLLM_GPU" || {
  set_status "$RUN_ROOT" waiting_resources "two idle GPUs required"
  exit 75
}

"$PYTHON" - "$VLLM_PORT" "$VLLM_GROUP_PORT" <<'PY'
import socket
import sys
for raw_port in sys.argv[1:]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", int(raw_port)))
PY

trainer_pgid=
vllm_pgid=
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  stop_group "${trainer_pgid:-}"
  stop_group "${vllm_pgid:-}"
  if [[ "$code" -eq 0 ]]; then
    set_status "$RUN_ROOT" trained_pending_audit "checkpoint-4 candidate written; owned resources released"
  else
    set_status "$RUN_ROOT" failed "SDPA recovery exit=$code; owned resources released"
  fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv >"$RUN_ROOT/logs/gpus_before.csv"
set_status "$RUN_ROOT" starting_server "8B recovery vLLM gpu=$VLLM_GPU port=$VLLM_PORT"
setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
  bash "$PROJECT_DIR/src/rl/frameworks/trl/start_vllm_server.sh" \
  >"$RUN_ROOT/logs/vllm.log" 2>&1 &
vllm_pgid=$!
printf '%s\n' "$vllm_pgid" >"$RUN_ROOT/vllm.pid"
wait_for_health "$vllm_pgid" "http://127.0.0.1:$VLLM_PORT/health/"
"$PYTHON" - "$VLLM_PORT" <<'PY'
import json
import sys
from rl.frameworks.trl.serving_contract import probe_trl_server
print(json.dumps(probe_trl_server("127.0.0.1", int(sys.argv[1]))))
PY

gpu_idle "$TRAIN_GPU" || exit 75
set_status "$RUN_ROOT" starting_trainer "resume checkpoint-3; SDPA; microbatch=1/4096"
setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  bash "$PROJECT_DIR/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" \
  "${COMMON_ARGS[@]}" >"$RUN_ROOT/logs/train.log" 2>&1 &
trainer_pgid=$!
printf '%s\n' "$trainer_pgid" >"$RUN_ROOT/trainer.pid"
wait "$trainer_pgid"
