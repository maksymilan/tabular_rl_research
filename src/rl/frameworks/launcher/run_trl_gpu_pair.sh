#!/usr/bin/env bash
# Shared single-trainer + TRL serving lifecycle; no experiment-specific reward logic.
set -Eeuo pipefail
PROJECT_DIR=${PROJECT_DIR:?set PROJECT_DIR to the isolated implementation root}
RUN_ROOT=${RUN_ROOT:?set an isolated RUN_ROOT}
TRAIN_GPU=${TRAIN_GPU:?set TRAIN_GPU}
VLLM_GPU=${VLLM_GPU:?set VLLM_GPU}
PYTHON_ENV=${PYTHON_ENV:-/home/dengyan/miniconda3/envs/trl-table}
export PYTHON="$PYTHON_ENV/bin/python"
export PROJECT_DIR
export OUTPUT_DIR="$RUN_ROOT/train"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export VLLM_PORT=${VLLM_PORT:?set VLLM_PORT}
export VLLM_GROUP_PORT=${VLLM_GROUP_PORT:?set VLLM_GROUP_PORT}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
export PYTHON_ENV
source "$PROJECT_DIR/src/rl/frameworks/launcher/launch_common.sh"
assert_distinct_allowlisted_gpus "$TRAIN_GPU" "$VLLM_GPU" 7
mkdir -p "$RUN_ROOT/logs"
exec 9>"$RUN_ROOT/pipeline.lock"
flock -n 9 || exit 2
exec 8>"/tmp/atomic-v26-gpu-$TRAIN_GPU.lock"
flock -n 8 || exit 2
exec 7>"/tmp/atomic-v26-gpu-$VLLM_GPU.lock"
flock -n 7 || exit 2
trainer_pgid=
vllm_pgid=
set_status "$RUN_ROOT" preflight "CPU identity and cohort validation"
CUDA_VISIBLE_DEVICES="" bash "$PROJECT_DIR/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" \
  --preflight-only "$@" >"$RUN_ROOT/logs/preflight.json"
gpu_idle "$TRAIN_GPU" && gpu_idle "$VLLM_GPU" || {
  set_status "$RUN_ROOT" waiting_resources "two idle GPUs required"
  exit 75
}
install_process_cleanup_trap "$RUN_ROOT"
"$PYTHON" - "$VLLM_PORT" "$VLLM_GROUP_PORT" <<'PY'
import socket, sys
for port in sys.argv[1:]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", int(port)))
PY
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv >"$RUN_ROOT/logs/gpus_before.csv"
set_status "$RUN_ROOT" starting_server "TRL online weight-sync server"
setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
  bash "$PROJECT_DIR/src/rl/frameworks/trl/start_vllm_server.sh" \
  >"$RUN_ROOT/logs/vllm.log" 2>&1 &
vllm_pgid=$!
printf '%s\n' "$vllm_pgid" >"$RUN_ROOT/vllm.pid"
wait_for_health "$vllm_pgid" "http://127.0.0.1:$VLLM_PORT/health/"
"$PYTHON" - "$VLLM_PORT" <<'PY'
import json, sys
from rl.frameworks.trl.serving_contract import probe_trl_server
print(json.dumps(probe_trl_server("127.0.0.1", int(sys.argv[1]))))
PY
gpu_idle "$TRAIN_GPU" || exit 75
set_status "$RUN_ROOT" starting_trainer "first effective update not yet verified"
setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  bash "$PROJECT_DIR/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" "$@" \
  >"$RUN_ROOT/logs/train.log" 2>&1 &
trainer_pgid=$!
printf '%s\n' "$trainer_pgid" >"$RUN_ROOT/trainer.pid"
wait "$trainer_pgid"
trainer_pgid=
stop_group "$vllm_pgid"
vllm_pgid=
set_status "$RUN_ROOT" trained_pending_audit "trainer exit 0; checkpoint and effective-update audit required"
