#!/usr/bin/env bash
# Fail-closed preflight for the checkpoint matched-eval path (runs ON the eval host).
#
# Checks the three environment traps that killed the 2026-09-17 eval chain:
#   1. eval controllers need PYTHONPATH=<project>/src, otherwise `import rl` fails instantly;
#   2. triton needs a linker-visible libcuda.so (TRITON_LIBCUDA_PATH), otherwise vLLM's CUDA
#      graph capture dies with a corrupted-shim checksum error or a gcc `-lcuda` link failure;
#   3. the eval launcher is fail-closed on busy GPUs and it requires a fresh output directory.
#
# Usage: preflight_eval_environment.sh <project_src> <gpu0> <gpu1> [port0] [port1]
set -Eeuo pipefail

PROJECT_SRC=${1:?usage: preflight_eval_environment.sh <project_src> <gpu0> <gpu1> [port0] [port1]}
GPU0=${2:?gpu0}
GPU1=${3:?gpu1}
PORT0=${4:-18406}
PORT1=${5:-18408}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
PYTHON_ENV=$(dirname "$(dirname "$PYTHON_BIN")")
TRITON_LIBCUDA_PATH=${TRITON_LIBCUDA_PATH:-"$PYTHON_ENV/var/triton-libcuda"}
export PYTHONPATH="$PROJECT_SRC${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_LIBCUDA_PATH

failures=0
ok() { printf 'PASS  %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; failures=$((failures + 1)); }

[[ -x "$PYTHON_BIN" ]] && ok "python: $PYTHON_BIN" || bad "python missing: $PYTHON_BIN"
[[ -e "$TRITON_LIBCUDA_PATH/libcuda.so" ]] && ok "triton shim: $TRITON_LIBCUDA_PATH/libcuda.so" \
  || bad "triton libcuda shim missing under $TRITON_LIBCUDA_PATH (vLLM CUDA graph capture will fail)"
[[ -f "$PROJECT_SRC/rl/evaluation/runners/formal_v26_rollout_passk.py" ]] \
  && ok "eval controllers present under $PROJECT_SRC" \
  || bad "eval controllers missing under $PROJECT_SRC (check PYTHONPATH target)"

# Import check runs with CUDA hidden: it only validates PYTHONPATH, not the driver.
if CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -c "
import importlib
importlib.import_module('rl.evaluation.runners.make_eval_shards')
importlib.import_module('rl.evaluation.runners.formal_v26_rollout_passk')
" >/dev/null 2>&1; then
  ok "'import rl.*' succeeds with PYTHONPATH=$PROJECT_SRC"
else
  bad "'import rl.*' failed with this PYTHONPATH (run the import manually to see the traceback)"
fi

gpu0_idle=0
for gpu in "$GPU0" "$GPU1"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  apps=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' \n')
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 512 )) && [[ -z "$apps" ]]; then
    ok "GPU $gpu idle (${used} MiB)"
    [[ "$gpu" == "$GPU0" ]] && gpu0_idle=1
  else
    bad "GPU $gpu not idle: ${used} MiB used, compute pids='${apps}'"
  fi
done

# The runtime probe needs a real CUDA context, so only run it on an idle card; a busy card is
# already reported as a failure above.
if (( gpu0_idle == 1 )); then
  if CUDA_VISIBLE_DEVICES="$GPU0" "$PYTHON_BIN" -c "
import torch.utils._triton as triton_utils
triton_utils.triton_backend()
" >/dev/null 2>&1; then
    ok "triton_backend() initialises on GPU $GPU0 (libcuda shim usable)"
  else
    bad "triton_backend() failed on GPU $GPU0 (check TRITON_LIBCUDA_PATH=$TRITON_LIBCUDA_PATH)"
  fi
else
  printf 'SKIP  triton_backend() probe (GPU %s busy)\n' "$GPU0"
fi

for port in "$PORT0" "$PORT1"; do
  if "$PYTHON_BIN" - "$port" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.settimeout(0.5)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) != 0 else 1)
PY
  then
    ok "port $port free"
  else
    bad "port $port already in use"
  fi
done

if (( failures == 0 )); then
  echo "preflight OK"
  exit 0
fi
echo "preflight FAILED (${failures} checks) - do not launch the eval"
exit 2
