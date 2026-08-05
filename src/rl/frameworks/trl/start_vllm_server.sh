#!/usr/bin/env bash
set -euo pipefail

PYTHON_ENV=${PYTHON_ENV:-/home/dengyan/miniconda3/envs/trl-table}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the local base model}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_SERVER_PORT=${VLLM_PORT:-8000}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.88}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}
# `VLLM_PORT` is also consumed by vLLM as its distributed master port.
# Preserve the user-facing value for the CLI, but do not leak that name.
unset VLLM_PORT

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_LAUNCH_MODE=${NCCL_LAUNCH_MODE:-GROUP}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

# Some GPU hosts expose only the versioned 64-bit driver library
# (`libcuda.so.1`). Triton's runtime compiler links with `-lcuda`, which
# requires an unversioned `libcuda.so`. Keep the compatibility link inside
# this isolated environment rather than modifying a system library directory.
if [[ -z "${TRITON_LIBCUDA_PATH:-}" ]] \
  && [[ -e /lib/x86_64-linux-gnu/libcuda.so.1 ]] \
  && [[ ! -e /lib/x86_64-linux-gnu/libcuda.so ]]; then
  triton_libcuda_link_dir=${TRITON_LIBCUDA_LINK_DIR:-"$PYTHON_ENV/var/triton-libcuda"}
  mkdir -p "$triton_libcuda_link_dir"
  ln -sfn /lib/x86_64-linux-gnu/libcuda.so.1 "$triton_libcuda_link_dir/libcuda.so"
  ln -sfn /lib/x86_64-linux-gnu/libcuda.so.1 "$triton_libcuda_link_dir/libcuda.so.1"
  export TRITON_LIBCUDA_PATH="$triton_libcuda_link_dir"
fi

server_args=(
  vllm-serve
  --model "$MODEL_PATH"
  --host "$VLLM_HOST"
  --port "$VLLM_SERVER_PORT"
  --tensor_parallel_size 1
  --gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --max_model_len "$MAX_MODEL_LEN"
  --dtype bfloat16
  --enable_prefix_caching true
  --trust_remote_code true
)
if [[ "$VLLM_ENFORCE_EAGER" == "1" ]]; then
  server_args+=(--enforce_eager)
fi

exec "$PYTHON_ENV/bin/trl" "${server_args[@]}"
