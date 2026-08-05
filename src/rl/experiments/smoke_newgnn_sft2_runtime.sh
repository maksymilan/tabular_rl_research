#!/usr/bin/env bash
# Bounded NewGNN CUDA, QLoRA+SFT2, and vLLM+SFT2 readiness smoke.
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
PYTHON_ENV=${PYTHON_ENV:-/home/dengyan/miniconda3/envs/trl-table}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
SFT2_ADAPTER_PATH=${SFT2_ADAPTER_PATH:-"$OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682"}
TRAIN_GPU_ID=${TRAIN_GPU_ID:-6}
VLLM_GPU_ID=${VLLM_GPU_ID:-7}
GPU_FREE_THRESHOLD_MIB=${GPU_FREE_THRESHOLD_MIB:-512}
PORT=${PORT:-18060}
LOG=${LOG:-"$OUTPUT_ROOT/logs/newgnn_sft2_runtime_smoke_20260731.log"}
VLLM_LOG=${VLLM_LOG:-"$OUTPUT_ROOT/logs/newgnn_sft2_runtime_smoke_20260731.vllm.log"}
PASSED_MARKER=${PASSED_MARKER:-"$OUTPUT_ROOT/logs/newgnn_sft2_runtime_smoke_20260731.passed"}

gpu_memory() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    sed -n "$(( $1 + 1 ))p" | tr -d '[:space:]'
}
require_free_gpu() {
  local gpu=$1
  local used
  used=$(gpu_memory "$gpu")
  if [[ -z "$used" || "$used" -gt "$GPU_FREE_THRESHOLD_MIB" ]]; then
    printf 'candidate GPU %s is occupied (%s MiB); smoke deferred\n' "$gpu" "${used:-unknown}" >&2
    exit 75
  fi
}

mkdir -p "$OUTPUT_ROOT/logs"
exec >>"$LOG" 2>&1
test -x "$PYTHON_ENV/bin/python"
test -f "$MODEL_PATH/model.safetensors.index.json"
test -f "$SFT2_ADAPTER_PATH/adapter_model.safetensors"
require_free_gpu "$TRAIN_GPU_ID"
require_free_gpu "$VLLM_GPU_ID"

printf 'starting CUDA+QLoRA SFT2 smoke train_gpu=%s\n' "$TRAIN_GPU_ID"
CUDA_VISIBLE_DEVICES="$TRAIN_GPU_ID" HF_HUB_OFFLINE=1 \
  "$PYTHON_ENV/bin/python" - "$MODEL_PATH" "$SFT2_ADAPTER_PATH" <<'PY'
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_path, adapter_path = sys.argv[1:]
assert torch.cuda.is_available()
x = torch.randn(128, 128, device="cuda")
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all()
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
base = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quant,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    local_files_only=True,
)
model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
inputs = tokenizer("Return one JSON object.", return_tensors="pt").to("cuda")
with torch.no_grad():
    logits = model(**inputs).logits
torch.cuda.synchronize()
assert logits.shape[0] == 1 and torch.isfinite(logits[:, -1, :]).all()
print("qlora_sft2_forward=passed")
PY

for _ in {1..60}; do
  [[ "$(gpu_memory "$TRAIN_GPU_ID")" -le "$GPU_FREE_THRESHOLD_MIB" ]] && break
  sleep 1
done
require_free_gpu "$TRAIN_GPU_ID"
require_free_gpu "$VLLM_GPU_ID"

vllm_pid=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$vllm_pid" ]] && kill -0 "$vllm_pid" 2>/dev/null; then
    kill -TERM -- "-$vllm_pid" 2>/dev/null || kill -TERM "$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

printf 'starting vLLM SFT2 smoke vllm_gpu=%s\n' "$VLLM_GPU_ID"
CUDA_VISIBLE_DEVICES="$VLLM_GPU_ID" HF_HUB_OFFLINE=1 \
  setsid "$PYTHON_ENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name qwen25-coder7b-base-smoke \
    --enable-lora \
    --lora-modules "sft2-smoke=$SFT2_ADAPTER_PATH" \
    --max-lora-rank 16 \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.80 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 2048 \
    --generation-config vllm \
    >"$VLLM_LOG" 2>&1 &
vllm_pid=$!

ready=0
for _ in {1..180}; do
  if ! kill -0 "$vllm_pid" 2>/dev/null; then
    printf 'vLLM smoke exited before readiness\n' >&2
    exit 1
  fi
  if curl --noproxy '*' -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" |
      grep -q 'sft2-smoke'; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" -eq 1 ]]
curl --noproxy '*' -fsS --max-time 120 \
  -H 'Content-Type: application/json' \
  -d '{"model":"sft2-smoke","messages":[{"role":"user","content":"Reply with one short word."}],"max_tokens":4,"temperature":0}' \
  "http://127.0.0.1:$PORT/v1/chat/completions" |
  "$PYTHON_ENV/bin/python" -c 'import json,sys; x=json.load(sys.stdin); assert x["choices"][0]["message"]["content"] is not None; print("vllm_sft2_request=passed")'

kill -TERM -- "-$vllm_pid" 2>/dev/null || kill -TERM "$vllm_pid" 2>/dev/null || true
wait "$vllm_pid" 2>/dev/null || true
vllm_pid=""
for _ in {1..90}; do
  [[ "$(gpu_memory "$VLLM_GPU_ID")" -le "$GPU_FREE_THRESHOLD_MIB" ]] && break
  sleep 1
done
require_free_gpu "$VLLM_GPU_ID"
printf '%s\tpassed\tqlora_sft2_forward+vllm_sft2_request\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >"$PASSED_MARKER"
printf 'NewGNN SFT2 runtime smoke passed\n'
