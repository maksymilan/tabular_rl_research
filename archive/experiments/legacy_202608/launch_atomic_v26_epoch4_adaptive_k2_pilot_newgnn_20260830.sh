#!/usr/bin/env bash
set -euo pipefail

# Local-Qwen3 result-only rollout screen.  This is intentionally separate from
# the old 4K pass@8 screen and never calls the external DeepSeek endpoint.
cd /home/dengyan/tabular_rl_checkpoint_e4fd09e
ROOT="/home/dengyan/tabular_rl_outputs/evaluations/qwen3_atomic_v26_epoch4_adaptive_pool3000_k2_pilot_20260830_r3"
POOL="/home/dengyan/tabular_rl_outputs/data/atomic_v26_epoch4_adaptive_rollout_pool_20260830"
TASKS="$POOL/rollout_tasks.jsonl"
SELECTION="$POOL/selection_manifest.json"
RUNTIME="/home/dengyan/tabular_rl_outputs/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de-eval-clean-20260829"
MODEL="/home/dengyan/models/Qwen3-8B-TrustSQL-baseline"
ADAPTER="/home/dengyan/tabular_rl_outputs/checkpoints/qwen3_atomic_v26_cumulative_augmented_fresh4ep_4gpu_newgnn_20260827/checkpoint-6380"
VENV="/home/dengyan/miniconda3/envs/vllm-qwen35"
PORT=8061
GPU=1

[[ ! -e "$ROOT" ]] || { echo "refusing existing result root: $ROOT" >&2; exit 2; }
[[ -s "$TASKS" && -s "$SELECTION" ]] || { echo "missing frozen adaptive pool" >&2; exit 3; }
[[ -s "$ADAPTER/adapter_model.safetensors" && -s "$ADAPTER/adapter_config.json" ]] || { echo "missing Epoch4 adapter" >&2; exit 4; }
[[ -f "$RUNTIME/src/eval/rollout_passk.py" ]] || { echo "missing frozen version26 runtime" >&2; exit 5; }
used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
apps=$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]' || true)
[[ -z "$apps" && "$used" -le 2048 ]] || { echo "GPU$GPU not free: memory=$used apps=$apps" >&2; exit 6; }
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "port $PORT is already in use" >&2
  exit 7
fi

mkdir -p "$ROOT"
python3 - "$ROOT" "$POOL" "$TASKS" "$SELECTION" "$ADAPTER" <<'PY'
import hashlib, json, os, sys, time
root, pool, tasks, selection, adapter = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
rows = [json.loads(line) for line in open(tasks, encoding="utf-8") if line.strip()]
sel = json.load(open(selection, encoding="utf-8"))
assert len(rows) == 3000 and sel["selection"]["selected_count"] == 3000
assert sel["outputs"]["rollout_tasks"]["sha256"] == sha(tasks)
metadata = {
    "schema_version": "atomic-v26-epoch4-adaptive-k2-pilot-launch-v1",
    "status": "running_local_qwen3_only",
    "repository_commit": "6f7b38e694b7fa0a50a827bfd88ee1db96a0e230",
    "launcher_revision": "adaptive-k2-pilot-r2",
    "runtime_commit": "4cd47c957fc6ae791e76a10594c8cd22f4d3b6de",
    "protocol_version": "version26",
    "tool_scheme": "atomic",
    "carrier": "think-json-v1",
    "model_checkpoint": "Epoch4-checkpoint-6380",
    "adapter_path": adapter,
    "selection_manifest": {"path": os.path.abspath(selection), "sha256": sha(selection)},
    "tasks": {"path": os.path.abspath(tasks), "sha256": sha(tasks), "records": len(rows)},
    "rollout": {
        "n_samples": 2, "pass_k": [1, 2], "temperature": 0.7, "top_p": 0.95,
        "workers": 24, "sample_workers": 2, "max_inflight_requests": 24,
        "max_steps": 30, "max_tokens": 2048, "history_turns": 4,
        "context_mode": "rolling-legal-history", "denotation_comparison": "bird-set",
        "enable_thinking": True, "external_api": False,
    },
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(os.path.join(root, "RUN_METADATA.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.write("\n")
PY

SERVER_PID=""
cleanup() {
  set +e
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 \
  "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --tokenizer "$MODEL" \
  --served-model-name qwen3-v26-epoch4-adaptive-k2-backbone \
  --enable-lora --lora-modules qwen3-v26-epoch4-adaptive-k2="$ADAPTER" \
  --max-lora-rank 16 --host 127.0.0.1 --port "$PORT" --dtype bfloat16 \
  --trust-remote-code --enable-prefix-caching --max-model-len 32768 \
  --max-num-batched-tokens 8192 --max-num-seqs 24 --gpu-memory-utilization 0.92 \
  --generation-config vllm >"$ROOT/vllm.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$ROOT/vllm.pid"

ready=0
for _ in $(seq 1 180); do
  if curl --noproxy '*' -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q 'qwen3-v26-epoch4-adaptive-k2-backbone'; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM exited before readiness" >&2
    tail -80 "$ROOT/vllm.log" >&2
    exit 20
  fi
  sleep 5
done
[[ "$ready" == 1 ]] || { echo "vLLM readiness timeout" >&2; tail -80 "$ROOT/vllm.log" >&2; exit 21; }

env -u PYTHONPATH -u EVAL_SYSTEM_PROMPT_VARIANT PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 EVAL_ENABLE_THINKING=1 \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  "$VENV/bin/python" -u "$RUNTIME/src/eval/rollout_passk.py" \
  --base-url "http://127.0.0.1:$PORT/v1" --model qwen3-v26-epoch4-adaptive-k2 \
  --examples-json "$TASKS" --n 3000 --n-samples 2 --pass-k 1,2 \
  --workers 24 --sample-workers 2 --max-inflight-requests 24 \
  --max-steps 30 --max-tokens 2048 --temperature 0.7 --top-p 0.95 \
  --sample-detail full --summary-every 10 --api-retries 3 \
  --context-mode rolling-legal-history --history-turns 4 \
  --rolling-prompt-variant full --rolling-observation-style resident \
  --denotation-comparison bird-set --result-dir "$ROOT/result" \
  >"$ROOT/rollout.log" 2>&1

rm -f "$ROOT/vllm.pid"
SERVER_PID=""
echo "rollout finished: $ROOT"
