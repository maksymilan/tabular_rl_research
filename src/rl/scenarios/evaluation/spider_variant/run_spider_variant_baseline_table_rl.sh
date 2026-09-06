#!/usr/bin/env bash
set -euo pipefail

# Isolated Spider-variant evaluation launcher for table_rl.  This is deliberately
# separate from the BIRD matched-evaluation launchers and uses the frozen
# version26 runtime copied to the server.

ROOT="${ROOT:-/home/dengyan/tabular_rl_outputs/evaluations/spider_variants/qwen3_8b_4k_sft_20260904}"
PY="${PY:-/home/dengyan/miniconda3/envs/sft/bin/python}"
VLLM_PY="${VLLM_PY:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}"
RUNTIME="${RUNTIME:-/home/dengyan/tabular_rl_outputs/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}"
BASE_MODEL="${BASE_MODEL:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}"
ADAPTER="${ADAPTER:-$ROOT/model/checkpoint-6380}"
MODE="${1:-full}"

case "$MODE" in
  smoke) N=2; PREFIX=smoke2; SUMMARY_EVERY=1 ;;
  full) N=; PREFIX=full; SUMMARY_EVERY=25 ;;
  *) echo "usage: $0 [smoke|full]" >&2; exit 2 ;;
esac

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 EVAL_ENABLE_THINKING=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/log" "$ROOT/results"

start_server() {
  local gpu="$1" port="$2" alias="$3"
  local pidfile="$ROOT/log/vllm_${port}.pid" logfile="$ROOT/log/vllm_${port}.log"
  if test -r "$pidfile" && kill -0 "$(tr -d '[:space:]' < "$pidfile")" 2>/dev/null; then
    return 0
  fi
  rm -f "$pidfile"
  if ss -ltnH "sport = :$port" | grep -q .; then
    echo "port $port is already in use but has no managed pid" >&2
    exit 43
  fi
  setsid env CUDA_VISIBLE_DEVICES="$gpu" "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" --tokenizer "$BASE_MODEL" \
    --served-model-name "qwen3-spider-4k-backbone-$port" \
    --enable-lora --lora-modules "$alias=$ADAPTER" --max-lora-rank 64 \
    --host 127.0.0.1 --port "$port" --dtype bfloat16 --tensor-parallel-size 1 \
    --max-model-len 16384 --max-num-batched-tokens 16384 --max-num-seqs 4 \
    --gpu-memory-utilization 0.90 --generation-config vllm \
    >"$logfile" 2>&1 < /dev/null &
  echo "$!" > "$pidfile"
}

wait_server() {
  local port="$1"
  for _ in $(seq 1 180); do
    if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "vLLM on port $port did not become ready" >&2
  exit 44
}

run_eval() {
  local ds="$1" port="$2" alias="$3"
  local narg=()
  test -n "$N" && narg=(--n "$N")
  "$PY" -u "$RUNTIME/src/eval/rollout_passk.py" \
    --base-url "http://127.0.0.1:$port/v1" --model "$alias" \
    --examples-json "$ROOT/input/${ds}_scored.jsonl" --allow-eval-tasks \
    "${narg[@]}" --n-samples 1 --pass-k 1 --workers 4 --sample-workers 1 \
    --max-inflight-requests 4 --max-steps 30 --max-tokens 2048 \
    --temperature 0 --top-p 1 --sample-detail full --summary-every "$SUMMARY_EVERY" \
    --context-mode rolling-legal-history --history-turns 4 \
    --rolling-prompt-variant full --rolling-observation-style resident \
    --denotation-comparison bird-set \
    --result-dir "$ROOT/results/${PREFIX}_${ds}" --resume \
    >"$ROOT/log/eval_${PREFIX}_${ds}.log" 2>&1
}

start_server 0 8091 qwen3-spider-4k-sft
start_server 1 8092 qwen3-spider-4k-sft-gpu1
wait_server 8091
wait_server 8092

run_eval spider_test 8091 qwen3-spider-4k-sft & p1=$!
run_eval spider_syn 8091 qwen3-spider-4k-sft & p2=$!
run_eval spider_dk 8092 qwen3-spider-4k-sft-gpu1 & p3=$!
run_eval spider_realistic 8092 qwen3-spider-4k-sft-gpu1 & p4=$!

status=0
for p in "$p1" "$p2" "$p3" "$p4"; do
  wait "$p" || status=1
done
exit "$status"
