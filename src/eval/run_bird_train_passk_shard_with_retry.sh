#!/usr/bin/env bash
# Resume-safe pass@k rollout for one prefiltered BIRD-train task shard on one local GPU.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:?set PROJECT_DIR to the frozen evaluation runtime}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TASKS_JSON=${TASKS_JSON:?set TASKS_JSON to one prefiltered trainer task artifact}
RESULT_DIR=${RESULT_DIR:?set an isolated RESULT_DIR}
ADAPTER=${ADAPTER:?set ADAPTER to the frozen SFT adapter}
SERVED_MODEL=${SERVED_MODEL:?set a unique served model name}
GPU_ID=${GPU_ID:?set GPU_ID}
PORT=${PORT:?set a unique local port}
STATUS=${STATUS:?set STATUS}
RUN_LOG=${RUN_LOG:?set RUN_LOG}
VLLM_LOG=${VLLM_LOG:?set VLLM_LOG}
VLLM_PID_FILE=${VLLM_PID_FILE:?set VLLM_PID_FILE}
EVAL_LOG=${EVAL_LOG:?set EVAL_LOG}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python}
BASE_MODEL=${BASE_MODEL:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
MAX_INFLIGHT=${MAX_INFLIGHT:-24}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-24}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

set_status() {
  printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"
}

expected_rows() {
  "$PYTHON_BIN" - "$TASKS_JSON" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
rows = payload.get("examples", payload) if isinstance(payload, dict) else payload
if not isinstance(rows, list) or not rows:
    raise SystemExit("prefiltered task shard must contain a non-empty examples list")
print(len(rows))
PY
}

completed_rows() {
  if [[ -f "$RESULT_DIR/all.jsonl" ]]; then
    wc -l <"$RESULT_DIR/all.jsonl" | tr -d '[:space:]'
  else
    printf '0\n'
  fi
}

validate_complete() {
  "$PYTHON_BIN" - "$RESULT_DIR/all.jsonl" "$RESULT_DIR/manifest.json" "$EXPECTED" <<'PY'
import json
import sys

all_path, manifest_path, expected_text = sys.argv[1:]
expected = int(expected_text)
rows = [json.loads(line) for line in open(all_path, encoding="utf-8") if line.strip()]
assert len(rows) == expected, (len(rows), expected)
indices = [int(row["example_index"]) for row in rows]
assert len(set(indices)) == expected, len(set(indices))
assert all((row.get("dataset_split") or "train") == "train" for row in rows)
infra_types = {
    "api_error",
    "context_overflow",
    "generation_oom",
    "incomplete_api_response",
    "provider_carrier_error",
    "task_timeout",
    "transport_error",
}
infra = []
for row in rows:
    if row.get("failure_type") in infra_types:
        infra.append((row["example_index"], row.get("failure_type")))
    for sample in row.get("samples") or []:
        if sample.get("failure_type") in infra_types:
            infra.append((row["example_index"], sample.get("failure_type")))
assert not infra, infra[:20]
manifest = json.load(open(manifest_path, encoding="utf-8"))
assert manifest.get("denotation_comparison") == "bird-set"
assert manifest.get("n_samples") == 8
assert manifest.get("temperature") == 0.7
PY
}

mkdir -p "$OUTPUT_ROOT/logs"
exec >>"$RUN_LOG" 2>&1
cd "$PROJECT_DIR"

test -f "$TASKS_JSON"
test -f "$ADAPTER/adapter_model.safetensors"
EXPECTED=$(expected_rows)

if [[ "$(completed_rows)" -eq "$EXPECTED" ]]; then
  validate_complete
  set_status complete "already_complete rows=$EXPECTED gpu=$GPU_ID"
  exit 0
fi

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  set_status running \
    "attempt=$attempt rows=$(completed_rows)/$EXPECTED gpu=$GPU_ID k=8 concurrency=$MAX_INFLIGHT max_model_len=$MAX_MODEL_LEN"
  set +e
  PROJECT_DIR="$PROJECT_DIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  BASE_MODEL="$BASE_MODEL" \
  ADAPTER="$ADAPTER" \
  SERVED_MODEL="$SERVED_MODEL" \
  GPU_ID="$GPU_ID" \
  PORT="$PORT" \
  RESULT_DIR="$RESULT_DIR" \
  EXAMPLES_JSON="$TASKS_JSON" \
  N="$EXPECTED" \
  N_SAMPLES=8 \
  PASS_K=1,2,4,8 \
  WORKERS=3 \
  SAMPLE_WORKERS=8 \
  MAX_INFLIGHT="$MAX_INFLIGHT" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  GPU_MEMORY_UTILIZATION=0.90 \
  MAX_STEPS=30 \
  MAX_TOKENS=1024 \
  TEMPERATURE=0.7 \
  TOP_P=0.95 \
  HISTORY_TURNS=4 \
  EVAL_ENABLE_THINKING=0 \
  SERVER_CONFIG_ID="vllm-generation-config-${SERVED_MODEL}" \
  ALLOW_OPERATIONAL_CONCURRENCY_RESUME=1 \
  TOOL_EXECUTION_TIMEOUT_SECONDS=20 \
  ALLOW_OPERATIONAL_TOOL_TIMEOUT_RESUME=1 \
  VLLM_LOG="$VLLM_LOG" \
  VLLM_PID_FILE="$VLLM_PID_FILE" \
  EVAL_LOG="$EVAL_LOG" \
    bash src/eval/run_bird_lora_tool_passk_local_gpu.sh
  exit_code=$?
  set -e
  if [[ "$exit_code" -eq 0 ]]; then
    validate_complete
    set_status complete "rows=$EXPECTED gpu=$GPU_ID result=$RESULT_DIR"
    exit 0
  fi
  set_status retrying \
    "attempt=$attempt exit=$exit_code rows=$(completed_rows)/$EXPECTED gpu=$GPU_ID"
  sleep 60
done

set_status failed "exhausted_retries rows=$(completed_rows)/$EXPECTED gpu=$GPU_ID"
exit 1
