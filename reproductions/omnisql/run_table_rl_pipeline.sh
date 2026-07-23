#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/dengyan/tabular_rl_outputs/reproductions/omnisql}"
MODEL_DIR="${MODEL_DIR:-/home/dengyan/models/text2sql_reproduction/OmniSQL-7B-af4eed67}"
HF="${HF:-/home/dengyan/miniconda3/envs/vllm-qwen35/bin/hf}"
RUNNER="${RUNNER:-$ROOT/source/run_bird_official.sh}"
DOWNLOAD_PID_FILE="${DOWNLOAD_PID_FILE:-$ROOT/logs/model_curl_download.pid}"
PIPELINE_LOG_DIR="$ROOT/logs"
STATUS_FILE="$PIPELINE_LOG_DIR/pipeline.status"
LOCK_FILE="$PIPELINE_LOG_DIR/pipeline.lock"

mkdir -p "$PIPELINE_LOG_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another OmniSQL pipeline owns $LOCK_FILE" >&2
  exit 1
fi

finish() {
  local status=$?
  if [[ "$status" -eq 0 ]]; then
    printf 'completed %s\n' "$(date --iso-8601=seconds)" > "$STATUS_FILE"
  else
    printf 'failed exit=%s %s\n' "$status" "$(date --iso-8601=seconds)" > "$STATUS_FILE"
  fi
}
trap finish EXIT
printf 'waiting_for_model %s\n' "$(date --iso-8601=seconds)" > "$STATUS_FILE"

if [[ -f "$DOWNLOAD_PID_FILE" ]]; then
  download_pid="$(cat "$DOWNLOAD_PID_FILE")"
  while kill -0 "$download_pid" 2>/dev/null; do
    sleep 30
  done
fi

printf 'verifying_model %s\n' "$(date --iso-8601=seconds)" > "$STATUS_FILE"
if ! HF_ENDPOINT=https://hf-mirror.com timeout 300 "$HF" cache verify seeklhy/OmniSQL-7B \
    --revision af4eed67f561bbeea555c017dae4b38b93bac2eb \
    --local-dir "$MODEL_DIR" \
    --fail-on-missing-files; then
  echo "HF remote verification was unavailable; enforcing pinned size and SHA256 locally" >&2
fi

model_bytes="$(stat -c %s "$MODEL_DIR/model.safetensors")"
model_sha256="$(sha256sum "$MODEL_DIR/model.safetensors" | cut -d' ' -f1)"
if [[ "$model_bytes" != "15231272152" ]]; then
  echo "unexpected model.safetensors size: $model_bytes" >&2
  exit 1
fi
if [[ "$model_sha256" != "dbdd444b3233a3decb600547c8d3cda0e0118c7cfe8085b68a05c519b0e80b01" ]]; then
  echo "unexpected model.safetensors SHA256: $model_sha256" >&2
  exit 1
fi

gpu_pids="$(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d'
)"
if [[ -n "$gpu_pids" ]]; then
  echo "refusing to launch: GPU compute processes appeared: $gpu_pids" >&2
  exit 1
fi

printf 'smoke %s\n' "$(date --iso-8601=seconds)" > "$STATUS_FILE"
bash "$RUNNER" smoke

printf 'full %s\n' "$(date --iso-8601=seconds)" > "$STATUS_FILE"
bash "$RUNNER" full
