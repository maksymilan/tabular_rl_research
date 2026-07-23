#!/usr/bin/env bash
set -euo pipefail

# Resumable fallback for old curl/HF-client combinations on table_rl. Each
# transfer is capped at ten minutes, then a fresh signed object-store URL is
# resolved and curl resumes the existing partial file.

MODEL_DIR="${MODEL_DIR:-/home/dengyan/models/text2sql_reproduction/OmniSQL-7B-af4eed67}"
REVISION="${REVISION:-af4eed67f561bbeea555c017dae4b38b93bac2eb}"
EXPECTED_BYTES=15231272152
EXPECTED_SHA256=dbdd444b3233a3decb600547c8d3cda0e0118c7cfe8085b68a05c519b0e80b01
TRANSFER_MAX_TIME="${TRANSFER_MAX_TIME:-600}"
RESOLVE_URL="https://hf-mirror.com/seeklhy/OmniSQL-7B/resolve/$REVISION/model.safetensors"
PART_FILE="$MODEL_DIR/model.safetensors.part"
FINAL_FILE="$MODEL_DIR/model.safetensors"

mkdir -p "$MODEL_DIR"
if [[ -f "$FINAL_FILE" ]]; then
  actual_bytes="$(stat -c %s "$FINAL_FILE")"
  actual_sha256="$(sha256sum "$FINAL_FILE" | cut -d' ' -f1)"
  if [[ "$actual_bytes" == "$EXPECTED_BYTES" && "$actual_sha256" == "$EXPECTED_SHA256" ]]; then
    echo "verified existing $FINAL_FILE"
    exit 0
  fi
  echo "existing final file does not match the pinned artifact" >&2
  exit 1
fi

previous_bytes=-1
no_progress_count=0
while true; do
  current_bytes=0
  if [[ -f "$PART_FILE" ]]; then
    current_bytes="$(stat -c %s "$PART_FILE")"
  fi
  if (( current_bytes == EXPECTED_BYTES )); then
    break
  fi
  if (( current_bytes > EXPECTED_BYTES )); then
    echo "partial file is larger than the expected artifact: $current_bytes" >&2
    exit 1
  fi
  if (( current_bytes == previous_bytes )); then
    no_progress_count=$((no_progress_count + 1))
  else
    no_progress_count=0
  fi
  if (( no_progress_count >= 12 )); then
    echo "download made no progress across 12 fresh signed URLs" >&2
    exit 1
  fi
  previous_bytes="$current_bytes"
  echo "$(date --iso-8601=seconds) resuming at $current_bytes/$EXPECTED_BYTES"

  signed_url="$(
    curl \
      --silent \
      --show-error \
      --head \
      --connect-timeout 30 \
      --max-time 60 \
      --retry 5 \
      --retry-delay 5 \
      "$RESOLVE_URL" |
      tr -d '\r' |
      sed -n 's/^location: //Ip' |
      tail -1
  )"
  if [[ -z "$signed_url" ]]; then
    echo "mirror did not return a signed object URL; retrying" >&2
    sleep 5
    continue
  fi

  curl \
    --show-error \
    --fail \
    --location \
    --continue-at - \
    --output "$PART_FILE" \
    --connect-timeout 30 \
    --max-time "$TRANSFER_MAX_TIME" \
    --speed-limit 1024 \
    --speed-time 60 \
    "$signed_url" || true
done

actual_sha256="$(sha256sum "$PART_FILE" | cut -d' ' -f1)"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "downloaded artifact SHA256 mismatch: $actual_sha256" >&2
  exit 1
fi
mv "$PART_FILE" "$FINAL_FILE"
echo "downloaded and verified $FINAL_FILE"
