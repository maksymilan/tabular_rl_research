#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="Qwen/Qwen3-8B"
MODEL_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
MODEL_ROOT="${MODEL_ROOT:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}"
DOWNLOAD_BACKEND="${DOWNLOAD_BACKEND:-mirror-curl}"
HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"
HF_BIN="${HF_BIN:-hf}"

files=(
  .gitattributes
  LICENSE
  README.md
  config.json
  generation_config.json
  merges.txt
  model-00001-of-00005.safetensors
  model-00002-of-00005.safetensors
  model-00003-of-00005.safetensors
  model-00004-of-00005.safetensors
  model-00005-of-00005.safetensors
  model.safetensors.index.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
)

# Metadata returned by the Hugging Face model-info API for the pinned revision.
# LFS objects use their published SHA256; ordinary Git blobs use Git blob IDs.
declare -A expected_size=(
  [.gitattributes]=1570
  [LICENSE]=11343
  [README.md]=16660
  [config.json]=728
  [generation_config.json]=239
  [merges.txt]=1671853
  [model-00001-of-00005.safetensors]=3996250744
  [model-00002-of-00005.safetensors]=3993160032
  [model-00003-of-00005.safetensors]=3959604768
  [model-00004-of-00005.safetensors]=3187841392
  [model-00005-of-00005.safetensors]=1244659840
  [model.safetensors.index.json]=32878
  [tokenizer.json]=11422654
  [tokenizer_config.json]=9732
  [vocab.json]=2776833
)

declare -A digest_kind=(
  [.gitattributes]=git
  [LICENSE]=git
  [README.md]=git
  [config.json]=git
  [generation_config.json]=git
  [merges.txt]=git
  [model-00001-of-00005.safetensors]=sha256
  [model-00002-of-00005.safetensors]=sha256
  [model-00003-of-00005.safetensors]=sha256
  [model-00004-of-00005.safetensors]=sha256
  [model-00005-of-00005.safetensors]=sha256
  [model.safetensors.index.json]=git
  [tokenizer.json]=sha256
  [tokenizer_config.json]=git
  [vocab.json]=git
)

declare -A expected_digest=(
  [.gitattributes]=52373fe24473b1aa44333d318f578ae6bf04b49b
  [LICENSE]=6634c8cc3133b3848ec74b9f275acaaa1ea618ab
  [README.md]=ecc3ebd0849aa08d9484bd911dddfd5261b10d30
  [config.json]=d46195ac87f837ad233d02b2f80f148bf7c005e0
  [generation_config.json]=20a8a9156fc8c3f25295ca067f61fdf120d517c5
  [merges.txt]=31349551d90c7606f325fe0f11bbb8bd5fa0d7c7
  [model-00001-of-00005.safetensors]=31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f
  [model-00002-of-00005.safetensors]=5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282
  [model-00003-of-00005.safetensors]=c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836
  [model-00004-of-00005.safetensors]=b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a
  [model-00005-of-00005.safetensors]=20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff
  [model.safetensors.index.json]=2b85c00f1b118961cd7a477e2bba0fe197a4ce1a
  [tokenizer.json]=aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
  [tokenizer_config.json]=417d038a63fa3de29cfde265caedae14d1a58d92
  [vocab.json]=4783fe10ac3adce15ac8f358ef5462739852c569
)

verify_file() {
  local path="$1"
  local name="$2"
  local actual_size actual_digest

  [[ -f "$path" ]] || return 1
  actual_size="$(stat -c '%s' "$path")"
  [[ "$actual_size" == "${expected_size[$name]}" ]] || return 1

  if [[ "${digest_kind[$name]}" == "sha256" ]]; then
    actual_digest="$(sha256sum "$path")"
    actual_digest="${actual_digest%% *}"
  else
    actual_digest="$(git hash-object "$path")"
  fi
  [[ "$actual_digest" == "${expected_digest[$name]}" ]]
}

download_one_with_mirror_curl() {
    local name="$1"
    local destination="$MODEL_ROOT/$name"
    local partial="$destination.partial"
    local url="$HF_MIRROR/$MODEL_REPO/resolve/$MODEL_REVISION/$name"
    local download_ok=0
    local attempt partial_size

    if verify_file "$destination" "$name"; then
      echo "Verified existing $name"
      return 0
    fi
    if [[ -e "$destination" ]]; then
      echo "Refusing to overwrite an invalid existing file: $destination" >&2
      exit 3
    fi
    if [[ -f "$partial" ]]; then
      partial_size="$(stat -c '%s' "$partial")"
      if (( partial_size > expected_size[$name] )); then
        echo "Partial file is larger than expected; inspect it manually: $partial" >&2
        exit 3
      fi
      if (( partial_size == expected_size[$name] )); then
        if ! verify_file "$partial" "$name"; then
          echo "Complete-size partial file has the wrong digest: $partial" >&2
          exit 4
        fi
        mv "$partial" "$destination"
        echo "Verified completed partial $name"
        return 0
      fi
    fi

    echo "Downloading $name from $HF_MIRROR"
    for attempt in $(seq 1 20); do
      if curl --fail --location --silent --show-error \
        --connect-timeout 30 \
        --speed-limit 1024 \
        --speed-time 60 \
        --retry 5 \
        --retry-delay 2 \
        --continue-at - \
        --output "$partial" \
        "$url"; then
        download_ok=1
        break
      fi
      if [[ -f "$partial" ]] && (( $(stat -c '%s' "$partial") > expected_size[$name] )); then
        echo "Partial file became larger than expected: $partial" >&2
        exit 3
      fi
      echo "curl attempt $attempt failed for $name; retrying from the partial file" >&2
      sleep 2
    done
    if [[ "$download_ok" -ne 1 ]]; then
      echo "Mirror download exhausted retries: $name" >&2
      exit 5
    fi

    if ! verify_file "$partial" "$name"; then
      echo "Downloaded file failed pinned size/digest validation: $partial" >&2
      exit 4
    fi
    mv "$partial" "$destination"
    echo "Verified downloaded $name"
}

download_with_mirror_curl() {
  local jobs="${MIRROR_JOBS:-3}"
  local name batch_failed index worker_pid
  local -a active_pids=()
  local -a active_names=()

  stop_download_workers() {
    trap - HUP INT TERM
    for worker_pid in "${active_pids[@]}"; do
      pkill -TERM -P "$worker_pid" >/dev/null 2>&1 || true
      kill "$worker_pid" >/dev/null 2>&1 || true
    done
    wait >/dev/null 2>&1 || true
    exit 130
  }
  trap stop_download_workers HUP INT TERM

  command -v curl >/dev/null 2>&1 || {
    echo "curl is required for DOWNLOAD_BACKEND=mirror-curl" >&2
    exit 2
  }
  command -v git >/dev/null 2>&1 || {
    echo "git is required to validate non-LFS Hugging Face blobs" >&2
    exit 2
  }
  command -v sha256sum >/dev/null 2>&1 || {
    echo "sha256sum is required to validate model weights" >&2
    exit 2
  }
  command -v flock >/dev/null 2>&1 || {
    echo "flock is required to prevent concurrent writers" >&2
    exit 2
  }
  command -v pkill >/dev/null 2>&1 || {
    echo "pkill is required for child-process cleanup" >&2
    exit 2
  }
  if [[ ! "$jobs" =~ ^[1-5]$ ]]; then
    echo "MIRROR_JOBS must be an integer from 1 through 5" >&2
    exit 2
  fi

  mkdir -p "$MODEL_ROOT"
  exec 9>"$MODEL_ROOT/.trustsql-download.lock"
  if ! flock -n 9; then
    echo "Another verified model download is already active for $MODEL_ROOT" >&2
    exit 6
  fi
  for name in "${files[@]}"; do
    download_one_with_mirror_curl "$name" &
    active_pids+=("$!")
    active_names+=("$name")

    if (( ${#active_pids[@]} >= jobs )); then
      batch_failed=0
      for index in "${!active_pids[@]}"; do
        if ! wait "${active_pids[$index]}"; then
          echo "Download worker failed: ${active_names[$index]}" >&2
          batch_failed=1
        fi
      done
      [[ "$batch_failed" -eq 0 ]] || exit 5
      active_pids=()
      active_names=()
    fi
  done

  batch_failed=0
  for index in "${!active_pids[@]}"; do
    if ! wait "${active_pids[$index]}"; then
      echo "Download worker failed: ${active_names[$index]}" >&2
      batch_failed=1
    fi
  done
  [[ "$batch_failed" -eq 0 ]] || exit 5
  trap - HUP INT TERM
}

download_with_hf_cli() {
  command -v "$HF_BIN" >/dev/null 2>&1 || {
    echo "hf CLI is required for DOWNLOAD_BACKEND=hf-cli" >&2
    exit 2
  }
  HF_ENDPOINT="${HF_ENDPOINT:-$HF_MIRROR}" "$HF_BIN" download "$MODEL_REPO" \
    --revision "$MODEL_REVISION" \
    --local-dir "$MODEL_ROOT" \
    --max-workers "${HF_MAX_WORKERS:-4}"

  for name in "${files[@]}"; do
    if ! verify_file "$MODEL_ROOT/$name" "$name"; then
      echo "hf CLI output failed pinned validation: $MODEL_ROOT/$name" >&2
      exit 4
    fi
  done
}

case "$DOWNLOAD_BACKEND" in
  mirror-curl)
    download_with_mirror_curl
    ;;
  hf-cli)
    download_with_hf_cli
    ;;
  *)
    echo "DOWNLOAD_BACKEND must be mirror-curl or hf-cli" >&2
    exit 2
    ;;
esac

test -f "$MODEL_ROOT/config.json"
test -f "$MODEL_ROOT/model.safetensors.index.json"
echo "Verified $MODEL_REPO@$MODEL_REVISION at $MODEL_ROOT"
