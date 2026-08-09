#!/usr/bin/env bash
set -euo pipefail

# Resumable, fail-closed mirror download for the exact public Qwen3-4B
# revision used by every 4B arm in this reproduction. Ordinary repository
# files are checked as Git blobs; LFS objects are checked by SHA-256.

MODEL_REPO="Qwen/Qwen3-4B"
MODEL_REVISION="1cfa9a7208912126459214e8b04321603b3df60c"
MODEL_ROOT="${MODEL_ROOT:-/home/dengyan/models/Qwen3-4B-TrustSQL-baseline}"
HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"
MIRROR_JOBS="${MIRROR_JOBS:-3}"

[[ "$MODEL_ROOT" == /home/dengyan/* ]] || {
  echo "MODEL_ROOT must be below /home/dengyan" >&2
  exit 2
}
[[ "$HF_MIRROR" == "https://hf-mirror.com" ]] || {
  echo "this manifest is pinned to https://hf-mirror.com" >&2
  exit 2
}
[[ "$MIRROR_JOBS" =~ ^[1-5]$ ]] || {
  echo "MIRROR_JOBS must be an integer from 1 through 5" >&2
  exit 2
}

files=(
  .gitattributes LICENSE README.md config.json generation_config.json merges.txt
  model-00001-of-00003.safetensors
  model-00002-of-00003.safetensors
  model-00003-of-00003.safetensors
  model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json
)

declare -A expected_size=(
  [.gitattributes]=1570
  [LICENSE]=11343
  [README.md]=16857
  [config.json]=726
  [generation_config.json]=239
  [merges.txt]=1671853
  [model-00001-of-00003.safetensors]=3957900840
  [model-00002-of-00003.safetensors]=3987450520
  [model-00003-of-00003.safetensors]=99630640
  [model.safetensors.index.json]=32819
  [tokenizer.json]=11422654
  [tokenizer_config.json]=9732
  [vocab.json]=2776833
)

declare -A digest_kind=(
  [.gitattributes]=git [LICENSE]=git [README.md]=git [config.json]=git
  [generation_config.json]=git [merges.txt]=git
  [model-00001-of-00003.safetensors]=sha256
  [model-00002-of-00003.safetensors]=sha256
  [model-00003-of-00003.safetensors]=sha256
  [model.safetensors.index.json]=git [tokenizer.json]=sha256
  [tokenizer_config.json]=git [vocab.json]=git
)

declare -A expected_digest=(
  [.gitattributes]=52373fe24473b1aa44333d318f578ae6bf04b49b
  [LICENSE]=6634c8cc3133b3848ec74b9f275acaaa1ea618ab
  [README.md]=2de5ee7eee214bb55ea33ec7505c5838a7adf7f6
  [config.json]=e49eccdc32f36da9c09cfa0e737084f9e0105e5e
  [generation_config.json]=20a8a9156fc8c3f25295ca067f61fdf120d517c5
  [merges.txt]=31349551d90c7606f325fe0f11bbb8bd5fa0d7c7
  [model-00001-of-00003.safetensors]=328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223
  [model-00002-of-00003.safetensors]=6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5
  [model-00003-of-00003.safetensors]=e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1
  [model.safetensors.index.json]=95c0a0059df040d75dc6c396b174382cf61d2f91
  [tokenizer.json]=aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
  [tokenizer_config.json]=417d038a63fa3de29cfde265caedae14d1a58d92
  [vocab.json]=4783fe10ac3adce15ac8f358ef5462739852c569
)

for command in curl git sha256sum flock pkill stat; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done

verify_file() {
  local path="$1" name="$2" actual_size actual_digest
  [[ -f "$path" ]] || return 1
  actual_size="$(stat -c '%s' "$path")"
  [[ "$actual_size" == "${expected_size[$name]}" ]] || return 1
  if [[ "${digest_kind[$name]}" == sha256 ]]; then
    actual_digest="$(sha256sum "$path")"
    actual_digest="${actual_digest%% *}"
  else
    actual_digest="$(git hash-object "$path")"
  fi
  [[ "$actual_digest" == "${expected_digest[$name]}" ]]
}

download_one() {
  local name="$1" destination partial url attempt partial_size
  destination="$MODEL_ROOT/$name"
  partial="$destination.partial"
  url="$HF_MIRROR/$MODEL_REPO/resolve/$MODEL_REVISION/$name"
  if verify_file "$destination" "$name"; then
    echo "verified existing $name"
    return 0
  fi
  [[ ! -e "$destination" ]] || {
    echo "refusing to overwrite invalid file: $destination" >&2
    return 3
  }
  if [[ -f "$partial" ]]; then
    partial_size="$(stat -c '%s' "$partial")"
    (( partial_size <= expected_size[$name] )) || {
      echo "oversized partial: $partial" >&2
      return 3
    }
  fi
  echo "downloading $name"
  for attempt in $(seq 1 20); do
    if curl --fail --location --silent --show-error \
      --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
      --retry 5 --retry-delay 2 --continue-at - --output "$partial" "$url"; then
      break
    fi
    echo "curl attempt $attempt failed for $name; resuming" >&2
    sleep 2
  done
  verify_file "$partial" "$name" || {
    echo "pinned validation failed: $partial" >&2
    return 4
  }
  mv "$partial" "$destination"
  echo "verified downloaded $name"
}

mkdir -p "$MODEL_ROOT"
exec 9>"$MODEL_ROOT/.trustsql-download.lock"
flock -n 9 || {
  echo "another downloader owns $MODEL_ROOT" >&2
  exit 6
}

active_pids=()
active_names=()
stop_workers() {
  trap - HUP INT TERM
  local worker
  for worker in "${active_pids[@]}"; do
    pkill -TERM -P "$worker" >/dev/null 2>&1 || true
    kill "$worker" >/dev/null 2>&1 || true
  done
  wait >/dev/null 2>&1 || true
  exit 130
}
trap stop_workers HUP INT TERM

wait_batch() {
  local failed=0 index
  for index in "${!active_pids[@]}"; do
    wait "${active_pids[$index]}" || {
      echo "download worker failed: ${active_names[$index]}" >&2
      failed=1
    }
  done
  active_pids=()
  active_names=()
  (( failed == 0 ))
}

for name in "${files[@]}"; do
  download_one "$name" &
  active_pids+=("$!")
  active_names+=("$name")
  if (( ${#active_pids[@]} >= MIRROR_JOBS )); then
    wait_batch
  fi
done
if (( ${#active_pids[@]} > 0 )); then
  wait_batch
fi
trap - HUP INT TERM

for name in "${files[@]}"; do
  verify_file "$MODEL_ROOT/$name" "$name" || {
    echo "final validation failed: $name" >&2
    exit 4
  }
done

printf '%s@%s\n' "$MODEL_REPO" "$MODEL_REVISION" >"$MODEL_ROOT/pinned_revision.txt"
echo "verified $MODEL_REPO@$MODEL_REVISION at $MODEL_ROOT"
