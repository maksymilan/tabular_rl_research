#!/usr/bin/env bash
set -euo pipefail

SFT_CHECKPOINT="${TRUSTSQL_SFT_CHECKPOINT:-}"
GPU_ROWS="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"
GPU_COUNT="$(printf '%s\n' "${GPU_ROWS}" | sed '/^[[:space:]]*$/d' | wc -l)"
A100_COUNT="$(printf '%s\n' "${GPU_ROWS}" | grep -c 'A100' || true)"

if [[ "${GPU_COUNT}" -lt 8 || "${A100_COUNT}" -lt 8 ]]; then
  echo "Paper-scale Qwen3-4B RL requires 8 A100 GPUs; this host exposes ${GPU_COUNT} GPU(s), ${A100_COUNT} A100." >&2
  exit 2
fi
if [[ -z "${SFT_CHECKPOINT}" || ! -d "${SFT_CHECKPOINT}" ]]; then
  echo "TRUSTSQL_SFT_CHECKPOINT must point to the unreleased paper SFT warm-up checkpoint." >&2
  exit 3
fi
python3 -c 'import sglang, megatron.core, ray'
echo "Hardware and runtime prerequisites pass; a paper checkpoint was supplied."
