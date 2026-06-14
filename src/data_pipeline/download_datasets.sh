#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"

mkdir -p "${DATA_DIR}"

if ! command -v hf >/dev/null 2>&1; then
  echo "error: Hugging Face CLI 'hf' is required. Install huggingface_hub first." >&2
  exit 1
fi

download_dataset() {
  local repo_id="$1"
  local local_name="$2"
  local target_dir="${DATA_DIR}/${local_name}"

  echo "Downloading ${repo_id} -> ${target_dir}"
  hf download "${repo_id}" \
    --repo-type dataset \
    --local-dir "${target_dir}"
}

download_dataset "table-benchmark/tqabench" "tqabench"
download_dataset "DongfuJiang/FeTaQA" "FeTaQA"
download_dataset "Multilingual-Multimodal-NLP/TableBench" "TableBench"

cat <<EOF

Done.
Data directory: ${DATA_DIR}

Useful dataset viewer URLs:
- https://huggingface.co/datasets/table-benchmark/tqabench/viewer/default/train
- https://huggingface.co/datasets/DongfuJiang/FeTaQA/viewer/default/train
- https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench/viewer/table_bench/TQA_test
- https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench/viewer/table_bench/Instruct_test
EOF
