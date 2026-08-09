#!/usr/bin/env bash
set -euo pipefail

REPRO_ROOT="${TRUSTSQL_REPRO_ROOT:-/home/dengyan/tabular_rl_outputs/reproductions/trust_sql}"
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${TRUSTSQL_PYTHON:-/home/dengyan/miniconda3/envs/sft/bin/python}"
BIRD_ROOT="${TRUSTSQL_BIRD_ROOT:-/home/dengyan/tabular_rl_project/data/bird/train/train_databases}"
SPIDER_ROOT="${TRUSTSQL_SPIDER_ROOT:-/home/dengyan/tabular_rl_project/data/spider_data/database}"
UPSTREAM_DIR="${TRUSTSQL_UPSTREAM_DIR:-${REPRO_ROOT}/third_party/TrustSQL}"

export TRUSTSQL_REPRO_ROOT="${REPRO_ROOT}"
export TRUSTSQL_UPSTREAM_DIR="${UPSTREAM_DIR}"
export TRUSTSQL_PYTHON="${PYTHON_BIN}"

bash "${SOURCE_DIR}/prepare_official.sh"

"${PYTHON_BIN}" "${SOURCE_DIR}/collect_preflight.py" \
  --output "${REPRO_ROOT}/manifests/host_preflight.json"

"${PYTHON_BIN}" "${SOURCE_DIR}/audit_released_rl_data.py" \
  --input "${UPSTREAM_DIR}/data_for_sql/filtered_questions.jsonl" \
  --bird-database-root "${BIRD_ROOT}" \
  --spider-database-root "${SPIDER_ROOT}" \
  --output "${REPRO_ROOT}/data/filtered_questions.table_rl.jsonl" \
  --manifest "${REPRO_ROOT}/manifests/released_rl_data_audit.json"

echo "TRUST-SQL preflight completed under ${REPRO_ROOT}"
