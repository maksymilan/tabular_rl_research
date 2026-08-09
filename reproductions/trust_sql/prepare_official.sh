#!/usr/bin/env bash
set -euo pipefail

PINNED_COMMIT="89df0661ad6b8e29ed8e61f7c950fbc2c1678b08"
REPRO_ROOT="${TRUSTSQL_REPRO_ROOT:-/home/dengyan/tabular_rl_outputs/reproductions/trust_sql}"
UPSTREAM_DIR="${TRUSTSQL_UPSTREAM_DIR:-${REPRO_ROOT}/third_party/TrustSQL}"
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${TRUSTSQL_PYTHON:-python3}"

mkdir -p \
  "${REPRO_ROOT}/data" \
  "${REPRO_ROOT}/logs" \
  "${REPRO_ROOT}/manifests" \
  "${REPRO_ROOT}/results" \
  "${REPRO_ROOT}/third_party"

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  if [[ -e "${UPSTREAM_DIR}" ]]; then
    echo "Refusing to replace non-Git path: ${UPSTREAM_DIR}" >&2
    exit 1
  fi
  git clone https://github.com/JaneEyre0530/TrustSQL.git "${UPSTREAM_DIR}"
  git -C "${UPSTREAM_DIR}" checkout --detach "${PINNED_COMMIT}"
fi

ACTUAL_COMMIT="$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${PINNED_COMMIT}" ]]; then
  echo "TrustSQL checkout is not pinned: ${ACTUAL_COMMIT}" >&2
  exit 1
fi
if [[ -n "$(git -C "${UPSTREAM_DIR}" status --porcelain)" ]]; then
  echo "TrustSQL checkout has local modifications; refusing an unaudited snapshot." >&2
  exit 1
fi

"${PYTHON_BIN}" "${SOURCE_DIR}/verify_upstream.py" \
  --upstream "${UPSTREAM_DIR}" \
  --output "${REPRO_ROOT}/manifests/upstream_audit.json"

echo "Pinned TRUST-SQL source is ready at ${UPSTREAM_DIR}"
