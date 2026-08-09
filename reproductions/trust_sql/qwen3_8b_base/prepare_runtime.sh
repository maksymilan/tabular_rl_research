#!/usr/bin/env bash
set -euo pipefail

EXPECTED_COMMIT="89df0661ad6b8e29ed8e61f7c950fbc2c1678b08"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-/home/dengyan/tabular_rl_outputs/reproductions/trust_sql/third_party/TrustSQL}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/dengyan/tabular_rl_outputs/reproductions/trust_sql/runtime/TrustSQL-qwen3-8b-base}"
PATCH_FILE="$SCRIPT_DIR/official_eval_vllm019.patch"

test -d "$UPSTREAM_ROOT/.git"
test -f "$PATCH_FILE"

actual_commit="$(git -C "$UPSTREAM_ROOT" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "Unexpected TRUST-SQL commit: $actual_commit" >&2
  exit 2
fi

if [[ -e "$RUNTIME_ROOT" ]]; then
  echo "Runtime already exists; refusing to overwrite: $RUNTIME_ROOT" >&2
  exit 2
fi

mkdir -p "$(dirname -- "$RUNTIME_ROOT")"
git clone --quiet --no-hardlinks "$UPSTREAM_ROOT" "$RUNTIME_ROOT"
git -C "$RUNTIME_ROOT" apply --check "$PATCH_FILE"
git -C "$RUNTIME_ROOT" apply "$PATCH_FILE"
git -C "$RUNTIME_ROOT" diff --check

test "$(git -C "$RUNTIME_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
echo "Prepared patched evaluation runtime: $RUNTIME_ROOT"
git -C "$RUNTIME_ROOT" status --short
