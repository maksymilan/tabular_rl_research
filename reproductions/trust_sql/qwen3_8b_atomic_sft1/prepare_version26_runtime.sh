#!/usr/bin/env bash
set -euo pipefail

# Export immutable historical code instead of importing the current version39/version54 tree.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
EXPECTED_COMMIT="4cd47c957fc6ae791e76a10594c8cd22f4d3b6de"
RUNTIME_ROOT="${RUNTIME_ROOT:-$PROJECT_DIR/tmp/version26-runtime-$EXPECTED_COMMIT}"
VERIFY_PYTHON="${VERIFY_PYTHON:-python3}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

test -f "$SCRIPT_DIR/runtime_lock.json" || die "missing runtime_lock.json"
test -f "$SCRIPT_DIR/verify_version26_runtime.py" || die "missing runtime verifier"
git -C "$PROJECT_DIR" cat-file -e "${EXPECTED_COMMIT}^{commit}" 2>/dev/null ||
  die "source commit is unavailable in $PROJECT_DIR: $EXPECTED_COMMIT"

resolved_commit="$(git -C "$PROJECT_DIR" rev-parse "$EXPECTED_COMMIT")"
test "$resolved_commit" = "$EXPECTED_COMMIT" ||
  die "source commit resolved unexpectedly: $resolved_commit"

for item in src/eval src/sft src/harness; do
  expected_tree="$($VERIFY_PYTHON -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["source_git_trees"][sys.argv[2]])' \
    "$SCRIPT_DIR/runtime_lock.json" "$item")"
  actual_tree="$(git -C "$PROJECT_DIR" rev-parse "${EXPECTED_COMMIT}:${item}")"
  test "$actual_tree" = "$expected_tree" ||
    die "source tree mismatch for $item: expected $expected_tree, got $actual_tree"
done

if test -d "$RUNTIME_ROOT"; then
  "$VERIFY_PYTHON" "$SCRIPT_DIR/verify_version26_runtime.py" \
    --runtime-root "$RUNTIME_ROOT"
  printf 'existing exact runtime is valid: %s\n' "$RUNTIME_ROOT"
  exit 0
fi
test ! -e "$RUNTIME_ROOT" || die "runtime target exists and is not a directory: $RUNTIME_ROOT"

runtime_parent="$(dirname -- "$RUNTIME_ROOT")"
mkdir -p "$runtime_parent"
staging="$(mktemp -d "$runtime_parent/.version26-runtime.XXXXXX")"

cleanup_staging() {
  if test -n "${staging:-}" && test -d "$staging"; then
    rm -rf -- "$staging"
  fi
}
trap cleanup_staging EXIT INT TERM

git -C "$PROJECT_DIR" archive "$EXPECTED_COMMIT" src/eval src/sft src/harness |
  tar -x -C "$staging"
cp "$SCRIPT_DIR/runtime_lock.json" "$staging/runtime_lock.json"

"$VERIFY_PYTHON" "$SCRIPT_DIR/verify_version26_runtime.py" \
  --runtime-root "$staging"
mv "$staging" "$RUNTIME_ROOT"
staging=""
trap - EXIT INT TERM

printf 'prepared exact version26 runtime: %s\n' "$RUNTIME_ROOT"
