#!/usr/bin/env bash
# Stable project-level entrypoint for the common post-training evaluation handoff.
# Keep the implementation in experiments/ for backward compatibility with the
# first span-balanced diagnostic launcher; all behavior is controlled by the
# environment variables documented in docs/current/evaluation_handoff.md.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
IMPLEMENTATION=""
for candidate in \
  "$HERE/../experiments/run_qwen3_8b_v26_spanbalanced_saam_matched_eval_newgnn.sh" \
  "$HERE/../run_qwen3_8b_v26_spanbalanced_saam_matched_eval_newgnn.sh" \
  "$HERE/run_qwen3_8b_v26_spanbalanced_saam_matched_eval_newgnn.sh"; do
  if [[ -f "$candidate" ]]; then
    IMPLEMENTATION=$candidate
    break
  fi
done
if [[ -z "$IMPLEMENTATION" ]]; then
  echo "matched eval handoff implementation is missing beside $0" >&2
  exit 2
fi
exec bash "$IMPLEMENTATION" "$@"
