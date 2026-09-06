#!/usr/bin/env bash
# Idempotent cron entrypoint for one mixed-pool search shard.
set -euo pipefail

: "${STATUS:?STATUS is required}"
: "${LAUNCHER:?LAUNCHER is required}"
state=""
if [[ -f "$STATUS" ]]; then
  state=$(awk -F '\t' 'NR==1 {print $2}' "$STATUS")
fi
if [[ "$state" == complete ]]; then
  exit 0
fi
exec bash "$LAUNCHER"
