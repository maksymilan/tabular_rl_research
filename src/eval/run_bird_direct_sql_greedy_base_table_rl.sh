#!/usr/bin/env bash
set -euo pipefail

# Greedy current-prompt baseline scored with official BIRD set denotation and a 20-second SQL limit.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
N_SAMPLES=1 PASS_K=1 TEMPERATURE=0 TOP_P=1 RESULT_VARIANT=greedy \
  exec "$SCRIPT_DIR/run_bird_direct_sql_passk4_base_table_rl.sh"
