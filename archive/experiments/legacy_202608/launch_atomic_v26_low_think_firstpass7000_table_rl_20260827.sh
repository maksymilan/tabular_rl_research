#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/dengyan/miniconda3/envs/sft/bin/python
RUNTIME=/home/dengyan/tabular_rl_outputs/runtime/atomic_v26_low_think_scaleout_20260827
DATA=/home/dengyan/tabular_rl_outputs/data/atomic_v26_low_think_scaleout_20260827
RESULT=/home/dengyan/tabular_rl_outputs/trajectories/atomic_v26_low_think_diverse_firstpass7000_20260827
TASKS=$DATA/atomic_v26_low_think_diverse_firstpass7000.table_rl.tasks.jsonl
MANIFEST=$DATA/atomic_v26_low_think_diverse_firstpass7000.manifest.json
SUPERVISOR=$RUNTIME/run_atomic_v26_low_think_scaleout_server.py

mkdir -p "$RESULT"
nohup "$PYTHON" -u "$SUPERVISOR" \
  --tasks "$TASKS" \
  --selection-manifest "$MANIFEST" \
  --result-root "$RESULT" \
  --expected-tasks-sha256 2b397d73f5cde9fe4cc355f0ba877e8088949d799388540e7e0c2920d614c581 \
  --expected-manifest-sha256 7d479c86cda9f316c53a47be6cc01b82a5247606012ef048900ce7a52bbe39c0 \
  --episodes 7000 \
  --total-token-cap 320000000 \
  --shard-size 25 \
  --max-processes 3 \
  --workers 8 \
  --token-reservation-per-shard 2500000 \
  --max-provider-failures 3 \
  > "$RESULT/supervisor.log" 2>&1 < /dev/null &
echo $!
