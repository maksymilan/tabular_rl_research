#!/usr/bin/env bash
# Start SFT2 full-dev greedy shards only after the matching candidate480 shard completes.
set -euo pipefail
GPU_ID=${1:?usage: watchdog_sft2_full_dev_greedy_newgnn.sh 6|7}
case "$GPU_ID" in
  6) PARITY=0 ;;
  7) PARITY=1 ;;
  *) exit 2 ;;
esac
O=/home/dengyan/tabular_rl_outputs
R=$O/eval_runtime_version36_20260728
dependency="$O/logs/mixed_pool_append480_newgnn_gpu${GPU_ID}_20260801.status"
status="$O/logs/sft2_full_dev_greedy_watchdog_gpu${GPU_ID}_20260801.status"
if ! grep -q $'\tcomplete\t' "$dependency" 2>/dev/null; then
  printf '%s\twaiting_dependency\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$dependency" >"$status"
  exit 0
fi
target="$O/logs/sft2_version36_dev1534_greedy_t0_logprobs20_evenodd${PARITY}_20260801.status"
if grep -q $'\tcomplete\t' "$target" 2>/dev/null; then
  printf '%s\tcomplete\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$target" >"$status"
  exit 0
fi
if pgrep -u dengyan -f "run_sft2_full_dev_greedy_shard_newgnn.sh.*GPU_ID=$GPU_ID" >/dev/null 2>&1; then
  printf '%s\trunning\tgpu=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$GPU_ID" >"$status"
  exit 0
fi
printf '%s\tstarting\tgpu=%s parity=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$GPU_ID" "$PARITY" >"$status"
cd "$R"
nohup env GPU_ID="$GPU_ID" PARITY="$PARITY" \
  bash src/rl/experiments/run_sft2_full_dev_greedy_shard_newgnn.sh \
  >>"$O/logs/sft2_full_dev_greedy_watchdog_gpu${GPU_ID}_20260801.log" 2>&1 &
