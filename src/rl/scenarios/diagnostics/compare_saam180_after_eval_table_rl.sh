#!/usr/bin/env bash
# Wait for the 180-question arm's matched eval to finish, then print the paired comparisons
# against the other three arms into one log file. Read-only; no GPU work.
set -Eeuo pipefail

BASE=/home/dengyan/tabular_rl_outputs
EVAL180=$BASE/evaluations/qwen3_4b_saam180_checkpoint6_actionable_20260917_table_rl
SFT_SAAM60=$BASE/evaluations/qwen3_4b_correctness_only_saam60_checkpoint4_graph_actionable_20260914_r4
NOSPAN=$BASE/evaluations/qwen3_4b_saam_nospan60_checkpoint4_actionable_20260917_table_rl_r3
GRPO_SPAN=$BASE/evaluations/qwen3_4b_grpo_span60_checkpoint4_actionable_20260917_table_rl_r4
COMPARE=/tmp/20260917_paired_eval_compare.py
OUT=$BASE/single_matched_evals_20260917/saam180_paired_comparison.txt

deadline=$((SECONDS + 4 * 3600))
while (( SECONDS < deadline )); do
  [[ -f "$EVAL180/results/merged/summary.json" ]] && break
  sleep 60
done
if [[ ! -f "$EVAL180/results/merged/summary.json" ]]; then
  echo "TIMEOUT: merged summary never appeared" >"$OUT"
  exit 1
fi

{
  echo "generated: $(date -Is)"
  echo
  echo "### SAAM180(ckpt-6) vs SAAM60+span (909) ###"
  python3 "$COMPARE" --base "$SFT_SAAM60" --candidate "$EVAL180" \
    --base-label SAAM60+span_909 --candidate-label SAAM180
  echo
  echo "### SAAM180(ckpt-6) vs GRPO+span (886) ###"
  python3 "$COMPARE" --base "$GRPO_SPAN" --candidate "$EVAL180" \
    --base-label GRPO+span --candidate-label SAAM180
  echo
  echo "### SAAM180(ckpt-6) vs SAAM-nospan (894) ###"
  python3 "$COMPARE" --base "$NOSPAN" --candidate "$EVAL180" \
    --base-label SAAM_nospan --candidate-label SAAM180
} >"$OUT" 2>&1
echo "wrote $OUT"
