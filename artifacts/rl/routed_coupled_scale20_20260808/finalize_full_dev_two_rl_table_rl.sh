#!/usr/bin/env bash
set -euo pipefail

output_root=/home/dengyan/tabular_rl_outputs
eval_runtime="$output_root/eval_runtime_version36_20260728"
summary_runtime="$output_root/rl_runtime_streaming_teacher_union_scale120_v2_20260804"
diagnostic_root="$output_root/diagnostics/routed_coupled_scale20_20260808"
python_bin=/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python

scale5_status="$diagnostic_root/scale5_full_dev.status"
scale20_status="$diagnostic_root/scale20_full_dev.status"
final_status="$diagnostic_root/two_rl_full_dev_summary.status"
scale5_result="$eval_runtime/data/results/routed_coupled_scale5_keyweighted_version36_dev1534_greedy_20260809/all.jsonl"
scale20_result="$eval_runtime/data/results/routed_coupled_scale20_lr6e6_version36_dev1534_greedy_20260809/all.jsonl"
sft2_result="$eval_runtime/data/results/sft2_version36_dev1534_greedy_t0_logprobs20_20260801/all.jsonl"
eval_inputs="$eval_runtime/data/eval_inputs/bird_dev_20240627.jsonl"
summarizer="$summary_runtime/src/rl/distillation/summarize_full_dev.py"

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$final_status"; }
state_of() { awk -F '\t' 'NR==1 {print $2}' "$1" 2>/dev/null || printf missing; }

for ((check = 1; check <= 1440; check++)); do
  scale5_state=$(state_of "$scale5_status")
  scale20_state=$(state_of "$scale20_status")
  if [[ "$scale5_state" == failed || "$scale20_state" == failed ]]; then
    set_status failed "scale5=$scale5_state scale20=$scale20_state"
    exit 1
  fi
  if [[ "$scale5_state" == complete && "$scale20_state" == complete ]]; then
    break
  fi
  set_status waiting "scale5=$scale5_state scale20=$scale20_state"
  sleep 60
done

if [[ "$(state_of "$scale5_status")" != complete || "$(state_of "$scale20_status")" != complete ]]; then
  set_status failed "timed out waiting for both full-dev runs"
  exit 1
fi

test "$(wc -l <"$scale5_result" | tr -d '[:space:]')" -eq 1534
test "$(wc -l <"$scale20_result" | tr -d '[:space:]')" -eq 1534
test "$(wc -l <"$sft2_result" | tr -d '[:space:]')" -eq 1534

scale5_summary="$diagnostic_root/scale5_full_dev_summary.json"
if [[ ! -f "$scale5_summary" ]]; then
  "$python_bin" "$summarizer" \
    --candidate "$scale5_result" \
    --baseline "sft2:$sft2_result" \
    --baseline "scale20:$scale20_result" \
    --eval-inputs "$eval_inputs" \
    --output "$scale5_summary"
fi

scale20_summary="$diagnostic_root/scale20_full_dev_summary.json"
if [[ ! -f "$scale20_summary" ]]; then
  "$python_bin" "$summarizer" \
    --candidate "$scale20_result" \
    --baseline "sft2:$sft2_result" \
    --baseline "scale5:$scale5_result" \
    --eval-inputs "$eval_inputs" \
    --output "$scale20_summary"
fi

set_status complete "scale5=$scale5_summary scale20=$scale20_summary"
