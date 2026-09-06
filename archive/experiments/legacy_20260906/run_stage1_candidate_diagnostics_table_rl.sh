#!/usr/bin/env bash
# Fixed-prefix diagnostic -> full-dev greedy@1 -> predeclared test requirements.
set -euo pipefail
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TR=${TRAIN_RUNTIME:-$O/rl_runtime_rank_score_v3_20260731}
ER=${EVAL_RUNTIME:-$O/eval_runtime_version36_20260728}
FP=${FIXED_PREFIX_DIR:-$ER/data/diagnostics/fixed_prefix_dev300_union4_v2}
DECISION_ROOT=${DECISION_ROOT:-$O/phase8_controlled_20260801}
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
DIAGNOSTIC_GPU_ID=${DIAGNOSTIC_GPU_ID:-1}
PORT=${PORT:-18083}
for name in CANDIDATE_NAME ADAPTER STATUS RUN_LOG; do [[ -n "${!name:-}" ]] || exit 2; done
timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
on_exit() { local c=$?; trap - EXIT; if [[ "$c" -ne 0 ]]; then set_status failed "exit=$c see=$RUN_LOG"; fi; exit "$c"; }
trap on_exit EXIT
mkdir -p "$O/logs" "$FP/scores"
exec >>"$RUN_LOG" 2>&1
dataset="$FP/fixed_prefix_dev300.jsonl"
prefix_scores="$FP/scores/$CANDIDATE_NAME.jsonl"
prefix_parquet="$FP/scores/$CANDIDATE_NAME.parquet"
pairs=$(wc -l < "$dataset" | tr -d '[:space:]')
if [[ ! -f "$prefix_scores" || ! -f "$prefix_parquet" \
  || "$(wc -l < "$prefix_scores")" -ne "$pairs" ]]; then
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | sed -n "$((DIAGNOSTIC_GPU_ID + 1))p")
    [[ -n "$used" && "$used" -le 512 ]] && break
    set_status waiting_gpu "stage=prefix gpu=$DIAGNOSTIC_GPU_ID mib=${used:-unknown}"
    sleep 30
  done
  set_status fixed_prefix "candidate=$CANDIDATE_NAME pairs=$pairs"
  cd "$ER"
  CUDA_VISIBLE_DEVICES="$DIAGNOSTIC_GPU_ID" "$PY" src/rl/diagnostics/score_action_candidates.py \
    --dataset "$dataset" --checkpoint-name "$CANDIDATE_NAME" \
    --base-model /home/dengyan/models/Qwen2.5-Coder-7B-Instruct \
    --adapter "$ADAPTER" --output-jsonl "$prefix_scores" \
    --output-parquet "$prefix_parquet" --device cuda:0
fi
greedy="$ER/data/results/stage1_${CANDIDATE_NAME}_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260802"
set_status full_dev_greedy "candidate=$CANDIDATE_NAME questions=1534"
ADAPTER="$ADAPTER" SERVED_MODEL="stage1-$CANDIDATE_NAME-full-dev-greedy" RESULT_DIR="$greedy" \
STATUS="$O/logs/stage1_${CANDIDATE_NAME}.full_dev_greedy.status" \
RUN_LOG="$O/logs/stage1_${CANDIDATE_NAME}.full_dev_greedy.run.log" PORT="$PORT" \
EVAL_GPU_ID="$DIAGNOSTIC_GPU_ID" RUNTIME="$ER" OUTPUT_ROOT="$O" \
  bash "$TR/src/rl/experiments/run_full_dev_greedy_table_rl.sh"

sft2_greedy="$ER/data/results/sft2_version36_dev1534_greedy_t0_logprobs20_20260801/all.jsonl"
while [[ ! -f "$sft2_greedy" || "$(wc -l < "$sft2_greedy")" -ne 1534 ]]; do
  set_status waiting_baseline "candidate=$CANDIDATE_NAME baseline=sft2_greedy"; sleep 60
done
requirements="$DECISION_ROOT/$CANDIDATE_NAME/decision_requirements.json"
set_status checking_requirements "candidate=$CANDIDATE_NAME"
"$PY" "$TR/src/rl/experiments/evaluate_stage1_candidate_gate.py" \
  --candidate-name "$CANDIDATE_NAME" --sft2-greedy "$sft2_greedy" \
  --candidate-greedy "$greedy/all.jsonl" --sft2-prefix "$FP/scores/sft2.jsonl" \
  --candidate-prefix "$prefix_scores" --expected-questions 1534 --output "$requirements"
passed=$("$PY" - "$requirements" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1]))["status"] == "passed"))
PY
)
if [[ "$passed" == 1 ]]; then
  set_status complete "candidate=$CANDIDATE_NAME requirements=passed full_dev_greedy=complete k4=deferred_to_finalist"
else
  set_status complete "candidate=$CANDIDATE_NAME requirements=failed full_dev_greedy=complete k4=skipped"
fi
