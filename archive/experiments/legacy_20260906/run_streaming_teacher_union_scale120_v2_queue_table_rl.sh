#!/usr/bin/env bash
# GPU0 queue: expand frozen teacher eligibility -> strict balanced120 v2 -> full dev.
set -euo pipefail

O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TR=${TRAIN_RUNTIME:-$O/rl_runtime_streaming_teacher_union_scale120_v2_20260804}
ER=${EVAL_RUNTIME:-$O/eval_runtime_version36_20260728}
DATA_PROJECT_ROOT=${DATA_PROJECT_ROOT:-/home/dengyan/tabular_rl_project}
ROOT=${RUN_ROOT:-$O/streaming_teacher_union_scale120_v2_20260804}
OLD_ROOT=${OLD_RUN_ROOT:-$O/streaming_teacher_union_20260803}
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL=${MODEL_PATH:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
SFT2=${SFT2_ADAPTER:-$O/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
EXP15=${EXP15_ADAPTER:-$O/checkpoints/trl-transition-v26-exp15-fixed-prefix-action-dpo-sft2-seed101-20260801/final}
CANDIDATE=${CANDIDATE_POOL:-$O/phase8_mixed_pool_search_20260802/candidate_960_seed101}
SFT2_EXISTING=${SFT2_EXISTING_ROLLOUTS:-$O/phase8_dense_stage2_20260802/balanced_mixed120_dense_uniform_seed101/trajectories.jsonl}
EXP15_EXISTING=${EXP15_EXISTING_ROLLOUTS:-$OLD_ROOT/exp15_k4_balanced120/trajectories.jsonl}
EXISTING_ELIGIBILITY=${EXISTING_ELIGIBILITY:-$OLD_ROOT/teacher_eligibility_balanced120.jsonl}
STATUS=${STATUS:-$O/logs/streaming_teacher_union_scale120_v2_20260804.status}
RUN_LOG=${RUN_LOG:-$O/logs/streaming_teacher_union_scale120_v2_20260804.log}
LOCK=${LOCK:-$O/logs/streaming_teacher_union_scale120_v2_20260804.lock}
GPU_ID=${GPU_ID:-0}
SFT2_SHA=d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-$PYTORCH_CUDA_ALLOC_CONF}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
state_of() { awk -F '\t' 'NR==1 {print $2}' "$1" 2>/dev/null || printf missing; }
gpu_used() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | sed -n "$((GPU_ID + 1))p"
}
wait_gpu_free() {
  while true; do
    used=$(gpu_used)
    if [[ -n "$used" && "$used" -le 512 ]]; then return 0; fi
    set_status waiting_gpu "gpu=$GPU_ID mib=${used:-unknown} next=$1"
    sleep 30
  done
}
on_exit() {
  code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then set_status failed "exit=$code see=$RUN_LOG"; fi
  exit "$code"
}
trap on_exit EXIT

mkdir -p "$O/logs" "$ROOT"
exec 9>"$LOCK"
flock -n 9 || exit 0
exec >>"$RUN_LOG" 2>&1
cd "$TR"
[[ "$(grep -c '^PROTOCOL_VERSION = "version26"' src/sft/protocol.py)" -eq 1 ]]
[[ "$(sha256sum "$SFT2/adapter_model.safetensors" | awk '{print $1}')" == "$SFT2_SHA" ]]
for required in \
  "$EXP15/adapter_model.safetensors" \
  "$CANDIDATE/tasks.jsonl" \
  "$CANDIDATE/group_integrity_audit.json" \
  "$SFT2_EXISTING" \
  "$EXP15_EXISTING" \
  "$EXISTING_ELIGIBILITY"; do
  [[ -f "$required" ]] || { set_status blocked "missing=$required"; exit 4; }
done

EXPANSION=$ROOT/qualification_expansion118
if [[ ! -f "$EXPANSION/manifest.json" ]]; then
  [[ ! -e "$EXPANSION" ]] || {
    set_status blocked "partial qualification expansion requires audit: $EXPANSION"
    exit 4
  }
  set_status preparing_qualification_expansion "unused mixed SFT2-K4 candidates"
  "$PY" src/rl/distillation/prepare_teacher_union_expansion.py \
    --candidate-tasks "$CANDIDATE/tasks.jsonl" \
    --groups-dir "$CANDIDATE/groups" \
    --existing-eligibility "$EXISTING_ELIGIBILITY" \
    --output-dir "$EXPANSION" --group-size 4 --protocol-version version26
fi
"$PY" - "$EXPANSION/manifest.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"]=="frozen" and p["tasks"]==118 and p["trajectories"]==472
assert p["difficulty_counts"]=={"challenging":53,"moderate":35,"simple":30}
PY

EXP15_EXPANSION=$ROOT/exp15_k4_expansion118
if [[ ! -f "$EXP15_EXPANSION/manifest.pending.json" ]]; then
  wait_gpu_free teacher_qualification
  set_status teacher_qualification "gpu=$GPU_ID teacher=Exp15 tasks=118 K4 scheduler=dynamic window=8"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 \
  TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda \
    "$PY" src/rl/fixed_pool/generate_fixed_rollout_pool.py \
      --model-path "$MODEL" --adapter-path "$EXP15" \
      --tasks "$EXPANSION/tasks.jsonl" --output-dir "$EXP15_EXPANSION" \
      --group-size 4 --temperature 0.7 --top-p 0.95 --max-steps 30 \
      --max-new-tokens 1024 --max-context-tokens 8192 --history-turns 4 \
      --seed 1515 --gpu-memory-utilization 0.82 \
      --scheduler dynamic --question-window 8 --task-batch-size 4
  generation_code=$?
  set -e
  if [[ "$generation_code" -ne 0 ]]; then
    wait_gpu_free teacher_qualification_retry
    set_status teacher_qualification_retry "gpu=$GPU_ID preserve_groups=1 window=4 prior_exit=$generation_code"
    CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 \
    TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda \
      "$PY" src/rl/fixed_pool/generate_fixed_rollout_pool.py \
        --model-path "$MODEL" --adapter-path "$EXP15" \
        --tasks "$EXPANSION/tasks.jsonl" --output-dir "$EXP15_EXPANSION" \
        --group-size 4 --temperature 0.7 --top-p 0.95 --max-steps 30 \
        --max-new-tokens 1024 --max-context-tokens 8192 --history-turns 4 \
        --seed 1515 --gpu-memory-utilization 0.82 \
        --scheduler dynamic --question-window 4 --task-batch-size 4
  fi
fi
"$PY" - "$EXP15_EXPANSION/manifest.pending.json" "$EXP15_EXPANSION/trajectories.jsonl" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); rows=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
assert p["protocol_version"]=="version26" and p["tasks"]==118
assert p["group_size"]==4 and p["trajectories"]==472 and len(rows)==472
assert [r["sequence"] for r in rows]==list(range(472))
PY

COMBINED=$ROOT/combined_teacher_k4_mixed238
mkdir -p "$COMBINED"
if [[ ! -f "$COMBINED/sft2.manifest.json" ]]; then
  set_status combining_teacher_rollouts "teacher=SFT2 tasks=238"
  "$PY" src/rl/distillation/combine_teacher_rollouts.py \
    --input "$SFT2_EXISTING" --input "$EXPANSION/sft2_trajectories.jsonl" \
    --output "$COMBINED/sft2.trajectories.jsonl" \
    --manifest "$COMBINED/sft2.manifest.json" --expected-k 4
fi
if [[ ! -f "$COMBINED/exp15.manifest.json" ]]; then
  set_status combining_teacher_rollouts "teacher=Exp15 tasks=238"
  "$PY" src/rl/distillation/combine_teacher_rollouts.py \
    --input "$EXP15_EXISTING" --input "$EXP15_EXPANSION/trajectories.jsonl" \
    --output "$COMBINED/exp15.trajectories.jsonl" \
    --manifest "$COMBINED/exp15.manifest.json" --expected-k 4
fi

ALL_ELIGIBILITY=$ROOT/teacher_eligibility_mixed238.jsonl
ALL_ELIGIBILITY_MANIFEST=$ROOT/teacher_eligibility_mixed238.manifest.json
if [[ ! -f "$ALL_ELIGIBILITY_MANIFEST" ]]; then
  set_status freezing_expanded_routes "tasks=238 threshold=2of4"
  "$PY" src/rl/distillation/build_teacher_eligibility.py \
    --sft2-rollouts "$COMBINED/sft2.trajectories.jsonl" \
    --exp15-rollouts "$COMBINED/exp15.trajectories.jsonl" \
    --output "$ALL_ELIGIBILITY" --manifest "$ALL_ELIGIBILITY_MANIFEST" \
    --expected-k 4 --minimum-dense-correct 2 --protocol-version version26 \
    --order-seed 120
fi

ELIGIBILITY=$ROOT/teacher_eligibility_strict120.jsonl
ELIGIBILITY_MANIFEST=$ROOT/teacher_eligibility_strict120.manifest.json
if [[ ! -f "$ELIGIBILITY_MANIFEST" ]]; then
  set_status selecting_strict120 "category=30x4 difficulty=40x3"
  "$PY" src/rl/distillation/select_strict_teacher_union_scale.py \
    --eligibility "$ALL_ELIGIBILITY" \
    --source-manifest "$ALL_ELIGIBILITY_MANIFEST" \
    --tasks "$CANDIDATE/tasks.jsonl" \
    --output "$ELIGIBILITY" --manifest "$ELIGIBILITY_MANIFEST" \
    --per-category 30 --per-difficulty 40 --seed 120
fi
"$PY" - "$ELIGIBILITY_MANIFEST" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["tasks"]==120
assert set(p["category_counts"].values())=={30}
assert set(p["difficulty_counts"].values())=={40}
assert len(p["training_order"])==len(set(p["training_order"]))==120
PY

SMOKE=$ROOT/smoke2_batch2_repairk2_v2
if [[ "$($PY - "$SMOKE/status.json" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get("state","missing"))
except Exception: print("missing")
PY
)" != complete ]]; then
  [[ ! -e "$SMOKE" ]] || { set_status blocked "incomplete smoke requires audit: $SMOKE"; exit 4; }
  wait_gpu_free smoke
  set_status hardware_smoke "gpu=$GPU_ID tasks=2 batch=2 multi-repair-v2"
  CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 \
  TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda \
    "$PY" src/rl/distillation/train_streaming_teacher_union.py \
      --project-root "$DATA_PROJECT_ROOT" --eligibility "$ELIGIBILITY" \
      --eligibility-manifest "$ELIGIBILITY_MANIFEST" --model-path "$MODEL" \
      --sft2-adapter "$SFT2" --exp15-adapter "$EXP15" --output-dir "$SMOKE" \
      --limit 2 --question-batch-size 2 --repair-k 2 \
      --max-positive-repairs-per-group 2 --repair-generation-batch-size 4 \
      --seed 120 --experiment-name streaming_teacher_union_scale120_v2_smoke2
fi

TRAIN=$ROOT/formal120_strict_batch2_repairk4_v2
if [[ "$($PY - "$TRAIN/status.json" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get("state","missing"))
except Exception: print("missing")
PY
)" != complete ]]; then
  RESUME_ARGS=()
  if [[ -e "$TRAIN" ]]; then
    latest=$($PY - "$TRAIN/checkpoints" <<'PY'
import sys
from pathlib import Path
from src.rl.distillation.streaming_checkpoint import latest_complete_checkpoint
print(latest_complete_checkpoint(Path(sys.argv[1])) or "")
PY
)
    [[ -n "$latest" ]] || { set_status blocked "formal has no complete checkpoint: $TRAIN"; exit 4; }
    RESUME_ARGS=(--resume-latest)
    set_status formal_resuming "gpu=$GPU_ID checkpoint=$latest"
  fi
  wait_gpu_free formal_train
  [[ ${#RESUME_ARGS[@]} -ne 0 ]] || set_status formal_training "gpu=$GPU_ID tasks=120 batch=2 repairK=4 multi-positive=2"
  CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 \
  TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda \
    "$PY" src/rl/distillation/train_streaming_teacher_union.py \
      --project-root "$DATA_PROJECT_ROOT" --eligibility "$ELIGIBILITY" \
      --eligibility-manifest "$ELIGIBILITY_MANIFEST" --model-path "$MODEL" \
      --sft2-adapter "$SFT2" --exp15-adapter "$EXP15" --output-dir "$TRAIN" \
      --limit 120 --question-batch-size 2 --repair-k 4 \
      --max-positive-repairs-per-group 2 --repair-generation-batch-size 4 \
      --checkpoint-every-batches 5 --checkpoint-total-limit 3 \
      --seed 120 "${RESUME_ARGS[@]}" \
      --experiment-name streaming_sft2_exp15_strict120_task_balanced_multi_repair_v2
fi

FINAL_ADAPTER=$($PY - "$TRAIN/run_manifest.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p["status"]=="complete"; print(p["final_adapter"])
PY
)
[[ -f "$FINAL_ADAPTER/adapter_model.safetensors" ]]

RESULT=$ER/data/results/streaming_sft2_exp15_strict120_multi_repair_v2_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260804
EVAL_STATUS=$O/logs/streaming_sft2_exp15_strict120_multi_repair_v2.full_dev_greedy.status
if [[ "$(state_of "$EVAL_STATUS")" != complete ]]; then
  wait_gpu_free full_dev_greedy
  set_status full_dev_greedy "gpu=$GPU_ID questions=1534 temperature=0 top_p=1 concurrency=24"
  ADAPTER="$FINAL_ADAPTER" \
  SERVED_MODEL=streaming-sft2-exp15-strict120-multi-repair-v2-full-dev-greedy \
  RESULT_DIR="$RESULT" STATUS="$EVAL_STATUS" \
  RUN_LOG=$O/logs/streaming_sft2_exp15_strict120_multi_repair_v2.full_dev_greedy.log \
  PORT=18203 EVAL_GPU_ID="$GPU_ID" RUNTIME="$ER" OUTPUT_ROOT="$O" \
    bash "$TR/src/rl/experiments/run_full_dev_greedy_table_rl.sh"
fi
[[ "$(state_of "$EVAL_STATUS")" == complete ]]

SUMMARY=$ROOT/full_dev_summary.json
if [[ ! -f "$SUMMARY" ]]; then
  "$PY" src/rl/diagnostics/analyze_evaluation_results.py \
    --examples "$ER/data/eval_inputs/bird_dev_20240627.jsonl" \
    --arm "candidate=$RESULT/all.jsonl" \
    --arm "sft2=$ER/data/results/sft2_version36_dev1534_greedy_t0_logprobs20_20260801/all.jsonl" \
    --arm "exp14=$ER/data/results/stage1_exp14_fixed_process_rank_action_mean_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260802/all.jsonl" \
    --arm "exp15=$ER/data/results/stage1_exp15_fixed_prefix_action_dpo_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260802/all.jsonl" \
    --compare candidate:sft2 --compare candidate:exp14 --compare candidate:exp15 \
    --expected-count 1534 --protocol-version version36 \
    --temperature 0 --top-p 1 --denotation-comparison bird-set \
    --output "$SUMMARY"
fi
set_status complete "strict120+full_dev complete summary=$SUMMARY exploratory_single_run=1"
