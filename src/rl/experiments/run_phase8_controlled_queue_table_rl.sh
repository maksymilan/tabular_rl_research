#!/usr/bin/env bash
# Server-native Stage 1 queue: Exp12/13/14 controlled objectives plus Exp15 Action-DPO.
set -euo pipefail
O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TR=${TRAIN_RUNTIME:-$O/rl_runtime_rank_score_v3_20260731}
ER=${EVAL_RUNTIME:-$O/eval_runtime_version36_20260728}
RANK_POOL=${RANK_POOL_DIR:-$O/phase8_controlled_20260801/balanced_mixed60_rank_seed101}
PROCESS_POOL=${PROCESS_POOL_DIR:-$O/phase8_controlled_20260801/balanced_mixed60_process_seed101}
POOL_VALIDATION_STATUS=${POOL_VALIDATION_STATUS:-$O/logs/balanced_mixed60_views_table_rl_20260801.status}
ROOT=${PHASE_ROOT:-$O/phase8_controlled_20260801}
STATUS=${STATUS:-$O/logs/phase8_controlled_queue_table_rl_20260801.status}
RUN_LOG=${RUN_LOG:-$O/logs/phase8_controlled_queue_table_rl_20260801.log}
LOCK=${LOCK:-$O/logs/phase8_controlled_queue_table_rl_20260801.lock}
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
SFT2="$O/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682"
MODEL=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() {
  local state=$1 detail=$2
  printf '%s\t%s\t%s\n' "$(timestamp)" "$state" "$detail" >"$STATUS"
  "$PY" - "$ROOT/stage_status.json" "$state" "$detail" <<'PY'
import datetime,json,sys
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
old=json.load(open(p)) if p.exists() else {"schema_version":"phase8-controlled-stage-status-v1","history":[]}
event={"time":datetime.datetime.now(datetime.timezone.utc).isoformat(),"state":sys.argv[2],"detail":sys.argv[3]}
old["current"]=event; old["history"].append(event)
tmp=p.with_suffix(".json.next"); tmp.write_text(json.dumps(old,ensure_ascii=False,indent=2)+"\n"); tmp.replace(p)
PY
}
state_of() { awk -F '\t' 'NR==1 {print $2}' "$1" 2>/dev/null || printf missing; }
on_exit() { local c=$?; trap - EXIT; if [[ "$c" -ne 0 ]]; then set_status failed "exit=$c see=$RUN_LOG"; fi; exit "$c"; }
trap on_exit EXIT
mkdir -p "$O/logs" "$ROOT"
exec 8>"$LOCK"; if ! flock -n 8; then exit 0; fi
exec >>"$RUN_LOG" 2>&1

while [[ "$(state_of "$POOL_VALIDATION_STATUS")" != complete ]]; do
  state=$(state_of "$POOL_VALIDATION_STATUS")
  if [[ "$state" == failed || "$state" == blocked ]]; then set_status blocked "fixed-pool validation=$state"; exit 3; fi
  set_status waiting_fixed_pool "validation=$state"; sleep 60
done
while [[ "$(state_of "$O/logs/stage0_fixed_prefix_union_queue_table_rl_20260801.status")" != complete ]]; do
  state=$(state_of "$O/logs/stage0_fixed_prefix_union_queue_table_rl_20260801.status")
  if [[ "$state" == failed || "$state" == blocked ]]; then set_status blocked "fixed-prefix union=$state"; exit 3; fi
  set_status waiting_fixed_prefix "union=$state"; sleep 60
done

run_train() {
  local ordinal=$1 name=$2 config=$3 artifact=$4 gpu_id=$5
  local train_status="$O/logs/stage1_${name}.train.status"
  local pool="$PROCESS_POOL"
  [[ "$name" == exp13_fixed_rank_only_action_mean ]] && pool="$RANK_POOL"
  local output="$O/checkpoints/$artifact"
  while [[ -e "$output" && ! -f "$output/final/adapter_model.safetensors" ]]; do
    local active_state
    active_state=$(state_of "$train_status")
    if [[ "$active_state" == training || "$active_state" == waiting_gpu ]]; then
      set_status waiting_existing_train "candidate=$name state=$active_state"
      sleep 30
    else
      break
    fi
  done
  EXPERIMENT_NAME="$name" EXPERIMENT_CONFIG="$TR/src/rl/configs/experiments/$config" \
  ARTIFACT="$artifact" STATUS="$train_status" RUN_LOG="$O/logs/stage1_${name}.train.log" \
  TRAIN_RUNTIME="$TR" OUTPUT_ROOT="$O" POOL_DIR="$pool" GPU_ID="$gpu_id" \
    bash "$TR/src/rl/experiments/run_fixed_pool_controlled_train_table_rl.sh"
  [[ "$(state_of "$train_status")" == complete ]]
}

run_diagnostics() {
  local ordinal=$1 name=$2 artifact=$3 gpu_id=$4 port=$5
  local diag_status="$O/logs/stage1_${name}.diagnostics.status"
  CANDIDATE_NAME="$name" ADAPTER="$O/checkpoints/$artifact/final" \
  STATUS="$diag_status" RUN_LOG="$O/logs/stage1_${name}.diagnostics.log" \
  TRAIN_RUNTIME="$TR" EVAL_RUNTIME="$ER" OUTPUT_ROOT="$O" \
  DIAGNOSTIC_GPU_ID="$gpu_id" PORT="$port" \
    bash "$TR/src/rl/experiments/run_stage1_candidate_diagnostics_table_rl.sh"
  [[ "$(state_of "$diag_status")" == complete ]]
}

exp12_artifact=trl-transition-v26-exp12-fixed-process-only-sft2-60xk4-seed101-20260801
exp13_artifact=trl-transition-v26-exp13-fixed-rank-only-action-mean-sft2-60xk4-seed101-20260801
exp14_artifact=trl-transition-v26-exp14-fixed-process-rank-action-mean-sft2-60xk4-seed101-20260801

set_status parallel_training "Exp12 gpu0 + Exp13 gpu1; independent SFT2 initializations"
run_train 1 exp12_fixed_process_only exp12_fixed_process_only.yaml "$exp12_artifact" 0 &
pid_exp12=$!
run_train 2 exp13_fixed_rank_only_action_mean exp13_fixed_rank_only_action_mean.yaml "$exp13_artifact" 1 &
pid_exp13=$!
set +e
wait "$pid_exp12"; code_exp12=$?
wait "$pid_exp13"; code_exp13=$?
set -e
if [[ "$code_exp12" -ne 0 || "$code_exp13" -ne 0 ]]; then
  set_status failed "parallel train failed exp12=$code_exp12 exp13=$code_exp13"
  exit 5
fi

set_status train_and_diagnose "Exp14 train gpu0 + Exp12 diagnostics gpu1"
run_train 3 exp14_fixed_process_rank_action_mean exp14_fixed_process_rank_action_mean.yaml "$exp14_artifact" 0 &
pid_exp14=$!
run_diagnostics 1 exp12_fixed_process_only "$exp12_artifact" 1 18083 &
pid_diag12=$!
set +e
wait "$pid_exp14"; code_exp14=$?
wait "$pid_diag12"; code_diag12=$?
set -e
if [[ "$code_exp14" -ne 0 || "$code_diag12" -ne 0 ]]; then
  set_status failed "train/diagnostics failed exp14=$code_exp14 diag12=$code_diag12"
  exit 5
fi

set_status parallel_diagnostics "Exp13 gpu0 + Exp14 gpu1"
run_diagnostics 2 exp13_fixed_rank_only_action_mean "$exp13_artifact" 0 18082 &
pid_diag13=$!
run_diagnostics 3 exp14_fixed_process_rank_action_mean "$exp14_artifact" 1 18083 &
pid_diag14=$!
set +e
wait "$pid_diag13"; code_diag13=$?
wait "$pid_diag14"; code_diag14=$?
set -e
if [[ "$code_diag13" -ne 0 || "$code_diag14" -ne 0 ]]; then
  set_status failed "parallel diagnostics failed diag13=$code_diag13 diag14=$code_diag14"
  exit 5
fi

pair_dir="$ROOT/fixed_prefix_train_seed101"
mkdir -p "$pair_dir"
raw_pairs="$pair_dir/fixed_prefix_train.raw.jsonl"
verified_pairs="$pair_dir/fixed_prefix_train.verified.jsonl"
if [[ ! -f "$verified_pairs" ]]; then
  set_status building_action_dpo "extracting exact same-state train pairs"
  cd "$TR"
  "$PY" src/rl/action_dpo/extract_first_divergence_pairs.py \
    --pool "$RANK_POOL/validated_trajectories.jsonl" --pool-manifest "$RANK_POOL/manifest.json" \
    --output "$raw_pairs" --manifest "$pair_dir/extraction_manifest.json" \
    --max-pairs-per-question 8 --seed 101
  "$PY" src/rl/action_dpo/verify_corrected_actions.py \
    --pairs "$raw_pairs" --pool "$RANK_POOL/validated_trajectories.jsonl" \
    --output "$verified_pairs" --audit "$pair_dir/verification_audit.json"
  chmod 444 "$raw_pairs" "$verified_pairs" "$pair_dir/extraction_manifest.json" "$pair_dir/verification_audit.json"
fi

name=exp15_fixed_prefix_action_dpo
artifact=trl-transition-v26-exp15-fixed-prefix-action-dpo-sft2-seed101-20260801
output="$O/checkpoints/$artifact"
train_status="$O/logs/stage1_${name}.train.status"
if [[ ! -f "$output/final/adapter_model.safetensors" ]]; then
  if [[ -e "$output" ]]; then set_status failed "partial Exp15 output exists; audit required"; exit 4; fi
  while true; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '2p'); [[ "$used" -le 512 ]] && break; set_status waiting_gpu "candidate=$name mib=$used"; sleep 30; done
  set_status training "ordinal=4/4 candidate=$name"
  printf '%s\ttraining\tcandidate=%s\n' "$(timestamp)" "$name" >"$train_status"
  cd "$TR"
  CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 "$PY" src/rl/action_dpo/train_fixed_prefix_action_dpo.py \
    --dataset "$verified_pairs" --verification-audit "$pair_dir/verification_audit.json" \
    --model-path "$MODEL" --adapter-path "$SFT2" --output-dir "$output" \
    --beta 0.1 --learning-rate 1e-6 --epochs 1 --gradient-clip 1.0 \
    --max-length 8192 --seed 101
  printf '%s\tcomplete\tcandidate=%s\n' "$(timestamp)" "$name" >"$train_status"
fi
clip_retry=$("$PY" - "$output/run_manifest.json" <<'PY'
import json,sys
print(int(float(json.load(open(sys.argv[1]))["gradient_clip_fraction"]) > 0.20))
PY
)
if [[ "$clip_retry" == 1 ]]; then
  artifact=trl-transition-v26-exp15-fixed-prefix-action-dpo-sft2-seed101-lr5e7-20260801
  retry_output="$O/checkpoints/$artifact"
  if [[ ! -f "$retry_output/final/adapter_model.safetensors" ]]; then
    if [[ -e "$retry_output" ]]; then set_status failed "partial Exp15 lr5e-7 output exists; audit required"; exit 4; fi
    while true; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '2p'); [[ "$used" -le 512 ]] && break; set_status waiting_gpu "candidate=$name lr=5e-7 mib=$used"; sleep 30; done
    set_status training "ordinal=4/4 candidate=$name automatic_lr_followup=5e-7 beta=0.1"
    cd "$TR"
    CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 "$PY" src/rl/action_dpo/train_fixed_prefix_action_dpo.py \
      --dataset "$verified_pairs" --verification-audit "$pair_dir/verification_audit.json" \
      --model-path "$MODEL" --adapter-path "$SFT2" --output-dir "$retry_output" \
      --beta 0.1 --learning-rate 5e-7 --epochs 1 --gradient-clip 1.0 \
      --max-length 8192 --seed 101
  fi
  output="$retry_output"
fi
diag_status="$O/logs/stage1_${name}.diagnostics.status"
set_status diagnosing "ordinal=4/4 candidate=$name"
CANDIDATE_NAME="$name" ADAPTER="$output/final" STATUS="$diag_status" \
RUN_LOG="$O/logs/stage1_${name}.diagnostics.log" TRAIN_RUNTIME="$TR" \
EVAL_RUNTIME="$ER" OUTPUT_ROOT="$O" \
  bash "$TR/src/rl/experiments/run_stage1_candidate_diagnostics_table_rl.sh"
[[ "$(state_of "$diag_status")" == complete ]]

set_status stage1_complete "Exp12-Exp15 trained; cheap gates applied; eligible K4 complete"
