#!/usr/bin/env bash
# Independent Exp15 rerun: ignore authored historical reasoning when matching prefixes.
set -euo pipefail

O=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TR=${TRAIN_RUNTIME:-$O/rl_runtime_rank_score_v3_20260731}
ER=${EVAL_RUNTIME:-$O/eval_runtime_version36_20260728}
PY=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct
SFT2="$O/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682"
EXPECTED_SFT2_SHA=d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e
POOL="$O/phase8_controlled_20260801/balanced_mixed60_rank_seed101"
PAIR_DIR="$O/phase8_controlled_20260801/reasoning_normalized_action_dpo_seed101"
NAME=exp15_reasoning_normalized_action_dpo
ARTIFACT=trl-transition-v26-exp15-reasoning-normalized-action-dpo-sft2-seed101-20260802
OUTPUT="$O/checkpoints/$ARTIFACT"
STATUS=${STATUS:-$O/logs/stage1_${NAME}.queue.status}
TRAIN_STATUS="$O/logs/stage1_${NAME}.train.status"
DIAG_STATUS="$O/logs/stage1_${NAME}.diagnostics.status"
RUN_LOG=${RUN_LOG:-$O/logs/stage1_${NAME}.queue.log}
LOCK=${LOCK:-$O/logs/stage1_${NAME}.queue.lock}
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-18082}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
state_of() { awk -F '\t' 'NR==1 {print $2}' "$1" 2>/dev/null || printf missing; }
on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -ne 0 ]]; then
    set_status failed "exit=$code see=$RUN_LOG"
  fi
  exit "$code"
}
trap on_exit EXIT

mkdir -p "$O/logs" "$PAIR_DIR"
exec 8>"$LOCK"
if ! flock -n 8; then exit 0; fi
exec >>"$RUN_LOG" 2>&1

actual_sft2_sha=$(sha256sum "$SFT2/adapter_model.safetensors" | awk '{print $1}')
if [[ "$actual_sft2_sha" != "$EXPECTED_SFT2_SHA" ]]; then
  set_status failed "SFT2 adapter hash mismatch actual=$actual_sft2_sha"
  exit 4
fi

raw_pairs="$PAIR_DIR/reasoning_normalized.raw.jsonl"
verified_pairs="$PAIR_DIR/reasoning_normalized.verified.jsonl"
extraction_manifest="$PAIR_DIR/extraction_manifest.json"
verification_audit="$PAIR_DIR/verification_audit.json"
pair_files=("$raw_pairs" "$verified_pairs" "$extraction_manifest" "$verification_audit")
complete_pair_files=0
for path in "${pair_files[@]}"; do
  if [[ -s "$path" ]]; then complete_pair_files=$((complete_pair_files + 1)); fi
done
if [[ "$complete_pair_files" -ne 0 && "$complete_pair_files" -ne 4 ]]; then
  set_status failed "partial reasoning-normalized pair artifacts require audit"
  exit 4
fi
if [[ "$complete_pair_files" -eq 0 ]]; then
  set_status extracting "normalization=assistant-think-placeholder-v1"
  cd "$TR"
  "$PY" src/rl/action_dpo/extract_first_divergence_pairs.py \
    --pool "$POOL/validated_trajectories.jsonl" \
    --pool-manifest "$POOL/manifest.json" \
    --output "$raw_pairs" --manifest "$extraction_manifest" \
    --max-pairs-per-question 8 --seed 101 \
    --normalize-history-reasoning
  set_status verifying "positive source suffix replay"
  "$PY" src/rl/action_dpo/verify_corrected_actions.py \
    --pairs "$raw_pairs" --pool "$POOL/validated_trajectories.jsonl" \
    --output "$verified_pairs" --audit "$verification_audit"
  chmod 444 "${pair_files[@]}"
fi

normalization=$(
  "$PY" - "$extraction_manifest" "$verification_audit" <<'PY'
import json,sys
extraction=json.load(open(sys.argv[1]))
audit=json.load(open(sys.argv[2]))
assert extraction["state_normalization"] == "assistant-think-placeholder-v1"
assert extraction["authored_reasoning_used_for_state_matching"] is False
assert audit["status"] == "passed"
assert audit["state_normalization_modes"] == ["assistant-think-placeholder-v1"]
assert audit["rejected_pairs"] == 0
print(extraction["state_normalization"])
PY
)
set_status pairs_ready "normalization=$normalization pairs=$(wc -l < "$verified_pairs")"

if [[ ! -f "$OUTPUT/final/adapter_model.safetensors" ]]; then
  if [[ -e "$OUTPUT" ]]; then
    set_status failed "partial training output exists; audit required output=$OUTPUT"
    exit 4
  fi
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | sed -n "$((GPU_ID + 1))p")
    [[ -n "$used" && "$used" -le 512 ]] && break
    set_status waiting_gpu "gpu=$GPU_ID mib=${used:-unknown}"
    sleep 30
  done
  set_status training "candidate=$NAME gpu=$GPU_ID beta=0.1 lr=1e-6"
  printf '%s\ttraining\tcandidate=%s gpu=%s\n' "$(timestamp)" "$NAME" "$GPU_ID" >"$TRAIN_STATUS"
  cd "$TR"
  CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 "$PY" \
    src/rl/action_dpo/train_fixed_prefix_action_dpo.py \
    --dataset "$verified_pairs" --verification-audit "$verification_audit" \
    --model-path "$MODEL" --adapter-path "$SFT2" --output-dir "$OUTPUT" \
    --experiment-name "$NAME" --beta 0.1 --learning-rate 1e-6 --epochs 1 \
    --gradient-clip 1.0 --max-length 8192 --seed 101
  printf '%s\tcomplete\tcandidate=%s gpu=%s\n' "$(timestamp)" "$NAME" "$GPU_ID" >"$TRAIN_STATUS"
fi

clip_retry=$(
  "$PY" - "$OUTPUT/run_manifest.json" <<'PY'
import json,sys
print(int(float(json.load(open(sys.argv[1]))["gradient_clip_fraction"]) > 0.20))
PY
)
if [[ "$clip_retry" == 1 ]]; then
  RETRY_ARTIFACT=${ARTIFACT}-lr5e7
  RETRY_OUTPUT="$O/checkpoints/$RETRY_ARTIFACT"
  if [[ ! -f "$RETRY_OUTPUT/final/adapter_model.safetensors" ]]; then
    if [[ -e "$RETRY_OUTPUT" ]]; then
      set_status failed "partial lr5e-7 output exists; audit required output=$RETRY_OUTPUT"
      exit 4
    fi
    set_status training "candidate=$NAME gpu=$GPU_ID beta=0.1 lr=5e-7 clip_followup=1"
    cd "$TR"
    CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 "$PY" \
      src/rl/action_dpo/train_fixed_prefix_action_dpo.py \
      --dataset "$verified_pairs" --verification-audit "$verification_audit" \
      --model-path "$MODEL" --adapter-path "$SFT2" --output-dir "$RETRY_OUTPUT" \
      --experiment-name "${NAME}_lr5e7" --beta 0.1 --learning-rate 5e-7 \
      --epochs 1 --gradient-clip 1.0 --max-length 8192 --seed 101
  fi
  OUTPUT="$RETRY_OUTPUT"
fi

set_status diagnostics "candidate=$NAME fixed_prefix_auxiliary+full_dev1534_greedy"
CANDIDATE_NAME="$NAME" ADAPTER="$OUTPUT/final" STATUS="$DIAG_STATUS" \
RUN_LOG="$O/logs/stage1_${NAME}.diagnostics.log" TRAIN_RUNTIME="$TR" \
EVAL_RUNTIME="$ER" OUTPUT_ROOT="$O" DIAGNOSTIC_GPU_ID="$GPU_ID" PORT="$PORT" \
  bash "$TR/src/rl/experiments/run_stage1_candidate_diagnostics_table_rl.sh"
if [[ "$(state_of "$DIAG_STATUS")" != complete ]]; then
  set_status failed "diagnostics did not complete state=$(state_of "$DIAG_STATUS")"
  exit 5
fi
set_status complete "candidate=$NAME full_dev1534_greedy=complete"
