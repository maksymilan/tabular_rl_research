#!/usr/bin/env bash
# Exact, recoverable resume wrapper for the operator-requested early-stop arm.
# It never creates a new experiment arm.  It resumes only the single immutable
# run below, archives every uncommitted rollout suffix, and retries only a
# positively identified CUDA OOM after restarting both owned processes.
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_earlystop_mixed180_grpo_20260813}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_earlystop_mixed180_vanilla_grpo_20260813}
TRAIN_OUT=$RUN_ROOT/train180_two_pass_seed20260812
TASKS=$OUTPUT_ROOT/qwen3_8b_atomic_v26_boundary_screen_s1_earlystop_568_20260813/cohort/train180.jsonl
COHORT_MANIFEST=$OUTPUT_ROOT/qwen3_8b_atomic_v26_boundary_screen_s1_earlystop_568_20260813/cohort/earlystop_mixed180_manifest.json
CONFIG=$RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_vanilla_grpo_earlystop_mixed180.yaml
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
ADAPTER_PATH=${ADAPTER_PATH:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
PROTOCOL_RUNTIME=$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de
RESUME_HELPER=$RUNTIME/src/rl/diagnostics/prepare_vanilla_grpo_resume.py
TRAIN_GPU=${TRAIN_GPU:-0}
VLLM_GPU=${VLLM_GPU:-1}
VLLM_PORT=${VLLM_PORT:-8079}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51279}
MAX_OOM_ATTEMPTS=${MAX_OOM_ATTEMPTS:-3}
WAIT_FOR_GPU_SECONDS=${WAIT_FOR_GPU_SECONDS:-172800}

EXPECTED_TASKS_SHA256=a015c6513b850576ff95388238d45ad5cc0130da53823bf5d43869668592fb5d
EXPECTED_COHORT_MANIFEST_SHA256=e940c996d854a1756ec9787d375517125d9b7b35872b6ce9251232cd7526eaad
EXPECTED_CONFIG_SHA256=58a9096030d242a36069ea026f5f27d3f12823b5912f25785756f217a100c6f6
EXPECTED_IMPLEMENTATION_LOCK_SHA256=f3e52d1e5224955fe4ca436c2274cc205f61ae668c9687183c71b4a737b6ca68
EXPECTED_RESUME_HELPER_SHA256=0b1b254ec94dd92d33f8ba2a05b96f8f8e6dc64a365d46d8e0f0728d9b81f60c
EXPECTED_RUNNER_SHA256=2990f15e303b110845d222e76c9e5ae92edc8111b9942cd8c89b655688b28caa
EXPECTED_TRAIN_SHELL_SHA256=b2fa3a0b6277fcfb30d30f0f14b08a2a288c6f07435efbb6a1c4135864a97657
EXPECTED_VLLM_SHELL_SHA256=627f0a07f90d91702acb2f5379d2c111668a97ee90dbf37294d0ee2bd3324d57

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
require_sha() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "$path" && ! -L "$path" ]] || { printf 'missing/non-regular %s: %s\n' "$label" "$path" >&2; exit 3; }
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] || {
    printf '%s SHA mismatch expected=%s actual=%s\n' "$label" "$expected" "$actual" >&2
    exit 3
  }
}
atomic_status() {
  local state=$1 detail=$2 temporary=$RUN_ROOT/status.next
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" "$detail" >"$temporary"
  mv "$temporary" "$RUN_ROOT/status"
}
gpu_idle() {
  local gpu=$1
  [[ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]
}
wait_for_gpu() {
  local gpu=$1 waited=0
  while ! gpu_idle "$gpu"; do
    if (( waited >= WAIT_FOR_GPU_SECONDS )); then
      printf 'timed out waiting for GPU%s after %s seconds\n' "$gpu" "$waited" >&2
      return 75
    fi
    atomic_status waiting_for_gpu "gpu=$gpu waited_seconds=$waited owner=external_process"
    sleep 60
    waited=$((waited + 60))
  done
}
wait_for_gpu_pair() {
  local waited=0
  while true; do
    if gpu_idle "$TRAIN_GPU" && gpu_idle "$VLLM_GPU"; then
      # Close the obvious check/start race without ever reserving or signalling
      # an external process.  A new process in this short window makes vLLM or
      # trainer fail closed instead of being killed by this launcher.
      sleep 5
      if gpu_idle "$TRAIN_GPU" && gpu_idle "$VLLM_GPU"; then return 0; fi
    fi
    if (( waited >= WAIT_FOR_GPU_SECONDS )); then
      printf 'timed out waiting for idle GPU pair %s,%s after %s seconds\n' "$TRAIN_GPU" "$VLLM_GPU" "$waited" >&2
      return 75
    fi
    atomic_status waiting_for_gpu "gpus=$TRAIN_GPU,$VLLM_GPU waited_seconds=$waited owner=external_process"
    sleep 60
    waited=$((waited + 60))
  done
}

[[ "$MAX_OOM_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || { printf 'MAX_OOM_ATTEMPTS must be positive\n' >&2; exit 2; }
[[ "$WAIT_FOR_GPU_SECONDS" =~ ^[1-9][0-9]*$ ]] || { printf 'WAIT_FOR_GPU_SECONDS must be positive\n' >&2; exit 2; }
[[ "$TRAIN_GPU" != "$VLLM_GPU" ]] || { printf 'trainer and vLLM GPUs must differ\n' >&2; exit 2; }
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/resume_audit/discarded_rollouts"
exec 9>"$RUN_ROOT/launcher.lock"
flock -n 9 || { printf 'another launcher owns the run lock\n' >&2; exit 75; }
exec >>"$RUN_ROOT/logs/resume_supervisor.log" 2>&1

require_sha "$TASKS" "$EXPECTED_TASKS_SHA256" train180
require_sha "$COHORT_MANIFEST" "$EXPECTED_COHORT_MANIFEST_SHA256" cohort_manifest
require_sha "$CONFIG" "$EXPECTED_CONFIG_SHA256" experiment_config
require_sha "$TRAIN_OUT/implementation_lock.json" "$EXPECTED_IMPLEMENTATION_LOCK_SHA256" implementation_lock
require_sha "$RESUME_HELPER" "$EXPECTED_RESUME_HELPER_SHA256" resume_helper
require_sha "$RUNTIME/src/rl/frameworks/trl/run_transition_grpo.py" "$EXPECTED_RUNNER_SHA256" runner
require_sha "$RUNTIME/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" "$EXPECTED_TRAIN_SHELL_SHA256" train_shell
require_sha "$RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" "$EXPECTED_VLLM_SHELL_SHA256" vllm_shell
[[ -d "$PROTOCOL_RUNTIME" && ! -L "$PROTOCOL_RUNTIME" ]] || { printf 'invalid protocol runtime\n' >&2; exit 3; }
[[ -d "$TRAIN_OUT" && ! -L "$TRAIN_OUT" ]] || { printf 'invalid training output\n' >&2; exit 3; }
if [[ ! -f "$RUN_ROOT/status" ]] \
  || [[ "$(grep -Fc 'torch.OutOfMemoryError: CUDA out of memory' "$RUN_ROOT/logs/train.log")" -ne 1 ]] \
  || [[ "$(grep -Ec '^Traceback \(most recent call last\):' "$RUN_ROOT/logs/train.log")" -ne 1 ]]; then
  printf 'original failure is not the unique admitted CUDA OOM\n' >&2
  exit 3
fi
# Operator/supervisor restarts may update the mutable status marker, but the
# immutable original log remains the admitted failure evidence.  Accept only
# known local supervisor terminal/wait states in addition to the original
# launcher failure; never infer OOM from status alone.
if ! grep -Eq $'^[^[:space:]]+\t(failed\t(exit=1|resume_supervisor_exit=(3|143))|waiting_for_gpu\t.*)$' "$RUN_ROOT/status"; then
  printf 'unexpected mutable run status before resume: ' >&2
  cat "$RUN_ROOT/status" >&2
  exit 3
fi

# The implementation lock is the authoritative list of all Python/shell code
# imported by the trainer.  Verify both the deployed source and frozen snapshot
# before allowing a resume helper to mutate the rollout audit prefix.
env PYTHONPATH= "$PYTHON" - "$RUNTIME" "$TRAIN_OUT" <<'PY'
import hashlib, json, sys
from pathlib import Path
runtime, out = map(Path, sys.argv[1:])
lock = json.loads((out / "implementation_lock.json").read_text())
assert lock["schema_version"] == "trl-implementation-lock-v1"
assert lock["protocol_version"] == "version26"
assert lock["protocol_hash"] == "4da19387399bd3a5"
assert lock["student_prompt_sha256"] == "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
assert lock["experiment_config_sha256"] == "58a9096030d242a36069ea026f5f27d3f12823b5912f25785756f217a100c6f6"
assert lock["base_model_identity"]["aggregate_sha256"] == "85bd3b7d908acb3a9b9c7ec57b98d6b9e3b2fb427685ae808d1c43173279cecc"
files = lock["files"]
assert isinstance(files, dict) and len(files) == 19
for relative, expected in files.items():
    for path in (runtime / relative, out / "implementation_source_snapshot" / relative):
        assert path.is_file() and not path.is_symlink(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
manifest = json.loads((out / "run_manifest.json").read_text())
assert manifest["records"] == manifest["expected_records"] == 180
assert manifest["optimizer_steps"] == 12
assert manifest["prompts_per_update"] == 30
assert manifest["group_size"] == 8
assert manifest["save_steps"] == 2 and manifest["save_total_limit"] == 1
assert manifest["seed"] == 20260812
assert manifest["reward_mode"] == "result-only" and manifest["result_reward_profile"] == "binary"
assert manifest["policy_reduction"] == "trajectory_token_mean"
assert manifest["base_model_identity"] == lock["base_model_identity"]
assert manifest["implementation_source_sha256"] == files
PY

vllm_pgid=""
trainer_pgid=""
stop_group() {
  local pgid=${1:-}
  [[ -n "$pgid" ]] || return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 -- "-$pgid" 2>/dev/null || break; sleep 1; done
    kill -0 -- "-$pgid" 2>/dev/null && kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
  wait "$pgid" 2>/dev/null || true
}
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  stop_group "$trainer_pgid"
  stop_group "$vllm_pgid"
  if [[ "$code" -ne 0 ]]; then atomic_status failed "resume_supervisor_exit=$code"; fi
  exit "$code"
}
trap cleanup EXIT INT TERM

start_vllm() {
  local log=$1
  setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
    PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table \
    MODEL_PATH="$MODEL_PATH" VLLM_PORT="$VLLM_PORT" \
    VLLM_GPU_MEMORY_UTILIZATION=0.82 MAX_MODEL_LEN=16384 \
    bash "$RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" >>"$log" 2>&1 &
  vllm_pgid=$!
  local ready=0
  for _ in $(seq 1 240); do
    kill -0 "$vllm_pgid" 2>/dev/null || break
    if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 2
  done
  [[ "$ready" -eq 1 ]] || { printf 'vLLM readiness failed\n' >&2; return 1; }
}

for attempt in $(seq 1 "$MAX_OOM_ATTEMPTS"); do
  if [[ -d "$TRAIN_OUT/checkpoint-12" && -d "$TRAIN_OUT/final" ]]; then
    atomic_status complete "final=$TRAIN_OUT/final checkpoint=12"
    trap - EXIT INT TERM
    exit 0
  fi
  wait_for_gpu_pair
  if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
    printf 'refusing occupied vLLM port %s\n' "$VLLM_PORT" >&2
    exit 75
  fi

  rows=$(wc -l <"$TRAIN_OUT/rollouts.jsonl")
  pretrim_sha=$(sha256_file "$TRAIN_OUT/rollouts.jsonl")
  archive="$RUN_ROOT/resume_audit/discarded_rollouts/rows-${rows}.sha-${pretrim_sha}.jsonl"
  if [[ ! -e "$archive" ]]; then
    cp --preserve=mode,timestamps "$TRAIN_OUT/rollouts.jsonl" "$archive"
    chmod a-w "$archive"
  fi
  archive_sha=$(sha256_file "$archive")
  [[ "$archive_sha" == "$pretrim_sha" ]] || { printf 'rollout archive verification failed\n' >&2; exit 1; }
  printf '%s attempt=%s pretrim_rows=%s pretrim_sha256=%s archive=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$attempt" "$rows" "$archive_sha" "$archive" \
    >>"$RUN_ROOT/resume_audit/archive_receipts.log"

  prep="$RUN_ROOT/resume_audit/resume_preparation_attempt_${attempt}.json"
  "$PYTHON" "$RESUME_HELPER" \
    --train-output "$TRAIN_OUT" --tasks "$TASKS" --expected-records 180 \
    --optimizer-steps 12 --prompts-per-update 30 --group-size 8 --save-steps 2 \
    --output "$prep"
  checkpoint=$(
    env PYTHONPATH= "$PYTHON" - "$prep" "$TRAIN_OUT" <<'PY'
import json, sys
from pathlib import Path
p = json.load(open(sys.argv[1]))
root = Path(sys.argv[2]).resolve()
assert p["schema_version"] == "vanilla-grpo-resume-preparation-v1"
assert p["status"] == "resume_ready"
assert p["contract"] == {"optimizer_steps": 12, "prompts_per_update": 30, "group_size": 8, "save_steps": 2, "committed_rollouts_per_step": 240}
checkpoint = Path(p["checkpoint"]["path"]).resolve()
assert checkpoint.parent == root and checkpoint.name == f"checkpoint-{p['checkpoint']['global_step']}"
assert 0 < p["checkpoint"]["global_step"] < 12
print(checkpoint)
PY
  )

  train_log="$RUN_ROOT/logs/resume_attempt_${attempt}.train.log"
  vllm_log="$RUN_ROOT/logs/resume_attempt_${attempt}.vllm.log"
  atomic_status starting_vllm "resume_attempt=$attempt checkpoint=$checkpoint"
  start_vllm "$vllm_log"

  atomic_status training "resume_attempt=$attempt checkpoint=$checkpoint alloc=expandable_segments"
  setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PROJECT_DIR="$RUNTIME" PYTHON="$PYTHON" \
    MODEL_PATH="$MODEL_PATH" ADAPTER_PATH="$ADAPTER_PATH" EXAMPLES_JSON="$TASKS" \
    OUTPUT_DIR="$TRAIN_OUT" EXPERIMENT_CONFIG="$CONFIG" VLLM_PORT="$VLLM_PORT" \
    VLLM_GROUP_PORT="$VLLM_GROUP_PORT" PYTHONPATH= \
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF" \
    bash "$RUNTIME/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" \
      --optimizer-steps 12 --ppo-iterations 1 --prompts-per-update 30 --group-size 8 \
      --kl-beta 0 --protocol-runtime-root "$PROTOCOL_RUNTIME" \
      --transition-micro-batch-size 1 --save-steps 2 --save-total-limit 1 --seed 20260812 \
      --resume-from-checkpoint "$checkpoint" >>"$train_log" 2>&1 &
  trainer_pgid=$!
  code=0
  wait "$trainer_pgid" || code=$?
  trainer_pgid=""
  stop_group "$vllm_pgid"
  vllm_pgid=""
  if [[ "$code" -eq 0 ]]; then
    [[ -d "$TRAIN_OUT/checkpoint-12" && -d "$TRAIN_OUT/final" ]] || {
      printf 'trainer exited zero without checkpoint-12/final\n' >&2
      exit 1
    }
    env PYTHONPATH= "$PYTHON" - "$TRAIN_OUT" "$TASKS" <<'PY'
import hashlib, json, sys
from collections import Counter
from pathlib import Path
out, tasks = map(Path, sys.argv[1:])
state = json.loads((out / "checkpoint-12" / "trainer_state.json").read_text())
assert state["global_step"] == state["max_steps"] == 12
rows = [json.loads(line) for line in (out / "rollouts.jsonl").read_text().splitlines() if line.strip()]
assert len(rows) == 2880
task_rows = [json.loads(line) for line in tasks.read_text().splitlines() if line.strip()]
ids = [row["example_index"] for row in task_rows]
assert len(ids) == len(set(ids)) == 180
for step in range(12):
    block = rows[step * 240:(step + 1) * 240]
    assert {row["policy_global_step"] for row in block} == {step}
    counts = Counter(row["example_index"] for row in block)
    assert len(counts) == 30 and set(counts.values()) == {8}
for pass_index in range(2):
    counts = Counter(row["example_index"] for row in rows[pass_index * 1440:(pass_index + 1) * 1440])
    assert counts == Counter({identity: 8 for identity in ids})
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(out / "checkpoint-12" / "adapter_model.safetensors") == sha(out / "final" / "adapter_model.safetensors")
assert (out / "training_precision.json").is_file()
PY
    atomic_status complete "final=$TRAIN_OUT/final checkpoint=12 attempts=$attempt"
    trap - EXIT INT TERM
    exit 0
  fi
  if ! grep -Fq 'torch.OutOfMemoryError: CUDA out of memory' "$train_log"; then
    printf 'resume attempt %s failed with non-OOM exit=%s\n' "$attempt" "$code" >&2
    exit "$code"
  fi
  printf 'resume attempt %s hit a confirmed CUDA OOM; restarting from latest durable checkpoint\n' "$attempt" >&2
  atomic_status retrying_oom "attempt=$attempt exit=$code"
done

printf 'exhausted %s confirmed-OOM resume attempts\n' "$MAX_OOM_ATTEMPTS" >&2
exit 1
