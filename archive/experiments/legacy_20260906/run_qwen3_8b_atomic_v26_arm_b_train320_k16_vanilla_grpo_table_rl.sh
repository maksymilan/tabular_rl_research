#!/usr/bin/env bash
# Fail-closed preregistered Arm B training launcher.  Validation64 is produced
# by a separate frozen pipeline; this launcher only verifies its explicit audit.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODE=${1:-dry-run}
if [[ $# -gt 1 ]]; then
  printf 'usage: %s [dry-run|preflight|train]\n' "$0" >&2
  exit 2
fi
case "$MODE" in
  dry-run|preflight|train) ;;
  *) printf 'usage: %s [dry-run|preflight|train]\n' "$0" >&2; exit 2 ;;
esac

if [[ "$MODE" == dry-run ]]; then
  printf '%s\n' \
    'Arm B vanilla-GRPO dry-run: no files written, no GPU inspected' \
    'preflight: bind frozen Arm-B selection manifest, train320 and admitted validation64 audit' \
    'train: train320, K=16, 32 x 20 prompts, exactly two passes, LR=8e-7, beta=0, binary reward' \
    'checkpoint policy: save every 4 steps for exact resume; steps 4..28 are diagnostic/resume-only and forbidden for evaluation; final step32 is the sole primary' \
    'required active bindings: selection/train320/F3 audit/source cohort/confirmatory trigger explicit SHA-256 values'
  exit 0
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TRAIN_RUNTIME=${TRAIN_RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_arm_b_grpo_20260812}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
SFT1_ADAPTER=${SFT1_ADAPTER:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}

SCREEN_RUN_ROOT=${SCREEN_RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_vanilla_grpo_arm_b_screen_20260812}
ARM_B_SELECTION_DIR=${ARM_B_SELECTION_DIR:-$SCREEN_RUN_ROOT/f2_confirmation640_k16_seed20260815/selection}
ARM_B_SELECTION_MANIFEST=${ARM_B_SELECTION_MANIFEST:-$ARM_B_SELECTION_DIR/arm_b_selection_manifest.json}
TRAIN320_TASKS=${TRAIN320_TASKS:-$ARM_B_SELECTION_DIR/train320.jsonl}
VALIDATION64_AUDIT=${VALIDATION64_AUDIT:-$SCREEN_RUN_ROOT/f3_validation64_k16_seed20260816/audit/arm_b_k16_validation_audit.json}
SOURCE_COHORT_MANIFEST=${SOURCE_COHORT_MANIFEST:-$SCREEN_RUN_ROOT/inputs/wide3000.manifest.json}
CONFIRMATORY_TRIGGER=${CONFIRMATORY_TRIGGER:-$SCREEN_RUN_ROOT/f3_validation64_k16_seed20260816/audit/confirmatory_arm_trigger.json}
FROZEN_ARM_B_SELECTION_MANIFEST_SHA256=${FROZEN_ARM_B_SELECTION_MANIFEST_SHA256:-}
FROZEN_TRAIN320_SHA256=${FROZEN_TRAIN320_SHA256:-}
FROZEN_VALIDATION64_AUDIT_SHA256=${FROZEN_VALIDATION64_AUDIT_SHA256:-}
FROZEN_SOURCE_COHORT_MANIFEST_SHA256=${FROZEN_SOURCE_COHORT_MANIFEST_SHA256:-}
FROZEN_CONFIRMATORY_TRIGGER_SHA256=${FROZEN_CONFIRMATORY_TRIGGER_SHA256:-}

EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-$TRAIN_RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_vanilla_grpo_arm_b_train320_k16.yaml}
INPUT_VALIDATOR=${INPUT_VALIDATOR:-$TRAIN_RUNTIME/src/rl/diagnostics/validate_arm_b_training_inputs.py}
ARM_B_SHARED_CONTRACT=${ARM_B_SHARED_CONTRACT:-$TRAIN_RUNTIME/src/rl/vanilla_grpo_arm_b.py}
TRIGGER_PREPARER=${TRIGGER_PREPARER:-$TRAIN_RUNTIME/src/rl/diagnostics/prepare_vanilla_grpo_confirmatory_arm_trigger.py}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_arm_b_train320_k16_vanilla_grpo_20260812}
TRAIN_OUT=${TRAIN_OUT:-$RUN_ROOT/train320_two_pass_k16_seed20260812}
TRAINING_CONTRACT=$TRAIN_OUT/arm_b_training_contract.json
STATUS=$RUN_ROOT/status/${MODE}.status
RUN_LOG=$RUN_ROOT/logs/${MODE}.log
TRAIN_LOG=$RUN_ROOT/logs/train_worker.log
VLLM_LOG=$RUN_ROOT/logs/train_vllm.log
LOCK=$RUN_ROOT/launcher.lock

TRAIN_GPU=${TRAIN_GPU:-0}
VLLM_GPU=${VLLM_GPU:-1}
VLLM_PORT=${VLLM_PORT:-8078}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51278}
TRAIN_SEED=20260812

EXPECTED_PROTOCOL_VERSION=version26
EXPECTED_PROTOCOL_HASH=4da19387399bd3a5
EXPECTED_STUDENT_PROMPT_SHA256=848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316
EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256=5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab
EXPECTED_SFT1_SHA256=3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5
EXPECTED_SFT1_CONFIG_SHA256=537dbc946a7131d59e294a8729ace4aa145d73e59f4cf8556360ea507ce4435a
EXPECTED_SFT1_STATE_SHA256=97e529475d8c4f68dc88bc1f80370dacab478a681629f5f99003c39c70e9600a
EXPECTED_CONFIG_SHA256=d2fb14b029b667928664e6b7d3dfd614d060862dd67cc6ba280712fced3fd2dc
EXPECTED_INPUT_VALIDATOR_SHA256=a416b9490e8c6287c7f05fff5c2c314c528b8524dddbd782c5e25ed6eb7125e0
EXPECTED_ARM_B_SHARED_CONTRACT_SHA256=d596a01a89a125ec0375af7d3a6dbdd2c1b5a65a2b625dc8d07cb502c29e5a80
EXPECTED_TRIGGER_PREPARER_SHA256=554fe25b39fe7c012a73b8d4045762476347bfd6fbcc70355f98f4cb8bdd7b98
EXPECTED_RESUME_PREPARER_SHA256=0b1b254ec94dd92d33f8ba2a05b96f8f8e6dc64a365d46d8e0f0728d9b81f60c
EXPECTED_RUN_TRANSITION_GRPO_SHA256=2990f15e303b110845d222e76c9e5ae92edc8111b9942cd8c89b655688b28caa

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
terminal_status_written=0
set_status() {
  local state=$1 detail=$2 temporary=$STATUS.next
  printf '%s\t%s\t%s\n' "$(timestamp)" "$state" "$detail" >"$temporary"
  mv "$temporary" "$STATUS"
}
set_terminal_status() { set_status "$1" "$2"; terminal_status_written=1; }
blocked() {
  set_terminal_status blocked "$1"
  printf 'blocked: %s\n' "$1" >&2
  exit 3
}
require_sha() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "$path" && ! -L "$path" ]] || blocked "missing/non-regular $label: $path"
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] \
    || blocked "$label SHA-256 mismatch: expected=$expected actual=$actual path=$path"
}

for binding_name in \
  FROZEN_ARM_B_SELECTION_MANIFEST_SHA256 \
  FROZEN_TRAIN320_SHA256 \
  FROZEN_VALIDATION64_AUDIT_SHA256 \
  FROZEN_SOURCE_COHORT_MANIFEST_SHA256 \
  FROZEN_CONFIRMATORY_TRIGGER_SHA256
do
  binding_value=${!binding_name}
  [[ "$binding_value" =~ ^[0-9a-f]{64}$ ]] \
    || { printf '%s must be an explicit lowercase SHA-256\n' "$binding_name" >&2; exit 3; }
done
for path in "$ARM_B_SELECTION_MANIFEST" "$TRAIN320_TASKS" "$VALIDATION64_AUDIT" \
  "$SOURCE_COHORT_MANIFEST" "$CONFIRMATORY_TRIGGER"; do
  [[ "$path" == /* ]] || { printf 'Arm B input path must be absolute: %s\n' "$path" >&2; exit 3; }
done
[[ "$TRAIN_RUNTIME" == "$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_arm_b_grpo_20260812" ]] \
  || { printf 'TRAIN_RUNTIME must be the dedicated Arm B overlay\n' >&2; exit 3; }
case "$TRAIN_OUT" in "$RUN_ROOT"/*) ;; *) printf 'TRAIN_OUT is outside RUN_ROOT\n' >&2; exit 3 ;; esac
[[ "$TRAIN_GPU" =~ ^[0-9]+$ && "$VLLM_GPU" =~ ^[0-9]+$ && "$TRAIN_GPU" != "$VLLM_GPU" ]] \
  || { printf 'trainer and vLLM require distinct nonnegative GPU ids\n' >&2; exit 3; }

mkdir -p "$RUN_ROOT/status" "$RUN_ROOT/logs"
exec 9>"$LOCK"
if ! flock -n 9; then
  printf 'another Arm B launcher owns %s\n' "$LOCK" >&2
  exit 75
fi
exec >>"$RUN_LOG" 2>&1

worker_pgid=""
vllm_pgid=""
terminate_session() {
  local pgid=${1:-}
  [[ -n "$pgid" ]] || return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in {1..30}; do kill -0 -- "-$pgid" 2>/dev/null || break; sleep 1; done
    kill -0 -- "-$pgid" 2>/dev/null && kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
  wait "$pgid" 2>/dev/null || true
}
cleanup_owned_sessions() {
  terminate_session "$worker_pgid"; worker_pgid=""
  terminate_session "$vllm_pgid"; vllm_pgid=""
}
on_exit() {
  local code=$?
  trap - EXIT INT TERM
  cleanup_owned_sessions
  if [[ "$code" -ne 0 && "$terminal_status_written" -eq 0 ]]; then
    set_status failed "mode=$MODE exit=$code log=$RUN_LOG"
  fi
  exit "$code"
}
on_signal() { set_status interrupted "mode=$MODE signal=$1"; terminal_status_written=1; exit "$2"; }
trap on_exit EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
set_status validating "mode=$MODE arm=arm_b"

verify_arm_b_inputs() {
  env PYTHONPATH="$TRAIN_RUNTIME" "$PYTHON_BIN" "$INPUT_VALIDATOR" \
    --selection-manifest "$ARM_B_SELECTION_MANIFEST" \
    --expected-selection-manifest-sha256 "$FROZEN_ARM_B_SELECTION_MANIFEST_SHA256" \
    --train320 "$TRAIN320_TASKS" \
    --expected-train320-sha256 "$FROZEN_TRAIN320_SHA256" \
    --validation64-audit "$VALIDATION64_AUDIT" \
    --expected-validation64-audit-sha256 "$FROZEN_VALIDATION64_AUDIT_SHA256"
  env PYTHONPATH="$TRAIN_RUNTIME" "$PYTHON_BIN" "$TRIGGER_PREPARER" \
    --source-cohort "$SOURCE_COHORT_MANIFEST" \
    --expected-source-cohort-sha256 "$FROZEN_SOURCE_COHORT_MANIFEST_SHA256" \
    --selection-manifest "$ARM_B_SELECTION_MANIFEST" \
    --expected-selection-manifest-sha256 "$FROZEN_ARM_B_SELECTION_MANIFEST_SHA256" \
    --train320 "$TRAIN320_TASKS" \
    --expected-train320-sha256 "$FROZEN_TRAIN320_SHA256" \
    --validation64-audit "$VALIDATION64_AUDIT" \
    --expected-validation64-audit-sha256 "$FROZEN_VALIDATION64_AUDIT_SHA256" \
    --output "$CONFIRMATORY_TRIGGER" --verify-existing
}

validate_static_inputs() {
  for forbidden in src/eval src/sft src/harness src/tool_modules; do
    [[ ! -e "$TRAIN_RUNTIME/$forbidden" ]] \
      || blocked "Arm B overlay shadows frozen protocol source: $TRAIN_RUNTIME/$forbidden"
  done
  [[ -x "$PYTHON_BIN" ]] || blocked "missing Python runtime: $PYTHON_BIN"
  [[ -d "$MODEL_PATH" && -d "$SFT1_ADAPTER" ]] || blocked 'missing model or SFT1 adapter'
  [[ -d "$PROTOCOL_RUNTIME/src/eval" && -d "$PROTOCOL_RUNTIME/src/sft" && -d "$PROTOCOL_RUNTIME/src/harness" ]] \
    || blocked "incomplete protocol runtime: $PROTOCOL_RUNTIME"
  require_sha "$MODEL_PATH/model-00001-of-00005.safetensors" 31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f model_shard_1
  require_sha "$MODEL_PATH/model-00002-of-00005.safetensors" 5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282 model_shard_2
  require_sha "$MODEL_PATH/model-00003-of-00005.safetensors" c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836 model_shard_3
  require_sha "$MODEL_PATH/model-00004-of-00005.safetensors" b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a model_shard_4
  require_sha "$MODEL_PATH/model-00005-of-00005.safetensors" 20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff model_shard_5
  require_sha "$MODEL_PATH/config.json" f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30 model_config
  require_sha "$MODEL_PATH/generation_config.json" 2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2 generation_config
  require_sha "$MODEL_PATH/merges.txt" 8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5 tokenizer_merges
  require_sha "$MODEL_PATH/model.safetensors.index.json" f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc model_index
  require_sha "$MODEL_PATH/tokenizer.json" aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 tokenizer
  require_sha "$MODEL_PATH/tokenizer_config.json" d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101 tokenizer_config
  require_sha "$MODEL_PATH/vocab.json" ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 tokenizer_vocab
  require_sha "$SFT1_ADAPTER/adapter_model.safetensors" "$EXPECTED_SFT1_SHA256" sft1_adapter
  require_sha "$SFT1_ADAPTER/adapter_config.json" "$EXPECTED_SFT1_CONFIG_SHA256" sft1_adapter_config
  require_sha "$SFT1_ADAPTER/trainer_state.json" "$EXPECTED_SFT1_STATE_SHA256" sft1_trainer_state
  require_sha "$EXPERIMENT_CONFIG" "$EXPECTED_CONFIG_SHA256" arm_b_config
  require_sha "$INPUT_VALIDATOR" "$EXPECTED_INPUT_VALIDATOR_SHA256" arm_b_input_validator
  require_sha "$ARM_B_SHARED_CONTRACT" "$EXPECTED_ARM_B_SHARED_CONTRACT_SHA256" arm_b_shared_contract
  require_sha "$TRIGGER_PREPARER" "$EXPECTED_TRIGGER_PREPARER_SHA256" confirmatory_trigger_preparer
  require_sha "$TRAIN_RUNTIME/src/rl/diagnostics/prepare_vanilla_grpo_resume.py" "$EXPECTED_RESUME_PREPARER_SHA256" resume_preparer
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/run_transition_grpo.py" "$EXPECTED_RUN_TRANSITION_GRPO_SHA256" run_transition_grpo
  require_sha "$TRAIN_RUNTIME/src/rl/experiment_config.py" 6bd4b836984a4cbf3156170c230464bfaa113a645d21df31f8c3e54416f05ccf experiment_config
  require_sha "$TRAIN_RUNTIME/src/rl/task_loader.py" d79d41ea5f32ddfa45bb7c1496e8de234f96a6b48d84dfa2363d0de37a024d4b task_loader
  require_sha "$TRAIN_RUNTIME/src/rl/rollout_scoring.py" a11233e7a04a9efa34393e7d77a4c4aca34151144bf531978235ef6d644c1ac5 rollout_scoring
  require_sha "$TRAIN_RUNTIME/src/rl/terminal_reward.py" 2b45d45f562a903fe6e83589bc3aa9b9cc8d22036be6d0544c562ced82757b7b terminal_reward
  require_sha "$TRAIN_RUNTIME/src/rl/tool_environment.py" fcd7e77c668623da7027ddd5b35c3e945f1f39ba93f8eb50adf742551d68e6a1 current_tool_environment
  require_sha "$TRAIN_RUNTIME/src/rl/tool_environment_v26.py" c7a84bdb2d259eea91cde5a77a5758ac8ea57e02828330088ecbd903b1d0ec5c tool_environment_v26
  require_sha "$TRAIN_RUNTIME/src/rl/counterfactual_suite.py" 13e88a98cb8b9d0cd350a1b963315b80c5fe951f8ac668d5b8d4140b5d00df92 counterfactual_suite
  require_sha "$TRAIN_RUNTIME/src/rl/reference_result_filter.py" 2d5b1e61181facfa2883f5dd926358e9f75d8eb0abe12dab3204628f5631dc92 reference_result_filter
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/fixed_rollout_pool.py" e0be4c4c830ca2e87172175553ab47d1b19561510f8fa6f77410e96846cf200d fixed_rollout_pool
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/transition_grpo.py" 1ef68a63b9857a43aa399710d2e58e84aadb4e0106e917237333471501be3b37 transition_grpo
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/transition_batch.py" c6477393589581209ad581230eb598160e0674469103ce197b7d7d7f26a0bcd8 transition_batch
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/rollout.py" 87c36224ba0f01954db0d58db3bc3e5685bc5c86736c0679299c76872b6c821f rollout
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/tool_loss_mask.py" b15072423600c663d91651e640f7738087bdc1ea551758704f98cc6882d14abb tool_loss_mask
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/training_precision.py" f5dd61eb2bc307ac422243a6d2be14e99382797aa3acd76396c0d553e76b215f training_precision
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/trajectory_ranking.py" 2097282799b321b32236f32686ea25d1d078139bae7fecc5f4be58574ddb9eb7 trajectory_ranking
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" b2fa3a0b6277fcfb30d30f0f14b08a2a288c6f07435efbb6a1c4135864a97657 trainer_shell
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" 627f0a07f90d91702acb2f5379d2c111668a97ee90dbf37294d0ee2bd3324d57 vllm_shell
  require_sha "$ARM_B_SELECTION_MANIFEST" "$FROZEN_ARM_B_SELECTION_MANIFEST_SHA256" arm_b_selection_manifest
  require_sha "$TRAIN320_TASKS" "$FROZEN_TRAIN320_SHA256" train320
  require_sha "$VALIDATION64_AUDIT" "$FROZEN_VALIDATION64_AUDIT_SHA256" validation64_audit
  require_sha "$SOURCE_COHORT_MANIFEST" "$FROZEN_SOURCE_COHORT_MANIFEST_SHA256" source_cohort_manifest
  require_sha "$CONFIRMATORY_TRIGGER" "$FROZEN_CONFIRMATORY_TRIGGER_SHA256" confirmatory_trigger
  verify_arm_b_inputs >/dev/null || blocked 'Arm B selection/train/validation admission failed'

  env PYTHONPATH= "$PYTHON_BIN" - "$TRAIN_RUNTIME" "$PROTOCOL_RUNTIME" \
    "$EXPERIMENT_CONFIG" "$EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" \
    "$EXPECTED_PROTOCOL_VERSION" "$EXPECTED_PROTOCOL_HASH" \
    "$EXPECTED_STUDENT_PROMPT_SHA256" "$EXPECTED_SFT1_SHA256" <<'PY'
import hashlib, importlib, importlib.metadata, inspect, sys
from pathlib import Path
overlay, runtime, config_path = map(Path, sys.argv[1:4])
expected_tree, expected_version, expected_hash, expected_prompt, expected_adapter = sys.argv[4:9]
sys.path[:0] = [str(overlay / "src/rl"), str(runtime / "src/eval"), str(runtime / "src/harness"), str(runtime / "src/sft")]
protocol = importlib.import_module("protocol")
prompt = protocol.student_runtime_system_prompt(context_mode="rolling-legal-history", compact=False)
assert protocol.PROTOCOL_VERSION == expected_version
assert protocol.protocol_hash(prompt) == expected_hash
assert hashlib.sha256(prompt.encode()).hexdigest() == expected_prompt
from rl.configuration.experiment_config import RLExperimentConfig, runtime_content_tree_sha256
defaults = RLExperimentConfig.load(config_path).argparse_defaults(overlay)
required = {
    "reward_mode": "result-only", "result_reward_profile": "binary",
    "policy_reduction": "trajectory_token_mean", "expected_records": 320,
    "optimizer_steps": 32, "prompts_per_update": 20, "group_size": 16,
    "ppo_iterations": 1, "learning_rate": 8e-7, "kl_beta": 0.0,
    "max_new_tokens": 2048, "max_context_tokens": 16384,
    "enable_thinking": True, "expected_protocol_version": expected_version,
    "expected_protocol_hash": expected_hash,
    "expected_student_prompt_sha256": expected_prompt,
    "expected_initial_adapter_sha256": expected_adapter,
    "expected_reference_adapter_sha256": expected_adapter,
    "expected_runtime_content_tree_sha256": expected_tree,
}
for key, expected in required.items(): assert defaults.get(key) == expected, (key, defaults.get(key), expected)
assert defaults["protocol_runtime_root"].resolve() == runtime.resolve()
assert runtime_content_tree_sha256(runtime) == expected_tree
environment = importlib.import_module("tool_environment_v26")
assert Path(inspect.getfile(environment.ToolUseEnv)).resolve() == overlay / "src/rl/tool_environment_v26.py"

assert importlib.metadata.version("trl").startswith("0.29.")
from trl import GRPOConfig
from trl.trainer.utils import RepeatSampler
assert inspect.signature(GRPOConfig).parameters["shuffle_dataset"].default is True
def two_passes():
    sampler = RepeatSampler(
        data_source=range(320), mini_repeat_count=16, batch_size=20,
        repeat_count=1, shuffle=True, seed=20260812,
    )
    return [list(iter(sampler)), list(iter(sampler))]
first_run = two_passes()
assert first_run == two_passes()
for sampled in first_run:
    assert len(sampled) == 5120
    groups = [sampled[offset:offset + 16] for offset in range(0, len(sampled), 16)]
    collapsed = [group[0] for group in groups]
    assert all(len(set(group)) == 1 for group in groups)
    assert len(collapsed) == 320 and set(collapsed) == set(range(320))
PY
}

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | sed -n "$(( $1 + 1 ))p" | tr -d '[:space:]'
}
wait_for_gpu() {
  local gpu=$1 used
  while true; do
    used=$(gpu_used_mib "$gpu")
    [[ "$used" =~ ^[0-9]+$ ]] || blocked "cannot read GPU$gpu memory"
    [[ "$used" -le 512 ]] && return 0
    set_status waiting_gpu "gpu=$gpu used_mib=$used; no process will be stopped"
    sleep 30
  done
}

validate_training_artifacts() {
  "$PYTHON_BIN" - "$TRAIN_OUT" "$TRAIN320_TASKS" "$ARM_B_SELECTION_MANIFEST" \
    "$VALIDATION64_AUDIT" "$FROZEN_ARM_B_SELECTION_MANIFEST_SHA256" \
    "$FROZEN_VALIDATION64_AUDIT_SHA256" "$EXPECTED_CONFIG_SHA256" \
    "$EXPECTED_PROTOCOL_VERSION" "$EXPECTED_PROTOCOL_HASH" \
    "$EXPECTED_STUDENT_PROMPT_SHA256" "$EXPECTED_SFT1_SHA256" \
    "$PROTOCOL_RUNTIME" "$EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" <<'PY'
import collections, hashlib, json, sys
from pathlib import Path
root, tasks_path, selection_path, validation_path = map(Path, sys.argv[1:5])
expected_selection, expected_validation, expected_config, expected_version, expected_hash, expected_prompt, expected_adapter, expected_runtime, expected_tree = sys.argv[5:14]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(selection_path) == expected_selection
assert sha(validation_path) == expected_validation
manifest = json.loads((root / "run_manifest.json").read_text())
for key, expected in {
    "experiment_config_sha256": expected_config,
    "examples_json_sha256": sha(tasks_path),
    "protocol_version": expected_version,
    "protocol_hash": expected_hash,
    "student_prompt_sha256": expected_prompt,
    "initial_adapter_sha256": expected_adapter,
    "records": 320, "expected_records": 320,
    "optimizer_steps": 32, "save_steps": 4, "save_total_limit": 1,
    "seed": 20260812, "prompts_per_update": 20, "group_size": 16,
    "ppo_iterations": 1, "learning_rate": 8e-7, "kl_beta": 0.0,
    "policy_reduction": "trajectory_token_mean", "result_reward_profile": "binary",
}.items(): assert manifest[key] == expected, (key, manifest.get(key), expected)
assert str(manifest["framework_version"]).startswith("0.29.")
assert manifest["rollout_settings"]["max_new_tokens"] == 2048
assert manifest["rollout_settings"]["max_context_tokens"] == 16384
assert manifest["rollout_settings"]["enable_thinking"] is True
runtime = manifest["runtime_module_audit"]
assert runtime["runtime_root"] == expected_runtime
assert runtime["tool_environment_factory_module"] == "tool_environment_v26"
identity = manifest["runtime_identity_audit"]
assert identity["expected"]["runtime_content_tree_sha256"] == expected_tree
assert identity["actual"]["runtime_content_tree_sha256"] == expected_tree
reference = manifest["reference_policy"]
assert reference["enabled"] is False and reference["kl_beta"] == 0.0
assert reference["adapter_name"] is None and reference["adapter_path"] is None
assert reference["expected_adapter_sha256"] == expected_adapter

checkpoints = sorted(path.name for path in root.glob("checkpoint-*") if path.is_dir())
assert checkpoints == ["checkpoint-32"], checkpoints
state = json.loads((root / "checkpoint-32/trainer_state.json").read_text())
assert int(state["global_step"]) == 32
checkpoint_adapter = root / "checkpoint-32/adapter_model.safetensors"
final_adapter = root / "final/adapter_model.safetensors"
assert checkpoint_adapter.is_file() and final_adapter.is_file()
assert sha(checkpoint_adapter) == sha(final_adapter)
assert (root / "training_precision.json").is_file()

tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
indices = [int(row["example_index"]) for row in tasks]
assert len(indices) == len(set(indices)) == 320
rollouts = [json.loads(line) for line in (root / "rollouts.jsonl").read_text().splitlines() if line.strip()]
assert len(rollouts) == 10240
for step in range(32):
    block = rollouts[step * 320:(step + 1) * 320]
    assert len(block) == 320
    assert {int(row["policy_global_step"]) for row in block} == {step}
    counts = collections.Counter(int(row["example_index"]) for row in block)
    assert len(counts) == 20 and set(counts.values()) == {16}
for pass_index in range(2):
    block = rollouts[pass_index * 5120:(pass_index + 1) * 5120]
    counts = collections.Counter(int(row["example_index"]) for row in block)
    assert counts == collections.Counter({index: 16 for index in indices})
PY
}

publish_training_contract() {
  "$PYTHON_BIN" - "$TRAIN_OUT" "$TRAINING_CONTRACT" "$ARM_B_SELECTION_MANIFEST" \
    "$TRAIN320_TASKS" "$VALIDATION64_AUDIT" "$SOURCE_COHORT_MANIFEST" \
    "$CONFIRMATORY_TRIGGER" "$EXPECTED_CONFIG_SHA256" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root, output, selection, tasks, validation, source, trigger = map(Path, sys.argv[1:8])
config_sha = sys.argv[8]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
record = {
    "schema_version": "qwen3-v26-arm-b-training-contract-v1",
    "status": "completed_primary_final32",
    "arm": "arm_b",
    "inputs": {
        "selection_manifest": {"path": str(selection.resolve()), "sha256": sha(selection)},
        "train320": {"path": str(tasks.resolve()), "sha256": sha(tasks)},
        "validation64_audit": {"path": str(validation.resolve()), "sha256": sha(validation)},
        "source_cohort_manifest": {"path": str(source.resolve()), "sha256": sha(source)},
        "confirmatory_trigger": {"path": str(trigger.resolve()), "sha256": sha(trigger)},
    },
    "training": {
        "records": 320, "group_size": 16, "prompts_per_update": 20,
        "optimizer_steps": 32, "train_passes": 2,
        "prompt_appearances": 640, "fresh_online_trajectories": 10240,
        "learning_rate": 8e-7, "kl_beta": 0.0,
        "reward_mode": "result-only", "result_reward_profile": "binary",
        "experiment_config_sha256": config_sha,
    },
    "checkpoint_policy": {
        "save_steps": 4, "save_total_limit": 1,
        "intermediate_steps": [4, 8, 12, 16, 20, 24, 28],
        "intermediate_purpose": "resume-and-training-diagnostics-only",
        "intermediate_evaluation_allowed": False,
        "primary_checkpoint": "checkpoint-32",
        "primary_global_step": 32,
        "final_must_equal_primary": True,
    },
    "outputs": {
        "run_manifest_sha256": sha(root / "run_manifest.json"),
        "implementation_lock_sha256": sha(root / "implementation_lock.json"),
        "checkpoint32_adapter_sha256": sha(root / "checkpoint-32/adapter_model.safetensors"),
        "final_adapter_sha256": sha(root / "final/adapter_model.safetensors"),
    },
}
assert record["outputs"]["checkpoint32_adapter_sha256"] == record["outputs"]["final_adapter_sha256"]
payload = (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
if output.exists() and output.read_bytes() != payload:
    raise RuntimeError(f"refusing changed Arm B training contract: {output}")
temporary = output.with_name(output.name + ".next")
temporary.write_bytes(payload); os.replace(temporary, output)
PY
}

RESUME_CHECKPOINT=""
RESUME_PREPARATION=$TRAIN_OUT/resume_preparation.json
TRAIN_ALREADY_COMPLETE=0
prepare_train_output() {
  if [[ -f "$TRAIN_OUT/final/adapter_model.safetensors" ]]; then
    validate_training_artifacts || blocked 'existing Arm B final failed exact final32 audit'
    verify_arm_b_inputs >/dev/null || blocked 'Arm B admission changed before completed-manifest binding'
    publish_training_contract || blocked 'existing Arm B final contract publication failed'
    TRAIN_ALREADY_COMPLETE=1
    return
  fi
  [[ ! -e "$TRAIN_OUT/final" ]] || blocked "incomplete final directory: $TRAIN_OUT/final"
  if [[ ! -d "$TRAIN_OUT" ]] || [[ -z "$(find "$TRAIN_OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    mkdir -p "$TRAIN_OUT"
    return
  fi
  [[ -f "$TRAIN_OUT/run_manifest.json" && -f "$TRAIN_OUT/implementation_lock.json" ]] \
    || blocked 'nonempty Arm B train output lacks immutable run locks'
  "$PYTHON_BIN" "$TRAIN_RUNTIME/src/rl/diagnostics/prepare_vanilla_grpo_resume.py" \
    --train-output "$TRAIN_OUT" --tasks "$TRAIN320_TASKS" \
    --expected-records 320 --optimizer-steps 32 --prompts-per-update 20 \
    --group-size 16 --save-steps 4 --output "$RESUME_PREPARATION" \
    || blocked 'no safe Arm B checkpoint/rollout prefix is resumable'
  RESUME_CHECKPOINT=$("$PYTHON_BIN" - "$RESUME_PREPARATION" "$TRAIN_OUT" <<'PY'
import json, sys
from pathlib import Path
record = json.load(open(sys.argv[1])); root = Path(sys.argv[2]).resolve()
assert record["schema_version"] == "vanilla-grpo-resume-preparation-v1"
assert record["status"] == "resume_ready"
assert record["contract"] == {
    "optimizer_steps": 32, "prompts_per_update": 20, "group_size": 16,
    "save_steps": 4, "committed_rollouts_per_step": 320,
}
step = record["checkpoint"]["global_step"]
checkpoint = Path(record["checkpoint"]["path"]).resolve()
assert 0 < step < 32 and step % 4 == 0
assert checkpoint == root / f"checkpoint-{step}"
assert record["rollouts"]["retained_rows"] == step * 320
print(checkpoint)
PY
  ) || blocked 'Arm B resume preparation record failed verification'
}

start_train_vllm() {
  curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1 \
    && blocked "port $VLLM_PORT already has a server; refusing unowned reuse"
  set_status starting_vllm "gpu=$VLLM_GPU port=$VLLM_PORT"
  setsid env CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
    PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table MODEL_PATH="$MODEL_PATH" \
    VLLM_PORT="$VLLM_PORT" VLLM_GPU_MEMORY_UTILIZATION=0.82 MAX_MODEL_LEN=16384 \
    bash "$TRAIN_RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" >>"$VLLM_LOG" 2>&1 &
  vllm_pgid=$!
  local ready=0
  for _ in {1..240}; do
    kill -0 "$vllm_pgid" 2>/dev/null || break
    if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 2
  done
  [[ "$ready" -eq 1 ]] || blocked "owned Arm B vLLM failed readiness: $VLLM_LOG"
}

run_train() {
  verify_arm_b_inputs >/dev/null || blocked 'Arm B admission changed after preflight'
  prepare_train_output
  if [[ "$TRAIN_ALREADY_COMPLETE" -eq 1 ]]; then
    set_terminal_status complete "existing validated primary=checkpoint-32 contract=$TRAINING_CONTRACT"
    return
  fi
  wait_for_gpu "$TRAIN_GPU"; wait_for_gpu "$VLLM_GPU"
  verify_arm_b_inputs >/dev/null || blocked 'Arm B/Arm A state changed immediately before training launch'
  start_train_vllm
  local resume_args=()
  [[ -z "$RESUME_CHECKPOINT" ]] || resume_args=(--resume-from-checkpoint "$RESUME_CHECKPOINT")
  set_status training 'arm=arm_b records=320 k=16 updates=32 prompts=20 passes=2 save_steps=4 intermediate=resume-only primary=step32'
  setsid env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PROJECT_DIR="$TRAIN_RUNTIME" \
    PYTHON="$PYTHON_BIN" MODEL_PATH="$MODEL_PATH" ADAPTER_PATH="$SFT1_ADAPTER" \
    EXAMPLES_JSON="$TRAIN320_TASKS" OUTPUT_DIR="$TRAIN_OUT" \
    EXPERIMENT_CONFIG="$EXPERIMENT_CONFIG" VLLM_PORT="$VLLM_PORT" \
    VLLM_GROUP_PORT="$VLLM_GROUP_PORT" PYTHONPATH= \
    bash "$TRAIN_RUNTIME/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" \
      --optimizer-steps 32 --ppo-iterations 1 --prompts-per-update 20 \
      --group-size 16 --kl-beta 0 --protocol-runtime-root "$PROTOCOL_RUNTIME" \
      --transition-micro-batch-size 1 --save-steps 4 --save-total-limit 1 \
      --seed "$TRAIN_SEED" "${resume_args[@]}" >>"$TRAIN_LOG" 2>&1 &
  worker_pgid=$!
  local code=0
  if wait "$worker_pgid"; then code=0; else code=$?; fi
  terminate_session "$worker_pgid"; worker_pgid=""
  terminate_session "$vllm_pgid"; vllm_pgid=""
  [[ "$code" -eq 0 ]] || return "$code"
  validate_training_artifacts || blocked 'Arm B output failed exact two-pass/final32 audit'
  verify_arm_b_inputs >/dev/null || blocked 'Arm B admission changed before completed-manifest binding'
  publish_training_contract || blocked 'Arm B training contract publication failed'
  set_terminal_status complete "train=$TRAIN_OUT primary_checkpoint=step32 contract=$TRAINING_CONTRACT"
}

validate_static_inputs
case "$MODE" in
  preflight)
    verify_arm_b_inputs
    set_terminal_status complete 'Arm B preflight admitted; no GPU allocated'
    ;;
  train) run_train ;;
esac
