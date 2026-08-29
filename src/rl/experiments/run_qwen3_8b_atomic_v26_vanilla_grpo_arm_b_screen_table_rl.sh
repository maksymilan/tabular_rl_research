#!/usr/bin/env bash
# Staged, fail-closed screening for the preregistered Qwen3-8B v26 Arm B.
#
# This launcher is independent of Arm A and never starts a later stage
# automatically.  Workers publish only per-task atomic groups.  Finalization
# assembles an isolated transaction, verifies it, and only then renames the
# complete fixed pool into its canonical path.  The default is read-only.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONOPTIMIZE

MODE=${1:-dry-run}
if [[ $# -gt 1 ]]; then
  printf 'usage: %s [dry-run|prepare-f1|worker-f1|finalize-f1|prepare-f2|worker-f2|finalize-f2|prepare-f3|worker-f3|finalize-f3]\n' "$0" >&2
  exit 2
fi
case "$MODE" in
  dry-run|prepare-f1|worker-f1|finalize-f1|prepare-f2|worker-f2|finalize-f2|prepare-f3|worker-f3|finalize-f3) ;;
  *)
    printf 'usage: %s [dry-run|prepare-f1|worker-f1|finalize-f1|prepare-f2|worker-f2|finalize-f2|prepare-f3|worker-f3|finalize-f3]\n' "$0" >&2
    exit 2
    ;;
esac

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
PROJECT_ROOT=${PROJECT_ROOT:-/home/dengyan/tabular_rl_project}
TRAIN_RUNTIME=${TRAIN_RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_arm_b_screen_20260812}
PUBLIC_SELECTOR_ROOT=${PUBLIC_SELECTOR_ROOT:-$TRAIN_RUNTIME/support/public-selector-v1}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
SFT1_ADAPTER=${SFT1_ADAPTER:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
REMOTE_DB_ROOT=${REMOTE_DB_ROOT:-$PROJECT_ROOT/data/bird/train/train_databases}

RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_vanilla_grpo_arm_b_screen_20260812}
INPUT_DIR=${INPUT_DIR:-$RUN_ROOT/inputs}
F1_TASKS=${F1_TASKS:-$INPUT_DIR/wide3000.jsonl}
F1_COHORT_MANIFEST=${F1_COHORT_MANIFEST:-$INPUT_DIR/wide3000.manifest.json}
F1_ROOT=${F1_ROOT:-$RUN_ROOT/f1_wide3000_k8_seed20260814}
F1_AUDIT_DIR=${F1_AUDIT_DIR:-$F1_ROOT/audit}
F1_AUDIT=${F1_AUDIT:-$F1_AUDIT_DIR/arm_b_wide_k8_screen_audit.json}
F2_TASKS=${F2_TASKS:-$F1_AUDIT_DIR/confirmation640.jsonl}
F2_ROOT=${F2_ROOT:-$RUN_ROOT/f2_confirmation640_k16_seed20260815}
F2_SELECTION_DIR=${F2_SELECTION_DIR:-$F2_ROOT/selection}
F2_AUDIT=${F2_AUDIT:-$F2_SELECTION_DIR/arm_b_k16_confirmation_audit.json}
F2_SELECTION_MANIFEST=${F2_SELECTION_MANIFEST:-$F2_SELECTION_DIR/arm_b_selection_manifest.json}
F3_TASKS=${F3_TASKS:-$F2_SELECTION_DIR/validation64.jsonl}
F3_ROOT=${F3_ROOT:-$RUN_ROOT/f3_validation64_k16_seed20260816}
F3_AUDIT_DIR=${F3_AUDIT_DIR:-$F3_ROOT/audit}
F3_AUDIT=${F3_AUDIT:-$F3_AUDIT_DIR/arm_b_k16_validation_audit.json}
CONFIRMATORY_TRIGGER=${CONFIRMATORY_TRIGGER:-$F3_AUDIT_DIR/confirmatory_arm_trigger.json}

REFERENCE_TASKS=${REFERENCE_TASKS:-$TRAIN_RUNTIME/data/rl_inputs/s2_sources/bird_train_filtered.jsonl}
ELIGIBLE_TASKS=${ELIGIBLE_TASKS:-$TRAIN_RUNTIME/data/rl_inputs/s2_sources/bird_train_tool_compatible_nonempty_v1.jsonl}
BASELINE_EVAL300=${BASELINE_EVAL300:-$TRAIN_RUNTIME/data/rl_inputs/s2_sources/bird_train_baseline300_v1.jsonl}
SFT1_INDEX=${SFT1_INDEX:-}
OLD_MIXED60=${OLD_MIXED60:-}

# Immutable absence checks enforce Arm B's time ordering at every active step.
# They do not scan or signal processes and therefore cannot disturb Arm A,
# NewGNN, or any unrelated experiment.
ARM_A_FORMAL_TRAIN_DIR=${ARM_A_FORMAL_TRAIN_DIR:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_boundary300_vanilla_grpo_20260812/train300_two_pass_seed20260812}
ARM_A_FULL_DEV_DIR=${ARM_A_FULL_DEV_DIR:-$OUTPUT_ROOT/evaluations/qwen3_8b_atomic_v26_boundary300_vanilla_formal}

ARM_A_TRIGGER_MODE=${ARM_A_TRIGGER_MODE:-}
ARM_A_SCREEN_POOL_COUNT=${ARM_A_SCREEN_POOL_COUNT:-}
ARM_A_SELECTION_MANIFEST=${ARM_A_SELECTION_MANIFEST:-}
FROZEN_ARM_A_SELECTION_MANIFEST_SHA256=${FROZEN_ARM_A_SELECTION_MANIFEST_SHA256:-}
ARM_A_VALIDATION_MANIFEST=${ARM_A_VALIDATION_MANIFEST:-}
FROZEN_ARM_A_VALIDATION_MANIFEST_SHA256=${FROZEN_ARM_A_VALIDATION_MANIFEST_SHA256:-}
ARM_A_VALIDATION_TASKS=${ARM_A_VALIDATION_TASKS:-}
FROZEN_ARM_A_VALIDATION_TASKS_SHA256=${FROZEN_ARM_A_VALIDATION_TASKS_SHA256:-}
ARM_A_VALIDATION_TRAJECTORIES=${ARM_A_VALIDATION_TRAJECTORIES:-}
FROZEN_ARM_A_VALIDATION_TRAJECTORIES_SHA256=${FROZEN_ARM_A_VALIDATION_TRAJECTORIES_SHA256:-}

F1_SHARDS=${F1_SHARDS:-4}
F2_SHARDS=${F2_SHARDS:-4}
F3_SHARDS=${F3_SHARDS:-4}
# Active modes require comma-separated explicit mappings.  Any nonnegative GPU
# index is allowed; this source does not permanently reserve or forbid GPU5.
F1_GPU_MAP=${F1_GPU_MAP:-}
F2_GPU_MAP=${F2_GPU_MAP:-}
F3_GPU_MAP=${F3_GPU_MAP:-}
SHARD_INDEX=${SHARD_INDEX:-}
GPU_IDLE_MAX_MIB=${GPU_IDLE_MAX_MIB:-512}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.82}

FROZEN_F1_COHORT_MANIFEST_SHA256=${FROZEN_F1_COHORT_MANIFEST_SHA256:-}
FROZEN_F1_PLAN_SHA256=${FROZEN_F1_PLAN_SHA256:-}
FROZEN_F1_AUDIT_SHA256=${FROZEN_F1_AUDIT_SHA256:-}
FROZEN_F2_PLAN_SHA256=${FROZEN_F2_PLAN_SHA256:-}
FROZEN_F2_SELECTION_MANIFEST_SHA256=${FROZEN_F2_SELECTION_MANIFEST_SHA256:-}
FROZEN_F3_PLAN_SHA256=${FROZEN_F3_PLAN_SHA256:-}

WIDE_PREPARER=${WIDE_PREPARER:-$TRAIN_RUNTIME/src/rl/prepare_vanilla_grpo_arm_b_wide3000.py}
WIDE_AUDITOR=${WIDE_AUDITOR:-$TRAIN_RUNTIME/src/rl/diagnostics/audit_vanilla_grpo_arm_b_wide_k8_screen.py}
K16_SELECTOR=${K16_SELECTOR:-$TRAIN_RUNTIME/src/rl/select_vanilla_grpo_arm_b_k16.py}
K16_VALIDATION_AUDITOR=${K16_VALIDATION_AUDITOR:-$TRAIN_RUNTIME/src/rl/diagnostics/audit_vanilla_grpo_arm_b_k16_validation.py}
CONFIRMATORY_TRIGGER_PREPARER=${CONFIRMATORY_TRIGGER_PREPARER:-$TRAIN_RUNTIME/src/rl/diagnostics/prepare_vanilla_grpo_confirmatory_arm_trigger.py}
ARM_B_TRAINING_INPUT_VALIDATOR=${ARM_B_TRAINING_INPUT_VALIDATOR:-$TRAIN_RUNTIME/src/rl/diagnostics/validate_arm_b_training_inputs.py}
STAGE_MANAGER=${STAGE_MANAGER:-$TRAIN_RUNTIME/src/rl/diagnostics/manage_vanilla_grpo_arm_b_screen_stage.py}
GROUP_RESUME_HELPER=${GROUP_RESUME_HELPER:-$TRAIN_RUNTIME/src/rl/diagnostics/prepare_boundary_screen_group_resume.py}
ARM_B_LIBRARY=${ARM_B_LIBRARY:-$TRAIN_RUNTIME/src/rl/vanilla_grpo_arm_b.py}

EXPECTED_WIDE_PREPARER_SHA256=5210b767a600b744b6d47387528c823f4d6fd493e3815c6ff050b2747c2f7c83
EXPECTED_WIDE_AUDITOR_SHA256=59ed1d365d2257090ef9b19e319297b21b6594f57a50b42c9f25c2557008007b
EXPECTED_K16_SELECTOR_SHA256=9fb4cde0eef433240cde1e30f1e215ad76116bfc4631bac28e7479c6cbd7789b
EXPECTED_K16_VALIDATION_AUDITOR_SHA256=1b3e52770519d663cb6eb24f10fccba0fae974dde6247e20c5e8063fa66e90e0
EXPECTED_CONFIRMATORY_TRIGGER_PREPARER_SHA256=554fe25b39fe7c012a73b8d4045762476347bfd6fbcc70355f98f4cb8bdd7b98
EXPECTED_ARM_B_TRAINING_INPUT_VALIDATOR_SHA256=a416b9490e8c6287c7f05fff5c2c314c528b8524dddbd782c5e25ed6eb7125e0
EXPECTED_STAGE_MANAGER_SHA256=ba38dd608d652d9f98ae8f8d6119ab515cfa5b199a98e5aee779cde211305d3a
EXPECTED_GROUP_RESUME_HELPER_SHA256=4ef3642cf6737c35a231ba0419a4d319f2052a4a2bcae1cbbd6088e5186576fe
EXPECTED_ARM_B_LIBRARY_SHA256=d596a01a89a125ec0375af7d3a6dbdd2c1b5a65a2b625dc8d07cb502c29e5a80

EXPECTED_REFERENCE_SHA256=24d21a349c430df549f92724dd07de1c59b2365cd72802957120afba13c7a2e1
EXPECTED_ELIGIBLE_SHA256=c8ea3c6419ac57f05021eca470c7519843456e8ac02b04cca9af39e779ec7545
EXPECTED_BASELINE_EVAL300_SHA256=87b37b307ea2710acdff7d03bdc474a6ba2cdb8ed51b005cff0fe8fa54c67c03
EXPECTED_SFT1_INDEX_SHA256=aa6b8a72a1cfbcf44e37702e0498c9a85f615edc201ba3f4172b65f701e0cda5
EXPECTED_OLD_MIXED60_SHA256=ef97b6977735996fde505aaad4e00215570963b37a1c82d890c34c4b414f30ae
EXPECTED_GENERATOR_SHA256=db934c05cbf4f2d9beef3e01e8b42ff74312ef00834e681a9ae65647de12e191
EXPECTED_ROLLOUT_SHA256=87c36224ba0f01954db0d58db3bc3e5685bc5c86736c0679299c76872b6c821f
EXPECTED_SCORING_SHA256=a11233e7a04a9efa34393e7d77a4c4aca34151144bf531978235ef6d644c1ac5
EXPECTED_TASK_LOADER_SHA256=d79d41ea5f32ddfa45bb7c1496e8de234f96a6b48d84dfa2363d0de37a024d4b
EXPECTED_TOOL_ENV_V26_SHA256=c7a84bdb2d259eea91cde5a77a5758ac8ea57e02828330088ecbd903b1d0ec5c
EXPECTED_FIXED_POOL_RUNTIME_SHA256=e0be4c4c830ca2e87172175553ab47d1b19561510f8fa6f77410e96846cf200d
EXPECTED_TRANSITION_BATCH_SHA256=c6477393589581209ad581230eb598160e0674469103ce197b7d7d7f26a0bcd8
EXPECTED_TERMINAL_REWARD_SHA256=2b45d45f562a903fe6e83589bc3aa9b9cc8d22036be6d0544c562ced82757b7b
EXPECTED_ARM_A_VALIDATION_AUDITOR_SHA256=b855786dbeabf42633fe6775a436c1eb29e8e79d9bfa1c326453fcdc8ab67489
EXPECTED_ARM_A_SELECTOR_SHA256=2849de8109b74cce59ca590b94b478edc5cf1a4c120ebda0e1d71902791c01d8
EXPECTED_PUBLIC_SELECTOR_SHA256=afaf6b97cb80371054fdd96b21b5f44269ec690e6760cce460e22bc19a461b1c
EXPECTED_SFT1_SHA256=3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5
EXPECTED_SFT1_CONFIG_SHA256=537dbc946a7131d59e294a8729ace4aa145d73e59f4cf8556360ea507ce4435a
EXPECTED_SFT1_STATE_SHA256=97e529475d8c4f68dc88bc1f80370dacab478a681629f5f99003c39c70e9600a
EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256=5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab
EXPECTED_PROTOCOL_VERSION=version26
EXPECTED_PROTOCOL_HASH=4da19387399bd3a5
EXPECTED_STUDENT_PROMPT_SHA256=848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
die() { printf 'blocked: %s\n' "$1" >&2; exit 3; }
require_explicit_sha() {
  [[ "$1" =~ ^[0-9a-f]{64}$ ]] || die "$2 must be an explicit lowercase SHA-256"
}
require_sha() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "$path" && ! -L "$path" ]] || die "missing/non-regular $label: $path"
  require_explicit_sha "$expected" "$label expected digest"
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] \
    || die "$label SHA-256 mismatch: expected=$expected actual=$actual path=$path"
}
write_status() {
  local path=$1 state=$2 detail=$3 temporary=${1}.next
  mkdir -p "$(dirname "$path")"
  printf '%s\t%s\t%s\n' "$(timestamp)" "$state" "$detail" >"$temporary"
  mv "$temporary" "$path"
}

validate_owned_paths() {
  "$PYTHON_BIN" - "$RUN_ROOT" "$INPUT_DIR" "$F1_ROOT" "$F1_AUDIT_DIR" \
    "$F2_ROOT" "$F2_SELECTION_DIR" "$F3_ROOT" "$F3_AUDIT_DIR" <<'PY'
import sys
from pathlib import Path
root, *owned = map(Path, sys.argv[1:])
if not root.is_absolute():
    raise SystemExit("Arm B RUN_ROOT must be absolute")
resolved_root = root.resolve()
for path in owned:
    if not path.is_absolute() or not path.resolve().is_relative_to(resolved_root):
        raise SystemExit(f"owned Arm B path escapes RUN_ROOT: {path}")
    current = path
    while current != root.parent and current.parent != current:
        if current.exists() and current.is_symlink():
            raise SystemExit(f"owned Arm B path crosses symlink below RUN_ROOT: {current}")
        current = current.parent
if len({path.resolve() for path in owned}) != len(owned):
    raise SystemExit("Arm B stage/input/audit directories must be distinct")
PY
}

validate_arm_a_not_started() {
  [[ "$ARM_A_FORMAL_TRAIN_DIR" == /* && "$ARM_A_FULL_DEV_DIR" == /* ]] \
    || die 'Arm A formal-train/full-dev paths must be absolute'
  [[ ! -e "$ARM_A_FORMAL_TRAIN_DIR" ]] \
    || die "Arm A formal training has started or allocated its canonical directory: $ARM_A_FORMAL_TRAIN_DIR"
  [[ ! -e "$ARM_A_FULL_DEV_DIR" ]] \
    || die "Arm A confirmatory full-dev has started or allocated its canonical directory: $ARM_A_FULL_DEV_DIR"
}

verify_protocol_runtime() {
  env PYTHONPATH= "$PYTHON_BIN" - "$PROTOCOL_RUNTIME" \
    "$EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" "$EXPECTED_PROTOCOL_VERSION" \
    "$EXPECTED_PROTOCOL_HASH" "$EXPECTED_STUDENT_PROMPT_SHA256" <<'PY'
import hashlib, importlib, sys
from pathlib import Path
runtime = Path(sys.argv[1]).resolve()
digest, files = hashlib.sha256(), []
for relative in ("src/eval", "src/sft", "src/harness"):
    directory = runtime / relative
    if not directory.is_dir():
        raise SystemExit(f"incomplete protocol runtime: {directory}")
    files.extend(path for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
for path in sorted(files, key=lambda value: value.relative_to(runtime).as_posix()):
    digest.update(path.relative_to(runtime).as_posix().encode()); digest.update(b"\0")
    digest.update(path.read_bytes()); digest.update(b"\0")
if digest.hexdigest() != sys.argv[2]:
    raise SystemExit("protocol runtime content-tree SHA-256 mismatch")
sys.path.insert(0, str(runtime / "src/sft"))
protocol = importlib.import_module("protocol")
prompt = protocol.student_runtime_system_prompt(context_mode="rolling-legal-history", compact=False)
assert protocol.PROTOCOL_VERSION == sys.argv[3]
assert protocol.protocol_hash(prompt) == sys.argv[4]
assert hashlib.sha256(prompt.encode()).hexdigest() == sys.argv[5]
PY
}

require_common_assets() {
  [[ -x "$PYTHON_BIN" ]] || die "missing Python runtime: $PYTHON_BIN"
  validate_owned_paths || die 'Arm B resolved path ownership check failed'
  validate_arm_a_not_started
  [[ -d "$MODEL_PATH" && ! -L "$MODEL_PATH" ]] || die "missing/unsafe base model: $MODEL_PATH"
  [[ -d "$SFT1_ADAPTER" && ! -L "$SFT1_ADAPTER" ]] || die "missing/unsafe SFT1 adapter: $SFT1_ADAPTER"
  for forbidden in src/eval src/sft src/harness src/tool_modules; do
    [[ ! -e "$TRAIN_RUNTIME/$forbidden" ]] \
      || die "training overlay shadows frozen protocol source: $TRAIN_RUNTIME/$forbidden"
  done
  [[ "$PUBLIC_SELECTOR_ROOT" == "$TRAIN_RUNTIME/support/public-selector-v1" ]] \
    || die 'PUBLIC_SELECTOR_ROOT must be the dedicated pinned support root'
  env PYTHONPATH= "$PYTHON_BIN" - "$PUBLIC_SELECTOR_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
expected = root / "src/eval/select_bird_train_baseline.py"
if not root.is_dir() or root.is_symlink():
    raise SystemExit("public selector support root is missing or a symlink")
files = [path for path in root.rglob("*") if path.is_file() or path.is_symlink()]
if files != [expected] or expected.is_symlink():
    raise SystemExit(f"public selector support root must contain exactly {expected}: {files}")
PY
  require_sha "$STAGE_MANAGER" "$EXPECTED_STAGE_MANAGER_SHA256" arm_b_stage_manager
  require_sha "$GROUP_RESUME_HELPER" "$EXPECTED_GROUP_RESUME_HELPER_SHA256" group_resume_helper
  require_sha "$ARM_B_LIBRARY" "$EXPECTED_ARM_B_LIBRARY_SHA256" arm_b_shared_library
  require_sha "$CONFIRMATORY_TRIGGER_PREPARER" "$EXPECTED_CONFIRMATORY_TRIGGER_PREPARER_SHA256" confirmatory_trigger_preparer
  require_sha "$ARM_B_TRAINING_INPUT_VALIDATOR" "$EXPECTED_ARM_B_TRAINING_INPUT_VALIDATOR_SHA256" arm_b_training_input_validator
  require_sha "$TRAIN_RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py" "$EXPECTED_GENERATOR_SHA256" fixed_pool_generator
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/rollout.py" "$EXPECTED_ROLLOUT_SHA256" rollout_runtime
  require_sha "$TRAIN_RUNTIME/src/rl/rollout_scoring.py" "$EXPECTED_SCORING_SHA256" rollout_scoring
  require_sha "$TRAIN_RUNTIME/src/rl/task_loader.py" "$EXPECTED_TASK_LOADER_SHA256" task_loader
  require_sha "$TRAIN_RUNTIME/src/rl/tool_environment_v26.py" "$EXPECTED_TOOL_ENV_V26_SHA256" tool_environment_v26
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/fixed_rollout_pool.py" "$EXPECTED_FIXED_POOL_RUNTIME_SHA256" fixed_rollout_pool_runtime
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/transition_batch.py" "$EXPECTED_TRANSITION_BATCH_SHA256" transition_batch_runtime
  require_sha "$TRAIN_RUNTIME/src/rl/terminal_reward.py" "$EXPECTED_TERMINAL_REWARD_SHA256" terminal_reward
  require_sha "$MODEL_PATH/model-00001-of-00005.safetensors" 31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f model_shard_1
  require_sha "$MODEL_PATH/model-00002-of-00005.safetensors" 5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282 model_shard_2
  require_sha "$MODEL_PATH/model-00003-of-00005.safetensors" c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836 model_shard_3
  require_sha "$MODEL_PATH/model-00004-of-00005.safetensors" b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a model_shard_4
  require_sha "$MODEL_PATH/model-00005-of-00005.safetensors" 20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff model_shard_5
  require_sha "$MODEL_PATH/config.json" f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30 model_config
  require_sha "$MODEL_PATH/generation_config.json" 2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2 generation_config
  require_sha "$MODEL_PATH/merges.txt" 8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5 tokenizer_merges
  require_sha "$MODEL_PATH/model.safetensors.index.json" f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc model_index
  require_sha "$MODEL_PATH/tokenizer_config.json" d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101 tokenizer_config
  require_sha "$MODEL_PATH/tokenizer.json" aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 tokenizer
  require_sha "$MODEL_PATH/vocab.json" ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 tokenizer_vocab
  require_sha "$SFT1_ADAPTER/adapter_model.safetensors" "$EXPECTED_SFT1_SHA256" sft1_adapter
  require_sha "$SFT1_ADAPTER/adapter_config.json" "$EXPECTED_SFT1_CONFIG_SHA256" sft1_adapter_config
  require_sha "$SFT1_ADAPTER/trainer_state.json" "$EXPECTED_SFT1_STATE_SHA256" sft1_trainer_state
  verify_protocol_runtime || die 'frozen protocol runtime verification failed'
}

validate_gpu_map() {
  local shards=$1 raw=$2 label=$3 gpu
  [[ "$shards" =~ ^[1-9][0-9]*$ ]] || die "$label shard count must be positive"
  [[ -n "$raw" ]] || die "$label GPU map must be explicitly supplied"
  IFS=',' read -r -a parsed_gpu_map <<<"$raw"
  [[ "${#parsed_gpu_map[@]}" -eq "$shards" ]] \
    || die "$label GPU map must contain exactly $shards comma-separated indices"
  declare -A observed=()
  for gpu in "${parsed_gpu_map[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "$label GPU map contains a nonnegative-index violation: $gpu"
    [[ -z "${observed[$gpu]:-}" ]] || die "$label GPU map contains duplicate GPU$gpu"
    observed[$gpu]=1
  done
}

stage_init() {
  STAGE=$1
  case "$STAGE" in
    F1)
      STAGE_KEY=f1; STAGE_ROOT=$F1_ROOT; STAGE_TASKS=$F1_TASKS
      STAGE_TASK_MANIFEST=$F1_COHORT_MANIFEST; STAGE_ACTIVATION=$F1_COHORT_MANIFEST
      STAGE_ACTIVATION_SHA=$FROZEN_F1_COHORT_MANIFEST_SHA256
      STAGE_RECORDS=3000; STAGE_GROUP_SIZE=8; STAGE_SEED=20260814
      STAGE_SHARDS=$F1_SHARDS; STAGE_GPU_MAP=$F1_GPU_MAP
      STAGE_HOOK=$WIDE_AUDITOR; STAGE_HOOK_SHA=$EXPECTED_WIDE_AUDITOR_SHA256
      FROZEN_STAGE_PLAN_SHA=$FROZEN_F1_PLAN_SHA256
      ;;
    F2)
      STAGE_KEY=f2; STAGE_ROOT=$F2_ROOT; STAGE_TASKS=$F2_TASKS
      STAGE_TASK_MANIFEST=$F1_AUDIT; STAGE_ACTIVATION=$F1_AUDIT
      STAGE_ACTIVATION_SHA=$FROZEN_F1_AUDIT_SHA256
      STAGE_RECORDS=640; STAGE_GROUP_SIZE=16; STAGE_SEED=20260815
      STAGE_SHARDS=$F2_SHARDS; STAGE_GPU_MAP=$F2_GPU_MAP
      STAGE_HOOK=$K16_SELECTOR; STAGE_HOOK_SHA=$EXPECTED_K16_SELECTOR_SHA256
      FROZEN_STAGE_PLAN_SHA=$FROZEN_F2_PLAN_SHA256
      ;;
    F3)
      STAGE_KEY=f3; STAGE_ROOT=$F3_ROOT; STAGE_TASKS=$F3_TASKS
      STAGE_TASK_MANIFEST=$F2_SELECTION_MANIFEST; STAGE_ACTIVATION=$F2_SELECTION_MANIFEST
      STAGE_ACTIVATION_SHA=$FROZEN_F2_SELECTION_MANIFEST_SHA256
      STAGE_RECORDS=64; STAGE_GROUP_SIZE=16; STAGE_SEED=20260816
      STAGE_SHARDS=$F3_SHARDS; STAGE_GPU_MAP=$F3_GPU_MAP
      STAGE_HOOK=$K16_VALIDATION_AUDITOR; STAGE_HOOK_SHA=$EXPECTED_K16_VALIDATION_AUDITOR_SHA256
      FROZEN_STAGE_PLAN_SHA=$FROZEN_F3_PLAN_SHA256
      ;;
    *) die "unknown stage: $STAGE" ;;
  esac
  STAGE_ASSIGNMENTS=$STAGE_ROOT/assignments
  STAGE_PLAN=$STAGE_ROOT/stage_plan.json
  STAGE_WORKERS=$STAGE_ROOT/workers
  STAGE_POOL=$STAGE_ROOT/pool
  STAGE_TRANSACTION=$STAGE_ROOT/transactions/pool.next
  STAGE_QUARANTINE=$STAGE_ROOT/quarantine
  STAGE_LOGS=$STAGE_ROOT/logs
  STAGE_STATUS=$STAGE_ROOT/status
  STAGE_LOCKS=$STAGE_ROOT/locks
  validate_gpu_map "$STAGE_SHARDS" "$STAGE_GPU_MAP" "$STAGE"
}

stage_bind_args() {
  STAGE_BIND_ARGS=(
    --bind stage-manager "$STAGE_MANAGER" "$EXPECTED_STAGE_MANAGER_SHA256"
    --bind group-resume-helper "$GROUP_RESUME_HELPER" "$EXPECTED_GROUP_RESUME_HELPER_SHA256"
    --bind arm-b-library "$ARM_B_LIBRARY" "$EXPECTED_ARM_B_LIBRARY_SHA256"
    --bind wide-preparer "$WIDE_PREPARER" "$EXPECTED_WIDE_PREPARER_SHA256"
    --bind public-selector "$PUBLIC_SELECTOR_ROOT/src/eval/select_bird_train_baseline.py" "$EXPECTED_PUBLIC_SELECTOR_SHA256"
    --bind stage-hook "$STAGE_HOOK" "$STAGE_HOOK_SHA"
    --bind confirmatory-trigger-preparer "$CONFIRMATORY_TRIGGER_PREPARER" "$EXPECTED_CONFIRMATORY_TRIGGER_PREPARER_SHA256"
    --bind arm-b-training-input-validator "$ARM_B_TRAINING_INPUT_VALIDATOR" "$EXPECTED_ARM_B_TRAINING_INPUT_VALIDATOR_SHA256"
    --bind generator "$TRAIN_RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py" "$EXPECTED_GENERATOR_SHA256"
    --bind rollout "$TRAIN_RUNTIME/src/rl/frameworks/trl/rollout.py" "$EXPECTED_ROLLOUT_SHA256"
    --bind rollout-scoring "$TRAIN_RUNTIME/src/rl/rollout_scoring.py" "$EXPECTED_SCORING_SHA256"
    --bind task-loader "$TRAIN_RUNTIME/src/rl/task_loader.py" "$EXPECTED_TASK_LOADER_SHA256"
    --bind tool-environment-v26 "$TRAIN_RUNTIME/src/rl/tool_environment_v26.py" "$EXPECTED_TOOL_ENV_V26_SHA256"
    --bind fixed-pool-runtime "$TRAIN_RUNTIME/src/rl/frameworks/trl/fixed_rollout_pool.py" "$EXPECTED_FIXED_POOL_RUNTIME_SHA256"
    --bind transition-batch "$TRAIN_RUNTIME/src/rl/frameworks/trl/transition_batch.py" "$EXPECTED_TRANSITION_BATCH_SHA256"
    --bind terminal-reward "$TRAIN_RUNTIME/src/rl/terminal_reward.py" "$EXPECTED_TERMINAL_REWARD_SHA256"
    --bind model-shard-1 "$MODEL_PATH/model-00001-of-00005.safetensors" 31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f
    --bind model-shard-2 "$MODEL_PATH/model-00002-of-00005.safetensors" 5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282
    --bind model-shard-3 "$MODEL_PATH/model-00003-of-00005.safetensors" c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836
    --bind model-shard-4 "$MODEL_PATH/model-00004-of-00005.safetensors" b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a
    --bind model-shard-5 "$MODEL_PATH/model-00005-of-00005.safetensors" 20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff
    --bind model-config "$MODEL_PATH/config.json" f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30
    --bind generation-config "$MODEL_PATH/generation_config.json" 2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2
    --bind tokenizer-merges "$MODEL_PATH/merges.txt" 8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5
    --bind model-index "$MODEL_PATH/model.safetensors.index.json" f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc
    --bind tokenizer-config "$MODEL_PATH/tokenizer_config.json" d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101
    --bind tokenizer "$MODEL_PATH/tokenizer.json" aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
    --bind tokenizer-vocab "$MODEL_PATH/vocab.json" ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910
    --bind sft1-adapter "$SFT1_ADAPTER/adapter_model.safetensors" "$EXPECTED_SFT1_SHA256"
    --bind sft1-config "$SFT1_ADAPTER/adapter_config.json" "$EXPECTED_SFT1_CONFIG_SHA256"
    --bind sft1-state "$SFT1_ADAPTER/trainer_state.json" "$EXPECTED_SFT1_STATE_SHA256"
  )
}

stage_plan_args() {
  stage_bind_args
  STAGE_PLAN_ARGS=(
    --stage "$STAGE"
    --run-root "$RUN_ROOT"
    --tasks "$STAGE_TASKS"
    --tasks-manifest "$STAGE_TASK_MANIFEST"
    --activation "$STAGE_ACTIVATION"
    --expected-activation-sha256 "$STAGE_ACTIVATION_SHA"
    --records "$STAGE_RECORDS"
    --group-size "$STAGE_GROUP_SIZE"
    --seed "$STAGE_SEED"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --shards "$STAGE_SHARDS"
    --gpu-map "$STAGE_GPU_MAP"
    --assignments-dir "$STAGE_ASSIGNMENTS"
    --plan "$STAGE_PLAN"
    --model-path "$MODEL_PATH"
    --adapter-path "$SFT1_ADAPTER"
    --protocol-runtime "$PROTOCOL_RUNTIME"
    "${STAGE_BIND_ARGS[@]}"
  )
}

verify_wide_cohort() {
  local expected=$1
  env PYTHONPATH= "$PYTHON_BIN" - "$F1_COHORT_MANIFEST" "$F1_TASKS" "$expected" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
manifest_path, tasks_path = map(Path, sys.argv[1:3]); expected = sys.argv[3]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert re.fullmatch(r"[0-9a-f]{64}", expected) and sha(manifest_path) == expected
for path in (manifest_path, tasks_path): assert path.is_file() and not path.is_symlink()
manifest = json.loads(manifest_path.read_text())
assert manifest["schema_version"] == "vanilla-grpo-arm-b-wide3000-cohort-v1"
assert manifest["status"] == "frozen_arm_b_wide_screen_cohort"
assert manifest["cohort_namespace"] == "qwen3-v26-arm-b-wide3000-v1"
assert manifest["seeds"] == {
    "wide_cohort_selection": "qwen3-v26-arm-b-wide3000-v1-20260812",
    "wide_k8_generation": 20260814,
    "confirmation640_selection": "qwen3-v26-arm-b-confirmation640-v1-20260812",
    "k16_confirmation_generation": 20260815,
    "selected384_selection": "qwen3-v26-arm-b-k16-selected384-v1-20260812",
    "validation64_generation": 20260816,
    "formal_training_data": 20260812,
}
assert manifest["all_acceptance_gates_passed"] is True and all(manifest["acceptance_gates"].values())
activation = manifest["activation"]
assert activation["decision"] == "arm_b_pretraining_fallback"
assert activation["reason"] in {"arm_a_s1_s2_readiness_failed", "arm_a_validation_failed"}
assert activation["arm_a_training_started"] is False and activation["arm_a_full_dev_started"] is False
pools = activation["actual_screen_pools"]
assert len(pools) in {1, 2}
for pool in pools:
    for key in ("audit", "tasks"):
        path = Path(pool[f"{key}_path"]); assert path.is_file() and not path.is_symlink()
        assert pool[f"{key}_sha256"] == sha(path)
for record in (manifest["reference_population"], manifest["eligibility_population"], *manifest["exclusions"]):
    path = Path(record["path"]); assert path.is_file() and not path.is_symlink()
    assert record["sha256"] == sha(path)
output = manifest["output"]
assert Path(output["path"]).resolve() == tasks_path.resolve()
assert output["sha256"] == sha(tasks_path) and output["records"] == 3000
rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
ids = [str(row.get("example_id") or row.get("instance_id")) for row in rows]
assert len(rows) == len(set(ids)) == 3000 and output["task_ids_in_frozen_order"] == ids
assert manifest["downstream_contract"] == {
    "screen_policy": "fresh initial-SFT1", "group_size": 8, "seed": 20260814,
    "minimum_clean_mixed_groups": 640,
    "contaminated_groups": "audited and excluded; no replacement sampling",
    "training_admission": "none until fresh K16 confirmation and validation64 pass",
}
PY
}

verify_f1_audit() {
  local expected=$1
  env PYTHONPATH= "$PYTHON_BIN" - "$F1_AUDIT" "$F1_COHORT_MANIFEST" "$F1_TASKS" \
    "$STAGE_POOL/manifest.pending.json" "$STAGE_POOL/trajectories.jsonl" "$F2_TASKS" "$expected" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
audit_path, cohort, tasks, manifest, trajectories, confirmation = map(Path, sys.argv[1:7]); expected=sys.argv[7]
sha=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert re.fullmatch(r"[0-9a-f]{64}", expected) and sha(audit_path)==expected
audit=json.loads(audit_path.read_text()); status=audit["status"]
assert audit["schema_version"]=="vanilla-grpo-arm-b-wide-k8-screen-audit-v1"
assert audit["issues"]==[] and audit["issue_counts"]=={} and status["audit_passes"] is True
assert status["next_stage"] == ("confirm_k16" if status["confirmation_ready"] else "stop_arm_b")
inputs=audit["inputs"]
for key,path in (("cohort_manifest",cohort),("generation_manifest",manifest),("tasks",tasks),("trajectories",trajectories)):
    assert Path(inputs[key]).resolve()==path.resolve() and inputs[f"{key}_sha256"]==sha(path)
observed=audit["observed"]
assert (observed["tasks"],observed["group_size"],observed["trajectories"])==(3000,8,24000)
ready=status["confirmation_ready"]
assert audit["acceptance_gates"]["at_least_640_clean_mixed_groups"] is ready
assert observed["mixed_boundary_groups"] >= 640 if ready else observed["mixed_boundary_groups"] < 640
if ready:
    assert confirmation.is_file() and not confirmation.is_symlink()
    output=audit["outputs"]["confirmation"]
    assert Path(output["path"]).resolve()==confirmation.resolve() and output["sha256"]==sha(confirmation) and output["records"]==640
    rows=[json.loads(line) for line in confirmation.read_text().splitlines() if line.strip()]
    ids=[str(row.get("example_id") or row.get("instance_id")) for row in rows]
    assert len(rows)==len(set(ids))==640 and audit["task_ids"]["confirmation"]==ids
else:
    assert audit["outputs"]["confirmation"] is None and audit["task_ids"]["confirmation"]==[]
    assert not confirmation.exists()
print("ready" if ready else "stop")
PY
}

verify_f2_selection() {
  local expected=$1
  env PYTHONPATH= "$PYTHON_BIN" - "$F2_SELECTION_MANIFEST" "$F2_AUDIT" "$F1_AUDIT" \
    "$F2_TASKS" "$STAGE_POOL/manifest.pending.json" "$STAGE_POOL/trajectories.jsonl" \
    "$F2_SELECTION_DIR/selected384.jsonl" "$F2_SELECTION_DIR/train320.jsonl" "$F3_TASKS" "$expected" <<'PY'
import hashlib,json,re,sys
from pathlib import Path
selection_path,audit_path,wide_audit,tasks,generation,trajectories,selected,train,validation=map(Path,sys.argv[1:10]); expected=sys.argv[10]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert re.fullmatch(r"[0-9a-f]{64}",expected) and sha(selection_path)==expected
for path in (selection_path,audit_path,wide_audit,tasks,generation,trajectories,selected,train,validation): assert path.is_file() and not path.is_symlink()
selection=json.loads(selection_path.read_text()); audit=json.loads(audit_path.read_text())
assert selection["schema_version"]=="vanilla-grpo-arm-b-k16-selection-v1" and selection["status"]=="frozen_arm_b_training_cohort"
assert selection["cohort_namespace"]=="qwen3-v26-arm-b-train320-k16-v1"
assert selection["seeds"]=={
    "wide_k8_generation":20260814,
    "k16_confirmation_generation":20260815,
    "k16_selection":"qwen3-v26-arm-b-k16-selected384-v1-20260812",
    "validation64_generation":20260816,
    "formal_training_data":20260812,
}
assert audit["schema_version"]=="vanilla-grpo-arm-b-k16-confirmation-audit-v1"
assert audit["status"]=={"audit_passes":True,"selection_ready":True,"next_stage":"validate_k16"}
assert audit["issues"]==[] and audit["issue_counts"]=={} and all(audit["acceptance_gates"].values())
inputs=audit["inputs"]
assert Path(inputs["wide_screen_audit"]["path"]).resolve()==wide_audit.resolve() and inputs["wide_screen_audit"]["sha256"]==sha(wide_audit)
for key,path in (("generation_manifest",generation),("tasks",tasks),("trajectories",trajectories)):
    assert Path(inputs[key]["path"]).resolve()==path.resolve() and inputs[key]["sha256"]==sha(path)
assert audit["observed"]["tasks"]==640 and audit["observed"]["trajectories"]==10240
assert audit["observed"]["core_boundary_groups"]>=384
paths={"selected":selected,"train":train,"validation":validation}; counts={"selected":384,"train":320,"validation":64}; observed={}
for name,path in paths.items():
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]; ids=[str(row.get("example_id") or row.get("instance_id")) for row in rows]
    assert len(rows)==len(set(ids))==counts[name]
    assert selection["outputs"][name]=={"path":str(path.resolve()),"sha256":sha(path),"records":counts[name]}
    assert selection["task_ids"][name]==ids; observed[name]=set(ids)
assert observed["train"].isdisjoint(observed["validation"]) and observed["train"]|observed["validation"]==observed["selected"]
assert selection["contract"]["primary_checkpoint"]=="final-step32-only"
assert selection["contract"]["formal_training"]["fresh_online_trajectories"]==10240
assert selection["selection"]["confirmation_generation_seed"]==20260815
assert selection["selection"]["selection_seed"]=="qwen3-v26-arm-b-k16-selected384-v1-20260812"
assert selection["contract"]["validation"]=={
    "policy":"fresh initial-SFT1","records":64,"group_size":16,"seed":20260816,
    "confirmation_seed_must_differ":20260815,"generation_seed_scheme":"sha256-task-sample-turn-v1",
    "exact_clean_trajectories":1024,"minimum_mixed_groups":56,"minimum_core_groups":48,
    "screen_trajectories_reused":False,
}
PY
}

verify_confirmatory_trigger() {
  local expected_validation_sha=$1 expected_train_sha=$2
  env PYTHONPATH= "$PYTHON_BIN" - "$CONFIRMATORY_TRIGGER" "$F1_COHORT_MANIFEST" \
    "$F2_SELECTION_MANIFEST" "$F2_SELECTION_DIR/train320.jsonl" "$F3_AUDIT" \
    "$FROZEN_F1_COHORT_MANIFEST_SHA256" "$FROZEN_F2_SELECTION_MANIFEST_SHA256" \
    "$expected_train_sha" "$expected_validation_sha" <<'PY'
import hashlib,json,re,sys
from pathlib import Path
trigger_path,cohort,selection,train,audit=map(Path,sys.argv[1:6]); cohort_sha,selection_sha,train_sha,audit_sha=sys.argv[6:10]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
for value in (cohort_sha,selection_sha,train_sha,audit_sha): assert re.fullmatch(r"[0-9a-f]{64}",value)
for path in (trigger_path,cohort,selection,train,audit): assert path.is_file() and not path.is_symlink()
assert (sha(cohort),sha(selection),sha(train),sha(audit))==(cohort_sha,selection_sha,train_sha,audit_sha)
trigger=json.loads(trigger_path.read_text())
assert trigger["schema_version"]=="vanilla-grpo-confirmatory-arm-trigger-v1"
assert trigger["status"]=="arm_b_admitted_before_training"
assert trigger["admitted_arm"]=="arm_b" and trigger["confirmatory_arm_count"]==1
assert trigger["full_dev_policy"]=="exactly-one-admitted-arm"
assert trigger["reason"] in {"arm_a_s1_s2_readiness_failed","arm_a_validation_failed"}
assert trigger["arm_a_training_started"] is False and trigger["arm_a_full_dev_started"] is False
assert trigger["arm_a_failure_evidence"]=={
    "path":str(cohort.resolve()),"sha256":cohort_sha,
    "schema_version":"vanilla-grpo-arm-b-wide3000-cohort-v1",
    "status":"frozen_arm_b_wide_screen_cohort",
}
assert trigger["arm_b_selection_manifest"]=={
    "path":str(selection.resolve()),"sha256":selection_sha,
    "schema_version":"vanilla-grpo-arm-b-k16-selection-v1",
    "status":"frozen_arm_b_training_cohort",
}
validation=trigger["arm_b_validation_audit"]
assert validation["path"]==str(audit.resolve()) and validation["sha256"]==audit_sha
assert validation["schema_version"]=="vanilla-grpo-arm-b-k16-validation-audit-v1"
assert validation["status"]=={"passes":True,"validation_admitted":True,"next_stage":"train_arm_b"}
print(sha(trigger_path))
PY
}

verify_stage_plan() {
  require_explicit_sha "$FROZEN_STAGE_PLAN_SHA" "$STAGE frozen plan"
  stage_plan_args
  env PYTHONPATH= "$PYTHON_BIN" "$STAGE_MANAGER" verify-plan "${STAGE_PLAN_ARGS[@]}" >/dev/null \
    || die "$STAGE plan contract verification failed"
  require_sha "$STAGE_PLAN" "$FROZEN_STAGE_PLAN_SHA" "$STAGE frozen plan"
}

build_arm_a_trigger_args() {
  [[ "$ARM_A_TRIGGER_MODE" == readiness || "$ARM_A_TRIGGER_MODE" == validation ]] \
    || die 'ARM_A_TRIGGER_MODE must be explicitly readiness or validation'
  [[ "$ARM_A_SCREEN_POOL_COUNT" =~ ^[12]$ ]] || die 'ARM_A_SCREEN_POOL_COUNT must be exactly 1 or 2'
  [[ "$ARM_A_TRIGGER_MODE" != readiness || "$ARM_A_SCREEN_POOL_COUNT" == 2 ]] \
    || die 'readiness fallback requires exactly the actual S1 and S2 screen pools'
  ARM_A_TRIGGER_ARGS=(--trigger-mode "$ARM_A_TRIGGER_MODE")
  local index audit_var audit_sha_var task_var task_sha_var audit audit_sha tasks tasks_sha
  for ((index=0; index<ARM_A_SCREEN_POOL_COUNT; index++)); do
    audit_var=ARM_A_SCREEN_AUDIT${index}; audit_sha_var=FROZEN_ARM_A_SCREEN_AUDIT${index}_SHA256
    task_var=ARM_A_SCREEN_TASKS${index}; task_sha_var=FROZEN_ARM_A_SCREEN_TASKS${index}_SHA256
    audit=${!audit_var:-}; audit_sha=${!audit_sha_var:-}; tasks=${!task_var:-}; tasks_sha=${!task_sha_var:-}
    require_sha "$audit" "$audit_sha" "Arm A screen audit $index"
    require_sha "$tasks" "$tasks_sha" "Arm A screen tasks $index"
    ARM_A_TRIGGER_ARGS+=(--screen-audit "$audit" --expected-screen-audit-sha256 "$audit_sha" --screen-tasks "$tasks" --expected-screen-tasks-sha256 "$tasks_sha")
  done
  if [[ "$ARM_A_TRIGGER_MODE" == validation ]]; then
    require_sha "$ARM_A_SELECTION_MANIFEST" "$FROZEN_ARM_A_SELECTION_MANIFEST_SHA256" 'Arm A selection manifest'
    require_sha "$ARM_A_VALIDATION_MANIFEST" "$FROZEN_ARM_A_VALIDATION_MANIFEST_SHA256" 'Arm A validation generation manifest'
    require_sha "$ARM_A_VALIDATION_TASKS" "$FROZEN_ARM_A_VALIDATION_TASKS_SHA256" 'Arm A validation tasks'
    require_sha "$ARM_A_VALIDATION_TRAJECTORIES" "$FROZEN_ARM_A_VALIDATION_TRAJECTORIES_SHA256" 'Arm A validation trajectories'
    ARM_A_TRIGGER_ARGS+=(
      --arm-a-selection-manifest "$ARM_A_SELECTION_MANIFEST"
      --expected-arm-a-selection-manifest-sha256 "$FROZEN_ARM_A_SELECTION_MANIFEST_SHA256"
      --arm-a-validation-manifest "$ARM_A_VALIDATION_MANIFEST"
      --expected-arm-a-validation-manifest-sha256 "$FROZEN_ARM_A_VALIDATION_MANIFEST_SHA256"
      --arm-a-validation-tasks "$ARM_A_VALIDATION_TASKS"
      --expected-arm-a-validation-tasks-sha256 "$FROZEN_ARM_A_VALIDATION_TASKS_SHA256"
      --arm-a-validation-trajectories "$ARM_A_VALIDATION_TRAJECTORIES"
      --expected-arm-a-validation-trajectories-sha256 "$FROZEN_ARM_A_VALIDATION_TRAJECTORIES_SHA256"
    )
  fi
}

prepare_f1() {
  require_common_assets
  stage_init F1
  require_sha "$WIDE_PREPARER" "$EXPECTED_WIDE_PREPARER_SHA256" wide3000_preparer
  require_sha "$WIDE_AUDITOR" "$EXPECTED_WIDE_AUDITOR_SHA256" wide_k8_auditor
  require_sha "$REFERENCE_TASKS" "$EXPECTED_REFERENCE_SHA256" reference6601
  require_sha "$ELIGIBLE_TASKS" "$EXPECTED_ELIGIBLE_SHA256" eligible5915
  require_sha "$BASELINE_EVAL300" "$EXPECTED_BASELINE_EVAL300_SHA256" baseline_eval300
  [[ -n "$SFT1_INDEX" && -n "$OLD_MIXED60" ]] || die 'SFT1_INDEX and OLD_MIXED60 must be explicit persistent input paths'
  require_sha "$SFT1_INDEX" "$EXPECTED_SFT1_INDEX_SHA256" sft1_index
  require_sha "$OLD_MIXED60" "$EXPECTED_OLD_MIXED60_SHA256" old_mixed60
  require_sha "$TRAIN_RUNTIME/src/rl/diagnostics/audit_vanilla_grpo_boundary_validation.py" "$EXPECTED_ARM_A_VALIDATION_AUDITOR_SHA256" arm_a_validation_auditor_dependency
  require_sha "$TRAIN_RUNTIME/src/rl/select_policy_boundary_grpo_tasks.py" "$EXPECTED_ARM_A_SELECTOR_SHA256" arm_a_selector_dependency
  require_sha "$PUBLIC_SELECTOR_ROOT/src/eval/select_bird_train_baseline.py" "$EXPECTED_PUBLIC_SELECTOR_SHA256" public_selector_dependency
  build_arm_a_trigger_args
  mkdir -p "$INPUT_DIR" "$STAGE_LOCKS" "$STAGE_STATUS" "$STAGE_LOGS"
  exec 9>"$STAGE_LOCKS/control.lock"; flock -n 9 || die 'another invocation owns F1 prepare/finalize control'
  write_status "$STAGE_STATUS/prepare.status" preparing "trigger=$ARM_A_TRIGGER_MODE actual_pools=$ARM_A_SCREEN_POOL_COUNT"
  env PYTHONPATH="$TRAIN_RUNTIME:$PUBLIC_SELECTOR_ROOT" "$PYTHON_BIN" "$WIDE_PREPARER" \
    "${ARM_A_TRIGGER_ARGS[@]}" \
    --reference "$REFERENCE_TASKS" --eligible "$ELIGIBLE_TASKS" \
    --baseline-eval300 "$BASELINE_EVAL300" --sft1-index "$SFT1_INDEX" \
    --old-mixed60 "$OLD_MIXED60" --output "$F1_TASKS" \
    --manifest "$F1_COHORT_MANIFEST" --remote-db-root "$REMOTE_DB_ROOT" \
    >>"$STAGE_LOGS/prepare.log" 2>&1 \
    || die "wide3000 preparer failed; see $STAGE_LOGS/prepare.log"
  STAGE_ACTIVATION_SHA=$(sha256_file "$F1_COHORT_MANIFEST")
  verify_wide_cohort "$STAGE_ACTIVATION_SHA" || die 'wide3000 cohort verification failed'
  stage_plan_args
  env PYTHONPATH= "$PYTHON_BIN" "$STAGE_MANAGER" prepare-plan "${STAGE_PLAN_ARGS[@]}" \
    >>"$STAGE_LOGS/prepare.log" 2>&1 || die 'F1 stage plan preparation failed'
  local plan_sha; plan_sha=$(sha256_file "$STAGE_PLAN")
  write_status "$STAGE_STATUS/prepare.status" complete "cohort_sha256=$STAGE_ACTIVATION_SHA plan_sha256=$plan_sha"
  printf 'prepared F1: FROZEN_F1_COHORT_MANIFEST_SHA256=%s FROZEN_F1_PLAN_SHA256=%s\n' "$STAGE_ACTIVATION_SHA" "$plan_sha"
}

prepare_f2() {
  require_common_assets
  stage_init F2
  require_explicit_sha "$FROZEN_F1_COHORT_MANIFEST_SHA256" FROZEN_F1_COHORT_MANIFEST_SHA256
  require_explicit_sha "$FROZEN_F1_AUDIT_SHA256" FROZEN_F1_AUDIT_SHA256
  verify_wide_cohort "$FROZEN_F1_COHORT_MANIFEST_SHA256" || die 'source wide3000 cohort changed'
  require_sha "$WIDE_AUDITOR" "$EXPECTED_WIDE_AUDITOR_SHA256" wide_k8_auditor
  require_sha "$K16_SELECTOR" "$EXPECTED_K16_SELECTOR_SHA256" k16_selector
  # Point the verifier at F1's canonical pool, not F2's not-yet-created pool.
  local STAGE_POOL=$F1_ROOT/pool
  [[ "$(verify_f1_audit "$FROZEN_F1_AUDIT_SHA256")" == ready ]] \
    || die 'F1 did not admit the frozen confirmation640 cohort'
  mkdir -p "$STAGE_LOCKS" "$STAGE_STATUS" "$STAGE_LOGS"
  exec 9>"$STAGE_LOCKS/control.lock"; flock -n 9 || die 'another invocation owns F2 prepare/finalize control'
  stage_plan_args
  env PYTHONPATH= "$PYTHON_BIN" "$STAGE_MANAGER" prepare-plan "${STAGE_PLAN_ARGS[@]}" \
    >>"$STAGE_LOGS/prepare.log" 2>&1 || die 'F2 stage plan preparation failed'
  local plan_sha; plan_sha=$(sha256_file "$STAGE_PLAN")
  write_status "$STAGE_STATUS/prepare.status" complete "source_f1_audit_sha256=$FROZEN_F1_AUDIT_SHA256 plan_sha256=$plan_sha"
  printf 'prepared F2: FROZEN_F2_PLAN_SHA256=%s\n' "$plan_sha"
}

prepare_f3() {
  require_common_assets
  stage_init F3
  require_explicit_sha "$FROZEN_F1_COHORT_MANIFEST_SHA256" FROZEN_F1_COHORT_MANIFEST_SHA256
  require_explicit_sha "$FROZEN_F2_SELECTION_MANIFEST_SHA256" FROZEN_F2_SELECTION_MANIFEST_SHA256
  verify_wide_cohort "$FROZEN_F1_COHORT_MANIFEST_SHA256" || die 'source wide3000 cohort changed'
  require_sha "$K16_VALIDATION_AUDITOR" "$EXPECTED_K16_VALIDATION_AUDITOR_SHA256" k16_validation_auditor
  local STAGE_POOL=$F2_ROOT/pool
  verify_f2_selection "$FROZEN_F2_SELECTION_MANIFEST_SHA256" \
    || die 'F2 selected384/train320/validation64 binding failed'
  mkdir -p "$STAGE_LOCKS" "$STAGE_STATUS" "$STAGE_LOGS"
  exec 9>"$STAGE_LOCKS/control.lock"; flock -n 9 || die 'another invocation owns F3 prepare/finalize control'
  stage_plan_args
  env PYTHONPATH= "$PYTHON_BIN" "$STAGE_MANAGER" prepare-plan "${STAGE_PLAN_ARGS[@]}" \
    >>"$STAGE_LOGS/prepare.log" 2>&1 || die 'F3 stage plan preparation failed'
  local plan_sha; plan_sha=$(sha256_file "$STAGE_PLAN")
  write_status "$STAGE_STATUS/prepare.status" complete "source_selection_sha256=$FROZEN_F2_SELECTION_MANIFEST_SHA256 plan_sha256=$plan_sha"
  printf 'prepared F3: FROZEN_F3_PLAN_SHA256=%s\n' "$plan_sha"
}

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d '[:space:]'
}

worker_pgid=''
terminate_owned_worker() {
  [[ -n "$worker_pgid" ]] || return 0
  if kill -0 -- "-$worker_pgid" 2>/dev/null; then
    kill -TERM -- "-$worker_pgid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 -- "-$worker_pgid" 2>/dev/null || break
      sleep 1
    done
    kill -0 -- "-$worker_pgid" 2>/dev/null && kill -KILL -- "-$worker_pgid" 2>/dev/null || true
  fi
  wait "$worker_pgid" 2>/dev/null || true
  worker_pgid=''
}
on_worker_signal() { terminate_owned_worker; exit "$1"; }

run_worker() {
  local requested_stage=$1
  require_common_assets
  stage_init "$requested_stage"
  require_explicit_sha "$FROZEN_F1_COHORT_MANIFEST_SHA256" FROZEN_F1_COHORT_MANIFEST_SHA256
  verify_wide_cohort "$FROZEN_F1_COHORT_MANIFEST_SHA256" || die 'source wide3000 cohort changed before worker'
  require_sha "$STAGE_HOOK" "$STAGE_HOOK_SHA" "$STAGE hook"
  verify_stage_plan
  [[ "$SHARD_INDEX" =~ ^[0-9]+$ && "$SHARD_INDEX" -lt "$STAGE_SHARDS" ]] \
    || die "SHARD_INDEX must be explicit in [0,$((STAGE_SHARDS-1))]"
  IFS=',' read -r -a parsed_gpu_map <<<"$STAGE_GPU_MAP"
  local gpu=${parsed_gpu_map[$SHARD_INDEX]}
  local shard_name assignment worker completion status log used unexpected
  shard_name=$(printf 'shard-%03d-of-%03d' "$SHARD_INDEX" "$STAGE_SHARDS")
  assignment=$STAGE_ASSIGNMENTS/$STAGE_KEY.$shard_name.txt
  worker=$STAGE_WORKERS/$shard_name
  completion=$worker/worker_complete.json
  status=$STAGE_STATUS/worker-$shard_name.status
  log=$STAGE_LOGS/worker-$shard_name.log
  mkdir -p "$worker/groups" "$STAGE_LOCKS" "$STAGE_LOGS" "$STAGE_STATUS"
  exec 8>"$STAGE_LOCKS/worker-$shard_name.lock"; flock -n 8 || die "another invocation owns $STAGE $shard_name"
  mkdir -p "$RUN_ROOT/locks"
  exec 7>"$RUN_ROOT/locks/physical-gpu-$gpu.lock"; flock -n 7 || die "another Arm B shard owns GPU$gpu"
  "$PYTHON_BIN" "$GROUP_RESUME_HELPER" --tasks "$STAGE_TASKS" \
    --task-id-file "$assignment" --groups-dir "$worker/groups" \
    --quarantine-dir "$STAGE_QUARANTINE/$shard_name/groups" >>"$log" 2>&1 \
    || die "$STAGE $shard_name partial-group resume preparation failed"
  if [[ -f "$completion" ]]; then
    "$PYTHON_BIN" "$STAGE_MANAGER" verify-worker --plan "$STAGE_PLAN" \
      --expected-plan-sha256 "$FROZEN_STAGE_PLAN_SHA" --shard-index "$SHARD_INDEX" \
      --worker-dir "$worker" --completion "$completion" \
      --quarantine-dir "$STAGE_QUARANTINE/$shard_name/markers" >/dev/null \
      || die "existing $STAGE $shard_name completion failed verification"
    write_status "$status" complete "existing_verified=$completion"
    return
  fi
  unexpected=$(find "$worker" -mindepth 1 -maxdepth 1 ! -name groups -print -quit)
  [[ -z "$unexpected" ]] || die "unknown worker artifact blocks $STAGE resume: $unexpected"
  used=$(gpu_used_mib "$gpu")
  [[ "$used" =~ ^[0-9]+$ ]] || die "cannot query explicit GPU$gpu"
  [[ "$used" -le "$GPU_IDLE_MAX_MIB" ]] \
    || die "GPU$gpu is not idle: used_mib=$used threshold=$GPU_IDLE_MAX_MIB; no process was stopped"
  write_status "$status" generating "stage=$STAGE shard=$SHARD_INDEX gpu=$gpu K=$STAGE_GROUP_SIZE resume_groups=true"
  trap 'on_worker_signal 130' INT
  trap 'on_worker_signal 143' TERM
  trap terminate_owned_worker EXIT
  setsid env CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 \
    TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$PROTOCOL_RUNTIME" \
    PYTHONPATH="$TRAIN_RUNTIME/src/rl" \
    "$PYTHON_BIN" "$TRAIN_RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py" \
      --model-path "$MODEL_PATH" --adapter-path "$SFT1_ADAPTER" \
      --tasks "$STAGE_TASKS" --output-dir "$worker" --task-id-file "$assignment" \
      --no-finalize --group-size "$STAGE_GROUP_SIZE" --temperature 0.8 --top-p 1 \
      --max-steps 30 --max-new-tokens 2048 --max-context-tokens 16384 \
      --history-turns 4 --enable-thinking --seed "$STAGE_SEED" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --scheduler static --task-batch-size 1 >>"$log" 2>&1 &
  worker_pgid=$!
  local code=0
  if wait "$worker_pgid"; then code=0; else code=$?; fi
  terminate_owned_worker; trap - EXIT INT TERM
  [[ "$code" -eq 0 ]] || { write_status "$status" failed "exit=$code log=$log"; return "$code"; }
  "$PYTHON_BIN" "$STAGE_MANAGER" complete-worker --plan "$STAGE_PLAN" \
    --expected-plan-sha256 "$FROZEN_STAGE_PLAN_SHA" --shard-index "$SHARD_INDEX" \
    --worker-dir "$worker" --completion "$completion" \
    --quarantine-dir "$STAGE_QUARANTINE/$shard_name/markers" >>"$log" 2>&1 \
    || die "$STAGE $shard_name completion validation/publication failed"
  write_status "$status" complete "stage=$STAGE gpu=$gpu completion=$completion"
}

assemble_and_publish_pool() {
  "$PYTHON_BIN" "$STAGE_MANAGER" prepare-pool-transaction --plan "$STAGE_PLAN" \
    --expected-plan-sha256 "$FROZEN_STAGE_PLAN_SHA" --workers-root "$STAGE_WORKERS" \
    --transaction "$STAGE_TRANSACTION" --pool "$STAGE_POOL" \
    --quarantine-dir "$STAGE_QUARANTINE/pool-transactions" \
    >>"$STAGE_LOGS/finalize.log" 2>&1 || die "$STAGE pool transaction preparation failed"
  if [[ ! -d "$STAGE_POOL" ]]; then
    env HF_HUB_OFFLINE=1 TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$PROTOCOL_RUNTIME" \
      PYTHONPATH="$TRAIN_RUNTIME/src/rl" \
      "$PYTHON_BIN" "$TRAIN_RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py" \
        --model-path "$MODEL_PATH" --adapter-path "$SFT1_ADAPTER" \
        --tasks "$STAGE_TASKS" --output-dir "$STAGE_TRANSACTION" \
        --group-size "$STAGE_GROUP_SIZE" --temperature 0.8 --top-p 1 \
        --max-steps 30 --max-new-tokens 2048 --max-context-tokens 16384 \
        --history-turns 4 --enable-thinking --seed "$STAGE_SEED" \
        --scheduler static --task-batch-size 1 --finalize-only \
        >>"$STAGE_LOGS/finalize.log" 2>&1 \
      || die "$STAGE generator --finalize-only failed inside isolated transaction"
    "$PYTHON_BIN" "$STAGE_MANAGER" verify-pool --plan "$STAGE_PLAN" \
      --expected-plan-sha256 "$FROZEN_STAGE_PLAN_SHA" --workers-root "$STAGE_WORKERS" \
      --pool "$STAGE_TRANSACTION" >>"$STAGE_LOGS/finalize.log" 2>&1 \
      || die "$STAGE isolated pool transaction failed verification"
    "$PYTHON_BIN" "$STAGE_MANAGER" publish-pool --plan "$STAGE_PLAN" \
      --expected-plan-sha256 "$FROZEN_STAGE_PLAN_SHA" --workers-root "$STAGE_WORKERS" \
      --transaction "$STAGE_TRANSACTION" --pool "$STAGE_POOL" \
      >>"$STAGE_LOGS/finalize.log" 2>&1 \
      || die "$STAGE canonical pool publication failed"
  else
    "$PYTHON_BIN" "$STAGE_MANAGER" verify-pool --plan "$STAGE_PLAN" \
      --expected-plan-sha256 "$FROZEN_STAGE_PLAN_SHA" --workers-root "$STAGE_WORKERS" \
      --pool "$STAGE_POOL" >>"$STAGE_LOGS/finalize.log" 2>&1 \
      || die "$STAGE existing canonical pool failed verification"
  fi
}

run_f1_audit() {
  mkdir -p "$F1_AUDIT_DIR"
  local code=0
  set +e
  env PYTHONPATH="$TRAIN_RUNTIME" "$PYTHON_BIN" "$WIDE_AUDITOR" \
    --cohort-manifest "$F1_COHORT_MANIFEST" \
    --expected-cohort-manifest-sha256 "$FROZEN_F1_COHORT_MANIFEST_SHA256" \
    --manifest "$STAGE_POOL/manifest.pending.json" \
    --trajectories "$STAGE_POOL/trajectories.jsonl" --tasks "$F1_TASKS" \
    --output-dir "$F1_AUDIT_DIR" >>"$STAGE_LOGS/audit.log" 2>&1
  code=$?
  set -e
  [[ "$code" -eq 0 || "$code" -eq 1 ]] \
    || die "F1 structural audit failed; see $STAGE_LOGS/audit.log"
  local audit_sha decision; audit_sha=$(sha256_file "$F1_AUDIT")
  decision=$(verify_f1_audit "$audit_sha") || die 'F1 audit/output verification failed'
  if [[ "$decision" == stop ]]; then
    write_status "$STAGE_STATUS/finalize.status" stop_arm_b "F1 clean mixed groups <640 audit_sha256=$audit_sha no_resample=true"
    printf 'stop_arm_b: F1 gate failed; no F2 was started; FROZEN_F1_AUDIT_SHA256=%s\n' "$audit_sha"
    return 4
  fi
  [[ "$code" -eq 0 ]] || die 'F1 auditor return code/status disagree'
  write_status "$STAGE_STATUS/finalize.status" complete "confirmation640=$F2_TASKS audit_sha256=$audit_sha"
  printf 'F1 complete: FROZEN_F1_AUDIT_SHA256=%s\n' "$audit_sha"
}

run_f2_selector() {
  mkdir -p "$F2_SELECTION_DIR"
  local code=0
  set +e
  env PYTHONPATH="$TRAIN_RUNTIME" "$PYTHON_BIN" "$K16_SELECTOR" \
    --wide-screen-audit "$F1_AUDIT" \
    --expected-wide-screen-audit-sha256 "$FROZEN_F1_AUDIT_SHA256" \
    --manifest "$STAGE_POOL/manifest.pending.json" \
    --trajectories "$STAGE_POOL/trajectories.jsonl" --tasks "$F2_TASKS" \
    --output-dir "$F2_SELECTION_DIR" >>"$STAGE_LOGS/selection.log" 2>&1
  code=$?
  set -e
  [[ "$code" -eq 0 || "$code" -eq 1 ]] \
    || die "F2 structural audit/selector failed; see $STAGE_LOGS/selection.log"
  require_sha "$F2_AUDIT" "$(sha256_file "$F2_AUDIT")" F2_confirmation_audit
  if [[ "$code" -eq 1 ]]; then
    env PYTHONPATH= "$PYTHON_BIN" - "$F2_AUDIT" "$F1_AUDIT" "$F2_TASKS" \
      "$STAGE_POOL/manifest.pending.json" "$STAGE_POOL/trajectories.jsonl" <<'PY'
import hashlib,json,sys
from pathlib import Path
audit_path,wide,tasks,manifest,trajectories=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest(); audit=json.loads(audit_path.read_text())
assert audit["schema_version"]=="vanilla-grpo-arm-b-k16-confirmation-audit-v1"
assert audit["status"]=={"audit_passes":True,"selection_ready":False,"next_stage":"stop_arm_b"}
assert audit["issues"]==[] and audit["issue_counts"]=={}
assert audit["observed"]["tasks"]==640 and audit["observed"]["trajectories"]==10240
assert audit["observed"]["core_boundary_groups"]<384
for key,path in (("wide_screen_audit",wide),("generation_manifest",manifest),("tasks",tasks),("trajectories",trajectories)):
    record=audit["inputs"][key]; assert Path(record["path"]).resolve()==path.resolve() and record["sha256"]==sha(path)
assert audit["outputs"]=={} and all(not values for values in audit["task_ids"].values())
PY
    [[ ! -e "$F2_SELECTION_MANIFEST" && ! -e "$F2_SELECTION_DIR/selected384.jsonl" \
      && ! -e "$F2_SELECTION_DIR/train320.jsonl" && ! -e "$F3_TASKS" ]] \
      || die 'F2 failed gate left forbidden selection/training artifacts'
    local audit_sha; audit_sha=$(sha256_file "$F2_AUDIT")
    write_status "$STAGE_STATUS/finalize.status" stop_arm_b "F2 clean robust core <384 audit_sha256=$audit_sha no_resample=true"
    printf 'stop_arm_b: F2 gate failed; no F3/training was started; audit_sha256=%s\n' "$audit_sha"
    return 4
  fi
  local selection_sha; selection_sha=$(sha256_file "$F2_SELECTION_MANIFEST")
  verify_f2_selection "$selection_sha" || die 'F2 selected384/train320/validation64 verification failed'
  write_status "$STAGE_STATUS/finalize.status" complete "selection_sha256=$selection_sha validation64=$F3_TASKS"
  printf 'F2 complete: FROZEN_F2_SELECTION_MANIFEST_SHA256=%s\n' "$selection_sha"
}

run_f3_audit() {
  mkdir -p "$F3_AUDIT_DIR"
  local code=0
  set +e
  env PYTHONPATH="$TRAIN_RUNTIME" "$PYTHON_BIN" "$K16_VALIDATION_AUDITOR" \
    --selection-manifest "$F2_SELECTION_MANIFEST" \
    --expected-selection-manifest-sha256 "$FROZEN_F2_SELECTION_MANIFEST_SHA256" \
    --manifest "$STAGE_POOL/manifest.pending.json" \
    --trajectories "$STAGE_POOL/trajectories.jsonl" --tasks "$F3_TASKS" \
    --output "$F3_AUDIT" >>"$STAGE_LOGS/audit.log" 2>&1
  code=$?
  set -e
  [[ "$code" -eq 0 || "$code" -eq 1 ]] \
    || die "F3 structural validation audit failed; see $STAGE_LOGS/audit.log"
  env PYTHONPATH= "$PYTHON_BIN" - "$F3_AUDIT" "$F2_SELECTION_MANIFEST" "$F3_TASKS" \
    "$STAGE_POOL/manifest.pending.json" "$STAGE_POOL/trajectories.jsonl" <<'PY'
import hashlib,json,sys
from pathlib import Path
audit_path,selection,tasks,manifest,trajectories=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest(); audit=json.loads(audit_path.read_text())
assert audit["schema_version"]=="vanilla-grpo-arm-b-k16-validation-audit-v1"
for key,path in (("selection_manifest",selection),("generation_manifest",manifest),("tasks",tasks),("trajectories",trajectories)):
    value=audit["inputs"][key]
    if isinstance(value,dict): assert Path(value["path"]).resolve()==path.resolve() and value["sha256"]==sha(path)
    else: assert Path(value).resolve()==path.resolve() and audit["inputs"][f"{key}_sha256"]==sha(path)
observed=audit["observed"]
assert (observed["tasks"],observed["group_size"],observed["trajectories"])==(64,16,1024)
assert observed["groups"]==64 and observed["eligible_trajectories"]<=1024
assert observed["mixed_outcome_groups"]==observed["mixed_boundary_groups"]
assert observed["core_outcome_groups"]==observed["core_boundary_groups"]
assert audit["status"]["passes"] is audit["status"]["validation_admitted"]
if audit["status"]["validation_admitted"]:
    assert observed["usable_groups"]==64 and observed["contaminated_groups"]==0
    assert observed["eligible_trajectories"]==1024 and observed["mixed_boundary_groups"]>=56 and observed["core_boundary_groups"]>=48
    assert audit["status"]["next_stage"]=="train_arm_b" and all(audit["checks"].values())
else:
    assert audit["status"]["next_stage"]=="stop_arm_b"
PY
  local audit_sha; audit_sha=$(sha256_file "$F3_AUDIT")
  if [[ "$code" -eq 1 ]]; then
    [[ ! -e "$CONFIRMATORY_TRIGGER" ]] \
      || die 'failed F3 gate left a forbidden confirmatory trigger artifact'
    write_status "$STAGE_STATUS/finalize.status" stop_arm_b "F3 validation gate failed audit_sha256=$audit_sha no_replacement=true"
    printf 'stop_arm_b: F3 validation64 failed; no training was started; audit_sha256=%s\n' "$audit_sha"
    return 4
  fi
  local train320=$F2_SELECTION_DIR/train320.jsonl train_sha trigger_sha
  require_sha "$train320" "$(sha256_file "$train320")" arm_b_train320
  train_sha=$(sha256_file "$train320")
  env PYTHONPATH="$TRAIN_RUNTIME" "$PYTHON_BIN" "$CONFIRMATORY_TRIGGER_PREPARER" \
    --source-cohort "$F1_COHORT_MANIFEST" \
    --expected-source-cohort-sha256 "$FROZEN_F1_COHORT_MANIFEST_SHA256" \
    --selection-manifest "$F2_SELECTION_MANIFEST" \
    --expected-selection-manifest-sha256 "$FROZEN_F2_SELECTION_MANIFEST_SHA256" \
    --train320 "$train320" --expected-train320-sha256 "$train_sha" \
    --validation64-audit "$F3_AUDIT" --expected-validation64-audit-sha256 "$audit_sha" \
    --output "$CONFIRMATORY_TRIGGER" >>"$STAGE_LOGS/trigger.log" 2>&1 \
    || die "canonical confirmatory trigger preparation failed; see $STAGE_LOGS/trigger.log"
  trigger_sha=$(verify_confirmatory_trigger "$audit_sha" "$train_sha") \
    || die 'canonical confirmatory trigger verification failed'
  write_status "$STAGE_STATUS/finalize.status" complete \
    "F3_audit_sha256=$audit_sha confirmatory_trigger_sha256=$trigger_sha formal_training_still_not_started=true"
  printf 'F3 complete: FROZEN_VALIDATION64_AUDIT_SHA256=%s FROZEN_CONFIRMATORY_TRIGGER_SHA256=%s; run the separate Arm B training launcher explicitly\n' \
    "$audit_sha" "$trigger_sha"
}

finalize_stage() {
  local requested_stage=$1
  require_common_assets
  stage_init "$requested_stage"
  require_explicit_sha "$FROZEN_F1_COHORT_MANIFEST_SHA256" FROZEN_F1_COHORT_MANIFEST_SHA256
  verify_wide_cohort "$FROZEN_F1_COHORT_MANIFEST_SHA256" || die 'source wide3000 cohort changed before finalize'
  require_sha "$STAGE_HOOK" "$STAGE_HOOK_SHA" "$STAGE hook"
  verify_stage_plan
  mkdir -p "$STAGE_LOCKS" "$STAGE_STATUS" "$STAGE_LOGS"
  exec 9>"$STAGE_LOCKS/control.lock"; flock -n 9 || die "another invocation owns $STAGE prepare/finalize control"
  write_status "$STAGE_STATUS/finalize.status" assembling 'all shards must be complete; canonical pool absent until transaction verifies'
  assemble_and_publish_pool
  case "$STAGE" in
    F1) run_f1_audit ;;
    F2)
      require_explicit_sha "$FROZEN_F1_AUDIT_SHA256" FROZEN_F1_AUDIT_SHA256
      run_f2_selector
      ;;
    F3)
      require_explicit_sha "$FROZEN_F2_SELECTION_MANIFEST_SHA256" FROZEN_F2_SELECTION_MANIFEST_SHA256
      run_f3_audit
      ;;
  esac
}

dry_run() {
  printf '%s\n' \
    'Arm B staged screen dry-run (no files written, no GPU inspected)' \
    'trigger=only recomputed Arm A pre-training readiness or fresh-validation termination; Arm A train/full-dev canonical roots must remain absent' \
    'F1=wide3000 fresh initial-SFT1 K8 seed20260814; require >=640 whole-group-clean mixed; no replacement' \
    'F2=confirmation640 fresh initial-SFT1 K16 seed20260815; require >=384 whole-group-clean c=2..14; freeze 384 -> train320 + validation64' \
    'F3=validation64 fresh initial-SFT1 K16 seed20260816; require 1024/1024 clean, mixed>=56, core>=48; no replacement' \
    'lifecycle=prepare-f1 -> worker-f1 shards -> finalize-f1 -> explicit prepare-f2 -> worker-f2 shards -> finalize-f2 -> explicit prepare-f3 -> worker-f3 shards -> finalize-f3' \
    'workers=explicit per-stage comma-separated GPU map + SHARD_INDEX; idle check; only spawned PGID can receive a signal' \
    'publication=atomic groups -> worker completion -> isolated finalize transaction -> verified canonical pool -> external audit/selector' \
    "run_root=$RUN_ROOT" \
    'active prepare-f1 additionally requires SFT1_INDEX, OLD_MIXED60, trigger mode, actual Arm A screen path/SHA tuples, and explicit GPU map'
}

case "$MODE" in
  dry-run) dry_run ;;
  prepare-f1) prepare_f1 ;;
  worker-f1) run_worker F1 ;;
  finalize-f1) finalize_stage F1 ;;
  prepare-f2) prepare_f2 ;;
  worker-f2) run_worker F2 ;;
  finalize-f2) finalize_stage F2 ;;
  prepare-f3) prepare_f3 ;;
  worker-f3) run_worker F3 ;;
  finalize-f3) finalize_stage F3 ;;
esac
