#!/usr/bin/env bash
# Fail-closed validation + training template for the Qwen3-8B v26 boundary300
# vanilla binary-GRPO baseline.  The default is deliberately read-only.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODE=${1:-dry-run}
if [[ $# -gt 1 ]]; then
  printf 'usage: %s [dry-run|validate|train]\n' "$0" >&2
  exit 2
fi
case "$MODE" in
  dry-run|validate|train) ;;
  *) printf 'usage: %s [dry-run|validate|train]\n' "$0" >&2; exit 2 ;;
esac

VALIDATION_SEED=20260813
TRAIN_SEED=20260812
if [[ "$MODE" == dry-run ]]; then
  printf '%s\n' \
    'boundary300 vanilla-GRPO dry-run: no files written, no GPU inspected' \
    'active lifecycle: validate -> train' \
    'validate: fresh validation32, initial-SFT1, K=8, seed=20260813, >=20 mixed, zero runtime contamination' \
    'train: train300, 20 x 30 prompts, K=8, exactly two full passes, final step20 only' \
    'required for active modes: FROZEN_BOUNDARY_MANIFEST_SHA256=<64 lowercase hex>'
  exit 0
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
TRAIN_RUNTIME=${TRAIN_RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_boundary300_grpo_20260812}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
SFT1_ADAPTER=${SFT1_ADAPTER:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}

BOUNDARY_SELECTION_DIR=${BOUNDARY_SELECTION_DIR:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_boundary_screen_s1_2gpu_20260812/boundary_selection}
BOUNDARY_MANIFEST=${BOUNDARY_MANIFEST:-$BOUNDARY_SELECTION_DIR/boundary_cohort_manifest.json}
BOUNDARY_VERIFICATION=${BOUNDARY_VERIFICATION:-$BOUNDARY_SELECTION_DIR/boundary_selection_verification.json}
BOUNDARY_TASKS=${BOUNDARY_TASKS:-$BOUNDARY_SELECTION_DIR/boundary332.jsonl}
TRAIN_TASKS=${TRAIN_TASKS:-$BOUNDARY_SELECTION_DIR/train300.jsonl}
VALIDATION_TASKS=${VALIDATION_TASKS:-$BOUNDARY_SELECTION_DIR/validation32.jsonl}
FROZEN_BOUNDARY_MANIFEST_SHA256=${FROZEN_BOUNDARY_MANIFEST_SHA256:-}

EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-$TRAIN_RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_vanilla_grpo_boundary300.yaml}
VALIDATION_AUDITOR=${VALIDATION_AUDITOR:-$TRAIN_RUNTIME/src/rl/diagnostics/audit_vanilla_grpo_boundary_validation.py}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_boundary300_vanilla_grpo_20260812}
VALIDATION_OUT=${VALIDATION_OUT:-$RUN_ROOT/validation32_k8_seed${VALIDATION_SEED}}
TRAIN_OUT=${TRAIN_OUT:-$RUN_ROOT/train300_two_pass_seed${TRAIN_SEED}}
VALIDATION_MANIFEST=$VALIDATION_OUT/manifest.pending.json
VALIDATION_TRAJECTORIES=$VALIDATION_OUT/trajectories.jsonl
VALIDATION_AUDIT=$VALIDATION_OUT/boundary_validation_audit.json
STATUS=$RUN_ROOT/status/${MODE}.status
RUN_LOG=$RUN_ROOT/logs/${MODE}.log
VALIDATION_LOG=$RUN_ROOT/logs/validation_generation.log
TRAIN_LOG=$RUN_ROOT/logs/train_worker.log
VLLM_LOG=$RUN_ROOT/logs/train_vllm.log
LOCK=$RUN_ROOT/launcher.lock

VALIDATION_GPU=${VALIDATION_GPU:-1}
TRAIN_GPU=${TRAIN_GPU:-0}
VLLM_GPU=${VLLM_GPU:-1}
VLLM_PORT=${VLLM_PORT:-8077}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51277}

EXPECTED_PROTOCOL_VERSION=version26
EXPECTED_PROTOCOL_HASH=4da19387399bd3a5
EXPECTED_STUDENT_PROMPT_SHA256=848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316
EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256=5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab
EXPECTED_SFT1_SHA256=3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5
EXPECTED_SFT1_CONFIG_SHA256=537dbc946a7131d59e294a8729ace4aa145d73e59f4cf8556360ea507ce4435a
EXPECTED_SFT1_STATE_SHA256=97e529475d8c4f68dc88bc1f80370dacab478a681629f5f99003c39c70e9600a
EXPECTED_CONFIG_SHA256=f03674d0fe693442df4f60cf948279b1bc46c3510a43a9320e77c8200b4923c7
EXPECTED_VALIDATION_AUDITOR_SHA256=b855786dbeabf42633fe6775a436c1eb29e8e79d9bfa1c326453fcdc8ab67489
EXPECTED_BASE_PROBE_AUDITOR_SHA256=d192753faf1d579db616840e3ccdcfade855180a5c08144e1b5fe495e7eb85a5
EXPECTED_BOUNDARY_SCREEN_AUDITOR_SHA256=b983bd5d8083de49328fb25b98fba6105b2ffc773faceb7a5d0c8eea7c9181e6
EXPECTED_BOUNDARY_SELECTOR_SHA256=2849de8109b74cce59ca590b94b478edc5cf1a4c120ebda0e1d71902791c01d8
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
set_terminal_status() {
  set_status "$1" "$2"
  terminal_status_written=1
}
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

worker_pgid=""
vllm_pgid=""
terminate_session() {
  local pgid=${1:-}
  [[ -n "$pgid" ]] || return 0
  if kill -0 -- "-$pgid" 2>/dev/null; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 -- "-$pgid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-$pgid" 2>/dev/null; then
      kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
  fi
  wait "$pgid" 2>/dev/null || true
}
cleanup_owned_sessions() {
  terminate_session "$worker_pgid"
  worker_pgid=""
  terminate_session "$vllm_pgid"
  vllm_pgid=""
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
on_signal() {
  local signal=$1 code=$2
  set_status interrupted "mode=$MODE signal=$signal log=$RUN_LOG"
  terminal_status_written=1
  exit "$code"
}

mkdir -p "$RUN_ROOT/status" "$RUN_ROOT/logs"
case "$VALIDATION_OUT" in "$RUN_ROOT"/*) ;; *) blocked "validation output is outside RUN_ROOT: $VALIDATION_OUT" ;; esac
case "$TRAIN_OUT" in "$RUN_ROOT"/*) ;; *) blocked "training output is outside RUN_ROOT: $TRAIN_OUT" ;; esac
[[ "$VALIDATION_OUT" != "$TRAIN_OUT" ]] || blocked 'validation and training outputs must differ'
[[ ! -L "$RUN_ROOT" && ! -L "$VALIDATION_OUT" && ! -L "$TRAIN_OUT" ]] \
  || blocked 'owned output roots may not be symlinks'
[[ "$FROZEN_BOUNDARY_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || blocked 'FROZEN_BOUNDARY_MANIFEST_SHA256 must be an explicit 64-character lowercase digest'
[[ "$VALIDATION_SEED" != 20260812 ]] || blocked 'validation seed must differ from screening seed'
[[ "$VALIDATION_GPU" =~ ^[0-9]+$ && "$TRAIN_GPU" =~ ^[0-9]+$ && "$VLLM_GPU" =~ ^[0-9]+$ ]] \
  || blocked 'GPU indices must be explicit nonnegative integers'
[[ "$TRAIN_GPU" != "$VLLM_GPU" ]] || blocked 'trainer and vLLM require distinct GPUs'

exec 9>"$LOCK"
if ! flock -n 9; then
  printf 'another boundary300 launcher owns %s\n' "$LOCK" >&2
  terminal_status_written=1
  exit 75
fi
trap on_exit EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
exec >>"$RUN_LOG" 2>&1
set_status validating "mode=$MODE selection_manifest=$FROZEN_BOUNDARY_MANIFEST_SHA256"

verify_boundary_selection() {
  env PYTHONPATH= "$PYTHON_BIN" - \
    "$BOUNDARY_MANIFEST" "$BOUNDARY_VERIFICATION" "$BOUNDARY_TASKS" \
    "$TRAIN_TASKS" "$VALIDATION_TASKS" "$FROZEN_BOUNDARY_MANIFEST_SHA256" <<'PY'
import collections, hashlib, json, re, sys
from pathlib import Path

manifest_path, verification_path, boundary_path, train_path, validation_path = map(Path, sys.argv[1:6])
expected_manifest_sha = sys.argv[6]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha)
for path in (manifest_path, verification_path, boundary_path, train_path, validation_path):
    assert path.is_file() and not path.is_symlink(), path
assert sha(manifest_path) == expected_manifest_sha
manifest = json.loads(manifest_path.read_text())
verification = json.loads(verification_path.read_text())
assert manifest["schema_version"] == "policy-boundary-grpo-cohort-v1"
assert manifest["status"] == "frozen_boundary_training_cohort"
assert verification["schema_version"] == "policy-boundary-grpo-cohort-verification-v1"
assert verification["status"] == "verified"
assert Path(verification["manifest"]["path"]).resolve() == manifest_path.resolve()
assert verification["manifest"]["sha256"] == expected_manifest_sha
assert manifest["selection"]["seed"] == "qwen3-v26-boundary332-v1-20260812"
assert all(manifest["selection"]["acceptance_gates"].values())
assert manifest["contract"] == {
    "boundary_records": 332,
    "train_records": 300,
    "validation_records": 32,
    "formal_training": {
        "optimizer_updates": 20,
        "prompts_per_update": 30,
        "group_size": 8,
        "train_passes": 2,
        "prompt_appearances": 600,
        "fresh_online_trajectories": 4800,
        "sampler": "trl-0.29-repeat-sampler-v1",
        "shuffle_dataset": True,
        "data_seed": 20260812,
        "task_order": "two deterministic data-seed shuffled passes",
        "per_pass_coverage": "each train300 identity exactly once",
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
    },
    "validation": {
        "policy": "fresh initial-SFT1",
        "records": 32,
        "group_size": 8,
        "seed": 20260813,
        "screen_seed_must_differ": 20260812,
        "generation_seed_scheme": "sha256-task-sample-turn-v1",
        "gate": "same >=20/32 mixed probe gate",
        "runtime_contamination_allowed": False,
        "screen_trajectories_reused": False,
    },
    "primary_checkpoint": "final-step20-only",
}

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def task_id(row):
    value = row.get("example_id") or row.get("instance_id")
    assert isinstance(value, str) and value
    return value

paths = {"boundary": boundary_path, "train": train_path, "validation": validation_path}
counts = {"boundary": 332, "train": 300, "validation": 32}
observed = {}
rows_by_name = {}
for name, path in paths.items():
    rows = read_jsonl(path)
    ids = [task_id(row) for row in rows]
    assert len(rows) == counts[name] == len(set(ids))
    declared = manifest["outputs"][name]
    actual_sha = sha(path)
    assert declared == {"path": str(path.resolve()), "sha256": actual_sha, "records": counts[name]}
    assert manifest["task_ids"][name] == ids
    assert verification["outputs"][name] == {
        "path": str(path.resolve()), "records": counts[name], "sha256": actual_sha
    }
    observed[name] = ids
    rows_by_name[name] = rows
assert set(observed["train"]).isdisjoint(observed["validation"])
assert set(observed["train"]) | set(observed["validation"]) == set(observed["boundary"])
boundary_rows = {task_id(row): row for row in rows_by_name["boundary"]}
assert all(boundary_rows[task_id(row)] == row for row in rows_by_name["train"] + rows_by_name["validation"])

screen_pools = manifest["screen_pools"]
assert len(screen_pools) in {1, 2}
assert manifest["selection"]["screen_stage"] == ("S1" if len(screen_pools) == 1 else "S1+S2")
source_rows = {}
for pool in screen_pools:
    audit_path = Path(pool["audit_path"])
    tasks_path = Path(pool["tasks_path"])
    assert audit_path.is_file() and tasks_path.is_file()
    assert sha(audit_path) == pool["audit_sha256"]
    assert sha(tasks_path) == pool["tasks_sha256"]
    audit = json.loads(audit_path.read_text())
    assert audit["schema_version"] == "vanilla-grpo-boundary-screen-audit-v1"
    assert audit["status"]["audit_passes"] is True
    assert audit["status"]["pool_admitted"] is True
    assert audit["issues"] == [] and audit["issue_counts"] == {}
    for row in read_jsonl(tasks_path):
        identifier = task_id(row)
        assert identifier not in source_rows
        source_rows[identifier] = row
assert all(source_rows[identifier] == row for identifier, row in boundary_rows.items())
PY
}

validate_static_inputs() {
  [[ "$TRAIN_RUNTIME" == "$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_boundary300_grpo_20260812" ]] \
    || blocked 'TRAIN_RUNTIME must be the dedicated boundary300-GRPO overlay'
  for forbidden in src/eval src/sft src/harness src/tool_modules; do
    [[ ! -e "$TRAIN_RUNTIME/$forbidden" ]] \
      || blocked "training overlay shadows frozen protocol source: $TRAIN_RUNTIME/$forbidden"
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
  require_sha "$EXPERIMENT_CONFIG" "$EXPECTED_CONFIG_SHA256" boundary300_config
  require_sha "$VALIDATION_AUDITOR" "$EXPECTED_VALIDATION_AUDITOR_SHA256" boundary_validation_auditor
  require_sha "$TRAIN_RUNTIME/src/rl/diagnostics/audit_vanilla_grpo_rollout_probe.py" "$EXPECTED_BASE_PROBE_AUDITOR_SHA256" base_probe_auditor
  require_sha "$TRAIN_RUNTIME/src/rl/diagnostics/audit_vanilla_grpo_boundary_screen.py" "$EXPECTED_BOUNDARY_SCREEN_AUDITOR_SHA256" boundary_screen_auditor
  require_sha "$TRAIN_RUNTIME/src/rl/select_policy_boundary_grpo_tasks.py" "$EXPECTED_BOUNDARY_SELECTOR_SHA256" boundary_selector
  require_sha "$TRAIN_RUNTIME/src/rl/experiment_config.py" 6bd4b836984a4cbf3156170c230464bfaa113a645d21df31f8c3e54416f05ccf experiment_config_py
  require_sha "$TRAIN_RUNTIME/src/rl/task_loader.py" d79d41ea5f32ddfa45bb7c1496e8de234f96a6b48d84dfa2363d0de37a024d4b task_loader_py
  require_sha "$TRAIN_RUNTIME/src/rl/rollout_scoring.py" a11233e7a04a9efa34393e7d77a4c4aca34151144bf531978235ef6d644c1ac5 rollout_scoring_py
  require_sha "$TRAIN_RUNTIME/src/rl/terminal_reward.py" 2b45d45f562a903fe6e83589bc3aa9b9cc8d22036be6d0544c562ced82757b7b terminal_reward_py
  require_sha "$TRAIN_RUNTIME/src/rl/tool_environment.py" fcd7e77c668623da7027ddd5b35c3e945f1f39ba93f8eb50adf742551d68e6a1 current_tool_environment_py
  require_sha "$TRAIN_RUNTIME/src/rl/tool_environment_v26.py" c7a84bdb2d259eea91cde5a77a5758ac8ea57e02828330088ecbd903b1d0ec5c isolated_v26_tool_environment_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/run_transition_grpo.py" "$EXPECTED_RUN_TRANSITION_GRPO_SHA256" run_transition_grpo_py
  require_sha "$TRAIN_RUNTIME/src/rl/diagnostics/prepare_vanilla_grpo_resume.py" "$EXPECTED_RESUME_PREPARER_SHA256" resume_preparer_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/transition_grpo.py" 1ef68a63b9857a43aa399710d2e58e84aadb4e0106e917237333471501be3b37 transition_grpo_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/transition_batch.py" c6477393589581209ad581230eb598160e0674469103ce197b7d7d7f26a0bcd8 transition_batch_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/rollout.py" 87c36224ba0f01954db0d58db3bc3e5685bc5c86736c0679299c76872b6c821f trl_rollout_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/training_precision.py" f5dd61eb2bc307ac422243a6d2be14e99382797aa3acd76396c0d553e76b215f training_precision_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" b2fa3a0b6277fcfb30d30f0f14b08a2a288c6f07435efbb6a1c4135864a97657 run_atomic_transition_grpo_sh
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" 627f0a07f90d91702acb2f5379d2c111668a97ee90dbf37294d0ee2bd3324d57 start_vllm_server_sh
  require_sha "$TRAIN_RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py" db934c05cbf4f2d9beef3e01e8b42ff74312ef00834e681a9ae65647de12e191 fixed_pool_generator_py
  require_sha "$TRAIN_RUNTIME/src/rl/counterfactual_suite.py" 13e88a98cb8b9d0cd350a1b963315b80c5fe951f8ac668d5b8d4140b5d00df92 counterfactual_suite_py
  require_sha "$TRAIN_RUNTIME/src/rl/reference_result_filter.py" 2d5b1e61181facfa2883f5dd926358e9f75d8eb0abe12dab3204628f5631dc92 reference_result_filter_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/fixed_rollout_pool.py" e0be4c4c830ca2e87172175553ab47d1b19561510f8fa6f77410e96846cf200d fixed_rollout_pool_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/tool_loss_mask.py" b15072423600c663d91651e640f7738087bdc1ea551758704f98cc6882d14abb tool_loss_mask_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/trajectory_ranking.py" 2097282799b321b32236f32686ea25d1d078139bae7fecc5f4be58574ddb9eb7 trajectory_ranking_py
  verify_boundary_selection || blocked 'boundary selection manifest/verification/output binding failed'

  env PYTHONPATH= "$PYTHON_BIN" - "$TRAIN_RUNTIME" "$PROTOCOL_RUNTIME" \
    "$EXPERIMENT_CONFIG" "$EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" \
    "$EXPECTED_PROTOCOL_VERSION" "$EXPECTED_PROTOCOL_HASH" \
    "$EXPECTED_STUDENT_PROMPT_SHA256" "$EXPECTED_SFT1_SHA256" <<'PY'
import hashlib, importlib, inspect, sys
import importlib.metadata
from pathlib import Path

overlay, runtime, config_path = map(Path, sys.argv[1:4])
expected_tree, expected_version, expected_hash, expected_prompt, expected_adapter = sys.argv[4:9]
sys.path[:0] = [str(overlay / "src/rl"), str(runtime / "src/eval"), str(runtime / "src/harness"), str(runtime / "src/sft")]
protocol = importlib.import_module("protocol")
prompt = protocol.student_runtime_system_prompt(context_mode="rolling-legal-history", compact=False)
assert protocol.PROTOCOL_VERSION == expected_version
assert protocol.protocol_hash(prompt) == expected_hash
assert hashlib.sha256(prompt.encode()).hexdigest() == expected_prompt
from experiment_config import RLExperimentConfig, runtime_content_tree_sha256
defaults = RLExperimentConfig.load(config_path).argparse_defaults(overlay)
required = {
    "reward_mode": "result-only", "result_reward_profile": "binary",
    "policy_reduction": "trajectory_token_mean", "expected_records": 300,
    "optimizer_steps": 20, "prompts_per_update": 30, "group_size": 8,
    "ppo_iterations": 1, "learning_rate": 8e-7, "kl_beta": 0.0,
    "max_new_tokens": 2048, "max_context_tokens": 16384,
    "enable_thinking": True, "expected_protocol_version": expected_version,
    "expected_protocol_hash": expected_hash,
    "expected_student_prompt_sha256": expected_prompt,
    "expected_initial_adapter_sha256": expected_adapter,
    "expected_reference_adapter_sha256": expected_adapter,
    "expected_runtime_content_tree_sha256": expected_tree,
}
for key, expected in required.items():
    assert defaults.get(key) == expected, (key, defaults.get(key), expected)
assert defaults["protocol_runtime_root"].resolve() == runtime.resolve()
assert runtime_content_tree_sha256(runtime) == expected_tree
environment = importlib.import_module("tool_environment_v26")
assert Path(inspect.getfile(environment.ToolUseEnv)).resolve() == overlay / "src/rl/tool_environment_v26.py"

# Prove the pinned TRL 0.29 sampler contract without assuming JSONL order.
# Each iterator is one deterministic shuffled pass: 300 unique indices,
# grouped into 30-prompt generation batches, each repeated exactly K=8.
assert importlib.metadata.version("trl").startswith("0.29.")
from trl import GRPOConfig
from trl.trainer.utils import RepeatSampler
shuffle_default = inspect.signature(GRPOConfig).parameters["shuffle_dataset"].default
assert shuffle_default is True
def two_passes():
    sampler = RepeatSampler(
        data_source=range(300), mini_repeat_count=8, batch_size=30,
        repeat_count=1, shuffle=True, seed=20260812,
    )
    return [list(iter(sampler)), list(iter(sampler))]
first_run = two_passes()
assert first_run == two_passes()
for sampled in first_run:
    assert len(sampled) == 2400
    groups = [sampled[offset:offset + 8] for offset in range(0, len(sampled), 8)]
    collapsed = [group[0] for group in groups]
    assert all(len(set(group)) == 1 for group in groups)
    assert len(collapsed) == 300 and set(collapsed) == set(range(300))
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
    if [[ "$used" -le 512 ]]; then return 0; fi
    set_status waiting_gpu "mode=$MODE gpu=$gpu used_mib=$used; no process will be stopped"
    sleep 30
  done
}

run_validation_audit() {
  "$PYTHON_BIN" "$VALIDATION_AUDITOR" \
    --selection-manifest "$BOUNDARY_MANIFEST" \
    --expected-selection-manifest-sha256 "$FROZEN_BOUNDARY_MANIFEST_SHA256" \
    --manifest "$VALIDATION_MANIFEST" \
    --trajectories "$VALIDATION_TRAJECTORIES" \
    --tasks "$VALIDATION_TASKS" \
    --output "$VALIDATION_AUDIT" \
    --overwrite
}

verify_validation_gate() {
  [[ -f "$VALIDATION_MANIFEST" && -f "$VALIDATION_TRAJECTORIES" && -f "$VALIDATION_AUDIT" ]] \
    || blocked 'fresh boundary validation artifacts are incomplete; run validate first'
  run_validation_audit || blocked 'fresh boundary validation32 gate failed'
  "$PYTHON_BIN" - "$VALIDATION_AUDIT" "$BOUNDARY_MANIFEST" \
    "$VALIDATION_MANIFEST" "$VALIDATION_TRAJECTORIES" "$VALIDATION_TASKS" \
    "$FROZEN_BOUNDARY_MANIFEST_SHA256" <<'PY'
import hashlib, json, sys
from pathlib import Path
audit_path, selection_path, generation_path, trajectories_path, tasks_path = map(Path, sys.argv[1:6])
expected_selection_sha = sys.argv[6]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
audit = json.loads(audit_path.read_text())
assert audit["schema_version"] == "vanilla-grpo-boundary-validation-audit-v1"
assert audit["status"] == {"passes": True, "validation_admitted": True}
assert "probe_admitted" not in audit["status"]
assert audit["contract"]["validation_seed"] == 20260813
assert audit["contract"]["runtime_contamination_allowed"] is False
assert audit["observed"]["eligible_trajectories"] == 256
assert audit["observed"]["mixed_outcome_groups"] >= 20
assert audit["observed"]["tokenization_warnings"] == 0
assert audit["checks"]["no_runtime_contamination"] is True
assert audit["inputs"]["selection_manifest_sha256"] == expected_selection_sha == sha(selection_path)
for key, path in (("generation_manifest", generation_path), ("tasks", tasks_path), ("trajectories", trajectories_path)):
    assert Path(audit["inputs"][key]).resolve() == path.resolve()
    assert audit["inputs"][f"{key}_sha256"] == sha(path)
PY
}

run_validation() {
  local generate=1 unexpected=""
  if [[ -d "$VALIDATION_OUT" ]] && [[ -n "$(find "$VALIDATION_OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    if [[ -f "$VALIDATION_MANIFEST" && -f "$VALIDATION_TRAJECTORIES" ]]; then
      generate=0
      unexpected=$(find "$VALIDATION_OUT" -mindepth 1 -maxdepth 1 \
        ! -name groups ! -name manifest.pending.json ! -name trajectories.jsonl \
        ! -name boundary_validation_audit.json -print -quit)
      [[ -z "$unexpected" ]] || blocked "validation output contains unknown entry: $unexpected"
    else
      unexpected=$(find "$VALIDATION_OUT" -mindepth 1 -maxdepth 1 ! -name groups -print -quit)
      [[ -z "$unexpected" && -d "$VALIDATION_OUT/groups" ]] \
        || blocked "unknown/incomplete validation output: ${unexpected:-$VALIDATION_OUT}"
      if find "$VALIDATION_OUT/groups" -mindepth 1 -maxdepth 1 ! -type f -print -quit | grep -q .; then
        blocked 'validation groups contain a non-file entry'
      fi
      if find "$VALIDATION_OUT/groups" -mindepth 1 -maxdepth 1 -type f ! -name '*.json' -print -quit | grep -q .; then
        blocked 'validation groups contain an unknown file'
      fi
    fi
  else
    mkdir -p "$VALIDATION_OUT"
  fi
  if [[ "$generate" -eq 1 ]]; then
    wait_for_gpu "$VALIDATION_GPU"
    set_status validating_policy "gpu=$VALIDATION_GPU validation32 k=8 seed=$VALIDATION_SEED initial=SFT1"
    setsid env \
      CUDA_VISIBLE_DEVICES="$VALIDATION_GPU" \
      HF_HUB_OFFLINE=1 \
      TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$PROTOCOL_RUNTIME" \
      PYTHONPATH="$TRAIN_RUNTIME/src/rl" \
      "$PYTHON_BIN" "$TRAIN_RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py" \
        --model-path "$MODEL_PATH" \
        --adapter-path "$SFT1_ADAPTER" \
        --tasks "$VALIDATION_TASKS" \
        --output-dir "$VALIDATION_OUT" \
        --group-size 8 \
        --temperature 0.8 \
        --top-p 1 \
        --max-steps 30 \
        --max-new-tokens 2048 \
        --max-context-tokens 16384 \
        --history-turns 4 \
        --enable-thinking \
        --seed "$VALIDATION_SEED" \
        --gpu-memory-utilization 0.82 \
        --scheduler dynamic \
        --question-window 4 >>"$VALIDATION_LOG" 2>&1 &
    worker_pgid=$!
    local code=0
    if wait "$worker_pgid"; then code=0; else code=$?; fi
    terminate_session "$worker_pgid"
    worker_pgid=""
    [[ "$code" -eq 0 ]] || return "$code"
  fi
  set_status auditing_validation "seed=$VALIDATION_SEED manifest=$FROZEN_BOUNDARY_MANIFEST_SHA256"
  verify_validation_gate
  set_terminal_status complete "validation_admitted=true audit=$VALIDATION_AUDIT"
}

validate_training_artifacts() {
  "$PYTHON_BIN" - "$TRAIN_OUT" "$TRAIN_TASKS" "$BOUNDARY_MANIFEST" \
    "$FROZEN_BOUNDARY_MANIFEST_SHA256" "$EXPECTED_CONFIG_SHA256" \
    "$EXPECTED_PROTOCOL_VERSION" "$EXPECTED_PROTOCOL_HASH" \
    "$EXPECTED_STUDENT_PROMPT_SHA256" "$EXPECTED_SFT1_SHA256" \
    "$PROTOCOL_RUNTIME" "$EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" <<'PY'
import collections, hashlib, json, sys
from pathlib import Path
root, tasks_path, selection_path = map(Path, sys.argv[1:4])
expected_selection, expected_config, expected_version, expected_hash, expected_prompt, expected_adapter, expected_runtime, expected_tree = sys.argv[4:12]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(selection_path) == expected_selection
manifest = json.loads((root / "run_manifest.json").read_text())
assert manifest["experiment_config_sha256"] == expected_config
assert manifest["examples_json_sha256"] == sha(tasks_path)
assert manifest["protocol_version"] == expected_version
assert manifest["protocol_hash"] == expected_hash
assert manifest["student_prompt_sha256"] == expected_prompt
assert manifest["initial_adapter_sha256"] == expected_adapter
assert manifest["records"] == manifest["expected_records"] == 300
assert manifest["optimizer_steps"] == 20
assert manifest["save_steps"] == 2
assert manifest["save_total_limit"] == 1
assert str(manifest["framework_version"]).startswith("0.29.")
assert manifest["seed"] == 20260812
assert manifest["prompts_per_update"] == 30
assert manifest["group_size"] == 8
assert manifest["ppo_iterations"] == 1
assert manifest["learning_rate"] == 8e-7
assert manifest["kl_beta"] == 0.0
assert manifest["policy_reduction"] == "trajectory_token_mean"
assert manifest["result_reward_profile"] == "binary"
assert manifest["rollout_settings"]["max_new_tokens"] == 2048
assert manifest["rollout_settings"]["max_context_tokens"] == 16384
assert manifest["rollout_settings"]["enable_thinking"] is True
runtime = manifest["runtime_module_audit"]
assert runtime["runtime_root"] == expected_runtime
assert runtime["tool_environment_factory_module"] == "tool_environment_v26"
assert runtime["module_paths"] == {
    "protocol": expected_runtime + "/src/sft/protocol.py",
    "rollout": expected_runtime + "/src/eval/rollout.py",
    "executor": expected_runtime + "/src/harness/executor.py",
    "tool_schemes": expected_runtime + "/src/sft/tool_schemes.py",
}
identity = manifest["runtime_identity_audit"]
assert identity["expected"]["runtime_content_tree_sha256"] == expected_tree
assert identity["actual"]["runtime_content_tree_sha256"] == expected_tree
reference = manifest["reference_policy"]
assert reference["enabled"] is False and reference["kl_beta"] == 0.0
assert reference["adapter_name"] is None and reference["adapter_path"] is None
assert reference["expected_adapter_sha256"] == expected_adapter

checkpoints = sorted(path.name for path in root.glob("checkpoint-*") if path.is_dir())
assert checkpoints == ["checkpoint-20"], checkpoints
state = json.loads((root / "checkpoint-20/trainer_state.json").read_text())
assert int(state["global_step"]) == 20
assert (root / "checkpoint-20/adapter_model.safetensors").is_file()
assert (root / "final/adapter_model.safetensors").is_file()
assert (root / "training_precision.json").is_file()

tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
indices = [int(row["example_index"]) for row in tasks]
assert len(indices) == len(set(indices)) == 300
rollouts = [json.loads(line) for line in (root / "rollouts.jsonl").read_text().splitlines() if line.strip()]
assert len(rollouts) == 4800
for step in range(20):
    block = rollouts[step * 240:(step + 1) * 240]
    assert len(block) == 240
    assert {int(row["policy_global_step"]) for row in block} == {step}
    block_counts = collections.Counter(int(row["example_index"]) for row in block)
    assert len(block_counts) == 30 and set(block_counts.values()) == {8}
for pass_index in range(2):
    block = rollouts[pass_index * 2400:(pass_index + 1) * 2400]
    counts = collections.Counter(int(row["example_index"]) for row in block)
    assert counts == collections.Counter({index: 8 for index in indices})
PY
}

RESUME_CHECKPOINT=""
RESUME_PREPARATION=$TRAIN_OUT/resume_preparation.json
TRAIN_ALREADY_COMPLETE=0
prepare_train_output() {
  if [[ -f "$TRAIN_OUT/final/adapter_model.safetensors" ]]; then
    validate_training_artifacts || blocked "existing final training output failed its two-pass/step20 audit: $TRAIN_OUT"
    TRAIN_ALREADY_COMPLETE=1
    return
  fi
  [[ ! -e "$TRAIN_OUT/final" ]] || blocked "incomplete final directory: $TRAIN_OUT/final"
  if [[ ! -d "$TRAIN_OUT" ]] || [[ -z "$(find "$TRAIN_OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    mkdir -p "$TRAIN_OUT"
    return
  fi
  [[ -f "$TRAIN_OUT/run_manifest.json" && -f "$TRAIN_OUT/implementation_lock.json" ]] \
    || blocked 'nonempty train output lacks immutable run locks'
  "$PYTHON_BIN" "$TRAIN_RUNTIME/src/rl/diagnostics/prepare_vanilla_grpo_resume.py" \
    --train-output "$TRAIN_OUT" \
    --tasks "$TRAIN_TASKS" \
    --expected-records 300 \
    --optimizer-steps 20 \
    --prompts-per-update 30 \
    --group-size 8 \
    --save-steps 2 \
    --output "$RESUME_PREPARATION" \
    || blocked 'no safe complete periodic checkpoint/rollout prefix is resumable'
  RESUME_CHECKPOINT=$("$PYTHON_BIN" - "$RESUME_PREPARATION" "$TRAIN_OUT" <<'PY'
import json, sys
from pathlib import Path
record = json.load(open(sys.argv[1]))
root = Path(sys.argv[2]).resolve()
assert record["schema_version"] == "vanilla-grpo-resume-preparation-v1"
assert record["status"] == "resume_ready"
assert record["contract"] == {
    "optimizer_steps": 20,
    "prompts_per_update": 30,
    "group_size": 8,
    "save_steps": 2,
    "committed_rollouts_per_step": 240,
}
step = record["checkpoint"]["global_step"]
checkpoint = Path(record["checkpoint"]["path"]).resolve()
assert 0 < step < 20 and step % 2 == 0
assert checkpoint == root / f"checkpoint-{step}"
assert record["rollouts"]["retained_rows"] == step * 240
assert record["rollouts"]["removed_uncommitted_rows"] >= 0
print(checkpoint)
PY
  ) || blocked 'resume preparation manifest failed verification'
}

start_train_vllm() {
  if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
    blocked "port $VLLM_PORT already has a server; refusing unowned reuse"
  fi
  set_status starting_vllm "gpu=$VLLM_GPU port=$VLLM_PORT max_model_len=16384"
  setsid env \
    CUDA_VISIBLE_DEVICES="$VLLM_GPU" \
    PYTHON_ENV=/home/dengyan/miniconda3/envs/trl-table \
    MODEL_PATH="$MODEL_PATH" \
    VLLM_PORT="$VLLM_PORT" \
    VLLM_GPU_MEMORY_UTILIZATION=0.82 \
    MAX_MODEL_LEN=16384 \
    bash "$TRAIN_RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" >>"$VLLM_LOG" 2>&1 &
  vllm_pgid=$!
  local ready=0
  for _ in {1..240}; do
    kill -0 "$vllm_pgid" 2>/dev/null || break
    if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 2
  done
  [[ "$ready" -eq 1 ]] || blocked "owned vLLM failed readiness: $VLLM_LOG"
}

run_train() {
  verify_validation_gate
  prepare_train_output
  if [[ "$TRAIN_ALREADY_COMPLETE" -eq 1 ]]; then
    set_terminal_status complete "existing validated final=$TRAIN_OUT/final"
    return
  fi
  wait_for_gpu "$TRAIN_GPU"
  wait_for_gpu "$VLLM_GPU"
  start_train_vllm
  local resume_args=()
  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    resume_args=(--resume-from-checkpoint "$RESUME_CHECKPOINT")
  fi
  set_status training 'records=300 passes=2 prompts_per_update=30 k=8 updates=20 save_steps=2 save_total_limit=1 primary=step20 reward=binary'
  setsid env \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
    PROJECT_DIR="$TRAIN_RUNTIME" \
    PYTHON="$PYTHON_BIN" \
    MODEL_PATH="$MODEL_PATH" \
    ADAPTER_PATH="$SFT1_ADAPTER" \
    EXAMPLES_JSON="$TRAIN_TASKS" \
    OUTPUT_DIR="$TRAIN_OUT" \
    EXPERIMENT_CONFIG="$EXPERIMENT_CONFIG" \
    VLLM_PORT="$VLLM_PORT" \
    VLLM_GROUP_PORT="$VLLM_GROUP_PORT" \
    PYTHONPATH= \
    bash "$TRAIN_RUNTIME/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" \
      --optimizer-steps 20 \
      --ppo-iterations 1 \
      --prompts-per-update 30 \
      --group-size 8 \
      --kl-beta 0 \
      --protocol-runtime-root "$PROTOCOL_RUNTIME" \
      --transition-micro-batch-size 1 \
      --save-steps 2 \
      --save-total-limit 1 \
      --seed "$TRAIN_SEED" \
      "${resume_args[@]}" >>"$TRAIN_LOG" 2>&1 &
  worker_pgid=$!
  local code=0
  if wait "$worker_pgid"; then code=0; else code=$?; fi
  terminate_session "$worker_pgid"
  worker_pgid=""
  terminate_session "$vllm_pgid"
  vllm_pgid=""
  [[ "$code" -eq 0 ]] || return "$code"
  validate_training_artifacts || blocked 'formal output failed exact two-pass/final-step20 audit'
  set_terminal_status complete "train=$TRAIN_OUT final=$TRAIN_OUT/final primary_checkpoint=step20"
}

validate_static_inputs
case "$MODE" in
  validate) run_validation ;;
  train) run_train ;;
esac
