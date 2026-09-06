#!/usr/bin/env bash
# Staged, fail-closed Qwen3-8B atomic-v26 vanilla-GRPO launcher for table_rl.
#
# probe (default): sample the frozen first 32 training tasks at K=8 without an
# optimizer, then require the binary-reward rollout-probe audit to pass.
# train: require that immutable probe gate, then run 20 on-policy updates on
# GPU0 while an owned TRL vLLM server serves the base model on GPU1.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODE=${1:-probe}
if [[ $# -gt 1 || ( "$MODE" != probe && "$MODE" != train ) ]]; then
  printf 'usage: %s [probe|train]\n' "$0" >&2
  exit 2
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
# This is a newly deployed, hash-pinned GRPO code overlay. Model-visible
# eval/sft/harness imports come from the separate read-only PROTOCOL_RUNTIME.
# Never point this launcher at, or modify, the historical 20260811 RL runtime.
TRAIN_RUNTIME=${TRAIN_RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_vanilla_grpo_20260812}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
SFT1_ADAPTER=${SFT1_ADAPTER:-$OUTPUT_ROOT/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560}
TASKS_JSONL=${TASKS_JSONL:-$TRAIN_RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.jsonl}
TASKS_MANIFEST=${TASKS_MANIFEST:-$TRAIN_RUNTIME/data/rl_inputs/qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.manifest.json}
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-$TRAIN_RUNTIME/src/rl/configs/experiments/qwen3_8b_atomic_v26_vanilla_grpo.yaml}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}

RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_8b_atomic_v26_vanilla_grpo_20260812}
PROBE_OUT=${PROBE_OUT:-$RUN_ROOT/probe_first32_k8_seed20260812}
TRAIN_OUT=${TRAIN_OUT:-$RUN_ROOT/train600_seed20260812}
PROBE_AUDIT=$PROBE_OUT/rollout_probe_audit.json
PROBE_MANIFEST=$PROBE_OUT/manifest.pending.json
PROBE_TRAJECTORIES=$PROBE_OUT/trajectories.jsonl
STATUS=$RUN_ROOT/status/${MODE}.status
RUN_LOG=$RUN_ROOT/logs/${MODE}.log
PROBE_LOG=$RUN_ROOT/logs/probe_generation.log
VLLM_LOG=$RUN_ROOT/logs/train_vllm.log
TRAIN_LOG=$RUN_ROOT/logs/train_worker.log
LOCK=$RUN_ROOT/launcher.lock

VLLM_PORT=${VLLM_PORT:-8076}
VLLM_GROUP_PORT=${VLLM_GROUP_PORT:-51276}
EXPECTED_PROTOCOL_VERSION=version26
EXPECTED_PROTOCOL_HASH=4da19387399bd3a5
EXPECTED_STUDENT_PROMPT_SHA256=848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316
EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256=5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab
EXPECTED_SFT1_SHA256=3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5
EXPECTED_SFT1_CONFIG_SHA256=537dbc946a7131d59e294a8729ace4aa145d73e59f4cf8556360ea507ce4435a
EXPECTED_SFT1_STATE_SHA256=97e529475d8c4f68dc88bc1f80370dacab478a681629f5f99003c39c70e9600a
EXPECTED_TASKS_SHA256=b5a83c373e9be094ea7355c0bc23c9212249491491f44dfdcb3b09ea49b2457e
EXPECTED_TASKS_MANIFEST_SHA256=9212d1f7ca4fb63576eb7e846a9b05f7d159e131fe992495b99e350b6aae8dd6
# Filled from the reviewed local artifact; a partial or later runtime sync must
# fail here instead of silently changing the experiment contract.
EXPECTED_CONFIG_SHA256=c5b933e576ac04ef932d9e2900585429c424e5d0a595b8dc903a4233ce37df89

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
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
  # Only process groups created by this invocation are targeted. In particular,
  # this never searches for or signals unrelated table_rl or NewGNN processes.
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
trap on_exit EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

mkdir -p "$RUN_ROOT/status" "$RUN_ROOT/logs"
case "$PROBE_OUT" in "$RUN_ROOT"/*) ;; *) blocked "probe output is outside isolated run root: $PROBE_OUT" ;; esac
case "$TRAIN_OUT" in "$RUN_ROOT"/*) ;; *) blocked "training output is outside isolated run root: $TRAIN_OUT" ;; esac
[[ "$PROBE_OUT" != "$TRAIN_OUT" ]] || blocked "probe and training outputs must differ"
[[ ! -L "$PROBE_OUT" && ! -L "$TRAIN_OUT" ]] || blocked "output directories may not be symlinks"

exec 9>"$LOCK"
if ! flock -n 9; then
  # Do not overwrite the active owner's status file from a duplicate caller.
  printf 'another vanilla-GRPO launcher owns %s\n' "$LOCK" >&2
  terminal_status_written=1
  exit 0
fi
exec >>"$RUN_LOG" 2>&1
set_status validating "mode=$MODE runtime=$TRAIN_RUNTIME"

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
require_sha() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "$path" ]] || blocked "missing $label: $path"
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] || blocked "$label SHA-256 mismatch: expected=$expected actual=$actual path=$path"
}

validate_static_inputs() {
  [[ "$TRAIN_RUNTIME" == "$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_vanilla_grpo_20260812" ]] \
    || blocked "TRAIN_RUNTIME must be the isolated vanilla-GRPO runtime"
  for forbidden_tree in src/eval src/harness src/sft src/tool_modules; do
    [[ ! -e "$TRAIN_RUNTIME/$forbidden_tree" ]] \
      || blocked "GRPO overlay must not shadow frozen protocol sources: $TRAIN_RUNTIME/$forbidden_tree"
  done
  [[ -x "$PYTHON_BIN" ]] || blocked "missing Python runtime: $PYTHON_BIN"
  [[ -d "$MODEL_PATH" ]] || blocked "missing Qwen3 base model: $MODEL_PATH"
  [[ -d "$SFT1_ADAPTER" ]] || blocked "missing SFT1 checkpoint-560: $SFT1_ADAPTER"

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
  require_sha "$TASKS_JSONL" "$EXPECTED_TASKS_SHA256" train600_tasks
  require_sha "$TASKS_MANIFEST" "$EXPECTED_TASKS_MANIFEST_SHA256" train600_manifest
  require_sha "$EXPERIMENT_CONFIG" "$EXPECTED_CONFIG_SHA256" vanilla_grpo_config

  [[ "$PROTOCOL_RUNTIME" == "$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de" ]] \
    || blocked "PROTOCOL_RUNTIME must be the frozen version26 source tree"
  [[ -d "$PROTOCOL_RUNTIME/src/eval" && -d "$PROTOCOL_RUNTIME/src/sft" && -d "$PROTOCOL_RUNTIME/src/harness" ]] \
    || blocked "frozen protocol runtime is incomplete: $PROTOCOL_RUNTIME"

  # Current reviewed GRPO overlay. Hashes are deliberately literal so a stale,
  # partial, or post-review deployment is rejected before a GPU is allocated.
  require_sha "$TRAIN_RUNTIME/src/rl/experiment_config.py" 6bd4b836984a4cbf3156170c230464bfaa113a645d21df31f8c3e54416f05ccf experiment_config_py
  require_sha "$TRAIN_RUNTIME/src/rl/task_loader.py" d79d41ea5f32ddfa45bb7c1496e8de234f96a6b48d84dfa2363d0de37a024d4b task_loader_py
  require_sha "$TRAIN_RUNTIME/src/rl/rollout_scoring.py" a11233e7a04a9efa34393e7d77a4c4aca34151144bf531978235ef6d644c1ac5 rollout_scoring_py
  require_sha "$TRAIN_RUNTIME/src/rl/terminal_reward.py" 2b45d45f562a903fe6e83589bc3aa9b9cc8d22036be6d0544c562ced82757b7b terminal_reward_py
  require_sha "$TRAIN_RUNTIME/src/rl/tool_environment.py" fcd7e77c668623da7027ddd5b35c3e945f1f39ba93f8eb50adf742551d68e6a1 current_tool_environment_py
  require_sha "$TRAIN_RUNTIME/src/rl/tool_environment_v26.py" c7a84bdb2d259eea91cde5a77a5758ac8ea57e02828330088ecbd903b1d0ec5c isolated_v26_tool_environment_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/run_transition_grpo.py" 2990f15e303b110845d222e76c9e5ae92edc8111b9942cd8c89b655688b28caa run_transition_grpo_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/transition_grpo.py" 1ef68a63b9857a43aa399710d2e58e84aadb4e0106e917237333471501be3b37 transition_grpo_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/transition_batch.py" c6477393589581209ad581230eb598160e0674469103ce197b7d7d7f26a0bcd8 transition_batch_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/rollout.py" 87c36224ba0f01954db0d58db3bc3e5685bc5c86736c0679299c76872b6c821f trl_rollout_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/training_precision.py" f5dd61eb2bc307ac422243a6d2be14e99382797aa3acd76396c0d553e76b215f training_precision_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/run_atomic_transition_grpo.sh" b2fa3a0b6277fcfb30d30f0f14b08a2a288c6f07435efbb6a1c4135864a97657 run_atomic_transition_grpo_sh
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/start_vllm_server.sh" 627f0a07f90d91702acb2f5379d2c111668a97ee90dbf37294d0ee2bd3324d57 start_vllm_server_sh
  require_sha "$TRAIN_RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py" db934c05cbf4f2d9beef3e01e8b42ff74312ef00834e681a9ae65647de12e191 fixed_pool_generator_py
  require_sha "$TRAIN_RUNTIME/src/rl/diagnostics/audit_vanilla_grpo_rollout_probe.py" d192753faf1d579db616840e3ccdcfade855180a5c08144e1b5fe495e7eb85a5 probe_auditor_py
  require_sha "$TRAIN_RUNTIME/src/rl/counterfactual_suite.py" 13e88a98cb8b9d0cd350a1b963315b80c5fe951f8ac668d5b8d4140b5d00df92 counterfactual_suite_py
  require_sha "$TRAIN_RUNTIME/src/rl/reference_result_filter.py" 2d5b1e61181facfa2883f5dd926358e9f75d8eb0abe12dab3204628f5631dc92 reference_result_filter_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/fixed_rollout_pool.py" e0be4c4c830ca2e87172175553ab47d1b19561510f8fa6f77410e96846cf200d fixed_rollout_pool_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/tool_loss_mask.py" b15072423600c663d91651e640f7738087bdc1ea551758704f98cc6882d14abb tool_loss_mask_py
  require_sha "$TRAIN_RUNTIME/src/rl/frameworks/trl/trajectory_ranking.py" 2097282799b321b32236f32686ea25d1d078139bae7fecc5f4be58574ddb9eb7 trajectory_ranking_py

  env PYTHONPATH= "$PYTHON_BIN" - "$TRAIN_RUNTIME" "$PROTOCOL_RUNTIME" "$TASKS_JSONL" "$TASKS_MANIFEST" "$EXPERIMENT_CONFIG" \
    "$EXPECTED_PROTOCOL_VERSION" "$EXPECTED_PROTOCOL_HASH" "$EXPECTED_STUDENT_PROMPT_SHA256" \
    "$EXPECTED_SFT1_SHA256" "$EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" <<'PY'
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path

runtime = Path(sys.argv[1]).resolve()
protocol_runtime = Path(sys.argv[2]).resolve()
tasks_path = Path(sys.argv[3]).resolve()
tasks_manifest_path = Path(sys.argv[4]).resolve()
config_path = Path(sys.argv[5]).resolve()
expected_version, expected_hash, expected_prompt_sha, expected_adapter_sha, expected_tree_sha = sys.argv[6:11]
sys.path[:0] = [
    str(runtime / "src" / "rl"),
    str(protocol_runtime / "src" / "eval"),
    str(protocol_runtime / "src" / "harness"),
    str(protocol_runtime / "src" / "sft"),
]

protocol = importlib.import_module("protocol")
prompt = protocol.student_runtime_system_prompt(
    context_mode="rolling-legal-history", compact=False
)
assert protocol.PROTOCOL_VERSION == expected_version
assert protocol.protocol_hash(prompt) == expected_hash
assert hashlib.sha256(prompt.encode()).hexdigest() == expected_prompt_sha
tool_environment = importlib.import_module("tool_environment_v26")
assert Path(inspect.getfile(tool_environment.ToolUseEnv)).resolve() == (
    runtime / "src" / "rl" / "tool_environment_v26.py"
)

from rl.configuration.experiment_config import RLExperimentConfig, runtime_content_tree_sha256
defaults = RLExperimentConfig.load(config_path).argparse_defaults(runtime)
required = {
    "reward_mode": "result-only",
    "result_reward_profile": "binary",
    "policy_reduction": "trajectory_token_mean",
    "expected_records": 600,
    "optimizer_steps": 20,
    "ppo_iterations": 1,
    "prompts_per_update": 30,
    "group_size": 8,
    "kl_beta": 0.0,
    "max_new_tokens": 2048,
    "max_context_tokens": 16384,
    "enable_thinking": True,
    "expected_protocol_version": expected_version,
    "expected_protocol_hash": expected_hash,
    "expected_student_prompt_sha256": expected_prompt_sha,
    "expected_initial_adapter_sha256": expected_adapter_sha,
    "expected_reference_adapter_sha256": expected_adapter_sha,
    "expected_runtime_content_tree_sha256": expected_tree_sha,
}
for key, expected in required.items():
    assert defaults.get(key) == expected, (key, defaults.get(key), expected)
assert defaults["protocol_runtime_root"].resolve() == protocol_runtime
assert runtime_content_tree_sha256(protocol_runtime) == expected_tree_sha

rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
assert len(rows) == 600
ids = [str(row.get("example_id") or row.get("instance_id")) for row in rows]
assert "None" not in ids and len(set(ids)) == 600
assert all(
    str(row["db_path"]).startswith(
        "/home/dengyan/tabular_rl_project/data/bird/train/train_databases/"
    )
    for row in rows
)
cohort = json.loads(tasks_manifest_path.read_text())
assert cohort["schema_version"] == "bird-train-vanilla-grpo-cohort-v1"
assert cohort["status"] == "frozen_training_cohort"
assert cohort["all_acceptance_gates_passed"] is True
assert cohort["output"]["records"] == 600
assert cohort["output"]["sha256"] == hashlib.sha256(tasks_path.read_bytes()).hexdigest()
assert cohort["output"]["task_ids_in_frozen_order"] == ids
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
    [[ "$used" =~ ^[0-9]+$ ]] || blocked "cannot read memory usage for table_rl GPU$gpu"
    if [[ "$used" -le 512 ]]; then return 0; fi
    set_status waiting_gpu "mode=$MODE gpu=$gpu used_mib=$used; no process will be stopped"
    sleep 30
  done
}

run_probe_audit() {
  "$PYTHON_BIN" "$TRAIN_RUNTIME/src/rl/diagnostics/audit_vanilla_grpo_rollout_probe.py" \
    --manifest "$PROBE_MANIFEST" \
    --trajectories "$PROBE_TRAJECTORIES" \
    --output "$PROBE_AUDIT" \
    --overwrite
}

verify_probe_gate() {
  [[ -f "$PROBE_MANIFEST" && -f "$PROBE_TRAJECTORIES" && -f "$PROBE_AUDIT" ]] \
    || blocked "probe artifacts are incomplete; run probe mode first"
  "$PYTHON_BIN" - "$PROBE_MANIFEST" "$PROBE_TRAJECTORIES" "$PROBE_AUDIT" \
    "$TASKS_JSONL" "$EXPECTED_PROTOCOL_VERSION" "$EXPECTED_PROTOCOL_HASH" \
    "$EXPECTED_STUDENT_PROMPT_SHA256" "$EXPECTED_SFT1_SHA256" \
    "$PROTOCOL_RUNTIME" "$EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

manifest_path, trajectories_path, audit_path, tasks_path = map(Path, sys.argv[1:5])
expected_version, expected_hash, expected_prompt, expected_adapter = sys.argv[5:9]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads(manifest_path.read_text())
audit = json.loads(audit_path.read_text())
assert manifest["protocol_version"] == expected_version
assert manifest["protocol_hash"] == expected_hash
assert manifest["student_prompt_sha256"] == expected_prompt
assert manifest["protocol_runtime_root"] == str(Path(sys.argv[9]).resolve())
assert manifest["protocol_runtime_content_tree_sha256"] == sys.argv[10]
assert manifest["adapter_sha256"] == expected_adapter
assert manifest["tasks_sha256"] == sha(tasks_path)
assert manifest["tasks"] == 32
assert manifest["group_size"] == 8
assert manifest["trajectories"] == 256
assert manifest["max_new_tokens"] == 2048
assert manifest["max_context_tokens"] == 16384
assert manifest["history_turns"] == 4
assert manifest["enable_thinking"] is True
assert manifest["temperature"] == 0.8
assert manifest["top_p"] == 1.0
assert audit["schema_version"] == "vanilla-grpo-rollout-probe-audit-v1"
assert audit["status"]["passes"] is True
assert audit["status"]["probe_admitted"] is True
assert audit["inputs"]["manifest_sha256"] == sha(manifest_path)
assert audit["inputs"]["trajectories_sha256"] == sha(trajectories_path)

task_rows = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
expected_ids = [
    str(row.get("example_id") or row.get("instance_id")) for row in task_rows[:32]
]
trajectory_rows = [
    json.loads(line) for line in trajectories_path.read_text().splitlines() if line.strip()
]
observed_ids = [str(row["environment"]["task_id"]) for row in trajectory_rows]
assert set(observed_ids) == set(expected_ids)
assert Counter(observed_ids) == Counter({task_id: 8 for task_id in expected_ids})
PY
}

run_probe() {
  local generate_probe=1 unexpected=""
  if [[ -d "$PROBE_OUT" ]] && [[ -n "$(find "$PROBE_OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    if [[ -f "$PROBE_MANIFEST" && -f "$PROBE_TRAJECTORIES" ]]; then
      generate_probe=0
      unexpected=$(find "$PROBE_OUT" -mindepth 1 -maxdepth 1 \
        ! -name groups ! -name manifest.pending.json ! -name trajectories.jsonl \
        ! -name rollout_probe_audit.json -print -quit)
      [[ -z "$unexpected" ]] \
        || blocked "probe output contains an unknown top-level entry: $unexpected"
    else
      # The generator commits each completed K-group atomically and can safely
      # resume an interrupted probe. Only that exact partial shape is admitted;
      # any top-level output besides `groups/` remains fail-closed.
      unexpected=$(find "$PROBE_OUT" -mindepth 1 -maxdepth 1 ! -name groups -print -quit)
      [[ -z "$unexpected" && -d "$PROBE_OUT/groups" ]] \
        || blocked "refusing unknown/incomplete probe output: ${unexpected:-$PROBE_OUT}"
      if find "$PROBE_OUT/groups" -mindepth 1 -maxdepth 1 ! -type f -print -quit | grep -q .; then
        blocked "probe groups contain a non-file entry: $PROBE_OUT/groups"
      fi
      if find "$PROBE_OUT/groups" -mindepth 1 -maxdepth 1 -type f ! -name '*.json' -print -quit | grep -q .; then
        blocked "probe groups contain an unknown file: $PROBE_OUT/groups"
      fi
    fi
  else
    mkdir -p "$PROBE_OUT"
  fi
  if [[ "$generate_probe" -eq 1 ]]; then
    wait_for_gpu 1
    set_status probing "gpu=1 tasks=first32 k=8 reward=binary max_new_tokens=2048 context=16384 thinking=true resume_groups=true"
    setsid env \
      CUDA_VISIBLE_DEVICES=1 \
      HF_HUB_OFFLINE=1 \
      TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$PROTOCOL_RUNTIME" \
      PYTHONPATH="$TRAIN_RUNTIME/src/rl" \
      "$PYTHON_BIN" "$TRAIN_RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py" \
        --model-path "$MODEL_PATH" \
        --adapter-path "$SFT1_ADAPTER" \
        --tasks "$TASKS_JSONL" \
        --output-dir "$PROBE_OUT" \
        --limit 32 \
        --group-size 8 \
        --temperature 0.8 \
        --top-p 1 \
        --max-steps 30 \
        --max-new-tokens 2048 \
        --max-context-tokens 16384 \
        --history-turns 4 \
        --enable-thinking \
        --seed 20260812 \
        --gpu-memory-utilization 0.82 \
        --scheduler dynamic \
        --question-window 4 >>"$PROBE_LOG" 2>&1 &
    worker_pgid=$!
    local code=0
    if wait "$worker_pgid"; then code=0; else code=$?; fi
    terminate_session "$worker_pgid"
    worker_pgid=""
    [[ "$code" -eq 0 ]] || return "$code"
  fi
  set_status auditing_probe "manifest=$PROBE_MANIFEST trajectories=$PROBE_TRAJECTORIES"
  run_probe_audit
  verify_probe_gate
  set_terminal_status complete "probe_admitted=true audit=$PROBE_AUDIT"
}

validate_existing_train_manifest() {
  "$PYTHON_BIN" - "$TRAIN_OUT/run_manifest.json" "$EXPECTED_CONFIG_SHA256" \
    "$EXPECTED_TASKS_SHA256" "$EXPECTED_PROTOCOL_VERSION" "$EXPECTED_PROTOCOL_HASH" \
    "$EXPECTED_STUDENT_PROMPT_SHA256" "$EXPECTED_SFT1_SHA256" \
    "$PROTOCOL_RUNTIME" "$EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" <<'PY'
import json
import sys

path = sys.argv[1]
expected_config, expected_tasks, expected_version, expected_hash, expected_prompt, expected_adapter, expected_runtime, expected_tree = sys.argv[2:10]
manifest = json.load(open(path))
assert manifest["experiment_config_sha256"] == expected_config
assert manifest["examples_json_sha256"] == expected_tasks
assert manifest["protocol_version"] == expected_version
assert manifest["protocol_hash"] == expected_hash
assert manifest["student_prompt_sha256"] == expected_prompt
assert manifest["initial_adapter_sha256"] == expected_adapter
runtime_modules = manifest["runtime_module_audit"]
assert runtime_modules["runtime_root"] == expected_runtime
assert runtime_modules["tool_environment_factory_module"] == "tool_environment_v26"
assert runtime_modules["module_paths"] == {
    "protocol": expected_runtime + "/src/sft/protocol.py",
    "rollout": expected_runtime + "/src/eval/rollout.py",
    "executor": expected_runtime + "/src/harness/executor.py",
    "tool_schemes": expected_runtime + "/src/sft/tool_schemes.py",
}
identity = manifest["runtime_identity_audit"]
assert identity["expected"]["runtime_content_tree_sha256"] == expected_tree
assert identity["actual"]["runtime_content_tree_sha256"] == expected_tree
assert manifest["reward_mode"] == "result-only"
assert manifest["result_reward_profile"] == "binary"
assert manifest["policy_reduction"] == "trajectory_token_mean"
assert manifest["records"] == manifest["expected_records"] == 600
assert manifest["group_size"] == 8
assert manifest["prompts_per_update"] == 30
assert manifest["optimizer_steps"] == 20
assert manifest["ppo_iterations"] == 1
assert manifest["kl_beta"] == 0.0
assert manifest["rollout_settings"]["max_new_tokens"] == 2048
assert manifest["rollout_settings"]["max_context_tokens"] == 16384
assert manifest["rollout_settings"]["enable_thinking"] is True
reference = manifest["reference_policy"]
assert reference["enabled"] is False
assert reference["kl_beta"] == 0.0
assert reference["adapter_name"] is None
assert reference["adapter_path"] is None
assert reference["adapter_sha256"] is None
assert reference["expected_adapter_sha256"] == expected_adapter
assert reference["equals_initial_adapter"] is True
assert reference["load_audit"] is None
assert reference["after_trainer_init_audit"] is None
PY
}

RESUME_CHECKPOINT=""
TRAIN_ALREADY_COMPLETE=0
prepare_train_output() {
  if [[ -f "$TRAIN_OUT/final/adapter_model.safetensors" ]]; then
    [[ -f "$TRAIN_OUT/run_manifest.json" && -f "$TRAIN_OUT/training_precision.json" ]] \
      || blocked "final training output is incomplete: $TRAIN_OUT"
    validate_existing_train_manifest
    TRAIN_ALREADY_COMPLETE=1
    return
  fi
  if [[ -e "$TRAIN_OUT/final" ]]; then
    blocked "refusing to overwrite an incomplete final directory: $TRAIN_OUT/final"
  fi
  if [[ ! -d "$TRAIN_OUT" ]] || [[ -z "$(find "$TRAIN_OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    mkdir -p "$TRAIN_OUT"
    return
  fi
  [[ -f "$TRAIN_OUT/run_manifest.json" && -f "$TRAIN_OUT/implementation_lock.json" ]] \
    || blocked "nonempty training output lacks immutable run locks: $TRAIN_OUT"
  validate_existing_train_manifest

  local checkpoint base step latest_step=-1 latest=""
  shopt -s nullglob
  for checkpoint in "$TRAIN_OUT"/checkpoint-*; do
    [[ -d "$checkpoint" ]] || continue
    base=${checkpoint##*/}
    [[ "$base" =~ ^checkpoint-([0-9]+)$ ]] || blocked "unexpected checkpoint directory: $checkpoint"
    step=${BASH_REMATCH[1]}
    if (( 10#$step > latest_step )); then
      latest_step=$((10#$step))
      latest=$checkpoint
    fi
  done
  shopt -u nullglob
  [[ -n "$latest" ]] || blocked "nonempty training output has no resumable checkpoint: $TRAIN_OUT"
  for required in adapter_model.safetensors adapter_config.json optimizer.pt scheduler.pt trainer_state.json rng_state.pth training_args.bin; do
    [[ -f "$latest/$required" ]] || blocked "latest checkpoint is incomplete: $latest/$required"
  done
  "$PYTHON_BIN" - "$latest/trainer_state.json" "$latest_step" <<'PY'
import json, sys
state = json.load(open(sys.argv[1]))
step = int(sys.argv[2])
assert int(state["global_step"]) == step
assert 0 < step <= 20
PY
  RESUME_CHECKPOINT=$latest
}

start_train_vllm() {
  if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
    blocked "port $VLLM_PORT already has a server; refusing to reuse an unowned vLLM"
  fi
  set_status starting_vllm "gpu=1 port=$VLLM_PORT max_model_len=16384"
  setsid env \
    CUDA_VISIBLE_DEVICES=1 \
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
    if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "$ready" -eq 1 ]] || blocked "owned vLLM failed readiness; see $VLLM_LOG"
}

run_train() {
  verify_probe_gate
  prepare_train_output
  if [[ "$TRAIN_ALREADY_COMPLETE" -eq 1 ]]; then
    set_terminal_status complete "existing validated final=$TRAIN_OUT/final"
    return
  fi
  wait_for_gpu 0
  wait_for_gpu 1
  start_train_vllm

  local resume_args=()
  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    resume_args=(--resume-from-checkpoint "$RESUME_CHECKPOINT")
    set_status resuming "checkpoint=$RESUME_CHECKPOINT gpu0=trainer gpu1=vllm"
  else
    set_status training "records=600 prompts_per_update=30 k=8 updates=20 save_every=2 reward=binary"
  fi
  setsid env \
    CUDA_VISIBLE_DEVICES=0 \
    PROJECT_DIR="$TRAIN_RUNTIME" \
    PYTHON="$PYTHON_BIN" \
    MODEL_PATH="$MODEL_PATH" \
    ADAPTER_PATH="$SFT1_ADAPTER" \
    EXAMPLES_JSON="$TASKS_JSONL" \
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
      --seed 20260812 \
      "${resume_args[@]}" >>"$TRAIN_LOG" 2>&1 &
  worker_pgid=$!
  local code=0
  if wait "$worker_pgid"; then code=0; else code=$?; fi
  terminate_session "$worker_pgid"
  worker_pgid=""
  terminate_session "$vllm_pgid"
  vllm_pgid=""
  [[ "$code" -eq 0 ]] || return "$code"

  [[ -f "$TRAIN_OUT/final/adapter_model.safetensors" ]] \
    || blocked "trainer exited without a final adapter: $TRAIN_OUT"
  [[ -f "$TRAIN_OUT/training_precision.json" ]] \
    || blocked "trainer exited without precision audit: $TRAIN_OUT"
  validate_existing_train_manifest
  set_terminal_status complete "train=$TRAIN_OUT final=$TRAIN_OUT/final"
}

validate_static_inputs
case "$MODE" in
  probe) run_probe ;;
  train) run_train ;;
esac
