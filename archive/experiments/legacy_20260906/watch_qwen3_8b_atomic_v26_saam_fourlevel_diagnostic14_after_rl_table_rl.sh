#!/usr/bin/env bash
# After the one-update diagnostic finishes, compare initial and final adapters
# on the same frozen 14-task set. This is an in-sample behavior diagnostic.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
RUNTIME=${RUNTIME:-$OUTPUT_ROOT/rl_runtime_qwen3_8b_v26_diagnostic_current_20260904}
PYTHON=${PYTHON:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
INITIAL_ADAPTER=${INITIAL_ADAPTER:-$OUTPUT_ROOT/checkpoints/checkpoint-6380}
PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de}
TASKS=${TASKS:-$OUTPUT_ROOT/qwen3_v26_4k_rl_diagnostic14_table_rl_20260904/tasks/qwen3_8b_atomic_v26_saam_fourlevel_diagnostic14_v1.jsonl}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/qwen3_v26_4k_rl_diagnostic14_table_rl_20260904_run3}
TRAIN_OUT=${TRAIN_OUT:-$RUN_ROOT/train}
EVAL_ROOT=${EVAL_ROOT:-$RUN_ROOT/matched_eval}

# The fixed-pool entrypoint adds src/rl to sys.path itself, while the
# Atomic-v26 environment imports the sibling src/tool_modules package.
# Keep both runtime roots explicit so the post-training evaluator works when
# launched detached by the watcher/automation.
export PYTHONPATH="$RUNTIME/src:$RUNTIME/src/rl:${PYTHONPATH:-}"

mkdir -p "$EVAL_ROOT/logs"
for _ in $(seq 1 1440); do
  [[ -f "$RUN_ROOT/status" ]] || { sleep 60; continue; }
  state=$(awk -F '\t' 'END {print $2}' "$RUN_ROOT/status")
  [[ "$state" == "complete" ]] && break
  [[ "$state" == "failed" ]] && { echo "RL run failed" >&2; exit 1; }
  sleep 60
done
[[ -f "$RUN_ROOT/status" && "$(awk -F '\t' 'END {print $2}' "$RUN_ROOT/status")" == "complete" ]] || {
  echo "timed out waiting for RL completion" >&2; exit 1;
}
FINAL_ADAPTER="$TRAIN_OUT/final"
[[ -f "$FINAL_ADAPTER/adapter_model.safetensors" ]] || { echo "missing final adapter" >&2; exit 1; }

gpu_idle() {
  local gpu=$1 pids used
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
  [[ -z "$pids" && "$used" =~ ^[0-9]+$ && "$used" -le 512 ]]
}
for _ in $(seq 1 180); do gpu_idle 0 && gpu_idle 1 && break; sleep 5; done
gpu_idle 0 && gpu_idle 1 || { echo "evaluation GPUs are busy" >&2; exit 1; }

run_eval() {
  local gpu=$1 adapter=$2 out=$3 label=$4
  # RolloutSettings currently requires a strictly positive temperature;
  # 1e-5 is effectively greedy while remaining valid for the evaluator.
  CUDA_VISIBLE_DEVICES="$gpu" TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$PROTOCOL_RUNTIME" \
    "$PYTHON" -u "$RUNTIME/src/rl/fixed_pool/generate_fixed_rollout_pool.py" \
    --model-path "$MODEL_PATH" --adapter-path "$adapter" --tasks "$TASKS" \
    --output-dir "$out" --group-size 1 --temperature 1e-5 --top-p 1 \
    --max-steps 30 --max-new-tokens 2048 --max-context-tokens 16384 \
    --history-turns 4 --enable-thinking --seed 20260905 \
    --gpu-memory-utilization 0.82 --task-batch-size 14 >"$EVAL_ROOT/logs/${label}.log" 2>&1
}
run_eval 0 "$INITIAL_ADAPTER" "$EVAL_ROOT/initial" initial & pid_initial=$!
run_eval 1 "$FINAL_ADAPTER" "$EVAL_ROOT/final" final & pid_final=$!
wait "$pid_initial"; wait "$pid_final"

INITIAL="$EVAL_ROOT/initial/trajectories.jsonl" FINAL="$EVAL_ROOT/final/trajectories.jsonl" \
  OUT="$EVAL_ROOT/summary.json" TASKS="$TASKS" "$PYTHON" - <<'PY'
import json, os
from pathlib import Path

def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def summary(data):
    correct=sum(bool(x.get("sample",{}).get("correct",False)) for x in data)
    legal=sum(bool(x.get("sample",{}).get("legal",False)) for x in data)
    n=len(data)
    return {"records":n,"correct":correct,"legal":legal,"correct_rate":correct/n if n else 0.0,"legal_rate":legal/n if n else 0.0}

a=rows(os.environ["INITIAL"]); b=rows(os.environ["FINAL"])
def key(x):
    audit=x.get("sample",{}).get("audit_record",{})
    return str(audit.get("task_id",x.get("sequence")))
by_a={key(x):x for x in a}; by_b={key(x):x for x in b}
changes=[]
for task_id in sorted(set(by_a)&set(by_b)):
    ca=bool(by_a[task_id].get("sample",{}).get("correct",False))
    cb=bool(by_b[task_id].get("sample",{}).get("correct",False))
    if ca != cb: changes.append({"task_id":task_id,"initial_correct":ca,"final_correct":cb})
out={"schema_version":"qwen3-v26-diagnostic14-matched-eval-v1","initial":summary(a),"final":summary(b),"changed_tasks":changes,"initial_path":os.environ["INITIAL"],"final_path":os.environ["FINAL"],"tasks_path":os.environ["TASKS"]}
Path(os.environ["OUT"]).write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(out,ensure_ascii=False))
PY
echo "MATCHED_EVAL_COMPLETE summary=$EVAL_ROOT/summary.json"
