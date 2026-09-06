#!/usr/bin/env bash
# Isolated end-to-end agent-rollout throughput benchmark for one 24 GiB RTX 3090.
set -euo pipefail

RUNTIME=${RUNTIME:-/home/dengyan/tabular_rl_outputs/rl_runtime_phase8_parallel_20260801}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
SOURCE_POOL=${SOURCE_POOL:-$OUTPUT_ROOT/phase8_mixed_pool_search_20260801/candidate_360_seed101}
BENCHMARK_ID=${BENCHMARK_ID:-rtx3090_agent_rollout_20260801}
BENCH_ROOT=${BENCH_ROOT:-$OUTPUT_ROOT/benchmarks/$BENCHMARK_ID}
PYTHON_BIN=${PYTHON_BIN:-/home/dengyan/miniconda3/envs/trl-table/bin/python}
MODEL=${MODEL:-/home/dengyan/models/Qwen2.5-Coder-7B-Instruct}
ADAPTER=${ADAPTER:-$OUTPUT_ROOT/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682}
GPU_ID=${GPU_ID:-6}
GROUP_SIZE=4
EXPECTED_TASKS=12
TASK_BATCH_SIZES=(${TASK_BATCH_SIZES:-2 3 4 6 8 12})
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.82}
SCHEDULER=${SCHEDULER:-static}
QUESTION_WINDOW=${QUESTION_WINDOW:-8}
STATUS=${STATUS:-$OUTPUT_ROOT/logs/${BENCHMARK_ID}.status}
RUN_LOG=${RUN_LOG:-$OUTPUT_ROOT/logs/${BENCHMARK_ID}.log}
LOCK=${LOCK:-$OUTPUT_ROOT/logs/${BENCHMARK_ID}.lock}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }

mkdir -p "$BENCH_ROOT" "$OUTPUT_ROOT/logs"
exec 9>"$LOCK"
if ! flock -n 9; then
  set_status duplicate "another benchmark owns the lock"
  exit 0
fi
exec >>"$RUN_LOG" 2>&1

[[ "$GPU_ID" == 6 ]] || { set_status failed "shared-server benchmark is restricted to GPU6"; exit 2; }
test -x "$PYTHON_BIN"
test -d "$MODEL"
test -f "$ADAPTER/adapter_model.safetensors"
test -f "$SOURCE_POOL/tasks.jsonl"
test "$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')" -eq 0
test "$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')" -le 512

cat >"$BENCH_ROOT/task_ids.txt.next" <<'EOF'
bird_train_04596
bird_train_00562
bird_train_02010
bird_train_03809
bird_train_04460
bird_train_02886
bird_train_02913
bird_train_03931
bird_train_03642
bird_train_01902
bird_train_01378
bird_train_02084
EOF
if [[ -f "$BENCH_ROOT/task_ids.txt" ]]; then
  cmp "$BENCH_ROOT/task_ids.txt" "$BENCH_ROOT/task_ids.txt.next"
  rm -f "$BENCH_ROOT/task_ids.txt.next"
else
  mv "$BENCH_ROOT/task_ids.txt.next" "$BENCH_ROOT/task_ids.txt"
fi
test "$(wc -l <"$BENCH_ROOT/task_ids.txt" | tr -d '[:space:]')" -eq "$EXPECTED_TASKS"

"$PYTHON_BIN" - "$SOURCE_POOL/tasks.jsonl" "$BENCH_ROOT/task_ids.txt" "$BENCH_ROOT/benchmark_manifest.json" "$GPU_MEMORY_UTILIZATION" "$SCHEDULER" "$QUESTION_WINDOW" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

tasks_path, ids_path, output_path = map(Path, sys.argv[1:4])
gpu_memory_utilization = float(sys.argv[4])
scheduler = sys.argv[5]
question_window = int(sys.argv[6])
rows = [json.loads(line) for line in tasks_path.open() if line.strip()]
task_map = {
    str(row.get("example_id") or row.get("instance_id") or row.get("trajectory_id")): row
    for row in rows
}
ids = [line.strip() for line in ids_path.open() if line.strip()]
assert len(ids) == len(set(ids)) == 12
assert not (set(ids) - set(task_map))
levels = [str((task_map[value].get("metadata") or {}).get("fixed_pool_difficulty")) for value in ids]
assert {level: levels.count(level) for level in set(levels)} == {
    "simple": 4,
    "moderate": 4,
    "challenging": 4,
}
payload = {
    "schema_version": "rtx3090-agent-rollout-benchmark-v1",
    "purpose": "operational throughput only; outputs are excluded from training pools",
    "task_ids": ids,
    "task_ids_sha256": hashlib.sha256(ids_path.read_bytes()).hexdigest(),
    "tasks_source": str(tasks_path),
    "tasks_source_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
    "difficulty_counts": {level: levels.count(level) for level in sorted(set(levels))},
    "gpu": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
    ).splitlines()[6],
    "group_size": 4,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_steps": 30,
    "max_new_tokens": 1024,
    "max_context_tokens": 8192,
    "history_turns": 4,
    "gpu_memory_utilization": gpu_memory_utilization,
    "enable_prefix_caching": True,
    "enforce_eager": True,
    "scheduler": scheduler,
    "question_window": question_window if scheduler == "dynamic" else None,
}
if output_path.exists():
    assert json.loads(output_path.read_text()) == payload
else:
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

summarize_variant() {
  local batch_size=$1 variant=$2 wall=$3 code=$4 gpu_csv=$5
  "$PYTHON_BIN" - "$variant" "$batch_size" "$wall" "$code" "$gpu_csv" <<'PY'
import csv
import json
import math
import statistics
import sys
from pathlib import Path

variant = Path(sys.argv[1])
batch_size = int(sys.argv[2])
wall = float(sys.argv[3])
code = int(sys.argv[4])
gpu_csv = Path(sys.argv[5])
groups = sorted((variant / "groups").glob("*.json")) if (variant / "groups").exists() else []
episodes = turns = prompt_tokens = completion_tokens = correct = 0
for path in groups:
    for row in json.loads(path.read_text()):
        episodes += 1
        correct += int(bool(row["sample"]["correct"]))
        for turn in row["policy_turns"]:
            turns += 1
            prompt_tokens += len(turn["prompt_ids"])
            completion_tokens += len(turn["response_ids"])

gpu_rows = []
if gpu_csv.exists():
    with gpu_csv.open() as source:
        for raw in csv.reader(source):
            if len(raw) < 5:
                continue
            try:
                gpu_rows.append(tuple(float(value.strip()) for value in raw[:5]))
            except ValueError:
                continue

def pct(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]

utils = [row[0] for row in gpu_rows]
memory = [row[2] for row in gpu_rows]
power = [row[3] for row in gpu_rows]
payload = {
    "schema_version": "rtx3090-agent-rollout-variant-result-v1",
    "task_batch_size": batch_size,
    "max_active_trajectories": batch_size * 4,
    "exit_code": code,
    "stable": code == 0 and len(groups) == 12 and episodes == 48,
    "wall_seconds": wall,
    "groups": len(groups),
    "episodes": episodes,
    "correct_episodes": correct,
    "turns": turns,
    "prompt_tokens": prompt_tokens,
    "completion_tokens": completion_tokens,
    "tasks_per_hour": len(groups) / wall * 3600 if wall else None,
    "episodes_per_hour": episodes / wall * 3600 if wall else None,
    "turns_per_second": turns / wall if wall else None,
    "completion_tokens_per_second": completion_tokens / wall if wall else None,
    "gpu_samples": len(gpu_rows),
    "gpu_util_mean": statistics.fmean(utils) if utils else None,
    "gpu_util_p95": pct(utils, 0.95),
    "gpu_memory_peak_mib": max(memory) if memory else None,
    "gpu_power_mean_w": statistics.fmean(power) if power else None,
}
(variant / "benchmark_result.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
}

for batch_size in "${TASK_BATCH_SIZES[@]}"; do
  variant="$BENCH_ROOT/task_batch_${batch_size}"
  result="$variant/benchmark_result.json"
  if [[ -f "$result" ]] && "$PYTHON_BIN" -c 'import json,sys; assert json.load(open(sys.argv[1]))["stable"]' "$result"; then
    echo "skip stable batch_size=$batch_size"
    continue
  fi
  if [[ -e "$variant" ]]; then
    set_status failed "partial benchmark directory requires audit batch_size=$batch_size path=$variant"
    exit 3
  fi
  mkdir -p "$variant"
  gpu_csv="$variant/gpu_samples.csv"
  : >"$gpu_csv"
  (
    while true; do
      nvidia-smi -i "$GPU_ID" \
        --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm \
        --format=csv,noheader,nounits >>"$gpu_csv" || true
      sleep 1
    done
  ) &
  sampler_pid=$!
  start_ns=$(date +%s%N)
  set_status running "scheduler=$SCHEDULER batch_size=$batch_size max_active=$((batch_size * GROUP_SIZE))"
  set +e
  cd "$RUNTIME"
  CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HUB_OFFLINE=1 \
    /usr/bin/time -v -o "$variant/time_verbose.txt" \
    "$PYTHON_BIN" src/rl/fixed_pool/generate_fixed_rollout_pool.py \
      --model-path "$MODEL" \
      --adapter-path "$ADAPTER" \
      --tasks "$SOURCE_POOL/tasks.jsonl" \
      --task-id-file "$BENCH_ROOT/task_ids.txt" \
      --output-dir "$variant" \
      --group-size "$GROUP_SIZE" \
      --temperature 0.7 --top-p 0.95 \
      --max-steps 30 --max-new-tokens 1024 --max-context-tokens 8192 \
      --history-turns 4 --seed 101 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --task-batch-size "$batch_size" --scheduler "$SCHEDULER" \
      --question-window "$QUESTION_WINDOW" --no-finalize \
      >"$variant/generation.log" 2>&1
  code=$?
  set -e
  end_ns=$(date +%s%N)
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  wall=$("$PYTHON_BIN" -c 'import sys; print((int(sys.argv[2])-int(sys.argv[1]))/1e9)' "$start_ns" "$end_ns")
  summarize_variant "$batch_size" "$variant" "$wall" "$code" "$gpu_csv"
  if [[ "$code" -ne 0 ]]; then
    set_status stopped_at_limit "batch_size=$batch_size exit=$code; see $variant/generation.log"
    break
  fi
  sleep 5
done

"$PYTHON_BIN" - "$BENCH_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("task_batch_*/benchmark_result.json")):
    rows.append(json.loads(path.read_text()))
stable = [row for row in rows if row["stable"]]
if not stable:
    raise SystemExit("no stable benchmark variant")
best = max(stable, key=lambda row: (row["tasks_per_hour"], row["completion_tokens_per_second"]))
summary = {
    "schema_version": "rtx3090-agent-rollout-benchmark-summary-v1",
    "selection_rule": "maximum end-to-end tasks/hour; completion-token throughput breaks ties",
    "tested": rows,
    "best": best,
    "largest_stable_task_batch_size": max(row["task_batch_size"] for row in stable),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

set_status complete "summary=$BENCH_ROOT/summary.json"
