#!/usr/bin/env python3
"""Experiment dashboard API and static server with no third-party Python dependencies."""
from __future__ import annotations

import argparse
import importlib.util
import json
import mimetypes
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DASHBOARD_ROOT.parent
REGISTRY_PATH = DASHBOARD_ROOT / "data" / "experiments.json"
DIST_ROOT = DASHBOARD_ROOT / "frontend" / "dist"
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:18000/v1").rstrip("/")
SSH_HOST = os.environ.get("EXPERIMENT_SSH_HOST", "NewGNN")
REMOTE_PROJECT_ROOT = os.environ.get(
    "EXPERIMENT_REMOTE_ROOT", "/home/dengyan/tabular_rl_project"
)
REMOTE_VLLM_ENV = os.environ.get(
    "EXPERIMENT_VLLM_ENV", "/home/dengyan/miniconda3/envs/vllm"
)
REMOTE_VLLM_PORT = int(os.environ.get("EXPERIMENT_VLLM_PORT", "8000"))
REMOTE_VLLM_PID = f"{REMOTE_PROJECT_ROOT}/logs/dashboard_vllm.pid"
REMOTE_VLLM_META = f"{REMOTE_PROJECT_ROOT}/logs/dashboard_vllm.json"
REMOTE_VLLM_LOG = f"{REMOTE_PROJECT_ROOT}/logs/dashboard_vllm.log"
MAX_BODY = 2 * 1024 * 1024
_eval_cache: dict[str, tuple[float, dict]] = {}


def load_protocol():
    path = REPO_ROOT / "src" / "sft" / "protocol.py"
    spec = importlib.util.spec_from_file_location("dashboard_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load protocol from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROTOCOL = load_protocol()


def load_attribution():
    """Load the sibling failure-attribution module by path (same idiom as load_protocol)."""
    path = Path(__file__).resolve().parent / "attribution.py"
    spec = importlib.util.spec_from_file_location("dashboard_attribution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load attribution from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ATTRIBUTION = load_attribution()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError("path escapes repository")
    return path


def load_registry() -> list[dict]:
    return read_json(REGISTRY_PATH)


def find_experiment(experiment_id: str) -> dict:
    for experiment in load_registry():
        if experiment["id"] == experiment_id:
            return experiment
    raise KeyError(experiment_id)


def jsonl_page(
    path: Path,
    page: int,
    page_size: int,
    min_steps: int | None = None,
    max_steps: int | None = None,
) -> dict:
    start = (page - 1) * page_size
    records = []
    total = 0
    has_filter = min_steps is not None or max_steps is not None
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            # When filtering by trajectory length we must parse every line to count its steps;
            # otherwise keep the cheap path that only deserialises the records on the current page.
            record = json.loads(line) if has_filter else None
            if has_filter:
                steps = ATTRIBUTION.record_step_count(record)
                if (min_steps is not None and steps < min_steps) or (
                    max_steps is not None and steps > max_steps
                ):
                    continue
            if start <= total < start + page_size:
                records.append(
                    {"index": total, "record": record if record is not None else json.loads(line)}
                )
            total += 1
    return {
        "records": records,
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": max(1, (total + page_size - 1) // page_size),
    }


def json_page(path: Path) -> dict:
    value = read_json(path)
    if isinstance(value, list):
        records = [{"index": index, "record": record} for index, record in enumerate(value)]
    else:
        records = [{"index": 0, "record": value}]
    return {
        "records": records,
        "page": 1,
        "page_size": len(records),
        "total": len(records),
        "pages": 1,
    }


def data_sources(experiment: dict) -> list[dict]:
    candidates = []

    def add(source_id: str, label: str, group: str, value: str | None):
        if not value:
            return
        path = resolve_repo_path(value)
        if not path.is_file():
            return
        candidates.append({
            "id": source_id,
            "label": label,
            "group": group,
            "path": value,
            "format": "jsonl" if path.suffix == ".jsonl" else "json",
            "size_bytes": path.stat().st_size,
        })

    dataset = experiment.get("dataset", {})
    add("train", "训练集", "dataset", dataset.get("train"))
    add("dev", "验证集", "dataset", dataset.get("dev"))
    add("train_manifest", "训练 Manifest", "metadata", dataset.get("train_manifest"))
    add("dev_manifest", "验证 Manifest", "metadata", dataset.get("dev_manifest"))

    training = experiment.get("training", {})
    add("trainer_state", "Trainer State", "training", training.get("trainer_state"))

    evaluation_dirs = []
    primary_dir = experiment.get("evaluation", {}).get("directory")
    if primary_dir:
        evaluation_dirs.append(("eval", "完整评测", primary_dir))
    for index, related in enumerate(experiment.get("related_evaluations", []), start=1):
        evaluation_dirs.append((
            f"related_{index}",
            related.get("label") or f"附加评测 {index}",
            related.get("directory"),
        ))
    for prefix, run_label, directory in evaluation_dirs:
        if not directory:
            continue
        add(f"{prefix}_all", f"{run_label} · 全部案例", "evaluation", f"{directory}/all.jsonl")
        add(
            f"{prefix}_success",
            f"{run_label} · 成功案例",
            "evaluation",
            f"{directory}/success.jsonl",
        )
        add(
            f"{prefix}_failure",
            f"{run_label} · 失败案例",
            "evaluation",
            f"{directory}/failure.jsonl",
        )
        add(
            f"{prefix}_manifest",
            f"{run_label} · 运行配置",
            "metadata",
            f"{directory}/manifest.json",
        )
        add(
            f"{prefix}_summary",
            f"{run_label} · 结果汇总",
            "metadata",
            f"{directory}/summary.json",
        )
    return candidates


def evaluation_summary(directory: Path) -> dict:
    all_path = directory / "all.jsonl"
    if not all_path.exists():
        return {"available": False}
    cache_key = str(all_path)
    mtime = all_path.stat().st_mtime
    cached = _eval_cache.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1]

    total = correct = legal = legal_observed = 0
    total_steps = total_elapsed = 0.0
    steps_observed = elapsed_observed = 0
    failures = Counter()
    with all_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            total += 1
            correct += int(record.get("correct") is True)
            if isinstance(record.get("legal"), bool):
                legal_observed += 1
                legal += int(record["legal"])
            if record.get("steps") is not None:
                steps_observed += 1
                total_steps += float(record["steps"])
            if record.get("elapsed_seconds") is not None:
                elapsed_observed += 1
                total_elapsed += float(record["elapsed_seconds"])
            if record.get("correct") is not True:
                failures[record.get("failure_type") or "unknown"] += 1
    summary = {
        "available": True,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "legal": legal if legal_observed else None,
        "legal_rate": legal / legal_observed if legal_observed else None,
        "legal_observed": legal_observed,
        "average_steps": total_steps / steps_observed if steps_observed else None,
        "average_elapsed_seconds": total_elapsed / elapsed_observed if elapsed_observed else None,
        "failure_types": dict(failures),
        "complete_dev": total == 1034,
    }
    _eval_cache[cache_key] = (mtime, summary)
    return summary


def training_metrics(experiment: dict) -> dict:
    trainer_state = experiment.get("training", {}).get("trainer_state")
    if not trainer_state:
        return {"available": False}
    state_path = resolve_repo_path(trainer_state)
    if not state_path.exists():
        return {"available": False}
    state = read_json(state_path)
    train_points = []
    eval_points = []
    final_metrics = {}
    for entry in state.get("log_history", []):
        common = {"step": entry.get("step"), "epoch": entry.get("epoch")}
        if "loss" in entry:
            train_points.append({
                **common,
                "loss": entry["loss"],
                "learning_rate": entry.get("learning_rate"),
                "grad_norm": entry.get("grad_norm"),
            })
        if "eval_loss" in entry:
            eval_points.append({**common, "loss": entry["eval_loss"]})
        if "train_loss" in entry:
            final_metrics = {
                "train_loss": entry.get("train_loss"),
                "train_runtime": entry.get("train_runtime"),
                "train_samples_per_second": entry.get("train_samples_per_second"),
                "train_steps_per_second": entry.get("train_steps_per_second"),
                "total_flos": entry.get("total_flos"),
            }
    summary = {
        "global_step": state.get("global_step"),
        "max_steps": state.get("max_steps"),
        "epoch": state.get("epoch"),
        "best_metric": state.get("best_metric"),
        **final_metrics,
        "last_train_loss": train_points[-1]["loss"] if train_points else None,
        "last_eval_loss": eval_points[-1]["loss"] if eval_points else None,
    }
    return {
        "available": True,
        "summary": summary,
        "train": train_points,
        "eval": eval_points,
        "updated_at": state_path.stat().st_mtime,
    }


def dataset_summary(experiment: dict) -> dict:
    result = {}
    for split in ("train", "dev"):
        manifest_value = experiment.get("dataset", {}).get(f"{split}_manifest")
        if manifest_value:
            path = resolve_repo_path(manifest_value)
            result[split] = read_json(path) if path.exists() else {"available": False}
    return result


def enrich_experiment(experiment: dict) -> dict:
    result = dict(experiment)
    result["training_metrics"] = training_metrics(experiment)
    result["dataset_summary"] = dataset_summary(experiment)
    eval_dir = experiment.get("evaluation", {}).get("directory")
    result["evaluation_summary"] = (
        evaluation_summary(resolve_repo_path(eval_dir)) if eval_dir else {"available": False}
    )
    result["data_sources"] = data_sources(experiment)
    return result


def refresh_trainer_state(experiment: dict) -> dict:
    remote = experiment.get("training", {}).get("remote_trainer_state")
    if not remote:
        raise ValueError("experiment has no remote trainer state")
    process = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", SSH_HOST, "cat", remote],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    state = json.loads(process.stdout)
    local = resolve_repo_path(experiment["training"]["trainer_state"])
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return training_metrics(experiment)


def update_experiment(experiment_id: str, patch: dict) -> dict:
    allowed = {"notes", "status", "tags", "description", "completed_at"}
    unknown = set(patch) - allowed
    if unknown:
        raise ValueError(f"unsupported fields: {sorted(unknown)}")
    registry = load_registry()
    for experiment in registry:
        if experiment["id"] == experiment_id:
            experiment.update(patch)
            REGISTRY_PATH.write_text(
                json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return experiment
    raise KeyError(experiment_id)


def ssh_run(
    command: list[str],
    *,
    input_text: str | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    remote_command = " ".join(shlex.quote(part) for part in command)
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", SSH_HOST, remote_command],
        check=True,
        capture_output=True,
        text=True,
        input=input_text,
        timeout=timeout,
    )


def launchable_models() -> list[dict]:
    models = []
    for experiment in load_registry():
        training = experiment.get("training", {})
        checkpoint = training.get("checkpoint")
        base_model = experiment.get("model")
        if not checkpoint or not base_model:
            continue
        models.append({
            "experiment_id": experiment["id"],
            "name": experiment["name"],
            "served_model_name": experiment["id"],
            "base_model": base_model,
            "adapter_path": checkpoint,
            "max_model_len": training.get("cutoff_len", 8192),
            "lora_rank": training.get("lora_rank", 16),
        })
    return models


def playground_config() -> dict:
    return {
        "system_prompt": PROTOCOL.SYSTEM_PROMPT,
        "protocol_version": PROTOCOL.PROTOCOL_VERSION,
        "protocol_hash": PROTOCOL.protocol_hash(),
        "models": launchable_models(),
        "vllm_base_url": VLLM_BASE_URL,
    }


GPU_STATUS_SCRIPT = r"""
import json
import subprocess

def output(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()

gpu_lines = output([
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
]).splitlines()
gpus = []
by_uuid = {}
for line in gpu_lines:
    index, uuid, name, total, used, utilization = [part.strip() for part in line.split(",", 5)]
    gpu = {
        "index": int(index),
        "uuid": uuid,
        "name": name,
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "utilization_percent": int(utilization),
        "processes": [],
    }
    gpus.append(gpu)
    by_uuid[uuid] = gpu

apps = subprocess.run([
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
    "--format=csv,noheader,nounits",
], capture_output=True, text=True).stdout.strip().splitlines()
app_rows = []
for line in apps:
    if not line.strip():
        continue
    uuid, pid, process_name, used = [part.strip() for part in line.split(",", 3)]
    app_rows.append((uuid, int(pid), process_name, int(used)))

pids = [str(row[1]) for row in app_rows]
owners = {}
commands = {}
if pids:
    ps = subprocess.run(
        ["ps", "-o", "pid=,user=,args=", "-p", ",".join(pids)],
        capture_output=True, text=True,
    ).stdout.splitlines()
    for line in ps:
        parts = line.strip().split(None, 2)
        if len(parts) >= 2:
            owners[int(parts[0])] = parts[1]
            commands[int(parts[0])] = parts[2] if len(parts) == 3 else ""

for uuid, pid, process_name, used in app_rows:
    gpu = by_uuid.get(uuid)
    if gpu is not None:
        gpu["processes"].append({
            "pid": pid,
            "user": owners.get(pid, "unknown"),
            "name": process_name,
            "command": commands.get(pid, ""),
            "memory_used_mib": used,
        })

print(json.dumps(gpus))
"""


def remote_gpu_status() -> list[dict]:
    process = ssh_run(["python3", "-"], input_text=GPU_STATUS_SCRIPT)
    gpus = json.loads(process.stdout)
    for gpu in gpus:
        gpu["available"] = (
            gpu["memory_used_mib"] <= 512
            and gpu["utilization_percent"] <= 10
            and not gpu["processes"]
        )
    return gpus


VLLM_STATUS_SCRIPT = r"""
import json
import os
import subprocess
import sys

pid_path, meta_path, log_path = sys.argv[1:4]
result = {"managed": False, "running": False, "log_path": log_path}
if os.path.exists(meta_path):
    try:
        result.update(json.load(open(meta_path)))
        result["managed"] = True
    except Exception as error:
        result["metadata_error"] = str(error)
if os.path.exists(pid_path):
    try:
        pid = int(open(pid_path).read().strip())
        check = subprocess.run(
            ["ps", "-o", "user=,args=", "-p", str(pid)],
            capture_output=True, text=True,
        )
        result["pid"] = pid
        result["running"] = check.returncode == 0 and bool(check.stdout.strip())
        if check.stdout.strip():
            owner, _, command = check.stdout.strip().partition(" ")
            result["owner"] = owner
            result["command"] = command.strip()
    except Exception as error:
        result["pid_error"] = str(error)
if os.path.exists(log_path):
    with open(log_path, errors="replace") as source:
        result["log_tail"] = source.readlines()[-30:]
print(json.dumps(result))
"""


def managed_vllm_status(include_gpus: bool = True) -> dict:
    process = ssh_run(
        ["python3", "-", REMOTE_VLLM_PID, REMOTE_VLLM_META, REMOTE_VLLM_LOG],
        input_text=VLLM_STATUS_SCRIPT,
    )
    result = json.loads(process.stdout)
    status, _ = vllm_request("GET", "models", timeout=3)
    result["ready"] = status == HTTPStatus.OK
    if include_gpus:
        result["gpus"] = remote_gpu_status()
    return result


START_VLLM_SCRIPT = r"""
set -euo pipefail
gpu="$1"
adapter="$2"
served_name="$3"
base_model="$4"
max_model_len="$5"
lora_rank="$6"
vllm_bin="$7"
port="$8"
pid_file="$9"
meta_file="${10}"
log_file="${11}"

mkdir -p "$(dirname "$pid_file")"
if [[ -f "$pid_file" ]]; then
  old_pid="$(cat "$pid_file")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "dashboard-managed vLLM is already running with PID $old_pid" >&2
    exit 20
  fi
fi
if [[ ! -f "$adapter/adapter_config.json" ]]; then
  echo "adapter_config.json not found in $adapter" >&2
  exit 21
fi
if ss -ltn 2>/dev/null | grep -q ":${port} "; then
  echo "port $port is already in use" >&2
  exit 22
fi
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((gpu + 1))p")"
util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sed -n "$((gpu + 1))p")"
uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n "$((gpu + 1))p" | xargs)"
compute_count="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -Fxc "$uuid" || true)"
if (( used > 512 || util > 10 || compute_count > 0 )); then
  echo "GPU $gpu is busy: ${used} MiB, ${util}% utilization" >&2
  exit 23
fi

: > "$log_file"
nohup env \
  CUDA_VISIBLE_DEVICES="$gpu" \
  HF_HUB_OFFLINE=1 \
  PYTHONUNBUFFERED=1 \
  "$vllm_bin" serve "$base_model" \
  --host 127.0.0.1 \
  --port "$port" \
  --served-model-name "$base_model" \
  --enable-lora \
  --lora-modules "${served_name}=${adapter}" \
  --max-lora-rank "$lora_rank" \
  --max-model-len "$max_model_len" \
  --gpu-memory-utilization 0.90 \
  > "$log_file" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$pid_file"
python3 - "$meta_file" "$pid" "$gpu" "$served_name" "$base_model" "$adapter" <<'PY'
import datetime
import json
import sys

path, pid, gpu, served_name, base_model, adapter = sys.argv[1:]
payload = {
    "pid": int(pid),
    "gpu_index": int(gpu),
    "served_model_name": served_name,
    "base_model": base_model,
    "adapter_path": adapter,
    "started_at": datetime.datetime.now().astimezone().isoformat(),
}
with open(path, "w") as target:
    json.dump(payload, target)
PY
sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
  tail -40 "$log_file" >&2 || true
  exit 24
fi
printf '{"pid":%s,"gpu_index":%s,"served_model_name":"%s"}\n' "$pid" "$gpu" "$served_name"
"""


def start_managed_vllm(experiment_id: str, gpu_index: int) -> dict:
    if not isinstance(gpu_index, int) or not 0 <= gpu_index <= 7:
        raise ValueError("gpu_index must be an integer from 0 to 7")
    model = next(
        (item for item in launchable_models() if item["experiment_id"] == experiment_id),
        None,
    )
    if model is None:
        raise ValueError("model is not registered for dashboard launch")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", model["served_model_name"]):
        raise ValueError("served model name contains unsupported characters")
    gpu = next((item for item in remote_gpu_status() if item["index"] == gpu_index), None)
    if gpu is None:
        raise ValueError(f"GPU {gpu_index} does not exist")
    if not gpu["available"]:
        owners = sorted({item["user"] for item in gpu["processes"]})
        detail = f"; users={','.join(owners)}" if owners else ""
        raise ValueError(
            f"GPU {gpu_index} is not idle: {gpu['memory_used_mib']} MiB, "
            f"{gpu['utilization_percent']}% utilization{detail}"
        )
    args = [
        str(gpu_index),
        model["adapter_path"],
        model["served_model_name"],
        model["base_model"],
        str(model["max_model_len"]),
        str(model["lora_rank"]),
        f"{REMOTE_VLLM_ENV}/bin/vllm",
        str(REMOTE_VLLM_PORT),
        REMOTE_VLLM_PID,
        REMOTE_VLLM_META,
        REMOTE_VLLM_LOG,
    ]
    process = ssh_run(["bash", "-s", "--", *args], input_text=START_VLLM_SCRIPT, timeout=30)
    result = json.loads(process.stdout.strip().splitlines()[-1])
    result["status"] = managed_vllm_status(include_gpus=False)
    return result


STOP_VLLM_SCRIPT = r"""
set -euo pipefail
pid_file="$1"
meta_file="$2"
log_file="$3"
if [[ ! -f "$pid_file" ]]; then
  echo '{"stopped":false,"reason":"no managed pid file"}'
  exit 0
fi
pid="$(cat "$pid_file")"
details="$(ps -o user=,args= -p "$pid" 2>/dev/null || true)"
if [[ -z "$details" ]]; then
  rm -f "$pid_file" "$meta_file"
  echo '{"stopped":false,"reason":"process already exited"}'
  exit 0
fi
owner="${details%% *}"
command="${details#* }"
if [[ "$owner" != "dengyan" || "$command" != *"vllm"* || "$command" != *"serve"* ]]; then
  echo "refusing to stop PID $pid because ownership/command validation failed" >&2
  exit 30
fi
kill "$pid"
for _ in $(seq 1 30); do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pid_file" "$meta_file"
    echo "{\"stopped\":true,\"pid\":$pid}"
    exit 0
  fi
  sleep 1
done
kill -9 "$pid"
rm -f "$pid_file" "$meta_file"
echo "{\"stopped\":true,\"pid\":$pid,\"forced\":true}"
"""


def stop_managed_vllm() -> dict:
    process = ssh_run(
        ["bash", "-s", "--", REMOTE_VLLM_PID, REMOTE_VLLM_META, REMOTE_VLLM_LOG],
        input_text=STOP_VLLM_SCRIPT,
        timeout=45,
    )
    return json.loads(process.stdout.strip().splitlines()[-1])


def vllm_request(
    method: str,
    suffix: str,
    payload=None,
    *,
    timeout: int = 300,
) -> tuple[int, object]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{VLLM_BASE_URL}/{suffix.lstrip('/')}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
            return response.status, {
                "data": body,
                "latency_seconds": round(time.monotonic() - started, 3),
                "base_url": VLLM_BASE_URL,
            }
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        return error.code, {"error": detail, "base_url": VLLM_BASE_URL}
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
        return HTTPStatus.BAD_GATEWAY, {"error": str(error), "base_url": VLLM_BASE_URL}


class Handler(BaseHTTPRequestHandler):
    server_version = "TabularExperimentDashboard/1.0"

    def log_message(self, fmt, *args):
        print(f"[dashboard] {self.address_string()} {fmt % args}")

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False, default=str).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def read_body(self):
        size = int(self.headers.get("Content-Length", "0"))
        if size > MAX_BODY:
            raise ValueError("request body too large")
        return json.loads(self.rfile.read(size) or b"{}")

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                return self.send_json({
                    "ok": True,
                    "repo_root": str(REPO_ROOT),
                    "vllm_base_url": VLLM_BASE_URL,
                })
            if parsed.path == "/api/experiments":
                return self.send_json([enrich_experiment(item) for item in load_registry()])
            if parsed.path == "/api/construction":
                # Enriched reflection/perception trajectories under data/trajectories/. No file param
                # lists the available *enriched*.jsonl files; with a file param it pages records.
                query = parse_qs(parsed.query)
                fname = query.get("file", [""])[0]
                tdir = REPO_ROOT / "data" / "trajectories"
                if not fname:
                    files = []
                    for name in os.listdir(tdir) if tdir.exists() else []:
                        if "enriched" not in name or not name.endswith(".jsonl"):
                            continue
                        path = tdir / name
                        stat = path.stat()
                        files.append({
                            "name": name,
                            "mtime": stat.st_mtime,
                            "mtime_iso": time.strftime(
                                "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
                            ),
                            "size_bytes": stat.st_size,
                        })
                    files.sort(key=lambda item: (item["mtime"], item["name"]), reverse=True)
                    return self.send_json({"files": files})
                page = max(1, int(query.get("page", ["1"])[0]))
                page_size = min(50, max(1, int(query.get("page_size", ["8"])[0])))
                path = resolve_repo_path(f"data/trajectories/{os.path.basename(fname)}")
                return self.send_json(jsonl_page(path, page, page_size))
            if parsed.path == "/api/playground/config":
                return self.send_json(playground_config())
            if parsed.path == "/api/vllm/models":
                status, payload = vllm_request("GET", "models")
                return self.send_json(payload, status)
            if parsed.path == "/api/vllm/server":
                return self.send_json(managed_vllm_status())

            parts = [unquote(part) for part in parsed.path.split("/") if part]
            if len(parts) >= 3 and parts[:2] == ["api", "experiments"]:
                experiment = find_experiment(parts[2])
                if len(parts) == 3:
                    return self.send_json(enrich_experiment(experiment))
                if len(parts) == 4 and parts[3] == "metrics":
                    return self.send_json(training_metrics(experiment))
                if len(parts) == 4 and parts[3] == "attribution":
                    eval_dir = experiment.get("evaluation", {}).get("directory")
                    if not eval_dir:
                        return self.send_json({"available": False})
                    return self.send_json(
                        ATTRIBUTION.attribution_summary(resolve_repo_path(eval_dir))
                    )
                if len(parts) == 4 and parts[3] == "records":
                    query = parse_qs(parsed.query)
                    sources = {item["id"]: item for item in data_sources(experiment)}
                    default_source = next(iter(sources), "")
                    source = query.get("source", [default_source])[0]
                    page = max(1, int(query.get("page", ["1"])[0]))
                    page_size = min(50, max(1, int(query.get("page_size", ["10"])[0])))
                    min_raw = query.get("min_steps", [""])[0]
                    max_raw = query.get("max_steps", [""])[0]
                    min_steps = int(min_raw) if min_raw else None
                    max_steps = int(max_raw) if max_raw else None
                    descriptor = sources.get(source)
                    if descriptor is None:
                        raise ValueError(f"unknown record source {source!r}")
                    path = resolve_repo_path(descriptor["path"])
                    payload = (
                        jsonl_page(path, page, page_size, min_steps, max_steps)
                        if descriptor["format"] == "jsonl"
                        else json_page(path)
                    )
                    payload["source"] = descriptor
                    return self.send_json(payload)
            return self.serve_static(parsed.path)
        except KeyError as error:
            self.send_json({"error": f"unknown experiment: {error.args[0]}"}, HTTPStatus.NOT_FOUND)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except subprocess.CalledProcessError as error:
            self.send_json(
                {"error": error.stderr.strip() or error.stdout.strip() or "remote command failed"},
                HTTPStatus.BAD_GATEWAY,
            )
        except subprocess.TimeoutExpired:
            self.send_json({"error": "remote command timed out"}, HTTPStatus.GATEWAY_TIMEOUT)
        except Exception as error:
            self.send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            body = self.read_body()
            if parsed.path == "/api/vllm/chat":
                messages = body.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a non-empty list")
                payload = {
                    "model": body.get("model"),
                    "messages": messages,
                    "temperature": float(body.get("temperature", 0)),
                    "max_tokens": min(4096, max(1, int(body.get("max_tokens", 1024)))),
                    "stream": False,
                }
                status, response = vllm_request("POST", "chat/completions", payload)
                return self.send_json(response, status)
            if parsed.path == "/api/vllm/server/start":
                experiment_id = body.get("experiment_id")
                if not isinstance(experiment_id, str) or not experiment_id:
                    raise ValueError("experiment_id is required")
                gpu_index = body.get("gpu_index")
                if not isinstance(gpu_index, int) or isinstance(gpu_index, bool):
                    raise ValueError("gpu_index must be an integer")
                return self.send_json(start_managed_vllm(experiment_id, gpu_index))
            if parsed.path == "/api/vllm/server/stop":
                return self.send_json(stop_managed_vllm())

            parts = [unquote(part) for part in parsed.path.split("/") if part]
            if (
                len(parts) == 4
                and parts[:2] == ["api", "experiments"]
                and parts[3] == "refresh"
            ):
                return self.send_json(refresh_trainer_state(find_experiment(parts[2])))
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except subprocess.CalledProcessError as error:
            self.send_json(
                {"error": error.stderr.strip() or error.stdout.strip() or "remote command failed"},
                HTTPStatus.BAD_GATEWAY,
            )
        except subprocess.TimeoutExpired:
            self.send_json({"error": "remote command timed out"}, HTTPStatus.GATEWAY_TIMEOUT)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self.send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_PATCH(self):
        parsed = urlparse(self.path)
        try:
            parts = [unquote(part) for part in parsed.path.split("/") if part]
            if len(parts) == 3 and parts[:2] == ["api", "experiments"]:
                return self.send_json(update_experiment(parts[2], self.read_body()))
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except KeyError as error:
            self.send_json({"error": f"unknown experiment: {error.args[0]}"}, HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def serve_static(self, request_path: str):
        if not DIST_ROOT.exists():
            return self.send_json(
                {"error": "frontend is not built; run npm run dev or npm run build"},
                HTTPStatus.NOT_FOUND,
            )
        relative = request_path.lstrip("/") or "index.html"
        path = (DIST_ROOT / relative).resolve()
        if DIST_ROOT not in path.parents and path != DIST_ROOT:
            return self.send_json({"error": "invalid path"}, HTTPStatus.BAD_REQUEST)
        if not path.is_file():
            path = DIST_ROOT / "index.html"
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Experiment dashboard API: http://{args.host}:{args.port}")
    print(f"vLLM proxy target: {VLLM_BASE_URL}")
    server.serve_forever()


if __name__ == "__main__":
    main()
