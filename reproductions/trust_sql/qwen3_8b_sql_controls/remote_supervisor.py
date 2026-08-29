#!/usr/bin/env python3
"""Detached evaluator for pinned raw Qwen3-4B/8B SQL interface controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALLOWED_ROOT = Path(
    "/home/dengyan/tabular_rl_outputs/evaluations/qwen3_sql_controls"
)
DB_ROOT = Path(
    "/home/dengyan/tabular_rl_project/data/bird/dev_20240627/dev_databases"
)
FULL_TASKS = 1534


class TerminationRequested(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_idle_gpu(gpu: int, maximum_used_mib: int = 512) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
    used = int(completed.stdout.strip())
    if used > maximum_used_mib:
        raise ValueError(f"physical GPU {gpu} is not idle: {used} MiB used")
    return used


def require_free_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", port))


def wait_for_model(
    port: int,
    served_model: str,
    process: subprocess.Popen[bytes],
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited before readiness: {process.returncode}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=5
            ) as response:
                payload = json.loads(response.read())
            identifiers = {
                item.get("id")
                for item in payload.get("data", [])
                if isinstance(item, dict)
            }
            if served_model in identifiers:
                return
            last_error = f"served model absent: {sorted(x for x in identifiers if x)}"
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(3)
    raise TimeoutError(f"vLLM readiness timed out: {last_error}")


def stop_process_group(process: subprocess.Popen[bytes] | None) -> dict[str, Any]:
    if process is None:
        return {"requested": False, "stopped": True}
    report: dict[str, Any] = {"requested": True, "pid": process.pid}
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=15)
            report["forced_kill"] = True
    report["returncode"] = process.poll()
    report["stopped"] = process.poll() is not None
    return report


def run_json(command: list[str], output: Path) -> dict[str, Any]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): "
            f"{(completed.stderr or completed.stdout).strip()[:4000]}"
        )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError("JSON command did not return an object")
    atomic_json(output, payload)
    return payload


def validate_result(
    mode: str,
    result_dir: Path,
    served_model: str,
    expected_tasks: int,
) -> dict[str, Any]:
    all_path = result_dir / "all.jsonl"
    manifest_path = result_dir / "manifest.json"
    if not all_path.is_file() or not manifest_path.is_file():
        raise ValueError("evaluation result is missing all.jsonl or manifest.json")
    records = [
        json.loads(line)
        for line in all_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = [int(record["example_index"]) for record in records]
    if len(records) != expected_tasks or sorted(indices) != list(range(expected_tasks)):
        raise ValueError(
            f"coverage gate failed: records={len(records)} unique={len(set(indices))}"
        )
    manifest = load_object(manifest_path)
    expected_common = {
        "model": served_model,
        "denotation_comparison": "bird-set",
        "enable_thinking": "1" if mode == "direct" else True,
    }
    if mode == "direct":
        expected = {
            **expected_common,
            "runner": "direct_sql_passk",
            "dev_size": expected_tasks,
            "n_samples": 1,
            "pass_k": [1],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 2048,
            "execution_feedback": False,
            "prompt_profile": "canonical-json-v1",
        }
    else:
        expected = {
            **expected_common,
            "runner": "iterative_sql_feedback",
            "tool_scheme": "iterative-sql",
            "interface": "execute-sql-submit-sql-v6",
            "context_profile": "lazy-catalog-v1",
            "max_steps": 30,
            "max_tokens": 2048,
            "history_turns": 4,
        }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"manifest contract drift: {mismatches}")

    api_errors: list[int] = []
    correct = 0
    legal = 0
    context_overflow = 0
    total_steps = 0
    total_process_errors = 0
    for record in records:
        failure = str(record.get("failure_type") or "")
        error = str(record.get("error") or "")
        if failure == "api_error" or "chatapierror" in error.casefold():
            api_errors.append(int(record["example_index"]))
        context_overflow += int(failure == "context_overflow")
        correct += int(bool(record.get("correct")))
        if mode == "direct":
            samples = record.get("samples") or []
            if len(samples) == 1 and isinstance(samples[0], dict):
                legal += int(samples[0].get("predicted_row_count") is not None)
        else:
            legal += int(bool(record.get("legal")))
            total_steps += int(record.get("steps") or 0)
            total_process_errors += int(record.get("errors") or 0)
    if api_errors:
        raise ValueError(
            f"API error gate failed: count={len(api_errors)} first={api_errors[:10]}"
        )
    return {
        "status": "ok",
        "records": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "legal": legal,
        "legal_rate": legal / len(records),
        "context_overflow": context_overflow,
        "api_error_count": 0,
        "mean_steps": total_steps / len(records) if mode == "iterative" else 1.0,
        "process_errors": total_process_errors if mode == "iterative" else None,
        "all_jsonl_sha256": sha256(all_path),
        "manifest_sha256": sha256(manifest_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("direct", "iterative"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--model-size", choices=("4b", "8b"), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--n", type=int, default=FULL_TASKS)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--model-ready-timeout", type=int, default=900)
    return parser


def run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    if run_dir.parent != ALLOWED_ROOT:
        raise ValueError(f"run directory must be a direct child of {ALLOWED_ROOT}")
    if not run_dir.is_dir():
        raise ValueError(f"staged run directory is missing: {run_dir}")
    if not 1 <= args.n <= FULL_TASKS:
        raise ValueError(f"--n must be in [1,{FULL_TASKS}]")
    if not 0.5 <= args.gpu_memory_utilization <= 0.95:
        raise ValueError("--gpu-memory-utilization must be in [0.5,0.95]")
    if args.max_num_batched_tokens < 1024:
        raise ValueError("--max-num-batched-tokens must be at least 1024")
    if not 1 <= args.max_num_seqs <= 32:
        raise ValueError("--max-num-seqs must be in [1,32]")
    if not 1 <= args.workers <= 32:
        raise ValueError("--workers must be in [1,32]")
    if args.workers > args.max_num_seqs:
        raise ValueError("--workers cannot exceed --max-num-seqs")
    paths = {
        "status": run_dir / "status.json",
        "launch": run_dir / "launch_manifest.json",
        "runtime_tar": run_dir / "runtime.tar.gz",
        "runtime": run_dir / "runtime",
        "controller": run_dir / "controller",
        "source": run_dir / "input/bird_dev_20240627.jsonl",
        "derived": run_dir / "input/bird_dev_20240627.newgnn.jsonl",
        "input_manifest": run_dir / "input/bird_dev_20240627.newgnn.manifest.json",
        "input_gate": run_dir / "input_gate.json",
        "model_gate": run_dir / "model_gate.json",
        "result": run_dir / "result",
        "vllm_log": run_dir / "vllm.log",
        "eval_log": run_dir / "evaluation.log",
    }
    for required in (
        paths["runtime_tar"],
        paths["runtime"],
        paths["controller"],
        paths["source"],
        paths["controller"] / "qwen3_model_specs.json",
    ):
        if not required.exists():
            raise ValueError(f"staged artifact is missing: {required}")
    if sha256(paths["runtime_tar"]) != args.runtime_sha256:
        raise ValueError("runtime tar SHA-256 does not match launcher declaration")

    status: dict[str, Any] = {
        "schema_version": "qwen3-sql-controls-status-v2",
        "state": "starting",
        "success": False,
        "mode": args.mode,
        "model_size": args.model_size,
        "run_dir": str(run_dir),
        "gpu": args.gpu,
        "port": args.port,
        "requested_tasks": args.n,
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "stage": "initializing",
        "supervisor_pid": os.getpid(),
    }
    atomic_json(paths["status"], status)

    def update(**values: Any) -> None:
        status.update(values)
        atomic_json(paths["status"], status)

    def terminate(signum: int, _frame: Any) -> None:
        raise TerminationRequested(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    process: subprocess.Popen[bytes] | None = None
    try:
        update(state="running", stage="verifying_input")
        lock_path = paths["controller"] / "remote_eval_lock.json"
        input_report = run_json(
            [
                sys.executable,
                str(paths["controller"] / "prepare_remote_eval_inputs.py"),
                "--source",
                str(paths["source"]),
                "--output",
                str(paths["derived"]),
                "--manifest",
                str(paths["input_manifest"]),
                "--db-root",
                str(DB_ROOT),
                "--lock",
                str(lock_path),
            ],
            paths["input_gate"],
        )
        update(stage="verifying_model")
        model_report = run_json(
            [
                sys.executable,
                str(paths["controller"] / "verify_pinned_qwen3_model.py"),
                "--model-root",
                str(args.model_root),
                "--model-size",
                args.model_size,
                "--specs",
                str(paths["controller"] / "qwen3_model_specs.json"),
            ],
            paths["model_gate"],
        )
        if model_report.get("model_size") != args.model_size:
            raise ValueError("model-size gate drift")

        update(stage="preflight")
        initial_gpu_memory = require_idle_gpu(args.gpu)
        require_free_port(args.port)
        paths["result"].mkdir()
        served_model = f"qwen3-{args.model_size}-{args.mode}-sql-base"
        launch_manifest = {
            "schema_version": "qwen3-sql-controls-launch-v2",
            "created_at_utc": utc_now(),
            "mode": args.mode,
            "model_size": args.model_size,
            "run_dir": str(run_dir),
            "runtime_tar_sha256": args.runtime_sha256,
            "model_root": str(args.model_root),
            "model_repository": model_report["repository"],
            "model_revision": model_report["model_revision"],
            "model_gate_sha256": sha256(paths["model_gate"]),
            "input_gate_sha256": sha256(paths["input_gate"]),
            "derived_input_sha256": sha256(paths["derived"]),
            "physical_gpu": args.gpu,
            "initial_gpu_memory_mib": initial_gpu_memory,
            "port": args.port,
            "served_model": served_model,
            "vllm": {
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_num_seqs": args.max_num_seqs,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "generation_config": "vllm",
                "reasoning_parser": None,
            },
            "evaluation": {
                "tasks": args.n,
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 2048,
                "enable_thinking": True,
                "workers": args.workers,
                "denotation_comparison": "bird-set",
                "direct_prompt_profile": (
                    "canonical-json-v1" if args.mode == "direct" else None
                ),
                "iterative_interface": (
                    "execute-sql-submit-sql-v6"
                    if args.mode == "iterative"
                    else None
                ),
                "iterative_context_profile": (
                    "lazy-catalog-v1" if args.mode == "iterative" else None
                ),
            },
            "input_preparation": input_report,
        }
        atomic_json(paths["launch"], launch_manifest)

        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(args.gpu),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            }
        )
        vllm_command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(args.model_root),
            "--tokenizer",
            str(args.model_root),
            "--served-model-name",
            served_model,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--generation-config",
            "vllm",
        ]
        update(stage="starting_vllm")
        with paths["vllm_log"].open("wb") as handle:
            process = subprocess.Popen(
                vllm_command,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
        update(stage="waiting_for_vllm", vllm_pid=process.pid)
        wait_for_model(args.port, served_model, process, args.model_ready_timeout)

        update(stage="evaluating")
        eval_environment = environment.copy()
        eval_environment.update(
            {
                "EVAL_ENABLE_THINKING": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        if args.mode == "direct":
            eval_command = [
                sys.executable,
                "-u",
                str(paths["runtime"] / "src/eval/text2sql_passk.py"),
                "--base-url",
                f"http://127.0.0.1:{args.port}/v1",
                "--model",
                served_model,
                "--tasks-json",
                str(paths["derived"]),
                "--n",
                str(args.n),
                "--n-samples",
                "1",
                "--pass-k",
                "1",
                "--workers",
                str(args.workers),
                "--max-tokens",
                "2048",
                "--temperature",
                "0",
                "--top-p",
                "1",
                "--api-retries",
                "3",
                "--execution-timeout-seconds",
                "20",
                "--denotation-comparison",
                "bird-set",
                "--prompt-profile",
                "canonical-json-v1",
                "--result-dir",
                str(paths["result"]),
            ]
        else:
            eval_command = [
                sys.executable,
                "-u",
                str(paths["runtime"] / "src/eval/run_tool_scheme.py"),
                "--tool-scheme",
                "iterative-sql",
                "--",
                "--base-url",
                f"http://127.0.0.1:{args.port}/v1",
                "--enable-thinking",
                "--model",
                served_model,
                "--tasks-json",
                str(paths["derived"]),
                "--result-dir",
                str(paths["result"]),
                "--n",
                str(args.n),
                "--workers",
                str(args.workers),
                "--max-steps",
                "30",
                "--max-tokens",
                "2048",
                "--api-retries",
                "3",
                "--api-timeout",
                "600",
                "--execution-timeout-seconds",
                "20",
                "--history-turns",
                "4",
                "--context-profile",
                "lazy-catalog-v1",
                "--denotation-comparison",
                "bird-set",
            ]
        with paths["eval_log"].open("wb") as handle:
            evaluation = subprocess.run(
                eval_command,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=eval_environment,
                check=False,
            )
        if evaluation.returncode != 0:
            raise RuntimeError(
                f"evaluator exited with {evaluation.returncode}; see {paths['eval_log']}"
            )
        update(stage="validating_result")
        report = validate_result(args.mode, paths["result"], served_model, args.n)
        cleanup = stop_process_group(process)
        process = None
        update(
            state="completed",
            stage="completed",
            success=True,
            finished_at_utc=utc_now(),
            result=report,
            vllm_cleanup=cleanup,
        )
        return 0
    except Exception as exc:
        cleanup = stop_process_group(process)
        update(
            state="failed",
            success=False,
            finished_at_utc=utc_now(),
            failure={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            vllm_cleanup=cleanup,
        )
        print(f"remote SQL control failed at {status.get('stage')}: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"supervisor initialization failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
