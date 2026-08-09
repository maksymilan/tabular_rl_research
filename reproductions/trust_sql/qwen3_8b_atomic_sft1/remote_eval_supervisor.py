#!/usr/bin/env python3
"""Detached, all-NewGNN supervisor for one exact Qwen3/version26 evaluation arm."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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


HERE = Path(__file__).resolve().parent
ALLOWED_RUN_ROOT = Path(
    "/home/dengyan/tabular_rl_outputs/evaluations/qwen3_8b_atomic_v26"
)


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def safe_gpu_ids(value: str) -> list[int]:
    pieces = value.split(",")
    if not pieces or any(not piece.isdigit() for piece in pieces):
        raise ValueError(f"GPU IDs must be comma-separated non-negative integers: {value!r}")
    values = [int(piece) for piece in pieces]
    if len(set(values)) != len(values):
        raise ValueError(f"GPU IDs must be distinct: {value!r}")
    return values


def preflight_gpu(gpus: list[int], maximum_used_mib: int) -> dict[str, int]:
    report: dict[str, int] = {}
    for gpu in gpus:
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
            raise ValueError(
                f"could not query physical GPU {gpu}: {completed.stderr.strip()}"
            )
        try:
            used = int(completed.stdout.strip())
        except ValueError as exc:
            raise ValueError(
                f"invalid memory reading for physical GPU {gpu}: {completed.stdout!r}"
            ) from exc
        if used > maximum_used_mib:
            raise ValueError(
                f"physical GPU {gpu} is not idle: {used} MiB used, limit={maximum_used_mib}"
            )
        report[str(gpu)] = used
    return report


def require_free_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as exc:
            raise ValueError(f"localhost port {port} is already in use") from exc


def run_json_command(command: list[str], output: Path) -> dict[str, Any]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"command failed ({completed.returncode}): {detail}")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"command returned non-JSON output: {completed.stdout!r}") from exc
    atomic_json(output, report)
    return report


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def wait_for_model(port: int, served_model: str, process: subprocess.Popen[bytes], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited before readiness with status {process.returncode}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=5
            ) as response:
                payload = json.loads(response.read())
            identifiers = {
                item.get("id") for item in payload.get("data", []) if isinstance(item, dict)
            }
            if served_model in identifiers:
                return
            last_error = f"served model absent; advertised={sorted(x for x in identifiers if x)}"
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(3)
    raise TimeoutError(f"vLLM readiness timed out after {timeout}s: {last_error}")


def stop_process_group(process: subprocess.Popen[bytes] | None) -> dict[str, Any]:
    if process is None:
        return {"requested": False, "stopped": True}
    report: dict[str, Any] = {
        "requested": True,
        "pid": process.pid,
        "returncode_before_cleanup": process.poll(),
    }
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
    report["returncode_after_cleanup"] = process.poll()
    report["stopped"] = process.poll() is not None
    return report


def sample_api_error_reason(sample: dict[str, Any]) -> str | None:
    failure_type = str(sample.get("failure_type") or "")
    if failure_type.casefold() in {"api_error", "chatapierror"}:
        return f"failure_type={failure_type}"
    error = str(sample.get("error") or "")
    if "chatapierror" in error.casefold():
        return "error_contains_ChatAPIError"
    return None


def validate_final_result(
    result_dir: Path,
    *,
    served_model: str,
    derived: Path,
    port: int,
) -> dict[str, Any]:
    all_path = result_dir / "all.jsonl"
    summary_path = result_dir / "summary.json"
    manifest_path = result_dir / "manifest.json"
    runtime_gate_path = result_dir / "version26_runtime_gate.json"
    model_gate_path = result_dir / "qwen3_model_gate.json"
    for path in (
        all_path,
        summary_path,
        manifest_path,
        runtime_gate_path,
        model_gate_path,
    ):
        if not path.is_file():
            raise ValueError(f"evaluation artifact is missing: {path}")
    records = [
        json.loads(line) for line in all_path.read_text(encoding="utf-8").splitlines() if line
    ]
    indices = [int(record["example_index"]) for record in records]
    if len(records) != 1534 or sorted(indices) != list(range(1534)):
        raise ValueError(
            f"evaluation coverage gate failed: records={len(records)}, unique={len(set(indices))}"
        )
    api_error_samples: list[dict[str, Any]] = []
    for record in records:
        samples = record.get("samples") or []
        record_has_sample_api_error = False
        if not isinstance(samples, list):
            raise ValueError(
                f"evaluation sample container is not a list: q{record.get('example_index')}"
            )
        for sample_position, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(
                    f"evaluation sample is not an object: "
                    f"q{record.get('example_index')} sample={sample_position}"
                )
            reason = sample_api_error_reason(sample)
            if reason is not None:
                record_has_sample_api_error = True
                api_error_samples.append(
                    {
                        "example_index": record.get("example_index"),
                        "sample_index": sample.get("sample_index", sample_position),
                        "reason": reason,
                    }
                )
        # The pass@k wrapper normally summarizes sample failures as all_samples_failed, but keep
        # this top-level guard for fail-closed compatibility with older artifacts.
        if not record_has_sample_api_error:
            top_level_reason = sample_api_error_reason(record)
            if top_level_reason is not None:
                api_error_samples.append(
                    {
                        "example_index": record.get("example_index"),
                        "sample_index": None,
                        "reason": f"top_level_{top_level_reason}",
                    }
                )
    api_error_count = len(api_error_samples)
    if api_error_count:
        raise ValueError(
            f"API error gate failed: api_error_count={api_error_count}; "
            f"first_errors={api_error_samples[:10]}"
        )
    summary = load_object(summary_path)
    if summary.get("total") != 1534:
        raise ValueError(f"summary total is not 1534: {summary.get('total')!r}")
    manifest = load_object(manifest_path)
    expected_fields = {
        "protocol_version": "version26",
        "tool_scheme": "atomic",
        "assistant_carrier": "think-json-v1",
        "model": served_model,
        "base_url": f"http://127.0.0.1:{port}/v1",
        "dataset": str(derived),
        "requested_size": 1534,
        "n_samples": 1,
        "pass_k": [1],
        "sample_workers": 1,
        "max_inflight_requests": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "max_steps": 30,
        "enable_thinking": "1",
        "context_mode": "rolling-legal-history",
        "history_turns": 4,
        "rolling_prompt_variant": "full",
        "rolling_observation_style": "resident",
        "denotation_comparison": "bird-set",
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_fields.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"historical evaluation manifest drift: {mismatches}")
    return {
        "status": "ok",
        "records": len(records),
        "unique_example_indices": len(set(indices)),
        "correct": summary.get("correct"),
        "accuracy": summary.get("accuracy"),
        "legal_answers": summary.get("legal_answers"),
        "api_error_count": api_error_count,
        "all_jsonl_sha256": sha256(all_path),
        "summary_sha256": sha256(summary_path),
        "manifest_sha256": sha256(manifest_path),
        "version26_runtime_gate_sha256": sha256(runtime_gate_path),
        "qwen3_model_gate_sha256": sha256(model_gate_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("base", "adapter"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-size", choices=("4b", "8b"), default="8b")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-gpu-memory-mib", type=int, default=512)
    parser.add_argument("--model-ready-timeout", type=int, default=900)
    return parser


def run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    model_size = getattr(args, "model_size", "8b")
    allowed_run_root = (
        ALLOWED_RUN_ROOT
        if model_size == "8b"
        else Path("/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_atomic_v26")
    )
    if run_dir.parent != allowed_run_root:
        raise ValueError(
            f"run directory must be a direct child of {allowed_run_root}: {run_dir}"
        )
    if not run_dir.is_dir():
        raise ValueError(f"staged run directory is missing: {run_dir}")
    if not 1024 <= args.port <= 65535:
        raise ValueError(f"port must be in [1024,65535], got {args.port}")
    gpus = safe_gpu_ids(args.gpu_ids)
    if args.mode == "base" and args.adapter is not None:
        raise ValueError("base mode must not receive --adapter")
    if args.mode == "adapter" and args.adapter is None:
        raise ValueError("adapter mode requires --adapter")

    paths = {
        "status": run_dir / "status.json",
        "launch": run_dir / "launch_manifest.json",
        "input_dir": run_dir / "input",
        "derived": run_dir / "input/bird_dev_20240627.newgnn.jsonl",
        "input_manifest": run_dir / "input/bird_dev_20240627.newgnn.manifest.json",
        "prepare_gate": run_dir / "input/preparation_gate.json",
        "asset_gate": run_dir / "asset_gate.json",
        "result": run_dir / "result",
        "vllm_log": run_dir / "vllm.log",
        "eval_log": run_dir / "evaluation.log",
        "vllm_pid": run_dir / "vllm.pid",
    }
    for name in ("status", "launch", "input_dir", "result", "vllm_log", "eval_log", "vllm_pid"):
        if paths[name].exists():
            raise ValueError(f"refusing to reuse run artifact {name}: {paths[name]}")

    started = utc_now()
    status: dict[str, Any] = {
        "schema_version": "qwen3-atomic-v26-all-newgnn-status-v2",
        "state": "starting",
        "success": False,
        "exit_status": None,
        "stage": "initializing",
        "started_at_utc": started,
        "finished_at_utc": None,
        "supervisor_pid": os.getpid(),
        "mode": args.mode,
        "model_size": model_size,
        "run_dir": str(run_dir),
        "result_dir": str(paths["result"]),
        "gpu_ids": gpus,
        "port": args.port,
    }
    atomic_json(paths["status"], status)

    def update(**values: Any) -> None:
        status.update(values)
        atomic_json(paths["status"], status)

    def on_signal(signum: int, _frame: Any) -> None:
        raise TerminationRequested(f"received signal {signal.Signals(signum).name}")

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    process: subprocess.Popen[bytes] | None = None
    try:
        update(state="running", stage="preparing_input")
        lock_path = HERE / "remote_eval_lock.json"
        runtime_lock = HERE / "runtime_lock.json"
        runtime_verifier = HERE / "verify_version26_runtime.py"
        preparer = HERE / "prepare_remote_eval_inputs.py"
        asset_verifier = HERE / "verify_remote_eval_assets.py"
        model_verifier = HERE / "verify_qwen3_model.py"
        pinned_model_verifier = HERE / "verify_pinned_qwen3_model.py"
        model_specs = HERE / "qwen3_model_specs.json"
        adapter_lock = HERE / "qwen3_4b_adapter_lock.json"
        for required in (
            lock_path,
            runtime_lock,
            runtime_verifier,
            preparer,
            asset_verifier,
            model_verifier,
            args.source,
        ):
            if not required.is_file():
                raise ValueError(f"staged required file is missing: {required}")
        if model_size == "4b":
            for required in (pinned_model_verifier, model_specs):
                if not required.is_file():
                    raise ValueError(f"staged required file is missing: {required}")
            if args.mode == "adapter" and not adapter_lock.is_file():
                raise ValueError(f"staged adapter lock is missing: {adapter_lock}")
        lock = load_object(lock_path)
        db_root = Path(lock["evaluation_input"]["remote_db_root"])
        paths["input_dir"].mkdir()
        prepare_report = run_json_command(
            [
                sys.executable,
                str(preparer),
                "--source",
                str(args.source),
                "--output",
                str(paths["derived"]),
                "--manifest",
                str(paths["input_manifest"]),
                "--db-root",
                str(db_root),
                "--lock",
                str(lock_path),
            ],
            paths["prepare_gate"],
        )

        update(stage="verifying_assets")
        asset_command = [
            sys.executable,
            str(asset_verifier),
            "--mode",
            args.mode,
            "--source",
            str(args.source),
            "--derived",
            str(paths["derived"]),
            "--input-manifest",
            str(paths["input_manifest"]),
            "--db-root",
            str(db_root),
            "--runtime-root",
            str(args.runtime_root),
            "--model-root",
            str(args.model_root),
            "--lock",
            str(lock_path),
            "--runtime-lock",
            str(runtime_lock),
            "--runtime-verifier",
            str(runtime_verifier),
            "--model-size",
            model_size,
        ]
        if model_size == "4b":
            asset_command.extend(("--model-specs", str(model_specs)))
            if args.mode == "adapter":
                asset_command.extend(("--adapter-lock", str(adapter_lock)))
        if args.adapter is not None:
            asset_command.extend(("--adapter", str(args.adapter)))
        asset_report = run_json_command(asset_command, paths["asset_gate"])

        update(stage="preflight")
        initial_gpu_memory = preflight_gpu(gpus, args.max_gpu_memory_mib)
        require_free_port(args.port)
        paths["result"].mkdir()
        # Keep the historical analyzer/archive contract.  Because the runtime report contains no
        # arm-specific field and is serialized canonically, these bytes are identical for base
        # and adapter runs.
        atomic_json(paths["result"] / "version26_runtime_gate.json", asset_report["runtime"])
        atomic_json(paths["result"] / "qwen3_model_gate.json", asset_report["model"])

        served_model = (
            f"qwen3-{model_size}-atomic-v26-base"
            if args.mode == "base"
            else f"qwen3-{model_size}-atomic-v26-sft1-qlora"
        )
        backbone_name = (
            served_model
            if args.mode == "base"
            else f"qwen3-{model_size}-atomic-v26-backbone"
        )
        bundle_hashes = {
            path.name: sha256(path)
            for path in sorted(HERE.iterdir())
            if path.is_file()
        }
        launch_manifest = {
            "schema_version": "qwen3-atomic-v26-all-newgnn-launch-v2",
            "created_at_utc": utc_now(),
            "mode": args.mode,
            "model_size": model_size,
            "run_dir": str(run_dir),
            "result_dir": str(paths["result"]),
            "supervisor_pid": os.getpid(),
            "python": sys.executable,
            "package_versions": package_versions(),
            "bundle_files_sha256": bundle_hashes,
            "physical_gpu_ids": gpus,
            "initial_gpu_memory_mib": initial_gpu_memory,
            "tensor_parallel_size": len(gpus),
            "port": args.port,
            "model_ready_timeout_seconds": args.model_ready_timeout,
            "base_model": str(args.model_root),
            "adapter": str(args.adapter) if args.adapter is not None else None,
            "served_model": served_model,
            "backbone_served_model": backbone_name,
            "runtime_root": str(args.runtime_root),
            "source_input": str(args.source),
            "source_input_sha256": sha256(args.source),
            "derived_input": str(paths["derived"]),
            "derived_input_sha256": sha256(paths["derived"]),
            "input_manifest_sha256": sha256(paths["input_manifest"]),
            "asset_gate_sha256": sha256(paths["asset_gate"]),
            "preparation_gate_sha256": sha256(paths["prepare_gate"]),
            "evaluation_contract": lock["evaluation_contract"],
            "vllm_contract": {
                "host": "127.0.0.1",
                "port": args.port,
                "dtype": "bfloat16",
                "max_model_len": 16384,
                "max_num_batched_tokens": 16384,
                "max_num_seqs": 4,
                "gpu_memory_utilization": 0.9,
                "generation_config": "vllm",
                "reasoning_parser": None,
                "chat_template_override": None,
            },
            "asset_identity": {
                "runtime_source_commit": asset_report["runtime"]["source_commit"],
                "runtime_content_tree_sha256": asset_report["runtime"][
                    "content_tree_sha256"
                ],
                "rolling_system_prompt_sha256": asset_report["runtime"]["protocol"][
                    "rolling_system_prompt_sha256"
                ],
                "model_revision": asset_report["model"]["model_revision"],
                "model_shards": asset_report["model"]["model_shards"],
                "adapter": asset_report["model"].get("adapter"),
                "databases": asset_report["evaluation_input"]["databases"],
            },
            "path_remap": prepare_report,
        }
        atomic_json(paths["launch"], launch_manifest)

        vllm_command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(args.model_root),
            "--tokenizer",
            str(args.model_root),
            "--served-model-name",
            backbone_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--dtype",
            "bfloat16",
            "--tensor-parallel-size",
            str(len(gpus)),
            "--max-model-len",
            "16384",
            "--max-num-batched-tokens",
            "16384",
            "--max-num-seqs",
            "4",
            "--gpu-memory-utilization",
            "0.90",
            "--generation-config",
            "vllm",
        ]
        if args.adapter is not None:
            vllm_command.extend(
                (
                    "--enable-lora",
                    "--lora-modules",
                    f"{served_model}={args.adapter}",
                    "--max-lora-rank",
                    "64",
                )
            )
        vllm_environment = os.environ.copy()
        vllm_environment.update(
            {
                "CUDA_VISIBLE_DEVICES": args.gpu_ids,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            }
        )
        update(stage="starting_vllm")
        with paths["vllm_log"].open("wb") as vllm_log:
            process = subprocess.Popen(
                vllm_command,
                stdin=subprocess.DEVNULL,
                stdout=vllm_log,
                stderr=subprocess.STDOUT,
                env=vllm_environment,
                start_new_session=True,
            )
        paths["vllm_pid"].write_text(f"{process.pid}\n", encoding="ascii")
        update(stage="waiting_for_vllm", vllm_pid=process.pid)
        wait_for_model(args.port, served_model, process, args.model_ready_timeout)

        update(stage="evaluating")
        eval_environment = os.environ.copy()
        eval_environment.pop("PYTHONPATH", None)
        eval_environment.pop("EVAL_SYSTEM_PROMPT_VARIANT", None)
        eval_environment.update(
            {
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "EVAL_ENABLE_THINKING": "1",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            }
        )
        evaluator = args.runtime_root / "src/eval/rollout_passk.py"
        eval_command = [
            sys.executable,
            "-u",
            str(evaluator),
            "--base-url",
            f"http://127.0.0.1:{args.port}/v1",
            "--model",
            served_model,
            "--examples-json",
            str(paths["derived"]),
            "--allow-eval-tasks",
            "--n",
            "1534",
            "--n-samples",
            "1",
            "--pass-k",
            "1",
            "--workers",
            "4",
            "--sample-workers",
            "1",
            "--max-inflight-requests",
            "4",
            "--max-steps",
            "30",
            "--max-tokens",
            "2048",
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--sample-detail",
            "full",
            "--summary-every",
            "10",
            "--context-mode",
            "rolling-legal-history",
            "--history-turns",
            "4",
            "--rolling-prompt-variant",
            "full",
            "--rolling-observation-style",
            "resident",
            "--denotation-comparison",
            "bird-set",
            "--result-dir",
            str(paths["result"]),
        ]
        with paths["eval_log"].open("wb") as eval_log:
            evaluation = subprocess.run(
                eval_command,
                stdin=subprocess.DEVNULL,
                stdout=eval_log,
                stderr=subprocess.STDOUT,
                env=eval_environment,
                check=False,
            )
        if evaluation.returncode != 0:
            raise RuntimeError(
                f"historical evaluator exited with status {evaluation.returncode}; "
                f"see {paths['eval_log']}"
            )

        update(stage="validating_result")
        result_report = validate_final_result(
            paths["result"],
            served_model=served_model,
            derived=paths["derived"],
            port=args.port,
        )
        cleanup_report = stop_process_group(process)
        process = None
        update(
            state="completed",
            stage="completed",
            success=True,
            exit_status=0,
            finished_at_utc=utc_now(),
            result=result_report,
            vllm_cleanup=cleanup_report,
        )
        return 0
    except Exception as exc:
        cleanup_report = stop_process_group(process)
        update(
            state="failed",
            success=False,
            exit_status=1,
            finished_at_utc=utc_now(),
            failure={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            vllm_cleanup=cleanup_report,
        )
        print(f"remote evaluation failed during {status.get('stage')}: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        # This is only reachable before the status path is safely established.
        print(f"remote evaluation supervisor initialization failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
