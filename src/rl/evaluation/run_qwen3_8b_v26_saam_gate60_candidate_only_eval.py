#!/usr/bin/env python3
"""Evaluate only the completed SAAM Gate60 adapter on frozen BIRD-dev.

SFT1 is intentionally not rerun here: its full-dev result is already the
trusted version26 anchor.  This launcher reuses the exact formal v26 arm
runner and all of its fail-closed runtime/model/input/result checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import traceback
from pathlib import Path

from src.rl.evaluation import formal_v26_matched_eval as formal


def bind_completed_saam_contract(
    contract_path: Path,
    training_run: Path,
    run_dir: Path,
) -> dict:
    contract = formal.load_object(contract_path)
    training_run = training_run.resolve()
    manifest = formal.load_object(training_run / "run_manifest.json")

    contract["status"] = "completed_saam_gate60_candidate_only_eval"
    contract["host_paths"]["training_run"] = str(training_run)
    contract["host_paths"]["allowed_run_parent"] = str(run_dir.resolve().parent)
    contract["training_final"] = {
        "checkpoint_name": "checkpoint-4",
        "global_step": 4,
        "experiment_config_sha256": manifest["experiment_config_sha256"],
        "expected_manifest": manifest,
    }

    for name, implementation_path in formal.implementation_paths().items():
        contract["implementation_sha256"][name] = formal.sha256_file(
            Path(implementation_path).resolve()
        )

    runtime_path = Path(contract["host_paths"]["runtime"]).resolve()
    runtime_lock_path = runtime_path / "runtime_lock.json"
    runtime_lock = formal.load_object(runtime_lock_path)
    contract["runtime"]["runtime_lock_sha256"] = formal.sha256_file(runtime_lock_path)
    for field in (
        "source_commit",
        "exported_paths",
        "exported_file_count",
        "content_tree_sha256",
    ):
        contract["runtime"][field] = runtime_lock[field]
    contract["runtime"]["key_file_sha256"] = {
        relative: formal.sha256_file(runtime_path / relative)
        for relative in contract["runtime"]["key_file_sha256"]
    }
    formal.validate_contract_shape(contract)
    return contract


def execute_candidate(
    contract: dict,
    *,
    run_dir: Path,
    gpu_id: int,
    port: int,
    maximum_used_mib: int,
    model_ready_timeout: int,
    implementation: dict[str, str],
) -> int:
    paths = {name: Path(value) for name, value in contract["host_paths"].items()}
    allowed_parent = paths["allowed_run_parent"].resolve()
    run_dir = run_dir.resolve()
    if run_dir.parent != allowed_parent:
        raise ValueError(
            f"run directory must be a direct child of {allowed_parent}: {run_dir}"
        )
    if run_dir.exists():
        raise ValueError(f"candidate run directory already exists: {run_dir}")

    assets = formal.verify_host_assets(
        contract,
        gpu_id=gpu_id,
        maximum_used_mib=maximum_used_mib,
    )
    formal.require_free_port(port)
    allowed_parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    status_path = run_dir / "status.json"
    status = {
        "schema_version": "qwen3-v26-formal-candidate-only-eval-status-v1",
        "state": "running",
        "success": False,
        "stage": "preparing_input",
        "started_at_utc": formal.utc_now(),
        "run_dir": str(run_dir),
        "physical_gpu_id": gpu_id,
        "port": port,
        "evaluated_arms": ["final"],
        "sft1_rerun": False,
    }
    formal.atomic_json(status_path, status)

    def update(**values) -> None:
        status.update(values)
        formal.atomic_json(status_path, status)

    try:
        input_dir = run_dir / "input"
        derived = input_dir / "bird_dev_20240627.newgnn.jsonl"
        input_manifest = input_dir / "bird_dev_20240627.newgnn.manifest.json"
        formal.prepare_derived_input(
            contract,
            assets.pop("source_rows"),
            paths["database_root"],
            derived,
            input_manifest,
        )
        formal.atomic_json(
            run_dir / "preflight_gate.json",
            {
                key: value for key, value in assets.items() if key != "adapters"
            }
            | {
                "adapters": assets["adapters"],
                "evaluated_arms": ["final"],
                "sft1_rerun": False,
                "contract_canonical_sha256": hashlib.sha256(
                    formal.canonical_json_bytes(contract)
                ).hexdigest(),
                "implementation_sha256": implementation,
                "candidate_launcher_sha256": formal.sha256_file(Path(__file__)),
                "derived_input_sha256": formal.sha256_file(derived),
                "derived_manifest_sha256": formal.sha256_file(input_manifest),
            },
        )
        update(stage="evaluating_final")
        result = formal.run_arm(
            contract,
            arm="final",
            adapter=assets["adapters"]["final"],
            runtime=paths["runtime"],
            model=paths["base_model"],
            derived=derived,
            input_rows=formal.load_jsonl(derived),
            run_root=run_dir,
            python=paths["python"],
            gpu_id=gpu_id,
            port=port,
            maximum_used_mib=maximum_used_mib,
            model_ready_timeout=model_ready_timeout,
            versions=assets["package_versions"],
            implementation=implementation,
        )
        update(
            state="completed",
            success=True,
            stage="completed",
            finished_at_utc=formal.utc_now(),
            final=result,
            accuracy={"correct": result["correct"], "total": result["records"]},
            legal={"count": result["legal"], "total": result["records"]},
        )
        return 0
    except Exception as exc:
        update(
            state="failed",
            success=False,
            finished_at_utc=formal.utc_now(),
            failure={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--maximum-used-mib", type=int, default=512)
    parser.add_argument("--model-ready-timeout", type=int, default=900)
    args = parser.parse_args()

    contract = bind_completed_saam_contract(
        args.contract,
        args.training_run,
        args.run_dir,
    )
    implementation = formal.verify_implementation(contract)

    def terminate(signum: int, _frame) -> None:
        raise formal.TerminationRequested(
            f"received {signal.Signals(signum).name}"
        )

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    with formal.exclusive_gpu_lock(args.gpu_id):
        return execute_candidate(
            contract,
            run_dir=args.run_dir,
            gpu_id=args.gpu_id,
            port=args.port,
            maximum_used_mib=args.maximum_used_mib,
            model_ready_timeout=args.model_ready_timeout,
            implementation=implementation,
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"formal SAAM-only evaluation failed: {exc}")
        raise SystemExit(2) from exc
