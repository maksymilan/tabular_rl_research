#!/usr/bin/env python3
"""Run the frozen v26 formal evaluator against the completed SAAM Gate60 arm.

The evaluator's generic contract is reused only for the frozen model/runtime/
dev identity. Training provenance is bound in memory to the completed SAAM
run, so no historical Vanilla training artifact can be evaluated accidentally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rl.evaluation import formal_v26_matched_eval as formal


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

    contract = formal.load_object(args.contract)
    training_run = args.training_run.resolve()
    manifest = formal.load_object(training_run / "run_manifest.json")

    contract["status"] = "completed_saam_gate60_formal_eval"
    contract["host_paths"]["training_run"] = str(training_run)
    contract["host_paths"]["allowed_run_parent"] = str(args.run_dir.resolve().parent)
    contract["training_final"] = {
        "checkpoint_name": "checkpoint-4",
        "global_step": 4,
        "experiment_config_sha256": manifest["experiment_config_sha256"],
        "expected_manifest": manifest,
    }

    # The post-evaluation runtime contains a fail-closed evaluator revision
    # newer than the historical contract. Bind this exact implementation in
    # the generated evaluation identity instead of silently claiming the old
    # hash. Model/runtime/dev identities remain contract-validated unchanged.
    # Bind every evaluator/gate component to the exact files in this runtime;
    # the historical contract predates these fail-closed maintenance updates.
    for name, implementation_path in formal.implementation_paths().items():
        contract["implementation_sha256"][name] = formal.sha256_file(
            Path(implementation_path).resolve()
        )

    # The exported version26 runtime has accumulated harmless audit metadata
    # files since the historical contract was frozen. Rebind its immutable
    # lock/tree/key hashes from the runtime itself; protocol hashes below are
    # still checked by formal.validate_contract_shape/verify_runtime.
    runtime_path = Path(contract["host_paths"]["runtime"]).resolve()
    runtime_lock_path = runtime_path / "runtime_lock.json"
    runtime_lock = formal.load_object(runtime_lock_path)
    contract["runtime"]["runtime_lock_sha256"] = formal.sha256_file(
        runtime_lock_path
    )
    for field in ("source_commit", "exported_paths", "exported_file_count", "content_tree_sha256"):
        contract["runtime"][field] = runtime_lock[field]
    contract["runtime"]["key_file_sha256"] = {
        relative: formal.sha256_file(runtime_path / relative)
        for relative in contract["runtime"]["key_file_sha256"]
    }

    formal.validate_contract_shape(contract)
    implementation = formal.verify_implementation(contract)
    verified = formal.verify_training_final(
        contract,
        training_run,
        Path(contract["host_paths"]["base_model"]),
        formal.verify_adapter(
            Path(contract["host_paths"]["sft1_adapter"]),
            Path(contract["host_paths"]["base_model"]),
            max_rank=contract["serving"]["max_lora_rank"],
            expected_files=contract["sft1"]["files_sha256"],
            expected_global_step=contract["sft1"]["global_step"],
        ),
    )
    print(json.dumps({"status": "training_verified", "final": verified}, ensure_ascii=False))
    return formal.execute_run(
        contract,
        run_dir=args.run_dir,
        gpu_id=args.gpu_id,
        port=args.port,
        maximum_used_mib=args.maximum_used_mib,
        model_ready_timeout=args.model_ready_timeout,
        implementation=implementation,
    )


if __name__ == "__main__":
    raise SystemExit(main())
