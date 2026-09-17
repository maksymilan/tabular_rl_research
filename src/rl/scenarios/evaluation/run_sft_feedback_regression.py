"""Prepare/run one user-confirmed SFT feedback-only BIRD-dev regression."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

from rl.evaluation.runners.feedback_regression import prepare_feedback_regression, verify_feedback_receipt
from rl.evaluation.runners.formal_v26_matched_eval import (
    atomic_json, exclusive_gpu_lock, require_free_port, stop_owned_process_group, utc_now,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--asset-contract", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--databases", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--adapter-sha256")
    parser.add_argument("--baseline-sha256")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--ports", type=int, nargs=2, default=[18320, 18321])
    parser.add_argument("--checkpoint-global-step", type=int, default=6380)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--max-num-seqs", type=int, default=24)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.mode == "preflight":
        prepare_feedback_regression(
            run_root=args.run_root, control_root=args.control_root, contract_path=args.asset_contract,
            runtime=args.runtime, model=args.model, adapter=args.adapter, source=args.source,
            databases=args.databases, baseline=args.baseline, baseline_manifest=args.baseline_manifest,
            adapter_sha256=args.adapter_sha256, baseline_sha256=args.baseline_sha256,
            max_tokens=args.max_tokens, gpu_ids=args.gpus, ports=args.ports,
            checkpoint_global_step=args.checkpoint_global_step,
            workers=args.workers, max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len, max_num_batched_tokens=args.max_num_batched_tokens,
        )
        print(f"preflight passed: {args.run_root}", flush=True)
        return 0
    manifest = verify_feedback_receipt(args.run_root, args.control_root)
    source_root = args.control_root / "src"
    runners = source_root / "rl/evaluation/runners"
    evaluation = args.run_root / "evaluation"
    gpu0, gpu1 = manifest["operational"]["gpu_ids"]
    port0, port1 = manifest["operational"]["ports"]
    env = {**os.environ, "PYTHONPATH": str(source_root), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHON_BIN": sys.executable, "RUNTIME": manifest["runtime"]["runtime_root"],
           "MODEL": manifest["model"]["path"], "ADAPTER": manifest["adapter"],
           "SOURCE_CHECKPOINT": manifest["adapter"], "CHECKPOINT_GLOBAL_STEP": str(manifest.get("checkpoint_global_step", 6380)),
           "SOURCE_INPUT": manifest["dataset"]["path"],
           "DATABASE_ROOT": manifest["dataset"]["database_root"],
           "EVAL_SYSTEM_PROMPT_VARIANT": "default",
           "RUN_DIR": str(evaluation), "GPU0": str(gpu0), "GPU1": str(gpu1),
           "PORT0": str(port0), "PORT1": str(port1),
           "MAX_TOKENS": str(manifest["max_tokens_user_confirmed"]),
           "ERROR_FEEDBACK_VERSION": manifest["error_feedback_version"],
           "VLLM_MAX_MODEL_LEN": str(manifest["serving"]["max_model_len"]),
           "VLLM_BATCH_TOKENS": str(manifest["serving"]["max_num_batched_tokens"]),
           "VLLM_MAX_NUM_SEQS": str(manifest["serving"]["max_num_seqs"]),
           "EVAL_WORKERS": str(manifest["serving"].get("evaluator_workers", 24)),
           "VLLM_MEMORY_UTILIZATION": str(manifest["serving"]["gpu_memory_utilization"]),
           "WRAPPER": str(runners / "feedback_rollout_passk.py"),
           "SHARDER": str(runners / "make_eval_shards.py"), "MERGER": str(runners / "merge_eval_shards.py")}
    env.pop("TRITON_LIBCUDA_PATH", None)
    env.update(manifest["serving"].get("library_environment", {}).get("environment", {}))
    process = None
    try:
        with ExitStack() as locks:
            for gpu in (gpu0, gpu1):
                locks.enter_context(exclusive_gpu_lock(gpu))
            for port in (port0, port1):
                require_free_port(port)
            atomic_json(args.run_root / "status.json", {"status": "evaluating", "time": utc_now()})
            process = subprocess.Popen(["bash", str(source_root / "rl/scenarios/evaluation/run_qwen3_8b_v26_checkpoint_dataparallel_table_rl.sh")],
                                       env=env, start_new_session=True)
            if process.wait() != 0:
                raise RuntimeError(f"evaluation launcher exited {process.returncode}")
        atomic_json(args.run_root / "status.json", {"status": "comparing", "time": utc_now()})
        subprocess.run([sys.executable, str(source_root / "rl/scenarios/diagnostics/analyze_evaluation_results.py"),
            "--examples", str(evaluation / "input/bird_dev_20240627.table_rl.jsonl"),
            "--arm", f"sft_old={manifest['baseline']['path']}",
            "--arm", f"sft_new={evaluation / 'results/merged/all.jsonl'}",
            "--compare", "sft_new:sft_old", "--expected-count", "1534", "--include-per-example",
            "--output", str(args.run_root / "paired_analysis.json")], env=env, check=True)
        atomic_json(args.run_root / "status.json", {"status": "complete", "time": utc_now()})
        return 0
    except BaseException as exc:
        stop_owned_process_group(process)
        atomic_json(args.run_root / "status.json", {"status": "failed", "time": utc_now(), "error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
