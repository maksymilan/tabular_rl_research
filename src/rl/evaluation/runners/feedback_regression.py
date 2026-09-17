"""Preflight and immutable identities for same-adapter feedback regression."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path

from rl.diagnostics.io import iter_jsonl
from rl.diagnostics.records import only_sample
from rl.evaluation.runners.formal_v26_matched_eval import (
    atomic_json, content_tree, load_object, require_equal, sha256_file,
    utc_now, verify_model, verify_runtime, verify_source_and_databases,
)
from rl.runtime.error_feedback import FEEDBACK_VERSION


def serving_library_environment() -> dict:
    """Record an explicit existing driver link; never install or mutate libraries."""
    directory = os.environ.get("TRITON_LIBCUDA_PATH")
    if not directory:
        return {"environment": {}, "driver": None}
    link = Path(directory).absolute() / "libcuda.so"
    if not link.is_file():
        raise FileNotFoundError(f"TRITON_LIBCUDA_PATH has no usable libcuda.so: {link}")
    return {"environment": {"TRITON_LIBCUDA_PATH": str(link.parent)},
            "driver": {"link": str(link), "resolved": str(link.resolve()),
                       "sha256": sha256_file(link)}}


def prepare_feedback_regression(*, run_root: Path, control_root: Path, contract_path: Path,
                                runtime: Path, model: Path, adapter: Path,
                                source: Path, databases: Path, baseline: Path,
                                baseline_manifest: Path, adapter_sha256: str,
                                baseline_sha256: str, max_tokens: int,
                                gpu_ids: list[int], ports: list[int],
                                checkpoint_global_step: int = 6380,
                                workers: int = 24, max_num_seqs: int = 24,
                                max_model_len: int = 32768,
                                max_num_batched_tokens: int = 8192) -> dict:
    if min(checkpoint_global_step, workers, max_num_seqs, max_model_len, max_num_batched_tokens) < 1:
        raise ValueError("checkpoint step and serving limits must be positive")
    state = load_object(adapter / "trainer_state.json")
    require_equal(state["global_step"], checkpoint_global_step, "SFT checkpoint step")
    for name in ("run_manifest.json", "implementation_lock.json", "preflight.json"):
        if (run_root / name).exists():
            raise FileExistsError(run_root / name)
    library_environment = serving_library_environment()
    contract = load_object(contract_path)
    # Reuse only pinned runtime/model/input assets. Historical RL arms/checkpoints
    # in this asset contract do not choose this experiment's SFT adapter.
    runtime_audit = verify_runtime(contract, runtime)
    rows, database_audit = verify_source_and_databases(contract, source, databases)
    model_audit = verify_model(contract, model)
    require_equal(sha256_file(adapter / "adapter_model.safetensors"), adapter_sha256, "SFT adapter")
    require_equal(sha256_file(baseline), baseline_sha256, "historical baseline results")
    baseline_config = load_object(baseline_manifest)
    expected = {
        "max_tokens": max_tokens, "max_steps": 30, "temperature": 0.0, "top_p": 1.0,
        "n_samples": 1, "pass_k": [1], "history_turns": 4,
        "context_mode": "rolling-legal-history", "denotation_comparison": "bird-set",
        "protocol_version": contract["runtime"]["protocol_version"],
        "protocol_hash": contract["runtime"]["protocol_hash"],
    }
    for key, value in expected.items():
        require_equal(baseline_config.get(key), value, f"baseline {key}")
    import hashlib
    prompt_hash = hashlib.sha256(baseline_config["system_prompt"].encode()).hexdigest()
    require_equal(prompt_hash, contract["runtime"]["student_prompt_sha256"], "baseline prompt")
    seen = set()
    baseline_correct = baseline_legal = 0
    for record in iter_jsonl(baseline):
        index = int(record["example_index"])
        if index in seen or not 0 <= index < len(rows):
            raise ValueError(f"invalid/duplicate baseline index: {index}")
        seen.add(index)
        for key in ("db_id", "question", "gold_sql"):
            require_equal(record[key], rows[index][key], f"baseline task {index} {key}")
        sample = only_sample(record)
        if sample.get("failure_type") in {"api_error", "context_overflow"}:
            raise ValueError(f"contaminated baseline at {index}")
        baseline_correct += bool(sample.get("correct"))
        baseline_legal += bool(sample.get("legal"))
    require_equal(seen, set(range(len(rows))), "baseline coverage")
    code_count, code_hash = content_tree(control_root, ["src"])
    implementation = {str(path.relative_to(control_root)): sha256_file(path)
                      for path in sorted((control_root / "src").rglob("*"))
                      if path.is_file() and path.suffix in {".py", ".sh"}}
    lock = {"schema_version": "feedback-implementation-lock-v1", "files_sha256": implementation,
            "content_tree_sha256": code_hash, "file_count": code_count}
    manifest = {
        "schema_version": "sft-feedback-regression-v1", "created_at_utc": utc_now(),
        "status": "preflight_passed", "error_feedback_version": FEEDBACK_VERSION,
        "candidate_only": True, "diagnostic_only": True,
        "max_tokens_user_confirmed": max_tokens,
        "adapter": str(adapter), "adapter_sha256": adapter_sha256,
        "checkpoint_global_step": checkpoint_global_step, "checkpoint_epoch": state.get("epoch"),
        "trainer_state_sha256": sha256_file(adapter / "trainer_state.json"),
        "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
        "model": model_audit, "runtime": runtime_audit, "decode_agent": expected,
        "tool_timeout_policy": "unchanged-frozen-runner-native-no-added-sqlite-timeout",
        "baseline": {"path": str(baseline), "sha256": baseline_sha256,
                     "manifest_sha256": sha256_file(baseline_manifest),
                     "correct": baseline_correct, "legal": baseline_legal, "total": len(rows)},
        "dataset": {**contract["input"], "path": str(source), "database_root": str(databases),
                    "databases_verified": database_audit},
        "control_root": str(control_root), "implementation_sha256": code_hash,
        "operational": {"gpu_ids": gpu_ids, "ports": ports, "shard_sizes": [767, 767]},
        "serving": {"dtype": "bfloat16", "tensor_parallel_size": 1, "enable_lora": True,
                    "library_environment": library_environment,
                    "max_model_len": max_model_len, "max_num_batched_tokens": max_num_batched_tokens,
                    "max_num_seqs": max_num_seqs, "evaluator_workers": workers,
                    "gpu_memory_utilization": 0.92, "enable_thinking": True,
                    "versions": {name: importlib.metadata.version(name) for name in
                                 ("torch", "transformers", "vllm", "peft")}},
        "comparison_limit": "historical cross-host baseline; no fresh run-to-run variance control",
    }
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_json(run_root / "implementation_lock.json", lock)
    atomic_json(run_root / "run_manifest.json", manifest)
    atomic_json(run_root / "preflight.json", {
        "passed": True, "run_manifest_sha256": sha256_file(run_root / "run_manifest.json"),
        "implementation_sha256": code_hash, "baseline_correct": baseline_correct,
        "coverage": len(seen), "created_at_utc": utc_now(),
    })
    return manifest


def verify_feedback_receipt(run_root: Path, control_root: Path) -> dict:
    receipt = load_object(run_root / "preflight.json")
    require_equal(sha256_file(run_root / "run_manifest.json"), receipt["run_manifest_sha256"], "immutable run manifest")
    _, code_hash = content_tree(control_root, ["src"])
    require_equal(code_hash, receipt["implementation_sha256"], "controller since preflight")
    manifest = load_object(run_root / "run_manifest.json")
    driver = manifest["serving"].get("library_environment", {}).get("driver")
    if driver:
        require_equal(str(Path(driver["link"]).resolve()), driver["resolved"], "CUDA driver link")
        require_equal(sha256_file(Path(driver["link"])), driver["sha256"], "CUDA driver since preflight")
    return manifest
