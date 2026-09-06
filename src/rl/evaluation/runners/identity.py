"""Pure identity record construction for formal Atomic v26 evaluations."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def evaluation_identity(contract: dict[str, Any], *, arm: str, adapter: dict[str, Any],
                        runtime: Path, model: Path, derived: Path,
                        versions: dict[str, str], implementation: dict[str, str],
                        gpu_id: int, port: int, served_model: str) -> dict[str, Any]:
    evaluation = contract["evaluation"]
    serving = contract["serving"]
    return {
        "schema_version": "qwen3-v26-formal-evaluation-identity-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": arm, "base_model": str(model),
        "base_model_revision": contract["model"]["revision"],
        "base_model_identity_sha256": contract["model"]["base_model_identity"]["aggregate_sha256"],
        "adapter_path": adapter["path"], "adapter_sha256": adapter["adapter_sha256"],
        "adapter_config_sha256": adapter["adapter_config_sha256"],
        "adapter_config_semantic_sha256": adapter["adapter_config_semantic_sha256"],
        "dataset": contract["input"]["dataset"], "input_path": str(derived),
        "input_sha256": contract["input"]["derived_sha256"],
        "task_identity_sha256_without_db_path": contract["input"]["task_identity_sha256_without_db_path"],
        "protocol_version": contract["runtime"]["protocol_version"],
        "protocol_hash": contract["runtime"]["protocol_hash"], "runtime": str(runtime),
        "runtime_sha256": contract["runtime"]["content_tree_sha256"],
        "serving": {**serving, "package_versions": versions,
                     "formal_wrapper_sha256": implementation["formal_v26_rollout_passk.py"],
                     "frozen_rollout_passk_sha256": contract["runtime"]["key_file_sha256"]["src/eval/rollout_passk.py"],
                     "tokenizer_config_sha256": contract["model"]["tokenizer_config_sha256"],
                     "chat_template_sha256": contract["model"]["chat_template_sha256"]},
        "concurrency": {"workers": evaluation["workers"], "sample_workers": evaluation["sample_workers"],
                        "first_sample_workers": evaluation["first_sample_workers"],
                        "max_inflight_requests": evaluation["max_inflight_requests"],
                        "max_num_seqs": serving["max_num_seqs"]},
        "decode": {key: evaluation[key] for key in ("temperature", "top_p", "max_tokens", "n_samples", "pass_k", "enable_thinking", "api_retries")},
        "agent": {key: evaluation[key] for key in ("max_steps", "context_mode", "history_turns", "rolling_prompt_variant", "rolling_observation_style", "denotation_comparison", "few_shot")},
        "tool_execution_timeout_seconds": evaluation["tool_execution_timeout_seconds"],
        "operational": {"physical_gpu_id": gpu_id, "port": port, "served_model": served_model},
    }
