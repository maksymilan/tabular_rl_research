#!/usr/bin/env python3
"""Fail-closed preflight for the frozen A100 SAAM/four-level 700-task run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from rl.configuration.experiment_config import RLExperimentConfig  # noqa: E402


EXPECTED_METHOD = {
    "credit_assignment": "saam-asymmetric-error",
    "error_penalty": 1.0,
    "result_reward_profile": "four-level",
    "span_balance_alpha": 0.5,
    "reason_token_loss_weight": 0.5,
    "tool_token_loss_weight": 0.5,
    "pcgrad": False,
}
EXPECTED_TERMINAL_REWARDS = {
    "correct_clean": 1.5,
    "correct_recovered_error": 1.0,
    "wrong_clean": -0.5,
    "wrong_error_or_policy_failure": -1.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    args = parser.parse_args()

    tasks = _jsonl(args.tasks)
    if len(tasks) != 700:
        raise ValueError(f"expected 700 tasks, found {len(tasks)}")
    identifiers = [row.get("example_id") for row in tasks]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("every task must have a non-empty example_id")
    if len(set(identifiers)) != 700:
        raise ValueError("task identities are not unique")
    indices = [row.get("example_index") for row in tasks]
    if sorted(indices) != list(range(700)):
        raise ValueError("example_index must be a unique contiguous 0..699 range")
    missing_databases = [
        str(row.get("db_path"))
        for row in tasks
        if not Path(str(row.get("db_path") or "")).is_file()
    ]
    if missing_databases:
        raise ValueError(
            f"{len(missing_databases)} tasks have missing databases; first={missing_databases[0]}"
        )

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "frozen_training_input":
        raise ValueError("cohort manifest is not frozen_training_input")
    method = manifest.get("method_contract") or {}
    if any(method.get(key) != value for key, value in EXPECTED_METHOD.items()):
        raise ValueError("cohort method contract drifted")
    if method.get("terminal_rewards") != EXPECTED_TERMINAL_REWARDS:
        raise ValueError("cohort four-level terminal reward contract drifted")
    outputs = manifest.get("outputs") or {}
    if outputs.get("tasks", {}).get("records") != 700:
        raise ValueError("manifest task record count drifted")
    if outputs.get("tasks", {}).get("sha256") != _sha256(args.tasks):
        raise ValueError("task JSONL hash does not match its frozen manifest")
    if outputs.get("manual_audit", {}).get("sha256") != _sha256(args.manual_audit):
        raise ValueError("manual audit hash does not match its frozen manifest")
    buckets = manifest["bucket_counts"]
    if set(buckets) != {
        "manual_old_all_correct",
        "manual_old_all_wrong",
        "mixed_new_bird_adaptive",
        "mixed_new_nonbird",
        "mixed_old_s1",
    }:
        raise ValueError("cohort bucket names drifted")
    if (
        buckets["manual_old_all_correct"] != 50
        or buckets["manual_old_all_wrong"] != 50
        or buckets["mixed_old_s1"] != 193
    ):
        raise ValueError("fixed historical/manual bucket counts drifted")
    if sum(buckets.values()) != 700 or sum(
        value for key, value in buckets.items() if key.startswith("mixed_")
    ) != 600:
        raise ValueError("cohort is not exactly 600 mixed + 100 manual")

    manual = _jsonl(args.manual_audit)
    if len(manual) != 100:
        raise ValueError("manual audit must contain 100 records")
    manual_classes = Counter(row.get("historical_outcome_class") for row in manual)
    if manual_classes != {"all_correct": 50, "all_wrong": 50}:
        raise ValueError(f"manual outcome balance drifted: {dict(manual_classes)}")

    config = RLExperimentConfig.load(args.experiment_config).payload
    if config.get("expected_records") != 700:
        raise ValueError("experiment config expected_records must equal 700")
    for key in ("credit_assignment", "error_penalty", "result_reward_profile"):
        if config.get(key) != EXPECTED_METHOD[key]:
            raise ValueError(f"experiment config {key} drifted")
    optimizer = config["optimizer"]
    rollout = config["rollout"]
    if optimizer["steps"] != 200 or rollout["prompts_per_update"] != 14:
        raise ValueError("schedule must use 14 task groups per update: 200 updates")
    if optimizer["learning_rate"] != 4.0e-7:
        raise ValueError("batch-14 schedule requires learning_rate=4e-7")
    if rollout["group_size"] != 8:
        raise ValueError("fresh online group size must equal K=8")
    weight = args.adapter_path / "adapter_model.safetensors"
    expected_adapter = config["runtime_contract"]["initial_adapter_sha256"]
    if not weight.is_file() or _sha256(weight) != expected_adapter:
        raise ValueError("checkpoint-6380 adapter identity mismatch")

    print(
        json.dumps(
            {
                "status": "passed",
                "tasks": 700,
                "mixed": 600,
                "manual_all_correct": 50,
                "manual_all_wrong": 50,
                "epochs": 4,
                "prompts_per_update": 14,
                "optimizer_updates": 200,
                "learning_rate": 4.0e-7,
                "online_rollouts": 4 * 700 * 8,
                "task_sha256": _sha256(args.tasks),
                "adapter_sha256": expected_adapter,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
