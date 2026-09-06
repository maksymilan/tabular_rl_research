#!/usr/bin/env python3
"""Strict full-dev gate for the operator-requested mixed180/step12 arm.

This controller deliberately does not reuse the representative-600 or
boundary300 training provenance.  It binds the one already-started mixed180
run, requires its immutable cohort and final artifacts, and then delegates the
fresh SFT1-versus-final evaluation to the shared v26 evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
from pathlib import Path
from typing import Any, Mapping

from rl.evaluation.runners import formal_v26_matched_eval as formal


from rl.evaluation.runners.contracts import default_contract
HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = default_contract("qwen3_8b_v26_earlystop_mixed180_formal_matched_contract.json")
SCHEMA_VERSION = "qwen3-8b-v26-earlystop-mixed180-formal-matched-eval-v1"
GENERIC_SCHEMA = "qwen3-8b-v26-vanilla-formal-matched-eval-v1"
EXPECTED_TASKS_SHA256 = (
    "a015c6513b850576ff95388238d45ad5cc0130da53823bf5d43869668592fb5d"
)
EXPECTED_COHORT_MANIFEST_SHA256 = (
    "e940c996d854a1756ec9787d375517125d9b7b35872b6ce9251232cd7526eaad"
)
EXPECTED_BASE_MODEL_IDENTITY_SHA256 = (
    "85bd3b7d908acb3a9b9c7ec57b98d6b9e3b2fb427685ae808d1c43173279cecc"
)


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} mismatch: expected={expected!r}, actual={actual!r}"
        )


def _regular(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")


def _generic_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    bound = json.loads(json.dumps(contract))
    bound["schema_version"] = GENERIC_SCHEMA
    formal.validate_contract_shape(bound)
    return bound


def validate_contract(contract: dict[str, Any]) -> None:
    _require(contract.get("schema_version"), SCHEMA_VERSION, "earlystop schema")
    _require(
        contract.get("status"),
        "frozen_operator_requested_earlystop_arm",
        "earlystop status",
    )
    _require(contract.get("arms"), ["sft1", "final"], "evaluation arms")
    _require(contract.get("baseline"), "sft1", "evaluation baseline")
    _require(contract.get("primary_candidate"), "final", "primary candidate")
    paths = contract.get("host_paths") or {}
    _require(
        paths.get("training_run"),
        "/home/dengyan/tabular_rl_outputs/"
        "qwen3_8b_atomic_v26_earlystop_mixed180_vanilla_grpo_20260813/"
        "train180_two_pass_seed20260812",
        "training run",
    )
    expected = contract.get("training_final") or {}
    _require(expected.get("checkpoint_name"), "checkpoint-12", "primary checkpoint")
    _require(expected.get("global_step"), 12, "primary global step")
    _require(
        expected.get("experiment_config_sha256"),
        "58a9096030d242a36069ea026f5f27d3f12823b5912f25785756f217a100c6f6",
        "experiment config",
    )
    manifest = expected.get("expected_manifest") or {}
    fixed = {
        "examples_json_sha256": EXPECTED_TASKS_SHA256,
        "records": 180,
        "expected_records": 180,
        "optimizer_steps": 12,
        "prompts_per_update": 30,
        "group_size": 8,
        "save_steps": 2,
        "save_total_limit": 1,
        "seed": 20260812,
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
        "policy_reduction": "trajectory_token_mean",
        "learning_rate": 8e-7,
        "kl_beta": 0.0,
    }
    for field, value in fixed.items():
        _require(manifest.get(field), value, f"training manifest {field}")
    _require(
        (contract.get("model") or {})
        .get("base_model_identity", {})
        .get("aggregate_sha256"),
        EXPECTED_BASE_MODEL_IDENTITY_SHA256,
        "base model identity",
    )
    _generic_contract(contract)


def verify_earlystop_artifacts(contract: dict[str, Any]) -> dict[str, Any]:
    training = Path(contract["host_paths"]["training_run"])
    cohort = Path(contract["earlystop_training"]["cohort_manifest"])
    tasks = Path(contract["earlystop_training"]["train_tasks"])
    manifest_path = training / "run_manifest.json"
    lock_path = training / "implementation_lock.json"
    precision_path = training / "training_precision.json"
    checkpoint = training / "checkpoint-12"
    final = training / "final"
    paths = {
        "cohort_manifest": cohort,
        "train_tasks": tasks,
        "run_manifest": manifest_path,
        "implementation_lock": lock_path,
        "training_precision": precision_path,
        "checkpoint_adapter": checkpoint / "adapter_model.safetensors",
        "checkpoint_config": checkpoint / "adapter_config.json",
        "checkpoint_state": checkpoint / "trainer_state.json",
        "final_adapter": final / "adapter_model.safetensors",
        "final_config": final / "adapter_config.json",
    }
    for label, path in paths.items():
        _regular(path, label)
    _require(formal.sha256_file(tasks), EXPECTED_TASKS_SHA256, "train180 SHA-256")
    _require(
        formal.sha256_file(cohort),
        EXPECTED_COHORT_MANIFEST_SHA256,
        "cohort manifest SHA-256",
    )
    checkpoint_state = formal.load_object(paths["checkpoint_state"])
    _require(checkpoint_state.get("global_step"), 12, "checkpoint global step")
    _require(
        formal.sha256_file(paths["checkpoint_adapter"]),
        formal.sha256_file(paths["final_adapter"]),
        "final/checkpoint-12 adapter weights",
    )
    _require(
        formal.sha256_file(paths["checkpoint_config"]),
        formal.sha256_file(paths["final_config"]),
        "final/checkpoint-12 adapter config",
    )
    checkpoint_names = sorted(
        path.name
        for path in training.glob("checkpoint-*")
        if path.is_dir() and not path.is_symlink()
    )
    _require(checkpoint_names, ["checkpoint-12"], "sole durable checkpoint")
    manifest = formal.load_object(manifest_path)
    lock = formal.load_object(lock_path)
    _require(
        manifest.get("base_model_identity"),
        contract["model"]["base_model_identity"],
        "training base model identity",
    )
    _require(
        lock.get("base_model_identity"),
        manifest.get("base_model_identity"),
        "implementation-lock base model identity",
    )
    _require(
        lock.get("files"),
        manifest.get("implementation_source_sha256"),
        "training implementation lock files",
    )
    cohort_value = formal.load_object(cohort)
    _require(
        cohort_value.get("schema_version"),
        "qwen3-v26-earlystop-mixed-grpo-cohort-v1",
        "cohort schema",
    )
    _require(
        cohort_value.get("status"),
        "frozen_operator_requested_earlystop_mixed180",
        "cohort status",
    )
    outputs = cohort_value.get("outputs") or {}
    train_output = outputs.get("train180") or {}
    _require(train_output.get("sha256"), EXPECTED_TASKS_SHA256, "cohort train SHA")
    _require(train_output.get("records"), 180, "cohort train records")
    selection = cohort_value.get("selection") or {}
    _require(selection.get("eligible_mixed"), 193, "eligible mixed records")
    _require(selection.get("selected"), 180, "selected records")
    training_contract = cohort_value.get("training_contract") or {}
    _require(
        training_contract,
        {
            "records": 180,
            "group_size": 8,
            "prompts_per_update": 30,
            "optimizer_steps": 12,
            "passes": 2,
            "fresh_online_trajectories": 2880,
            "reward": "binary-result-only",
            "kl_beta": 0.0,
        },
        "cohort training contract",
    )
    return {
        "status": "verified",
        "training_run": str(training),
        "cohort_manifest_sha256": formal.sha256_file(cohort),
        "train_tasks_sha256": formal.sha256_file(tasks),
        "run_manifest_sha256": formal.sha256_file(manifest_path),
        "implementation_lock_sha256": formal.sha256_file(lock_path),
        "checkpoint12_adapter_sha256": formal.sha256_file(
            paths["checkpoint_adapter"]
        ),
    }


def print_plan(contract: dict[str, Any], implementation: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": "qwen3-v26-earlystop-mixed180-formal-plan-v1",
        "status": "frozen_waiting_for_training_completion",
        "remote_operations_performed": False,
        "training": {
            "records": 180,
            "optimizer_steps": 12,
            "primary_policy": "final equals checkpoint-12",
        },
        "evaluation": {
            "arms": ["sft1", "final"],
            "order": "fresh SFT1 then final",
            "questions_per_arm": 1534,
            "no_resume": True,
        },
        "promotion_gate": contract["promotion_gate"],
        "implementation_sha256": implementation,
        "host_paths": contract["host_paths"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-plan", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--maximum-used-mib", type=int, default=512)
    parser.add_argument("--model-ready-timeout", type=int, default=900)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    contract = formal.load_object(args.contract)
    validate_contract(contract)
    generic = _generic_contract(contract)
    implementation = formal.verify_implementation(generic)
    if args.print_plan:
        print(json.dumps(print_plan(contract, implementation), ensure_ascii=False, indent=2))
        return 0
    if args.gpu_id is None or args.gpu_id < 0:
        raise SystemExit("--gpu-id is required and must be non-negative")
    verified = verify_earlystop_artifacts(contract)
    with formal.exclusive_gpu_lock(args.gpu_id):
        if args.preflight:
            report = formal.verify_host_assets(
                generic,
                gpu_id=args.gpu_id,
                maximum_used_mib=args.maximum_used_mib,
            )
            report.pop("source_rows", None)
            print(
                json.dumps(
                    {"status": "earlystop_preflight_ok", "earlystop": verified, **report},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.run_dir is None:
            raise SystemExit("--run requires --run-dir")

        def terminate(signum: int, _frame: Any) -> None:
            raise formal.TerminationRequested(
                f"received {signal.Signals(signum).name}"
            )

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        return formal.execute_run(
            generic,
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
        print(f"earlystop mixed180 formal matched evaluation failed: {exc}")
        raise SystemExit(2) from exc
