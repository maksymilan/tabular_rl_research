"""One-off receipts for the user-approved frozen-runtime alpha=0.25 experiment.

Training stays in the copied remote runtime. File I/O and identity verification
reuse the project's shared APIs; this script does not implement a trainer.
"""
from __future__ import annotations

import argparse
import math
import shutil
import socket
import sys
from pathlib import Path

import yaml

from rl.diagnostics.io import read_json, read_jsonl, sha256_file, write_json


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def prepare(args: argparse.Namespace) -> None:
    previous = read_json(args.previous / "run_manifest.json")
    check(previous["span_balance_alpha"] == 0.5, "reference must be alpha=0.5")
    check(not args.runtime.exists(), "refusing to replace an existing runtime")
    shutil.copytree(args.source_runtime, args.runtime, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Restore the exact implementation files recorded in the reference run.
    shutil.copytree(args.previous / "implementation_source_snapshot", args.runtime, dirs_exist_ok=True)
    launcher = args.runtime / "src/rl/experiments/run_qwen3_8b_atomic_v26_saam_fourlevel_gate60_table_rl.sh"
    original = launcher.read_text()
    check(original.count("--span-balance-alpha 0.5") == 1, "unexpected legacy launcher shape")
    launcher.write_text(original.replace("--span-balance-alpha 0.5", '--span-balance-alpha "${SPAN_BALANCE_ALPHA:?set SPAN_BALANCE_ALPHA}"'))
    config = previous["experiment_config"]
    config["experiment_name"] = "qwen3_v26_saam_alpha025_balanced60_diagnostic"
    config["expected_records"] = previous["records"]
    config["optimizer"]["steps"] = previous["optimizer_steps"]
    config["rollout"]["prompts_per_update"] = previous["prompts_per_update"]
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text("# alpha=0.25 is passed explicitly to the frozen trainer CLI.\n" + yaml.safe_dump(config, sort_keys=False))
    write_json(args.run_root / "deployment.json", {
        "diagnostic_only": True, "source_runtime": str(args.source_runtime),
        "runtime": str(args.runtime), "reference_run": str(args.previous),
        "launcher_sha256": sha256_file(launcher),
        "change": "parameterize the launcher alpha; training implementation restored from reference snapshot",
    })


def preflight(args: argparse.Namespace) -> None:
    previous = read_json(args.previous / "run_manifest.json")
    sys.path.insert(0, str(args.runtime / "src/rl"))
    from experiment_config import RLExperimentConfig, runtime_content_tree_sha256, verify_base_model_identity

    config = RLExperimentConfig.load(args.config)
    defaults = config.argparse_defaults(args.runtime)
    for key in ("expected_records", "optimizer_steps", "prompts_per_update", "group_size", "learning_rate", "kl_beta", "credit_assignment", "result_reward_profile"):
        check(defaults[key] == previous[key], f"effective config mismatch: {key}")
    check(sha256_file(args.tasks) == previous["examples_json_sha256"], "cohort changed")
    rows = read_jsonl(args.tasks)
    check(len(rows) == 60, "expected 60 tasks")
    check(len({row["example_index"] for row in rows}) == 60, "duplicate example indices")
    for row in rows:
        db = Path(row["db_path"])
        check((db if db.is_absolute() else args.runtime / db).is_file(), "missing task database")
    contract = config.payload["runtime_contract"]
    protocol_root = Path(contract["runtime_root"])
    check(runtime_content_tree_sha256(protocol_root) == contract["runtime_content_tree_sha256"], "protocol runtime changed")
    adapter = Path(previous["adapter_path"])
    check(sha256_file(adapter / "adapter_model.safetensors") == previous["initial_adapter_sha256"], "initial adapter changed")
    model = verify_base_model_identity(Path(previous["model_path"]), contract["base_model_identity"])
    for relative, digest in previous["implementation_source_sha256"].items():
        check(sha256_file(args.runtime / relative) == digest, f"training implementation changed: {relative}")
    check(not (args.run_root / "train").exists(), "training output already exists")
    check(not (args.run_root / "evaluation/results").exists(), "evaluation output already exists")
    for port in (18285, 51385, 18310, 18311):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    baseline = read_jsonl(args.run_root / "reference/sft6380.all.jsonl")
    check(len(baseline) == 1534 and sum(bool(r["correct"]) for r in baseline) == 924, "wrong SFT baseline")
    check({r["example_index"] for r in baseline} == set(range(1534)), "SFT baseline coverage mismatch")
    output = args.run_root / "preflight.json"
    check(not output.exists(), "refusing to replace preflight receipt")
    write_json(output, {
        "status": "passed", "diagnostic_only": True, "span_balance_alpha": 0.25,
        "reference_run": str(args.previous), "reference_manifest_sha256": sha256_file(args.previous / "run_manifest.json"),
        "runtime": str(args.runtime), "protocol": contract, "base_model": model,
        "configuration": str(args.config), "config_sha256": sha256_file(args.config),
        "tasks_sha256": sha256_file(args.tasks), "initial_adapter_sha256": previous["initial_adapter_sha256"],
        "implementation_source_sha256": previous["implementation_source_sha256"],
        "records": 60, "prompts_per_update": 30, "group_size": 8, "optimizer_steps": 4,
        "seed": 20260901, "train_gpu": 0, "rollout_gpu": 1, "evaluation_gpus": [0, 1],
        "sft_baseline_sha256": sha256_file(args.run_root / "reference/sft6380.all.jsonl"),
        "precision": "same frozen 4-bit base and FP32 LoRA/AdamW as reference run",
    })
    print("PREFLIGHT_PASSED alpha=0.25 records=60 updates=4 baseline=checkpoint-6380")


def trained(args: argparse.Namespace) -> None:
    previous = read_json(args.previous / "run_manifest.json")
    train = args.run_root / "train"
    current = read_json(train / "run_manifest.json")
    keys = (
        "protocol_version", "protocol_hash", "student_prompt_sha256", "initial_adapter_sha256",
        "reward_mode", "result_reward_profile", "result_advantage_profile", "credit_assignment",
        "policy_reduction", "error_penalty", "records", "expected_records", "group_size",
        "prompts_per_update", "optimizer_steps", "ppo_iterations", "gradient_accumulation_steps",
        "seed", "learning_rate", "adam_beta1", "adam_beta2", "clip_epsilon", "kl_beta",
        "transition_micro_batch_size", "transition_micro_batch_tokens", "examples_json_sha256",
    )
    for key in keys:
        check(current[key] == previous[key], f"unexpected matched-arm difference: {key}")
    check(current["span_balance_alpha"] == 0.25, "wrong effective alpha")
    state = read_json(train / "checkpoint-4/trainer_state.json")
    check(state["global_step"] == state["max_steps"] == 4, "wrong actual optimizer step")
    check(state["epoch"] == 2.0, "expected two passes over 60 tasks")
    gradients = [r["grad_norm"] for r in state["log_history"] if "grad_norm" in r]
    check(len(gradients) == 4 and all(math.isfinite(g) and g > 0 for g in gradients), "missing effective updates")
    check(len(read_jsonl(train / "rollouts.jsonl")) == 960, "rollout coverage mismatch")
    final_sha = sha256_file(train / "final/adapter_model.safetensors")
    check(final_sha == sha256_file(train / "checkpoint-4/adapter_model.safetensors"), "final adapter differs from step4")
    for name in ("implementation_lock.json", "training_precision.json"):
        check((train / name).is_file(), f"missing audit: {name}")
    write_json(args.run_root / "training_receipt.json", {
        "status": "passed", "global_step": 4, "span_balance_alpha": 0.25,
        "rollouts": 960, "gradient_norms": gradients, "final_adapter_sha256": final_sha,
        "training_manifest_sha256": sha256_file(train / "run_manifest.json"),
    })
    print("TRAINING_AUDIT_PASSED final=checkpoint-4 alpha=0.25 rollouts=960")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "preflight", "trained"))
    for key in ("run-root", "runtime", "source-runtime", "previous", "config", "tasks"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    {"prepare": prepare, "preflight": preflight, "trained": trained}[args.mode](args)


if __name__ == "__main__":
    main()
