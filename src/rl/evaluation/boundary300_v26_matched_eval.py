#!/usr/bin/env python3
"""Bind and execute the strict boundary300 SFT1-versus-final matched gate.

The frozen full-dev evaluator remains ``formal_v26_matched_eval.py``.  This
small outer controller adds the policy-boundary provenance that cannot be
known until cohort selection and training have completed.  Preflight and run
therefore require five explicit SHA-256 values; none is inferred from the
current contents of the remote paths.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from src.rl.evaluation import formal_v26_matched_eval as formal


HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE / "qwen3_8b_v26_boundary300_formal_matched_contract.json"
BOUNDARY_SCHEMA = "qwen3-8b-v26-boundary300-vanilla-formal-matched-eval-v1"
GENERIC_SCHEMA = "qwen3-8b-v26-vanilla-formal-matched-eval-v1"
PLACEHOLDER = "__REQUIRED_EXPLICIT_BINDING_AT_LAUNCH__"
PATH_PLACEHOLDER = "__REQUIRED_EXPLICIT_PATH_AT_LAUNCH__"
TRAIN300_PLACEHOLDER = "__BOUND_FROM_TRAIN300_AT_LAUNCH__"
SHA_FIELDS = (
    "boundary_manifest",
    "train300",
    "training_run_manifest",
    "training_implementation_lock",
    "final_checkpoint20_adapter",
)
SHA_PATTERN = re.compile(r"[0-9a-f]{64}")
PATH_FIELDS = ("boundary_manifest", "train_tasks", "training_run")


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} mismatch: expected={expected!r}, actual={actual!r}"
        )


def _require_regular(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")


def _task_id(row: Mapping[str, Any]) -> str:
    value = row.get("example_id") or row.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError("boundary task is missing example_id/instance_id")
    return value


def validate_boundary_template(template: dict[str, Any]) -> None:
    """Reject any attempt to weaken or reuse the representative train600 arm."""

    _require(template.get("schema_version"), BOUNDARY_SCHEMA, "boundary schema")
    _require(
        template.get("status"),
        "preregistered_waiting_for_explicit_artifact_bindings",
        "boundary status",
    )
    _require(template.get("arms"), ["sft1", "final"], "formal arms")
    _require(template.get("baseline"), "sft1", "formal baseline")
    _require(template.get("primary_candidate"), "final", "primary candidate")
    serialized = json.dumps(template, sort_keys=True)
    if "train600" in serialized or "records\": 600" in serialized:
        raise ValueError("boundary300 contract must not reuse the train600 contract")

    paths = template.get("host_paths") or {}
    for name in (
        "python",
        "allowed_run_parent",
        "source_input",
        "database_root",
        "runtime",
        "base_model",
        "sft1_adapter",
    ):
        if not isinstance(paths.get(name), str) or not paths[name]:
            raise ValueError(f"boundary host path is missing: {name}")
    for name in PATH_FIELDS:
        _require(paths.get(name), PATH_PLACEHOLDER, f"dynamic host path {name}")

    boundary = template.get("boundary_training") or {}
    for field, expected in {
        "selection_schema_version": "policy-boundary-grpo-cohort-v1",
        "selection_status": "frozen_boundary_training_cohort",
        "boundary_records": 332,
        "train_records": 300,
        "validation_records": 32,
        "primary_checkpoint": "final-step20-only",
    }.items():
        _require(boundary.get(field), expected, f"boundary_training.{field}")
    dynamic = boundary.get("dynamic_sha256") or {}
    _require(set(dynamic), set(SHA_FIELDS), "dynamic binding fields")
    for field in SHA_FIELDS:
        _require(dynamic.get(field), PLACEHOLDER, f"dynamic binding {field}")
    _require(
        boundary.get("allowed_artifact_roots"),
        {
            "selection_runs": "/home/dengyan/tabular_rl_outputs",
            "training_run_parent": (
                "/home/dengyan/tabular_rl_outputs/"
                "qwen3_8b_atomic_v26_boundary300_vanilla_grpo_20260812"
            ),
        },
        "allowed boundary artifact roots",
    )

    training = template.get("training_final") or {}
    _require(training.get("checkpoint_name"), "checkpoint-20", "final checkpoint")
    _require(training.get("global_step"), 20, "final global step")
    _require(
        training.get("experiment_config_sha256"),
        "f03674d0fe693442df4f60cf948279b1bc46c3510a43a9320e77c8200b4923c7",
        "boundary300 experiment config",
    )
    expected_manifest = training.get("expected_manifest") or {}
    for field, expected in {
        "examples_json_sha256": TRAIN300_PLACEHOLDER,
        "records": 300,
        "expected_records": 300,
        "optimizer_steps": 20,
        "save_steps": 2,
        "save_total_limit": 1,
        "prompts_per_update": 30,
        "group_size": 8,
        "seed": 20260812,
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
        "policy_reduction": "trajectory_token_mean",
        "learning_rate": 8e-7,
        "kl_beta": 0.0,
    }.items():
        _require(expected_manifest.get(field), expected, f"training manifest {field}")

    # Reuse every established full-dev/runtime/serving/gate invariant after
    # temporarily supplying well-formed inert digests.
    generic = bind_contract(
        template,
        {field: "0" * 64 for field in SHA_FIELDS},
        {
            "boundary_manifest": (
                "/home/dengyan/tabular_rl_outputs/inert_boundary_screen/"
                "boundary_selection/boundary_cohort_manifest.json"
            ),
            "train_tasks": (
                "/home/dengyan/tabular_rl_outputs/inert_boundary_screen/"
                "boundary_selection/train300.jsonl"
            ),
            "training_run": (
                "/home/dengyan/tabular_rl_outputs/"
                "qwen3_8b_atomic_v26_boundary300_vanilla_grpo_20260812/"
                "train300_two_pass_seed20260812"
            ),
        },
        _validated=True,
    )
    formal.validate_contract_shape(generic)


def validate_bound_paths(template: dict[str, Any], paths: Mapping[str, str]) -> None:
    """Permit S1 or S1+S2 output roots without permitting arbitrary assets."""

    _require(set(paths), set(PATH_FIELDS), "supplied dynamic path fields")
    converted: dict[str, Path] = {}
    for field, value in paths.items():
        if not isinstance(value, str) or not value or value != str(Path(value)):
            raise ValueError(f"{field} must be a normalized absolute path")
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{field} must be a normalized absolute path")
        converted[field] = path
    roots = template["boundary_training"]["allowed_artifact_roots"]
    selection_root = Path(roots["selection_runs"])
    manifest = converted["boundary_manifest"]
    train = converted["train_tasks"]
    _require(manifest.parent, train.parent, "selection artifact directory")
    _require(manifest.name, "boundary_cohort_manifest.json", "selection manifest name")
    _require(train.name, "train300.jsonl", "train300 name")
    _require(manifest.parent.name, "boundary_selection", "selection directory name")
    _require(manifest.parent.parent.parent, selection_root, "selection run parent")
    training_parent = Path(roots["training_run_parent"])
    training = converted["training_run"]
    _require(training.parent, training_parent, "boundary300 training parent")
    _require(training.name, "train300_two_pass_seed20260812", "training run name")


def bind_contract(
    template: dict[str, Any],
    bindings: Mapping[str, str],
    paths: Mapping[str, str],
    *,
    _validated: bool = False,
) -> dict[str, Any]:
    """Return an execution contract containing only explicitly supplied hashes."""

    if not _validated:
        validate_boundary_template(template)
    _require(set(bindings), set(SHA_FIELDS), "supplied dynamic binding fields")
    for field, digest in bindings.items():
        if not isinstance(digest, str) or SHA_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"{field} must be an explicit lowercase SHA-256")
    validate_bound_paths(template, paths)
    bound = copy.deepcopy(template)
    bound["schema_version"] = GENERIC_SCHEMA
    bound["status"] = "explicitly_bound_for_preflight_or_run"
    dynamic = bound["boundary_training"]["dynamic_sha256"]
    for field in SHA_FIELDS:
        dynamic[field] = bindings[field]
    for field in PATH_FIELDS:
        bound["host_paths"][field] = paths[field]
    bound["boundary_training"]["binding_status"] = "explicitly_bound"
    bound["training_final"]["expected_manifest"]["examples_json_sha256"] = bindings[
        "train300"
    ]
    formal.validate_contract_shape(bound)
    return bound


def verify_boundary_artifacts(
    contract: dict[str, Any], training_run: Path
) -> dict[str, Any]:
    """Bind selected tasks, training locks, and the sole step-20 policy exactly."""

    paths = {name: Path(value) for name, value in contract["host_paths"].items()}
    _require(training_run.resolve(), paths["training_run"].resolve(), "training run path")
    boundary_path = paths["boundary_manifest"]
    train_path = paths["train_tasks"]
    run_manifest_path = training_run / "run_manifest.json"
    lock_path = training_run / "implementation_lock.json"
    checkpoint_adapter = training_run / "checkpoint-20/adapter_model.safetensors"
    final_adapter = training_run / "final/adapter_model.safetensors"
    required = {
        "boundary_manifest": boundary_path,
        "train300": train_path,
        "training_run_manifest": run_manifest_path,
        "training_implementation_lock": lock_path,
        "checkpoint20_adapter": checkpoint_adapter,
        "final_adapter": final_adapter,
    }
    for label, path in required.items():
        _require_regular(path, label)
    allowed_roots = contract["boundary_training"]["allowed_artifact_roots"]
    selection_root = Path(allowed_roots["selection_runs"]).resolve()
    _require(
        boundary_path.resolve().parent.parent.parent,
        selection_root,
        "resolved selection run parent",
    )
    _require(
        train_path.resolve().parent,
        boundary_path.resolve().parent,
        "resolved selection artifact directory",
    )
    _require(
        training_run.resolve().parent,
        Path(allowed_roots["training_run_parent"]).resolve(),
        "resolved boundary300 training parent",
    )

    dynamic = contract["boundary_training"]["dynamic_sha256"]
    observed = {
        "boundary_manifest": formal.sha256_file(boundary_path),
        "train300": formal.sha256_file(train_path),
        "training_run_manifest": formal.sha256_file(run_manifest_path),
        "training_implementation_lock": formal.sha256_file(lock_path),
        "checkpoint20_adapter": formal.sha256_file(checkpoint_adapter),
        "final_adapter": formal.sha256_file(final_adapter),
    }
    for field in (
        "boundary_manifest",
        "train300",
        "training_run_manifest",
        "training_implementation_lock",
    ):
        _require(observed[field], dynamic[field], f"bound {field} SHA-256")
    expected_adapter = dynamic["final_checkpoint20_adapter"]
    _require(
        observed["checkpoint20_adapter"], expected_adapter, "checkpoint-20 adapter SHA-256"
    )
    _require(observed["final_adapter"], expected_adapter, "final adapter SHA-256")

    selection = formal.load_object(boundary_path)
    boundary = contract["boundary_training"]
    _require(
        selection.get("schema_version"),
        boundary["selection_schema_version"],
        "selection schema",
    )
    _require(selection.get("status"), boundary["selection_status"], "selection status")
    screen_pools = selection.get("screen_pools")
    if not isinstance(screen_pools, list) or len(screen_pools) not in {1, 2}:
        raise ValueError("selection must bind exactly an S1 or S1+S2 screen pool")
    expected_stage = "S1" if len(screen_pools) == 1 else "S1+S2"
    _require(
        (selection.get("selection") or {}).get("screen_stage"),
        expected_stage,
        "selection screen stage",
    )
    selection_contract = selection.get("contract") or {}
    for field in ("boundary_records", "train_records", "validation_records"):
        _require(
            selection_contract.get(field), boundary[field], f"selection contract {field}"
        )
    _require(
        selection_contract.get("primary_checkpoint"),
        "final-step20-only",
        "selection primary checkpoint",
    )
    expected_training_contract = {
        "optimizer_updates": 20,
        "prompts_per_update": 30,
        "group_size": 8,
        "train_passes": 2,
        "prompt_appearances": 600,
        "fresh_online_trajectories": 4800,
        "sampler": "trl-0.29-repeat-sampler-v1",
        "shuffle_dataset": True,
        "data_seed": 20260812,
        "task_order": "two deterministic data-seed shuffled passes",
        "per_pass_coverage": "each train300 identity exactly once",
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
    }
    _require(
        selection_contract.get("formal_training"),
        expected_training_contract,
        "selection formal training contract",
    )
    output = (selection.get("outputs") or {}).get("train") or {}
    _require(Path(str(output.get("path"))).resolve(), train_path.resolve(), "selection train path")
    _require(output.get("records"), 300, "selection train records")
    _require(output.get("sha256"), dynamic["train300"], "selection train SHA-256")

    rows = formal.load_jsonl(train_path)
    _require(len(rows), 300, "train300 row count")
    task_ids = [_task_id(row) for row in rows]
    indices = [row.get("example_index") for row in rows]
    if len(set(task_ids)) != 300 or any(type(index) is not int for index in indices):
        raise ValueError("train300 task identities are not 300 unique valid rows")
    if len(set(indices)) != 300:
        raise ValueError("train300 example_index values are not unique")
    _require((selection.get("task_ids") or {}).get("train"), task_ids, "train task order")

    run_manifest = formal.load_object(run_manifest_path)
    _require(
        run_manifest.get("examples_json_sha256"), dynamic["train300"], "run train300 SHA-256"
    )
    implementation_lock = formal.load_object(lock_path)
    _require(
        implementation_lock.get("schema_version"),
        "trl-implementation-lock-v1",
        "implementation lock schema",
    )
    _require(
        implementation_lock.get("files"),
        run_manifest.get("implementation_source_sha256"),
        "implementation lock files",
    )
    checkpoint_names = sorted(
        path.name
        for path in training_run.glob("checkpoint-*")
        if path.is_dir() and not path.is_symlink()
    )
    _require(checkpoint_names, ["checkpoint-20"], "sole checkpoint directory")
    return {
        "binding_status": "verified",
        "selection_manifest_path": str(boundary_path),
        "train300_path": str(train_path),
        "training_run": str(training_run),
        "observed_sha256": observed,
        "primary_policy": "final-equals-checkpoint-20",
    }


@contextmanager
def install_boundary_training_verifier(
    contract: dict[str, Any],
) -> Iterator[None]:
    """Extend the shared verifier without changing the frozen evaluator."""

    original = formal.verify_training_final

    def wrapped(
        active_contract: dict[str, Any],
        training_run: Path,
        base_model: Path,
        sft1: dict[str, Any],
    ) -> dict[str, Any]:
        report = original(active_contract, training_run, base_model, sft1)
        binding = verify_boundary_artifacts(active_contract, training_run)
        _require(
            report["adapter_sha256"],
            active_contract["boundary_training"]["dynamic_sha256"][
                "final_checkpoint20_adapter"
            ],
            "verified final adapter",
        )
        return {**report, "boundary_training_binding": binding}

    formal.verify_training_final = wrapped
    try:
        yield
    finally:
        formal.verify_training_final = original


def pending_plan(
    template: dict[str, Any], implementation: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": "qwen3-v26-boundary300-formal-matched-eval-plan-v1",
        "status": "dry_run_waiting_for_explicit_artifact_bindings",
        "remote_operations_performed": False,
        "arms": ["sft1", "final"],
        "baseline": "sft1",
        "primary_candidate": "final",
        "primary_policy": "unique final equal to checkpoint-20",
        "questions_per_arm": 1534,
        "execution": "fresh sequential arms on the same explicitly selected physical GPU",
        "runtime_and_decode": {
            "same_runtime": True,
            "workers": 24,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 2048,
            "max_model_len": 16384,
            "enable_thinking": True,
        },
        "promotion_gate": template["promotion_gate"],
        "required_explicit_sha256": list(SHA_FIELDS),
        "required_explicit_paths": list(PATH_FIELDS),
        "no_resume": True,
        "gpu_policy": "fail immediately unless explicitly selected GPU is already idle",
        "implementation_sha256": implementation,
        "host_paths": template["host_paths"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind, preflight, or run the boundary300 matched full-dev gate."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-plan", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--boundary-manifest-sha256")
    parser.add_argument("--train300-sha256")
    parser.add_argument("--training-run-manifest-sha256")
    parser.add_argument("--training-implementation-lock-sha256")
    parser.add_argument("--final-checkpoint20-adapter-sha256")
    parser.add_argument("--boundary-manifest-path")
    parser.add_argument("--train300-path")
    parser.add_argument("--training-run-path")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--maximum-used-mib", type=int, default=512)
    parser.add_argument("--model-ready-timeout", type=int, default=900)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    template = formal.load_object(args.contract)
    validate_boundary_template(template)
    inert = {field: "0" * 64 for field in SHA_FIELDS}
    inert_paths = {
        "boundary_manifest": (
            "/home/dengyan/tabular_rl_outputs/inert_boundary_screen/"
            "boundary_selection/boundary_cohort_manifest.json"
        ),
        "train_tasks": (
            "/home/dengyan/tabular_rl_outputs/inert_boundary_screen/"
            "boundary_selection/train300.jsonl"
        ),
        "training_run": (
            "/home/dengyan/tabular_rl_outputs/"
            "qwen3_8b_atomic_v26_boundary300_vanilla_grpo_20260812/"
            "train300_two_pass_seed20260812"
        ),
    }
    implementation = formal.verify_implementation(
        bind_contract(template, inert, inert_paths)
    )
    if args.print_plan:
        print(json.dumps(pending_plan(template, implementation), ensure_ascii=False, indent=2))
        return 0

    bindings = {
        "boundary_manifest": args.boundary_manifest_sha256,
        "train300": args.train300_sha256,
        "training_run_manifest": args.training_run_manifest_sha256,
        "training_implementation_lock": args.training_implementation_lock_sha256,
        "final_checkpoint20_adapter": args.final_checkpoint20_adapter_sha256,
    }
    paths = {
        "boundary_manifest": args.boundary_manifest_path,
        "train_tasks": args.train300_path,
        "training_run": args.training_run_path,
    }
    contract = bind_contract(template, bindings, paths)
    if args.gpu_id is None or args.gpu_id < 0:
        raise SystemExit("--gpu-id is required and must be non-negative")
    with install_boundary_training_verifier(contract):
        with formal.exclusive_gpu_lock(args.gpu_id):
            if args.preflight:
                report = formal.verify_host_assets(
                    contract,
                    gpu_id=args.gpu_id,
                    maximum_used_mib=args.maximum_used_mib,
                )
                report.pop("source_rows", None)
                print(
                    json.dumps(
                        {"status": "boundary300_preflight_ok", **report},
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
        print(f"boundary300 formal matched evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
