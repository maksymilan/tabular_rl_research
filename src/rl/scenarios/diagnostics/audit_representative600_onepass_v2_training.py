#!/usr/bin/env python3
"""Fail-closed completion audit for representative600-v2 one-pass GRPO.

The audit rechecks all 4,800 trajectories with the exact same v2 runtime-
exclusion classifier used by the synchronous step-5 gate.  Known structured
generation-length, timeout, and context-overflow trajectories may be excluded
from optimization; unknown, contradictory, unexcluded, OOM, or tokenization-
warning trajectories fail closed.  Only ``final`` byte-identical to
``checkpoint-20`` is admitted for evaluation.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl.diagnostics.validation import require as _diagnostic_require

try:
    from rl.scenarios.diagnostics import audit_vanilla_grpo_train600_v2_step5 as gate
except ImportError:
    import rl.scenarios.diagnostics.audit_vanilla_grpo_train600_v2_step5 as gate


SCHEMA_VERSION = "representative600-onepass-v2-training-completion-audit-v1"
EXPECTED_EXPERIMENT_NAME = (
    "qwen3_8b_atomic_v26_vanilla_grpo_representative600_onepass_v2"
)
EXPECTED_CONFIG_SHA256 = (
    "c932d7f61ea55d20739fe433808aca1fe95117e0d68a3d7ee93717e5cd175a1e"
)
EXPECTED_MAX_NEW_TOKENS = 3072
EXPECTED_MAX_CONTEXT_TOKENS = 20480
GATE_AUDITOR = "src/rl/scenarios/diagnostics/audit_vanilla_grpo_train600_v2_step5.py"


def _load_private_engine():
    source = Path(__file__).with_name(
        "audit_representative600_onepass_training.py"
    )
    diagnostics = str(source.parent)
    inserted = diagnostics not in sys.path
    if inserted:
        sys.path.insert(0, diagnostics)
    try:
        spec = importlib.util.spec_from_file_location(
            "_representative600_v2_completion_engine", source
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load completion engine: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted:
            sys.path.remove(diagnostics)


_engine = _load_private_engine()
_original_validate_experiment_config = _engine._validate_experiment_config
_original_validate_manifest = _engine._validate_manifest


def _require(condition: bool, message: str) -> None:
    _diagnostic_require(condition, message)


def _validate_experiment_config(config: Mapping[str, Any]) -> None:
    _require(
        config.get("experiment_name") == EXPECTED_EXPERIMENT_NAME,
        "embedded experiment config experiment_name mismatch",
    )
    rollout = config.get("rollout") or {}
    _require(
        rollout.get("max_new_tokens") == EXPECTED_MAX_NEW_TOKENS,
        "embedded experiment config max_new_tokens mismatch",
    )
    _require(
        rollout.get("max_context_tokens") == EXPECTED_MAX_CONTEXT_TOKENS,
        "embedded experiment config max_context_tokens mismatch",
    )
    translated = copy.deepcopy(dict(config))
    translated["experiment_name"] = (
        "qwen3_8b_atomic_v26_vanilla_grpo_representative600_onepass"
    )
    translated["rollout"]["max_new_tokens"] = 2048
    translated["rollout"]["max_context_tokens"] = 16384
    _original_validate_experiment_config(translated)


def _validate_manifest(
    manifest: Mapping[str, Any], *, run_dir: Path, tasks_manifest_path: Path
) -> None:
    checkpoint_gate = manifest.get("checkpoint_gate") or {}
    _require(
        checkpoint_gate.get("script_relative_path") == GATE_AUDITOR,
        "v2 checkpoint gate script path mismatch",
    )
    rollout = manifest.get("rollout_settings") or {}
    _require(
        rollout.get("max_new_tokens") == EXPECTED_MAX_NEW_TOKENS,
        "run manifest max_new_tokens mismatch",
    )
    _require(
        rollout.get("max_context_tokens") == EXPECTED_MAX_CONTEXT_TOKENS,
        "run manifest max_context_tokens mismatch",
    )
    translated = copy.deepcopy(dict(manifest))
    translated["checkpoint_gate"]["script_relative_path"] = (
        "src/rl/scenarios/diagnostics/audit_vanilla_grpo_train600_step5.py"
    )
    translated["rollout_settings"]["max_new_tokens"] = 2048
    translated["rollout_settings"]["max_context_tokens"] = 16384
    _original_validate_manifest(
        translated,
        run_dir=run_dir,
        tasks_manifest_path=tasks_manifest_path,
    )


def _validate_runtime_outcome(row: Mapping[str, Any], position: int) -> None:
    try:
        gate._runtime_exclusion_kind(
            dict(row),
            expected_max_tokens=EXPECTED_MAX_NEW_TOKENS,
            expected_max_context_tokens=EXPECTED_MAX_CONTEXT_TOKENS,
        )
    except gate.InputStructureError as exc:
        raise ValueError(f"rollout {position} runtime exclusion invalid: {exc}") from exc


_engine.SCHEMA_VERSION = SCHEMA_VERSION
_engine.EXPECTED_CONFIG_SHA256 = EXPECTED_CONFIG_SHA256
_engine.REQUIRED_IMPLEMENTATION_FILES = (
    set(_engine.REQUIRED_IMPLEMENTATION_FILES)
    - {"src/rl/scenarios/diagnostics/audit_vanilla_grpo_train600_step5.py"}
) | {GATE_AUDITOR}
_engine._validate_experiment_config = _validate_experiment_config
_engine._validate_manifest = _validate_manifest
_engine._validate_runtime_outcome = _validate_runtime_outcome


def _runtime_exclusion_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    excluded_ids: dict[str, list[str]] = {
        kind: [] for kind in gate.KNOWN_RUNTIME_EXCLUSIONS
    }
    timeout_trajectories = 0
    timeout_events = 0
    for position, row in enumerate(rows):
        structured_timeouts = gate._deduplicated_structured_timeout_events(row)
        timeout_marker = gate.v1._timeout_observed(row)
        timeout_trajectories += int(timeout_marker)
        timeout_events += len(structured_timeouts)
        try:
            kind = gate._runtime_exclusion_kind(
                row,
                expected_max_tokens=EXPECTED_MAX_NEW_TOKENS,
                expected_max_context_tokens=EXPECTED_MAX_CONTEXT_TOKENS,
            )
        except gate.InputStructureError as exc:
            raise ValueError(
                f"rollout {position} runtime exclusion invalid: {exc}"
            ) from exc
        counts[kind or "eligible"] += 1
        if kind is not None:
            excluded_ids[kind].append(str(row.get("trajectory_id")))
    eligible_by_step = [
        sum(
            row.get("process_update") is True
            for row in rows[
                step * _engine.ROWS_PER_STEP : (step + 1) * _engine.ROWS_PER_STEP
            ]
        )
        for step in range(_engine.EXPECTED_STEPS)
    ]
    eligible_fraction_by_step = [
        eligible / _engine.ROWS_PER_STEP for eligible in eligible_by_step
    ]
    denominator = len(rows)
    eligible_fraction = counts["eligible"] / denominator if denominator else 0.0
    length_fraction = (
        counts["generation_length"] / denominator if denominator else 0.0
    )
    timeout_fraction = timeout_trajectories / denominator if denominator else 0.0
    context_fraction = (
        counts["context_overflow"] / denominator if denominator else 0.0
    )
    checks = {
        "exact_4800_runtime_classifications": (
            denominator == _engine.EXPECTED_ROLLOUTS
        ),
        "overall_eligible_at_least_95_percent": (
            eligible_fraction >= gate.MIN_OVERALL_ELIGIBLE_FRACTION
        ),
        "each_step_eligible_at_least_90_percent": (
            len(eligible_fraction_by_step) == _engine.EXPECTED_STEPS
            and all(
                fraction >= gate.MIN_STEP_ELIGIBLE_FRACTION
                for fraction in eligible_fraction_by_step
            )
        ),
        "generation_length_exclusions_at_most_5_percent": (
            length_fraction <= gate.MAX_GENERATION_LENGTH_FRACTION
        ),
        "timeout_trajectories_at_most_2_percent": (
            timeout_fraction <= gate.MAX_TIMEOUT_TRAJECTORY_FRACTION
        ),
        "context_overflow_at_most_0_25_percent": (
            context_fraction <= gate.MAX_CONTEXT_OVERFLOW_FRACTION
        ),
    }
    return {
        "policy": (
            "known structured exclusions are audit-only and receive no gradient; "
            "unknown or contradictory exclusions fail closed"
        ),
        "expected_max_new_tokens": EXPECTED_MAX_NEW_TOKENS,
        "counts": {
            "eligible": counts["eligible"],
            **{
                kind: counts[kind]
                for kind in gate.KNOWN_RUNTIME_EXCLUSIONS
            },
        },
        "checks": checks,
        "eligible_fraction": eligible_fraction,
        "eligible_by_step": eligible_by_step,
        "eligible_fraction_by_step": eligible_fraction_by_step,
        "known_exclusion_fraction": 1.0 - eligible_fraction,
        "generation_length_exclusion_fraction": length_fraction,
        "structured_timeout_trajectories": timeout_trajectories,
        "structured_timeout_events": timeout_events,
        "structured_timeout_trajectory_fraction": timeout_fraction,
        "context_overflow_exclusion_fraction": context_fraction,
        "excluded_trajectory_ids": excluded_ids,
        "unknown_or_contradictory": 0,
    }


def audit(
    run_dir: Path,
    tasks_path: Path,
    cohort_manifest_path: Path,
    experiment_config_path: Path,
    expected_implementation_lock_sha256: str,
) -> dict[str, Any]:
    common = _engine.common
    _require(
        run_dir.is_dir() and not run_dir.is_symlink(), f"invalid run dir: {run_dir}"
    )
    required = {
        "run_manifest": run_dir / "run_manifest.json",
        "implementation_lock": run_dir / "implementation_lock.json",
        "rollouts": run_dir / "rollouts.jsonl",
        "training_precision": run_dir / "training_precision.json",
        "checkpoint_state": run_dir / "checkpoint-20/trainer_state.json",
        "checkpoint_weights": run_dir / "checkpoint-20/adapter_model.safetensors",
        "checkpoint_config": run_dir / "checkpoint-20/adapter_config.json",
        "checkpoint_optimizer": run_dir / "checkpoint-20/optimizer.pt",
        "checkpoint_scheduler": run_dir / "checkpoint-20/scheduler.pt",
        "checkpoint_rng_state": run_dir / "checkpoint-20/rng_state.pth",
        "checkpoint_training_args": run_dir / "checkpoint-20/training_args.bin",
        "final_weights": run_dir / "final/adapter_model.safetensors",
        "final_config": run_dir / "final/adapter_config.json",
        "final_training_args": run_dir / "final/training_args.bin",
        "step5_gate": run_dir / "step5_gate.json",
        "step5_gate_invocation": run_dir / "checkpoint_gate_step5_invocation.json",
        "tasks": tasks_path,
        "cohort_manifest": cohort_manifest_path,
        "experiment_config": experiment_config_path,
    }
    for label, path in required.items():
        common._regular(path, label)
    _require(
        common.sha256_file(experiment_config_path) == EXPECTED_CONFIG_SHA256,
        "experiment config SHA mismatch",
    )
    _, _, tasks = _engine._validate_cohort(tasks_path, cohort_manifest_path)

    manifest = common.load_object(required["run_manifest"])
    _validate_manifest(
        manifest,
        run_dir=run_dir,
        tasks_manifest_path=cohort_manifest_path,
    )
    initial_adapter_path = Path(str(manifest.get("adapter_path") or ""))
    _require(
        str(initial_adapter_path) == _engine.EXPECTED_INITIAL_ADAPTER_PATH,
        "run manifest initial adapter path mismatch",
    )
    step5_receipt = common.load_object(required["step5_gate"])
    recomputed_step5 = gate.audit(
        run_dir,
        tasks_path,
        cohort_manifest_path,
        initial_adapter_path,
        allow_continued_run=True,
    )
    _require(
        step5_receipt == recomputed_step5,
        "step5 gate receipt differs from its stable full recomputation",
    )
    _require(
        step5_receipt.get("schema_version") == gate.SCHEMA_VERSION,
        "step5 gate receipt schema mismatch",
    )
    _require(
        step5_receipt.get("status")
        == {
            "outcome": "pass",
            "passes": True,
            "continuation_admitted": True,
            "exit_code": 0,
        },
        "step5 gate did not admit continuation",
    )
    checkpoint_gate = manifest.get("checkpoint_gate") or {}
    _require(
        step5_receipt.get("auditor_sha256")
        == checkpoint_gate.get("script_sha256"),
        "step5 receipt/auditor source identity mismatch",
    )
    invocation = common.load_object(required["step5_gate_invocation"])
    _engine._require_fields(
        invocation,
        {
            "schema_version": "trl-checkpoint-gate-invocation-v1",
            "step": 5,
            "verify_existing": False,
            "script_relative_path": checkpoint_gate.get("script_relative_path"),
            "script_sha256": checkpoint_gate.get("script_sha256"),
            "returncode": 0,
            "receipt": str((run_dir / "step5_gate.json").resolve()),
            "receipt_sha256": common.sha256_file(required["step5_gate"]),
        },
        "step5 gate invocation",
    )
    resume_path = manifest.get("resume_from_checkpoint")
    if resume_path is not None:
        match = _engine.CHECKPOINT_PATTERN.fullmatch(Path(str(resume_path)).name)
        _require(match is not None, "run manifest resume checkpoint path invalid")
        if int(match.group(1)) >= 5:
            verification_path = (
                run_dir / "checkpoint_gate_step5_resume_verification.json"
            )
            common._regular(verification_path, "step5_gate_resume_verification")
            verification = common.load_object(verification_path)
            _engine._require_fields(
                verification,
                {
                    "schema_version": "trl-checkpoint-gate-invocation-v1",
                    "step": 5,
                    "verify_existing": True,
                    "script_sha256": checkpoint_gate.get("script_sha256"),
                    "returncode": 0,
                    "receipt": str((run_dir / "step5_gate.json").resolve()),
                    "receipt_sha256": common.sha256_file(required["step5_gate"]),
                },
                "step5 gate resume verification",
            )

    implementation_files = _engine._validate_implementation_lock(
        run_dir,
        manifest,
        required["implementation_lock"],
        expected_implementation_lock_sha256,
    )
    primary_checkpoint, intermediate_checkpoints = (
        _engine._validate_checkpoint_inventory(run_dir)
    )
    observed_checkpoint_steps = {
        item["step"] for item in [primary_checkpoint, *intermediate_checkpoints]
    }
    _require(
        observed_checkpoint_steps == set(range(1, _engine.EXPECTED_STEPS + 1)),
        "checkpoint inventory is not exact steps 1..20",
    )
    rows = common.load_jsonl(required["rollouts"])
    step_summaries, micro_step_audit = _engine._validate_rollouts(rows, tasks)
    exclusion_summary = _runtime_exclusion_summary(rows)
    failed_runtime_checks = [
        name
        for name, passed in exclusion_summary["checks"].items()
        if not passed
    ]
    _require(
        not failed_runtime_checks,
        "full-run runtime exclusion thresholds failed: "
        f"{failed_runtime_checks}; "
        f"eligible_fraction={exclusion_summary['eligible_fraction']}; "
        f"eligible_fraction_by_step="
        f"{exclusion_summary['eligible_fraction_by_step']}; "
        f"length_fraction="
        f"{exclusion_summary['generation_length_exclusion_fraction']}; "
        f"timeout_fraction="
        f"{exclusion_summary['structured_timeout_trajectory_fraction']}; "
        f"context_fraction="
        f"{exclusion_summary['context_overflow_exclusion_fraction']}",
    )

    state = common.load_object(required["checkpoint_state"])
    _require(
        state.get("global_step") == _engine.EXPECTED_STEPS,
        "checkpoint global_step is not 20",
    )
    _require(
        state.get("max_steps") == _engine.EXPECTED_STEPS,
        "checkpoint max_steps is not 20",
    )
    trainer_metrics = _engine._trainer_metrics(state)
    precision = common._precision_audit(required["training_precision"])
    checkpoint_adapter_sha = common.sha256_file(required["checkpoint_weights"])
    checkpoint_config_sha = common.sha256_file(required["checkpoint_config"])
    _require(
        checkpoint_adapter_sha == common.sha256_file(required["final_weights"]),
        "final/checkpoint-20 adapter weights differ",
    )
    _require(
        checkpoint_config_sha == common.sha256_file(required["final_config"]),
        "final/checkpoint-20 adapter config differs",
    )
    _require(
        common.sha256_file(required["checkpoint_training_args"])
        == common.sha256_file(required["final_training_args"]),
        "final/checkpoint-20 training args differ",
    )
    transients = _engine._forbidden_run_paths(run_dir)
    _require(
        not transients,
        f"transient or symlink artifacts remain: {transients[:3]}",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": {"passes": True, "evaluation_admitted": True},
        "run_dir": str(run_dir),
        "artifact_sha256": {
            label: common.sha256_file(path) for label, path in required.items()
        },
        "identity_lock": {
            "implementation_lock_sha256": expected_implementation_lock_sha256,
            "implementation_files": implementation_files,
            "base_model_identity": _engine.EXPECTED_BASE_MODEL_IDENTITY,
            "protocol_version": _engine.EXPECTED_PROTOCOL_VERSION,
            "protocol_hash": _engine.EXPECTED_PROTOCOL_HASH,
            "task_sha256": _engine.EXPECTED_TASKS_SHA256,
            "task_manifest_sha256": _engine.EXPECTED_COHORT_MANIFEST_SHA256,
            "experiment_config_sha256": EXPECTED_CONFIG_SHA256,
            "step5_gate_script_sha256": checkpoint_gate.get("script_sha256"),
        },
        "contract": {
            "records": _engine.EXPECTED_RECORDS,
            "passes": 1,
            "optimizer_steps": _engine.EXPECTED_STEPS,
            "prompts_per_step": _engine.PROMPTS_PER_STEP,
            "group_size": _engine.GROUP_SIZE,
            "rollout_rows": _engine.EXPECTED_ROLLOUTS,
            "reward": "binary-result-only",
            "process_rank_custom_credit": "disabled",
            "max_new_tokens": EXPECTED_MAX_NEW_TOKENS,
            "max_context_tokens": EXPECTED_MAX_CONTEXT_TOKENS,
        },
        "runtime_exclusions": exclusion_summary,
        "rollout_schedule": {
            "steps": step_summaries,
            "coverage": "all 600 frozen tasks exactly K8 once",
            "policy_micro_step": micro_step_audit,
        },
        "checkpoint_policy": {
            "primary": "final",
            "checkpoint_alias": "checkpoint-20",
            "primary_adapter_sha256": checkpoint_adapter_sha,
            "primary_adapter_config_sha256": checkpoint_config_sha,
            "posthoc_checkpoint_selection_allowed": False,
            "evaluation_candidates": ["final"],
            "primary_checkpoint": primary_checkpoint,
            "intermediate_checkpoints": intermediate_checkpoints,
            "intermediate_checkpoint_policy": (
                "resume-or-audit-only; never evaluation candidates"
            ),
        },
        "trainer_metrics": trainer_metrics,
        "training_precision": precision,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-implementation-lock-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit(
            args.run_dir,
            args.tasks,
            args.cohort_manifest,
            args.experiment_config,
            args.expected_implementation_lock_sha256,
        )
    except Exception as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": {
                "passes": False,
                "evaluation_admitted": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        }
        try:
            _engine._publish_immutable_json(args.output, result)
        except Exception as publish_exc:
            print(
                json.dumps(
                    {
                        "passes": False,
                        "evaluation_admitted": False,
                        "error": str(publish_exc),
                    },
                    ensure_ascii=False,
                )
            )
            return 2
        print(json.dumps(result["status"], ensure_ascii=False))
        return 2
    try:
        _engine._publish_immutable_json(args.output, result)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "passes": False,
                    "evaluation_admitted": False,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(result["status"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
