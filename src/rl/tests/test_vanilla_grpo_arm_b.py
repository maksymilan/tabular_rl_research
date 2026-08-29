from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.rl.diagnostics.audit_vanilla_grpo_arm_b_k16_validation import (
    audit_validation,
)
from src.rl.diagnostics.audit_vanilla_grpo_arm_b_wide_k8_screen import (
    audit_wide_screen,
)
from src.rl.diagnostics.validate_arm_b_training_inputs import validate
from src.rl.prepare_vanilla_grpo_arm_b_wide3000 import freeze_wide_cohort
from src.rl.select_vanilla_grpo_arm_b_k16 import (
    _selection_manifest,
    confirm_and_select,
)
from src.rl.tests.arm_b_test_fixtures import (
    generation_artifacts,
    task,
    write_json,
    write_jsonl,
)
from src.rl.vanilla_grpo_arm_b import (
    ARM_B_SEED_REGISTRY,
    F1_GENERATION_SEED,
    F2_GENERATION_SEED,
    F3_GENERATION_SEED,
    INITIAL_ADAPTER_SHA256,
    SELECTION_VALIDATION_CONTRACT,
    VALIDATION_AUDIT_CONTRACT,
    WIDE_COHORT_NAMESPACE,
    WIDE_COHORT_SCHEMA,
    WIDE_COHORT_STATUS,
    WIDE_SELECTION_SEED,
    audit_fixed_policy_groups,
    parse_json_bytes,
    parse_jsonl_bytes,
    sha256_bytes,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def _boundary_pool(
    root: Path,
    name: str,
    tasks: list[dict],
    *,
    correct_counts: list[int],
) -> dict[str, object]:
    """Write one real 600-task Arm-A screen pool consumed by ``_load_pool``."""

    assert len(tasks) == len(correct_counts) == 600
    tasks_path = root / f"{name}.jsonl"
    task_bytes = write_jsonl(tasks_path, tasks)
    groups = []
    for row, correct in zip(tasks, correct_counts, strict=True):
        usable = True
        mixed = 1 <= correct <= 7
        groups.append(
            {
                "task_id": row["example_id"],
                "example_index": row["example_index"],
                "correct_count": correct,
                "uncertainty": correct * (8 - correct) if mixed else 0,
                "usable": usable,
                "mixed_boundary": mixed,
                "core_boundary": usable and 2 <= correct <= 6,
                "contamination": [],
            }
        )
    audit = {
        "schema_version": "vanilla-grpo-boundary-screen-audit-v1",
        "contract": {
            "tasks": 600,
            "group_size": 8,
            "optimizer_updates": 0,
            "initial_adapter_sha256": INITIAL_ADAPTER_SHA256,
            "reward_mode": "result-only",
            "result_reward_profile": "binary",
        },
        "inputs": {
            "tasks": str(tasks_path.resolve()),
            "tasks_sha256": sha256_bytes(task_bytes),
            "manifest_sha256": "a" * 64,
            "trajectories_sha256": "b" * 64,
        },
        "groups": groups,
        "issues": [],
        "issue_counts": {},
        "status": {"audit_passes": True, "pool_admitted": True},
    }
    audit_path = root / f"{name}.audit.json"
    audit_bytes = write_json(audit_path, audit)
    return {
        "audit_path": audit_path,
        "audit_sha256": sha256_bytes(audit_bytes),
        "tasks_path": tasks_path,
        "tasks_sha256": sha256_bytes(task_bytes),
    }


def _wide_sources(root: Path) -> tuple[dict[str, Path], dict[str, str], dict[str, tuple[int, int]]]:
    """Build a small public-only source universe with one example-index alias."""

    source_rows = [task(index) for index in range(1841)]
    for index, row in enumerate(source_rows):
        # Keep the public length distribution stationary across index ranges;
        # decimal-id width must not become an accidental selection axis here.
        row["question"] = "question " + ("x" * (20 + index % 50))
    paths = {
        "reference": root / "reference.jsonl",
        "eligible": root / "eligible.jsonl",
        "baseline_eval300": root / "baseline_alias.jsonl",
        "sft1_index": root / "sft1_empty.jsonl",
        "old_mixed60": root / "old_mixed_empty.jsonl",
    }
    payloads = {
        "reference": write_jsonl(paths["reference"], source_rows),
        "eligible": write_jsonl(paths["eligible"], source_rows),
        # Different task id, same underlying BIRD example.  The preparer must
        # exclude bird_train_01200 by example_index, not merely this alias id.
        "baseline_eval300": write_jsonl(
            paths["baseline_eval300"],
            [{"example_id": "legacy_alias_01200", "example_index": 1200}],
        ),
        "sft1_index": write_jsonl(paths["sft1_index"], []),
        "old_mixed60": write_jsonl(paths["old_mixed60"], []),
    }
    hashes = {role: sha256_bytes(payload) for role, payload in payloads.items()}
    counts = {
        "reference": (1841, 1841),
        "eligible": (1841, 1841),
        "baseline_eval300": (1, 1),
        "sft1_index": (0, 0),
        "old_mixed60": (0, 0),
    }
    return paths, hashes, counts


def _freeze_readiness_wide(root: Path) -> tuple[Path, Path]:
    paths, hashes, counts = _wide_sources(root / "sources")
    rows = _rows(paths["eligible"])
    s1 = _boundary_pool(root, "s1", rows[:600], correct_counts=[0] * 600)
    s2 = _boundary_pool(root, "s2", rows[600:1200], correct_counts=[0] * 600)
    cohort_path = root / "wide_manifest.json"
    output_path = root / "wide640.jsonl"
    freeze_wide_cohort(
        trigger_mode="readiness",
        screen_audit_paths=[s1["audit_path"], s2["audit_path"]],  # type: ignore[list-item]
        expected_screen_audit_hashes=[s1["audit_sha256"], s2["audit_sha256"]],  # type: ignore[list-item]
        screen_task_paths=[s1["tasks_path"], s2["tasks_path"]],  # type: ignore[list-item]
        expected_screen_task_hashes=[s1["tasks_sha256"], s2["tasks_sha256"]],  # type: ignore[list-item]
        reference_path=paths["reference"],
        eligible_path=paths["eligible"],
        baseline_eval300_path=paths["baseline_eval300"],
        sft1_index_path=paths["sft1_index"],
        old_mixed60_path=paths["old_mixed60"],
        output_path=output_path,
        manifest_path=cohort_path,
        remote_db_root=root / "remote_db",
        expected_count=640,
        expected_source_hashes=hashes,
        expected_source_counts=counts,
    )
    selected = _rows(output_path)
    assert len(selected) == 640
    assert 1200 not in {row["example_index"] for row in selected}
    return cohort_path, output_path


def _arm_a_selection_contract() -> dict:
    return {
        "boundary_records": 332,
        "train_records": 300,
        "validation_records": 32,
        "formal_training": {
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
        },
        "validation": {
            "policy": "fresh initial-SFT1",
            "records": 32,
            "group_size": 8,
            "seed": 20260813,
            "screen_seed_must_differ": 20260812,
            "generation_seed_scheme": "sha256-task-sample-turn-v1",
            "gate": "same >=20/32 mixed probe gate",
            "runtime_contamination_allowed": False,
            "screen_trajectories_reused": False,
        },
        "primary_checkpoint": "final-step20-only",
    }


def _freeze_validation_wide(root: Path) -> Path:
    """Exercise a structurally valid but terminal-gate-failed Arm-A validation."""

    paths, hashes, counts = _wide_sources(root / "sources")
    rows = _rows(paths["eligible"])
    pool = _boundary_pool(
        root,
        "s1_ready",
        rows[:600],
        correct_counts=[4] * 300 + [1] * 100 + [8] * 200,
    )
    validation_tasks = rows[:32]
    generation_path, trajectories_path, validation_bytes, trajectories_bytes = (
        generation_artifacts(
            root / "arm_a_validation",
            validation_tasks,
            group_size=8,
            seed=20260813,
            correct_counts=[4] * 19 + [8] * 13,
        )
    )
    validation_tasks_path = root / "arm_a_validation" / "tasks.jsonl"
    pool_binding = {
        "audit_path": str(Path(pool["audit_path"]).resolve()),
        "audit_sha256": pool["audit_sha256"],
        "tasks_path": str(Path(pool["tasks_path"]).resolve()),
        "tasks_sha256": pool["tasks_sha256"],
    }
    selection = {
        "schema_version": "policy-boundary-grpo-cohort-v1",
        "status": "frozen_boundary_training_cohort",
        "screen_pools": [pool_binding],
        "selection": {
            "seed": "qwen3-v26-boundary332-v1-20260812",
            "acceptance_gates": {"all": True},
        },
        "contract": _arm_a_selection_contract(),
        "outputs": {
            "validation": {
                "path": str(validation_tasks_path.resolve()),
                "sha256": sha256_bytes(validation_bytes),
                "records": 32,
            }
        },
        "task_ids": {
            "validation": [row["example_id"] for row in validation_tasks]
        },
    }
    selection_path = root / "arm_a_selection.json"
    selection_bytes = write_json(selection_path, selection)
    cohort_path = root / "wide_validation_trigger_manifest.json"
    freeze_wide_cohort(
        trigger_mode="validation",
        screen_audit_paths=[pool["audit_path"]],  # type: ignore[list-item]
        expected_screen_audit_hashes=[pool["audit_sha256"]],  # type: ignore[list-item]
        screen_task_paths=[pool["tasks_path"]],  # type: ignore[list-item]
        expected_screen_task_hashes=[pool["tasks_sha256"]],  # type: ignore[list-item]
        reference_path=paths["reference"],
        eligible_path=paths["eligible"],
        baseline_eval300_path=paths["baseline_eval300"],
        sft1_index_path=paths["sft1_index"],
        old_mixed60_path=paths["old_mixed60"],
        output_path=root / "wide_validation_trigger640.jsonl",
        manifest_path=cohort_path,
        remote_db_root=root / "remote_db",
        arm_a_selection_manifest_path=selection_path,
        expected_arm_a_selection_manifest_sha256=sha256_bytes(selection_bytes),
        arm_a_validation_manifest_path=generation_path,
        expected_arm_a_validation_manifest_sha256=sha256_bytes(
            generation_path.read_bytes()
        ),
        arm_a_validation_tasks_path=validation_tasks_path,
        expected_arm_a_validation_tasks_sha256=sha256_bytes(validation_bytes),
        arm_a_validation_trajectories_path=trajectories_path,
        expected_arm_a_validation_trajectories_sha256=sha256_bytes(
            trajectories_bytes
        ),
        expected_count=640,
        expected_source_hashes=hashes,
        expected_source_counts=counts,
    )
    return cohort_path


def test_fixed_group_audit_rejects_missing_process_update_and_row_reorders(
    tmp_path: Path,
) -> None:
    tasks = [task(1), task(2)]
    manifest_path, trajectory_path, task_bytes, trajectory_bytes = generation_artifacts(
        tmp_path, tasks, group_size=8, seed=F1_GENERATION_SEED, correct_counts=[4, 4]
    )

    def run(rows: list[dict]) -> None:
        audit_fixed_policy_groups(
            generation_manifest=_load(manifest_path),
            tasks=tasks,
            trajectories=rows,
            tasks_sha256=sha256_bytes(task_bytes),
            trajectories_sha256=sha256_bytes(trajectory_bytes),
            tasks_path=tmp_path / "tasks.jsonl",
            expected_tasks=2,
            group_size=8,
            seed=F1_GENERATION_SEED,
        )

    original = _rows(trajectory_path)
    missing = copy.deepcopy(original)
    del missing[0]["sample"]["process_update"]
    with pytest.raises(ValueError, match="explicit boolean"):
        run(missing)

    reordered = copy.deepcopy(original)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    reordered[0]["sequence"], reordered[1]["sequence"] = 0, 1
    with pytest.raises(ValueError, match="identity mismatch"):
        run(reordered)

    interleaved = copy.deepcopy(original)
    interleaved[1], interleaved[8] = interleaved[8], interleaved[1]
    interleaved[1]["sequence"], interleaved[8]["sequence"] = 1, 8
    with pytest.raises(ValueError, match="identity mismatch"):
        run(interleaved)


def test_fixed_group_audit_retains_recovered_timeout_as_contamination(
    tmp_path: Path,
) -> None:
    tasks = [task(10), task(11)]
    manifest_path, trajectory_path, task_bytes, _ = generation_artifacts(
        tmp_path,
        tasks,
        group_size=8,
        seed=F1_GENERATION_SEED,
        correct_counts=[8, 0],
    )
    rows = _rows(trajectory_path)
    recovered = rows[0]["sample"]
    recovered["process_update"] = False
    recovered["reward"] = 0.0
    recovered["audit_record"]["result_reward"]["value"] = 0.0
    recovered["audit_record"]["error_events"] = [
        {"error_type": "timeout_error", "recovered": True}
    ]
    # The second task is an ordinary semantic miss: binary reward zero, but
    # process_update=true and no runtime marker means it remains usable.
    trajectory_bytes = write_jsonl(trajectory_path, rows)
    manifest = _load(manifest_path)
    manifest["trajectories_sha256"] = sha256_bytes(trajectory_bytes)
    write_json(manifest_path, manifest)

    observed = audit_fixed_policy_groups(
        generation_manifest=manifest,
        tasks=tasks,
        trajectories=rows,
        tasks_sha256=sha256_bytes(task_bytes),
        trajectories_sha256=sha256_bytes(trajectory_bytes),
        tasks_path=tmp_path / "tasks.jsonl",
        expected_tasks=2,
        group_size=8,
        seed=F1_GENERATION_SEED,
    )
    recovered_group, semantic_wrong_group = observed["groups"]
    assert recovered_group["usable"] is False
    assert set(recovered_group["contamination"]) == {
        "process_update_false",
        "timeout_evidence",
    }
    assert semantic_wrong_group["usable"] is True
    assert semantic_wrong_group["correct_count"] == 0
    assert semantic_wrong_group["contamination"] == []


def test_wide_preparer_reaudits_validation_trigger_and_binds_raw_inputs(
    tmp_path: Path,
) -> None:
    cohort = _load(_freeze_validation_wide(tmp_path / "validation_trigger"))
    activation = cohort["activation"]
    assert activation["reason"] == "arm_a_validation_failed"
    assert activation["recomputed_status"] == {
        "passes": False,
        "validation_admitted": False,
    }
    assert activation["failed_terminal_gates"] == {
        "no_runtime_contamination": True,
        "at_least_20_mixed_outcome_groups": False,
    }
    bound = activation["arm_a_failure_evidence"]["bound_inputs"]
    assert bound["screen_pools"] == activation["actual_screen_pools"]
    assert bound["validation"] == activation["bound_inputs"]


def build_real_arm_b_pipeline(tmp_path: Path) -> dict[str, Path]:
    """Build preparer -> F1 -> F2 -> F3 -> training admission artifacts."""

    cohort_path, wide_tasks_path = _freeze_readiness_wide(tmp_path / "preparer")
    cohort = _load(cohort_path)
    cohort_bytes = cohort_path.read_bytes()
    wide_tasks = _rows(wide_tasks_path)
    f1_root = tmp_path / "f1"
    f1_manifest, f1_trajectories, wide_bytes, f1_trajectory_bytes = generation_artifacts(
        f1_root,
        wide_tasks,
        group_size=8,
        seed=F1_GENERATION_SEED,
        correct_counts=[4] * 640,
    )
    # The rollout generator writes an identical convenience task file.  Bind
    # its manifest back to the immutable preparer output used by F1.
    assert wide_bytes == wide_tasks_path.read_bytes()
    f1_generation = _load(f1_manifest)
    f1_generation["tasks_path"] = str(wide_tasks_path.resolve())
    f1_generation["tasks_sha256"] = sha256_bytes(wide_bytes)
    write_json(f1_manifest, f1_generation)
    f1_audit, confirmation_bytes = audit_wide_screen(
        cohort_manifest=cohort,
        generation_manifest=f1_generation,
        tasks=wide_tasks,
        trajectories=_rows(f1_trajectories),
        expected_cohort_manifest_sha256=sha256_bytes(cohort_bytes),
        cohort_manifest_sha256=sha256_bytes(cohort_bytes),
        generation_manifest_sha256=sha256_bytes(f1_manifest.read_bytes()),
        tasks_sha256=sha256_bytes(wide_bytes),
        trajectories_sha256=sha256_bytes(f1_trajectory_bytes),
        cohort_manifest_path=cohort_path,
        generation_manifest_path=f1_manifest,
        tasks_path=wide_tasks_path,
        trajectories_path=f1_trajectories,
        output_dir=f1_root,
        expected_tasks=640,
    )
    assert f1_audit["status"]["confirmation_ready"] is True
    assert confirmation_bytes is not None
    confirmation_path = f1_root / "confirmation640.jsonl"
    confirmation_path.write_bytes(confirmation_bytes)
    f1_audit_path = f1_root / "arm_b_wide_k8_screen_audit.json"
    f1_audit_bytes = write_json(f1_audit_path, f1_audit)

    confirmation_tasks = parse_jsonl_bytes(confirmation_bytes, path=confirmation_path)
    f2_root = tmp_path / "selection"
    f2_manifest, f2_trajectories, confirmation_bytes2, f2_trajectory_bytes = generation_artifacts(
        tmp_path / "f2_generation",
        confirmation_tasks,
        group_size=16,
        seed=F2_GENERATION_SEED,
        correct_counts=[8] * 640,
    )
    # F2 generation has a different physical tasks path.  Bind the manifest to
    # the immutable confirmation640 path exactly as a real launcher does.
    f2_generation = _load(f2_manifest)
    f2_generation["tasks_path"] = str(confirmation_path.resolve())
    f2_generation["tasks_sha256"] = sha256_bytes(confirmation_bytes)
    write_json(f2_manifest, f2_generation)
    f2_audit, rendered = confirm_and_select(
        wide_audit=f1_audit,
        generation_manifest=f2_generation,
        tasks=confirmation_tasks,
        trajectories=_rows(f2_trajectories),
        expected_wide_audit_sha256=sha256_bytes(f1_audit_bytes),
        wide_audit_sha256=sha256_bytes(f1_audit_bytes),
        generation_manifest_sha256=sha256_bytes(f2_manifest.read_bytes()),
        tasks_sha256=sha256_bytes(confirmation_bytes),
        trajectories_sha256=sha256_bytes(f2_trajectory_bytes),
        wide_audit_path=f1_audit_path,
        generation_manifest_path=f2_manifest,
        tasks_path=confirmation_path,
        trajectories_path=f2_trajectories,
        output_dir=f2_root,
    )
    assert f2_audit["status"]["selection_ready"] is True
    f2_audit_path = f2_root / "arm_b_k16_confirmation_audit.json"
    f2_audit_bytes = write_json(f2_audit_path, f2_audit)
    for role, name in (
        ("selected", "selected384.jsonl"),
        ("train", "train320.jsonl"),
        ("validation", "validation64.jsonl"),
    ):
        (f2_root / name).write_bytes(rendered[role])
    selection = _selection_manifest(
        audit=f2_audit,
        audit_path=f2_audit_path,
        audit_sha256=sha256_bytes(f2_audit_bytes),
        rendered=rendered,
        output_dir=f2_root,
    )
    selection_path = f2_root / "arm_b_selection_manifest.json"
    selection_bytes = write_json(selection_path, selection)

    validation_tasks = parse_jsonl_bytes(
        rendered["validation"], path=f2_root / "validation64.jsonl"
    )
    f3_root = tmp_path / "f3"
    f3_manifest, f3_trajectories, validation_bytes, f3_trajectory_bytes = generation_artifacts(
        f3_root,
        validation_tasks,
        group_size=16,
        seed=F3_GENERATION_SEED,
        correct_counts=[8] * 64,
    )
    f3_generation = _load(f3_manifest)
    validation_path = f2_root / "validation64.jsonl"
    f3_generation["tasks_path"] = str(validation_path.resolve())
    f3_generation["tasks_sha256"] = sha256_bytes(rendered["validation"])
    write_json(f3_manifest, f3_generation)
    f3_audit = audit_validation(
        selection_manifest=selection,
        generation_manifest=f3_generation,
        tasks=validation_tasks,
        trajectories=_rows(f3_trajectories),
        expected_selection_manifest_sha256=sha256_bytes(selection_bytes),
        selection_manifest_sha256=sha256_bytes(selection_bytes),
        generation_manifest_sha256=sha256_bytes(f3_manifest.read_bytes()),
        tasks_sha256=sha256_bytes(rendered["validation"]),
        trajectories_sha256=sha256_bytes(f3_trajectory_bytes),
        selection_manifest_path=selection_path,
        generation_manifest_path=f3_manifest,
        tasks_path=validation_path,
        trajectories_path=f3_trajectories,
    )
    assert f3_audit["contract"] == VALIDATION_AUDIT_CONTRACT
    assert selection["contract"]["validation"] == SELECTION_VALIDATION_CONTRACT
    validation_audit_path = f3_root / "arm_b_k16_validation_audit.json"
    validation_audit_bytes = write_json(validation_audit_path, f3_audit)

    admitted = validate(
        selection_manifest_path=selection_path,
        expected_selection_manifest_sha256=sha256_bytes(selection_bytes),
        train320_path=f2_root / "train320.jsonl",
        expected_train320_sha256=sha256_bytes(rendered["train"]),
        validation64_audit_path=validation_audit_path,
        expected_validation64_audit_sha256=sha256_bytes(validation_audit_bytes),
    )
    assert admitted["status"] == "training_inputs_admitted"
    return {
        "cohort": cohort_path,
        "selection": selection_path,
        "train": f2_root / "train320.jsonl",
        "validation_audit": validation_audit_path,
    }


def test_real_preparer_f1_f2_f3_builders_feed_training_validator(
    tmp_path: Path,
) -> None:
    paths = build_real_arm_b_pipeline(tmp_path)
    assert _load(paths["cohort"])["activation"]["reason"] == (
        "arm_a_s1_s2_readiness_failed"
    )
