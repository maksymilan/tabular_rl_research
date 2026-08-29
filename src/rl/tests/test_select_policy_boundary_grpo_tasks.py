from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_DIR))

from select_policy_boundary_grpo_tasks import (  # noqa: E402
    DEFAULT_SEED,
    _load_pool,
    main,
    select_boundary_cohort,
)


def _pool(*, offset: int = 0, mixed: int = 400, core: int = 300) -> dict:
    tasks = {}
    groups = {}
    for position in range(600):
        index = offset + position
        task_id = f"bird_train_{index:05d}"
        task = {
            "example_id": task_id,
            "instance_id": task_id,
            "example_index": index,
            "db_id": f"db_{position % 60}",
            "question": "q" * (30 + position % 50),
            "external_knowledge": "hint" if position % 2 else None,
            "gold_sql": f"select {index}",
        }
        tasks[task_id] = task
        correct = 8
        if position < mixed:
            correct = 4 if position < core else 1
        groups[task_id] = {
            "task_id": task_id,
            "example_index": index,
            "correct_count": correct,
            "uncertainty": correct * (8 - correct) if 1 <= correct <= 7 else 0,
            "usable": True,
            "mixed_boundary": 1 <= correct <= 7,
            "core_boundary": 2 <= correct <= 6,
            "contamination": [],
        }
    return {"tasks": tasks, "groups": groups}


def test_selection_freezes_exact_300_train_and_32_validation() -> None:
    selected, train, validation, details = select_boundary_cohort([_pool()])
    assert len(selected) == 332
    assert len(train) == 300
    assert len(validation) == 32
    assert not ({item["task_id"] for item in train} & {item["task_id"] for item in validation})
    assert details["eligible_mixed_groups"] == 400
    assert details["eligible_core_groups"] == 300
    assert details["distribution"]["unique_databases"] >= 50
    assert all(details["acceptance_gates"].values())


def test_uncertainty_priority_selects_core_before_one_of_eight() -> None:
    selected, *_ = select_boundary_cohort([_pool(mixed=400, core=332)])
    assert all(item["correct_count"] == 4 for item in selected)
    assert all(item["uncertainty"] == 16 for item in selected)


def test_real_representative600_all_c4_passes_distribution_gates() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "data"
        / "rl_inputs"
        / "qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.jsonl"
    )
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    pool = {"tasks": {}, "groups": {}}
    for row in rows:
        task_id = row["example_id"]
        pool["tasks"][task_id] = row
        pool["groups"][task_id] = {
            "task_id": task_id,
            "example_index": row["example_index"],
            "correct_count": 4,
            "uncertainty": 16,
            "usable": True,
            "mixed_boundary": True,
            "core_boundary": True,
            "contamination": [],
        }
    _selected, _train, _validation, details = select_boundary_cohort([pool])
    assert details["distribution"]["database_tv"] <= 0.15
    assert details["distribution"]["length_tv"] <= 0.10
    assert details["distribution"]["knowledge_tv"] <= 0.05
    assert details["distribution"]["unique_databases"] >= 50


def test_gold_mutation_cannot_change_selection_or_split() -> None:
    first = _pool()
    second = copy.deepcopy(first)
    for task in second["tasks"].values():
        task["gold_sql"] = "SECRET CHANGED"
    a = select_boundary_cohort([first], seed=DEFAULT_SEED)
    b = select_boundary_cohort([second], seed=DEFAULT_SEED)
    assert [item["task_id"] for item in a[0]] == [item["task_id"] for item in b[0]]
    assert [item["task_id"] for item in a[1]] == [item["task_id"] for item in b[1]]
    assert [item["task_id"] for item in a[2]] == [item["task_id"] for item in b[2]]


def test_s1_s2_overlap_fails_closed() -> None:
    with pytest.raises(ValueError, match="overlap"):
        select_boundary_cohort([_pool(mixed=180, core=120), _pool(mixed=180, core=120)])


def test_s1_s2_example_index_alias_overlap_fails_closed() -> None:
    second = _pool(offset=600, mixed=180, core=120)
    alias_id = next(iter(second["tasks"]))
    second["tasks"][alias_id]["example_index"] = 0
    second["groups"][alias_id]["example_index"] = 0
    with pytest.raises(ValueError, match="overlap at example_index"):
        select_boundary_cohort([_pool(mixed=180, core=120), second])


def test_combined_s1_s2_must_still_meet_frozen_thresholds() -> None:
    with pytest.raises(ValueError, match="boundary readiness failed"):
        select_boundary_cohort(
            [_pool(mixed=180, core=120), _pool(offset=600, mixed=179, core=120)]
        )


def test_s1_s2_can_cross_the_frozen_thresholds_exactly() -> None:
    selected, train, validation, details = select_boundary_cohort(
        [_pool(mixed=180, core=120), _pool(offset=600, mixed=180, core=120)]
    )
    assert (len(selected), len(train), len(validation)) == (332, 300, 32)
    assert details["screen_stage"] == "S1+S2"
    assert details["eligible_mixed_groups"] == 360
    assert details["eligible_core_groups"] == 240


def test_s2_is_forbidden_when_s1_already_meets_thresholds() -> None:
    with pytest.raises(ValueError, match="S2 is forbidden"):
        select_boundary_cohort([_pool(), _pool(offset=600)])


def _write_pool(tmp_path: Path, pool: dict, name: str) -> tuple[Path, Path]:
    tasks_path = tmp_path / f"{name}.jsonl"
    audit_path = tmp_path / f"{name}.audit.json"
    task_rows = list(pool["tasks"].values())
    task_bytes = "".join(
        json.dumps(row, separators=(",", ":")) + "\n" for row in task_rows
    ).encode()
    tasks_path.write_bytes(task_bytes)
    audit = {
        "schema_version": "vanilla-grpo-boundary-screen-audit-v1",
        "contract": {
            "tasks": 600,
            "group_size": 8,
            "optimizer_updates": 0,
            "initial_adapter_sha256": "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5",
            "reward_mode": "result-only",
            "result_reward_profile": "binary",
        },
        "inputs": {
            "tasks": str(tasks_path.resolve()),
            "tasks_sha256": __import__("hashlib").sha256(task_bytes).hexdigest(),
            "manifest_sha256": "a" * 64,
            "trajectories_sha256": "b" * 64,
        },
        "groups": list(pool["groups"].values()),
        "issues": [],
        "issue_counts": {},
        "status": {"audit_passes": True, "pool_admitted": True},
    }
    audit_path.write_text(json.dumps(audit))
    return audit_path, tasks_path


def test_cli_writes_frozen_output_names_and_manifest(tmp_path: Path) -> None:
    audit, tasks = _write_pool(tmp_path, _pool(), "s1")
    output = tmp_path / "selected"
    assert main(
        [
            "--screen-audit",
            str(audit),
            "--tasks",
            str(tasks),
            "--output-dir",
            str(output),
        ]
    ) == 0
    manifest = json.loads((output / "boundary_cohort_manifest.json").read_text())
    assert (output / "boundary332.jsonl").exists()
    assert (output / "train300.jsonl").exists()
    assert (output / "validation32.jsonl").exists()
    assert manifest["contract"]["primary_checkpoint"] == "final-step20-only"
    assert manifest["contract"]["formal_training"]["fresh_online_trajectories"] == 4800
    assert manifest["contract"]["formal_training"]["sampler"] == "trl-0.29-repeat-sampler-v1"
    assert manifest["contract"]["formal_training"]["shuffle_dataset"] is True
    assert manifest["contract"]["formal_training"]["data_seed"] == 20260812
    assert manifest["contract"]["formal_training"]["per_pass_coverage"] == "each train300 identity exactly once"
    assert manifest["contract"]["validation"]["seed"] == 20260813
    assert manifest["contract"]["validation"]["screen_seed_must_differ"] == 20260812
    assert manifest["contract"]["validation"]["runtime_contamination_allowed"] is False
    assert manifest["contract"]["validation"]["screen_trajectories_reused"] is False
    assert manifest["outputs"]["train"]["records"] == 300
    assert manifest["outputs"]["validation"]["records"] == 32


def test_loader_rejects_task_digest_mismatch(tmp_path: Path) -> None:
    audit, tasks = _write_pool(tmp_path, _pool(), "s1")
    tasks.write_text(tasks.read_text() + "{}\n")
    with pytest.raises(ValueError, match="binding mismatch"):
        _load_pool(audit, tasks)


def test_loader_rejects_internally_inconsistent_group_summary(
    tmp_path: Path,
) -> None:
    audit, tasks = _write_pool(tmp_path, _pool(), "s1")
    payload = json.loads(audit.read_text())
    payload["groups"][0]["mixed_boundary"] = False
    audit.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid group summary"):
        _load_pool(audit, tasks)
