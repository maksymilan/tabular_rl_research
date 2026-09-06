from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


RL_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl.scenarios.data.prepare_policy_boundary_s2_tasks import (  # noqa: E402
    CANONICAL_INPUT_COUNTS,
    CANONICAL_INPUT_SHA256,
    DEFAULT_SEED,
    EXPECTED_INITIAL_ADAPTER_SHA256,
    EXPECTED_PROTOCOL_HASH,
    EXPECTED_PROTOCOL_VERSION,
    SCHEMA_VERSION,
    prepare_s2_pool,
)
from rl.scenarios.data.select_vanilla_grpo_tasks import exclusion_ids  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task(index: int) -> dict:
    db_id = f"db_{index % 5}"
    return {
        "example_id": f"bird_train_{index:05d}",
        "instance_id": f"bird_train_{index:05d}",
        "example_index": index,
        "db_id": db_id,
        "db_path": f"/local/train_databases/{db_id}/{db_id}.sqlite",
        "question": "q" * (20 + index % 10),
        "external_knowledge": "hint" if index % 2 else None,
        "gold_sql": f"select {index}",
        "query": f"select {index}",
        "gold_exec_results": [[index]],
        "metadata": {"tool_round_trip": "verified"},
    }


def _write_fixture(tmp_path: Path) -> dict:
    reference_rows = [_task(index) for index in range(1000)]
    eligible_rows = copy.deepcopy(reference_rows)
    s1_rows = copy.deepcopy(reference_rows[:600])
    baseline_rows = copy.deepcopy(reference_rows[600:630])
    sft_ids = [row["example_id"] for row in reference_rows[630:670]]
    sft_rows = [
        {
            "source_episode_id": sft_ids[position % len(sft_ids)],
            "source_step_id": f"step_{position}",
        }
        for position in range(100)
    ]
    mixed_rows = copy.deepcopy(reference_rows[670:690])

    paths = {
        "reference": tmp_path / "reference.jsonl",
        "eligible": tmp_path / "eligible.jsonl",
        "s1_tasks": tmp_path / "s1_tasks.jsonl",
        "s1_cohort_manifest": tmp_path / "s1_cohort_manifest.json",
        "baseline_eval300": tmp_path / "baseline.jsonl",
        "sft1_index": tmp_path / "sft.index.jsonl",
        "old_mixed60": tmp_path / "mixed.jsonl",
    }
    for role, rows in (
        ("reference", reference_rows),
        ("eligible", eligible_rows),
        ("s1_tasks", s1_rows),
        ("baseline_eval300", baseline_rows),
        ("sft1_index", sft_rows),
        ("old_mixed60", mixed_rows),
    ):
        _write_jsonl(paths[role], rows)

    expected_hashes = {
        role: _sha(path)
        for role, path in paths.items()
        if role != "s1_cohort_manifest"
    }
    exclusions = []
    for role in ("baseline_eval300", "sft1_index", "old_mixed60"):
        ids, artifacts = exclusion_ids([paths[role]])
        artifact = artifacts[0]
        assert artifact["unique_task_ids"] == len(ids)
        exclusions.append(artifact)
    s1_manifest = {
        "schema_version": "bird-train-vanilla-grpo-cohort-v1",
        "status": "frozen_training_cohort",
        "all_acceptance_gates_passed": True,
        "reference_population": {
            "sha256": expected_hashes["reference"],
            "records": len(reference_rows),
        },
        "eligibility_population": {
            "sha256": expected_hashes["eligible"],
            "records": len(eligible_rows),
        },
        "exclusions": exclusions,
        "output": {
            "sha256": expected_hashes["s1_tasks"],
            "records": 600,
            "task_ids_in_frozen_order": [row["example_id"] for row in s1_rows],
        },
    }
    paths["s1_cohort_manifest"].write_text(json.dumps(s1_manifest))
    expected_hashes["s1_cohort_manifest"] = _sha(paths["s1_cohort_manifest"])

    pending = tmp_path / "s1_manifest.pending.json"
    trajectories = tmp_path / "s1_trajectories.jsonl"
    pending.write_text('{"frozen":true}\n')
    trajectories.write_text('{"frozen":true}\n')
    s1_audit = tmp_path / "s1_boundary_screen_audit.json"
    audit = {
        "schema_version": "vanilla-grpo-boundary-screen-audit-v1",
        "contract": {
            "tasks": 600,
            "group_size": 8,
            "optimizer_updates": 0,
            "protocol_version": EXPECTED_PROTOCOL_VERSION,
            "protocol_hash": EXPECTED_PROTOCOL_HASH,
            "initial_adapter_sha256": EXPECTED_INITIAL_ADAPTER_SHA256,
            "reward_mode": "result-only",
            "result_reward_profile": "binary",
            "screen_seed": 20260812,
            "generation_seed_scheme": "sha256-task-sample-turn-v1",
            "minimum_mixed_groups": 360,
            "minimum_core_groups": 240,
        },
        "inputs": {
            "manifest": str(pending.resolve()),
            "manifest_sha256": _sha(pending),
            "trajectories": str(trajectories.resolve()),
            "trajectories_sha256": _sha(trajectories),
            "tasks": str(paths["s1_tasks"].resolve()),
            "tasks_sha256": expected_hashes["s1_tasks"],
        },
        "groups": [
            {
                "task_id": row["example_id"],
                "example_index": row["example_index"],
                "trajectories": 8,
                "usable": True,
                "contamination": [],
                "correct_count": 4 if position < 200 else 8,
                "mixed_boundary": position < 200,
                "core_boundary": position < 200,
            }
            for position, row in enumerate(s1_rows)
        ],
        "observed": {
            "tasks": 600,
            "trajectories": 4800,
            "mixed_boundary_groups": 200,
            "core_boundary_groups": 200,
        },
        "issues": [],
        "issue_counts": {},
        "status": {
            "audit_passes": True,
            "pool_admitted": True,
            "selection_ready_without_s2": False,
            "next_stage": "screen_s2",
        },
    }
    s1_audit.write_text(json.dumps(audit))
    expected_counts = {
        "reference": (1000, 1000),
        "eligible": (1000, 1000),
        "s1_tasks": (600, 600),
        "baseline_eval300": (30, 30),
        "sft1_index": (100, 40),
        "old_mixed60": (20, 20),
    }
    return {
        "paths": paths,
        "s1_audit": s1_audit,
        "expected_s1_audit_sha256": _sha(s1_audit),
        "expected_hashes": expected_hashes,
        "expected_counts": expected_counts,
        "output": tmp_path / "s2_extra.jsonl",
        "manifest": tmp_path / "s2_extra.manifest.json",
        "remote_db_root": Path("/remote/bird/train/train_databases"),
    }


def _prepare(fixture: dict, **overrides) -> dict:
    paths = fixture["paths"]
    kwargs = {
        "s1_audit_path": fixture["s1_audit"],
        "expected_s1_audit_sha256": fixture["expected_s1_audit_sha256"],
        "reference_path": paths["reference"],
        "eligible_path": paths["eligible"],
        "s1_tasks_path": paths["s1_tasks"],
        "s1_cohort_manifest_path": paths["s1_cohort_manifest"],
        "baseline_eval300_path": paths["baseline_eval300"],
        "sft1_index_path": paths["sft1_index"],
        "old_mixed60_path": paths["old_mixed60"],
        "output_path": fixture["output"],
        "manifest_path": fixture["manifest"],
        "remote_db_root": fixture["remote_db_root"],
        "seed": "synthetic-s2-seed",
        "count": 200,
        "expected_hashes": fixture["expected_hashes"],
        "expected_counts": fixture["expected_counts"],
    }
    kwargs.update(overrides)
    return prepare_s2_pool(**kwargs)


def test_requires_s2_freezes_disjoint_public_field_pool_and_identity_manifest(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    manifest = _prepare(fixture)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["status"] == "frozen_s2_screening_cohort"
    assert manifest["activation_gate"]["decision"] == "requires_s2"
    assert manifest["selection"]["gold_used_for_selection_or_order"] is False
    assert manifest["selection"]["public_fields_used"] == [
        "example_id", "db_id", "question", "external_knowledge"
    ]
    assert manifest["output"]["records"] == 200
    assert all(manifest["acceptance_gates"].values())
    selected = set(manifest["output"]["task_ids_in_frozen_order"])
    for role in ("s1_tasks", "baseline_eval300", "sft1_index", "old_mixed60"):
        ids, _ = exclusion_ids([fixture["paths"][role]])
        assert selected.isdisjoint(ids)
    output_rows = [
        json.loads(line) for line in fixture["output"].read_text().splitlines()
    ]
    assert len(output_rows) == 200
    assert all(
        row["example_id"] == f"bird_train_{row['example_index']:05d}"
        and row["instance_id"] == row["example_id"]
        for row in output_rows
    )
    assert all(
        row["metadata"]["rl_training_cohort"] == SCHEMA_VERSION
        and row["metadata"]["policy_boundary_screen_stage"] == "S2"
        and row["db_path"].startswith(str(fixture["remote_db_root"]))
        for row in output_rows
    )
    assert _sha(fixture["output"]) == manifest["output"]["sha256"]
    assert json.loads(fixture["manifest"].read_text()) == manifest


def test_ready_s1_audit_forbids_s2_and_writes_nothing(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    audit = json.loads(fixture["s1_audit"].read_text())
    audit["status"]["selection_ready_without_s2"] = True
    audit["status"]["next_stage"] = "select_s1"
    fixture["s1_audit"].write_text(json.dumps(audit))
    fixture["expected_s1_audit_sha256"] = _sha(fixture["s1_audit"])
    with pytest.raises(ValueError, match="S2 is forbidden"):
        _prepare(fixture)
    assert not fixture["output"].exists()
    assert not fixture["manifest"].exists()


def test_wrong_audit_or_universe_hash_fails_before_output(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _prepare(fixture, expected_s1_audit_sha256="0" * 64)
    assert not fixture["output"].exists()

    fixture["paths"]["eligible"].write_text(
        fixture["paths"]["eligible"].read_text() + "{}\n"
    )
    with pytest.raises(ValueError, match="eligible: SHA-256 mismatch"):
        _prepare(fixture)
    assert not fixture["output"].exists()


def test_task_example_alias_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in fixture["paths"]["baseline_eval300"].read_text().splitlines()
    ]
    rows[0]["example_index"] += 1
    _write_jsonl(fixture["paths"]["baseline_eval300"], rows)
    fixture["expected_hashes"] = dict(fixture["expected_hashes"])
    fixture["expected_hashes"]["baseline_eval300"] = _sha(
        fixture["paths"]["baseline_eval300"]
    )
    with pytest.raises(ValueError, match="task/example identity mismatch"):
        _prepare(fixture)
    assert not fixture["output"].exists()


def test_production_contract_pins_all_known_inputs_and_cli_is_explicit() -> None:
    assert CANONICAL_INPUT_SHA256 == {
        "reference": "24d21a349c430df549f92724dd07de1c59b2365cd72802957120afba13c7a2e1",
        "eligible": "c8ea3c6419ac57f05021eca470c7519843456e8ac02b04cca9af39e779ec7545",
        "s1_tasks": "b5a83c373e9be094ea7355c0bc23c9212249491491f44dfdcb3b09ea49b2457e",
        "s1_cohort_manifest": "9212d1f7ca4fb63576eb7e846a9b05f7d159e131fe992495b99e350b6aae8dd6",
        "baseline_eval300": "87b37b307ea2710acdff7d03bdc474a6ba2cdb8ed51b005cff0fe8fa54c67c03",
        "sft1_index": "aa6b8a72a1cfbcf44e37702e0498c9a85f615edc201ba3f4172b65f701e0cda5",
        "old_mixed60": "ef97b6977735996fde505aaad4e00215570963b37a1c82d890c34c4b414f30ae",
    }
    assert CANONICAL_INPUT_COUNTS["sft1_index"] == (4471, 678)
    assert DEFAULT_SEED == "qwen3-v26-policy-boundary-s2-extra600-v1-20260812"
    script = RL_DIR / "scenarios" / "data" / "prepare_policy_boundary_s2_tasks.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for option in (
        "--s1-audit",
        "--expected-s1-audit-sha256",
        "--reference",
        "--eligible",
        "--s1-tasks",
        "--s1-cohort-manifest",
        "--baseline-eval300",
        "--sft1-index",
        "--old-mixed60",
        "--output",
        "--manifest",
        "--remote-db-root",
    ):
        assert option in result.stdout
