from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.rl.evaluation import audit_earlystop_missing32_behavior_gate as audit
from src.rl.evaluation import prepare_earlystop_missing32_gate as prepare
from src.rl.evaluation import preserve_diagnostic_checkpoint as preserve


def _source_rows() -> list[dict]:
    return [
        {
            "example_id": f"bird_train_{index:05d}",
            "instance_id": f"bird_train_{index:05d}",
            "example_index": index,
            "db_id": f"db_{index % 3}",
            "db_path": f"/db/db_{index % 3}.sqlite",
            "question": f"question {index}",
            "gold_sql": f"select {index}",
            "external_knowledge": None,
        }
        for index in range(600)
    ]


def _earlystop(screened: set[str]) -> dict:
    return {
        "schema_version": "qwen3-v26-earlystop-mixed-grpo-cohort-v1",
        "status": "frozen_operator_requested_earlystop_mixed180",
        "operator_decision": {
            "completed_groups": 568,
            "missing_groups": 32,
            "stopped_before_full_600": True,
        },
        "inputs": {
            "tasks": {
                "sha256": prepare.EXPECTED_SOURCE_TASKS_SHA256,
                "records": 600,
            }
        },
        "screen": {"group_sha256": {identity: "a" * 64 for identity in screened}},
        # Poisoned reward-derived fields prove derive() ignores their contents.
        "selection": {"selected_task_ids": ["SHOULD_NOT_BE_READ"]},
    }


def test_reward_blind_ordered_complement(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _source_rows()
    missing_positions = {7, 33, *range(570, 600)}
    screened = {
        prepare.task_id(row) for index, row in enumerate(rows) if index not in missing_positions
    }
    monkeypatch.setattr(prepare, "EXPECTED_SOURCE_TASKS_SHA256", "1" * 64)
    monkeypatch.setattr(prepare, "EXPECTED_EARLYSTOP_MANIFEST_SHA256", "2" * 64)
    output, manifest = prepare.derive(
        rows,
        _earlystop(screened),
        source_tasks_sha256="1" * 64,
        earlystop_manifest_sha256="2" * 64,
    )
    expected = [prepare.task_id(rows[index]) for index in sorted(missing_positions)]
    assert [prepare.task_id(row) for row in output] == expected
    assert manifest["selection"]["ordered_task_ids"] == expected
    assert manifest["selection"]["reward_blind"] is True
    assert manifest["selection"]["reward_or_correctness_fields_read"] == 0


def test_checkpoint4_cohort_has_independent_seed_and_explicit_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _source_rows()
    screened = {prepare.task_id(row) for row in rows[:568]}
    monkeypatch.setattr(prepare, "EXPECTED_SOURCE_TASKS_SHA256", "1" * 64)
    monkeypatch.setattr(prepare, "EXPECTED_EARLYSTOP_MANIFEST_SHA256", "2" * 64)
    _, manifest = prepare.derive(
        rows,
        _earlystop(screened),
        source_tasks_sha256="1" * 64,
        earlystop_manifest_sha256="2" * 64,
        evaluation_arm="checkpoint-4",
        evaluation_seed=20260816,
    )
    assert manifest["evaluation_contract"]["arms"] == ["sft1", "checkpoint-4"]
    assert manifest["evaluation_contract"]["seed"] == 20260816
    with pytest.raises(ValueError, match="checkpoint-4 evaluation seed"):
        prepare.derive(
            rows,
            _earlystop(screened),
            source_tasks_sha256="1" * 64,
            earlystop_manifest_sha256="2" * 64,
            evaluation_arm="checkpoint-4",
            evaluation_seed=20260817,
        )


def test_checkpoint4_sibling_is_canonical_and_rejects_cp6_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _source_rows()[:32]
    tasks_bytes = prepare.canonical_jsonl(rows)
    source_manifest = {
        "schema_version": prepare.SCHEMA_VERSION,
        "status": prepare.STATUS,
        "scope": {
            "dataset_split": "BIRD-train",
            "diagnostic_only": True,
            "formal_checkpoint_selection_authority": False,
            "formal_dev_consumed": False,
        },
        "selection": {
            "ordered_task_ids": [prepare.task_id(row) for row in rows],
        },
        "inputs": {},
        "output": {
            "path": "/frozen/cp6/missing32.jsonl",
            "records": 32,
            "sha256": hashlib.sha256(tasks_bytes).hexdigest(),
        },
        "evaluation_contract": {
            "arms": ["sft1", "checkpoint-6"],
            "seed": 20260817,
        },
    }
    source_manifest_bytes = (
        json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    monkeypatch.setattr(
        prepare, "EXPECTED_MISSING32_TASKS_SHA256", hashlib.sha256(tasks_bytes).hexdigest()
    )
    monkeypatch.setattr(
        prepare,
        "EXPECTED_CHECKPOINT6_COHORT_MANIFEST_SHA256",
        hashlib.sha256(source_manifest_bytes).hexdigest(),
    )
    source = tmp_path / "cohort"
    source.mkdir()
    (source / "missing32.jsonl").write_bytes(tasks_bytes)
    (source / "missing32_manifest.json").write_bytes(source_manifest_bytes)
    target = tmp_path / "cohort_checkpoint4_seed20260816"
    remote_tasks = Path(
        "/home/dengyan/tabular_rl_outputs/evaluations/"
        "qwen3_8b_atomic_v26_earlystop_missing32_cp4_20260813/"
        "cohort_checkpoint4_seed20260816/missing32.jsonl"
    )
    prepare.publish_checkpoint4_sibling(
        source, target, manifest_tasks_path=remote_tasks
    )
    manifest = prepare.verify_checkpoint4_sibling(
        source, target, manifest_tasks_path=remote_tasks
    )
    assert (target / "missing32.jsonl").read_bytes() == tasks_bytes
    assert manifest["evaluation_contract"] == {
        "arms": ["sft1", "checkpoint-4"],
        "seed": 20260816,
    }
    assert manifest["output"]["path"] == str(remote_tasks)
    with pytest.raises(ValueError, match="canonical contract"):
        prepare.verify_checkpoint4_sibling(
            source, source, manifest_tasks_path=remote_tasks
        )


def _write_checkpoint(root: Path, step: int = 6, max_steps: int = 12) -> None:
    root.mkdir()
    for name in preserve.REQUIRED_FILES + preserve.OPTIONAL_STATE_FILES:
        if name == "trainer_state.json":
            (root / name).write_text(json.dumps({"global_step": step, "max_steps": max_steps}))
        else:
            (root / name).write_bytes((name + "\n").encode())


def test_checkpoint_preservation_is_verified_atomic_and_source_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-6"
    target = tmp_path / "diagnostic_checkpoint-6"
    _write_checkpoint(source)
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    receipt = preserve.preserve(
        source, target, expected_step=6, expected_max_steps=12,
        stable_seconds=0.001, poll_seconds=0.001, timeout_seconds=1,
    )
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    assert before == after
    assert receipt["source_mutated"] is False
    assert (target / "preservation_receipt.json").is_file()
    assert not list(tmp_path.glob(".diagnostic_checkpoint-6.next-*"))
    for name, record in receipt["files"].items():
        assert hashlib.sha256((target / name).read_bytes()).hexdigest() == record["sha256"]


def test_checkpoint_preservation_rejects_wrong_step(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-6"
    _write_checkpoint(source, step=5)
    with pytest.raises(TimeoutError):
        preserve.preserve(
            source, tmp_path / "out", expected_step=6, expected_max_steps=12,
            stable_seconds=0.001, poll_seconds=0.001, timeout_seconds=0.01,
        )


def test_cluster_bootstrap_and_gate_threshold_direction() -> None:
    assert audit.cluster_bootstrap_lower([0.125] * 32) == pytest.approx(0.125)
    assert audit.cluster_bootstrap_lower([-0.125] * 32) == pytest.approx(-0.125)


def test_checkpoint4_audit_schema_and_labels_cannot_masquerade_as_checkpoint6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = _source_rows()[:32]
    ordered = [prepare.task_id(row) for row in tasks]
    task_sha = "f" * 64
    cohort = {
        "schema_version": prepare.SCHEMA_VERSION,
        "status": prepare.STATUS,
        "inputs": {
            "source_tasks": {"sha256": audit.EXPECTED_SOURCE_TASKS_SHA256},
            "earlystop_manifest": {"sha256": audit.EXPECTED_EARLYSTOP_MANIFEST_SHA256},
        },
        "selection": {"ordered_task_ids": ordered},
        "output": {"sha256": task_sha},
        "evaluation_contract": {"arms": ["sft1", "checkpoint-4"], "seed": 20260816},
    }

    def fake_arm(*, arm: str, **_kwargs):
        correct = 4 if arm == "sft1" else 5
        return {
            "arm": arm,
            "adapter_sha256": "1" * 64 if arm == "sft1" else "2" * 64,
            "tasks": 32,
            "trajectories": 256,
            "runtime_clean_trajectories": 256,
            "correct": correct * 32,
            "legal": 256,
            "groups": {identity: {"correct": correct, "legal": 8} for identity in ordered},
        }

    monkeypatch.setattr(audit, "audit_arm", fake_arm)
    result = audit.audit_pair(
        cohort_manifest=cohort,
        tasks=tasks,
        tasks_sha256=task_sha,
        sft1_manifest={},
        sft1_rows=[],
        sft1_rows_sha256="a" * 64,
        checkpoint_manifest={},
        checkpoint_rows=[],
        checkpoint_rows_sha256="b" * 64,
        expected_checkpoint_adapter_sha256="2" * 64,
        checkpoint_arm="checkpoint-4",
    )
    assert result["schema_version"] == audit.SCHEMA_VERSIONS["checkpoint-4"]
    assert set(result["arms"]) == {"sft1", "checkpoint-4"}
    assert "checkpoint-6" not in json.dumps(result)
    assert result["contract"]["checkpoint_global_step"] == 4
    assert result["contract"]["seed"] == 20260816
    assert "checkpoint4_correct" in result["task_deltas"][0]


def test_launcher_defaults_to_dry_run_and_names_explicit_watch_mode() -> None:
    launcher = (
        Path(__file__).parent
        / "run_qwen3_8b_v26_earlystop_missing32_behavior_gate_table_rl.sh"
    ).read_text()
    assert "MODE=${1:-dry-run}" in launcher
    assert "watch-checkpoint6" in launcher
    assert "formal_dev_consumed=false" in launcher
    assert "--expected-step 6 --expected-max-steps 12" in launcher
    assert "--seed 20260817" in launcher
    assert "--temperature 0.8" in launcher
    assert "--max-new-tokens 2048" in launcher
    assert "--max-context-tokens 16384" in launcher


def test_newgnn_checkpoint4_launcher_is_independent_pinned_and_dry_by_default() -> None:
    launcher = (
        Path(__file__).parent
        / "run_qwen3_8b_v26_earlystop_missing32_checkpoint4_gate_newgnn.sh"
    ).read_text()
    assert "MODE=${1:-dry-run}" in launcher
    assert "GPU_ID=${GPU_ID:-6}" in launcher
    assert "--checkpoint-arm checkpoint-4" in launcher
    assert "--seed 20260816" in launcher
    assert "677746a38f9f16542b06ed4345e947207f5bd81c240efa4d56dd8fd95efdc2ee" in launcher
    assert "014e8ddc5cd6948510c9ef8dde0067915522b49d99e31472b75f4afaacde97bd" in launcher
    assert "6355325166bc4deb55852040146c90d6e958abe5a663abe262f5a08fd3a144a6" in launcher
    assert "COHORT=${COHORT_DIR:-$RUN_ROOT/cohort_checkpoint4_seed20260816}" in launcher
    assert "SOURCE_CP6_COHORT=${SOURCE_CP6_COHORT_DIR:-$RUN_ROOT/cohort}" in launcher
    assert "--checkpoint4-sibling" in launcher
    assert '--source-cohort-dir "$SOURCE_CP6_COHORT"' in launcher
    assert '--manifest-tasks-path "$TASKS" --verify' in launcher
    assert "european_football_1/european_football_1.sqlite" in launcher
    assert 'PYTHONPATH="$RUNTIME"' in launcher
    assert "cd /tmp" in launcher
    assert 'require_sha "$CP4/chat_template.jinja" "$EXPECTED_CHAT_TEMPLATE_SHA256"' in launcher
    assert "(root/'chat_template.jinja').read_bytes()" in launcher
    verify_cp4_body = launcher.split("verify_cp4() {", 1)[1].split("\n}\n", 1)[0]
    assert "json.loads((root/'tokenizer_config.json').read_text())['chat_template']" not in verify_cp4_body
    assert "TO_BE_PINNED" not in launcher
