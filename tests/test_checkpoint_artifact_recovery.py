from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from artifacts import ArtifactWriter


MANIFEST = {"runner": "synthetic-recovery-test", "requested_size": 2}


def _record(example_index: int, *, correct: bool = True) -> dict:
    return {
        "example_index": example_index,
        "correct": correct,
        "failure_type": None if correct else "synthetic_failure",
    }


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_resume_repairs_only_torn_all_jsonl_tail_and_records_hash_evidence(
    tmp_path: Path,
):
    result_dir = tmp_path / "torn-journal"
    first_manifest = {**MANIFEST, "run_started_at_utc": "2026-08-09T00:00:00+00:00"}
    writer = ArtifactWriter(str(result_dir), first_manifest, resume=False)
    writer.append(_record(1))
    torn_tail = b'{"example_index":2,"correct":'
    with (result_dir / "all.jsonl").open("ab") as handle:
        handle.write(torn_tail)

    resumed = ArtifactWriter(
        str(result_dir),
        {**first_manifest, "run_started_at_utc": "2026-08-09T00:05:00+00:00"},
        resume=True,
        operational_resume_fields={"run_started_at_utc"},
    )

    assert resumed.completed == {1}
    assert [record["example_index"] for record in _jsonl(result_dir / "all.jsonl")] == [1]
    event = _jsonl(result_dir / "artifact_recovery_events.jsonl")[0]
    assert event["event"] == "truncate_torn_all_jsonl_tail"
    assert event["bad_tail_bytes"] == len(torn_tail)
    assert event["bad_tail_sha256"] == hashlib.sha256(torn_tail).hexdigest()
    assert "content" not in event and "reasoning_content" not in event
    operational_event = _jsonl(result_dir / "operational_resume_events.jsonl")[0]
    assert operational_event["completed_before_resume"] == 1


def test_resume_fails_closed_on_malformed_middle_or_complete_conflict(tmp_path: Path):
    result_dir = tmp_path / "bad-middle"
    writer = ArtifactWriter(str(result_dir), dict(MANIFEST), resume=False)
    writer.append(_record(1))
    writer.append(_record(2, correct=False))
    all_path = result_dir / "all.jsonl"
    lines = all_path.read_bytes().splitlines(keepends=True)
    all_path.write_bytes(lines[0] + b"{not-json}\n" + lines[1])
    with pytest.raises(ValueError, match="malformed complete all.jsonl"):
        ArtifactWriter(str(result_dir), dict(MANIFEST), resume=True)

    conflict_dir = tmp_path / "complete-conflict"
    writer = ArtifactWriter(str(conflict_dir), dict(MANIFEST), resume=False)
    writer.append(_record(1))
    committed = _jsonl(conflict_dir / "all.jsonl")[0]
    conflicting = {**committed, "correct": False}
    with (conflict_dir / "all.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(conflicting) + "\n")
    with pytest.raises(ValueError, match="conflicting complete all.jsonl"):
        ArtifactWriter(str(conflict_dir), dict(MANIFEST), resume=True)


def test_resume_rebuilds_missing_split_tail_and_case_from_all_journal(tmp_path: Path):
    result_dir = tmp_path / "projection-rebuild"
    writer = ArtifactWriter(str(result_dir), dict(MANIFEST), resume=False)
    writer.append(_record(1))
    writer.append(_record(2))
    success_path = result_dir / "success.jsonl"
    success_lines = success_path.read_bytes().splitlines(keepends=True)
    success_path.write_bytes(success_lines[0])
    (result_dir / "success_cases" / "q0002.json").unlink()

    resumed = ArtifactWriter(str(result_dir), dict(MANIFEST), resume=True)

    assert resumed.completed == {1, 2}
    assert _jsonl(success_path) == _jsonl(result_dir / "all.jsonl")
    assert json.loads(
        (result_dir / "success_cases" / "q0002.json").read_text()
    )["example_index"] == 2
    events = _jsonl(result_dir / "artifact_recovery_events.jsonl")
    rebuild = events[-1]
    assert rebuild["event"] == "rebuild_derived_artifacts_from_all_jsonl"
    assert rebuild["record_count"] == 2
    assert rebuild["sft_eligible"] is False
    assert any("success.jsonl" in reason for reason in rebuild["reasons"])
    assert any("q0002.json" in reason for reason in rebuild["reasons"])


def test_resume_rejects_complete_projection_conflict(tmp_path: Path):
    result_dir = tmp_path / "projection-conflict"
    writer = ArtifactWriter(str(result_dir), dict(MANIFEST), resume=False)
    writer.append(_record(1))
    success_path = result_dir / "success.jsonl"
    conflicting = _jsonl(success_path)[0]
    conflicting["failure_type"] = "tampered"
    success_path.write_text(json.dumps(conflicting) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="conflicts with authoritative all.jsonl"):
        ArtifactWriter(str(result_dir), dict(MANIFEST), resume=True)


def test_append_commits_journal_and_publishes_case_atomically(tmp_path: Path):
    result_dir = tmp_path / "atomic-case"
    writer = ArtifactWriter(str(result_dir), dict(MANIFEST), resume=False)
    writer.append(_record(7, correct=False))

    assert _jsonl(result_dir / "all.jsonl") == _jsonl(result_dir / "failure.jsonl")
    assert json.loads(
        (result_dir / "failure_cases" / "q0007.json").read_text()
    ) == _jsonl(result_dir / "all.jsonl")[0]
    assert not list(result_dir.rglob(".*.tmp-*"))


def test_existing_artifacts_without_manifest_fail_closed(tmp_path: Path):
    result_dir = tmp_path / "unclaimed-artifacts"
    result_dir.mkdir()
    (result_dir / "all.jsonl").write_text(
        json.dumps(_record(1)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="without a manifest"):
        ArtifactWriter(str(result_dir), dict(MANIFEST), resume=True)
    assert not (result_dir / "manifest.json").exists()
