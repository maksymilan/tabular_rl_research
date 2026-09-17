from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.data_selection.passk_artifact import export_snapshot


PROTOCOL_VERSION = "version26"
PROTOCOL_HASH = "4da19387399bd3a5"


def _task(identifier: str, index: int, question: str) -> dict:
    return {
        "example_id": identifier,
        "example_index": index,
        "db_id": "demo_db",
        "dataset": "bird-sql",
        "question": question,
    }


def _result(index: int, question: str, **overrides) -> dict:
    row = {
        "example_index": index,
        "db_id": "demo_db",
        "question": question,
        "n_samples": 8,
        "attempted_samples": 8,
        "sample_correct_count": 3,
        "sample_legal_count": 8,
        "failure_type": None,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": PROTOCOL_HASH,
        "assistant_carrier": "think-json-v1",
        "temperature": 0.8,
        "top_p": 1.0,
        "max_tokens": 2048,
        "max_steps": 30,
        "recorded_at_utc": "2026-09-16T00:00:00+00:00",
    }
    row.update(overrides)
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_snapshot_pairs_by_identity_and_sorts_by_task_id(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    result_path = tmp_path / "result.jsonl"
    _write_jsonl(
        input_path,
        [
            _task("bird_train_00002", 2, "second question"),
            _task("bird_train_00001", 1, "first question"),
        ],
    )
    # Raw result files are written in completion order, not input order.
    _write_jsonl(
        result_path,
        [
            _result(2, "second question", sample_correct_count=7),
            _result(1, "first question", sample_correct_count=0),
        ],
    )

    rows, manifest = export_snapshot(
        input_path=input_path,
        result_path=result_path,
        source_label="screen-batch",
        protocol_version=PROTOCOL_VERSION,
        protocol_hash=PROTOCOL_HASH,
    )

    assert [row["task_id"] for row in rows] == ["bird_train_00001", "bird_train_00002"]
    assert [row["sample_correct_count"] for row in rows] == [0, 7]
    assert rows[0]["source"] == "screen-batch"
    assert rows[0]["question_sha256"]
    assert manifest["records"] == 2
    assert manifest["correct_count_histogram"] == {"0": 1, "7": 1}
    assert manifest["legal_count_histogram"] == {"8": 2}
    assert manifest["incomplete_screen_rows"] == 0
    assert manifest["input"]["sha256"]
    assert manifest["result"]["sha256"]


def test_missing_result_row_is_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    result_path = tmp_path / "result.jsonl"
    _write_jsonl(
        input_path,
        [_task("bird_train_00001", 1, "first question"), _task("bird_train_00002", 2, "second question")],
    )
    _write_jsonl(result_path, [_result(1, "first question")])

    with pytest.raises(ValueError, match="1 input tasks have no result row"):
        export_snapshot(
            input_path=input_path,
            result_path=result_path,
            source_label="screen-batch",
        )


def test_incomplete_screen_is_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    result_path = tmp_path / "result.jsonl"
    _write_jsonl(input_path, [_task("bird_train_00001", 1, "first question")])
    _write_jsonl(result_path, [_result(1, "first question", attempted_samples=6)])

    with pytest.raises(ValueError, match="incomplete K=8 screen"):
        export_snapshot(
            input_path=input_path,
            result_path=result_path,
            source_label="screen-batch",
        )


def test_unmatched_result_row_is_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    result_path = tmp_path / "result.jsonl"
    _write_jsonl(input_path, [_task("bird_train_00001", 1, "first question")])
    _write_jsonl(result_path, [_result(9, "unrelated question")])

    with pytest.raises(ValueError, match="does not match any input task"):
        export_snapshot(
            input_path=input_path,
            result_path=result_path,
            source_label="screen-batch",
        )


def test_protocol_mismatch_is_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    result_path = tmp_path / "result.jsonl"
    _write_jsonl(input_path, [_task("bird_train_00001", 1, "first question")])
    _write_jsonl(result_path, [_result(1, "first question", protocol_hash="deadbeef")])

    with pytest.raises(ValueError, match="protocol_hash"):
        export_snapshot(
            input_path=input_path,
            result_path=result_path,
            source_label="screen-batch",
            protocol_hash=PROTOCOL_HASH,
        )
