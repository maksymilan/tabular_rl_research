from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tool_modules.checkpoint_relalg.merge_sft_candidates import (
    _expected_identity,
    _file_sha256,
    _sha256_value,
    merge_candidate_lanes,
)
from tool_modules.checkpoint_relalg.sft_export import TRAINING_RECORD_VERSION


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _candidate(episode: str, turn: int, tool: str, *, prefix: str) -> tuple[dict, dict]:
    assistant = {
        "role": "assistant",
        "reasoning_content": f"reason-{prefix}-{episode}-{turn}",
        "content": json.dumps({"tool": tool, "arguments": {}}),
    }
    messages = [
        {"role": "system", "content": "student"},
        {"role": "user", "content": "state"},
        assistant,
    ]
    metadata = {
        "record_id": f"{prefix}_{episode}_{turn}",
        "source_episode_id": episode,
        "source_task_position": turn,
        "source_turn_index": turn,
        "tool_name": tool,
        "reasoning_sha256": hashlib.sha256(
            assistant["reasoning_content"].encode("utf-8")
        ).hexdigest(),
    }
    record = {
        "schema_version": TRAINING_RECORD_VERSION,
        "messages": messages,
        "loss_message_index": 2,
        "metadata": metadata,
    }
    index = {
        **metadata,
        "model_input_sha256": _sha256_value(messages[:-1]),
        "target_envelope_sha256": _sha256_value(assistant),
        "target_visible_content_sha256": hashlib.sha256(
            assistant["content"].encode("utf-8")
        ).hexdigest(),
    }
    return record, index


def _lane(tmp_path: Path, name: str, episodes: list[str]) -> tuple[Path, Path, Path]:
    records: list[dict] = []
    indexes: list[dict] = []
    for episode in episodes:
        record, index = _candidate(episode, 0, "answer", prefix=name)
        records.append(record)
        indexes.append(index)
    records_path = tmp_path / f"{name}.jsonl"
    index_path = tmp_path / f"{name}_index.jsonl"
    manifest_path = tmp_path / f"{name}.manifest.json"
    _write_jsonl(records_path, records)
    _write_jsonl(index_path, indexes)
    manifest = {
        "schema_version": f"{name}-manifest-v1",
        **_expected_identity(),
        "admitted_episodes": len(episodes),
        "records": len(records),
        "output_sha256": _file_sha256(records_path),
        "index_sha256": _file_sha256(index_path),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return records_path, index_path, manifest_path


def test_primary_episode_wins_and_fallback_only_fills_missing(tmp_path: Path) -> None:
    primary = _lane(tmp_path, "primary", ["shared", "primary_only"])
    fallback = _lane(tmp_path, "fallback", ["shared", "fallback_only"])
    out = tmp_path / "union.jsonl"
    out_index = tmp_path / "union_index.jsonl"
    manifest = merge_candidate_lanes(
        primary_records_path=primary[0],
        primary_index_path=primary[1],
        primary_manifest_path=primary[2],
        fallback_records_path=fallback[0],
        fallback_index_path=fallback[1],
        fallback_manifest_path=fallback[2],
        out_path=out,
        out_index_path=out_index,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    by_episode = {row["metadata"]["source_episode_id"]: row for row in rows}
    assert set(by_episode) == {"shared", "primary_only", "fallback_only"}
    assert by_episode["shared"]["metadata"]["record_id"].startswith("primary_")
    assert manifest["overlap_episodes"] == 1
    assert manifest["selected_primary_episodes"] == 2
    assert manifest["selected_fallback_episodes"] == 1
    assert manifest["episodes"] == 3


def test_rejects_tampered_target_envelope(tmp_path: Path) -> None:
    primary = _lane(tmp_path, "primary", ["episode"])
    fallback = _lane(tmp_path, "fallback", ["other"])
    rows = [json.loads(line) for line in primary[0].read_text().splitlines()]
    rows[0]["messages"][-1]["reasoning_content"] = "tampered"
    _write_jsonl(primary[0], rows)
    manifest = json.loads(primary[2].read_text())
    manifest["output_sha256"] = _file_sha256(primary[0])
    primary[2].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="target envelope hash mismatch"):
        merge_candidate_lanes(
            primary_records_path=primary[0],
            primary_index_path=primary[1],
            primary_manifest_path=primary[2],
            fallback_records_path=fallback[0],
            fallback_index_path=fallback[1],
            fallback_manifest_path=fallback[2],
            out_path=tmp_path / "union.jsonl",
            out_index_path=tmp_path / "union_index.jsonl",
        )
