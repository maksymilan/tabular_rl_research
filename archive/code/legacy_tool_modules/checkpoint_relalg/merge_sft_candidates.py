#!/usr/bin/env python3
"""Merge frozen-Atomic SFT candidate lanes with task-level source priority.

The official DeepSeek lane is primary.  A replay-verified Codex-manual lane may
fill only task identities for which the primary lane has no admitted episode.
No messages, reasoning, actions, or feedback are rewritten by this merger.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from tool_modules.checkpoint_relalg.sft_export import (
    ADMISSION_POLICY_VERSION,
    CARRIER,
    GUIDANCE,
    LOSS_POLICY,
    MODE,
    PROFILE,
    TRAINING_RECORD_VERSION,
)
from tool_modules.registry import build_checkpoint_relalg_tool_scheme


MERGE_POLICY_VERSION = "checkpoint-relalg-atomic-v24-frozen-task-priority-union-v1"


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _expected_identity() -> dict[str, Any]:
    scheme = build_checkpoint_relalg_tool_scheme(
        mode=MODE,
        carrier=CARRIER,
        atomic_operator_profile=PROFILE,
    )
    return {
        "admission_policy_version": ADMISSION_POLICY_VERSION,
        "training_record_version": TRAINING_RECORD_VERSION,
        "loss_policy": LOSS_POLICY,
        "tool_scheme": "checkpoint-relalg",
        "mode": MODE,
        "atomic_operator_profile": PROFILE,
        "carrier": CARRIER,
        "checkpoint_guidance_profile": GUIDANCE,
        "protocol_hash": scheme.protocol_hash,
        "student_prompt_sha256": scheme.student_prompt_hash,
        "tool_schema_sha256": scheme.tool_schema_hash,
    }


def _validate_episode(rows: list[tuple[dict[str, Any], dict[str, Any]]], episode: str) -> None:
    ordered = sorted(rows, key=lambda pair: pair[1]["source_turn_index"])
    turn_indexes = [pair[1]["source_turn_index"] for pair in ordered]
    if turn_indexes != sorted(set(turn_indexes)):
        raise ValueError(f"{episode}: duplicate or unordered source turn indexes")
    answer_positions = [
        position for position, (_, index) in enumerate(ordered) if index["tool_name"] == "answer"
    ]
    if answer_positions != [len(ordered) - 1]:
        raise ValueError(f"{episode}: expected one final answer target")


def _load_lane(
    records_path: Path,
    index_path: Path,
    manifest_path: Path,
    *,
    lane: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records_path = records_path.resolve()
    index_path = index_path.resolve()
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field, expected in _expected_identity().items():
        if manifest.get(field) != expected:
            raise ValueError(f"{lane}: manifest {field} differs from current frozen identity")
    if manifest.get("output_sha256") != _file_sha256(records_path):
        raise ValueError(f"{lane}: candidate file hash differs from manifest")
    if manifest.get("index_sha256") != _file_sha256(index_path):
        raise ValueError(f"{lane}: index file hash differs from manifest")
    records = _load_jsonl(records_path)
    indexes = _load_jsonl(index_path)
    if len(records) != len(indexes) or manifest.get("records") != len(records):
        raise ValueError(f"{lane}: candidate/index/manifest row counts differ")
    episodes: collections.defaultdict[
        str, list[tuple[dict[str, Any], dict[str, Any]]]
    ] = collections.defaultdict(list)
    record_ids: set[str] = set()
    for row_number, (record, index) in enumerate(zip(records, indexes), start=1):
        metadata = record.get("metadata")
        messages = record.get("messages")
        if record.get("schema_version") != TRAINING_RECORD_VERSION:
            raise ValueError(f"{lane}:{row_number}: training record version mismatch")
        if not isinstance(metadata, Mapping) or not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"{lane}:{row_number}: malformed candidate record")
        if record.get("loss_message_index") != len(messages) - 1:
            raise ValueError(f"{lane}:{row_number}: loss target is not the final assistant")
        assistant = messages[-1]
        if not isinstance(assistant, Mapping) or assistant.get("role") != "assistant":
            raise ValueError(f"{lane}:{row_number}: final message is not assistant")
        record_id = metadata.get("record_id")
        episode = metadata.get("source_episode_id")
        turn_index = metadata.get("source_turn_index")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError(f"{lane}:{row_number}: missing or duplicate record_id")
        if not isinstance(episode, str) or not episode:
            raise ValueError(f"{lane}:{row_number}: missing source_episode_id")
        if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
            raise ValueError(f"{lane}:{row_number}: invalid source_turn_index")
        record_ids.add(record_id)
        for key in ("record_id", "source_episode_id", "source_turn_index", "tool_name"):
            if index.get(key) != metadata.get(key):
                raise ValueError(f"{lane}:{row_number}: record/index {key} mismatch")
        model_input = messages[:-1]
        if index.get("model_input_sha256") != _sha256_value(model_input):
            raise ValueError(f"{lane}:{row_number}: model input hash mismatch")
        if index.get("target_envelope_sha256") != _sha256_value(assistant):
            raise ValueError(f"{lane}:{row_number}: target envelope hash mismatch")
        content = assistant.get("content")
        reasoning = assistant.get("reasoning_content")
        if not isinstance(content, str) or not isinstance(reasoning, str):
            raise ValueError(f"{lane}:{row_number}: assistant channels must be strings")
        if index.get("target_visible_content_sha256") != hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest():
            raise ValueError(f"{lane}:{row_number}: visible target hash mismatch")
        if metadata.get("reasoning_sha256") != hashlib.sha256(
            reasoning.encode("utf-8")
        ).hexdigest():
            raise ValueError(f"{lane}:{row_number}: reasoning hash mismatch")
        episodes[episode].append((record, index))
    for episode, episode_rows in episodes.items():
        _validate_episode(episode_rows, episode)
    if manifest.get("admitted_episodes") != len(episodes):
        raise ValueError(f"{lane}: admitted episode count differs from rows")
    return records, indexes, manifest


def merge_candidate_lanes(
    *,
    primary_records_path: Path,
    primary_index_path: Path,
    primary_manifest_path: Path,
    fallback_records_path: Path,
    fallback_index_path: Path,
    fallback_manifest_path: Path,
    out_path: Path,
    out_index_path: Path,
) -> dict[str, Any]:
    primary_records, primary_indexes, primary_manifest = _load_lane(
        primary_records_path,
        primary_index_path,
        primary_manifest_path,
        lane="primary",
    )
    fallback_records, fallback_indexes, fallback_manifest = _load_lane(
        fallback_records_path,
        fallback_index_path,
        fallback_manifest_path,
        lane="fallback",
    )
    primary_episodes = {
        str(row["metadata"]["source_episode_id"]) for row in primary_records
    }
    fallback_episodes = {
        str(row["metadata"]["source_episode_id"]) for row in fallback_records
    }
    selected_pairs = list(zip(primary_records, primary_indexes))
    selected_pairs.extend(
        (record, index)
        for record, index in zip(fallback_records, fallback_indexes)
        if str(record["metadata"]["source_episode_id"]) not in primary_episodes
    )
    record_ids = [str(record["metadata"]["record_id"]) for record, _ in selected_pairs]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("selected union has duplicate record ids")
    output_records = [record for record, _ in selected_pairs]
    output_indexes = [index for _, index in selected_pairs]
    _write_jsonl_atomic(out_path, output_records)
    _write_jsonl_atomic(out_index_path, output_indexes)
    selected_fallback = fallback_episodes - primary_episodes
    identity = _expected_identity()
    manifest = {
        "schema_version": "checkpoint-relalg-sft-candidate-union-manifest-v1",
        "merge_policy_version": MERGE_POLICY_VERSION,
        **identity,
        "source_priority": ["official_deepseek", "codex_manual_fallback"],
        "mutation": "none; complete episode rows copied byte-semantically from selected lanes",
        "primary": {
            "records": str(primary_records_path.resolve()),
            "records_sha256": _file_sha256(primary_records_path.resolve()),
            "index": str(primary_index_path.resolve()),
            "index_sha256": _file_sha256(primary_index_path.resolve()),
            "manifest": str(primary_manifest_path.resolve()),
            "manifest_sha256": _file_sha256(primary_manifest_path.resolve()),
            "episodes": len(primary_episodes),
            "records_count": len(primary_records),
        },
        "fallback": {
            "records": str(fallback_records_path.resolve()),
            "records_sha256": _file_sha256(fallback_records_path.resolve()),
            "index": str(fallback_index_path.resolve()),
            "index_sha256": _file_sha256(fallback_index_path.resolve()),
            "manifest": str(fallback_manifest_path.resolve()),
            "manifest_sha256": _file_sha256(fallback_manifest_path.resolve()),
            "episodes": len(fallback_episodes),
            "records_count": len(fallback_records),
        },
        "overlap_episodes": len(primary_episodes & fallback_episodes),
        "selected_primary_episodes": len(primary_episodes),
        "selected_fallback_episodes": len(selected_fallback),
        "episodes": len(primary_episodes | fallback_episodes),
        "records": len(output_records),
        "output": str(out_path.resolve()),
        "output_sha256": _file_sha256(out_path.resolve()),
        "index": str(out_index_path.resolve()),
        "index_sha256": _file_sha256(out_index_path.resolve()),
        "strict_artifact_is_admission_gate": False,
        "schema_match_is_admission_gate": False,
        "training_admission": "scheme_local_atomic_v24_frozen_sft_candidate_union",
        "rl_admission": False,
        "primary_source_manifest_sha256": _file_sha256(primary_manifest_path.resolve()),
        "fallback_source_manifest_sha256": _file_sha256(fallback_manifest_path.resolve()),
        "primary_export_manifest_schema": primary_manifest.get("schema_version"),
        "fallback_export_manifest_schema": fallback_manifest.get("schema_version"),
    }
    _write_json_atomic(out_path.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--primary-index", type=Path, required=True)
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--fallback-index", type=Path, required=True)
    parser.add_argument("--fallback-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-index", type=Path, required=True)
    args = parser.parse_args()
    manifest = merge_candidate_lanes(
        primary_records_path=args.primary,
        primary_index_path=args.primary_index,
        primary_manifest_path=args.primary_manifest,
        fallback_records_path=args.fallback,
        fallback_index_path=args.fallback_index,
        fallback_manifest_path=args.fallback_manifest,
        out_path=args.out,
        out_index_path=args.out_index,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
