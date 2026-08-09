#!/usr/bin/env python3
"""Rebind SFT index hashes after a documented model-visible projection.

The source index hashes the canonical model input.  A Qwen3-specific history view
changes that input while preserving record identity and the final target, so copying
the old ``model_input_sha256`` would make the derived index semantically false.  This
tool retains the original hash as lineage and computes the hash of the actual derived
messages using the same canonical JSON digest as ``build_rolling_sft_data.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROLE_MAP = {"human": "user", "gpt": "assistant"}


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(compact(value).encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def row_hashes(row: dict[str, Any], *, record_id: str) -> tuple[str, str]:
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"{record_id}: conversations must be non-empty")
    if conversations[-1].get("from") != "gpt":
        raise ValueError(f"{record_id}: final target is not a gpt turn")
    system = row.get("system")
    if not isinstance(system, str) or not system:
        raise ValueError(f"{record_id}: system must be non-empty")
    messages = [{"role": "system", "content": system}]
    for turn in conversations[:-1]:
        role = ROLE_MAP.get(turn.get("from"))
        content = turn.get("value")
        if role is None or not isinstance(content, str):
            raise ValueError(f"{record_id}: unsupported model-input turn")
        messages.append({"role": role, "content": content})
    target = conversations[-1].get("value")
    if not isinstance(target, str):
        raise ValueError(f"{record_id}: final target must be a string")
    return digest(messages), digest(target)


def rebind(
    view_path: Path,
    canonical_path: Path,
    source_index_path: Path,
    output_path: Path,
    projection: str,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    rows = read_jsonl(view_path)
    canonical_rows = read_jsonl(canonical_path)
    indexes = read_jsonl(source_index_path)
    if not (len(rows) == len(canonical_rows) == len(indexes)):
        raise ValueError("model view, canonical input, and source index record counts differ")

    rebound: list[dict[str, Any]] = []
    seen: set[str] = set()
    canonical_to_view_changed = 0
    source_input_matches_canonical = 0
    source_target_matches_canonical = 0
    for row, canonical_row, source_index in zip(rows, canonical_rows, indexes):
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("model-view row is missing metadata")
        record_id = metadata.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("model-view row is missing metadata.record_id")
        if record_id in seen:
            raise ValueError(f"duplicate record id: {record_id}")
        seen.add(record_id)
        canonical_record_id = (canonical_row.get("metadata") or {}).get("record_id")
        if source_index.get("record_id") != record_id or canonical_record_id != record_id:
            raise ValueError(f"row/index order mismatch at {record_id}")
        if canonical_row.get("system") != row.get("system"):
            raise ValueError(f"{record_id}: projection changed system")
        canonical_conversations = canonical_row.get("conversations")
        view_conversations = row.get("conversations")
        if (
            not isinstance(canonical_conversations, list)
            or not isinstance(view_conversations, list)
            or canonical_conversations[-1] != view_conversations[-1]
        ):
            raise ValueError(f"{record_id}: projection changed final target")
        canonical_input_hash, canonical_target_hash = row_hashes(
            canonical_row, record_id=record_id
        )
        view_input_hash, view_target_hash = row_hashes(row, record_id=record_id)
        if canonical_target_hash != view_target_hash:
            raise ValueError(f"{record_id}: projection changed target hash")
        source_input_hash = source_index.get("model_input_sha256")
        source_target_hash = source_index.get("target_sha256")
        if not isinstance(source_input_hash, str) or not source_input_hash:
            raise ValueError(f"{record_id}: source index has no model_input_sha256")
        if not isinstance(source_target_hash, str) or not source_target_hash:
            raise ValueError(f"{record_id}: source index has no target_sha256")
        canonical_to_view_changed += int(canonical_input_hash != view_input_hash)
        source_input_matches_canonical += int(source_input_hash == canonical_input_hash)
        source_target_matches_canonical += int(source_target_hash == canonical_target_hash)
        item = dict(source_index)
        item["source_index_model_input_sha256"] = source_input_hash
        item["source_index_target_sha256"] = source_target_hash
        item["canonical_model_input_sha256"] = canonical_input_hash
        item["canonical_target_sha256"] = canonical_target_hash
        item["model_input_sha256"] = view_input_hash
        item["target_sha256"] = view_target_hash
        item["model_input_projection"] = projection
        rebound.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for item in rebound:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    manifest = {
        "operation": "rebind-sft-index-to-derived-model-view-v1",
        "projection": projection,
        "model_view": str(view_path),
        "model_view_sha256": sha256(view_path),
        "canonical_input": str(canonical_path),
        "canonical_input_sha256": sha256(canonical_path),
        "source_index": str(source_index_path),
        "source_index_sha256": sha256(source_index_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "records": len(rebound),
        "unique_record_ids": len(seen),
        "canonical_to_view_model_input_hashes_changed": canonical_to_view_changed,
        "canonical_to_view_model_input_hashes_unchanged": len(rebound)
        - canonical_to_view_changed,
        "target_hashes_changed_by_projection": 0,
        "source_index_model_input_hashes_matching_canonical": source_input_matches_canonical,
        "source_index_target_hashes_matching_canonical": source_target_matches_canonical,
        "lineage_fields": [
            "source_index_model_input_sha256",
            "source_index_target_sha256",
            "canonical_model_input_sha256",
            "canonical_target_sha256",
        ],
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--projection", required=True)
    args = parser.parse_args()
    manifest = rebind(
        args.view.resolve(),
        args.canonical.resolve(),
        args.source_index.resolve(),
        args.out.resolve(),
        args.projection,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
