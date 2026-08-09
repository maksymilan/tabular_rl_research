#!/usr/bin/env python3
"""Repeat the longest audited SFT row for a worst-case distributed smoke test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sft_dataset_registry import write_sharegpt_dataset_info


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build(
    training_view: Path,
    index_path: Path,
    token_audit_path: Path,
    output_path: Path,
    dataset_name: str,
    repeats: int,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("repeats must be at least two")
    rows = read_jsonl(training_view)
    indexes = read_jsonl(index_path)
    audit = json.loads(token_audit_path.read_text(encoding="utf-8"))
    if len(rows) != len(indexes) or len(rows) != audit.get("records"):
        raise ValueError("training view, index, and token audit record counts differ")
    longest = audit.get("longest_records")
    if not isinstance(longest, list) or not longest:
        raise ValueError("token audit has no longest_records")
    record_id = longest[0].get("record_id")
    positions = [i for i, item in enumerate(indexes) if item.get("record_id") == record_id]
    if len(positions) != 1:
        raise ValueError(f"longest record {record_id!r} is not unique in index")
    selected = rows[positions[0]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for _ in range(repeats):
            handle.write(json.dumps(selected, ensure_ascii=False) + "\n")
    snippet, registry = write_sharegpt_dataset_info(output_path, dataset_name)
    manifest = {
        "selection": "repeat-longest-exact-token-audited-record-v1",
        "training_view": str(training_view),
        "training_view_sha256": sha256(training_view),
        "index": str(index_path),
        "index_sha256": sha256(index_path),
        "token_audit": str(token_audit_path),
        "token_audit_sha256": sha256(token_audit_path),
        "selected_record_id": record_id,
        "selected_original_tokens": longest[0].get("original_tokens"),
        "repeats": repeats,
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "dataset_name": dataset_name,
        "dataset_info_snippet": str(snippet),
        "dataset_info_registry": str(registry),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--token-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--repeats", type=int, default=32)
    args = parser.parse_args()
    manifest = build(
        args.training_view.resolve(),
        args.index.resolve(),
        args.token_audit.resolve(),
        args.out.resolve(),
        args.dataset_name,
        args.repeats,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
