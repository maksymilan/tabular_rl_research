#!/usr/bin/env python3
"""Select the longest fully admitted v3 records for a deterministic memory smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_checkpoint_relalg_qwen3_projection_v3 import read_jsonl, sha256, write_jsonl_atomic
from sft_dataset_registry import write_sharegpt_dataset_info


def build(
    training_view: Path,
    index_path: Path,
    token_audit_path: Path,
    output: Path,
    output_index: Path,
    dataset_name: str,
    count: int = 32,
) -> dict:
    if count <= 0:
        raise ValueError("count must be positive")
    rows = read_jsonl(training_view)
    indexes = read_jsonl(index_path)
    if len(rows) != len(indexes):
        raise ValueError("training view and index counts differ")
    audit = json.loads(token_audit_path.read_text(encoding="utf-8"))
    if audit.get("cutoff_len") != 8192:
        raise ValueError("token audit identity differs")
    details = audit.get("details")
    if not isinstance(details, list):
        raise ValueError("token audit does not contain per-record overflow details")
    positions = {item.get("record_id"): number for number, item in enumerate(indexes)}
    candidates = sorted(
        (
            (int(item.get("original_tokens", 0)), str(item.get("record_id")))
            for item in details
            if item.get("record_id") in positions
        ),
        key=lambda value: (-value[0], value[1]),
    )
    selected_ids = [item_id for _, item_id in candidates[:count]]
    if len(selected_ids) != count:
        raise ValueError("longest records do not identify enough training rows")
    selected_rows = [rows[positions[item_id]] for item_id in selected_ids]
    selected_indexes = [indexes[positions[item_id]] for item_id in selected_ids]
    write_jsonl_atomic(output, selected_rows)
    write_jsonl_atomic(output_index, selected_indexes)
    write_sharegpt_dataset_info(output, dataset_name)
    manifest = {
        "schema_version": "checkpoint-relalg-qwen3-longest-smoke-v1",
        "selection": (
            "longest v3-selected records among exact 8K overflow details; "
            "combined corpus separately requires a zero-drop 16K audit"
        ),
        "source_training_view": str(training_view),
        "source_training_view_sha256": sha256(training_view),
        "source_index": str(index_path),
        "source_index_sha256": sha256(index_path),
        "token_audit": str(token_audit_path),
        "token_audit_sha256": sha256(token_audit_path),
        "dataset_name": dataset_name,
        "records": count,
        "record_ids": selected_ids,
        "output": str(output),
        "output_sha256": sha256(output),
        "output_index": str(output_index),
        "output_index_sha256": sha256(output_index),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--token-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index-out", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    result = build(
        args.training_view.resolve(),
        args.index.resolve(),
        args.token_audit.resolve(),
        args.out.resolve(),
        args.index_out.resolve(),
        args.dataset_name,
        args.count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
