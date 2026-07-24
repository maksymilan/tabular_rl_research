#!/usr/bin/env python3
"""Filter rolling SFT records using an exact LLaMA-Factory token audit.

The source episodes and rendered records remain immutable. This module removes only
single-action records whose final target, current source, or oldest rolling-history
pair is incomplete under the audited tokenizer/template/cutoff configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_rolling_sft_data import write_dataset_info


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
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


def record_id(row: dict[str, Any]) -> str:
    value = (row.get("metadata") or {}).get("record_id")
    if not isinstance(value, str) or not value:
        raise ValueError("SFT record is missing metadata.record_id")
    return value


def validate_unique_ids(rows: list[dict[str, Any]], *, source: str) -> list[str]:
    ids = [record_id(row) for row in rows]
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"{source}: duplicate record ids: {duplicates[:5]}")
    return ids


def rejected_records(audit: dict[str, Any]) -> tuple[set[str], Counter[str]]:
    details = audit.get("details")
    if not isinstance(details, list):
        raise ValueError("token audit does not contain per-record details")

    rejected: set[str] = set()
    reasons: Counter[str] = Counter()
    for item in details:
        item_reasons = []
        if (
            item.get("target_status") != "complete"
            or item.get("processor_target_complete") is False
        ):
            item_reasons.append("incomplete_final_target")
        if item.get("current_source_tokens_kept") != item.get("current_source_tokens_original"):
            item_reasons.append("incomplete_current_source")
        if item.get("history_pairs_original") and not item.get("first_pair_complete"):
            item_reasons.append("incomplete_oldest_history_pair")
        if not item_reasons:
            continue
        item_id = item.get("record_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("token audit detail is missing record_id")
        rejected.add(item_id)
        reasons.update(item_reasons)
    return rejected, reasons


def build(
    input_paths: list[Path],
    index_paths: list[Path],
    token_audit_paths: list[Path],
    out_path: Path,
    index_out_path: Path,
    dataset_name: str,
) -> dict[str, Any]:
    if len(input_paths) != len(index_paths):
        raise ValueError("each SFT input must have one matching index input")
    if len(input_paths) != len(token_audit_paths):
        raise ValueError("each SFT input must have one matching exact token audit")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset_name):
        raise ValueError("dataset_name must contain only letters, digits, '.', '_' or '-'")

    records = [row for path in input_paths for row in read_jsonl(path)]
    indexes = [row for path in index_paths for row in read_jsonl(path)]
    record_ids = validate_unique_ids(records, source="SFT inputs")
    index_ids = [row.get("record_id") for row in indexes]
    if any(not isinstance(item, str) or not item for item in index_ids):
        raise ValueError("index input is missing record_id")
    if len(index_ids) != len(set(index_ids)):
        raise ValueError("index inputs contain duplicate record ids")
    if record_ids != index_ids:
        raise ValueError("SFT records and indexes are not in identical record-id order")

    audits = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in token_audit_paths
    ]
    for input_path, input_rows, audit_path, audit in zip(
        input_paths,
        (read_jsonl(path) for path in input_paths),
        token_audit_paths,
        audits,
    ):
        if audit.get("records") != len(input_rows):
            raise ValueError(
                f"{audit_path} covers {audit.get('records')} records, but matching input "
                f"{input_path} contains {len(input_rows)}"
            )
    rejected: set[str] = set()
    rejection_reasons: Counter[str] = Counter()
    for audit in audits:
        audit_rejected, audit_reasons = rejected_records(audit)
        overlap = rejected & audit_rejected
        if overlap:
            raise ValueError(f"token audits overlap record ids: {sorted(overlap)[:5]}")
        rejected.update(audit_rejected)
        rejection_reasons.update(audit_reasons)
    unknown_rejections = sorted(rejected - set(record_ids))
    if unknown_rejections:
        raise ValueError(f"token audit references unknown records: {unknown_rejections[:5]}")

    kept_records = [row for row in records if record_id(row) not in rejected]
    kept_indexes = [row for row in indexes if row["record_id"] not in rejected]
    write_jsonl_atomic(out_path, kept_records)
    write_jsonl_atomic(index_out_path, kept_indexes)
    snippet, registry = write_dataset_info(out_path, dataset_name)

    all_by_episode: defaultdict[str, int] = defaultdict(int)
    kept_by_episode: defaultdict[str, int] = defaultdict(int)
    for row in indexes:
        all_by_episode[row["source_episode_id"]] += 1
    for row in kept_indexes:
        kept_by_episode[row["source_episode_id"]] += 1
    episodes_with_dropped_records = sorted(
        episode_id
        for episode_id, count in all_by_episode.items()
        if kept_by_episode[episode_id] != count
    )
    complete_episodes = sorted(set(all_by_episode) - set(episodes_with_dropped_records))

    manifest = {
        "selection": "exact-token-full-causal-prefix",
        "selection_policy": {
            "final_target": "complete",
            "current_source": "complete",
            "oldest_rolling_history_pair": "complete_when_present",
            "reasoning_word_limit": None,
            "mutation": "none; rejected records are omitted without rewriting source episodes",
        },
        "inputs": [
            {"path": str(path), "sha256": sha256(path)}
            for path in input_paths
        ],
        "index_inputs": [
            {"path": str(path), "sha256": sha256(path)}
            for path in index_paths
        ],
        "token_audits": [
            {
                "path": str(path),
                "sha256": sha256(path),
                "model": audit.get("model_path") or audit.get("model"),
                "template": audit.get("template"),
                "cutoff_len": audit.get("cutoff_len"),
                "mask_history": audit.get("mask_history"),
            }
            for path, audit in zip(token_audit_paths, audits)
        ],
        "output": str(out_path),
        "output_sha256": sha256(out_path),
        "index": str(index_out_path),
        "index_sha256": sha256(index_out_path),
        "dataset_name": dataset_name,
        "dataset_info_snippet": str(snippet),
        "dataset_info_registry": str(registry),
        "input_records": len(records),
        "kept_records": len(kept_records),
        "dropped_records": len(rejected),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "source_episodes": len(all_by_episode),
        "contributing_episodes": len(kept_by_episode),
        "episodes_with_dropped_records": len(episodes_with_dropped_records),
        "complete_episodes": len(complete_episodes),
        "records_from_complete_episodes": sum(all_by_episode[item] for item in complete_episodes),
        "feedback_recovery_targets": sum(
            bool((row.get("metadata") or {}).get("feedback_recovery"))
            for row in kept_records
        ),
        "tool_hist": dict(Counter(row.get("tool_name", "unknown") for row in kept_indexes).most_common()),
        "dropped_record_ids": sorted(rejected),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--token-audit", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index-out", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    args = parser.parse_args()
    manifest = build(
        [path.resolve() for path in args.input],
        [path.resolve() for path in args.index],
        [path.resolve() for path in args.token_audit],
        args.out.resolve(),
        args.index_out.resolve(),
        args.dataset_name,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
